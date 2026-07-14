# src/eryn/flows/torch/mixture.py
"""Mode-mixture conditional flow wrapper.

:class:`ModeMixtureFlow` wraps :class:`~eryn.flows.torch.flows.ZukoFlow` with
per-leaf, buffer-driven mode clustering (:func:`eryn.flows.modes.estimate_modes`)
so that a single flow can represent a multimodal per-leaf posterior as an
explicit mixture over ``(leaf, mode-slot)`` composite conditions, instead of
one flow having to bridge disjoint islands under a single leaf condition.

This module implements the fit path, snapshot, and HDF5 persistence, plus
mixture sampling and density evaluation (marginalizing over mode slots) — see
the module docstring of ``ModeMixtureFlow`` for details.

Requires the ``flow`` optional extra (``pip install eryn[flow]``), which
installs ``torch`` and ``zuko`` — but only indirectly, through
:class:`~eryn.flows.torch.flows.ZukoFlow``.  This module itself does not
import torch directly.
"""
from __future__ import annotations

import copy
import os
import pickle

import h5py
import numpy as np
from scipy.special import logsumexp

from eryn.flows.conditioning import LeafModeConditioning
from eryn.flows.modes import ModeState, estimate_modes
from eryn.flows.torch.flows import ZukoFlow

__all__ = ["ModeMixtureFlow"]


class ModeMixtureFlow(ZukoFlow):
    """Per-leaf mode-mixture conditional flow.

    At :meth:`fit` time, each leaf's training buffer is clustered into
    ``K <= kmax`` mode-slot components via :func:`eryn.flows.modes.estimate_modes`
    (warm-started from the previous round's :class:`~eryn.flows.modes.ModeState`
    for stable slot identities).  Rows are then re-labeled onto composite
    condition ids ``cid = leaf * kmax + slot`` (see :meth:`_cid`) and handed to
    the underlying :class:`~eryn.flows.torch.flows.ZukoFlow` as one condition
    per populated slot — i.e. a single flow net, conditioned by
    :class:`~eryn.flows.conditioning.LeafModeConditioning`, that gets its own
    per-slot ``data_transform`` island.

    MH exactness never depends on the clustering: the mixture density
    (:meth:`log_prob`) marginalizes over the slots the flow was trained on
    via ``logsumexp_m [log w_m + logq(x | cid(leaf, m))]``, which is exact for
    any clustering choice — clustering quality only affects proposal
    efficiency, never validity.

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    nleaves_max : int
        Maximum number of leaves (components) in the parent RJ model.
    kmax : int, optional
        Maximum number of mode slots per leaf.  Default is ``8``.
    mode_floor : float, optional
        Minimum per-slot mixture weight floor passed to
        :func:`~eryn.flows.modes.estimate_modes` (``floor``).  Default is
        ``0.02``.
    min_rows_per_component : int, optional
        Minimum rows for a cluster to survive as its own slot, passed to
        :func:`~eryn.flows.modes.estimate_modes` (``min_rows``).  Default is
        ``25``.
    cluster_seed : int, optional
        Seed forwarded to :func:`~eryn.flows.modes.estimate_modes` at every
        :meth:`fit` call, and used to derive :attr:`_rng`
        (``np.random.default_rng(cluster_seed + 1)``, reserved for the
        mixture sampling path).  Default is ``0``.
    periodic : dict of {int: (float, float)} or None, optional
        Mapping from dimension index to ``(low, high)`` boundary of that
        periodic component — the same mapping given to
        :class:`~eryn.flows.torch.transforms.WhiteningTransform`.  Needed here
        so the wrapper can embed rows the same wrap-free way
        (:func:`eryn.flows.modes.embed`) before clustering.  Stored on
        :attr:`_periodic` with integer-keyed, float-valued tuples regardless
        of whether the caller passed that form directly or the
        JSON-round-tripped form (``{"2": [0.0, 6.28]}``) produced by
        :meth:`~eryn.flows.torch.flows.BaseTorchFlow.load`.  Default is
        ``None`` (no periodic dimensions).
    conditioning : ConditioningStrategy or None, optional
        Conditioning strategy.  When ``None`` (the default), a
        :class:`~eryn.flows.conditioning.LeafModeConditioning` is built
        internally from ``nleaves_max``/``kmax``.  The escape hatch exists so
        that :meth:`~eryn.flows.torch.flows.BaseTorchFlow.load` can reconstruct
        the instance with the exact (unpickled) conditioning object the
        checkpoint was trained with, rather than a freshly built one.
    **zuko_kwargs
        Forwarded to :class:`~eryn.flows.torch.flows.ZukoFlow.__init__`
        (``flow_class``, ``device``, ``data_transform``, ``seed``, and any
        flow-architecture kwargs such as ``transforms``/``hidden_features``/
        ``bins``).

    Attributes
    ----------
    mode_state : dict[int, ModeState]
        Per-leaf clustering state from the most recent :meth:`fit` call.
        Empty before the first fit.
    """

    def __init__(
        self,
        dims,
        nleaves_max,
        kmax=8,
        mode_floor=0.02,
        min_rows_per_component=25,
        cluster_seed=0,
        periodic=None,
        conditioning=None,
        **zuko_kwargs,
    ):
        self.nleaves_max = int(nleaves_max)
        self.kmax = int(kmax)
        self.mode_floor = float(mode_floor)
        self.min_rows_per_component = int(min_rows_per_component)
        self.cluster_seed = int(cluster_seed)
        self._periodic = _normalize_periodic(periodic)

        if conditioning is None:
            conditioning = LeafModeConditioning(self.nleaves_max, self.kmax)

        super().__init__(dims=dims, conditioning=conditioning, **zuko_kwargs)

        self.mode_state: dict[int, ModeState] = {}
        # Reserved for the mixture sampling path (slot draws); harmless here.
        self._rng = np.random.default_rng(self.cluster_seed + 1)

    # ------------------------------------------------------------------
    # Composite condition ids
    # ------------------------------------------------------------------

    def _cid(self, leaf: int, slot: int) -> int:
        """Return the composite condition id for ``(leaf, slot)``.

        Parameters
        ----------
        leaf : int
        slot : int

        Returns
        -------
        int
            ``leaf * kmax + slot`` — matches
            :meth:`~eryn.flows.conditioning.LeafModeConditioning.encode`'s
            ``divmod(cid, kmax)`` decoding.
        """
        return int(leaf) * self.kmax + int(slot)

    # ------------------------------------------------------------------
    # Fit: per-leaf clustering + composite-condition training
    # ------------------------------------------------------------------

    def fit(self, samples, **fit_kwargs):
        """Cluster each leaf's buffer into mode slots, then train on composite conditions.

        Parameters
        ----------
        samples : dict[int, np.ndarray]
            Per-leaf training rows, ``{leaf: (N, dims)}``.
        **fit_kwargs
            Forwarded to :meth:`~eryn.flows.torch.flows.ZukoFlow.fit`.

        Returns
        -------
        history : FlowHistory

        Notes
        -----
        For each leaf, :func:`~eryn.flows.modes.estimate_modes` is warm-started
        from ``self.mode_state.get(leaf)`` (the previous round's state, if any)
        for stable slot identities across rounds.  Rows are then split by
        cluster label into the composite dict ``{cid(leaf, slot): rows}``,
        preserving each leaf's row order within its slot (boolean masking
        preserves order) — ``val_split="temporal"`` depends on this.
        """
        composite: dict[int, np.ndarray] = {}
        for leaf, rows in samples.items():
            rows = np.asarray(rows, dtype=np.float64)
            st = estimate_modes(
                rows,
                self._periodic,
                self.kmax,
                prev=self.mode_state.get(leaf),
                floor=self.mode_floor,
                min_rows=self.min_rows_per_component,
                seed=self.cluster_seed,
            )
            self.mode_state[leaf] = st
            for slot in st.slots:
                composite[self._cid(leaf, slot)] = rows[st.labels == slot]

        return super().fit(composite, **fit_kwargs)

    # ------------------------------------------------------------------
    # Mixture sampling / density
    # ------------------------------------------------------------------

    def log_prob(self, x, context=None, base_scale: float | None = None) -> np.ndarray:
        """Return the exact per-leaf mixture log density at ``x``.

        Marginalizes over the leaf's mode slots::

            log q(x | leaf) = logsumexp_s [ log w_s + logq(x | cid(leaf, s)) ]

        where the inner ``logq`` is the plain :class:`~eryn.flows.torch.flows.ZukoFlow`
        component density (``super().log_prob``) evaluated at the composite
        condition id for each populated slot.  This is the SAME code path used
        for both proposed and current points in an MH step (see
        :meth:`sample_and_log_prob`), which is what makes the move's factors
        exact regardless of clustering quality.

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Points in coords space.
        context : int
            Bare leaf id (NOT a composite condition id).
        base_scale : float or None, optional
            Forwarded unchanged to every component's
            :meth:`~eryn.flows.torch.flows.ZukoFlow.log_prob` call; see that
            method for the temperature-scaled-base semantics.  ``None``
            (default) is bit-identical to the pre-``base_scale`` behaviour.

        Returns
        -------
        log_prob : np.ndarray, shape (N,), dtype float64

        Raises
        ------
        RuntimeError
            If ``context`` names a leaf with no fitted :attr:`mode_state`
            (i.e. :meth:`fit` has not been called for it yet).
        """
        leaf = int(context)
        st = self.mode_state.get(leaf)
        if st is None:
            raise RuntimeError(f"ModeMixtureFlow: no mode_state for leaf {leaf}; fit first.")
        comps = np.stack([
            np.log(st.weights[s])
            + super(ModeMixtureFlow, self).log_prob(x, context=self._cid(leaf, s),
                                                    base_scale=base_scale)
            for s in st.slots
        ])                                   # (K, N)
        return logsumexp(comps, axis=0)

    def sample_and_log_prob(self, n: int, context=None, base_scale: float | None = None) -> tuple:
        """Draw ``n`` samples from the per-leaf mixture and return their mixture density.

        Draws component counts from ``Multinomial(n, [weights[s] for s in slots])``
        via :attr:`_rng`, samples each populated slot's component from the
        underlying :class:`~eryn.flows.torch.flows.ZukoFlow`
        (``super().sample_and_log_prob``) at its composite condition id, then
        concatenates and shuffles the rows.

        The shuffle is mandatory: :class:`~eryn.moves.ConditionalFlowMove` maps
        returned rows positionally onto (temperature, walker) slots, so
        component-grouped (unshuffled) draws would correlate mode with
        temperature.

        The returned log density is **not** the per-component value from the
        inner sampling call — it is :meth:`log_prob` (the full mixture
        density) evaluated on the shuffled draws, so that proposed-point and
        current-point densities in an MH step come from the identical code
        path.  Within a component these differ only on rare wrap-cut crossers
        (the pinned wrap-cut contract).

        Parameters
        ----------
        n : int
            Number of samples to draw.
        context : int
            Bare leaf id (NOT a composite condition id).
        base_scale : float or None, optional
            Forwarded unchanged to every component's
            :meth:`~eryn.flows.torch.flows.ZukoFlow.sample_and_log_prob` call
            (and to the :meth:`log_prob` re-evaluation). ``None`` (default) is
            bit-identical to the pre-``base_scale`` behaviour.

        Returns
        -------
        x : np.ndarray, shape (n, dims), dtype float64
            Samples in coords space, shuffled across components.
        log_prob : np.ndarray, shape (n,), dtype float64
            Mixture log density at each returned sample (see :meth:`log_prob`).
        """
        leaf = int(context)
        st = self.mode_state[leaf]           # same guard as log_prob
        counts = self._rng.multinomial(n, [st.weights[s] for s in st.slots])
        xs = [super(ModeMixtureFlow, self).sample_and_log_prob(
                  int(c), context=self._cid(leaf, s), base_scale=base_scale)[0]
              for s, c in zip(st.slots, counts) if c > 0]
        x = np.concatenate(xs, axis=0)
        x = x[self._rng.permutation(n)]      # break component<->row-position correlation
        return x, self.log_prob(x, context=leaf, base_scale=base_scale)

    # ------------------------------------------------------------------
    # Snapshot / weight hand-off
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a snapshot with the mixture state alongside the net + transform.

        Returns
        -------
        dict
            ``super().get_snapshot()`` (``{"net", "data_transform"}``) plus a
            deep copy of :attr:`mode_state` under the ``"mixture_state"`` key.
        """
        d = super().get_snapshot()
        d["mixture_state"] = copy.deepcopy(self.mode_state)
        return d

    def set_weights(self, obj) -> None:
        """Restore weights, installing ``mixture_state`` first when present.

        When ``obj`` is a snapshot (``"net"`` sentinel) that also carries a
        ``"mixture_state"`` entry, :attr:`mode_state` is installed as a deep
        copy of that entry *before* delegating to
        :meth:`~eryn.flows.torch.flows.BaseTorchFlow.set_weights`, so the net,
        its matched data transform, and its matched mixture state move
        together atomically.  ``obj`` itself is never mutated.

        Parameters
        ----------
        obj : dict
            A snapshot (optionally with ``"mixture_state"``) or a bare net
            state_dict.
        """
        if isinstance(obj, dict) and "net" in obj and "mixture_state" in obj:
            self.mode_state = copy.deepcopy(obj["mixture_state"])
        super().set_weights(obj)

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------

    def save(self, h5_file, path: str = "flow") -> None:
        """Save the flow (via ``super().save``) plus the mixture state.

        Adds a ``"mixture_state"`` pickle-blob dataset next to the existing
        ``"data_transform"`` blob written by
        :meth:`~eryn.flows.torch.flows.BaseTorchFlow.save`.

        Parameters
        ----------
        h5_file : str or h5py.File
            Destination path or already-open handle.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.
        """
        handle, should_close = self._open_h5(h5_file, "a")
        try:
            super().save(handle, path=path)
            grp = handle[path]
            blob = pickle.dumps(self.mode_state)
            if "mixture_state" in grp:
                del grp["mixture_state"]
            grp.create_dataset("mixture_state", data=np.bytes_(blob))
        finally:
            if should_close:
                handle.close()

    @classmethod
    def load(cls, h5_file, path: str = "flow") -> "ModeMixtureFlow":
        """Load a flow (via ``super().load``) and restore the mixture state.

        Parameters
        ----------
        h5_file : str or h5py.File
            Source path or already-open handle.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.

        Returns
        -------
        ModeMixtureFlow
            Restored instance.  ``mode_state`` is restored from the
            ``"mixture_state"`` dataset if present, or set to ``{}`` for
            checkpoints written before this dataset existed.
        """
        if isinstance(h5_file, (str, os.PathLike)):
            handle = h5py.File(h5_file, "r")
            should_close = True
        else:
            handle, should_close = h5_file, False
        try:
            instance = super().load(handle, path=path)
            grp = handle[path]
            if "mixture_state" in grp:
                raw = bytes(grp["mixture_state"][()])
                instance.mode_state = pickle.loads(raw)
            else:
                instance.mode_state = {}
            return instance
        finally:
            if should_close:
                handle.close()


def _normalize_periodic(periodic) -> dict:
    """Coerce a ``periodic`` mapping to ``{int: (float, float)}``.

    Accepts both the caller-supplied form (int keys, tuple/list values) and
    the JSON-round-tripped form produced by
    :func:`~eryn.flows.torch.flows._make_serialisable` /
    :meth:`~eryn.flows.torch.flows.BaseTorchFlow.load` (``{"2": [0.0, 6.28]}``
    — string keys, list values).

    Parameters
    ----------
    periodic : dict or None

    Returns
    -------
    dict of {int: (float, float)}
        Empty dict if ``periodic`` is ``None`` or empty.
    """
    if not periodic:
        return {}
    out = {}
    for dim, bounds in periodic.items():
        low, high = bounds
        out[int(dim)] = (float(low), float(high))
    return out
