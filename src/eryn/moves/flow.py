# -*- coding: utf-8 -*-
"""Independent-Metropolis move backed by an eryn.flows Flow.

This module is intentionally torch-free at import time: ``eryn.flows.base``
(where ``FlowProposalDistribution`` lives) is pure NumPy, but
``eryn.flows.torch`` (where the concrete backends live) may import torch.
``FlowProposalDistribution`` is therefore imported lazily inside
``get_proposal`` rather than at module level, ensuring that
``import eryn.moves`` never triggers a torch import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mh import MHMove

if TYPE_CHECKING:
    # Only for static type-checkers; never executed at runtime.
    from eryn.flows.base import Flow, FlowProposalDistribution  # noqa: F401

__all__ = ["FlowMove"]


class FlowMove(MHMove):
    """Independent-Metropolis proposal drawing from a conditional normalizing flow.

    Proposes new coordinates for a single named branch by sampling from a
    :class:`eryn.flows.Flow` conditioned on :attr:`active_condition`.  The
    Hastings factor convention matches :class:`eryn.moves.DistributionGenerate`::

        factors += +log q(x_old) - log q(x_new)

    A poor or stale flow only lowers acceptance — it never biases the posterior.
    This no-bias guarantee relies on IEEE-754: NaN logq values propagate into
    lnpdiff and NaN comparisons are False, so any proposal with a NaN log-prob
    is always rejected rather than silently accepted.
    Accumulation uses :func:`numpy.add.at` (not ``+=``) so that duplicate
    ``(temp, walker)`` indices from multi-leaf proposals sum instead of being
    silently overwritten by the last write.

    :meth:`Flow.sample_and_log_prob` is used for new points to avoid a second
    forward pass through the flow: the log-probability returned by sampling is
    reused directly as ``-log q(new)``.

    **Online training (optional)**

    When an ``executor`` (a :class:`eryn.flows.executors.TrainerExecutor`) is
    supplied, :meth:`setup` — called at the top of every :meth:`propose` — gains
    two non-blocking online-training hooks:

    1. **Harvest.** Every ``harvest_every``-th call, the cold-chain coordinates
       of this move's branch are flattened and submitted to the executor as
       training data.  The submit is non-blocking by contract; the executor
       trains a *clone* of the flow off the proposal's critical path.
    2. **Poll + hot-reload.** On *every* call, the newest trained weights are
       polled; if a strictly newer version is available it is loaded into this
       move's flow via ``flow.set_weights``.  The proposal therefore tracks an
       improving flow without ever blocking.

    Failures propagate loudly: a dead/failed trainer surfaces as
    :class:`~eryn.flows.executors.TrainerError` out of :meth:`setup` and is NOT
    swallowed.  A multi-day run silently sampling from a frozen flow is worse
    than a crash; graceful degradation is a future scheduler policy, not the
    move's job.

    Parameters
    ----------
    flow : eryn.flows.Flow
        Fitted (or freshly initialised) conditional normalizing flow.  Must
        implement :meth:`~eryn.flows.Flow.log_prob`,
        :meth:`~eryn.flows.Flow.sample_and_log_prob`, and the rest of the
        :class:`eryn.flows.Flow` ABC.
    branch_name : str
        Name of the branch this move proposes for.  All other branches in
        ``branches_coords`` are copied unchanged.
    executor : eryn.flows.executors.TrainerExecutor or None, optional
        Online-training executor.  When ``None`` (the default), :meth:`setup`
        is a pure no-op and the move behaves exactly as a static flow proposal
        — zero overhead, no executor interaction.
    harvest_every : int, optional
        Submit cold-chain coordinates to the executor on every
        ``harvest_every``-th :meth:`setup` call.  Default is ``1`` (every call).
        Polling for new weights happens on every call regardless of this value.
    harvest_temp_index : int, optional
        Temperature index to harvest from.  Default is ``0`` (the cold chain).
        Harvesting flattens the ``nwalkers`` x ``nleaves`` coordinates of this
        temperature into ``(-1, ndim)`` training rows.
    *args, **kwargs
        Passed through to :class:`eryn.moves.MHMove`.

    Attributes
    ----------
    active_condition : int
        Condition id passed to the flow on every ``get_proposal`` call.
        Readable and writable; the setter coerces the value to ``int``.
    loaded_version : int
        Version of the most recently hot-loaded weights (``0`` if none have been
        loaded).  Read-only.

    Examples
    --------
    >>> move = FlowMove(my_flow, branch_name="x")
    >>> move.active_condition = 2   # switch to condition 2 before sampling

    >>> # With online training:
    >>> from eryn.flows import InlineExecutor
    >>> ex = InlineExecutor(my_flow, min_train_samples=1000)
    >>> move = FlowMove(my_flow, branch_name="x", executor=ex, harvest_every=10)
    """

    def __init__(
        self,
        flow,
        branch_name: str,
        executor=None,
        harvest_every: int = 1,
        harvest_temp_index: int = 0,
        *args,
        **kwargs,
    ):
        self.flow = flow
        self.branch_name = branch_name
        self._active_condition: int = 0
        self.harvest_every = int(harvest_every)
        self.harvest_temp_index = int(harvest_temp_index)
        self._setup_calls = 0
        self._loaded_version = 0
        # Initialise through the property so the executor-swap bookkeeping
        # (version + harvest-counter reset) is defined in exactly one place.
        self._executor = None
        self.executor = executor
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # active_condition property
    # ------------------------------------------------------------------

    @property
    def active_condition(self) -> int:
        """Condition id used when calling the flow."""
        return self._active_condition

    @active_condition.setter
    def active_condition(self, c: int) -> None:
        self._active_condition = int(c)

    @property
    def loaded_version(self) -> int:
        """Version of the most recently hot-loaded flow weights (``0`` = none)."""
        return self._loaded_version

    # ------------------------------------------------------------------
    # executor property — resets hot-reload bookkeeping on a real swap
    # ------------------------------------------------------------------

    @property
    def executor(self):
        """Online-training executor (or ``None``).

        Assigning a **new** executor object resets this move's hot-reload
        memory (:attr:`loaded_version` back to ``0`` and the internal harvest
        counter back to ``0``).  This is essential because
        :attr:`~eryn.flows.executors.TrainerExecutor.version` is *per-executor-
        instance*: a replacement executor (e.g. a recreated
        :class:`~eryn.flows.executors.ProcessExecutor` after a worker died)
        restarts its version counter at ``1``.  Without the reset, :meth:`setup`
        would compare the new executor's version ``1`` against a remembered
        ``loaded_version`` of, say, ``7`` and silently ignore every hot-reload
        for the rest of the run.  Assigning the *same* object that is already
        installed is a no-op (identity comparison) and preserves the counters.
        """
        return self._executor

    @executor.setter
    def executor(self, new_executor) -> None:
        # Identity comparison: re-assigning the same object must NOT reset the
        # counters (that would discard legitimate hot-reload progress); only a
        # genuinely different executor instance restarts the bookkeeping.
        if new_executor is self._executor:
            return
        self._executor = new_executor
        self._loaded_version = 0
        self._setup_calls = 0

    # ------------------------------------------------------------------
    # Online-training hook (called at the top of every propose)
    # ------------------------------------------------------------------

    def setup(self, branches_coords):
        """Harvest cold-chain samples and hot-reload trained weights.

        Called by :meth:`eryn.moves.mh.MHMove.propose` at the top of every
        proposal.  With no executor configured this is a pure no-op.  With an
        executor it performs two non-blocking steps:

        - **Harvest** (every ``harvest_every``-th call): the coordinates of this
          move's branch at temperature ``harvest_temp_index`` (the cold chain by
          default) are reshaped to ``(-1, ndim)`` — flattening walkers x leaves —
          and submitted to the executor under :attr:`active_condition`.
        - **Poll + hot-reload** (every call): the newest trained weights are
          polled; a strictly newer version is loaded into :attr:`flow`.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` current
            coordinates.

        Raises
        ------
        KeyError
            If this move's ``branch_name`` is not present in ``branches_coords``
            (same loud guard as :meth:`get_proposal`).
        eryn.flows.executors.TrainerError
            If the executor's trainer has failed.  Propagated, never swallowed:
            silently sampling from a frozen flow for a long run is worse than a
            crash.
        """
        if self.executor is None:
            return

        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        self._setup_calls += 1

        # --- harvest (non-blocking submit) ---
        if self._setup_calls % self.harvest_every == 0:
            coords = branches_coords[self.branch_name][self.harvest_temp_index]
            ndim = coords.shape[-1]
            flat = np.asarray(coords).reshape(-1, ndim)
            self.executor.submit({self.active_condition: flat})

        # --- poll + hot-reload (non-blocking; TrainerError propagates) ---
        lw = self.executor.latest_weights()
        if lw is not None:
            version, snapshot = lw
            if version > self._loaded_version:
                # ``snapshot`` is a self-contained {"net", "data_transform"}
                # payload: set_weights installs the matched transform + net
                # atomically, so this flow never disagrees with the trainer on
                # the coords-latent map.
                self.flow.set_weights(snapshot)
                self._loaded_version = version

    # ------------------------------------------------------------------
    # MHMove interface
    # ------------------------------------------------------------------

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate independent-Metropolis proposals from the flow.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` with current
            walker coordinates.
        random : object
            Current random state (unused by this move, passed for API compatibility).
        branches_inds : dict, optional
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max]`` booleans indicating
            which leaves are active.  Defaults to all-``True`` when ``None``.
        **kwargs
            Accepted and ignored for compatibility.

        Returns
        -------
        q : dict
            Proposed coordinates; same structure as ``branches_coords``.
        factors : np.ndarray, shape (ntemps, nwalkers)
            Hastings log-factors: ``+log q(x_old) - log q(x_new)`` accumulated
            with :func:`numpy.add.at` over all active leaves of ``branch_name``.
        """
        # Lazy import: FlowProposalDistribution is pure NumPy (no torch), but
        # importing it here guarantees that even if eryn.flows.base were ever
        # changed to pull in a backend, this module stays torch-free at the
        # top level.
        from eryn.flows.base import FlowProposalDistribution

        dist = FlowProposalDistribution(self.flow, condition=self.active_condition)

        if self.branch_name not in branches_coords:
            raise KeyError(
                f"{type(self).__name__}: branch_name {self.branch_name!r} not in"
                f" branches_coords (keys: {list(branches_coords)})."
            )

        q = {}

        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords.items()
            }

        first = next(iter(branches_coords.values()))
        factors = np.zeros(first.shape[:2])

        for name, coords in branches_coords.items():
            q[name] = coords.copy()

            if name != self.branch_name:
                continue  # FlowMove only proposes for its own branch

            where = np.where(branches_inds[name])
            old_points = coords[where]
            num = len(where[0])
            if num == 0:
                continue

            # + log q(old): np.add.at so multi-leaf duplicate (temp, walker) indices
            # accumulate instead of last-write-wins on repeated fancy-index +=.
            # NaN logq values (e.g. from out-of-support old points) propagate into
            # lnpdiff.  NaN comparisons are always False, so such proposals are
            # unconditionally rejected — the "no-bias" claim in the class docstring
            # relies on this IEEE-754 property rather than on an explicit NaN guard.
            np.add.at(factors, where[:2], dist.logpdf(old_points))

            # Draw and - log q(new): sample_and_log_prob avoids a second forward
            # pass — the log-prob returned during sampling is reused directly.
            # Same NaN-rejection applies if the flow returns NaN for new points.
            new_points, logq_new = self.flow.sample_and_log_prob(
                num, context=self.active_condition
            )
            np.add.at(factors, where[:2], -logq_new)

            q[name][where] = new_points

        return q, factors
