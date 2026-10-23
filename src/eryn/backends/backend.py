# -*- coding: utf-8 -*-
"""In-memory chain storage: :class:`Backend`.

:class:`Backend` stores every saved step of an :class:`~eryn.ensemble.EnsembleSampler` run
(per-branch coordinates and ``inds``, log-likelihood, log-prior, inverse temperatures,
blobs, the running-sampler mask, acceptance counters, per-move acceptance fractions) in
``numpy`` arrays, and provides the getters (``get_chain``, ``get_log_like``, ...), the last
sample as a :class:`~eryn.state.State`, and diagnostics (autocorrelation time, evidence
estimates, Gelman-Rubin). :class:`~eryn.backends.HDFBackend` subclasses it to store the
same layout in an HDF5 file.
"""

import numpy as np

from ..state import State
from ..utils.utility import (
    get_integrated_act,
    psrf,
    stepping_stone_log_evidence,
    thermodynamic_integration_log_evidence,
)

__all__ = ["Backend"]


class Backend(object):
    """In-memory backend that stores the chain in ``numpy`` arrays.

    The sampler calls :meth:`reset` once (through
    :meth:`EnsembleSampler.reset <eryn.ensemble.EnsembleSampler.reset>`), :meth:`grow`
    before each :meth:`~eryn.ensemble.EnsembleSampler.sample` call with ``store=True``, and
    :meth:`save_step` at every stored step. The storage is allocated in :meth:`reset`;
    until then most attributes below do not exist.

    Storage layout: the raw arrays ALWAYS carry an independent-samplers axis directly
    after the step axis (size ``nsamplers``, default 1), for example ``chain[name]`` has
    shape ``(nsteps, nsamplers, ntemps, nwalkers, nleaves_max, ndim)``. The getters
    (``get_chain``, ``get_log_like``, ...) and the :attr:`accepted` / :attr:`rj_accepted` /
    :attr:`swaps_accepted` properties squeeze that axis away when ``nsamplers == 1``, so
    single-sampler users see ``(nsteps, ntemps, nwalkers, nleaves_max, ndim)`` and so on.
    Coordinates of inactive leaves are stored as ``store_missing_leaves``.

    Args:
        store_missing_leaves (float, optional): Value stored for the coordinates of
            inactive leaves. Default ``np.nan``.
        dtype (numpy.dtype, optional): Floating dtype of the stored arrays. ``None`` means
            ``np.float64``. Default ``None``.

    Attributes:
        initialized (bool): ``True`` once :meth:`reset` has run.
        iteration (int): Number of steps saved so far (next write index).
        chain (dict): Branch name to coordinates, raw shape
            ``(nsteps, nsamplers, ntemps, nwalkers, nleaves_max, ndim)``.
        inds (dict): Branch name to boolean active-leaf flags, raw shape
            ``(nsteps, nsamplers, ntemps, nwalkers, nleaves_max)``.
        log_like (np.ndarray): Log-likelihood, raw shape
            ``(nsteps, nsamplers, ntemps, nwalkers)``.
        log_prior (np.ndarray): Log-prior, raw shape ``(nsteps, nsamplers, ntemps, nwalkers)``.
        betas (np.ndarray): Inverse-temperature ladder at each step, raw shape
            ``(nsteps, nsamplers, ntemps)``. Only written when the saved state has
            ``betas`` (tempered runs).
        samplers_running (np.ndarray): Boolean running-sampler mask, raw shape
            ``(nsteps, nsamplers)``.
        blobs (np.ndarray or None): Blobs, raw shape
            ``(nsteps, nsamplers, ntemps, nwalkers, nblobs)``; ``None`` without blobs.
        random_state (object): Random state of the last saved step.
        move_info (dict or None): Move name to ``{"acceptance_fraction": array}``, each of
            shape ``(nsamplers * ntemps, nwalkers)``; ``None`` without tracked moves.
        move_keys (list of str): Tracked move names, in order (only with tracked moves).
        branch_names (list of str): Branch names.
        nbranches (int): Number of branches.
        ndims (dict): Dimensionality per branch.
        nleaves_max (dict): Maximum leaf count per branch.
        nwalkers (int): Walkers per temperature.
        ntemps (int): Temperatures per sampler.
        nsamplers (int): Number of independent samplers.
        rj (bool): Whether reversible-jump acceptance counts are stored.
        key_order (dict or None): Branch name to parameter key order.
        reset_args (tuple): Positional arguments of the last :meth:`reset`.
        reset_kwargs (dict): Keyword arguments of the last :meth:`reset`.
        dtype (numpy.dtype): Storage dtype.
        store_missing_leaves (float): Value stored for inactive leaves.

    Example:
        >>> from eryn.backends import Backend
        >>> backend = Backend()
        >>> backend.reset(8, 2, ntemps=3)
        >>> backend.shape
        {'model_0': (3, 8, 1, 2)}
        >>> backend.iteration
        0
    """

    def __init__(self, store_missing_leaves=np.nan, dtype=None):
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    def reset_base(self):
        """Call :meth:`reset` again with the arguments of the previous call.

        The previous extra ``info`` pairs come back as one ``info`` attribute (a dict)
        rather than as separate attributes.
        """
        self.reset(*self.reset_args, **self.reset_kwargs)

    # class-level default so legacy/unpickled backends still work
    nsamplers = 1

    def reset(
        self,
        nwalkers,
        ndims,
        nleaves_max=1,
        ntemps=1,
        branch_names=None,
        nbranches=1,
        rj=False,
        moves=None,
        key_order=None,
        nsamplers=1,
        **info,
    ):
        """Clear the backend and allocate empty storage.

        Sets ``iteration`` to 0, zeroes the acceptance counters, creates zero-length
        storage arrays (filled by :meth:`grow` / :meth:`save_step`) and, when ``moves`` is
        given, one zeroed acceptance-fraction array per move.

        Args:
            nwalkers (int): Walkers per temperature.
            ndims (int, list of int, np.ndarray, or dict): Dimensionality of each branch. An
                ``int`` is only valid for a single branch; a list or array follows
                ``branch_names``; a dict is keyed by branch name.
            nleaves_max (int, list of int, np.ndarray, or dict, optional): Maximum leaf count
                per branch, same forms as ``ndims``. Default ``1``.
            ntemps (int, optional): Temperatures per sampler. Default ``1``.
            branch_names (str or list of str, optional): Branch names. ``None`` means
                ``["model_0", ..., f"model_{nbranches - 1}"]``. Default ``None``.
            nbranches (int, optional): Number of branches; only used if ``branch_names`` is
                ``None``. Default ``1``.
            rj (bool, optional): If ``True``, also store reversible-jump acceptance counts.
                Default ``False``.
            moves (list of str, optional): Names of the tracked moves (the sampler's
                ``move_keys``, ``f"{ClassName}_{k}"``). ``None`` disables per-move
                acceptance storage. Default ``None``.
            key_order (dict, optional): Branch name to the list of parameter keys, for
                example ``{"model_0": ["x1", "x2", "x3"]}``. Default ``None``.
            nsamplers (int, optional): Number of independent samplers run simultaneously.
                Storage always carries a sampler axis directly after the step axis
                (e.g. chain shape ``(nsteps, nsamplers, ntemps, nwalkers, nleaves_max, ndim)``);
                getters squeeze that axis when ``nsamplers == 1`` so return shapes
                match the single-sampler layout. Default ``1``.
            **info: Extra key/value pairs set as attributes on the backend.

        Raises:
            ValueError: ``branch_names`` is not a str or list, or ``ndims`` /
                ``nleaves_max`` has an unsupported type or a key not in ``branch_names``.
            AssertionError: A list length (or an ``int`` for several branches) does not
                match the number of branches.
        """
        # store inputs for later resets
        self.reset_args = (nwalkers, ndims)
        self.reset_kwargs = dict(
            nleaves_max=nleaves_max,
            ntemps=ntemps,
            branch_names=branch_names,
            rj=rj,
            moves=moves,
            key_order=key_order,
            nsamplers=nsamplers,
            info=info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.nsamplers = int(nsamplers)
        self.rj = rj

        # turn things into lists/dicts if needed
        if branch_names is not None:
            if isinstance(branch_names, str):
                branch_names = [branch_names]

            elif not isinstance(branch_names, list):
                raise ValueError("branch_names must be string or list of strings.")

        else:
            branch_names = [f"model_{i}" for i in range(nbranches)]

        nbranches = len(branch_names)

        if isinstance(ndims, int):
            assert len(branch_names) == 1
            ndims = {branch_names[0]: ndims}

        elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
            assert len(branch_names) == len(ndims)
            ndims = {bn: nd for bn, nd in zip(branch_names, ndims)}

        elif isinstance(ndims, dict):
            assert len(list(ndims.keys())) == len(branch_names)
            for key in ndims:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in ndims but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("ndims is to be a scalar int, list or dict.")

        if isinstance(nleaves_max, int):
            assert len(branch_names) == 1
            nleaves_max = {branch_names[0]: nleaves_max}

        elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
            assert len(branch_names) == len(nleaves_max)
            nleaves_max = {bn: nl for bn, nl in zip(branch_names, nleaves_max)}

        elif isinstance(nleaves_max, dict):
            assert len(list(nleaves_max.keys())) == len(branch_names)
            for key in nleaves_max:
                if key not in branch_names:
                    raise ValueError(
                        f"{key} is in nleaves_max but does not appear in branch_names: {branch_names}."
                    )
        else:
            raise ValueError("nleaves_max is to be a scalar int, list, or dict.")

        self.nbranches = len(branch_names)

        self.branch_names = branch_names
        self.ndims = ndims
        self.nleaves_max = nleaves_max
        self.key_order = key_order 

        self.iteration = 0

        # setup all the holder arrays
        # cumulative counters always carry the sampler axis; the public
        # ``accepted`` / ``rj_accepted`` / ``swaps_accepted`` properties squeeze
        # it when nsamplers == 1
        self._accepted = np.zeros(
            (self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
        )
        self._swaps_accepted = np.zeros(
            (self.nsamplers, self.ntemps - 1), dtype=self.dtype
        )
        if self.rj:
            self._rj_accepted = np.zeros(
                (self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
            )

        else:
            self._rj_accepted = None

        # chains are stored in dictionaries
        self.chain = {
            name: np.empty(
                (
                    0,
                    self.nsamplers,
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[name],
                    self.ndims[name],
                ),
                dtype=self.dtype,
            )
            for name in self.branch_names
        }

        # inds correspond to leaves used or not
        self.inds = {
            name: np.empty(
                (0, self.nsamplers, self.ntemps, self.nwalkers, self.nleaves_max[name]),
                dtype=bool,
            )
            for name in self.branch_names
        }

        # log likelihood and prior
        self.log_like = np.empty(
            (0, self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
        )
        self.log_prior = np.empty(
            (0, self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
        )

        # temperature ladder
        self.betas = np.empty((0, self.nsamplers, self.ntemps), dtype=self.dtype)

        # which independent samplers were running at each step
        self.samplers_running = np.empty((0, self.nsamplers), dtype=bool)

        self.blobs = None

        self.random_state = None
        self.initialized = True

        # store move specific information
        if moves is not None:
            # setup info and keys
            self.move_info = {}
            self.move_keys = []
            for move in moves:
                # prepare information dictionary
                # stored FOLDED (nsamplers * ntemps, nwalkers): identical to the
                # legacy layout when nsamplers == 1
                self.move_info[move] = {
                    "acceptance_fraction": np.zeros(
                        (self.nsamplers * self.ntemps, self.nwalkers),
                        dtype=self.dtype,
                    )
                }

                # update the move keys to keep proper order
                self.move_keys.append(move)

        else:
            self.move_info = None

    @property
    def accepted(self):
        """np.ndarray: Cumulative in-model acceptance counts per walker over saved steps.

        Shape ``(ntemps, nwalkers)`` for ``nsamplers == 1``, else
        ``(nsamplers, ntemps, nwalkers)``.
        """
        return self._accepted[0] if self.nsamplers == 1 else self._accepted

    @property
    def rj_accepted(self):
        """np.ndarray or None: Cumulative reversible-jump acceptance counts per walker.

        Same shapes as :attr:`accepted`; ``None`` when ``rj`` is ``False``.
        """
        if self._rj_accepted is None:
            return None
        return self._rj_accepted[0] if self.nsamplers == 1 else self._rj_accepted

    @property
    def swaps_accepted(self):
        """np.ndarray: Cumulative accepted temperature swaps per adjacent rung pair.

        Shape ``(ntemps - 1,)`` for ``nsamplers == 1``, else ``(nsamplers, ntemps - 1)``.
        """
        return self._swaps_accepted[0] if self.nsamplers == 1 else self._swaps_accepted

    def _step_indexer(self, slice_vals, temp_index, sampler_index, has_temp_axis=True):
        """Build the (step, sampler[, temp]) indexing tuple for stored arrays.

        The sampler axis is squeezed (integer-indexed with 0) when
        ``nsamplers == 1`` and no explicit ``sampler_index`` is given, so
        return shapes match the legacy single-sampler layout. ``temp_index`` and
        ``sampler_index`` must be ``int`` or ``None`` (asserted).
        """
        if sampler_index is None:
            sampler_sel = 0 if self.nsamplers == 1 else slice(None)
        else:
            assert isinstance(sampler_index, int)
            sampler_sel = sampler_index

        if not has_temp_axis:
            return (slice_vals, sampler_sel)

        if temp_index is None:
            temp_sel = slice(None)
        else:
            assert isinstance(temp_index, int)
            temp_sel = temp_index

        return (slice_vals, sampler_sel, temp_sel)

    def has_blobs(self):
        """Return ``True`` if blobs are stored."""
        return self.blobs is not None

    def get_value(self, name, thin=1, discard=0, slice_vals=None, temp_index=None, sampler_index=None, branch_names=None):
        """Return one stored quantity, selected by step, temperature and sampler.

        The steps kept are ``discard + thin - 1``, ``discard + 2 * thin - 1``, ... up to
        :attr:`iteration`. The independent-sampler axis is removed when ``nsamplers == 1``
        or ``sampler_index`` is given, and the temperature axis when ``temp_index`` is
        given. With ``nsamplers == 1`` and no indices the shapes are
        ``(nsteps, ntemps, nwalkers, nleaves_max, ndim)`` for ``"chain"``,
        ``(nsteps, ntemps, nwalkers, nleaves_max)`` for ``"inds"``,
        ``(nsteps, ntemps, nwalkers)`` for ``"log_like"`` / ``"log_prior"``,
        ``(nsteps, ntemps)`` for ``"betas"``, ``(nsteps, ntemps, nwalkers, nblobs)`` for
        ``"blobs"`` and ``(nsteps,)`` for ``"samplers_running"``; ``nsamplers > 1`` adds an
        ``nsamplers`` axis after the step axis.

        Args:
            name (str): ``"chain"``, ``"inds"``, ``"log_like"``, ``"log_prior"``,
                ``"betas"``, ``"blobs"``, ``"samplers_running"``, or the name of another
                stored array with the same leading axes.
            thin (int, optional): Keep every ``thin``-th step. Default ``1``.
            discard (int, optional): Number of initial steps to drop (burn-in). Default
                ``0``.
            slice_vals (slice or np.ndarray, optional): Step index used directly instead of
                ``thin`` / ``discard``; only supported by
                :class:`~eryn.backends.HDFBackend` and must be ``None`` here. Default
                ``None``.
            temp_index (int, optional): Single temperature index. ``None`` keeps all.
                Default ``None``.
            sampler_index (int, optional): Integer for the desired independent-sampler
                index. If ``None``, returns all samplers (the sampler axis is squeezed
                away entirely when ``nsamplers == 1``). Default ``None``.
            branch_names (list of str, optional): Branches returned for ``"chain"`` /
                ``"inds"``; ``None`` returns all. Default ``None``.

        Returns:
            dict or np.ndarray or None: A dict keyed by branch name for ``"chain"`` and
            ``"inds"``; ``None`` for ``"blobs"`` when no blobs are stored; otherwise an
            array.

        Raises:
            ValueError: ``slice_vals`` is given.
            AttributeError: Nothing has been stored yet.
            AssertionError: ``temp_index`` or ``sampler_index`` is not an ``int``.
        """
        if slice_vals is not None:
            raise ValueError("slice_vals can only be used with an HDF Backend.")

        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        # check against blobs
        if name == "blobs" and not self.has_blobs():
            return None

        step_sel = slice(discard + thin - 1, self.iteration, thin)
        indexer = self._step_indexer(step_sel, temp_index, sampler_index)

        # make sure branch_names input is a list
        if branch_names is not None:
            if isinstance(branch_names, str):
                branches_names = [branch_names]

        branch_names_in = self.branch_names if branch_names is None else branch_names
        # prepare chain for output
        if name == "chain":
            v_all = {key: self.chain[key][indexer] for key in branch_names_in}
            return v_all

        # prepare inds for output
        if name == "inds":
            v_all = {key: self.inds[key][indexer] for key in branch_names_in}
            return v_all

        # samplers_running has no temperature axis
        if name == "samplers_running":
            indexer = self._step_indexer(
                step_sel, temp_index, sampler_index, has_temp_axis=False
            )
            return self.samplers_running[indexer]

        # all other requests can filter through array output
        # rather than the dictionary output used above
        v = getattr(self, name)[indexer]
        return v

    def get_chain(self, **kwargs):
        """Get the stored coordinates of every branch.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``, ``branch_names``.

        Returns:
            dict: Branch name to coordinates of shape
            ``(nsteps, ntemps, nwalkers, nleaves_max, ndim)`` (axes changed by the
            selection keywords and ``nsamplers`` as described in
            :meth:`~eryn.backends.Backend.get_value`). Inactive leaves hold the backend's
            ``store_missing_leaves`` value (``np.nan`` by default).
        """
        return self.get_value("chain", **kwargs)

    def get_autocorr_thin_burn(self):
        """Suggest ``(discard, thin)`` from the autocorrelation times.

        ``discard = int(2 * max(tau))`` and ``thin = int(0.5 * min(tau))`` over all
        parameters of all branches, with ``tau`` from :meth:`get_autocorr_time` (so the same
        restrictions apply: one temperature, no reversible jump, ``nsamplers == 1``).

        Returns:
            tuple[int, int]: ``(discard, thin)``.
        """

        # get the autocorrelation times
        tau = self.get_autocorr_time()

        # find the proper maximum
        tau_max = 0.0
        for name, values in tau.items():
            temp_max = np.max(values)
            tau_max = tau_max if tau_max > temp_max else temp_max

        discard = int(2 * tau_max)

        # find proper minimum
        tau_min = 1e10
        for name, values in tau.items():
            temp_min = np.min(values)
            tau_min = tau_min if tau_min < temp_min else temp_min

        thin = int(0.5 * tau_min)

        return (discard, thin)

    def get_inds(self, **kwargs):
        """Get the stored active-leaf flags (``inds``) of every branch.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``, ``branch_names``.

        Returns:
            dict: Branch name to boolean arrays of shape
            ``(nsteps, ntemps, nwalkers, nleaves_max)`` (axes changed as described in
            :meth:`~eryn.backends.Backend.get_value`); ``True`` marks a leaf that was in the
            model at that step.
        """
        return self.get_value("inds", **kwargs)

    def get_nleaves(self, **kwargs):
        """Get the number of active leaves of every walker at every stored step.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``, ``branch_names``.

        Returns:
            dict: Branch name to integer arrays of shape ``(nsteps, ntemps, nwalkers)``
            (the ``inds`` summed over the leaf axis).
        """
        inds = self.get_value("inds", **kwargs)
        nleaves = {name: np.sum(inds[name], axis=-1, dtype=int) for name in inds}
        return nleaves

    def get_blobs(self, **kwargs):
        """Get the stored blobs (extra values returned by the likelihood).

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``.

        Returns:
            np.ndarray or None: Blobs of shape ``(nsteps, ntemps, nwalkers, nblobs)`` (axes
            changed as described in :meth:`~eryn.backends.Backend.get_value`), or ``None``
            when no blobs are stored.
        """
        return self.get_value("blobs", **kwargs)

    def get_log_like(self, **kwargs):
        """Get the stored log-likelihood of every walker.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``.

        Returns:
            np.ndarray: Log-likelihood of shape ``(nsteps, ntemps, nwalkers)`` (axes changed
            as described in :meth:`~eryn.backends.Backend.get_value`).
        """
        return self.get_value("log_like", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the stored log-prior of every walker.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``.

        Returns:
            np.ndarray: Log-prior of shape ``(nsteps, ntemps, nwalkers)`` (axes changed as
            described in :meth:`~eryn.backends.Backend.get_value`).
        """
        return self.get_value("log_prior", **kwargs)

    def get_log_posterior(self, temper: bool = False, **kwargs):
        """Get the log-posterior ``beta * log_like + log_prior`` of every stored walker.

        Args:
            temper (bool, optional): If ``True``, weight the log-likelihood by the stored
                inverse temperatures; otherwise use ``beta = 1``. Default ``False``.
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``.

        Returns:
            np.ndarray: Log-posterior of shape ``(nsteps, ntemps, nwalkers)`` (axes changed
            as described in :meth:`~eryn.backends.Backend.get_value`).
        """
        if temper:
            betas = self.get_betas(**kwargs)

        else:
            betas = np.ones_like(self.get_betas(**kwargs))

        log_like = self.get_log_like(**kwargs)
        log_prior = self.get_log_prior(**kwargs)

        # betas is (nsteps, ntemps) for nsamplers == 1, (nsteps, nsamplers, ntemps) otherwise
        return betas[..., None] * log_like + log_prior

    def get_betas(self, **kwargs):
        """Get the stored inverse-temperature ladders.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`:
                ``thin``, ``discard``, ``slice_vals`` (HDF5 backend only), ``temp_index``,
                ``sampler_index``.

        Returns:
            np.ndarray: Inverse temperatures of shape ``(nsteps, ntemps)`` (axes changed as
            described in :meth:`~eryn.backends.Backend.get_value`). Only meaningful for
            tempered runs: the ladder is written only when the saved state carries
            ``betas``.
        """
        return self.get_value("betas", **kwargs)

    def get_samplers_running(self, **kwargs):
        """Get the stored running-sampler masks.

        Args:
            **kwargs: Selection keywords of :meth:`~eryn.backends.Backend.get_value`
                (``thin``, ``discard``, ``slice_vals``, ``sampler_index``).

        Returns:
            np.ndarray or None: Boolean mask of shape ``(nsteps, nsamplers)``
            (``(nsteps,)`` when ``nsamplers == 1``), or ``None`` for legacy HDF5 files that
            do not store it.
        """
        try:
            return self.get_value("samplers_running", **kwargs)
        except KeyError:
            # legacy backends/files have no samplers_running storage
            return None

    def get_a_sample(self, it):
        """Rebuild the :class:`~eryn.state.State` of one stored step.

        Fills coordinates, ``inds``, ``log_like``, ``log_prior``, ``betas`` (never written,
        so not meaningful, for untempered runs; see :meth:`get_betas`), blobs, the
        running-sampler mask (``nsamplers > 1``) and the backend's latest
        ``random_state`` (the random state is not stored per step). Supplemental
        information is not stored and is not restored.

        Args:
            it (int): Stored step index, ``0 <= it < iteration``.

        Returns:
            State: The state at step ``it`` (with ``nsamplers`` set from the backend).

        Raises:
            AttributeError: The backend is not initialized or nothing has been stored.
        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin
        # check for blobs
        blobs = self.get_blobs(discard=discard, thin=thin)
        if blobs is not None:
            blobs = blobs[0]

        # the running mask for this step (nsamplers > 1 only)
        if self.nsamplers > 1:
            samplers_running = self.get_samplers_running(discard=discard, thin=thin)
            if samplers_running is not None:
                samplers_running = samplers_running[0]
        else:
            samplers_running = None

        # fill a State with quantities from the last sample in the chain
        # for nsamplers > 1 the arrays come back with the sampler axis;
        # State folds them back into the internal folded representation
        sample = State(
            {
                name: temp[0]
                for name, temp in self.get_chain(discard=discard, thin=thin).items()
            },
            log_like=self.get_log_like(discard=discard, thin=thin)[0],
            log_prior=self.get_log_prior(discard=discard, thin=thin)[0],
            inds={
                name: temp[0]
                for name, temp in self.get_inds(discard=discard, thin=thin).items()
            },
            betas=self.get_betas(discard=discard, thin=thin)[0],
            blobs=blobs,
            random_state=self.random_state,
            nsamplers=self.nsamplers,
            samplers_running=samplers_running,
        )
        return sample

    def get_last_sample(self):
        """Rebuild the :class:`~eryn.state.State` of the last stored step.

        Equivalent to ``get_a_sample(iteration - 1)``; used to restart a run.

        Returns:
            State: The last stored state.

        Raises:
            AttributeError: The backend is not initialized or nothing has been stored.
        """
        it = self.iteration - 1

        # get the state from the last iteration
        last_sample = self.get_a_sample(it)
        return last_sample

    def get_autocorr_time(
        self, discard=0, thin=1, all_temps=False, multiply_thin=True, **kwargs
    ):
        """Estimate the integrated autocorrelation time of every parameter.

        Only defined for untempered runs without reversible jump and with
        ``nsamplers == 1``; otherwise compute it per chain from :meth:`get_chain`.

        Args:
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            thin (int, optional): Use every ``thin``-th step. Default ``1``.
            all_temps (bool, optional): Use all temperatures instead of the first one. Has
                no effect, since more than one temperature raises. Default ``False``.
            multiply_thin (bool, optional): If ``True``, multiply the result by ``thin`` so
                it is in units of stored steps before thinning. Default ``True``.
            **kwargs: Forwarded to :func:`eryn.utils.utility.get_integrated_act`
                (``window``, ``fast``, ``average``).

        Returns:
            dict: Branch name to the autocorrelation times of its parameters, as returned by
            :func:`~eryn.utils.utility.get_integrated_act` (times ``thin`` when
            ``multiply_thin``).

        Raises:
            ValueError: ``ntemps > 1``, reversible jump, or ``nsamplers > 1``.
        """
        # stopping index into temperatures
        ind = self.ntemps if all_temps else 1

        if self.ntemps > 1 or self.rj:
            raise ValueError(
                "get_autocorr_time is not well-defined for number of temperatures > 1 or when using reversible jump."
            )

        if self.nsamplers > 1:
            raise ValueError(
                "get_autocorr_time is not implemented for nsamplers > 1. "
                "Compute it per sampler with get_chain(sampler_index=...)."
            )

        # get chain
        x = self.get_chain(discard=discard, thin=thin)
        x = {name: value[:, :ind] for name, value in x.items()}

        out = get_integrated_act(x, **kwargs)

        # apply thinning factor if desired
        thin_factor = thin if multiply_thin else 1

        return {name: values * thin_factor for name, values in out.items()}

    def get_evidence_estimate(
        self, discard=0, thin=1, return_error=True, method="therodynamic", **ss_kwargs
    ):
        """Estimate the log-evidence from the tempered chains.

        Uses :func:`~eryn.utils.utility.thermodynamic_integration_log_evidence` on the
        per-temperature mean log-likelihood (non-finite values ignored), or
        :func:`~eryn.utils.utility.stepping_stone_log_evidence` on the full log-likelihood
        chain. The ladder must have been fixed over the selected steps (stop the
        adaptation with ``stop_adaptation`` and discard the adaptive phase). With
        ``nsamplers > 1`` and no ``sampler_index``, one estimate per sampler is returned.

        Args:
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            thin (int, optional): Use every ``thin``-th step. Default ``1``.
            return_error (bool, optional): If ``True``, also return the error estimate.
                Default ``True``.
            method (str, optional): ``"therodynamic"`` (default; also
                ``"thermodynamic integration"``, ``"thermo"``, ``"ti"``) or
                ``"stepping-stone"`` (also ``"stepping stone"``, ``"ss"``, ``"step"``,
                ``"stone"``); case-insensitive. Default ``"therodynamic"``.
            **ss_kwargs: ``sampler_index`` (one independent sampler) and, for the
                stepping-stone method, keyword arguments of
                :func:`~eryn.utils.utility.stepping_stone_log_evidence` (``block_len``,
                ``repeats``).

        Returns:
            float or tuple[float, float]: ``logZ``, or ``(logZ, dlogZ)`` when
            ``return_error``. With ``nsamplers > 1`` and no ``sampler_index``, arrays of
            shape ``(nsamplers,)`` instead of floats.

        Raises:
            ValueError: The inverse temperatures vary over the selected steps, or an unknown
                ``method``.
        """
        # per-sampler estimates when running independent samplers
        if self.nsamplers > 1 and "sampler_index" not in ss_kwargs:
            out = [
                self.get_evidence_estimate(
                    discard=discard,
                    thin=thin,
                    return_error=True,
                    method=method,
                    sampler_index=s,
                    **ss_kwargs,
                )
                for s in range(self.nsamplers)
            ]
            logZ = np.asarray([tmp[0] for tmp in out])
            dlogZ = np.asarray([tmp[1] for tmp in out])
            return (logZ, dlogZ) if return_error else logZ

        # get all the likelihood and temperature values
        sampler_index = ss_kwargs.pop("sampler_index", None)
        logls_all = self.get_log_like(
            discard=discard, thin=thin, sampler_index=sampler_index
        )
        betas_all = self.get_betas(
            discard=discard, thin=thin, sampler_index=sampler_index
        )

        # make sure that the betas were fixed during sampling (after burn in)
        if not (betas_all == betas_all[0]).all():
            raise ValueError(
                """Cannot compute evidence estimation if betas are allowed to vary. Use stop_adaptation
                kwarg in temperature settings."""
            )

        # setup information
        betas = betas_all[0]

        # get log evidence and error
        if method.lower() in [
            "therodynamic",
            "thermodynamic integration",
            "thermo",
            "ti",
        ]:
            logls = logls_all.copy()
            logls[~np.isfinite(logls)] = np.nan
            meanlogls = np.nanmean(logls, axis=(0, -1))
            logZ, dlogZ = thermodynamic_integration_log_evidence(betas, meanlogls)
            
        elif method.lower() in [
            "stepping stone",
            "ss",
            "step",
            "stone",
            "stepping-stone",
        ]:
            logZ, dlogZ = stepping_stone_log_evidence(betas, logls_all, **ss_kwargs)
        else:
            raise ValueError(
                """Please choose only between 'thermodynamic' and 'stepping-stone' methods."""
            )

        if return_error:
            return (logZ, dlogZ)
        else:
            return logZ

    def get_gelman_rubin_convergence_diagnostic(
        self, discard=0, thin=1, doprint=True, **psrf_kwargs
    ):
        """Compute the Gelman-Rubin convergence diagnostic ``Rhat`` per branch and temperature.

        Compares within-walker and between-walker variances with
        :func:`~eryn.utils.utility.psrf`; values far from 1 indicate non-convergence. With
        several leaves per walker, each walker's chain is the first ``min_leaves`` active
        leaves it visited (``min_leaves`` = smallest number over walkers). Based on Brooks
        & Gelman (1998), J. Comput. Graph. Stat. 7, 434-455, and Gelman & Rubin (1992),
        Stat. Sci. 7, 457-511.

        Args:
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            thin (int, optional): Use every ``thin``-th step. Default ``1``.
            doprint (bool, optional): If ``True``, print a table of the mean ``Rhat`` per
                branch and temperature. Default ``True``.
            **psrf_kwargs: Forwarded to :func:`~eryn.utils.utility.psrf` (for example
                ``per_walker``).

        Returns:
            dict: Branch name to ``{temperature index: Rhat}``, with ``Rhat`` as returned
            by :func:`~eryn.utils.utility.psrf`.

        Raises:
            NotImplementedError: ``nsamplers > 1``.
        """
        if self.nsamplers > 1:
            raise NotImplementedError(
                "get_gelman_rubin_convergence_diagnostic is not implemented for nsamplers > 1. "
                "Compute it per sampler with get_chain(sampler_index=...)."
            )

        Rhat_all_branches = dict()
        # Loop over the different models
        for branch in self.branch_names:

            Rhat = dict()  # Initialize
            # Loop over the temperatures
            for temp in range(self.ntemps):

                # Get all the chains per branch
                chains = self.get_chain(discard=discard, thin=thin)[branch][:, temp]

                # Handle the cases of multiple leaves on a given branch
                if chains.shape[2] == 1:
                    # If no multiple leaves, we squeeze and transpose to the
                    # right shape to pass to the psrf function, which is  (nwalkers, nsamples, ndim)
                    chains_in = chains.squeeze(axis=2).transpose((1, 0, 2))
                else:
                    # Project onto the model dim all chains [in case of RJ and multiple leaves per branch]
                    inds = self.get_inds(discard=discard, thin=thin)[branch][
                        :, temp
                    ]  # [t, w, nleavesmax, dim]
                    min_leaves = inds.sum(axis=(0, 2)).min()
                    tmp = [inds[:, w].flatten() for w in range(self.nwalkers)]
                    keep = [np.where(tmp[w])[0][:min_leaves] for w in range(len(tmp))]
                    chains_in = np.asarray(
                        [
                            chains[:, w].reshape(-1, self.ndims[branch])[keep[w]]
                            for w in range(self.nwalkers)
                        ]
                    )

                Rhat[temp] = psrf(chains_in, self.ndims[branch], **psrf_kwargs)
            Rhat_all_branches[branch] = Rhat  # Store the Rhat per branch

        if doprint:  # Print table of results
            print("  Gelman-Rubin diagnostic \n  <R̂>: Mean value for all parameters\n")
            print("  --------------")
            for branch in self.branch_names:
                print(f" Model: {branch}")
                print("   T \t <R̂>")
                print("  --------------")
                for temp in range(self.ntemps):
                    print(
                        f"   {temp:01d}\t{np.mean(Rhat_all_branches[branch][temp]):3.2f}"
                    )
                print("\n")

        return Rhat_all_branches

    @property
    def shape(self):
        """dict: Branch name to the per-step shape of its coordinates.

        ``(ntemps, nwalkers, nleaves_max, ndim)``, or
        ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)`` when ``nsamplers > 1``.
        """
        if self.nsamplers > 1:
            return {
                key: (
                    self.nsamplers,
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[key],
                    self.ndims[key],
                )
                for key in self.branch_names
            }
        return {
            key: (self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key])
            for key in self.branch_names
        }

    def _check_blobs(self, blobs):
        """Raise ``ValueError`` if blobs appear or disappear between calls."""
        # check if the setup for blobs is correct
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def _check_rj_accepted(self, rj_accepted):
        """Raise ``ValueError`` unless ``rj_accepted`` is given exactly when ``rj`` is set."""
        # check fi rj_accepted is setup properly
        if not self.rj and rj_accepted is not None:
            raise ValueError("inconsistent use of rj_accepted")
        if self.rj and rj_accepted is None:
            raise ValueError("inconsistent use of rj_accepted")

    def grow(self, ngrow, blobs):
        """Make room for ``ngrow`` more steps after the current :attr:`iteration`.

        Only the missing space is allocated (existing unused rows count). Called by
        :meth:`~eryn.ensemble.EnsembleSampler.sample` before storing.

        Args:
            ngrow (int): Number of steps that will be saved next.
            blobs (np.ndarray or None): Current blobs of the state, folded shape
                ``(nsamplers * ntemps, nwalkers, nblobs)``; their dtype and trailing shape
                define the blobs storage on first use. ``None`` without blobs.

        Raises:
            ValueError: Blobs appear or disappear relative to what is stored.
        """
        self._check_blobs(blobs)

        # determine the number of entries in the chains
        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)

        # temperary addition to chains
        a = {
            key: np.empty(
                (
                    i,
                    self.nsamplers,
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[key],
                    self.ndims[key],
                ),
                dtype=self.dtype,
            )
            for key in self.branch_names
        }
        # combine with original chain
        self.chain = {
            key: np.concatenate((self.chain[key], a[key]), axis=0) for key in a
        }

        # temperorary addition to inds
        a = {
            key: np.empty(
                (i, self.nsamplers, self.ntemps, self.nwalkers, self.nleaves_max[key]),
                dtype=bool,
            )
            for key in self.branch_names
        }
        # combine with original inds
        self.inds = {key: np.concatenate((self.inds[key], a[key]), axis=0) for key in a}

        # temperorary addition for log likelihood
        a = np.empty(
            (i, self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
        )
        # combine with original log likelihood
        self.log_like = np.concatenate((self.log_like, a), axis=0)

        # temperorary addition for log prior
        a = np.empty(
            (i, self.nsamplers, self.ntemps, self.nwalkers), dtype=self.dtype
        )
        # combine with original log prior
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)

        # temperorary addition for betas
        a = np.empty((i, self.nsamplers, self.ntemps), dtype=self.dtype)
        # combine with original betas
        self.betas = np.concatenate((self.betas, a), axis=0)

        # temperorary addition for samplers_running
        a = np.empty((i, self.nsamplers), dtype=bool)
        self.samplers_running = np.concatenate((self.samplers_running, a), axis=0)

        if blobs is not None:
            # blobs come in folded: (nsamplers * ntemps, nwalkers, ...)
            dt = np.dtype((blobs.dtype, blobs.shape[2:]))
            # temperorary addition for blobs
            a = np.empty(
                (i, self.nsamplers, self.ntemps, self.nwalkers), dtype=dt
            )
            # combine with original blobs
            if self.blobs is None:
                self.blobs = a
            else:
                self.blobs = np.concatenate((self.blobs, a), axis=0)

    def _check(
        self,
        state,
        accepted,
        rj_accepted=None,
        swaps_accepted=None,
    ):
        """Check all the information going in is okay.

        The incoming ``state`` and acceptance arrays are FOLDED: the leading
        axis has size ``nsamplers * ntemps`` (unfolded ``(nsamplers, ntemps, ...)``
        acceptance arrays and betas are accepted too). Raises ``ValueError`` on any shape
        mismatch or inconsistent blobs / ``rj_accepted``.
        """
        self._check_blobs(state.blobs)
        self._check_rj_accepted(rj_accepted)

        has_blobs = self.has_blobs()

        ntemps, nwalkers, nsamplers = self.ntemps, self.nwalkers, self.nsamplers
        ntemps_eff = nsamplers * ntemps

        # make sure all of the coordinate and inds dimensions are okay
        for key in self.branch_names:
            ntemp1, nwalker1, nleaves1, ndim1 = state.branches[key].shape
            nleaves2 = self.nleaves_max[key]
            ndim2 = self.ndims[key]

            if (ntemp1, nwalker1, ndim1) != (
                ntemps_eff,
                nwalkers,
                ndim2,
            ) or nleaves1 > nleaves2:
                raise ValueError(
                    f"invalid coordinate dimensions for model {key} with shape {state.branches[key].shape}; expected {(ntemps_eff, nwalkers, nleaves2, ndim2)}"
                )

            if (ntemp1, nwalker1, nleaves1) != state.branches[key].inds.shape:
                raise ValueError(
                    f"invalid inds dimensions for model {key} with shape {state.branches[key].inds.shape}; expected {(ntemp1, nwalker1, nleaves1)}"
                )

        # make sure log likelihood, log prior, blobs, accepted, rj_accepted, betas are okay
        if state.log_like.shape != (
            ntemps_eff,
            nwalkers,
        ):
            raise ValueError(
                f"invalid log probability size; expected {(ntemps_eff, nwalkers)}"
            )
        if state.log_prior.shape != (
            ntemps_eff,
            nwalkers,
        ):
            raise ValueError(
                f"invalid log prior size; expected {(ntemps_eff, nwalkers)}"
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and state.blobs.shape[:2] != (ntemps_eff, nwalkers):
            raise ValueError(
                f"invalid blobs size; expected {(ntemps_eff, nwalkers)}"
            )
        if accepted.shape not in [
            (ntemps_eff, nwalkers),
            (nsamplers, ntemps, nwalkers),
        ]:
            raise ValueError(
                f"invalid acceptance size; expected {(ntemps_eff, nwalkers)}"
            )

        if swaps_accepted is not None and swaps_accepted.shape not in [
            (ntemps - 1,) if nsamplers == 1 else (nsamplers, ntemps - 1),
            (nsamplers, ntemps - 1),
        ]:
            raise ValueError(
                f"invalid swaps_accepted size; expected {(nsamplers, ntemps - 1)}"
            )
        if self.rj:
            if rj_accepted.shape not in [
                (ntemps_eff, nwalkers),
                (nsamplers, ntemps, nwalkers),
            ]:
                raise ValueError(
                    f"invalid rj acceptance size; expected {(ntemps_eff, nwalkers)}"
                )

        if state.betas is not None and state.betas.shape not in [
            (ntemps_eff,),
            (nsamplers, ntemps),
        ]:
            raise ValueError(f"invalid beta size; expected {ntemps_eff}")

    def get_move_info(self):
        """Get the stored per-move information.

        Returns:
            dict or None: Move name (``f"{ClassName}_{k}"``) to
            ``{"acceptance_fraction": np.ndarray}`` with the latest acceptance fractions,
            shape ``(nsamplers * ntemps, nwalkers)``; ``None`` when no moves are tracked.
        """
        return self.move_info

    def save_step(
        self,
        state,
        accepted,
        rj_accepted=None,
        swaps_accepted=None,
        moves_accepted_fraction=None,
    ):
        """Write one step at index :attr:`iteration` and advance it.

        Stores the coordinates (``store_missing_leaves`` on inactive leaves), ``inds``,
        ``log_like``, ``log_prior``, blobs, ``betas`` (only if the state has them) and the
        running-sampler mask (all ``True`` if the state has none); adds the acceptance
        counts to the cumulative counters; overwrites the per-move acceptance fractions;
        and keeps ``state.random_state``. :meth:`grow` must have made room first.

        Args:
            state (State): State to save, folded leading axis ``nsamplers * ntemps``.
            accepted (np.ndarray): In-model acceptance counts of this step, shape
                ``(nsamplers * ntemps, nwalkers)`` (or ``(nsamplers, ntemps, nwalkers)``).
            rj_accepted (np.ndarray, optional): Reversible-jump acceptance counts, same
                shape as ``accepted``; required when ``rj`` is set and forbidden otherwise.
                Default ``None``.
            swaps_accepted (np.ndarray, optional): Accepted swaps per adjacent rung pair,
                shape ``(ntemps - 1,)`` (``(nsamplers, ntemps - 1)`` when
                ``nsamplers > 1``). Default ``None``.
            moves_accepted_fraction (dict, optional): Move name to its acceptance-fraction
                array, with every key of ``move_keys``. Default ``None``.

        Raises:
            ValueError: A shape mismatch, inconsistent blobs or ``rj_accepted``, or
                ``moves_accepted_fraction`` without tracked moves.
        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            rj_accepted=rj_accepted,
            swaps_accepted=swaps_accepted,
        )

        # incoming state arrays are FOLDED; storage carries the sampler axis,
        # so unfold with cheap reshapes at this write boundary
        nsamplers, ntemps = self.nsamplers, self.ntemps

        # save the coordinates and inds
        for key, model in state.branches.items():
            self.inds[key][self.iteration] = model.inds.reshape(
                (nsamplers, ntemps) + model.inds.shape[1:]
            )
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = model.coords * model.inds[:, :, :, None]

            inds_all = np.repeat(model.inds, model.coords.shape[-1], axis=-1).reshape(
                model.inds.shape + (model.coords.shape[-1],)
            )
            coords_in[~inds_all] = self.store_missing_leaves
            self.chain[key][self.iteration] = coords_in.reshape(
                (nsamplers, ntemps) + coords_in.shape[1:]
            )

        # save higher level quantities
        self.log_like[self.iteration] = state.log_like.reshape(
            (nsamplers, ntemps) + state.log_like.shape[1:]
        )
        self.log_prior[self.iteration] = state.log_prior.reshape(
            (nsamplers, ntemps) + state.log_prior.shape[1:]
        )
        if state.blobs is not None:
            self.blobs[self.iteration] = state.blobs.reshape(
                (nsamplers, ntemps) + state.blobs.shape[1:]
            )
        if state.betas is not None:
            self.betas[self.iteration] = state.betas.reshape(nsamplers, ntemps)

        samplers_running = getattr(state, "samplers_running", None)
        self.samplers_running[self.iteration] = (
            samplers_running
            if samplers_running is not None
            else np.ones(nsamplers, dtype=bool)
        )

        self._accepted += np.reshape(accepted, self._accepted.shape)

        if swaps_accepted is not None:
            self._swaps_accepted += np.reshape(
                swaps_accepted, self._swaps_accepted.shape
            )
        if self.rj:
            self._rj_accepted += np.reshape(rj_accepted, self._rj_accepted.shape)

        # moves
        if moves_accepted_fraction is not None:
            if self.move_info is None:
                raise ValueError(
                    """moves_accepted_fraction was passed, but moves_info was not initialized. Use the moves kwarg 
                    in the reset function."""
                )

            # update acceptance fractions
            for move_key in self.move_keys:
                self.move_info[move_key]["acceptance_fraction"][:] = (
                    moves_accepted_fraction[move_key]
                )

        self.random_state = state.random_state
        self.iteration += 1

    def get_info(self, discard=0, thin=1):
        """Collect the stored chain and run metadata into one dict for diagnostics or plots.

        Calls :meth:`get_autocorr_time` without catching its error, so it only works for
        untempered runs without reversible jump and with ``nsamplers == 1``.

        Args:
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            thin (int, optional): Use every ``thin``-th step. Default ``1``.

        Returns:
            dict: Keys ``"samples"`` (:meth:`get_chain`), ``"thin"``, ``"burn"``,
            ``"log_like"``, ``"betas"``, ``"inds"``, ``"shapes"`` (:attr:`shape`),
            ``"ntemps"``, ``"nsamplers"``, ``"nwalkers"``, ``"nbranches"``,
            ``"branch names"``, ``"ndims"``, ``"tau"`` (autocorrelation times of the full
            chain), and ``"ac_burn"`` / ``"ac_thin"`` (``int(2 * max(tau))`` and
            ``int(0.5 * min(tau))``, with ``"ac_thin"`` raised to at least 1; both 1 if
            they cannot be computed).

        Raises:
            ValueError: From :meth:`get_autocorr_time` (``ntemps > 1``, reversible jump, or
                ``nsamplers > 1``).
        """
        samples = self.get_chain(discard=discard, thin=thin)

        # add all information that would be needed from backend
        out_info = dict(samples=samples)
        out_info["thin"] = thin
        out_info["burn"] = discard

        tau = self.get_autocorr_time()

        # get log prob
        out_info["log_like"] = self.get_log_like(thin=thin, discard=discard)

        # get temperatures
        out_info["betas"] = self.get_betas(thin=thin, discard=discard)

        # get inds
        out_info["inds"] = self.get_inds(thin=thin, discard=discard)

        out_info["shapes"] = self.shape
        out_info["ntemps"] = self.ntemps
        out_info["nsamplers"] = self.nsamplers
        out_info["nwalkers"] = self.nwalkers
        out_info["nbranches"] = self.nbranches
        out_info["branch names"] = self.branch_names
        out_info["ndims"] = self.ndims
        out_info["tau"] = tau

        try:
            out_info["ac_burn"] = int(2 * np.max(list(tau.values())))
            out_info["ac_thin"] = int(0.5 * np.min(list(tau.values())))
        except Exception as e:
            print(
                f"Failed to calculate the autocorrelation length. Will not output this piece of information. \n\n Actual error: [{e}]"
            )
            out_info["ac_thin"] = 1
            out_info["ac_burn"] = 1

        if out_info["ac_thin"] < 1:
            out_info["ac_thin"] = 1

        return out_info

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
