# -*- coding: utf-8 -*-
"""The ensemble sampler: Eryn's main entry point.

:class:`EnsembleSampler` drives a run. It parses the branches (models), priors, in-model and
reversible-jump moves, the tempering setup and the storage backend, then advances the ensemble
with :meth:`EnsembleSampler.sample` (a generator) or :meth:`EnsembleSampler.run_mcmc`.
:func:`walkers_independent` is the emcee-style check that the initial walkers are linearly
independent.

Inside the sampler the coordinates of branch ``name`` have shape
``(ntemps, nwalkers, nleaves_max[name], ndims[name])`` and a boolean ``inds`` array of shape
``(ntemps, nwalkers, nleaves_max[name])`` marks the active leaves. With ``nsamplers > 1``
independent samplers, the sampler axis is folded into the temperature axis (leading axis
``nsamplers * ntemps``).
"""

import warnings
from copy import deepcopy
from itertools import count

import numpy as np

from .backends import Backend, HDFBackend
from .model import Model
from .moves import (
    DistributionGenerateRJ,
    GaussianMove,
    MultipleTryMove,
    StretchMove,
    TemperatureControl,
)
from .pbar import get_progress_bar
from .prior import ProbDistContainer
from .state import State
from .utils import PeriodicContainer, PlotContainer
from .utils.utility import groups_from_inds

__all__ = ["EnsembleSampler", "walkers_independent"]


from collections.abc import Iterable


class EnsembleSampler(object):
    """Ensemble MCMC sampler with parallel tempering, reversible jump and multiple branches.

    The class controls the entire sampling run, from a basic untempered MCMC to a
    parallel-tempered global fit with several branches (models), each holding a variable
    number of leaves (sources). See the `tree metaphor
    <https://lisa-analysis-tools.github.io/Eryn/Eryn_tutorial.html#The-Tree-Metaphor>`_
    for a basic explainer.

    Coordinates of branch ``name`` have shape ``(ntemps, nwalkers, nleaves_max[name],
    ndims[name])`` inside the sampler; a boolean ``inds`` array of shape
    ``(ntemps, nwalkers, nleaves_max[name])`` marks the active leaves (``True`` = in the
    model at this step). Each sampler iteration (one "proposal iteration", see
    :meth:`sample`) runs ``num_repeats_in_model`` in-model proposals, each with a move
    drawn at random from ``moves`` (with ``weights``), then, when reversible jump is on,
    ``num_repeats_rj`` between-model proposals drawn from ``rj_moves``. Temperature swaps
    are done by the moves themselves through
    :class:`~eryn.moves.tempering.TemperatureControl` (reversible-jump moves swap without
    ladder adaptation); the swap counts stored in the backend come from the in-model moves.

    Args:
        nwalkers (int): Number of walkers per temperature (per sampler when
            ``nsamplers > 1``).
        ndims (int, list of int, np.ndarray, or dict): Dimensionality of each branch. An
            ``int`` is only valid for a single branch; a list or array is matched to
            ``branch_names`` in order; a dict is keyed by branch name.
        log_like_fn (callable): Natural log of the likelihood. It is called as
            ``log_like_fn(x, [groups,] *args, [supps=..., branch_supps=...,] **kwargs)``,
            where ``args`` / ``kwargs`` are the arguments below, ``groups`` is passed only
            when ``vectorize`` and ``provide_groups`` are both ``True``, and ``supps`` /
            ``branch_supps`` only when ``provide_supplemental`` is ``True``. The form of
            ``x`` depends on ``vectorize`` and on the number of branches ``N``:

            - ``vectorize=False``: one call per walker that has at least one active leaf
              (through ``pool.map`` if ``pool`` is given, else ``map``). For each branch,
              the walker's active leaves form an array of shape ``(nleaves_active, ndim)``;
              without reversible jump a single active leaf is passed as a 1D ``(ndim,)``
              array. With ``N == 1`` that array is ``x``; with ``N > 1``, ``x`` is the
              list over branches (``None`` for a branch with no active leaf). The call
              returns a scalar (or a 1D array: log-likelihood followed by blobs).
            - ``vectorize=True``: one call for the whole ensemble. With ``N == 1``, ``x`` is
              the 2D array of all active leaves, shape ``(total active leaves, ndim)``; with
              ``N > 1`` it is the list ``[X0, X1, ...]`` over branches. ``groups`` (``group0``
              or ``[group0, group1, ...]``) gives, for each leaf, the index of the walker
              it belongs to; leaves with the same index enter the same likelihood value.
              Indices are compacted to ``0 ... ngroups - 1`` over the walkers that have at
              least one active leaf and a finite log-prior, in flattened
              ``(ntemps, nwalkers)`` order. Example: 3 walkers with ``(1, 2, 1)`` leaves
              give ``X0 = array([p0, p1, p2, p3])`` and ``group0 = array([0, 1, 1, 2])``.
              The call returns an array of shape ``(ngroups,)``, or ``(ngroups, 1 + nblobs)``
              whose first column is the log-likelihood and the rest blobs.

            Walkers with no active leaf in any branch, or with a log-prior of ``-inf``, are
            not evaluated: their log-likelihood is set to ``fill_zero_leaves_val``. As
            written, :meth:`compute_log_like` reshapes the blobs to ``(ntemps, nwalkers)``,
            so only a single blob column is supported. See the
            `tutorial <https://lisa-analysis-tools.github.io/Eryn/Eryn_tutorial.html#>`_.
        priors (dict or ProbDistContainer): Prior of each branch. Accepted forms:

            - a :class:`~eryn.priors.ProbDistContainer`, used for a single branch and
              stored under the name ``"model_0"`` (the branch must keep that default
              name);
            - a dict keyed by branch name whose values are either a dict
              ``{parameter key: distribution}`` (each distribution needs a ``logpdf``
              method; the dict is wrapped in a :class:`~eryn.priors.ProbDistContainer`),
              a :class:`~eryn.priors.ProbDistContainer`, or any object with a ``logpdf``
              method and a ``key_order`` attribute.

            The special key ``"all_models_together"`` replaces the per-branch evaluation
            with one call ``logpdf(coords, inds, supps=supps, branch_supps=branch_supps)``
            on the coordinate and ``inds`` dicts (per-branch shapes
            ``(ntemps, nwalkers, nleaves_max, ndim)`` and ``(ntemps, nwalkers,
            nleaves_max)``), which must return an array of shape ``(ntemps, nwalkers)``.
            That object must also carry a ``key_order`` attribute.
        provide_groups (bool, optional): If ``True``, pass the leaf-group indices to a
            vectorized ``log_like_fn`` (see above), and evaluate the prior only on active
            leaves in :meth:`compute_log_prior`. Default ``False``.
        provide_supplemental (bool, optional): If ``True``, pass the ``supps``
            (``state.supplemental``) and ``branch_supps`` (per-branch
            :class:`~eryn.state.BranchSupplemental`) keyword arguments to ``log_like_fn``,
            indexed to the walkers / leaves of each call. Default ``False``.
        tempering_kwargs (dict, optional): Keyword arguments for
            :class:`~eryn.moves.tempering.TemperatureControl` (for example ``ntemps``,
            ``betas``, ``Tmax``, ``adaptive``, ``stop_adaptation``). Its effective
            dimension is ``sum(nleaves_max[b] * ndims[b])`` over branches. The default
            ``{}`` means no tempering: ``ntemps = 1`` and ``temperature_control = None``.
        branch_names (str or list of str, optional): Branch names. A ``str`` is one
            branch. If ``None``, the names are ``f"model_{i}"`` for ``i`` in
            ``range(nbranches)``. Default ``None``.
        nbranches (int, optional): Number of branches; only used if ``branch_names`` is
            ``None``. Default ``1``.
        nleaves_max (int, list of int, np.ndarray, or dict, optional): Maximum leaf count
            per branch, same forms as ``ndims``. Default ``1``.
        nleaves_min (int, list of int, np.ndarray, dict, or None, optional): Minimum leaf
            count per branch for the default reversible-jump move. Only parsed when
            ``rj_moves`` is ``True`` or a str; ``None`` then means 0 for every branch.
            User-supplied RJ moves carry their own minimum. Default ``0``.
        pool (object, optional): Object with a ``map`` method following the built-in
            ``map`` calling sequence, used for the per-walker likelihood calls when
            ``vectorize`` is ``False``. Pickling the sampler sets ``self.pool`` to ``None``
            (on the live object as well). Default ``None``.
        moves (Move, list of Move, or list of (Move, float), optional): In-model
            proposals: one move, a list of moves (equal weights), or a list of
            ``(move, weight)`` pairs (weights are normalized). One move is drawn at random
            per in-model proposal. ``None`` uses ``StretchMove(a=2.0)``. Moves without a
            ``temperature_control`` or ``periodic`` receive the sampler's. Default
            ``None``.
        rj_moves (Move, list, bool, str, or None, optional): Reversible-jump proposals.
            ``None`` or ``False``: no reversible jump. A move, a list of moves, or a list of
            ``(move, weight)`` pairs: drawn at random per RJ proposal. ``True`` or
            ``"together"``: one :class:`~eryn.moves.DistributionGenerateRJ` (prior draws)
            that proposes changes to all branches at once. ``"iterate_branches"``: one
            :class:`~eryn.moves.DistributionGenerateRJ` that updates the branches one by
            one, in ``branch_names`` order, within each RJ proposal.
            ``"separate_branches"``: one :class:`~eryn.moves.DistributionGenerateRJ` per
            branch, chosen with equal probability (recommended with several branches).
            Any other str raises ``ValueError``. Giving ``rj_moves`` (anything but
            ``None``) requires ``moves``. Default ``None``.
        dr_moves (bool or Move, optional): Delayed rejection for the default RJ move built
            from ``rj_moves=True`` or a str (passed as its ``dr`` argument; ignored for
            user-supplied RJ moves). ``True`` wraps a
            :class:`~eryn.moves.DistributionGenerate` over the RJ move's generating
            distribution (the priors) in
            :class:`~eryn.moves.delayedrejection.DelayedRejection`; a move object is used as
            the delayed-rejection proposal; ``None`` or ``False`` disables it. The wrapper
            is built, but the delayed-rejection branch of
            :meth:`~eryn.moves.ReversibleJumpMove.propose` is not implemented and raises
            ``NotImplementedError`` on the first RJ proposal, so only ``None`` / ``False``
            can run. Default ``None``.
        dr_max_iter (int, optional): Maximum number of delayed-rejection iterations.
            Default ``5``.
        args (list, optional): Extra positional arguments appended after the sampler's
            arguments in every ``log_like_fn`` call. Default ``None`` (none).
        kwargs (dict, optional): Extra keyword arguments for every ``log_like_fn`` call.
            Default ``None`` (none).
        backend (Backend or str, optional): Chain storage. ``None`` creates an in-memory
            :class:`~eryn.backends.Backend`; a str is the file name of a new
            :class:`~eryn.backends.HDFBackend`; any :class:`~eryn.backends.Backend`
            (sub)class instance is used as given. An already-initialized backend is
            resumed (see Note). Default ``None``.
        vectorize (bool, optional): If ``True``, ``log_like_fn`` is called once for all
            walkers (see above) and ``pool`` is not used. Default ``False``.
        blobs_dtype (numpy.dtype, optional): Stored as ``self.blobs_dtype``; not used by the
            sampler. Default ``None``.
        plot_iterations (int, optional): Produce runtime diagnostic plots every
            ``plot_iterations`` yielded states in :meth:`run_mcmc` (not in
            :meth:`sample`). ``<= 0`` disables plotting; ``> 0`` is not supported with
            ``nsamplers > 1``. Default ``-1``.
        plot_generator (PlotContainer, optional): Plot generator used when
            ``plot_iterations > 0``; its ``backend`` is set to the sampler's backend. If
            ``None``, a :class:`~eryn.utils.PlotContainer` with ``plots=['base', 'rj']``
            and ``discard=0.2`` is created. Default ``None``.
        plot_folder (str, optional): Output folder of the default plot generator. ``None``
            means ``"./runtime_plots"``. Default ``None``.
        periodic (dict or PeriodicContainer, optional): Periodic parameters, keyed by
            branch name; each value maps a parameter index to its period. A dict is wrapped
            in :class:`~eryn.utils.PeriodicContainer` without ``key_order``, so it must use
            integer indices (pass a ``PeriodicContainer`` built with ``key_order`` for
            string names). Given to every move that has no ``periodic`` of its own.
            Default ``None``.
        update_fn (callable, optional): Called as ``update_fn(i, state, sampler)`` inside
            :meth:`sample` every ``update_iterations`` proposal iterations, where ``i`` is
            the 0-based proposal counter of the current :meth:`sample` call, for example an
            :class:`~eryn.utils.AdjustStretchProposalScale`. Default ``None``.
        update_iterations (int, optional): Proposal iterations between ``update_fn`` calls
            (counted per proposal iteration, independent of ``thin_by``). ``<= 0`` disables
            updates. Default ``-1``.
        stopping_fn (callable, optional): Called as ``stopping_fn(i, state, sampler)`` by
            :meth:`run_mcmc` every ``stopping_iterations`` yielded states, that is every
            ``thin_by * stopping_iterations`` proposal iterations; a truthy return ends the
            run. Not called by :meth:`sample`. For example an
            :class:`~eryn.utils.Stopping` subclass. Default ``None``.
        stopping_iterations (int, optional): Yielded states between ``stopping_fn``
            checks. ``<= 0`` disables the checks. Default ``-1``.
        fill_zero_leaves_val (float, optional): Log-likelihood given to walkers that are
            not evaluated (no active leaf in any branch, or ``-inf`` log-prior). To keep the
            zero-leaf model possible, set it to the log-likelihood of the data alone.
            Default ``-1e300``.
        num_repeats_in_model (int, optional): In-model proposals per sampler iteration.
            The stored acceptance counts sum over the repeats, so divide
            :attr:`acceptance_fraction` by this number. Default ``1``.
        num_repeats_rj (int, optional): Reversible-jump proposals per sampler iteration;
            same caveat for :attr:`rj_acceptance_fraction`. Default ``1``.
        track_moves (bool, optional): If ``True``, store each move's acceptance fraction in
            the backend under ``f"{ClassName}_{k}"`` (``k`` counts moves of the same class).
            When resuming an initialized backend the move names must match the stored ones
            (``ValueError`` otherwise; start a new backend from the last state instead).
            Reordering moves of the same class is not detected and would mix their
            acceptance fractions. Default ``True``.
        nsamplers (int, optional): Number of fully independent samplers (ensembles) to run
            simultaneously. All samplers' likelihood evaluations are batched into single
            calls (useful for GPU likelihoods). Each sampler carries its own temperature
            ladder; temperature swaps and ladder adaptation are per sampler. Initial
            coordinates then take shape ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)``
            and stored/returned quantities gain a leading ``nsamplers`` axis (after the
            step axis). With the default of 1, nothing changes anywhere user-facing.
            Internally the sampler axis is folded into the temperature axis, so proposals
            simply see ``nsamplers * ntemps`` "temperatures". Use
            ``State.samplers_running`` (bool array of shape ``(nsamplers,)``) to freeze
            converged samplers: frozen samplers get a ``-inf`` log-prior, so they are
            skipped in likelihood evaluation, always reject, and are excluded from
            tempering. Default ``1``.
        prior_transform_fn (object or dict, optional): Per-sampler prior-basis transform
            (the counterpart of the deprecated ``ParaEnsembleSampler`` argument). A dict
            maps branch names to objects implementing
            ``transform_to_prior_basis(coords, samplers_running_idx)`` (in-place map of
            ``(num_running, ntemps, nwalkers, ndim)`` coordinates to the basis in which
            ``priors`` is defined) and ``adjust_logp(logp, samplers_running_idx)`` (in-place
            Jacobian adjustment of the ``(num_running, ntemps, nwalkers)`` log-prior);
            ``samplers_running_idx`` holds the integer indices of the running samplers. A
            single object is allowed for a single branch. Useful for per-sampler prior
            bounds (e.g. per-band frequency limits). Requires ``nleaves_max == 1`` for the
            transformed branch and is incompatible with reversible jump, multiple-try
            moves, ``provide_groups``, and ``all_models_together`` priors. Default
            ``None``.
        info (dict, optional): Extra key/value pairs passed as ``backend.reset(**info)``
            when the backend is not yet initialized (they become backend attributes; an
            :class:`~eryn.backends.HDFBackend` also writes them to the file). Default
            ``{}``.

    Raises:
        ValueError: Invalid ``branch_names``, ``ndims``, ``nleaves_max``, ``nleaves_min``,
            ``priors``, ``periodic``, ``rj_moves`` str or ``prior_transform_fn``;
            ``rj_moves`` without ``moves``; ``nsamplers < 1``; a move whose temperature
            control has another ``nsamplers``; runtime plotting with ``nsamplers > 1``; or,
            when resuming, a backend whose moves, key order, ``nsamplers`` or shapes differ.
        AssertionError: ``ndims`` / ``nleaves_max`` (or ``nleaves_min``) length does not
            match the number of branches, or an ``int`` is given for several branches.

    Attributes:
        backend (Backend): Chain storage.
        branch_names (list of str): Branch names, in order.
        ndims (dict): Dimensionality per branch.
        nleaves_max (dict): Maximum leaf count per branch.
        nleaves_min (dict): Minimum leaf count per branch; only set when ``rj_moves`` is
            ``True`` or a str.
        nwalkers (int): Walkers per temperature.
        ntemps (int): Temperatures per sampler.
        nsamplers (int): Number of independent samplers.
        ntemps_eff (int): Folded leading-axis size ``nsamplers * ntemps``.
        temperature_control (TemperatureControl or None): Tempering controller.
        moves (list of Move): In-model moves; ``weights`` holds their probabilities.
        rj_moves (list of Move or None): RJ moves; ``rj_weights`` holds their
            probabilities. ``None`` for ``rj_moves=False``; not set at all for the default
            ``rj_moves=None``.
        has_reversible_jump (bool): Whether RJ proposals are made.
        all_moves (dict): Tracked moves keyed by ``f"{ClassName}_{k}"`` (empty when
            ``track_moves`` is ``False``).
        key_order (dict): Branch name to the parameter ``key_order`` of its prior.
        log_like_fn (callable): ``log_like_fn`` wrapped with ``args`` / ``kwargs``.
        periodic (PeriodicContainer or None): Periodic-parameter container.

    Note:
        When ``backend`` is already initialized (for example an existing HDF5 file), the
        sampler checks it against the configuration and resumes: the random state is read
        from the backend and the last stored sample becomes the default starting point, so
        ``run_mcmc(None, nsteps)`` continues the chain. As written this fails for
        untempered runs: the backend never writes ``betas`` there, but the restored
        :class:`~eryn.state.State` still carries the (zero or uninitialized) stored rows,
        and :meth:`sample` then assigns them to a ``None`` temperature control
        (``AttributeError``).

    Example:
        >>> import numpy as np
        >>> from eryn.ensemble import EnsembleSampler
        >>> from eryn.priors import ProbDistContainer, UniformDistribution
        >>> def log_like_fn(x, mu, invcov):
        ...     diff = x - mu
        ...     return -0.5 * diff @ invcov @ diff
        >>> ndim, nwalkers = 3, 16
        >>> priors = ProbDistContainer(
        ...     {f"x{i}": UniformDistribution(-5.0, 5.0) for i in range(ndim)}
        ... )
        >>> sampler = EnsembleSampler(
        ...     nwalkers, ndim, log_like_fn, priors, args=[np.zeros(ndim), np.eye(ndim)]
        ... )
        >>> start = priors.rvs(size=(nwalkers,))
        >>> state = sampler.run_mcmc(start, 100, progress=False)
        >>> sampler.get_chain()["model_0"].shape
        (100, 1, 16, 1, 3)
    """

    def __init__(
        self,
        nwalkers,
        ndims,  # assumes ndim_max
        log_like_fn,
        priors,
        provide_groups=False,
        provide_supplemental=False,
        tempering_kwargs={},
        branch_names=None,
        nbranches=1,
        nleaves_max=1,
        nleaves_min=0,
        pool=None,
        moves=None,
        rj_moves=None,
        dr_moves=None,
        dr_max_iter=5,
        args=None,
        kwargs=None,
        backend=None,
        vectorize=False,
        blobs_dtype=None,  # TODO check this
        plot_iterations=-1,  # TODO: do plot stuff?
        plot_generator=None,
        plot_folder=None,
        periodic=None,
        update_fn=None,
        update_iterations=-1,
        stopping_fn=None,
        stopping_iterations=-1,
        fill_zero_leaves_val=-1e300,
        num_repeats_in_model=1,
        num_repeats_rj=1,
        track_moves=True,
        nsamplers=1,
        prior_transform_fn=None,
        info={},
    ):
        # store priors
        self.priors = priors

        # number of independent samplers run simultaneously; the sampler axis
        # is FOLDED into the temperature axis everywhere internally
        # (leading axis size nsamplers * ntemps), so proposals treat the
        # independent samplers as extra temperatures
        self.nsamplers = int(nsamplers)
        if self.nsamplers < 1:
            raise ValueError(f"nsamplers must be >= 1; got {nsamplers}.")

        # store some kwargs
        self.provide_groups = provide_groups
        self.provide_supplemental = provide_supplemental
        self.fill_zero_leaves_val = fill_zero_leaves_val
        self.num_repeats_in_model = num_repeats_in_model
        self.num_repeats_rj = num_repeats_rj
        self.track_moves = track_moves

        # setup emcee-like basics
        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

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

        # setup temperaing information
        # default is no temperatures
        if tempering_kwargs == {}:
            self.ntemps = 1
            self.temperature_control = None
        else:
            # get effective total dimension
            total_ndim = 0
            for key in self.branch_names:
                total_ndim += self.nleaves_max[key] * self.ndims[key]
            self.temperature_control = TemperatureControl(
                total_ndim, nwalkers, nsamplers=self.nsamplers, **tempering_kwargs
            )
            self.ntemps = self.temperature_control.ntemps

        # folded leading-axis size: independent samplers x temperatures
        self.ntemps_eff = self.nsamplers * self.ntemps

        # set basic variables for sampling settings
        self.nwalkers = nwalkers
        self.nbranches = nbranches

        # eryn wraps periodic parameters
        if periodic is not None:
            if not isinstance(periodic, PeriodicContainer) and not isinstance(
                periodic, dict
            ):
                raise ValueError(
                    "periodic must be PeriodicContainer or dict if not None."
                )
            elif isinstance(periodic, dict):
                periodic = PeriodicContainer(periodic)

        # Parse the move schedule
        if moves is None:
            if rj_moves is not None:
                raise ValueError(
                    "If providing rj_moves, must provide moves kwarg as well."
                )

            # defaults to stretch move
            self.moves = [
                StretchMove(
                    temperature_control=self.temperature_control,
                    periodic=periodic,
                    a=2.0,
                )
            ]
            self.weights = [1.0]

        elif isinstance(moves, Iterable):
            try:
                self.moves, self.weights = [list(tmp) for tmp in zip(*moves)]

            except TypeError:
                self.moves = moves
                self.weights = np.ones(len(moves))
        else:
            self.moves = [moves]
            self.weights = [1.0]

        self.weights = np.atleast_1d(self.weights).astype(float)
        self.weights /= np.sum(self.weights)

        # parse the reversible jump move schedule
        if rj_moves is None:
            self.has_reversible_jump = False
        elif (isinstance(rj_moves, bool) and rj_moves) or isinstance(rj_moves, str):
            self.has_reversible_jump = True

            if self.has_reversible_jump:
                if nleaves_min is None:
                    nleaves_min = {bn: 0 for bn in branch_names}
                elif isinstance(nleaves_min, int):
                    assert len(branch_names) == 1
                    nleaves_min = {branch_names[0]: nleaves_min}

                elif isinstance(nleaves_min, list) or isinstance(
                    nleaves_min, np.ndarray
                ):
                    assert len(branch_names) == len(nleaves_min)
                    nleaves_min = {bn: nl for bn, nl in zip(branch_names, nleaves_min)}

                elif isinstance(nleaves_min, dict):
                    assert len(list(nleaves_min.keys())) == len(branch_names)
                    for key in nleaves_min:
                        if key not in branch_names:
                            raise ValueError(
                                f"{key} is in nleaves_min but does not appear in branch_names: {branch_names}."
                            )
                else:
                    raise ValueError(
                        "If providing nleaves_min, nleaves_min is to be a scalar int, list, or dict."
                    )

                self.nleaves_min = nleaves_min

                if (isinstance(rj_moves, bool) and rj_moves) or (
                    isinstance(rj_moves, str) and rj_moves == "together"
                ):
                    # default to DistributionGenerateRJ

                    # gibbs sampling setup here means run all of them together
                    gibbs_sampling_setup = None

                    rj_move = DistributionGenerateRJ(
                        self.priors,
                        nleaves_max=self.nleaves_max,
                        nleaves_min=self.nleaves_min,
                        dr=dr_moves,
                        dr_max_iter=dr_max_iter,
                        tune=False,
                        temperature_control=self.temperature_control,
                        gibbs_sampling_setup=gibbs_sampling_setup,
                    )
                    self.rj_moves = [rj_move]
                    self.rj_weights = [1.0]

                elif isinstance(rj_moves, str) and rj_moves == "iterate_branches":
                    # will iterate through all branches within one RJ proposal
                    gibbs_sampling_setup = deepcopy(branch_names)

                    # default to DistributionGenerateRJ
                    rj_move = DistributionGenerateRJ(
                        self.priors,
                        nleaves_max=self.nleaves_max,
                        nleaves_min=self.nleaves_min,
                        dr=dr_moves,
                        dr_max_iter=dr_max_iter,
                        tune=False,
                        temperature_control=self.temperature_control,
                        gibbs_sampling_setup=gibbs_sampling_setup,
                    )
                    self.rj_moves = [rj_move]
                    self.rj_weights = [1.0]

                elif isinstance(rj_moves, str) and rj_moves == "separate_branches":
                    # will iterate through all branches within one RJ proposal
                    rj_moves = []
                    rj_weights = []
                    for branch_name in branch_names:

                        # only do one branch per move
                        gibbs_sampling_setup = [branch_name]

                        # default to DistributionGenerateRJ
                        rj_move_tmp = DistributionGenerateRJ(
                            self.priors,
                            nleaves_max=self.nleaves_max,
                            nleaves_min=self.nleaves_min,
                            dr=dr_moves,
                            dr_max_iter=dr_max_iter,
                            tune=False,
                            temperature_control=self.temperature_control,
                            gibbs_sampling_setup=gibbs_sampling_setup,
                        )
                        rj_moves.append(rj_move_tmp)
                        # will renormalize after
                        rj_weights.append(1.0)
                    self.rj_moves = rj_moves
                    self.rj_weights = rj_weights

                elif isinstance(rj_moves, str):
                    raise ValueError(
                        f"When providing a str for rj_moves, must be 'together', 'iterate_branches', or 'separate_branches'. Input is {rj_moves}"
                    )

        # same as above for moves
        elif isinstance(rj_moves, Iterable):
            self.has_reversible_jump = True

            try:
                self.rj_moves, self.rj_weights = zip(*rj_moves)
            except TypeError:
                self.rj_moves = rj_moves
                self.rj_weights = np.ones(len(rj_moves))

        elif isinstance(rj_moves, bool) and not rj_moves:
            self.has_reversible_jump = False
            self.rj_moves = None
            self.rj_weights = None

        elif not isinstance(rj_moves, bool):
            self.has_reversible_jump = True

            self.rj_moves = [rj_moves]
            self.rj_weights = [1.0]

        # adjust rj weights properly
        if self.has_reversible_jump:
            self.rj_weights = np.atleast_1d(self.rj_weights).astype(float)
            self.rj_weights /= np.sum(self.rj_weights)

            # warn if base stretch is used
            for move in self.moves:
                if type(move) == StretchMove:
                    warnings.warn(
                        "If using revisible jump, using the Stretch Move for in-model proposals is not advised. It will run and work, but it will not be using the correct complientary group of parameters meaning it will most likely be very inefficient."
                    )

        # make sure moves have temperature module
        if self.temperature_control is not None:
            for move in self.moves:
                if move.temperature_control is None:
                    move.temperature_control = self.temperature_control

            if self.has_reversible_jump:
                for move in self.rj_moves:
                    if move.temperature_control is None:
                        move.temperature_control = self.temperature_control

        # make sure moves have temperature module
        if periodic is not None:
            for move in self.moves:
                if move.periodic is None:
                    move.periodic = periodic

            if self.has_reversible_jump:
                for move in self.rj_moves:
                    if move.periodic is None:
                        move.periodic = periodic

        # store
        self.periodic = periodic

        # give every move the independent-samplers structure (also set by the
        # temperature_control setter, but needed for untempered runs and
        # user-constructed moves); guard against mismatched user-provided
        # temperature controls
        all_moves_for_setup = list(self.moves) + (
            list(self.rj_moves) if self.has_reversible_jump else []
        )
        for move in all_moves_for_setup:
            if (
                move.temperature_control is not None
                and getattr(move.temperature_control, "nsamplers", 1) != self.nsamplers
            ):
                raise ValueError(
                    f"Move {move.__class__.__name__} carries a TemperatureControl with "
                    f"nsamplers={getattr(move.temperature_control, 'nsamplers', 1)}, but the "
                    f"sampler was built with nsamplers={self.nsamplers}."
                )
            move.nsamplers = self.nsamplers
            move.ntemps_per_sampler = self.ntemps
            move.sampler_id_rows = np.repeat(np.arange(self.nsamplers), self.ntemps)

        # prepare the per proposal accepted values that are held as attributes in the specific classes
        for move in self.moves:
            move.accepted = np.zeros((self.ntemps_eff, self.nwalkers))

        if self.has_reversible_jump:
            for move in self.rj_moves:
                move.accepted = np.zeros((self.ntemps_eff, self.nwalkers))

        # setup backend if not provided or initialized
        if backend is None:
            self.backend = Backend()
        elif isinstance(backend, str):
            self.backend = HDFBackend(backend)
        else:
            self.backend = backend

        self.info = info

        all_moves_tmp = list(
            tuple(self.moves)
            if not self.has_reversible_jump
            else tuple(self.moves) + tuple(self.rj_moves)
        )

        self.all_moves = {}
        if self.track_moves:
            current_indices_move_keys = {}
            for move in all_moves_tmp:
                # get out of tuple if weights are given
                if isinstance(move, tuple):
                    move = move[0]

                # get the name of the class instance as a string
                move_name = move.__class__.__name__

                # need to keep track how many times each type of move class has been used
                if move_name not in current_indices_move_keys:
                    current_indices_move_keys[move_name] = 0

                else:
                    current_indices_move_keys[move_name] += 1

                # get the full name including the index
                full_move_name = move_name + f"_{current_indices_move_keys[move_name]}"
                self.all_moves[full_move_name] = move

            # get move keys out
            move_keys = list(self.all_moves.keys())

        else:
            move_keys = None

        self.move_keys = move_keys

        # Deal with re-used backends
        if not self.backend.initialized:
            self._previous_state = None
            self.reset(
                branch_names=branch_names,
                ntemps=self.ntemps,
                nleaves_max=nleaves_max,
                rj=self.has_reversible_jump,
                moves=move_keys,
                key_order=self.key_order,
                nsamplers=self.nsamplers,
                **info,
            )
            state = np.random.get_state()
        else:
            if self.track_moves:
                moves_okay = True
                if len(self.move_keys) != len(self.backend.move_keys):
                    moves_okay = False

                for key in self.move_keys:
                    if key not in self.backend.move_keys:
                        moves_okay = False

                if not moves_okay:
                    raise ValueError(
                        "Configuration of moves has changed. Cannot use the same backend. Declare a new backend and start from the previous state. If you would prefer not to track move acceptance fraction, set track_moves to False in the EnsembleSampler."
                    )

            if self.key_order != self.backend.key_order:
                raise ValueError("Input key order from priors does not match backend.")

            # legacy files / backends have no sampler axis and only support nsamplers == 1
            if getattr(self.backend, "nsamplers", 1) != self.nsamplers:
                raise ValueError(
                    f"Backend was initialized with nsamplers={getattr(self.backend, 'nsamplers', 1)}, "
                    f"but the sampler was built with nsamplers={self.nsamplers}. "
                    "Restarting with a different nsamplers (including from a legacy file "
                    "without the sampler axis) is not supported."
                )

            # Check the backend shape
            for i, (name, shape) in enumerate(self.backend.shape.items()):
                test_shape = ((self.nsamplers,) if self.nsamplers > 1 else ()) + (
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[name],
                    self.ndims[name],
                )
                if shape != test_shape:
                    raise ValueError(
                        (
                            f"the shape of the backend ({shape}) is incompatible with the "
                            f"shape of the sampler ({test_shape} for model {name})"
                        )
                    )

            # Get the last random state
            state = self.backend.random_state
            if state is None:
                state = np.random.get_state()

            # Grab the last step so that we can restart
            it = self.backend.iteration
            if it > 0:
                self._previous_state = self.get_last_sample()

        # This is a random number generator that we can easily set the state
        # of without affecting the numpy-wide generator
        self._random = np.random.mtrand.RandomState()
        self._random.set_state(state)

        # Do a little bit of _magic_ to make the likelihood call with
        # ``args`` and ``kwargs`` pickleable.
        self.log_like_fn = _FunctionWrapper(log_like_fn, args, kwargs)

        self.all_walkers = self.nwalkers * self.ntemps_eff

        # prepare plotting
        # TODO: adjust plotting maybe?
        self.plot_iterations = plot_iterations

        if self.nsamplers > 1 and plot_iterations > 0:
            raise ValueError(
                "Runtime plotting (plot_iterations > 0) is not supported for nsamplers > 1."
            )

        # per-sampler prior-basis transform (para parity):
        # dict of branch_name -> object with in-place
        # transform_to_prior_basis(coords, samplers_running_idx) and
        # adjust_logp(logp, samplers_running_idx) methods
        if prior_transform_fn is not None:
            if not isinstance(prior_transform_fn, dict):
                if len(self.branch_names) != 1:
                    raise ValueError(
                        "With multiple branches, prior_transform_fn must be a dict keyed by branch name."
                    )
                prior_transform_fn = {self.branch_names[0]: prior_transform_fn}

            if self.has_reversible_jump:
                raise ValueError(
                    "prior_transform_fn is not compatible with reversible jump: RJ moves "
                    "evaluate the priors directly on sampling-basis coordinates."
                )
            if self.provide_groups:
                raise ValueError(
                    "prior_transform_fn is not compatible with provide_groups=True."
                )
            if "all_models_together" in self.priors:
                raise ValueError(
                    "prior_transform_fn is not compatible with 'all_models_together' priors."
                )
            for name in prior_transform_fn:
                if name not in self.branch_names:
                    raise ValueError(
                        f"prior_transform_fn key {name} is not a branch name: {self.branch_names}."
                    )
                if self.nleaves_max[name] > 1:
                    raise ValueError(
                        "prior_transform_fn requires nleaves_max == 1 for the transformed branch."
                    )
            for move in all_moves_for_setup:
                if isinstance(move, MultipleTryMove):
                    raise ValueError(
                        "prior_transform_fn is not compatible with multiple-try moves: they "
                        "evaluate the priors directly on sampling-basis coordinates."
                    )
        self.prior_transform_fn = prior_transform_fn

        if plot_generator is None and self.plot_iterations > 0:
            # set to default if not provided
            if plot_folder is  None:
                plot_folder = "./runtime_plots"
            self.plot_generator = PlotContainer(
                backend=self.backend, plots=['base', 'rj'], parent_folder=plot_folder, discard=0.2
            )
        elif self.plot_iterations > 0:
            self.plot_generator = plot_generator
            self.plot_generator.backend = self.backend # make sure backend is correctly set

        # prepare stopping functions
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations

        # prepare update functions
        self.update_fn = update_fn
        self.update_iterations = update_iterations

    @property
    def random_state(self):
        """tuple: State of the sampler's own random number generator.

        The result of ``get_state()`` on the internal ``numpy.random.mtrand.RandomState``
        (separate from the global numpy generator). It is copied onto every
        ``state.random_state`` and saved by the backend. Setting it calls ``set_state``;
        a failing set is silently ignored.
        """
        return self._random.get_state()
    
    @random_state.setter  # NOQA
    def random_state(self, state):
        """Set the generator state; any exception from ``set_state`` is swallowed."""
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def priors(self):
        """dict: Branch name (or ``"all_models_together"``) to prior object.

        Setting it accepts the forms listed for the ``priors`` constructor argument: a
        :class:`~eryn.priors.ProbDistContainer` is stored as ``{"model_0": container}``,
        dict values that are dicts of distributions are wrapped in a
        :class:`~eryn.priors.ProbDistContainer`, containers and other objects with a
        ``logpdf`` method are stored as given. The setter also fills :attr:`key_order` from
        each prior's ``key_order`` attribute.

        Raises:
            ValueError: A distribution without ``logpdf``, a dict value of an unsupported
                type, or an input that is neither a dict nor a ``ProbDistContainer``.
        """
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Validate and store the priors (see the getter)."""
        if isinstance(priors, dict):
            self._priors = {}

            for key in priors.keys():
                test = priors[key]
                if isinstance(test, dict):
                    # check all dists
                    for ind, dist in test.items():
                        if not hasattr(dist, "logpdf"):
                            raise ValueError(
                                f"Distribution for model {key} and index {ind} does not have logpdf method."
                            )

                    self._priors[key] = ProbDistContainer(test)

                elif isinstance(test, ProbDistContainer):
                    self._priors[key] = test

                elif hasattr(test, "logpdf"):
                    self._priors[key] = test

                else:
                    raise ValueError(
                        "priors dictionary items must be dictionaries with prior information or instances of the ProbDistContainer class."
                    )

        elif isinstance(priors, ProbDistContainer):
            self._priors = {"model_0": priors}

        else:
            raise ValueError("Priors must be a dictionary.")

        self.key_order = {key: value.key_order for key, value in self._priors.items()}
        return

    @property
    def iteration(self):
        """int: Number of steps stored in the backend (``backend.iteration``)."""
        return self.backend.iteration

    def reset(self, **kwargs):
        """Clear the backend and re-initialize its storage for this sampler.

        Calls ``backend.reset(nwalkers, ndims, **kwargs)``, with ``nsamplers`` defaulting to
        the sampler's. Called by the constructor when the backend is not yet initialized.

        Args:
            **kwargs: Forwarded to :meth:`eryn.backends.Backend.reset` (for example
                ``branch_names``, ``ntemps``, ``nleaves_max``, ``rj``, ``moves``,
                ``key_order``, and extra ``info`` key/value pairs).
        """
        kwargs.setdefault("nsamplers", self.nsamplers)
        self.backend.reset(self.nwalkers, self.ndims, **kwargs)

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["pool"] = None
        return d

    def get_model(self):
        """Build the :class:`~eryn.model.Model` handed to every move's ``propose``.

        Called once per :meth:`sample` call. It can also be used to run a move by hand
        outside the sampler. Subclasses may override it to pass a different model object.

        Returns:
            Model: ``Model(log_like_fn, compute_log_like_fn, compute_log_prior_fn,
            temperature_control, map_fn, random)`` with the wrapped likelihood,
            :meth:`compute_log_like`, :meth:`compute_log_prior`, the sampler's
            :class:`~eryn.moves.tempering.TemperatureControl` (or ``None``),
            ``pool.map`` (or the built-in ``map``), and the sampler's ``RandomState``.
        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = Model(
            self.log_like_fn,
            self.compute_log_like,
            self.compute_log_prior,
            self.temperature_control,
            map_fn,
            self._random,
        )
        return model

    def sample(
        self,
        initial_state,
        iterations=1,
        tune=False,
        skip_initial_state_check=True,
        thin_by=1,
        store=True,
        progress=False,
    ):
        """Advance the ensemble as a generator.

        Each proposal iteration runs ``num_repeats_in_model`` in-model proposals (a move
        drawn from ``moves`` with ``weights`` each time), then ``num_repeats_rj``
        reversible-jump proposals if reversible jump is on. Every ``thin_by`` proposal
        iterations the state is saved to the backend (when ``store``) and yielded.
        ``update_fn`` runs every ``update_iterations`` proposal iterations; ``stopping_fn``
        and runtime plots are handled by :meth:`run_mcmc`, not here.

        Before the first proposal the initial state is completed: a missing ``log_prior``
        / ``log_like`` is computed with :meth:`compute_log_prior` /
        :meth:`compute_log_like`; ``betas`` stored on the state are loaded into the
        temperature control, otherwise the temperature control's ladder is copied onto the
        state.

        Args:
            initial_state (State, np.ndarray, or dict): Starting point. A :class:`State` (or
                subclass, copied with its own class), or coordinates as accepted by
                :class:`State`: one array for a single branch, or a dict keyed by branch
                name, each of shape ``(ntemps, nwalkers, nleaves_max, ndim)`` (2D/3D
                shortcuts are expanded by :class:`State`). With ``nsamplers > 1`` use 5D
                ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)`` coordinates or a State
                built with ``nsamplers``; for ``nleaves_max == 1`` a 4D
                ``(nsamplers, ntemps, nwalkers, ndim)`` array is also accepted. The input
                is copied.
            iterations (int or None, optional): Number of states to yield (and store).
                ``None`` generates an infinite stream (requires ``store=False``). Default
                ``1``.
            tune (bool, optional): If ``True``, call ``move.tune(state, accepted)`` after
                every proposal. Default ``False``.
            skip_initial_state_check (bool, optional): If ``False``, the intent is to raise
                ``ValueError`` when the initial walkers are not linearly independent
                (:func:`walkers_independent`, without reversible jump). As written the check
                reads ``state.coords``, which :class:`State` does not define, so ``False``
                raises ``AttributeError``. Default ``True``.
            thin_by (int, optional): Proposal iterations per stored / yielded state, so
                ``iterations * thin_by`` proposal iterations are made. Replaced by
                ``self.override_thin_by`` when that attribute exists. Default ``1``.
            store (bool, optional): If ``True``, grow the backend by ``iterations`` steps
                and save every yielded state (with acceptance counts and, if
                ``track_moves``, per-move acceptance fractions). Default ``True``.
            progress (bool or str, optional): ``True`` shows a ``tqdm`` progress bar; a str
                selects ``tqdm.tqdm_<str>`` (for example ``"notebook"``); ``False`` shows
                none. Default ``False``.

        Yields:
            State: The current state, every ``thin_by`` proposal iterations. The stock
            moves update this object in place on later iterations; keep a copy
            (``State(state, copy=True)``) if a snapshot is needed.

        Raises:
            ValueError: ``iterations=None`` with ``store=True``; coordinates, ``betas``,
                ``log_like`` or ``log_prior`` with the wrong shape; a state with another
                ``nsamplers``; a NaN or infinite initial ``log_like`` / ``log_prior`` on a
                running sampler; or ``thin_by <= 0``.
            AttributeError: ``skip_initial_state_check=False`` (see above), or the state
                carries ``betas`` while the sampler is untempered (``temperature_control``
                is ``None``), as for a state restored from an untempered backend.
        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")

        # Interpret the input as a walker state and check the dimensions.

        # convenience for nsamplers > 1: accept para-style 4D array input
        # (nsamplers, ntemps, nwalkers, ndim) by inserting the leaf axis; only
        # applied when the shape matches the sampler configuration exactly and
        # cannot be confused with the folded 4D layout
        if self.nsamplers > 1 and not hasattr(initial_state, "branches"):
            initial_state_in = initial_state
            if isinstance(initial_state_in, np.ndarray):
                initial_state_in = {self.branch_names[0]: initial_state_in}
            if isinstance(initial_state_in, dict):
                new_coords = {}
                for name, arr in initial_state_in.items():
                    if (
                        isinstance(arr, np.ndarray)
                        and arr.ndim == 4
                        and self.nleaves_max[name] == 1
                        and arr.shape
                        == (
                            self.nsamplers,
                            self.ntemps,
                            self.nwalkers,
                            self.ndims[name],
                        )
                        and arr.shape
                        != (
                            self.ntemps_eff,
                            self.nwalkers,
                            self.nleaves_max[name],
                            self.ndims[name],
                        )
                    ):
                        arr = arr[:, :, :, None, :]
                    new_coords[name] = arr
                initial_state = new_coords

        # initial_state.__class__ rather than State in case it is a subclass
        # of State
        if (
            hasattr(initial_state, "__class__")
            and issubclass(initial_state.__class__, State)
            and not isinstance(initial_state.__class__, State)
        ):
            state = initial_state.__class__(initial_state, copy=True)
            if getattr(state, "nsamplers", 1) != self.nsamplers:
                raise ValueError(
                    f"Input state has nsamplers={getattr(state, 'nsamplers', 1)}, but the "
                    f"sampler was built with nsamplers={self.nsamplers}. Build the State "
                    "with the nsamplers kwarg or pass 5D coordinates "
                    "(nsamplers, ntemps, nwalkers, nleaves_max, ndim)."
                )
        else:
            state = State(initial_state, copy=True, nsamplers=self.nsamplers)

        # source of truth for which independent samplers are running; the
        # array object is shared by reference through the whole iteration, so
        # in-place edits (e.g. from update_fn) propagate
        self._samplers_running = getattr(state, "samplers_running", None)

        # Check the backend shape
        for i, (name, branch) in enumerate(state.branches.items()):
            ntemps_, nwalkers_, nleaves_, ndim_ = branch.shape
            if (ntemps_, nwalkers_, nleaves_, ndim_) != (
                self.ntemps_eff,
                self.nwalkers,
                self.nleaves_max[name],
                self.ndims[name],
            ):
                raise ValueError(
                    f"incompatible input dimensions for branch {name}: folded shape "
                    f"{branch.shape} vs expected "
                    f"{(self.ntemps_eff, self.nwalkers, self.nleaves_max[name], self.ndims[name])}"
                    + (
                        " (for nsamplers > 1, provide coordinates with shape "
                        f"({self.nsamplers}, {self.ntemps}, {self.nwalkers}, nleaves_max, ndim))"
                        if self.nsamplers > 1
                        else ""
                    )
                )

        # do an initial state check if is requested and we are not using reversible jump
        if (not skip_initial_state_check) and (
            not walkers_independent(state.coords) and not self.has_reversible_jump
        ):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_prior = self.compute_log_prior(coords, inds=inds)

        if state.log_like is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_like, state.blobs = self.compute_log_like(
                coords,
                inds=inds,
                logp=state.log_prior,
                supps=state.supplemental,  # only used if self.provide_supplemental is True
                branch_supps=state.branches_supplemental,  # only used if self.provide_supplemental is True
            )

        # get betas out of state object if they are there
        if state.betas is not None:
            if state.betas.shape[0] != self.ntemps_eff:
                raise ValueError(
                    "Input state has inverse temperatures (betas), but not the correct number of temperatures according to sampler inputs."
                )

            self.temperature_control.betas = state.betas.copy()

        else:
            if hasattr(self, "temperature_control") and hasattr(
                self.temperature_control, "betas"
            ):
                state.betas = self.temperature_control.betas.copy()

        if np.shape(state.log_like) != (self.ntemps_eff, self.nwalkers):
            raise ValueError("incompatible input dimensions")
        if np.shape(state.log_prior) != (self.ntemps_eff, self.nwalkers):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``. Rows of non-running samplers hold fill values, so only
        # check the running rows.
        if self._samplers_running is None:
            check_rows = slice(None)
        else:
            check_rows = np.repeat(
                np.asarray(self._samplers_running, dtype=bool), self.ntemps
            )

        if np.any(np.isnan(state.log_like[check_rows])):
            raise ValueError("The initial log_like was NaN")

        if np.any(np.isinf(state.log_like[check_rows])):
            raise ValueError("The initial log_like was +/- infinite")

        if np.any(np.isnan(state.log_prior[check_rows])):
            raise ValueError("The initial log_prior was NaN")

        if np.any(np.isinf(state.log_prior[check_rows])):
            raise ValueError("The initial log_prior was +/- infinite")

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

        if hasattr(self, "override_thin_by"):
            thin_by = self.override_thin_by
        self.yield_step = thin_by
        self.checkpoint_step = thin_by
        if store:
            self.backend.grow(iterations, state.blobs)

        # get the model object
        model = self.get_model()

        # Inject the progress bar
        total = None if iterations is None else iterations * self.yield_step
        with get_progress_bar(progress, total) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(self.yield_step):
                    # re-sync the running mask (update_fn may replace the array
                    # on the state between iterations)
                    self._samplers_running = getattr(state, "samplers_running", None)

                    # in model moves
                    accepted = np.zeros((self.ntemps_eff, self.nwalkers))
                    for repeat in range(self.num_repeats_in_model):
                        # Choose a random move
                        move = self._random.choice(self.moves, p=self.weights)

                        # Propose (in model)
                        state, accepted_out = move.propose(model, state)
                        accepted += accepted_out
                        if self.ntemps > 1:
                            try:
                                in_model_swaps = move.temperature_control.swaps_accepted
                            except AttributeError:
                                in_model_swaps = None
                        else:
                            in_model_swaps = None

                        state.random_state = self.random_state

                        if tune:
                            move.tune(state, accepted_out)

                    if self.has_reversible_jump:
                        rj_accepted = np.zeros((self.ntemps_eff, self.nwalkers))
                        for repeat in range(self.num_repeats_rj):
                            rj_move = self._random.choice(
                                self.rj_moves, p=self.rj_weights
                            )

                            # Propose (Between models)
                            state, rj_accepted_out = rj_move.propose(model, state)
                            rj_accepted += rj_accepted_out
                            # Again commenting out this section: We do not control temperature on RJ moves
                            # if self.ntemps > 1:
                            #     rj_swaps = rj_move.temperature_control.swaps_accepted
                            # else:
                            #     rj_swaps = None
                            rj_swaps = None

                            state.random_state = self.random_state

                            if tune:
                                rj_move.tune(state, rj_accepted_out)

                    else:
                        rj_accepted = None
                        rj_swaps = None

                    # Save the new step
                    if store and (i + 1) % self.checkpoint_step == 0:
                        if self.track_moves:
                            moves_accepted_fraction = {
                                key: move_tmp.acceptance_fraction
                                for key, move_tmp in self.all_moves.items()
                            }
                        else:
                            moves_accepted_fraction = None

                        self.backend.save_step(
                            state,
                            accepted,
                            rj_accepted=rj_accepted,
                            swaps_accepted=in_model_swaps,
                            moves_accepted_fraction=moves_accepted_fraction,
                        )

                    # update after diagnostic and stopping check
                    # if updating and using burn_in, need to make sure it does not use
                    # previous chain samples since they are not stored.
                    if (
                        self.update_iterations > 0
                        and self.update_fn is not None
                        and (i + 1) % (self.update_iterations) == 0
                    ):
                        self.update_fn(i, state, self)

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state

    def run_mcmc(
        self, initial_state, nsteps, burn=None, post_burn_update=False, **kwargs
    ):
        """Run :meth:`sample` for ``nsteps`` stored steps and return the last state.

        An optional burn-in of ``burn`` proposal iterations runs first with ``store=False``
        and ``thin_by=1``. Then :meth:`sample` runs for ``nsteps`` yielded states; after
        each one, runtime plots are produced every ``plot_iterations`` states and
        ``stopping_fn`` is checked every ``stopping_iterations`` states (a truthy return
        stops the run early). The final state is kept so that ``run_mcmc(None, ...)``
        continues from it.

        Args:
            initial_state (State, np.ndarray, dict, or None): Starting point, as for
                :meth:`sample`. ``None`` continues from the state returned by the previous
                call (or, for a resumed backend, from its last stored sample).
            nsteps (int): Number of states to yield and (with ``store=True``) store. The
                number of proposal iterations is ``nsteps * thin_by``. ``0`` returns the
                (burned-in) initial state without sampling.
            burn (int, optional): Burn-in proposal iterations run before storing. Default
                ``None`` (no burn-in).
            post_burn_update (bool, optional): If ``True`` and ``update_fn`` is set, call
                ``update_fn(burn, state, sampler)`` once after the burn-in. Default
                ``False``.
            **kwargs: Forwarded to :meth:`sample` (``thin_by``, ``store``, ``progress``,
                ``tune``, ``skip_initial_state_check``).

        Returns:
            State: The last state yielded by :meth:`sample` (or the initial state when
            ``nsteps == 0``).

        Raises:
            ValueError: ``initial_state`` is ``None`` and there is no previous state, or
                any error raised by :meth:`sample`.
        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        # run burn in
        if burn is not None and burn != 0:
            # prepare kwargs that relate to burn
            burn_kwargs = deepcopy(kwargs)
            burn_kwargs["store"] = False
            burn_kwargs["thin_by"] = 1
            i = 0
            for results in self.sample(initial_state, iterations=burn, **burn_kwargs):
                i += 1

            # run post-burn update
            if post_burn_update and self.update_fn is not None:
                self.update_fn(i, results, self)

            initial_state = results

        if nsteps == 0:
            return initial_state

        results = None

        i = 0
        for results in self.sample(initial_state, iterations=nsteps, **kwargs):
            # diagnostic plots
            # TODO: adjust diagnostic plots
            if self.plot_iterations > 0 and (i + 1) % (self.plot_iterations) == 0:
                self.plot_generator.produce_plots(sampler=self)  # TODO: remove defaults

            # check for stopping before updating
            if (
                self.stopping_iterations > 0
                and self.stopping_fn is not None
                and (i + 1) % (self.stopping_iterations) == 0
            ):
                stop = self.stopping_fn(i, results, self)

                if stop:
                    break

            i += 1
            
        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results

    def compute_log_prior(self, coords, inds=None, supps=None, branch_supps=None):
        """Compute the log-prior of every walker, summed over branches and active leaves.

        Three paths: an ``"all_models_together"`` prior is called once on the full dicts;
        with ``provide_groups`` each branch prior is evaluated on the active leaves only
        and summed per walker; otherwise each branch prior is evaluated on all leaves and
        inactive leaves are zeroed before summing. A branch with a ``prior_transform_fn``
        is mapped to its prior basis first (running samplers only; the other samplers get
        ``-inf``). When ``nsamplers > 1`` and the input has the full folded leading axis,
        rows of samplers that are not running (``state.samplers_running``) are set to
        ``-inf``, which makes the likelihood skip them and every proposal there reject.

        Args:
            coords (dict): Branch name to coordinates of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``, for example
                ``state.branches_coords``. ``ntemps`` here is the leading-axis size of the
                input (folded ``nsamplers * ntemps`` for a full state).
            inds (dict, optional): Branch name to boolean active-leaf arrays of shape
                ``(ntemps, nwalkers, nleaves_max)``, for example ``state.branches_inds``.
                ``None`` treats every leaf as active. Default ``None``.
            supps (BranchSupplemental, optional): Forwarded to an ``"all_models_together"``
                prior only. Default ``None``.
            branch_supps (dict, optional): Forwarded to an ``"all_models_together"`` prior
                only. Default ``None``.

        Returns:
            np.ndarray: Log-prior, shape ``(ntemps, nwalkers)``.

        Raises:
            ValueError: The prior returns NaN.
        """

        # get number of temperature and walkers
        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        # independent-samplers row mask on the folded leading axis; only
        # applied when the input carries the full folded layout (some moves,
        # e.g. multiple-try internals, call this wrapper with other shapes)
        samplers_running = getattr(self, "_samplers_running", None)
        row_mask = None
        if samplers_running is not None and ntemps == self.ntemps_eff:
            running = np.asarray(samplers_running, dtype=bool)
            if not running.all():
                row_mask = np.repeat(running, self.ntemps)

        if inds is None:
            # default use all sources
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # take information out of dict and spread to x1..xn
        x_in = {}

        # for completely customizable priors
        if "all_models_together" in self.priors:
            prior_out = self.priors["all_models_together"].logpdf(
                coords, inds, supps=supps, branch_supps=branch_supps
            )
            assert prior_out.shape == (ntemps, nwalkers)

        elif self.provide_groups:
            # get leaf-group information from the inds dict
            # ("leaf groups" = leaf-groupings per likelihood call; unrelated to
            # the independent-samplers axis)
            leaf_groups = groups_from_inds(inds)

            # get the coordinates that are used
            for i, (name, coords_i) in enumerate(coords.items()):
                x_in[name] = coords_i[inds[name]]

            prior_out = np.zeros((ntemps * nwalkers))
            for name in x_in:
                # get prior for individual binaries
                prior_out_temp = self.priors[name].logpdf(x_in[name])

                # arrange prior values by leaf groups
                # TODO: vectorize this?
                for i in np.unique(leaf_groups[name]):
                    # which members are in the leaf group i
                    inds_temp = np.where(leaf_groups[name] == i)[0]
                    # num_in_group = len(inds_temp)

                    # add to the prior for this leaf group
                    prior_out[i] += prior_out_temp[inds_temp].sum()

            # reshape
            prior_out = prior_out.reshape(ntemps, nwalkers)

        else:
            prior_out = np.zeros((ntemps, nwalkers))
            for name, coords_i in coords.items():
                ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

                if (
                    self.prior_transform_fn is not None
                    and name in self.prior_transform_fn
                    and ntemps == self.ntemps_eff
                ):
                    # per-sampler prior-basis transform (para parity):
                    # slice to the running samplers, transform in place,
                    # evaluate, Jacobian-adjust in place, scatter back with
                    # -inf on non-running samplers (nleaves_max == 1 enforced
                    # at initialization)
                    running = (
                        np.full(self.nsamplers, True)
                        if samplers_running is None
                        else np.asarray(samplers_running, dtype=bool)
                    )
                    running_idx = np.arange(self.nsamplers)[running]

                    coords_buffer = coords_i.reshape(
                        self.nsamplers, self.ntemps, nwalkers, ndim
                    )[running_idx].copy()
                    self.prior_transform_fn[name].transform_to_prior_basis(
                        coords_buffer, running_idx
                    )

                    logp_tmp = (
                        self.priors[name]
                        .logpdf(coords_buffer.reshape(-1, ndim))
                        .reshape(len(running_idx), self.ntemps, nwalkers)
                    )
                    self.prior_transform_fn[name].adjust_logp(logp_tmp, running_idx)

                    prior_out_branch = np.full(
                        (self.nsamplers, self.ntemps, nwalkers), -np.inf
                    )
                    prior_out_branch[running_idx] = logp_tmp
                    prior_out += prior_out_branch.reshape(ntemps, nwalkers)

                else:
                    # flatten coordinate arrays
                    x_in[name] = coords_i.reshape(-1, ndim)
                    prior_out_temp = (
                        self.priors[name]
                        .logpdf(x_in[name])
                        .reshape(ntemps, nwalkers, nleaves_max)
                    )

                    # fix any infs / nans from binaries that are not being used (inds == False)
                    prior_out_temp[~inds[name]] = 0.0

                    # vectorized because everything is rectangular (no groups to indicate model difference)
                    prior_out += prior_out_temp.sum(axis=-1)

        # rows of non-running samplers get -inf: proposals there always reject
        # and the likelihood wrapper skips them entirely
        if row_mask is not None:
            prior_out[~row_mask] = -np.inf

        if np.any(np.isnan(prior_out)):
            raise ValueError("The prior function is returning Nan.")

        return prior_out

    def compute_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        """Compute the log-likelihood of every walker by calling ``log_like_fn``.

        Marshals the branch dicts into the call convention described for ``log_like_fn``
        in the class docstring (vectorized or one call per walker, with optional groups and
        supplementals). Walkers with a ``-inf`` log-prior are removed from the call (their
        ``inds`` are switched off in a copy, and the ``"inds_keep"`` entry of a branch
        supplemental that holds one is set to ``False`` there, in place); walkers left with
        no active leaf receive ``fill_zero_leaves_val``.

        Args:
            coords (dict): Branch name to coordinates of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``, for example
                ``state.branches_coords``.
            inds (dict, optional): Branch name to boolean active-leaf arrays of shape
                ``(ntemps, nwalkers, nleaves_max)``. ``None`` treats every leaf as active.
                Default ``None``.
            logp (np.ndarray, optional): Log-prior of shape ``(ntemps, nwalkers)``. If
                ``None`` it is computed with :meth:`compute_log_prior`, so that the
                likelihood is never evaluated outside the prior. Default ``None``.
            supps (BranchSupplemental, optional): Walker-level supplemental information
                (``state.supplemental``, base shape ``(ntemps, nwalkers)``), passed to
                ``log_like_fn`` as ``supps`` when ``provide_supplemental`` is set. Default
                ``None``.
            branch_supps (dict, optional): Branch name to
                :class:`~eryn.state.BranchSupplemental` (or ``None``), for example
                ``state.branches_supplemental``, passed as ``branch_supps`` when
                ``provide_supplemental`` is set. Default ``None``.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: The log-likelihood, shape
            ``(ntemps, nwalkers)``, and the blobs (``None`` when ``log_like_fn`` returns no
            blobs). The blobs are reshaped to ``(ntemps, nwalkers)``, which only succeeds
            for exactly one blob column; more columns raise ``ValueError`` in the reshape.
            If every walker has a ``-inf`` log-prior, a warning is issued and
            ``(full -1e300 array, None)`` is returned.

        Raises:
            ValueError: An infinite or NaN coordinate on an active leaf; both ``supps`` and
                ``branch_supps`` are ``None`` while ``provide_supplemental`` is set; or the
                likelihood returns NaN.
            NotImplementedError: ``log_like_fn`` returns an object-dtype array.
        """

        # if inds not provided, use all
        if inds is None:
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # Check that the parameters are in physical ranges.
        for name, ptemp in coords.items():
            if np.any(np.isinf(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was infinite")
            if np.any(np.isnan(ptemp[inds[name]])):
                raise ValueError("At least one parameter value was NaN")

        # if no prior values are added, compute_prior
        # this is necessary to ensure Likelihood is not evaluated outside of the prior
        if logp is None:
            logp = self.compute_log_prior(
                coords, inds=inds, supps=supps, branch_supps=branch_supps
            )

        # if all points are outside the prior
        if np.all(np.isinf(logp)):
            warnings.warn(
                "All points input for the Likelihood have a log prior of -inf."
            )
            return np.full_like(logp, -1e300), None

        # do not run log likelihood where logp = -inf
        inds_copy = deepcopy(inds)
        inds_bad = np.where(np.isinf(logp))
        for key in inds_copy:
            inds_copy[key][inds_bad] = False

            # if inds_keep in branch supps, indicate which to not keep
            if (
                branch_supps is not None
                and key in branch_supps
                and branch_supps[key] is not None
                and "inds_keep" in branch_supps[key]
            ):
                # TODO: indicate specialty of inds_keep in branch_supp
                branch_supps[key][inds_bad] = {"inds_keep": False}

        # take information out of dict and spread to x1..xn
        x_in = {}
        if self.provide_supplemental:
            if supps is None and branch_supps is None:
                raise ValueError(
                    """supps and branch_supps are both None. If self.provide_supplemental
                       is True, must provide some supplemental information."""
                )
            if branch_supps is not None:
                branch_supps_in = {}

        # determine leaf groupings from inds
        # ("leaf groups" = which leaves feed each individual likelihood call;
        # unrelated to the independent-samplers (nsamplers) axis)
        leaf_groups = groups_from_inds(inds_copy)

        # need to map group inds properly
        # this is the unique group indexes
        unique_groups = np.unique(
            np.concatenate([groups_i for groups_i in leaf_groups.values()])
        )

        # this is the map to those indexes that are used in the likelihood
        groups_map = np.arange(len(unique_groups))

        # get the indices with groups_map for the Likelihood
        ll_leaf_groups = {}
        for key, group in leaf_groups.items():
            # get unique groups in this sub-group (or branch)
            temp_unique_groups, inverse = np.unique(group, return_inverse=True)

            # use groups_map by finding where temp_unique_groups overlaps with unique_groups
            keep_groups = groups_map[np.isin(unique_groups, temp_unique_groups)]

            # fill group information for Likelihood
            ll_leaf_groups[key] = keep_groups[inverse]

        for i, (name, coords_i) in enumerate(coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords_i.shape
            nwalkers_all = ntemps * nwalkers

            # fill x_values properly into dictionary
            x_in[name] = coords_i[inds_copy[name]]

            # prepare branch supplementals for each branch
            if self.provide_supplemental:
                if branch_supps is not None:  #  and
                    if branch_supps[name] is not None:
                        # index the branch supps
                        # it will carry in a dictionary of information
                        branch_supps_in[name] = branch_supps[name][inds_copy[name]]
                    else:
                        # fill with None if this branch does not have a supplemental
                        branch_supps_in[name] = None

        # deal with overall supplemental not specific to the branches
        if self.provide_supplemental:
            if supps is not None:
                # get the flattened supplemental
                # this will produce the shape (ntemps * nwalkers,...)
                temp = supps.flat

                # unique_groups will properly index the flattened array
                supps_in = {
                    name: values[unique_groups] for name, values in temp.items()
                }

        # prepare group information
        # this gets the group_map indexing into a list
        groups_in = list(ll_leaf_groups.values())

        # if only one branch, take the group array out of the list
        if len(groups_in) == 1:
            groups_in = groups_in[0]

        # list of paramter arrays
        params_in = list(x_in.values())

        # Likelihoods are vectorized across groups
        if self.vectorize:
            # prepare args list
            args_in = []

            # when vectorizing, if params_in has one entry, take out of list
            if len(params_in) == 1:
                params_in = params_in[0]

            # add parameters to args
            args_in.append(params_in)

            # if providing groups, add to args
            if self.provide_groups:
                args_in.append(groups_in)

            # prepare supplementals as kwargs to the Likelihood
            kwargs_in = {}
            if self.provide_supplemental:
                if supps is not None:
                    kwargs_in["supps"] = supps_in
                if branch_supps is not None:
                    # get list of branch_supps values
                    branch_supps_in_2 = list(branch_supps_in.values())

                    # if only one entry, take out of list
                    if len(branch_supps_in_2) == 1:
                        kwargs_in["branch_supps"] = branch_supps_in_2[0]

                    else:
                        kwargs_in["branch_supps"] = branch_supps_in_2

            # provide args, kwargs as a tuple
            args_and_kwargs = (args_in, kwargs_in)

            # get vectorized results
            results = self.log_like_fn(args_and_kwargs)

        # each Likelihood is computed individually
        else:
            # if groups in is an array, need to put it in a list.
            if isinstance(groups_in, np.ndarray):
                groups_in = [groups_in]

            # prepare input args for all Likelihood calls
            # to be spread out with map functions below
            args_in = []

            # each individual group in the groups_map
            for group_i in groups_map:
                # args and kwargs for the individual Likelihood
                arg_i = [None for _ in self.branch_names]
                kwarg_i = {}

                # iterate over the group information from the branches
                for branch_i, groups_in_set in enumerate(groups_in):
                    # which entries in this branch are in the overall group tested
                    # this accounts for multiple leaves (or model counts)
                    inds_keep = np.where(groups_in_set == group_i)[0]

                    branch_name_i = self.branch_names[branch_i]

                    if inds_keep.shape[0] > 0:
                        # get parameters

                        params = params_in[branch_i][inds_keep]

                        # if leaf count is constant and leaf count is 1
                        # just give 1D parameters
                        if not self.has_reversible_jump and params.shape[0] == 1:
                            params = params[0]

                        # add them to the specific args for this Likelihood
                        arg_i[branch_i] = params
                        if self.provide_supplemental:
                            if supps is not None:
                                # supps are specific to each group
                                kwarg_i["supps"] = {
                                    key: supps_in[key][group_i] for key in supps_in
                                }
                            if branch_supps is not None:
                                # make sure there is a dictionary ready in this kwarg dictionary
                                if "branch_supps" not in kwarg_i:
                                    kwarg_i["branch_supps"] = {}

                                # fill these branch supplementals for the specific group
                                if branch_supps_in[branch_name_i] is not None:
                                    # get list of branch_supps values
                                    kwarg_i["branch_supps"][branch_name_i] = (
                                        branch_supps_in[branch_name_i][inds_keep]
                                    )
                                else:
                                    kwarg_i["branch_supps"][branch_name_i] = None

                # if only one model type, will take out of groups
                add_term = arg_i[0] if len(groups_in) == 1 else arg_i

                # based on how this is dealth with in the _FunctionWrapper
                # add_term is wrapped in a list
                args_in.append([[add_term], kwarg_i])

            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map

            else:
                map_func = map

            # get results and turn into an array
            results = np.asarray(list(map_func(self.log_like_fn, args_in)))

        assert isinstance(results, np.ndarray)

        # -1e300 because -np.inf screws up state acceptance transfer in proposals
        ll = np.full(nwalkers_all, -1e300)
        inds_fix_zeros = np.delete(np.arange(nwalkers_all), unique_groups)

        # make sure second dimension is not 1
        if results.ndim == 2 and results.shape[1] == 1:
            results = np.squeeze(results)

        # parse the results if it has blobs
        if results.ndim == 2:
            # get the results and put into groups that were analyzed
            ll[unique_groups] = results[:, 0]

            # fix groups that were not analyzed
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            # deal with blobs
            _blobs_out = np.zeros((nwalkers_all, results.shape[1] - 1))
            _blobs_out[unique_groups] = results[:, 1:]
            blobs_out = _blobs_out.reshape(ntemps, nwalkers)

        elif results.dtype == "object":
            # TODO: check blobs and add this capability
            raise NotImplementedError

        else:
            # no blobs
            ll[unique_groups] = results
            ll[inds_fix_zeros] = self.fill_zero_leaves_val

            blobs_out = None

        if False:  # self.provide_supplemental:
            # TODO: need to think about how to return information, we may need to add a function to do that
            if branch_supps is not None:
                for name_i, name in enumerate(branch_supps):
                    if branch_supps[name] is not None:
                        # TODO: better way to do this? limit to
                        if "inds_keep" in branch_supps[name]:
                            inds_back = branch_supps[name][:]["inds_keep"]
                            inds_back2 = branch_supps_in[name]["inds_keep"]
                        else:
                            inds_back = inds_copy[name]
                            inds_back2 = slice(None)
                        try:
                            branch_supps[name][inds_back] = {
                                key: branch_supps_in_2[name_i][key][inds_back2]
                                for key in branch_supps_in_2[name_i]
                            }
                        except ValueError:
                            breakpoint()
                            branch_supps[name][inds_back] = {
                                key: branch_supps_in_2[name_i][key][inds_back2]
                                for key in branch_supps_in_2[name_i]
                            }

        if np.any(np.isnan(ll)):
            raise ValueError("The likelihood function is returning Nan.")
        
        # return Likelihood and blobs
        return ll.reshape(ntemps, nwalkers), blobs_out

    @property
    def acceptance_fraction(self):
        """np.ndarray: In-model acceptance fraction per walker.

        ``backend.accepted / backend.iteration``, counted over the proposal iterations that
        were stored. Shape ``(ntemps, nwalkers)``, or ``(nsamplers, ntemps, nwalkers)`` when
        ``nsamplers > 1``. The counts sum over ``num_repeats_in_model`` repeats, so divide
        by that number for a per-proposal fraction.
        """
        return self.backend.accepted / float(self.backend.iteration)

    @property
    def rj_acceptance_fraction(self):
        """np.ndarray or None: Reversible-jump acceptance fraction per walker.

        Same layout as :attr:`acceptance_fraction` (sums over ``num_repeats_rj``);
        ``None`` without reversible jump.
        """
        if self.has_reversible_jump:
            return self.backend.rj_accepted / float(self.backend.iteration)
        else:
            return None

    @property
    def swap_acceptance_fraction(self):
        """np.ndarray: Fraction of accepted temperature swaps per adjacent rung pair.

        ``backend.swaps_accepted / (backend.iteration * nwalkers)``. Shape
        ``(ntemps - 1,)``, or ``(nsamplers, ntemps - 1)`` when ``nsamplers > 1``.
        """
        return self.backend.swaps_accepted / float(
            self.backend.iteration * self.nwalkers
        )

    # The getters below forward to the backend; each takes the docstring of the
    # matching Backend method through the ``__doc__`` assignment after its def.
    # Exception: get_log_like is assigned Backend.get_log_prior.__doc__ (a code
    # slip), so its rendered help describes the log-prior.
    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_like(self, **kwargs):
        return self.backend.get_log_like(**kwargs)

    get_log_like.__doc__ = Backend.get_log_prior.__doc__

    def get_log_prior(self, **kwargs):
        return self.backend.get_log_prior(**kwargs)

    get_log_prior.__doc__ = Backend.get_log_prior.__doc__

    def get_log_posterior(self, **kwargs):
        return self.backend.get_log_posterior(**kwargs)

    get_log_posterior.__doc__ = Backend.get_log_posterior.__doc__

    def get_inds(self, **kwargs):
        return self.get_value("inds", **kwargs)

    get_inds.__doc__ = Backend.get_inds.__doc__

    def get_nleaves(self, **kwargs):
        return self.backend.get_nleaves(**kwargs)

    get_nleaves.__doc__ = Backend.get_nleaves.__doc__

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    get_last_sample.__doc__ = Backend.get_last_sample.__doc__

    def get_betas(self, **kwargs):
        return self.backend.get_betas(**kwargs)

    get_betas.__doc__ = Backend.get_betas.__doc__

    def get_value(self, name, **kwargs):
        """Return a stored quantity from the backend.

        Args:
            name (str): Stored quantity, for example ``"chain"``, ``"inds"``,
                ``"log_like"``, ``"log_prior"``, ``"betas"``, ``"blobs"``.
            **kwargs: Forwarded to ``backend.get_value`` (see
                :meth:`eryn.backends.Backend.get_value`).

        Returns:
            dict or np.ndarray: What ``backend.get_value`` returns.
        """
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        """Compute autocorrelation time through backend."""
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _FunctionWrapper(object):
    """Picklable wrapper that appends fixed ``args`` / ``kwargs`` to every likelihood call.

    Args:
        f (callable): The user's ``log_like_fn``.
        args (list or None): Extra positional arguments (``None`` means none).
        kwargs (dict or None): Extra keyword arguments (``None`` means none).
    """

    def __init__(
        self,
        f,
        args,
        kwargs,
    ):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, args_and_kwargs):
        """Call ``f(*args_in, *self.args, **kwargs_in, **self.kwargs)``.

        Args:
            args_and_kwargs (tuple[list, dict]): The sampler-built positional arguments and
                keyword arguments of one call.

        Returns:
            object: Whatever ``f`` returns.

        Raises:
            Exception: Any exception from ``f`` is re-raised after the inputs and the
                traceback are printed.
        """

        args_in_add, kwargs_in_add = args_and_kwargs

        try:
            args_in = args_in_add + type(args_in_add)(self.args)
            kwargs_in = {**kwargs_in_add, **self.kwargs}

            out = self.f(*args_in, **kwargs_in)
            return out

        except:  # pragma: no cover
            import traceback

            print("eryn: Exception while calling your likelihood function:")
            print("  args added:", args_in_add)
            print("  args:", self.args)
            print("  kwargs added:", kwargs_in_add)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def walkers_independent(coords_in):
    """Check that the walkers are linearly independent (originally from ``emcee``).

    The coordinates are flattened to ``(ntemps * nwalkers, nleaves_max * ndim)``, centred,
    and normalized per column; the walkers count as independent when every coordinate is
    finite, no column is constant, and the condition number is at most ``1e8``.

    Args:
        coords_in (np.ndarray): Coordinates of shape
            ``(ntemps, nwalkers, nleaves_max, ndim)``.

    Returns:
        bool: ``True`` if the walkers are independent.

    Raises:
        AssertionError: ``coords_in`` is not 4D.
    """
    # make sure it is 4-dimensional and reshape
    # so it groups by temperature and walker
    assert coords_in.ndim == 4
    ntemps, nwalkers, nleaves_max, ndim = coords_in.shape
    coords = coords_in.reshape(ntemps * nwalkers, nleaves_max * ndim)

    # make sure all coordinates are finite
    if not np.all(np.isfinite(coords)):
        return False

    # roughly determine covariance information
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C**2, axis=0))
    C /= C_colsum
    return np.linalg.cond(C.astype(float)) <= 1e8
