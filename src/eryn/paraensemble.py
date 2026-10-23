# -*- coding: utf-8 -*-
"""Deprecated vectorized sampler for many independent single-branch ensembles.

:class:`ParaEnsembleSampler` advances ``ngroups`` independent parallel-tempered ensembles
of shape ``(ntemps, nwalkers, ndim)`` with one batched likelihood call per proposal, on CPU
(``numpy``) or GPU (``cupy``), with a built-in stretch proposal and temperature swaps. It
is superseded by :class:`~eryn.ensemble.EnsembleSampler` with ``nsamplers=...``, and works
with the deprecated :class:`~eryn.state.ParaState` and
:class:`~eryn.backends.parabackend.ParaBackend`. :func:`shuffle_along_axis` is its
independent-permutation helper.
"""

import warnings
from copy import deepcopy
from itertools import count
from typing import Callable

import numpy as np
from gpubackendtools import get_backend, get_first_backend
from gpubackendtools.exceptions import GPUBACKENDTOOLSException

from .backends.parabackend import ParaBackend
from .ensemble import EnsembleSampler
from .moves import StretchMove, TemperatureControl
from .pbar import get_progress_bar
from .state import ParaState
from .utils import PeriodicContainer

__all__ = ["ParaEnsembleSampler"]

# Eryn ships no compiled code of its own: the compute backend (numpy / cupy)
# is sourced from the GPUBackendTools (GBT) backend registry. Preference
# order used when a CUDA device is requested via the ``gpu`` argument.
_CUDA_BACKEND_PRIORITY = ("gbt_cuda13x", "gbt_cuda12x", "gbt_cuda11x")


def shuffle_along_axis(a, axis, xp=np):
    """Independently shuffle array ``a`` along ``axis``.

    Each 1D slice along ``axis`` gets its own random permutation (from
    ``xp.random.rand``, so it follows the global ``numpy`` / ``cupy`` random state).

    Args:
        a (np.ndarray or cupy.ndarray): Array to shuffle.
        axis (int): Axis along which to shuffle.
        xp (module, optional): Array module (``numpy`` or ``cupy``) matching
            the type of ``a``. Default ``numpy``.

    Returns:
        np.ndarray or cupy.ndarray: Copy of ``a`` shuffled along ``axis``.

    Example:
        >>> import numpy as np
        >>> from eryn.paraensemble import shuffle_along_axis
        >>> out = shuffle_along_axis(np.tile(np.arange(4), (3, 1)), -1)
        >>> sorted(out[0].tolist())
        [0, 1, 2, 3]
    """
    idx = xp.random.rand(*a.shape).argsort(axis=axis)
    return xp.take_along_axis(a, idx, axis=axis)


class ParaEnsembleSampler(EnsembleSampler):
    """Vectorized ensemble sampler running many independent ensembles at once (deprecated).

    This sampler advances ``ngroups`` independent parallel-tempered ensembles
    simultaneously. All groups share a single (vectorized) likelihood call per
    proposal, which makes this sampler well suited to GPU likelihoods where
    batching across groups is much cheaper than looping over them. Each group
    is a self-contained ensemble with shape ``(ntemps, nwalkers, ndim)`` and
    its own temperature ladder; groups can be switched on and off between
    steps via ``ParaState.groups_running``. Each proposal iteration is one
    stretch-move pass over the two halves of the walkers (see :meth:`propose`)
    followed, when ``ntemps > 1``, by adjacent-rung swaps and ladder adaptation
    (:meth:`tempering_operations`).

    The compute backend (CPU/GPU) is fixed at instantiation through the
    ``force_backend`` / ``gpu`` arguments and is sourced from the
    GPUBackendTools (GBT) backend registry; Eryn ships no compiled code of
    its own. All internal arrays use the matching array module (``self.xp``:
    ``numpy`` for the CPU backend, ``cupy`` for CUDA backends).

    Args:
        ndim: Number of dimensions in the parameter space.
        nwalkers: Number of walkers per temperature per group.
            Must be even (the stretch proposal splits walkers in half).
        ngroups: Number of independent ensembles run simultaneously.
        log_like_fn (callable): Called as
            ``log_like_fn(x, *args, **kwargs)`` (plus ``branch_supps=...`` and
            ``supps=...`` when ``provide_supplemental``), where ``x`` is the 2D
            ``(n, ndim)`` array (``xp``) of the in-prior points of all running
            groups, temperatures and walkers. Must return a 1D array of ``n``
            log-likelihood values; NaN results are replaced by ``-1e300``.
        priors (dict): ``{name: prior}``, where ``prior`` is a
            :class:`~eryn.priors.ProbDistContainer` (or a dict of distributions, or
            any object with ``logpdf``; see :attr:`EnsembleSampler.priors
            <eryn.ensemble.EnsembleSampler.priors>`). It is evaluated on the
            ``(n, ndim)`` coordinates, after ``prior_transform_fn`` if given.
        tempering_kwargs: Keyword arguments for
            :class:`~eryn.moves.tempering.TemperatureControl`, built with
            ``(ndim, nwalkers)``; its ladder is copied to every group (``betas``
            of shape ``(ngroups, ntemps)``). If ``None``, no tempering is used
            (``ntemps = 1``). Default ``None``.
        args: Positional arguments passed to ``log_like_fn``. Default ``()``.
        kwargs: Keyword arguments passed to ``log_like_fn``. Default ``None``
            (none).
        gpu: If provided, use this CUDA device and run with
            ``cupy`` (resolved through the first available GBT CUDA backend).
            If ``None`` (and no CUDA ``force_backend`` is given), run on CPU
            with ``numpy``. Default ``None``.
        force_backend: Name of the GBT backend to use, e.g.
            ``"cpu"``, ``"cuda12x"`` (or prefixed: ``"gbt_cpu"``). Overrides
            the default CPU resolution; combine with ``gpu`` to also select
            the CUDA device index. The jax backend is not supported (this
            sampler mutates arrays in place). Default ``None``.
        periodic: Periodic-parameter information passed to the stretch
            proposal, as a dict ``{name: {index: period}}`` or a
            :class:`~eryn.utils.PeriodicContainer`. Default ``None``.
        backend: Storage backend. If ``None``, an in-memory
            :class:`~eryn.backends.parabackend.ParaBackend` is created. A given,
            initialized backend must have shape ``(ngroups, ntemps, nwalkers, ndim)``.
            Default ``None``.
        update_fn: Called as ``update_fn(i, state, sampler)``
            every ``update_iterations`` proposals. Default ``None``.
        update_iterations: Number of proposals between calls
            to ``update_fn``. ``<= 0`` disables updates. Default ``-1``.
        stopping_fn: Called as ``stopping_fn(i, state, sampler)`` by :meth:`run_mcmc`
            every ``stopping_iterations`` yielded states; a truthy return
            stops sampling. Default ``None``.
        stopping_iterations: Number of yielded states between calls
            to ``stopping_fn``. ``<= 0`` disables stopping checks. Default ``-1``.
        prior_transform_fn (object, optional): Object implementing
            ``transform_to_prior_basis(coords, groups_running)`` (in-place map
            of the ``(num_running, ntemps, nwalkers, ndim)`` coordinates to the
            basis in which ``priors`` is defined) and
            ``adjust_logp(logp, groups_running)`` (in-place Jacobian adjustment of
            the ``(num_running, ntemps, nwalkers)`` log-prior), where
            ``groups_running`` holds the integer indices of the running groups.
            Used for per-group prior bounds (e.g. per-band frequency limits). If
            ``None``, coordinates are passed to ``priors`` unchanged. Default
            ``None``.
        name: Branch name for the single branch sampled. Default ``"model_0"``.
        provide_supplemental: If ``True``, pass ``supps`` and
            ``branch_supps`` keyword arguments through to ``log_like_fn``.
            Default ``False``.
        gibbs_sampling_setup (np.ndarray, optional): Boolean mask of shape
            ``(ndim,)`` (a ``numpy`` array). If provided, only parameters where
            this mask is ``True`` are sampled; the remaining dimensions stay fixed
            at their current values. Default ``None``.

    Raises:
        ValueError: ``nwalkers`` is odd, ``priors`` is not a dict containing ``name``,
            or the requested compute backend is unavailable or incompatible.
        AssertionError: ``gibbs_sampling_setup`` is not a boolean ``numpy`` array of
            length ``ndim``, or a given initialized backend has another shape.

    Attributes:
        ndim (int): Parameters per walker.
        nwalkers (int): Walkers per temperature per group.
        ngroups (int): Number of independent groups.
        ntemps (int): Temperatures per group.
        betas (xp.ndarray): Current inverse temperatures, shape ``(ngroups, ntemps)``.
        base_temperature_control (TemperatureControl or None): Source of the ladder and
            of the adaptation settings.
        move_proposal (StretchMove): The stretch proposal.
        swaps_accepted (xp.ndarray): Swaps accepted at the last tempering step, shape
            ``(ngroups, ntemps - 1)`` (set once tempering has run).

    Warning:
        Deprecated: construction emits a ``DeprecationWarning``. Use
        :class:`~eryn.ensemble.EnsembleSampler` with ``nsamplers=...`` (plus
        ``State.samplers_running`` and ``prior_transform_fn``). This class does not
        call ``EnsembleSampler.__init__``; inherited members that need attributes set
        there (``get_model`` needs ``pool``) or :class:`~eryn.backends.Backend` features
        that :class:`~eryn.backends.parabackend.ParaBackend` lacks (``get_inds``,
        ``get_autocorr_time``) do not work.

    Example:
        .. code-block:: python

            import numpy as np
            from eryn.paraensemble import ParaEnsembleSampler
            from eryn.priors import ProbDistContainer, UniformDistribution
            from eryn.state import ParaState

            ndim, nwalkers, ngroups, ntemps = 2, 8, 4, 3
            priors = {"model_0": ProbDistContainer(
                {i: UniformDistribution(-5.0, 5.0) for i in range(ndim)})}

            def log_like_fn(x):  # x: (n, ndim)
                return -0.5 * np.sum(x**2, axis=-1)

            sampler = ParaEnsembleSampler(
                ndim, nwalkers, ngroups, log_like_fn, priors,
                tempering_kwargs=dict(ntemps=ntemps),
            )
            start = np.random.uniform(-1, 1, size=(ngroups, ntemps, nwalkers, ndim))
            state = ParaState({"model_0": start}, groups_running=np.ones(ngroups, dtype=bool))
            state = sampler.run_mcmc(state, 100)
    """

    def __init__(
        self,
        ndim: int,
        nwalkers: int,
        ngroups: int,
        log_like_fn,
        priors,
        tempering_kwargs: dict | None = None,
        args: list | tuple = (),
        kwargs: dict | None = None,
        gpu: int | None = None,
        force_backend: str | None = None,
        periodic: dict | None = None,
        backend: ParaBackend | None = None,  # add ParaHDFBackend
        update_fn: Callable | None = None,
        update_iterations: int = -1,
        stopping_fn: Callable | None = None,
        stopping_iterations: int = -1,
        prior_transform_fn=None,
        name: str = "model_0",
        provide_supplemental: bool = False,
        gibbs_sampling_setup=None,
    ):
        warnings.warn(
            "ParaEnsembleSampler is deprecated; use "
            "EnsembleSampler(nsamplers=...) instead (it batches all samplers' "
            "likelihood evaluations the same way and supports samplers_running "
            "and prior_transform_fn). ParaEnsembleSampler will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if nwalkers % 2 != 0:
            raise ValueError(
                f"nwalkers must be even for the stretch proposal split. Got {nwalkers}."
            )

        if not isinstance(priors, dict) or name not in priors:
            raise ValueError(f"priors must be a dict containing the branch name {name!r} as a key.")

        self.ndim = ndim
        self.nwalkers = nwalkers
        self.ngroups = ngroups
        self.log_like_fn = log_like_fn
        self.priors = priors
        self.logl_args = args
        self.logl_kwargs = kwargs if kwargs is not None else {}
        # resolve the GBT compute backend before anything touches self.xp
        self._compute_backend_name = self._resolve_compute_backend(force_backend, gpu)
        self.gpu = gpu
        self.periodic = periodic
        self.update_fn = update_fn
        self.update_iterations = update_iterations
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations
        self.name = name
        self.prior_transform_fn = prior_transform_fn
        self.provide_supplemental = provide_supplemental
        self.gibbs_sampling_setup = gibbs_sampling_setup

        # for run_mcmc(initial_state=None, ...) continuation
        self._previous_state = None

        if self.gibbs_sampling_setup is not None:
            assert isinstance(self.gibbs_sampling_setup, np.ndarray)
            assert len(self.gibbs_sampling_setup) == self.ndim
            assert self.gibbs_sampling_setup.dtype == bool
            self.gibbs_sampling_setup = self.xp.asarray(self.gibbs_sampling_setup)

        if tempering_kwargs is None:
            self.ntemps = 1
            self.betas = self.xp.ones((self.ngroups, self.ntemps))
            self.base_temperature_control = None

        else:
            self.base_temperature_control = TemperatureControl(ndim, nwalkers, **tempering_kwargs)
            self.ntemps = self.base_temperature_control.ntemps
            self.betas = self.xp.tile(self.base_temperature_control.betas, (self.ngroups, 1))

        # hyperbolic-decay clock for adaptive temperature adjustments
        self.time_temp = 0

        self.backend = backend

        if self.backend is not None and self.backend.initialized:
            assert self.backend.shape == (
                self.ngroups,
                self.ntemps,
                self.nwalkers,
                self.ndim,
            )

        self.move_proposal = StretchMove(
            periodic=self.periodic,
            temperature_control=self.base_temperature_control,
            return_gpu=self.use_gpu,
            use_gpu=self.use_gpu,
        )

        # index helpers mapping (group, temp, walker) positions
        self.temp_guide = (
            self.xp.repeat(
                self.xp.arange(self.ntemps)[:, None],
                self.nwalkers * self.ngroups,
                axis=-1,
            )
            .reshape(self.ntemps, self.nwalkers, self.ngroups)
            .transpose(2, 0, 1)
        )
        self.walker_guide = (
            self.xp.repeat(
                self.xp.arange(self.nwalkers)[:, None],
                self.ntemps * self.ngroups,
                axis=-1,
            )
            .reshape(self.nwalkers, self.ntemps, self.ngroups)
            .transpose(2, 1, 0)
        )
        self.group_guide = self.xp.repeat(
            self.xp.arange(self.ngroups)[:, None], self.ntemps * self.nwalkers, axis=0
        ).reshape(self.ngroups, self.ntemps, self.nwalkers)
        self.random_state = self.xp.random

    @property
    def random_state(self):
        """module: Random number generator (the ``numpy.random`` or ``cupy.random`` module).

        Unlike :attr:`EnsembleSampler.random_state
        <eryn.ensemble.EnsembleSampler.random_state>`, this is the global generator module
        itself, not a saved state.
        """
        return self._random

    @random_state.setter
    def random_state(self, random):
        """Set the generator module."""
        self._random = random

    @property
    def periodic(self):
        """PeriodicContainer or None: Periodic-parameter container.

        Setting a dict wraps it in :class:`~eryn.utils.PeriodicContainer`; any type other
        than dict, ``PeriodicContainer`` or ``None`` raises ``ValueError``.
        """
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        """Validate and store ``periodic`` (see the getter)."""
        if isinstance(periodic, dict):
            self._periodic = PeriodicContainer(periodic)
        elif isinstance(periodic, PeriodicContainer) or periodic is None:
            self._periodic = periodic
        else:
            raise ValueError(
                f"periodic must be None, dict, or PeriodicContainer. Got {type(periodic)}."
            )

    @property
    def backend(self):
        """ParaBackend: Storage object.

        Setting ``None`` creates and resets an in-memory
        :class:`~eryn.backends.parabackend.ParaBackend` for this sampler's shape.
        """
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Store ``backend``, or build a fresh :class:`ParaBackend` for ``None``."""
        if backend is None:
            self._backend = ParaBackend()
            self._backend.reset(
                self.ndim,
                self.nwalkers,
                self.ngroups,
                ntemps=self.ntemps,
                branch_name=self.name,
            )
        else:
            self._backend = backend

    @staticmethod
    def _resolve_compute_backend(force_backend, gpu):
        """Map ``(force_backend, gpu)`` inputs to a GBT backend name.

        ``force_backend`` wins when given (bare names like ``"cpu"`` /
        ``"cuda12x"`` get the ``gbt_`` prefix). A bare ``gpu`` index implies
        the first available CUDA backend (``gbt_cuda13x``, then ``gbt_cuda12x``,
        then ``gbt_cuda11x``). Default is CPU. Raises ``ValueError`` for a non-str
        or jax backend, an unavailable backend, or ``gpu`` with a non-CUDA backend.
        """
        if force_backend is not None:
            if not isinstance(force_backend, str):
                raise ValueError(
                    f"force_backend must be a backend name string. Got {type(force_backend)}."
                )
            backend_name = (
                force_backend if force_backend.startswith("gbt_") else f"gbt_{force_backend}"
            )
            if "jax" in backend_name:
                raise ValueError(
                    "ParaEnsembleSampler mutates arrays in place and does not "
                    "support the jax backend. Use 'cpu' or a 'cuda*' backend."
                )
            try:
                backend = get_backend(backend_name)
            except GPUBACKENDTOOLSException as e:
                raise ValueError(f"Requested backend {backend_name!r} is not available: {e}") from e
            if gpu is not None and not backend.uses_cupy:
                raise ValueError(
                    f"gpu={gpu} conflicts with non-CUDA force_backend={force_backend!r}."
                )
            return backend.name

        if gpu is not None:
            try:
                return get_first_backend(_CUDA_BACKEND_PRIORITY).name
            except GPUBACKENDTOOLSException as e:
                raise ValueError(
                    f"gpu={gpu} requested, but no CUDA backend is available: {e}. "
                    "Use gpu=None to run on CPU."
                ) from e

        return get_backend("gbt_cpu").name

    @property
    def compute_backend(self):
        """GBT :class:`gpubackendtools.gpubackendtools.Backend` providing ``xp``."""
        return get_backend(self._compute_backend_name)

    @property
    def xp(self):
        """module: Array module from the GBT backend: ``cupy`` on GPU, else ``numpy``."""
        return self.compute_backend.xp

    @property
    def use_gpu(self):
        """bool: Whether this sampler instance runs on GPU (the backend uses ``cupy``)."""
        return self.compute_backend.uses_cupy

    @property
    def gpu(self):
        """int or None: CUDA device index, or ``None`` for CPU.

        Setting an index on a CPU-resolved sampler switches it to the first available
        CUDA backend (``ValueError`` if there is none) and calls
        ``cupy.cuda.runtime.setDevice``.
        """
        return self._gpu

    @gpu.setter
    def gpu(self, gpu):
        """Select the CUDA device (see the getter)."""
        if gpu is not None:
            if not self.compute_backend.uses_cupy:
                # explicit device request on a CPU-resolved sampler:
                # switch to the first available CUDA backend
                try:
                    self._compute_backend_name = get_first_backend(_CUDA_BACKEND_PRIORITY).name
                except GPUBACKENDTOOLSException as e:
                    raise ValueError(
                        f"gpu={gpu} requested, but no CUDA backend is available: {e}. "
                        "Use gpu=None to run on CPU."
                    ) from e
            self.xp.cuda.runtime.setDevice(gpu)
        self._gpu = gpu

    def add_gpu_index(self, gpu):
        """Set the CUDA device index (switches the sampler to GPU).

        Args:
            gpu (int): CUDA device index (see :attr:`gpu`).
        """
        self.gpu = gpu

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
        """Advance all groups as a generator.

        Completes the initial state first: a missing ``groups_running`` becomes all
        ``True``; a missing ``log_prior`` / ``log_like`` is computed for the running groups
        (``-inf`` / ``-1e300`` elsewhere); ``betas`` on the state replace the sampler's,
        otherwise the sampler's are copied onto the state. Each proposal iteration calls
        :meth:`propose`; every ``thin_by`` iterations the state is saved (when ``store``)
        and yielded. ``update_fn`` runs every ``update_iterations`` proposal iterations.

        Args:
            initial_state (ParaState, np.ndarray, cupy.ndarray, or dict): The initial
                :class:`~eryn.state.ParaState`, or positions of shape
                ``(ngroups, ntemps, nwalkers, ndim)`` (a dict is keyed by the sampler's
                ``name``). The input is copied.
            iterations (int or None, optional): The number of states to yield.
                ``None`` generates an infinite stream (requires ``store=False``).
                Default ``1``.
            tune (bool, optional): Included for signature compatibility with
                :class:`eryn.ensemble.EnsembleSampler`; not used here.
                Default ``False``.
            skip_initial_state_check (bool, optional): Included for signature
                compatibility with :class:`eryn.ensemble.EnsembleSampler`;
                not used here. Default ``True``.
            thin_by (int, optional): Proposal iterations per stored / yielded state, so
                ``iterations * thin_by`` proposal iterations are made. Default ``1``.
            store (bool, optional): If ``True``, grow the backend and save every yielded
                state. Default ``True``.
            progress (bool or str, optional): ``True`` shows a ``tqdm`` progress bar; a str
                selects ``tqdm.tqdm_<str>`` (for example ``"notebook"``); ``False`` shows
                none. Default ``False``.

        Yields:
            ParaState: The current state, every ``thin_by`` proposal iterations.

        Raises:
            ValueError: ``iterations=None`` with ``store=True``; coordinates, ``betas``,
                ``log_like`` or ``log_prior`` with the wrong shape; a NaN or infinite
                initial ``log_like`` / ``log_prior`` on a running group; or
                ``thin_by <= 0``.
        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")

        # Interpret the input as a walker state and check the dimensions.
        # type(initial_state) rather than ParaState in case it is a subclass.
        if isinstance(initial_state, ParaState):
            state = type(initial_state)(initial_state, copy=True)
        else:
            state = ParaState(initial_state, copy=True)

        if state.groups_running is None:
            state.groups_running = self.xp.ones(self.ngroups, dtype=bool)

        # Check the backend shape
        for name, branch in state.branches.items():
            ngroups_, ntemps_, nwalkers_, ndim_ = branch.shape
            if (ngroups_, ntemps_, nwalkers_, ndim_) != (
                self.ngroups,
                self.ntemps,
                self.nwalkers,
                self.ndim,
            ):
                raise ValueError(
                    f"incompatible input dimensions for branch {name}: "
                    f"{branch.shape} vs expected "
                    f"{(self.ngroups, self.ntemps, self.nwalkers, self.ndim)}"
                )

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = {
                name: value[state.groups_running] for name, value in state.branches_coords.items()
            }
            state.log_prior = self.xp.full((self.ngroups, self.ntemps, self.nwalkers), -np.inf)
            state.log_prior[state.groups_running] = self.compute_log_prior(
                coords,
                groups_running=self.xp.arange(self.ngroups)[state.groups_running],
            )

        if state.log_like is None:
            state.log_like = self.xp.full((self.ngroups, self.ntemps, self.nwalkers), -1e300)
            coords = {
                name: value[state.groups_running] for name, value in state.branches_coords.items()
            }
            supps_in = (
                None if state.supplemental is None else state.supplemental[state.groups_running]
            )
            branch_supps_in = {
                name: None if tmp is None else tmp[state.groups_running]
                for name, tmp in state.branches_supplemental.items()
            }

            state.log_like[state.groups_running] = self.compute_log_like(
                coords,
                logp=state.log_prior[state.groups_running],
                supps=supps_in,  # only used if self.provide_supplemental is True
                branch_supps=branch_supps_in,  # only used if self.provide_supplemental is True
            )

        # get betas out of state object if they are there
        if state.betas is not None:
            if state.betas.shape != (self.ngroups, self.ntemps):
                raise ValueError(
                    "Input state has inverse temperatures (betas) with shape "
                    f"{state.betas.shape}; expected {(self.ngroups, self.ntemps)}."
                )

            self.betas = state.betas.copy()

        else:
            if self.betas is not None:
                state.betas = self.betas.copy()

        if self.xp.shape(state.log_like) != (self.ngroups, self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")
        if self.xp.shape(state.log_prior) != (self.ngroups, self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``self.xp.nan``.
        if self.xp.any(self.xp.isnan(state.log_like[state.groups_running])):
            raise ValueError("The initial log_like was NaN")

        if self.xp.any(self.xp.isinf(state.log_like[state.groups_running])):
            raise ValueError("The initial log_like was +/- infinite")

        if self.xp.any(self.xp.isnan(state.log_prior[state.groups_running])):
            raise ValueError("The initial log_prior was NaN")

        if self.xp.any(self.xp.isinf(state.log_prior[state.groups_running])):
            raise ValueError("The initial log_prior was +/- infinite")

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

        yield_step = thin_by
        checkpoint_step = thin_by
        if store:
            self.backend.grow(iterations, state.blobs)

        # Inject the progress bar
        total = None if iterations is None else iterations * yield_step
        with get_progress_bar(progress, total) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(yield_step):
                    # in model moves
                    accepted = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
                    # Propose (in model)
                    state, accepted_out = self.propose(state)

                    accepted += accepted_out

                    if self.ntemps > 1:
                        in_model_swaps = self.swaps_accepted
                    else:
                        in_model_swaps = None

                    state.random_state = self.random_state

                    # Save the new step
                    if store and (i + 1) % checkpoint_step == 0:
                        self.backend.save_step(
                            state,
                            accepted,
                            swaps_accepted=in_model_swaps,
                        )

                    # update after diagnostic and stopping check
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

    def compute_log_like(
        self,
        coords,
        groups_running=None,
        logp=None,
        supps=None,  # only used if self.provide_supplemental is True
        branch_supps=None,
    ):
        """Compute the log-likelihood for in-prior points.

        Calls ``log_like_fn`` once on the ``(n, ndim)`` array of points with a finite
        log-prior. NaN results become ``-1e300``. On GPU the device is synchronized
        before returning.

        Args:
            coords (dict): Coordinates keyed by branch ``name`` with values of
                shape ``(num_groups_running, ntemps, nwalkers, ndim)``.
            groups_running (xp.ndarray, optional): Integer indices of the running
                groups along the leading axis of ``coords`` (used for
                ``prior_transform_fn`` when ``logp`` is computed here). Default ``None``.
            logp (xp.ndarray, optional): Pre-computed log-prior, shape
                ``coords[name].shape[:-1]``. If ``None``, it is computed here. Points with
                ``-inf`` log-prior are skipped and filled with ``-1e300``. Default
                ``None``.
            supps (object, optional): Supplemental information passed through to the
                likelihood when ``provide_supplemental`` is set. Default ``None``.
            branch_supps (dict, optional): ``{name: BranchSupplemental}``; restricted to the
                in-prior points and passed through to the likelihood when
                ``provide_supplemental`` is set. Default ``None``.

        Returns:
            xp.ndarray: Log-likelihood with the same shape as ``logp``.
        """
        if groups_running is not None:
            assert coords[self.name].shape[0] == len(groups_running)

        if logp is None:
            logp = self.compute_log_prior(coords, groups_running=groups_running)

        keep_logp = ~self.xp.isinf(logp)

        coords_arr = coords[self.name][keep_logp]

        logl = self.xp.full_like(logp, -1e300)

        if branch_supps is not None and branch_supps != {} and branch_supps[self.name] is not None:
            branch_supps = {self.name: branch_supps[self.name][keep_logp]}

        if self.provide_supplemental:
            kwargs = {**self.logl_kwargs, "branch_supps": branch_supps, "supps": supps}

        else:
            kwargs = self.logl_kwargs

        logl[keep_logp] = self.log_like_fn(coords_arr, *self.logl_args, **kwargs)

        # fix any nans that may come up
        logl[self.xp.isnan(logl)] = -1e300

        if self.use_gpu:
            self.xp.cuda.runtime.deviceSynchronize()

        return logl

    def compute_log_prior(self, coords, groups_running=None):
        """Compute the log-prior.

        If ``prior_transform_fn`` is set, coordinates are first mapped to the
        prior basis and the resulting log-prior is Jacobian-adjusted.

        Args:
            coords (dict): Coordinates keyed by branch ``name`` with values of
                shape ``(num_groups_running, ntemps, nwalkers, ndim)``.
            groups_running (xp.ndarray, optional): Integer indices of the running groups
                along the leading axis of ``coords``, passed to ``prior_transform_fn``.
                Default ``None``.

        Returns:
            xp.ndarray: Log-prior of shape ``coords[name].shape[:-1]``.
        """
        if groups_running is not None:
            assert coords[self.name].shape[0] == len(groups_running)

        shape_in = coords[self.name].shape[:-1]

        if self.prior_transform_fn is not None:
            coords_logp_buffer = coords[self.name].copy()
            self.prior_transform_fn.transform_to_prior_basis(coords_logp_buffer, groups_running)
        else:
            coords_logp_buffer = coords[self.name]

        coords_logp_in = coords_logp_buffer.reshape(-1, self.ndim)

        logp = self.priors[self.name].logpdf(coords_logp_in).reshape(shape_in)

        if self.prior_transform_fn is not None:
            self.prior_transform_fn.adjust_logp(logp, groups_running)

        return logp

    def run_mcmc(self, initial_state, nsteps, burn=None, post_burn_update=False, **kwargs):
        """Run :meth:`sample` for ``nsteps`` yielded states and return the last state.

        An optional burn-in of ``burn`` proposal iterations runs first with
        ``store=False`` and ``thin_by=1``. During the burn-in ``update_fn`` is called by
        :meth:`sample` and, in addition, by this loop every
        ``update_iterations * thin_by`` burn-in states. ``stopping_fn`` is checked every
        ``stopping_iterations`` yielded states after the burn-in.

        Args:
            initial_state (ParaState, np.ndarray, dict, or None): Starting point, as for
                :meth:`sample`; ``None`` continues from the previous call's final state.
            nsteps (int): Number of states to yield (and store). The number of proposal
                iterations is ``nsteps * thin_by``. ``0`` returns the (burned-in) initial
                state.
            burn (int, optional): Burn-in proposal iterations. Default ``None`` (none).
            post_burn_update (bool, optional): If ``True`` and ``update_fn`` is set, call
                it once after the burn-in. Default ``False``.
            **kwargs: Forwarded to :meth:`sample` (``thin_by``, ``store``, ``progress``).

        Returns:
            ParaState: The last state yielded by :meth:`sample`.

        Raises:
            ValueError: ``initial_state`` is ``None`` and ``run_mcmc`` has never been
                called, or any error raised by :meth:`sample`.
        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never been called."
                )
            initial_state = self._previous_state

        # setup thin_by info
        thin_by = 1 if "thin_by" not in kwargs else kwargs["thin_by"]

        results = None

        # run burn in
        if burn is not None and burn != 0:
            # prepare kwargs that relate to burn
            burn_kwargs = deepcopy(kwargs)
            burn_kwargs["store"] = False
            burn_kwargs["thin_by"] = 1
            i = 0
            for results in self.sample(initial_state, iterations=burn, **burn_kwargs):
                # if updating and using burn_in, need to make sure it does not use
                # previous chain samples since they are not stored.
                if (
                    self.update_iterations > 0
                    and self.update_fn is not None
                    and (i + 1) % (self.update_iterations * thin_by) == 0
                ):
                    self.update_fn(i, results, self)
                i += 1

            # run post-burn update
            if post_burn_update and self.update_fn is not None:
                self.update_fn(i, results, self)

            initial_state = results

        if nsteps == 0:
            return initial_state

        i = 0
        for results in self.sample(initial_state, iterations=nsteps, **kwargs):
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

    def propose(self, state):
        """Run one stretch proposal (plus temperature swaps) on all running groups.

        The walkers are split at random (global ``np.random``) into two halves; each half
        is moved with :class:`~eryn.moves.StretchMove` using the other half as the
        complement, then accepted when ``factors + logP_new - logP_old > log(u)``, where
        ``factors`` are the stretch move's log proposal factors, ``u`` is uniform and
        ``logP = beta * log_like + log_prior``. With ``gibbs_sampling_setup``, the
        dimensions outside the mask keep their current values. When ``ntemps > 1``,
        :meth:`tempering_operations` follows.

        Args:
            state (ParaState): Current state of all groups (not modified; a copy is
                updated and returned).

        Returns:
            tuple[ParaState, xp.ndarray]: ``(new_state, accepted)`` where ``accepted`` is
            an integer array of shape ``(ngroups, ntemps, nwalkers)`` (1 where the
            proposal was accepted).
        """
        new_state = ParaState(state, copy=True)
        groups_running = new_state.groups_running.copy()
        num_groups_running = groups_running.sum().item()

        inds_split = np.arange(self.nwalkers)

        np.random.shuffle(inds_split)

        accepted = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers), dtype=int)

        for split in range(2):
            inds_here = np.arange(self.nwalkers)[inds_split % 2 == split]
            inds_not_here = np.delete(np.arange(self.nwalkers), inds_here)

            inds_here = self.xp.asarray(inds_here)
            inds_not_here = self.xp.asarray(inds_not_here)

            s_in = (
                new_state.branches[self.name]
                .coords[:, :, inds_here][groups_running]
                .reshape(
                    (
                        self.ntemps * num_groups_running,
                        self.nwalkers // 2,
                        1,
                        self.ndim,
                    )
                )
            )
            c_in = [
                new_state.branches[self.name]
                .coords[:, :, inds_not_here][groups_running]
                .reshape((self.ntemps * num_groups_running, self.nwalkers // 2, 1, -1))
            ]

            temps_here = self.temp_guide[:, :, inds_here][groups_running]
            walkers_here = self.walker_guide[:, :, inds_here][groups_running]
            groups_here = self.group_guide[:, :, inds_here][groups_running]

            if getattr(new_state, "random_state", None) is None:
                new_state.random_state = self.random_state

            if self.gibbs_sampling_setup is not None:
                gibbs_ndim = self.gibbs_sampling_setup.sum()
            else:
                gibbs_ndim = self.ndim

            new_points_dict, factors = self.move_proposal.get_proposal(
                {self.name: s_in},
                {self.name: c_in},
                new_state.random_state,
                gibbs_ndim=gibbs_ndim,
            )
            new_points = {
                self.name: new_points_dict[self.name].reshape(
                    num_groups_running, self.ntemps, self.nwalkers // 2, -1
                )
            }

            if self.gibbs_sampling_setup is not None:
                # parameters outside the Gibbs mask stay fixed at the
                # *current* values of the walkers being updated
                new_points[self.name][:, :, :, ~self.gibbs_sampling_setup] = (
                    new_state.branches[self.name]
                    .coords[:, :, inds_here][groups_running]
                    .reshape(
                        (
                            num_groups_running,
                            self.ntemps,
                            self.nwalkers // 2,
                            self.ndim,
                        )
                    )[:, :, :, ~self.gibbs_sampling_setup]
                )

            logp = self.compute_log_prior(
                new_points,
                groups_running=self.xp.arange(self.ngroups)[groups_running],
            )
            factors = factors.reshape(logp.shape)

            supps_in = None  # new_state.supplemental[]

            branch_supps_in = {}
            if new_state.branches_supplemental[self.name] is not None:
                branch_supps_in[self.name] = {
                    key: tmp[groups_running]
                    for key, tmp in new_state.branches_supplemental[self.name][
                        :, :, inds_here
                    ].items()
                }

            logl = self.compute_log_like(
                new_points,
                groups_running=self.xp.arange(self.ngroups)[groups_running],
                logp=logp,
                supps=supps_in,
                branch_supps=branch_supps_in,
            )

            prev_logl_here = new_state.log_like[:, :, inds_here][groups_running]
            prev_logp_here = new_state.log_prior[:, :, inds_here][groups_running]

            prev_logP_here = (
                state.betas[groups_running][:, :, None] * prev_logl_here + prev_logp_here
            )

            logP = state.betas[groups_running][:, :, None] * logl + logp

            lnpdiff = factors + logP - prev_logP_here
            keep = lnpdiff > self.xp.asarray(self.xp.log(new_state.random_state.rand(*logP.shape)))

            keep_tuple = (groups_here[keep], temps_here[keep], walkers_here[keep])
            accepted[keep_tuple] = 1
            new_state.log_prior[keep_tuple] = logp[keep]
            new_state.log_like[keep_tuple] = logl[keep]
            new_state.branches[self.name].coords[keep_tuple] = new_points[self.name][keep]

        if self.ntemps > 1:
            self.tempering_operations(new_state)

        return new_state, accepted

    def tempering_operations(self, state):
        """Swap walkers between adjacent temperatures and adapt the ladder, in place.

        For every running group and each rung pair ``(i - 1, i)``, from the hottest down,
        randomly paired walkers swap coordinates, log-likelihood and log-prior with
        acceptance ``log u < (beta_{i-1} - beta_i) * (logL_i - logL_{i-1})``. Accepted
        counts go to ``self.swaps_accepted`` (shape ``(ngroups, ntemps - 1)``). If the
        base temperature control is adaptive (and ``stop_adaptation`` has not been
        reached), the interior inverse temperatures of ``state.betas`` are then adjusted
        from the per-pair acceptance ratios with a hyperbolically decaying step, the
        coldest and hottest rungs fixed (same scheme as
        :class:`~eryn.moves.tempering.TemperatureControl`).

        Args:
            state (ParaState): State to update in place (coordinates, ``log_like``,
                ``log_prior``, ``betas``).
        """

        groups_running = state.groups_running.copy()
        num_groups_running = groups_running.sum().item()

        # prepare information on how many swaps are accepted this time
        self.swaps_accepted = self.xp.zeros((self.ngroups, self.ntemps - 1), dtype=int)
        self.swaps_proposed = self.xp.full_like(self.swaps_accepted, self.nwalkers)

        swaps_accepted_tmp = self.xp.zeros((num_groups_running, self.ntemps - 1), dtype=int)
        swaps_proposed_tmp = self.xp.full_like(swaps_accepted_tmp, self.nwalkers)

        # iterate from highest to lowest temperatures
        for i in range(self.ntemps - 1, 0, -1):
            # get both temperature rungs
            bi = state.betas[groups_running, i]
            bi1 = state.betas[groups_running, i - 1]

            # difference in inverse temps
            dbeta = bi1 - bi

            # permute the indices for the walkers in each temperature to randomize swap positions
            iperm = shuffle_along_axis(
                self.xp.tile(self.xp.arange(self.nwalkers), (num_groups_running, 1)),
                -1,
                xp=self.xp,
            )
            i1perm = shuffle_along_axis(
                self.xp.tile(self.xp.arange(self.nwalkers), (num_groups_running, 1)),
                -1,
                xp=self.xp,
            )

            # random draw that produces log of the acceptance fraction
            raccept = self.xp.log(
                state.random_state.uniform(size=(num_groups_running, self.nwalkers))
            )

            # log of the detailed balance fraction
            walker_swap_i = iperm.flatten()
            walker_swap_i1 = i1perm.flatten()

            temp_swap_i = self.xp.full_like(walker_swap_i, i)
            temp_swap_i1 = self.xp.full_like(walker_swap_i1, i - 1)
            group_swap = self.xp.repeat(
                self.xp.arange(len(groups_running))[groups_running], self.nwalkers
            )

            paccept = dbeta[:, None] * (
                state.log_like[(group_swap, temp_swap_i, walker_swap_i)].reshape(
                    num_groups_running, self.nwalkers
                )
                - state.log_like[(group_swap, temp_swap_i1, walker_swap_i1)].reshape(
                    num_groups_running, self.nwalkers
                )
            )

            # How many swaps were accepted
            sel = paccept > raccept
            swaps_accepted_tmp[:, i - 1] = self.xp.sum(sel, axis=-1)

            temp_swap_i_keep = temp_swap_i[sel.flatten()]
            walker_swap_i_keep = walker_swap_i[sel.flatten()]
            group_swap_keep = group_swap[sel.flatten()]

            temp_swap_i1_keep = temp_swap_i1[sel.flatten()]
            walker_swap_i1_keep = walker_swap_i1[sel.flatten()]

            keep_i_tuple = (group_swap_keep, temp_swap_i_keep, walker_swap_i_keep)
            keep_i1_tuple = (group_swap_keep, temp_swap_i1_keep, walker_swap_i1_keep)

            coords_tmp_i = state.branches[self.name].coords[keep_i_tuple].copy()
            logl_tmp_i = state.log_like[keep_i_tuple].copy()
            logp_tmp_i = state.log_prior[keep_i_tuple].copy()

            state.branches[self.name].coords[keep_i_tuple] = state.branches[self.name].coords[
                keep_i1_tuple
            ]
            state.log_like[keep_i_tuple] = state.log_like[keep_i1_tuple]
            state.log_prior[keep_i_tuple] = state.log_prior[keep_i1_tuple]

            state.branches[self.name].coords[keep_i1_tuple] = coords_tmp_i
            state.log_like[keep_i1_tuple] = logl_tmp_i
            state.log_prior[keep_i1_tuple] = logp_tmp_i

        self.swaps_accepted[groups_running] = swaps_accepted_tmp

        # mirror TemperatureControl: respect adaptive + stop_adaptation settings
        if self.base_temperature_control.adaptive and (
            self.base_temperature_control.stop_adaptation < 0
            or self.time_temp < self.base_temperature_control.stop_adaptation
        ):
            ratios = swaps_accepted_tmp / swaps_proposed_tmp

            # adjust temps
            betas0 = state.betas[groups_running].copy()
            betas1 = state.betas[groups_running].copy()

            # Modulate temperature adjustments with a hyperbolic decay.
            decay = self.base_temperature_control.adaptation_lag / (
                self.time_temp + self.base_temperature_control.adaptation_lag
            )
            kappa = decay / self.base_temperature_control.adaptation_time

            self.time_temp += 1

            # Construct temperature adjustments.
            dSs = kappa * (ratios[:, :-1] - ratios[:, 1:])

            # Compute new ladder (hottest and coldest chains don't move).
            deltaTs = self.xp.diff(1 / betas1[:, :-1], axis=-1)
            deltaTs *= self.xp.exp(dSs)
            betas1[:, 1:-1] = 1 / (self.xp.cumsum(deltaTs, axis=-1) + 1 / betas1[:, 0][:, None])

            dbetas = betas1 - betas0
            state.betas[groups_running] += dbetas
