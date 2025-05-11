from .ensemble import EnsembleSampler
from .utils import TransformContainer, PeriodicContainer
from .moves import TemperatureControl, StretchMove
from .backends.parabackend import ParaBackend
import numpy as np
from copy import deepcopy
from .pbar import get_progress_bar
from itertools import count

from .state import ParaState

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

from typing import Union, Callable


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class ParaEnsembleSampler(EnsembleSampler):
    def __init__(
        self,
        ndim: int,
        nwalkers: int,
        ngroups: int,
        log_like_fn,
        priors,
        tempering_kwargs: Union[dict, None] = None,
        args: Union[list, tuple] = (),
        kwargs: dict = {},
        gpu: int = None,
        periodic: Union[dict, None] = None,
        backend: Union[ParaBackend] = None,  # add ParaHDFBackend
        update_fn: Callable = None,
        update_iterations=-1,
        stopping_fn: Callable = None,
        stopping_iterations: int=-1,
        prior_transform_fn=None,
        name="model_0",
        provide_supplimental=False,
    ):
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.ngroups = ngroups
        self.log_like_fn = log_like_fn
        self.priors = priors
        self.logl_args = args
        self.logl_kwargs = kwargs
        self.gpu = gpu
        self.periodic = periodic
        self.update_fn = update_fn
        self.update_iterations = update_iterations
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations
        self.name = name
        self.prior_transform_fn = prior_transform_fn
        self.provide_supplimental = provide_supplimental

        if tempering_kwargs is None:
            self.ntemps = 1
            self.betas = self.xp.ones((self.ngroups, self.ntemps))
            self.base_temperature_control = None

        else:
            self.base_temperature_control = TemperatureControl(
                ndim, nwalkers, **tempering_kwargs
            )
            self.ntemps = self.base_temperature_control.ntemps
            self.betas = self.xp.tile(
                self.base_temperature_control.betas, (self.ngroups, 1)
            )

        self.backend = backend

        if self.backend is not None and self.backend.initialized:
            assert self.backend.shape == (
                self.ngroups,
                self.ntemps,
                self.nwalkers,
                self.ndim,
            )

        self.periodic = periodic
        self.move_proposal = StretchMove(
            periodic=self.periodic,
            temperature_control=self.base_temperature_control,
            return_gpu=self.use_gpu,
            use_gpu=self.use_gpu,
        )

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
        return self._random

    @random_state.setter
    def random_state(self, random):
        self._random = random

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        if periodic is not None:
            if isinstance(periodic, dict):
                self._periodic = PeriodicContainer(periodic)
            elif isinstance(periodic, PeriodicContainer):
                self._periodic = periodic
        else:
            self._periodic = periodic

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
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

    @property
    def xp(self):
        xp = cp if self.use_gpu else np
        return xp

    @property
    def use_gpu(self):
        if self.gpu is not None:
            return True
        else:
            return False

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, gpu):
        self._gpu = gpu
        if gpu is not None:
            cp.cuda.runtime.setDevice(gpu)

    def add_gpu_index(self, gpu):
        self.gpu = gpu
        cp.cuda.runtime.setDevice(gpu)

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
        """Advance the chain as a generator

        Args:
            initial_state (:class:`ParaState` or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`ParaState` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            iterations (int or None, optional): The number of steps to generate.
                ``None`` generates an infinite stream (requires ``store=False``).
                (default: 1)
            tune (bool, optional): If ``True``, the parameters of some moves
                will be automatically tuned. (default: ``False``)
            thin_by (int, optional): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made. (default: 1)
            store (bool, optional): By default, the sampler stores in the backend
                the positions (and other information) of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``. (default: ``True``)
            progress (bool or str, optional): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown. (default: ``False``)
            skip_initial_state_check (bool, optional): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. If using reversible jump, the user needs to ensure this on their own
                (``skip_initial_state_check``is set to ``False`` in this case.
                (default: ``True``)

        Returns:
            ParaState: Every ``thin_by`` steps, this generator yields the :class:`ParaState` of the ensemble.

        Raises:
            ValueError: Improper initialization.

        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")

        # Interpret the input as a walker state and check the dimensions.

        # initial_state.__class__ rather than ParaState in case it is a subclass
        # of ParaState
        if (
            hasattr(initial_state, "__class__")
            and issubclass(initial_state.__class__, ParaState)
            and not isinstance(initial_state.__class__, ParaState)
        ):
            state = initial_state.__class__(initial_state, copy=True)
        else:
            state = ParaState(initial_state, copy=True)

        # Check the backend shape
        for i, (name, branch) in enumerate(state.branches.items()):
            ngroups_, ntemps_, nwalkers_, ndim_ = branch.shape
            if (ngroups_, ntemps_, nwalkers_, ndim_) != (
                self.ngroups,
                self.ntemps,
                self.nwalkers,
                self.ndim,
            ):
                raise ValueError("incompatible input dimensions")

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = {name: value[state.groups_running] for name, value in state.branches_coords.items()}
            state.log_prior = self.xp.full((self.ngroups, self.ntemps, self.nwalkers), -np.inf)
            state.log_prior[state.groups_running] = self.compute_log_prior(coords, groups_running=self.xp.arange(self.ngroups)[state.groups_running])

        if state.log_like is None:
            state.log_like = self.xp.full((self.ngroups, self.ntemps, self.nwalkers), -1e300)
            coords = {name: value[state.groups_running] for name, value in state.branches_coords.items()}
            supps_in = None if state.supplimental is None else state.supplimental[state.groups_running]
            branch_supps_in = {name: None if tmp is None else tmp[state.groups_running] for name, tmp in state.branches_supplimental.items()}

            state.log_like[state.groups_running] = self.compute_log_like(
                coords,
                logp=state.log_prior[state.groups_running],
                supps=supps_in,  # only used if self.provide_supplimental is True
                branch_supps=branch_supps_in,  # only used if self.provide_supplimental is True
            )

        # get betas out of state object if they are there
        if state.betas is not None:
            if (
                state.betas.shape[1] != self.ntemps
                or state.betas.shape[0] != self.ngroups
            ):
                raise ValueError(
                    "Input state has inverse temperatures (betas), but not the correct number of temperatures according to sampler inputs."
                )

            self.betas = self.betas.copy()

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
        supps=None,  # only used if self.provide_supplimental is True
        branch_supps=None,
    ):
        # if supps is not None:
        #     raise NotImplementedError

        # if branch_supps is not None:
        #     if branch_supps[self.name] is not None:
        #         raise NotImplementedError

        if groups_running is not None:
            assert coords[self.name].shape[0] == len(groups_running)

        if logp is None:
            logp = self.compute_log_prior(coords, groups_running=groups_running)

        keep_logp = ~self.xp.isinf(logp)

        coords_arr = coords[self.name][keep_logp]

        logl = self.xp.full_like(logp, -1e300)

        if branch_supps is not None:
            branch_supps = {self.name: {key: branch_supps[self.name][key][keep_logp] for key in branch_supps[self.name]}}

        if self.provide_supplimental:
            kwargs = {**self.logl_kwargs, "branch_supps": branch_supps, "supps": supps}

        else:
            kwargs = self.logl_kwargs

        logl[keep_logp] = self.log_like_fn(
            coords_arr, *self.logl_args, **kwargs
        )

        # fix any nans that may come up
        logl[self.xp.isnan(logl)] = -1e300

        if self.use_gpu:
            self.xp.cuda.runtime.deviceSynchronize()

        return logl

    def compute_log_prior(self, coords, groups_running=None):

        if groups_running is not None:
            assert coords[self.name].shape[0] == len(groups_running)

        shape_in = coords[self.name].shape[:-1]

        coords_logp_buffer = coords[self.name].copy()

        self.prior_transform_fn.transform_to_prior_basis(coords_logp_buffer, groups_running)
        coords_logp_in = coords_logp_buffer.reshape(-1, self.ndim)
        
        logp = self.priors[self.name].logpdf(coords_logp_in).reshape(shape_in)

        self.prior_transform_fn.adjust_logp(logp, groups_running)

        return logp

    def run_mcmc(
        self, initial_state, nsteps, burn=None, post_burn_update=False, **kwargs
    ):
        """
        Iterate :func:`sample` for ``nsteps`` iterations and return the result.

        Args:
            initial_state (ParaState or ndarray[ntemps, nwalkers, nleaves_max, ndim] or dict): The initial
                :class:`ParaState` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. If ``betas`` are
                provided in the state object, they will be loaded into the
                ``temperature_control``.
            nsteps (int): The number of steps to generate. The total number of proposals is ``nsteps * thin_by``.
            burn (int, optional): Number of burn steps to run before storing information. The ``thin_by`` kwarg is ignored when counting burn steps since there is no storage (equivalent to ``thin_by=1``).
            post_burn_update (bool, optional): If ``True``, run ``update_fn`` after burn in.

        Other parameters are directly passed to :func:`sample`.

        Returns:
            ParaState: This method returns the most recent result from :func:`sample`.

        Raises:
            ValueError: ``If initial_state`` is None and ``run_mcmc`` has never been called.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        # setup thin_by info
        thin_by = 1 if "thin_by" not in kwargs else kwargs["thin_by"]

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

        results = None

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
                    (self.ntemps * num_groups_running, int(self.nwalkers / 2), 1, -1)
                )
            )
            c_in = [
                new_state.branches[self.name]
                .coords[:, :, inds_not_here][groups_running]
                .reshape(
                    (self.ntemps * num_groups_running, int(self.nwalkers / 2), 1, -1)
                )
            ]

            temps_here = self.temp_guide[:, :, inds_here][groups_running]
            walkers_here = self.walker_guide[:, :, inds_here][groups_running]
            groups_here = self.group_guide[:, :, inds_here][groups_running]

            if not hasattr(new_state, "random_state") or new_state.random_state is None:
                new_state.random_state = self.random_state

            new_points_dict, factors = self.move_proposal.get_proposal(
                {self.name: s_in}, {self.name: c_in}, new_state.random_state
            )
            new_points = {
                self.name: new_points_dict[self.name].reshape(
                    num_groups_running, self.ntemps, int(self.nwalkers / 2), -1
                )
            }

            logp = self.compute_log_prior(new_points, groups_running=self.xp.arange(self.ngroups)[groups_running])
            factors = factors.reshape(logp.shape)
            
            supps_in = None  # new_state.supplimental[]

            branch_supps_in = {self.name: {key: tmp[groups_running] for key, tmp in new_state.branches_supplimental[self.name][:, :, inds_here].items()}}
            logl = self.compute_log_like(
                new_points, groups_running=self.xp.arange(self.ngroups)[groups_running], logp=logp, supps=supps_in, branch_supps=branch_supps_in
            )

            prev_logl_here = new_state.log_like[:, :, inds_here][groups_running]
            prev_logp_here = new_state.log_prior[:, :, inds_here][groups_running]

            prev_logP_here = (
                self.betas[groups_running][:, :, None] * prev_logl_here + prev_logp_here
            )

            logP = self.betas[groups_running][:, :, None] * logl + logp

            lnpdiff = factors + logP - prev_logP_here
            keep = lnpdiff > self.xp.asarray(
                self.xp.log(new_state.random_state.rand(*logP.shape))
            )

            keep_tuple = (groups_here[keep], temps_here[keep], walkers_here[keep])

            accepted[keep_tuple] = 1
            new_state.log_prior[keep_tuple] = logp[keep]
            new_state.log_like[keep_tuple] = logl[keep]
            new_state.branches[self.name].coords[keep_tuple] = new_points[self.name][
                keep
            ]

        if self.ntemps > 1:
            self.tempering_operations(new_state)

        return new_state, accepted

    def tempering_operations(self, state):
        """IN-PLACE temperature swapping"""

        groups_running = state.groups_running.copy()
        num_groups_running = groups_running.sum().item()

        # prepare information on how many swaps are accepted this time
        self.swaps_accepted = self.xp.zeros(
            (self.ngroups, self.ntemps - 1), dtype=int
        )
        self.swaps_proposed = self.xp.full_like(self.swaps_accepted, self.nwalkers)

        swaps_accepted_tmp = self.xp.zeros(
            (num_groups_running, self.ntemps - 1), dtype=int
        )
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
                self.xp.tile(self.xp.arange(self.nwalkers), (num_groups_running, 1)), -1
            )
            i1perm = shuffle_along_axis(
                self.xp.tile(self.xp.arange(self.nwalkers), (num_groups_running, 1)), -1
            )

            # random draw that produces log of the acceptance fraction
            raccept = self.xp.log(
                state.random_state.uniform(size=(num_groups_running, self.nwalkers))
            )

            # log of the detailed balance fraction
            walker_swap_i = iperm.flatten()
            walker_swap_i1 = i1perm.flatten()

            temp_swap_i = np.full_like(walker_swap_i, i)
            temp_swap_i1 = np.full_like(walker_swap_i1, i - 1)
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

            state.branches[self.name].coords[keep_i_tuple] = state.branches[
                self.name
            ].coords[keep_i1_tuple]
            state.log_like[keep_i_tuple] = state.log_like[keep_i1_tuple]
            state.log_prior[keep_i_tuple] = state.log_prior[keep_i1_tuple]

            state.branches[self.name].coords[keep_i1_tuple] = coords_tmp_i
            state.log_like[keep_i1_tuple] = logl_tmp_i
            state.log_prior[keep_i1_tuple] = logp_tmp_i

        self.swaps_accepted[groups_running] = swaps_accepted_tmp

        # print(prev_logl.max(axis=(1, 2)))
        if self.base_temperature_control.adaptive:
            # print(time.perf_counter() - st)
            ratios = swaps_accepted_tmp / swaps_proposed_tmp

            # adjust temps
            betas0 = self.betas[groups_running].copy()
            betas1 = self.betas[groups_running].copy()

            if not hasattr(self, "time_temp"):
                self.time_temp = 0

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
            betas1[:, 1:-1] = 1 / (np.cumsum(deltaTs, axis=-1) + 1 / betas1[:, 0][:, None])

            dbetas = betas1 - betas0
            state.betas[groups_running] += dbetas


"""
    convergence_iter_count = 500

    new_points = priors_good.rvs(size=(ntemps, nwalkers, len(band_inds_here)))  # , psds=lisasens_in[0][None, :])

    fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)
    while xp.any(fix):
        tmp = priors_good.rvs(size=int((fix.flatten() == True).sum()))
        new_points[fix == True] = tmp
        fix = xp.any(xp.isinf(new_points), axis=-1) | xp.any(xp.isnan(new_points), axis=-1)

    # TODO: fix fs stuff
    prev_logp = priors_good.logpdf(new_points.reshape(-1, ndim)).reshape(new_points.shape[:-1])
    assert not xp.any(xp.isinf(prev_logp))
    new_points_with_fs = new_points.copy()

    L = 2.5e9
    amp_transform = AmplitudeFromSNR(L, current_info.general_info['Tobs'], fd=current_info.general_info["fd"], sens_fn="lisasens", use_cupy=True)

    original_snr_params = new_points_with_fs[:, :, :, 0].copy()
    new_points_with_fs[:, :, :, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * new_points_with_fs[:, :, :, 1] + f0_mins[band_inds_here]
    new_points_with_fs[:, :, :, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * new_points_with_fs[:, :, :, 2] + fdot_mins[band_inds_here]
    new_points_with_fs[:, :, :, 0] = amp_transform(new_points_with_fs[:, :, :, 0].flatten(), new_points_with_fs[:, :, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(new_points_with_fs.shape[:-1])

    lp_factors = np.log(original_snr_params / new_points_with_fs[:, :, :, 0])
    prev_logp += lp_factors
    transform_fn = gb_info["transform"]

    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp).reshape(new_points_with_fs.shape[:-1] + (ndim + 1,)).reshape(-1, ndim + 1)
    inner_product = 4 * df * (xp.sum(data_in[0].conj() * data_in[0] / psd_in[0]) + xp.sum(data_in[1].conj() * data_in[1] / psd_in[1])).real
    ll = (-1/2 * inner_product - xp.sum(xp.log(xp.asarray(psd_in)))).item()
    gb.d_d = inner_product

    start_ll = -1/2 * inner_product
    print(ll)

    waveform_kwargs = gb_info["waveform_kwargs"].copy()
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    prev_logl = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs).reshape(prev_logp.shape))

    if xp.any(xp.isnan(prev_logl)):
        breakpoint()

    old_points = new_points.copy()
    
    best_logl = prev_logl.max(axis=(0, 1))
    best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)
    
    best_logl_coords = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here)))]

    start_best_logl = best_logl.copy()

    
    still_going_here = xp.ones(len(band_inds_here), dtype=bool)
    num_proposals_per = np.zeros_like(still_going_here, dtype=int)
    iter_count = np.zeros_like(still_going_here, dtype=int)
    betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)

    run_number = 0
    for prop_i in range(num_max_proposals):  # tqdm(range(num_max_proposals)):
        # st = time.perf_counter()
        
        original_snr_params = new_points_with_fs[:, :, :, 0].copy()

        

        new_best_logl = prev_logl.max(axis=(0, 1))

        improvement = (new_best_logl - best_logl > 0.01)

        # print(new_best_logl - best_logl, best_logl)
        best_logl[improvement] = new_best_logl[improvement]

        best_logl_ind = prev_logl.reshape(ntemps * nwalkers, len(band_inds_here)).argmax(axis=0)[improvement]
        best_logl_coords[improvement] = old_points.reshape(ntemps * nwalkers, len(band_inds_here), ndim)[(best_logl_ind, xp.arange(len(band_inds_here))[improvement])]

        best_binaries_coords_with_fs = best_logl_coords.copy()

        best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
        best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]
        best_binaries_coords_with_fs[:, 0] = amp_transform(best_binaries_coords_with_fs[:, 0].flatten(), best_binaries_coords_with_fs[:, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(best_binaries_coords_with_fs.shape[:-1])

        best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

        best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

        if prop_i > convergence_iter_count:
            iter_count[improvement] = 0
            iter_count[~improvement] += 1

        num_proposals_per[still_going_here] += 1
        still_going_here[iter_count >= convergence_iter_count] = False
        
        if prop_i % convergence_iter_count == 0:
            print(f"Proposal {prop_i}, Still going:", still_going_here.sum().item())  # , still_going_here[825], np.sort(prev_logl[0, :, 825] - start_ll))
        if run_number == 2:
            iter_count[:] = 0
            collect_sample_check_iter += 1
            if collect_sample_check_iter % thin_by == 0:
                coords_with_fs = old_points.transpose(2, 0, 1, 3)[still_going_here, 0, :].copy()
                coords_with_fs[:, :, 1] = (f0_maxs[band_inds_here[still_going_here]] - f0_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 1] + f0_mins[band_inds_here[still_going_here]][:, None]
                coords_with_fs[:, :, 2] = (fdot_maxs[band_inds_here[still_going_here]] - fdot_mins[band_inds_here][still_going_here])[:, None] * coords_with_fs[:, :, 2] + fdot_mins[band_inds_here[still_going_here]][:, None]
                coords_with_fs[:, :, 0] = amp_transform(coords_with_fs[:, :, 0].flatten(), coords_with_fs[:, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(coords_with_fs.shape[:-1])
    
                samples_store[:, collect_sample_iter] = coords_with_fs.get()
                collect_sample_iter += 1
                print(collect_sample_iter, num_samples_store)
                if collect_sample_iter == num_samples_store:
                    still_going_here[:] = False

        if still_going_here.sum().item() == 0:
            if run_number < 2:
                betas = xp.repeat(xp.asarray(temperature_control.betas[:, None].copy()), len(band_inds_here), axis=-1)
                
                old_points_old = old_points.copy()
                old_points[:] = best_logl_coords[None, None, :]

                gen_points = old_points.transpose(2, 0, 1, 3).reshape(best_logl_coords.shape[0], -1, ndim).copy()
                iter_count[:] = 0
                still_going_here[:] = True

                factor = 1e-5
                cov = xp.ones(ndim) * 1e-3
                cov[1] = 1e-8

                still_going_start_like = xp.ones(best_logl_coords.shape[0], dtype=bool)
                starting_points = np.zeros((best_logl_coords.shape[0], nwalkers * ntemps, ndim))

                iter_check = 0
                max_iter = 10000
                while np.any(still_going_start_like) and iter_check < max_iter:
                    num_still_going_start_like = still_going_start_like.sum().item()
                    
                    start_like = np.zeros((num_still_going_start_like, nwalkers * ntemps))
                
                    logp = np.full_like(start_like, -np.inf)
                    tmp = xp.zeros((num_still_going_start_like, ntemps * nwalkers, ndim))
                    fix = xp.ones((num_still_going_start_like, ntemps * nwalkers), dtype=bool)
                    while xp.any(fix):
                        tmp[fix] = (gen_points[still_going_start_like, :] * (1. + factor * cov * xp.random.randn(num_still_going_start_like, nwalkers * ntemps, ndim)))[fix]

                        tmp[:, :, 3] = tmp[:, :, 3] % (2 * np.pi)
                        tmp[:, :, 5] = tmp[:, :, 5] % (np.pi)
                        tmp[:, :, 6] = tmp[:, :, 6] % (2 * np.pi)
                        logp = priors_good.logpdf(tmp.reshape(-1, ndim)).reshape(tmp.shape[:-1])

                        fix = xp.isinf(logp)
                        if xp.all(fix):
                            factor /= 10.0

                    new_points_with_fs = tmp.copy()

                    original_snr_params = new_points_with_fs[:, :, 0].copy()
                    new_points_with_fs[:, :, 1] = (f0_maxs[None, band_inds_here[still_going_start_like]] - f0_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 1] + f0_mins[None, band_inds_here[still_going_start_like]].T
                    new_points_with_fs[:, :, 2] = (fdot_maxs[None, band_inds_here[still_going_start_like]] - fdot_mins[None, band_inds_here[still_going_start_like]]).T * new_points_with_fs[:, :, 2] + fdot_mins[None, band_inds_here[still_going_start_like]].T
                    new_points_with_fs[:, :, 0] = amp_transform(new_points_with_fs[:, :, 0].flatten(), new_points_with_fs[:, :, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(new_points_with_fs.shape[:-1])
 
                    lp_factors = np.log(original_snr_params / new_points_with_fs[:, :, 0])
                    logp += lp_factors
                    new_points_in = transform_fn.both_transforms(new_points_with_fs.reshape(-1, ndim), xp=xp)

                    start_like = xp.asarray(gb.get_ll(new_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs)).reshape(new_points_with_fs.shape[:-1])

                    old_points[:, :, still_going_start_like, :] = tmp.transpose(1, 0, 2).reshape(ntemps, nwalkers, -1, ndim)
                    prev_logl[:, :, still_going_start_like] = start_like.T.reshape(ntemps, nwalkers, -1)
                    prev_logp[:, :, still_going_start_like] = logp.T.reshape(ntemps, nwalkers, -1)
                    # fix any nans that may come up
                    start_like[xp.isnan(start_like)] = -1e300
                    
                    update = xp.arange(still_going_start_like.shape[0])[still_going_start_like][xp.std(start_like, axis=-1) > 15.0]
                    still_going_start_like[update] = False 

                    iter_check += 1
                    factor *= 1.5
                    
                    # if still_going_start_like[400]:
   
                    #     ind_check = np.where(np.arange(still_going_start_like.shape[0])[still_going_start_like] == 400)[0]
                    #     print(iter_check, still_going_start_like.sum(), start_like[ind_check].max(axis=-1), start_like[ind_check].min(axis=-1), start_like[ind_check].max(axis=-1) - start_like[ind_check].min(axis=-1), xp.std(start_like, axis=-1)[ind_check])

                # breakpoint()
                if run_number == 1:
                    best_binaries_coords_with_fs = best_logl_coords.copy()

                    best_binaries_coords_with_fs[:, 1] = (f0_maxs[band_inds_here] - f0_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 1] + f0_mins[band_inds_here]
                    best_binaries_coords_with_fs[:, 2] = (fdot_maxs[band_inds_here] - fdot_mins[band_inds_here]) * best_binaries_coords_with_fs[:, 2] + fdot_mins[band_inds_here]
                    best_binaries_coords_with_fs[:, 0] = amp_transform(best_binaries_coords_with_fs[:, 0].flatten(), best_binaries_coords_with_fs[:, 1].flatten() / 1e3, psds=lisasens_in[0][None, :])[0].reshape(best_binaries_coords_with_fs.shape[:-1])
            
                    best_logl_points_in = transform_fn.both_transforms(best_binaries_coords_with_fs, xp=xp)

                    best_logl_check = xp.asarray(gb.get_ll(best_logl_points_in, data_in, psd_in, phase_marginalize=True, **waveform_kwargs))

                    if not xp.allclose(best_logl, best_logl_check):
                        breakpoint()

                    snr_lim = gb_info["search_info"]["snr_lim"]
                    keep_binaries = gb.d_h / xp.sqrt(gb.h_h.real) > snr_lim

                    print(f"SNR lim: {snr_lim}")
                    
                    still_going_here = keep_binaries.copy()

                    num_new_binaries = keep_binaries.sum().item()
                    print(f"num new search: {num_new_binaries}")

                    thin_by = 25
                    num_samples_store = 30
                    samples_store = np.zeros((still_going_here.sum().item(), num_samples_store, nwalkers, ndim))
                    collect_sample_iter = 0
                    collect_sample_check_iter = 0

                    # # TODO: add in based on sensitivity changing
                    # # band_inds_running[band_inds_here[~keep_binaries].get()] = False
                    # keep_coords = best_binaries_coords_with_fs[keep_binaries].get()

                    # # adjust the phase from marginalization
                    # phase_change = np.angle(gb.non_marg_d_h)[keep_binaries.get()]
                    # keep_coords[:, 3] -= phase_change
                    # # best_logl_points_in[keep_binaries, 4] -= xp.asarray(phase_change)

                    # # check if there are sources near band edges that are overlapping
                    # assert np.all(keep_coords[:, 1] == np.sort(keep_coords[:, 1]))
                    # f_found = keep_coords[:, 1] / 1e3
                    # N = get_N(np.full_like(f_found, 1e-30), f_found, Tobs=waveform_kwargs["T"], oversample=waveform_kwargs["oversample"])
                    # inds_check = np.where((np.diff(f_found) / df).astype(int) < N[:-1])[0]

                    # params_add = keep_coords[inds_check]
                    # params_remove = keep_coords[inds_check + 1]
                    # N_check = N[inds_check]

                    # params_add_in = transform_fn.both_transforms(params_add)
                    # params_remove_in = transform_fn.both_transforms(params_remove)

                    # waveform_kwargs_tmp = waveform_kwargs.copy()
                    # if "N" in waveform_kwargs_tmp:
                    #     waveform_kwargs_tmp.pop("N")
                    # waveform_kwargs_tmp["use_c_implementation"] = False

                    # gb.swap_likelihood_difference(params_add_in, params_remove_in, data_in, psd_in, N=256, **waveform_kwargs_tmp)

                    # likelihood_difference = -1/2 * (gb.add_add + gb.remove_remove - 2 * gb.add_remove).real.get()
                    # overlap = (gb.add_remove.real / np.sqrt(gb.add_add.real * gb.remove_remove.real)).get()

                    # fix = np.where((likelihood_difference > -100.0) | (overlap > 0.4))

                    # if np.any(fix):
                    #     params_comp_add = params_add[fix]
                    #     params_comp_remove = params_remove[fix]

                    #     # not actually in the data yet, just using swap for quick likelihood comp
                    #     snr_add = (gb.d_h_add.real[fix] / gb.add_add.real[fix] ** (1/2)).get()
                    #     snr_remove = (gb.d_h_remove.real[fix] / gb.remove_remove.real[fix] ** (1/2)).get()

                    #     inds_add = inds_check[fix]
                    #     inds_remove = inds_add + 1

                    #     inds_delete = (inds_add) * (snr_add < snr_remove) + (inds_remove) * (snr_remove < snr_add)
                    #     keep_coords = np.delete(keep_coords, inds_delete, axis=0)
                        

                run_number += 1

            else:
                break

    return fit_gmm(samples_store, comm, comm_info)

    """
