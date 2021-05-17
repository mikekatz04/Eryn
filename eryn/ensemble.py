# -*- coding: utf-8 -*-

import warnings

import numpy as np
from itertools import count
from copy import deepcopy

from .backends import Backend
from .model import Model
from .moves import StretchMove, TemperatureControl, PriorGenerate, GaussianMove
from .pbar import get_progress_bar
from .state import State
from .prior import PriorContainer
from .utils import PlotContainer

__all__ = ["EnsembleSampler", "walkers_independent"]

try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable


class EnsembleSampler(object):
    """An ensemble MCMC sampler

    If you are upgrading from an earlier version of emcee, you might notice
    that some arguments are now deprecated. The parameters that control the
    proposals have been moved to the :ref:`moves-user` interface (``a`` and
    ``live_dangerously``), and the parameters related to parallelization can
    now be controlled via the ``pool`` argument (:ref:`parallel`).

    Args:
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): Number of dimensions in the parameter space.
        log_prob_fn (callable): A function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            posterior probability (up to an additive constant) for that
            position.
        moves (Optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(emcee.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            (default: :class:`StretchMove`)
        args (Optional): A list of extra positional arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_pprob_fn(p, *args, **kwargs)``.
        kwargs (Optional): A dict of extra keyword arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_pprob_fn(p, *args, **kwargs)``.
        pool (Optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        backend (Optional): Either a :class:`backends.Backend` or a subclass
            (like :class:`backends.HDFBackend`) that is used to store and
            serialize the state of the chain. By default, the chain is stored
            as a set of numpy arrays in memory, but new backends can be
            written to support other mediums.
        vectorize (Optional[bool]): If ``True``, ``log_prob_fn`` is expected
            to accept a list of position vectors instead of just one. Note
            that ``pool`` will be ignored if this is ``True``.
            (default: ``False``)

    """

    def __init__(
        self,
        nwalkers,
        ndims,  # assumes ndim_max
        log_prob_fn,
        priors,
        rj=False,
        provide_groups=False,  # TODO: improve this
        tempering_kwargs={},
        nbranches=1,
        nleaves_max=1,
        test_inds=None,
        pool=None,
        moves=None,
        args=None,
        kwargs=None,
        backend=None,
        subset=None,
        blobs_dtype=None,
        truth=None,
        autocorr_iter_count=100,
        autocorr_multiplier=1000,
        plot_iterations=-1,
        plot_generator=None,
        periodic=None,  # TODO: add periodic to proposals
        update_fn=None,
        update=-1,
        update_kwargs={},
        stopping_fn=None,
        stopping_iterations=-1,
        info={},
        branch_names=None,
        vectorize=True,
        verbose=False,
        cov=None,  # TODO: change this
    ):

        # TODO: check non-vectorized

        self.provide_groups = provide_groups

        if branch_names is None:
            branch_names = ["model_{}".format(i) for i in range(nbranches)]

        assert len(branch_names) == nbranches

        if isinstance(ndims, int):
            ndims = [ndims for _ in range(nbranches)]
        elif not isinstance(ndims, list):
            raise ValueError("ndims must be integer or list.")

        if isinstance(nleaves_max, int):
            nleaves_max = [nleaves_max]

        if tempering_kwargs == {}:
            self.ntemps = 1
            self.temperature_control = None
        else:
            # TODO: fix ndim
            self.temperature_control = TemperatureControl(
                ndims, nwalkers, **tempering_kwargs
            )
            self.ntemps = self.temperature_control.ntemps

        # Parse the move schedule
        if moves is None:

            if cov is None:
                # TODO: remove live_dangerously
                self._moves = [
                    StretchMove(
                        live_dangerously=True,
                        temperature_control=self.temperature_control,
                        a=1.1,
                    )
                ]
                self._weights = [1.0]

            else:
                # TODO: remove live_dangerously
                self._moves = [
                    GaussianMove(cov, temperature_control=self.temperature_control)
                ]
                self._weights = [1.0]
        elif isinstance(moves, Iterable):
            try:
                self._moves, self._weights = zip(*moves)
            except TypeError:
                self._moves = moves
                self._weights = np.ones(len(moves))
        else:
            self._moves = [moves]
            self._weights = [1.0]

        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= np.sum(self._weights)

        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

        # TODO: deal with priors consistent across leaves
        if isinstance(priors, dict):
            test = priors[list(priors.keys())[0]]
            if isinstance(test, dict):
                # check all dists
                for name, priors_temp in priors.items():
                    for ind, dist in priors_temp.items():
                        if not hasattr(dist, "logpdf"):
                            raise ValueError(
                                "Distribution for model {0} and index {1} does not have logpdf method.".format(
                                    name, ind
                                )
                            )
                self.priors = {
                    name: PriorContainer(priors_temp)
                    for name, priors_temp in priors.items()
                }

            elif isinstance(test, PriorContainer):
                self.priors = priors

            elif hasattr(test, "logpdf"):
                self.priors = {"model_0": PriorContainer(priors)}

            else:
                raise ValueError(
                    "priors dictionary items must be dictionaries with prior information or instances of the PriorContainer class."
                )
        else:
            raise ValueError("Priors must be a dictionary.")

        self.ndims = ndims  # interpeted as ndim_max
        self.nwalkers = nwalkers
        self.nbranches = nbranches
        self.nleaves_max = nleaves_max
        self.branch_names = branch_names

        # TODO: adjust for how we want to choose if rj / for now it is rj == True
        if rj:
            # TODO: make min_k adjustable
            # TODO: deal with tuning
            min_k = [1, 1]
            rj_move = PriorGenerate(
                self.priors,
                self._moves,
                self._weights,
                self.nleaves_max,
                min_k,
                tune=False,
                temperature_control=self.temperature_control,
            )
            self._moves = [rj_move]
            self._weights = [1.0]

        self.backend = Backend() if backend is None else backend
        self.info = info

        # Deal with re-used backends
        if not self.backend.initialized:
            self._previous_state = None
            self.reset(
                branch_names=branch_names,
                ntemps=self.ntemps,
                nleaves_max=nleaves_max,
                **info
            )
            state = np.random.get_state()
        else:
            # Check the backend shape
            for i, (name, shape) in enumerate(self.backend.shape.items()):
                test_shape = (
                    self.ntemps,
                    self.nwalkers,
                    self.nleaves_max[i],
                    self.ndims[i],
                )
                if shape != test_shape:
                    raise ValueError(
                        (
                            "the shape of the backend ({0}) is incompatible with the "
                            "shape of the sampler ({1} for model {2})"
                        ).format(shape, test_shape, name)
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
        self.log_prob_fn = _FunctionWrapper(
            log_prob_fn, args, kwargs, provide_groups=self.provide_groups
        )

        # update information
        self.update_fn = update_fn
        self.update = update
        self.update_kwargs = update_kwargs.copy()

        # stopping information
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations

        self.all_walkers = self.nwalkers * self.ntemps
        self.verbose = verbose

        # prepare plotting
        self.plot_iterations = plot_iterations

        if plot_generator is None and self.plot_iterations > 0:
            self.plot_generator = PlotContainer(
                "output", backend=self.backend, thin_chain_by_ac=True
            )

        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations

    @property
    def random_state(self):
        """
        The state of the internal random number generator. In practice, it's
        the result of calling ``get_state()`` on a
        ``numpy.random.mtrand.RandomState`` object. You can try to set this
        property but be warned that if you do this and it fails, it will do
        so silently.

        """
        return self._random.get_state()

    @random_state.setter  # NOQA
    def random_state(self, state):
        """
        Try to set the state of the random number generator but fail silently
        if it doesn't work. Don't say I didn't warn you...

        """
        try:
            self._random.set_state(state)
        except:
            pass

    @property
    def iteration(self):
        return self.backend.iteration

    def reset(self, **info):
        """
        Reset the bookkeeping parameters

        """
        self.backend.reset(self.nwalkers, self.ndims, **info)

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["pool"] = None
        return d

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
            initial_state (State or ndarray[nwalkers, ndim]): The initial
                :class:`State` or positions of the walkers in the
                parameter space.
            iterations (Optional[int or NoneType]): The number of steps to generate.
                ``None`` generates an infinite stream (requires ``store=False``).
            tune (Optional[bool]): If ``True``, the parameters of some moves
                will be automatically tuned.
            thin_by (Optional[int]): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
            store (Optional[bool]): By default, the sampler stores (in memory)
                the positions and log-probabilities of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``.
            progress (Optional[bool or str]): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown.
            skip_initial_state_check (Optional[bool]): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. (default: ``False``)


        Every ``thin_by`` steps, this generator yields the
        :class:`State` of the ensemble.

        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")

        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        # Check the backend shape
        for i, (name, branch) in enumerate(state.branches.items()):
            ntemps_, nwalkers_, nleaves_, ndim_ = branch.shape
            if (ntemps_, nwalkers_, ndim_) != (
                self.ntemps,
                self.nwalkers,
                self.ndims[i],
            ) or nleaves_ > self.nleaves_max[i]:
                raise ValueError("incompatible input dimensions")

        if (not skip_initial_state_check) and (not walkers_independent(state.coords)):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        if state.log_prior is None:
            coords = {name: branch.coords for name, branch in state.branches.items()}
            if self.provide_groups:
                inds = {name: branch.inds for name, branch in state.branches.items()}
            else:
                inds = None
            state.log_prior = self.compute_log_prior(coords, inds=inds)

        if state.log_prob is None:
            coords = {name: branch.coords for name, branch in state.branches.items()}

            if self.provide_groups:
                inds = {name: branch.inds for name, branch in state.branches.items()}
            else:
                inds = None
            state.log_prob, state.blobs = self.compute_log_prob(
                coords, inds=inds, logp=state.log_prior
            )

        if np.shape(state.log_prob) != (self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")
        if np.shape(state.log_prior) != (self.ntemps, self.nwalkers):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(state.log_prob)):
            raise ValueError("The initial log_prob was NaN")

        # Check that the thin keyword is reasonable.
        thin_by = int(thin_by)
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

        yield_step = thin_by
        checkpoint_step = thin_by
        if store:
            self.backend.grow(iterations, state.blobs)

        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        model = Model(
            self.log_prob_fn,
            self.compute_log_prob,
            self.compute_log_prior,
            self.temperature_control,
            map_fn,
            self._random,
        )

        # Inject the progress bar
        total = None if iterations is None else iterations * yield_step
        with get_progress_bar(progress, total) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(yield_step):
                    # Choose a random move
                    move = self._random.choice(self._moves, p=self._weights)

                    # Propose
                    state, accepted = move.propose(model, state)
                    state.random_state = self.random_state

                    if tune:
                        move.tune(state, accepted)

                    # Save the new step
                    if store and (i + 1) % checkpoint_step == 0:
                        self.backend.save_step(state, accepted)

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state

    def run_mcmc(self, initial_state, nsteps, burn=None, **kwargs):
        """
        Iterate :func:`sample` for ``nsteps`` iterations and return the result

        Args:
            initial_state: The initial state or position vector. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps: The number of steps to run.

        Other parameters are directly passed to :func:`sample`.

        This method returns the most recent result from :func:`sample`.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        if burn is not None:
            print("Start burn")
            burn_kwargs = deepcopy(kwargs)
            burn_kwargs["store"] = False
            burn_kwargs["thin_by"] = 1
            for results in self.sample(initial_state, iterations=burn, **burn_kwargs):
                pass
            initial_state = results
            print("Finish burn")

        thin_by = 1 if "thin_by" not in kwargs else kwargs["thin_by"]
        results = None
        i = 0
        for results in self.sample(initial_state, iterations=nsteps, **kwargs):

            if (
                self.plot_iterations > 0
                and (i + 1) % (self.plot_iterations * thin_by) == 0
            ):
                self.plot_generator.generate_update()  # TODO: remove defaults

            if (
                self.stopping_iterations > 0
                and self.stopping_fn is not None
                and (i + 1) % (self.stopping_iterations * thin_by) == 0
            ):
                stop = self.stopping_fn(i, results, self)

                if stop:
                    break

            i += 1

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results

    def compute_log_prior(self, coords, inds=None):

        # TODO: expand out like likelihood
        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        # take information out of dict and spread to x1..xn
        x_in = {}
        if self.provide_groups:
            if inds is None:
                inds = {
                    name: np.full(coords[name].shape[:-1], True, dtype=bool)
                    for name in coords
                }

            groups = groups_from_inds(inds)

            num_groups = groups[list(groups.keys())[0]].max() + 1

            for i, (name, coords_i) in enumerate(coords.items()):
                x_in[name] = coords_i[inds[name]]

            prior_out = np.zeros((ntemps * nwalkers))
            for name in x_in:
                prior_out_temp = self.priors[name].logpdf(x_in[name])
                for i in range(num_groups):
                    inds_temp = np.where(groups[name] == i)[0]
                    num_in_group = len(inds_temp)
                    check = prior_out_temp[inds_temp].sum() / num_in_group
                    prior_out[i] += check

            prior_out = prior_out.reshape(ntemps, nwalkers)
            return prior_out

        else:
            for i, (name, coords_i) in enumerate(coords.items()):
                ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

                # TODO: add copy here?
                x_in[name] = coords_i.reshape(-1, ndim)

                prior_out = np.zeros((ntemps, nwalkers))
                for name in x_in:
                    prior_out_temp = self.priors[name].logpdf(x_in[name])
                    prior_out += prior_out_temp.reshape(
                        ntemps, nwalkers, nleaves_max
                    ).sum(axis=-1)

            return prior_out

        # START HERE and add inds and then do tempering

    def compute_log_prob(self, coords, inds=None, logp=None):
        """Calculate the vector of log-probability for the walkers

        Args:
            coords: (ndarray[..., ndim]) The position vector in parameter
                space where the probability should be calculated.

        This method returns:

        * log_prob: A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.
        * blob: The list of meta data returned by the ``log_post_fn`` at
          this position or ``None`` if nothing was returned.

        """
        p = coords

        # Check that the parameters are in physical ranges.
        for ptemp in p.values():
            if np.any(np.isinf(ptemp)):
                raise ValueError("At least one parameter value was infinite")
            if np.any(np.isnan(ptemp)):
                raise ValueError("At least one parameter value was NaN")

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            # do not run log likelihood where logp = -inf
            if inds is None:
                inds = {
                    name: np.full(coords[name].shape[:-1], True, dtype=bool)
                    for name in coords
                }

            # if no prior values are added, compute_prior
            if logp is None:
                logp = self.compute_log_prior_fn(q, inds=inds)

            inds_copy = deepcopy(inds)
            inds_bad = np.where(np.isinf(logp))
            for key in inds_copy:
                inds_copy[key][inds_bad] = False

            results = self.log_prob_fn(p, inds=inds_copy)
            if not isinstance(results, list):
                results = [results]
        else:
            raise NotImplementedError
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map
            results = list(map_func(self.log_prob_fn, (p[i] for i in range(len(p)))))

        log_prob = results[0]
        try:
            blob = results[1]

        except (IndexError, TypeError):
            blob = None

        """
        # TODO: adjust this
        try:
            log_prob = np.array([float(l[0]) for l in results])
            blob = [l[1:] for l in results]
        except (IndexError, TypeError):
            log_prob = np.array([float(l) for l in results])
            blob = None
        else:
            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
            else:
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("error",
                                              np.VisibleDeprecationWarning)
                        try:
                            dt = np.atleast_1d(blob[0]).dtype
                        except Warning:
                            deprecation_warning(
                                "You have provided blobs that are not all the "
                                "same shape or size. This means they must be "
                                "placed in an object array. Numpy has "
                                "deprecated this automatic detection, so "
                                "please specify "
                                "blobs_dtype=np.dtype('object')")
                            dt = np.dtype("object")
                except ValueError:
                    dt = np.dtype("object")
                if dt.kind in "US":
                    # Strings need to be object arrays or we risk truncation
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")
        """
        return log_prob, blob

    @property
    def acceptance_fraction(self):
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend.iteration)

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    get_chain.__doc__ = Backend.get_chain.__doc__

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    get_blobs.__doc__ = Backend.get_blobs.__doc__

    def get_log_prob(self, **kwargs):
        return self.get_value("log_prob", **kwargs)

    get_log_prob.__doc__ = Backend.get_log_prob.__doc__

    def get_inds(self, **kwargs):
        return self.get_value("inds", **kwargs)

    get_inds.__doc__ = Backend.get_inds.__doc__

    def get_nleaves(self, **kwargs):
        return self.backend.get_nleaves(**kwargs)

    get_nleaves.__doc__ = Backend.get_nleaves.__doc__

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    get_last_sample.__doc__ = Backend.get_last_sample.__doc__

    def get_value(self, name, **kwargs):
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


def groups_from_inds(inds):
    groups = {}
    for name, inds_temp in inds.items():
        if inds_temp is None:
            inds_temp = np.full(x[name].shape[:-1], True, dtype=bool)
        ntemps, nwalkers, nleaves_max = inds_temp.shape
        num_groups = ntemps * nwalkers

        group_id = np.repeat(
            np.arange(num_groups).reshape(ntemps, nwalkers)[:, :, None],
            nleaves_max,
            axis=-1,
        )

        groups[name] = group_id[inds_temp]
    return groups


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(self, f, args, kwargs, provide_groups=True):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.provide_groups = provide_groups

    def __call__(self, x, inds=None):
        try:
            # take information out of dict and spread to x1..xn
            x_in = {}
            if inds is None:
                inds = {
                    name: np.full(x[name].shape[:-1], True, dtype=bool) for name in x
                }

            groups = groups_from_inds(inds)

            ll_groups = {}
            temp_unique_groups = []
            for key, group in groups.items():
                unique_groups, inverse = np.unique(group, return_inverse=True)
                ll_groups[key] = np.arange(len(unique_groups))[inverse]
                temp_unique_groups.append(unique_groups)

            unique_groups = np.unique(np.concatenate(temp_unique_groups))

            for i, (name, coords) in enumerate(x.items()):
                ntemps, nwalkers, nleaves_max, ndim = coords.shape
                nwalkers_all = ntemps * nwalkers
                x_in[name] = coords[inds[name]]

            if self.provide_groups:
                args_in = list(x_in.values()) + list(ll_groups.values())

            else:
                args_in = list(x_in.values())

            args_in += list(self.args)

            out = self.f(*args_in, **self.kwargs)

            # -1e300 because -np.inf screws up state acceptance transfer in proposals
            ll = np.full(nwalkers_all, -1e300)
            if out.ndim == 2:
                ll[unique_groups] = out[:, 0]
                blobs_out = np.zeros_like(out[:, 1:])
                blobs_out[unique_groups] = out[:, 1:]

                return [
                    ll.reshape(ntemps, nwalkers),
                    blobs_out.reshape(ntemps, nwalkers, -1),
                ]
            else:
                ll[unique_groups] = out
                return ll.reshape(ntemps, nwalkers)

        except:  # pragma: no cover
            import traceback

            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def walkers_independent(coords):
    raise NotImplementedError
    if not np.all(np.isfinite(coords)):
        return False
    C = coords - np.mean(coords, axis=(1, 2))[:, None, None, :, :]
    C_colmax = np.amax(np.abs(C), axis=(1, 2))[:, None, None, :, :]
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C ** 2, axis=(1, 2)))[:, None, None, :, :]
    C /= C_colsum
    cond = np.array([np.linalg.cond(C[i,].astype(float))])
    return np.all(cond <= 1e8)


def walkers_independent_cov(coords):
    C = np.cov(coords, rowvar=False)
    if np.any(np.isnan(C)):
        return False
    return _scaled_cond(np.atleast_2d(C)) <= 1e8


def _scaled_cond(a):
    asum = np.sqrt((a ** 2).sum(axis=0))[None, :]
    if np.any(asum == 0):
        return np.inf
    b = a / asum
    bsum = np.sqrt((b ** 2).sum(axis=1))[:, None]
    if np.any(bsum == 0):
        return np.inf
    c = b / bsum
    return np.linalg.cond(c.astype(float))
