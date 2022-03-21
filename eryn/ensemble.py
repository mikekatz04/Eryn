# -*- coding: utf-8 -*-

import warnings

import numpy as np
from itertools import count
from copy import deepcopy

from .backends import Backend, HDFBackend
from .model import Model
from .moves import StretchMove, TemperatureControl, PriorGenerate, GaussianMove
from .pbar import get_progress_bar
from .state import State
from .prior import PriorContainer
from .utils import PlotContainer
from .utils import PeriodicContainer
from .utils.utility import groups_from_inds


__all__ = ["EnsembleSampler", "walkers_independent"]

try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable


class EnsembleSampler(object):
    """An ensemble MCMC sampler

    The class controls the entire sampling run. It can handle
    everything from a basic non-tempered MCMC to a parallel-tempered,
    global fit containing multiple branches (models) and a variable
    number of leaves (sources) per branch. (# TODO: add link to tree explainer)
    Parameters related to parallelization can be
    controlled via the ``pool`` argument (:ref:`parallel`).

    Args:
        nwalkers (int): The number of walkers in the ensemble.
        ndims (int or list of ints): Number of dimensions in the parameter space
            for each branch tested.
        log_prob_fn (callable): A function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            likelihood for that position. When using reversible jump, this function
            must take a specific set of arguments: ``(x1, group1,...,xN, groupN)``,
            where ``N`` is the number of branches. ``x1,..,xN`` are the coordinates
            for each branch and ``group1,...,groupN`` are the groups of leaves that
            are combined for each walker. This is due to the possibly variable number
            of leaves associated with different walkers. ``x1`` is 2D with shape
            (all included individual leaves of branch 1, ndim1), ``xN`` is 2D with shape
            (all included individual leaves of branch N, ndimN). ``group1`` is
            a 1D int array array of indexes assigning the specific leaf to a
            specific walker. Please see the tutorial for more information. (# TODO: add link to tutorial)
        priors (dict): The prior dictionary can take four forms.
            1) A dictionary with keys as int or tuple containing the int or tuple of int
            that describe the parameter number over which to assess the prior, and values that
            are prior probability distributions that must have a ``logpdf`` class method.
            2) A :class:`eryn.prior.PriorContainer` object.
            3) A dictionary with keys that are ``branch_names`` and values that are dictionaries for
            each branch as described for (1).
            4) A dictionary with keys that are ``branch_names`` and values are
            :class:`eryn.prior.PriorContainer` objects.
        provide_groups (bool, optional): If True, provide groups as described in ``log_prob_fn`` above.
            A group parameter is added for each branch. (default: ``False``)
        tempering_kwargs (dict, optional): Keyword arguments for initialization of the
            tempering class (# TODO: add link to tempering class).  (default: ``{}``)
        nbranches (int, optional): Number of branches (models) tested. (default: ``1``)
        nleaves_max (int, list of int, or int np.ndarray[nbranches], optional):
            Number of maximum allowable leaves for each branch. (default: ``1``)
        nleaves_min (int, list of int, or int np.ndarray[nbranches], optional):
            Number of minimum allowable leaves for each branch. This is only
            used when using reversible jump. (default: ``1``)
        pool (optional): An object with a ``map`` method that follows the same
            calling sequence as the built-in ``map`` function. This is
            generally used to compute the log-probabilities for the ensemble
            in parallel.
        moves (optional): This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(emcee.moves.StretchMove(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``None``, the default will be :class:`StretchMove`.
            (default: ``None``)
        rj_moves (optional): If ``None`` ot ``False``, reversible jump will not be included in the run.
            This can be a single move object, a list of moves,
            or a "weighted" list of the form ``[(eryn.moves.PriorGenerate(),
            0.1), ...]``. When running, the sampler will randomly select a
            move from this list (optionally with weights) for each proposal.
            If ``True``, it defaults to :class:`PriorGenerate`.
            (default: ``None``)
        dr_moves (bool, optional): If ``None`` ot ``False``, delayed rejection when proposing "birth"
            of new components/models will be switched off for this run. Requires ``rj_moves`` set to ``True``.
            (default: ``None``)
        args (optional): A list of extra positional arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_prob_fn(p, *args, **kwargs)``.
        kwargs (optional): A dict of extra keyword arguments for
            ``log_prob_fn``. ``log_prob_fn`` will be called with the sequence
            ``log_prob_fn(p, *args, **kwargs)``.
        backend (optional): Either a :class:`backends.Backend` or a subclass
            (like :class:`backends.HDFBackend`) that is used to store and
            serialize the state of the chain. By default, the chain is stored
            as a set of numpy arrays in memory, but new backends can be
            written to support other mediums.
        vectorize (bool, optional): If ``True``, ``log_prob_fn`` is expected
            to accept a list of position vectors instead of just one. Note
            that ``pool`` will be ignored if this is ``True``.
            (default: ``False``)
        subset (int, optional): Set an amount of walkers to have their likelihood
            calculated (used when ``vectorize == True``). This allows the user
            to maintain memory constraints while adding more walkers to the ensemble.
            When subset is ``None``, all walkers are determined together.
            (default: ``None``) # TODO implement
        autocorr_iter_count (int, optional): Number of iterations of the sampler
            before the autocorrelations are checked and printed.
            ``autocorr_iter_count == -1`` will not check the autocorrelation lengths.
            (default: 100)
        autocorr_multiplier (int, optional): This will stop the sampler if the
            sampler iteration is greater than
            ``autocorr_multiplier * max(autocorrelation lengths)``.
            (default: 50)
        plot_iterations (int, optional): If ``plot_iterations == -1``, then the
            diagnostic plots will not be constructed. Otherwise, the diagnostic
            plots will be constructed every ``plot_iterations`` sampler iterations.
            (default: -1)
        plot_generator (optional): # TODO: add class object that controls
            the diagnostic plotting updates. If not provided and ``plot_iterations > 0``,
            the ensemble will initialize a default plotting setup.
            (default: None)
        plot_name (str, optional): Name of file to save diagnostic plots to. This only
            applies if ``plot_generator == None`` and ``plot_iterations > 0``.
            (default: ``None``)
        periodic (dict, optional): Keys are ``branch_names``. Values are dictionaries
            that have (key: value) pairs as (index to parameter: period). Periodic
            parameters are treated as having periodic boundary conditions in proposals.
        update_fn (optional): :class:`eryn.utils.updates.AdjustStretchProposalScale`
            object that allows the user to update the sampler in any preferred way
            every ``update_iterations`` sampler iterations.
        update_iterations (int, optional): Number of iterations between sampler
            updates using ``update_fn``.
        stopping_fn (optional): :class:`eryn.utils.stopping.Stopping` object that
            allows the user to end the sampler if specified criteria are met.
        stopping_iterations (int, optional): Number of iterations between sampler
            attempts to evaluate the ``stopping_fn``.
        info (dict, optional): Key and value pairs reprenting any information
            the user wants to add to the backend if the user is not inputing
            their own backend.
        fill_zero_leaves_val (double, optional): When there are zero leaves in a
            given walker (across all branches), fill the likelihood value with
            ``fill_zero_leaves_val``. If wanting to keep zero leaves as a possible
            model, this should be set to the value of the contribution to the Likelihood
            from the data. (Default: ``-1e300``).
        verbose (int, optional): # TODO

    Raises:
        ValueError: Any startup issues.

    """

    def __init__(
        self,
        nwalkers,
        ndims,  # assumes ndim_max
        log_prob_fn,
        priors,
        provide_groups=False,  # TODO: improve this
        provide_supplimental=False,  # TODO: improve this
        tempering_kwargs={},
        nbranches=1,
        nleaves_max=1,
        nleaves_min=1,
        # test_inds=None,  # TODO: add ?
        pool=None,
        moves=None,
        rj_moves=None,
        dr_moves=None,
        dr_max_iter=5,
        args=None,
        kwargs=None,
        backend=None,
        vectorize=True,
        subset=None,
        blobs_dtype=None,  # TODO check this
        truth=None,  # TODO: add this
        autocorr_iter_count=100,
        autocorr_multiplier=1000,  # TODO: adjust this to 50
        plot_iterations=-1,
        plot_generator=None,
        plot_name=None,
        periodic=None,  # TODO: add periodic to proposals
        update_fn=None,
        update_iterations=-1,
        stopping_fn=None,
        stopping_iterations=-1,
        info={},
        branch_names=None,
        fill_zero_leaves_val=-1e300,
        verbose=False,
    ):

        # TODO: check non-vectorized

        self.provide_groups = provide_groups
        self.provide_supplimental = provide_supplimental
        self.fill_zero_leaves_val = fill_zero_leaves_val

        # store default branch names if not given
        if branch_names is None:
            branch_names = ["model_{}".format(i) for i in range(nbranches)]

        assert len(branch_names) == nbranches

        # setup dimensions for branches
        # turn things into lists if ints are given
        if isinstance(ndims, int):
            ndims = [ndims for _ in range(nbranches)]
        elif not isinstance(ndims, list):
            raise ValueError("ndims must be integer or list.")

        if isinstance(nleaves_max, int):
            nleaves_max = [nleaves_max]

        # setup temperaing information
        # default is no temperatures
        if tempering_kwargs == {}:
            self.ntemps = 1
            self.temperature_control = None
        else:
            # TODO: fix ndim
            self.temperature_control = TemperatureControl(
                ndims, nwalkers, nleaves_max, **tempering_kwargs
            )
            self.ntemps = self.temperature_control.ntemps

        # setup emcee basics
        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

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

        elif isinstance(priors, PriorContainer):
            self.priors = {"model_0": priors}

        else:
            raise ValueError("Priors must be a dictionary.")

        # set basic variables for sampling settings
        self.ndims = ndims  # interpeted as ndim_max
        self.nwalkers = nwalkers
        self.nbranches = nbranches
        self.nleaves_max = nleaves_max
        self.branch_names = branch_names

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
                    "If providing rj_moves, must provide moves kwargs as well."
                )
            # TODO: remove live_dangerously
            self._moves = [
                StretchMove(
                    live_dangerously=True,
                    temperature_control=self.temperature_control,
                    periodic=periodic,
                    a=2.0,
                )
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

        # parse the reversible jump move schedule
        if isinstance(rj_moves, bool):
            self.has_reversible_jump = rj_moves
            # TODO: deal with tuning
            if self.has_reversible_jump:
                if isinstance(nleaves_min, int):
                    self.nleaves_min = [nleaves_min for _ in range(self.nbranches)]
                elif isinstance(nleaves_min, list):
                    self.nleaves_min = nleaves_min
                else:
                    raise ValueError(
                        "If providing a minimum number of leaves, must be int or list of ints."
                    )

                assert len(self.nleaves_min) == self.nbranches

                rj_move = PriorGenerate(
                    self.priors,
                    self.nleaves_max,
                    self.nleaves_min,
                    self._moves[0],  # TODO: check if necessary
                    dr=dr_moves,
                    dr_max_iter=dr_max_iter,
                    tune=False,
                    temperature_control=self.temperature_control,
                )
                self._rj_moves = [rj_move]
                self._rj_weights = [1.0]

        elif isinstance(rj_moves, Iterable):
            self.has_reversible_jump = True

            try:
                self._rj_moves, self._rj_weights = zip(*rj_moves)
            except TypeError:
                self._rj_moves = rj_moves
                self._rj_weights = np.ones(len(rj_moves))

        else:
            self.has_reversible_jump = False

        # adjust rj weights properly
        if self.has_reversible_jump:
            self._rj_weights = np.atleast_1d(self._rj_weights).astype(float)
            self._rj_weights /= np.sum(self._rj_weights)

        # make sure moves have temperature module
        if self.temperature_control is not None:
            for move in self._moves:
                if move.temperature_control is None:
                    move.temperature_control = self.temperature_control

            if self.has_reversible_jump:
                for move in self._rj_moves:
                    if move.temperature_control is None:
                        move.temperature_control = self.temperature_control

        if periodic is not None:
            for move in self._moves:
                if move.periodic is None:
                    move.periodic = periodic

            if self.has_reversible_jump:
                for move in self._rj_moves:
                    if move.periodic is None:
                        move.periodic = periodic

        # setup backend if not provided or initialized
        if backend is None:
            self.backend = Backend()
        elif isinstance(backend, str):
            self.backend = HDFBackend(backend)
        else:
            self.backend = backend

        self.info = info

        # Deal with re-used backends
        if not self.backend.initialized:
            self._previous_state = None
            self.reset(
                branch_names=branch_names,
                ntemps=self.ntemps,
                nleaves_max=nleaves_max,
                rj=self.has_reversible_jump,
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
            log_prob_fn,
            args,
            kwargs,
            provide_groups=self.provide_groups,
            provide_supplimental=provide_supplimental,
            fill_zero_leaves_val=self.fill_zero_leaves_val,
        )

        self.all_walkers = self.nwalkers * self.ntemps
        self.verbose = verbose

        # prepare plotting
        self.plot_iterations = plot_iterations

        if plot_generator is None and self.plot_iterations > 0:
            # set to default if not provided
            if plot_name is not None:
                name = plot_name
            else:
                name = "output"
            self.plot_generator = PlotContainer(
                fp=name, backend=self.backend, thin_chain_by_ac=True
            )
        elif self.plot_iterations > 0:
            self.plot_generator = plot_generator

        # prepare stopping functions
        self.stopping_fn = stopping_fn
        self.stopping_iterations = stopping_iterations

        # prepare update functions
        self.update_fn = update_fn
        self.update_iterations = update_iterations

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

        Args:
            **info (dict, optional): information to pass to backend reset method.

        """
        self.backend.reset(self.nwalkers, self.ndims, **info)

    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__
        d["pool"] = None
        return d

    def get_model(self):
        """Get ``Model`` object from sampler

        The model object is used to pass necessary information to the
        proposals. This method can be used to retrieve the ``model`` used
        in the sampler from outside the sampler.

        Returns:
            :class:`Model`: ``Model`` object used by sampler.


        """
        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map

        # setup model framework for passing necessary items
        model = Model(
            self.log_prob_fn,
            self.compute_log_prob,
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
        """Advance the chain as a generator

        Args:
            initial_state (State or ndarray[nwalkers, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions.
            iterations (int or NoneType, optional): The number of steps to generate.
                ``None`` generates an infinite stream (requires ``store=False``).
                (default: 1)
            tune (bool, optional): If ``True``, the parameters of some moves
                will be automatically tuned.
            thin_by (int, optional): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
            store (bool, optional): By default, the sampler stores in the backend
                the positions and log-probabilities of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``.
            progress (bool or str, optional): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown.
            skip_initial_state_check (bool, optional): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. (default: ``False``)

        Returns:
            State: Every ``thin_by`` steps, this generator yields the :class:`State` of the ensemble.

        Raises:
            ValueError: Improper initialization.

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

        # get log prior and likelihood if not provided in the initial state
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

        model = self.get_model()

        # Inject the progress bar
        total = None if iterations is None else iterations * yield_step
        with get_progress_bar(progress, total) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(yield_step):
                    # Choose a random move
                    move = self._random.choice(self._moves, p=self._weights)

                    # Propose (in model)
                    state, accepted = move.propose(model, state)
                    if self.ntemps > 1:
                        in_model_swaps = move.temperature_control.swaps_accepted
                    else:
                        in_model_swaps = None

                    state.random_state = self.random_state

                    if tune:
                        move.tune(state, accepted)

                    if self.has_reversible_jump:
                        rj_move = self._random.choice(
                            self._rj_moves, p=self._rj_weights
                        )

                        # Propose (Between models)
                        state, rj_accepted = rj_move.propose(model, state)
                        if self.ntemps > 1:
                            rj_swaps = rj_move.temperature_control.swaps_accepted
                        else:
                            rj_swaps = None

                        state.random_state = self.random_state

                        if tune:
                            rj_move.tune(state, rj_accepted)

                    else:
                        rj_accepted = None
                        rj_swaps = None

                    # Save the new step
                    if store and (i + 1) % checkpoint_step == 0:
                        self.backend.save_step(
                            state,
                            accepted,
                            rj_accepted=rj_accepted,
                            in_model_swaps_accepted=in_model_swaps,
                            rj_swaps_accepted=rj_swaps,
                        )

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state

    def run_mcmc(self, initial_state, nsteps, burn=None, **kwargs):
        """
        Iterate :func:`sample` for ``nsteps`` iterations and return the result

        Args:
            initial_state (State or ndarray[nwalkers, ndim] or dict): The initial
                :class:`State` or positions of the walkers in the
                parameter space. If multiple branches used, must be dict with keys
                as the ``branch_names`` and values as the positions. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps: The number of steps to run.
            burn (int, optional): Number of burn steps to run before storing information.

        Other parameters are directly passed to :func:`sample`.

        Returns:
            State: This method returns the most recent result from :func:`sample`.

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

        thin_by = 1 if "thin_by" not in kwargs else kwargs["thin_by"]

        if burn is not None and burn != 0:
            if self.verbose:
                print("Start burn")

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
                    stop = self.update_fn(i, results, self)
                i += 1

            initial_state = results
            if self.verbose:
                print("Finish burn")

        if nsteps == 0:
            return initial_state
            
        results = None

        i = 0
        for results in self.sample(initial_state, iterations=nsteps, **kwargs):

            # diagnostic plots
            if self.plot_iterations > 0 and (i + 1) % (self.plot_iterations) == 0:
                self.plot_generator.generate_plot_info()  # TODO: remove defaults

            # check for stopping before updating
            if (
                self.stopping_iterations > 0
                and self.stopping_fn is not None
                and (i + 1) % (self.stopping_iterations) == 0
            ):
                stop = self.stopping_fn(i, results, self)

                if stop:
                    break

            # update after diagnostic and stopping check
            if (
                self.update_iterations > 0
                and self.update_fn is not None
                and (i + 1) % (self.update_iterations) == 0
            ):
                self.update_fn(i, results, self)

            i += 1

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results

    def compute_log_prior(self, coords, inds=None):
        """Calculate the vector of log-prior for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the inds np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)

        Returns:
            np.ndarray[ntemps, nwalkers]: Prior Values

        """

        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        # take information out of dict and spread to x1..xn
        if inds is None:
            # default use all sources
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        x_in = {}
        if self.provide_groups:

            # get group information from the inds dict
            groups = groups_from_inds(inds)

            for i, (name, coords_i) in enumerate(coords.items()):
                x_in[name] = coords_i[inds[name]]

            prior_out = np.zeros((ntemps * nwalkers))
            for name in x_in:
                prior_out_temp = self.priors[name].logpdf(x_in[name])

                # arrange prior values by groups
                for i in np.unique(groups[name]):
                    inds_temp = np.where(groups[name] == i)[0]
                    num_in_group = len(inds_temp)
                    # check = (prior_out_temp[inds_temp].sum() / num_in_group)
                    prior_out[i] += prior_out_temp[inds_temp].sum()

            prior_out = prior_out.reshape(ntemps, nwalkers)
            return prior_out

        else:
            for i, (name, coords_i) in enumerate(coords.items()):
                ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

                # TODO: add copy here?
                x_in[name] = coords_i.reshape(-1, ndim)

            prior_out = np.zeros((ntemps, nwalkers))
            for name in x_in:
                prior_out_temp = self.priors[name].logpdf(x_in[name]) * inds[name].flatten()
                prior_out += prior_out_temp.reshape(ntemps, nwalkers, nleaves_max).sum(
                    axis=-1
                )

            return prior_out

    def compute_log_prob(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        """Calculate the vector of log-likelihood for the walkers

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the inds np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)
            logp (np.ndarray[ntemps, nwalkers], optional): Log prior values associated
                with all walkers. If not provided, it will be calculated because
                if a walker has logp = -inf, its likelihood is not calculated.
                This prevents evaluting likelihood outside the prior.
                (default: ``None``)

        Returns:
            tuple: Carries log-likelihood and blob information.
                First entry is np.ndarray[ntemps, nwalkers] with values corresponding
                to the log likelihood of each walker. Second entry is ``blobs``.

         Raises:
            ValueError: Infinite or NaN values in parameters.

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
            if inds is None:
                inds = {
                    name: np.full(coords[name].shape[:-1], True, dtype=bool)
                    for name in coords
                }

            # if no prior values are added, compute_prior
            if logp is None:
                logp = self.compute_log_prior(coords, inds=inds)

            if np.all(np.isinf(logp)):
                return np.full_like(logp, -1e300), None

            # do not run log likelihood where logp = -inf
            inds_copy = deepcopy(inds)
            inds_bad = np.where(np.isinf(logp))
            for key in inds_copy:
                inds_copy[key][inds_bad] = False

                if branch_supps is not None and branch_supps[key] is not None:
                    branch_supps[key][inds_bad] = {"inds_keep": False}

            results = self.log_prob_fn(p, inds=inds_copy, supps=supps, branch_supps=branch_supps)
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

    @property
    def rj_acceptance_fraction(self):
        """The fraction of proposed reversible jump steps that were accepted"""
        if self.has_reversible_jump:
            return self.backend.rj_accepted / float(self.backend.iteration)
        else:
            return None

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

    def get_betas(self, **kwargs):
        return self.backend.get_betas(**kwargs)

    get_betas.__doc__ = Backend.get_betas.__doc__

    def get_value(self, name, **kwargs):
        """Get a specific value"""
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        """Compute autocorrelation time through backend."""
        return self.backend.get_autocorr_time(**kwargs)

    get_autocorr_time.__doc__ = Backend.get_autocorr_time.__doc__


class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(
        self, f, args, kwargs, provide_groups=False, provide_supplimental=False, fill_zero_leaves_val=-1e300
    ):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.provide_groups = provide_groups
        self.provide_supplimental = provide_supplimental
        self.fill_zero_leaves_val = fill_zero_leaves_val

    def __call__(self, x, inds=None, supps=None, branch_supps=None):
        try:
            # take information out of dict and spread to x1..xn
            x_in = {}
            if self.provide_supplimental:
                if supps is None and branch_supps is None:
                    raise ValueError("supps and branch_supps are both None. If self.provide_supplimental is True, must provide some supplimental information.")
                if branch_supps is not None:
                    branch_supps_in = {}
                    

            if inds is None:
                inds = {
                    name: np.full(x[name].shape[:-1], True, dtype=bool) for name in x
                }

            # determine groupings from inds
            groups = groups_from_inds(inds)

            # need to map group inds properly
            # this is the unique group indexes
            unique_groups = np.unique(
                np.concatenate([groups_i for groups_i in groups.values()])
            )

            # this is the map to those indexes that are used in the likelihood
            groups_in = np.arange(len(unique_groups))

            ll_groups = {}
            for key, group in groups.items():
                temp_unique_groups, inverse = np.unique(group, return_inverse=True)
                keep_groups = groups_in[np.in1d(unique_groups, temp_unique_groups)]
                ll_groups[key] = keep_groups[inverse]

            for i, (name, coords) in enumerate(x.items()):
                ntemps, nwalkers, nleaves_max, ndim = coords.shape
                nwalkers_all = ntemps * nwalkers
                x_in[name] = coords[inds[name]]
                
                if self.provide_supplimental:
                    if branch_supps is not None and branch_supps[name] is not None:
                        branch_supps_in[name] = branch_supps[name][inds[name]]

            if self.provide_supplimental:
                if supps is not None:
                    temp = supps.flat
                    supps_in = {name: values[keep_groups] for name, values in temp.items()}

            args_in = []
            
            params_in = list(x_in.values()) 
            
            if len(params_in) == 1:
                params_in = params_in[0]

            args_in.append(params_in)
            
            if self.provide_groups:
                groups_in = list(ll_groups.values())
                if len(groups_in) == 1:
                    groups_in = groups_in[0]

                args_in.append(groups_in)

            args_in += self.args

            kwargs_in = self.kwargs.copy()
            if self.provide_supplimental:
                if supps is not None:
                    kwargs_in["supps"] = supps_in
                if branch_supps is not None:
                    branch_supps_in_2 = list(branch_supps_in.values())
                    if len(branch_supps_in_2) == 1:
                        kwargs_in["branch_supps"] = branch_supps_in_2

                    elif len(branch_supps_in_2) == 1:
                        kwargs_in["branch_supps"] = branch_supps_in_2[0]
                    
            # TODO: this may have pickle issue with multiprocessing (kwargs_in)
            out = self.f(*args_in, **kwargs_in)

            if self.provide_supplimental:
                if branch_supps is not None:
                    for name_i, name in enumerate(branch_supps):
                        if branch_supps[name] is not None:
                            # TODO: better way to do this? limit to 
                            if "inds_keep" in branch_supps[name]:
                                inds_back = branch_supps[name][:]["inds_keep"]
                                inds_back2 = branch_supps_in[name]["inds_keep"]
                            else:
                                inds_back = inds[name]
                                inds_back2 = slice(None)
                            try:
                                branch_supps[name][inds_back] = {key: branch_supps_in_2[name_i][key][inds_back2] for key in branch_supps_in_2[name_i]}
                            except ValueError:
                                breakpoint()
                                branch_supps[name][inds_back] = {key: branch_supps_in_2[name_i][key][inds_back2] for key in branch_supps_in_2[name_i]}
                                
            # -1e300 because -np.inf screws up state acceptance transfer in proposals
            ll = np.full(nwalkers_all, -1e300)
            inds_fix_zeros = np.delete(np.arange(nwalkers_all), unique_groups)

            if out.ndim == 2:
                ll[unique_groups] = out[:, 0]
                ll[inds_fix_zeros] = self.fill_zero_leaves_val
                blobs_out = np.zeros((nwalkers_all, out.shape[1] - 1))
                blobs_out[unique_groups] = out[:, 1:]
                return [
                    ll.reshape(ntemps, nwalkers),
                    blobs_out.reshape(ntemps, nwalkers, -1),
                ]
            else:
                try:
                    ll[unique_groups] = out
                except ValueError:
                    breakpoint()
                ll[inds_fix_zeros] = self.fill_zero_leaves_val
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
