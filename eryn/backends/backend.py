# -*- coding: utf-8 -*-

import numpy as np

from ..utils.utility import get_integrated_act, thermodynamic_integration_log_evidence
from ..state import State

__all__ = ["Backend"]


class Backend(object):
    """A simple default backend that stores the chain in memory

    Args:
        store_missing_leaves (double, optional): Number to store for leaves that are not
            used in a specific step. (default: ``np.nan``)
        dtype (dtype, optional): Dtype to use for data storage. If None,
            program uses np.float64. (default: ``None``)

    Attributes:
        accepted (2D int np.ndarray): Number of accepted steps for within-model moves.
            The shape is (ntemps, nwalkers).
        betas (2D double np.ndarray): Inverse temperature latter at each step.
            Shape is (nsteps, ntemps). This keeps track of adjustable temperatures.
        blobs (4D double np.ndarray): Stores extra blob information returned from
            likelihood function. Shape is (nsteps, ntemps, nwalkers, nblobs).
        branch_names (list of str): List of branch names.
        chain (dict): Dictionary with branch_names as keys. The values are
            5D double np.ndarray arrays with shape (nsteps, ntemps, nwalkers, nleaves_max, ndim).
            These are the locations of walkers over the MCMC run.
        dtype (dtype): Dtype to use for data storage.
        inds (dict): Keys are branch_names. Values are 4D bool np.ndarray
            of shape (nsteps, ntemps, nwalkers, nleaves_max). This array details which
            leaves are used in the current step. This is really only
            relevant for reversible jump.
        initiailized (bool): If True, backend object has been initialized.
        iteration (int): Current index within the data storage arrays.
        log_prior (3D double np.ndarray): Log of the prior values. Shape is
            (nsteps, nwalkers, ntemps).
        log_prob (3D double np.ndarray): Log of the likelihood values. Shape is
            (nsteps, nwalkers, ntemps).
        nbranches (int): Number of branches.
        ndims (1D int np.ndarray): Dimensionality of each branch.
        nleaves_max (1D int np.ndarray): Maximum allowable leaves for each branch.
        nwalkers (int): The size of the ensemble
        ntemps (int): Number of rungs in the temperature ladder.
        reset_args (tuple): Arguments to reset backend.
        reset_kwargs (dict): Keyword arguments to reset backend.
        rj (bool): If True, reversible-jump techniques are used.
        rj_accepted (2D int np.ndarray): Number of accepted steps for between-model moves.
            The shape is (ntemps, nwalkers).
        store_missing_leaves (double): Number to store for leaves that are not
            used in a specific step.

    """

    def __init__(self, store_missing_leaves=np.nan, dtype=None):
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    def reset_base(self):
        """Allows for simple reset based on previous inputs"""
        self.reset(*self.reset_args, **self.reset_kwargs)

    def reset(
        self,
        nwalkers,
        ndims,
        nleaves_max=1,
        ntemps=1,
        truth=None,
        branch_names=None,
        rj=False,
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble
            ndims (int or list of ints): The number of dimensions for each branch.
            nleaves_max (int or list of ints or 1D np.ndarray of ints, optional):
                Maximum allowable leaf count for each branch. It should have
                the same length as the number of branches.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            truth (list or 1D double np.ndarray, optional): injection parameters.
                (default: ``None``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend.

        """

        # store inputs for later resets
        self.reset_args = (nwalkers, ndims)
        self.reset_kwargs = dict(
            nleaves_max=nleaves_max,
            ntemps=ntemps,
            truth=truth,
            branch_names=branch_names,
            rj=rj,
            info=info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.rj = rj

        if isinstance(ndims, int):
            self.ndims = np.array([ndims])
        elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
            self.ndims = np.asarray(ndims)
        else:
            raise ValueError("ndims is to be a scalar int or a list.")

        if isinstance(nleaves_max, int):
            self.nleaves_max = np.array([nleaves_max])
        elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
            self.nleaves_max = np.asarray(nleaves_max)
        else:
            raise ValueError("nleaves_max is to be a scalar int or a list.")

        if len(self.nleaves_max) != len(self.ndims):
            raise ValueError(
                "Number of branches indicated by nleaves_max and ndims are not equivalent (nleaves_max: {}, ndims: {}).".format(
                    len(self.nleaves_max), len(self.ndims)
                )
            )

        self.nbranches = len(self.nleaves_max)

        # fill branch names accordingly
        if branch_names is not None:
            if isinstance(branch_names, str):
                branch_names = [branch_names]

            elif not isinstance(branch_names, list):
                raise ValueError("branch_names must be string or list of strings.")

            elif len(branch_names) != self.nbranches:
                raise ValueError(
                    "Number of branches indicated by nleaves_max and branch_names are not equivalent (nleaves_max: {}, branch_names: {}).".format(
                        len(self.nleaves_max), len(branch_names)
                    )
                )

        else:
            # fill default names if not given
            branch_names = ["model_{}".format(i) for i in range(self.nbranches)]

        self.branch_names = branch_names

        self.iteration = 0

        # setup all the holder arrays
        self.accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)
        self.in_model_swaps_accepted = np.zeros((self.ntemps - 1,), dtype=self.dtype)
        if self.rj:
            self.rj_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)
            self.rj_swaps_accepted = np.zeros((self.ntemps - 1,), dtype=self.dtype)
        else:
            self.rj_accepted = None
            self.rj_swaps_accepted = None

        # chains are stored in dictionaries
        self.chain = {
            name: np.empty(
                (0, self.ntemps, self.nwalkers, nleaves, ndim), dtype=self.dtype
            )
            for name, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }

        # inds correspond to leaves used or not
        self.inds = {
            name: np.empty((0, self.ntemps, self.nwalkers, nleaves), dtype=bool)
            for name, nleaves in zip(self.branch_names, self.nleaves_max)
        }

        # log likelihood and prior
        self.log_prob = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)

        # temperature ladder
        self.betas = np.empty((0, self.ntemps), dtype=self.dtype)

        self.blobs = None

        self.random_state = None
        self.initialized = True

        # injection parameters
        self.truth = truth

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        return self.blobs is not None

    def get_value(self, name, flat=False, thin=1, discard=0):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            dict or np.ndarray: Values requested.

        """
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        # check against blobs
        if name == "blobs" and not self.has_blobs():
            return None

        # prepare chain for output
        if name == "chain":
            v_all = {
                key: self.chain[key][discard + thin - 1 : self.iteration : thin]
                for key in self.chain
            }
            if flat:
                v_out = {}
                for key, v in v_all.items():
                    s = list(v.shape[1:])
                    s[0] = np.prod(v.shape[:2])
                    v.reshape(s)
                    v_out[key] = v
                return v_out
            return v_all

        # prepare inds for output
        if name == "inds":
            v_all = {
                key: self.inds[key][discard + thin - 1 : self.iteration : thin]
                for key in self.chain
            }
            if flat:
                v_out = {}
                for key, v in v_all.items():
                    s = list(v.shape[1:])
                    s[0] = np.prod(v.shape[:2])
                    v.reshape(s)
                    v_out[key] = v
                return v_out
            return v_all

        # all other requests can filter through array output
        # rather than the dictionary output used above
        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            dict: MCMC samples
                The dictionary contains np.ndarrays of samples
                across the branches.

        """
        return self.get_value("chain", **kwargs)

    def get_autocorr_thin_burn(self):
        """Return the discard and thin values based on the autocorrelation length.

        The ``discard`` is determined as 2 times the maximum correlation length among parameters.
        The ``thin`` is determined using 1/2 times the minimum correlation legnth among parameters.

        Returns:
            tuple: Information on thin and burn
                (discard, thin)
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
        """Get the stored chain of MCMC samples

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            dict: The ``inds`` associated with the MCMC samples.
                  The dictionary contains np.ndarrays of ``inds``
                  across the branches indicated which leaves were used at each step.

        """
        return self.get_value("inds", **kwargs)

    def get_nleaves(self, **kwargs):
        """Get the number of leaves for each walker

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            dict: nleaves on each branch.
                The number of leaves on each branch associated with the MCMC samples
                  within each branch.

        """
        inds = self.get_value("inds", **kwargs)
        nleaves = {name: np.sum(inds[name], axis=-1, dtype=int) for name in inds}
        return nleaves

    def get_blobs(self, **kwargs):
        """Get the chain of blobs for each sample in the chain

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers, nblobs]: The chain of blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log likelihood values.

        """
        return self.get_value("log_prob", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log prior values.

        """
        return self.get_value("log_prior", **kwargs)

    def get_betas(self, **kwargs):
        """Get the chain of inverse temperatures

        Args:
            flat (bool, optional): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            double np.ndarray[nsteps, ntemps]: The chain of temperatures.


        """
        return self.get_value("betas", **kwargs)

    def get_a_sample(self, it):
        """Access a sample in the chain

        Returns:
            State: :class:`eryn.state.State` object containing the sample from the chain.

        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        thin = self.iteration - it if it != self.iteration else 1
        # check for blobs
        blobs = self.get_blobs(discard=it - 1, thin=thin)
        if blobs is not None:
            blobs = blobs[0]

        # fill a State with quantities from the last sample in the chain
        sample = State(
            {name: temp[0] for name, temp in self.get_chain(discard=it - 1, thin=thin).items()},
            log_prob=self.get_log_prob(discard=it - 1, thin=thin)[0],
            log_prior=self.get_log_prior(discard=it - 1, thin=thin)[0],
            inds={
                name: temp[0] for name, temp in self.get_inds(discard=it - 1, thin=thin).items()
            },
            blobs=blobs,
            random_state=self.random_state,
        )
        return sample

    def get_last_sample(self):
        """Access the most recent sample in the chain

        Returns:
            State: :class:`eryn.state.State` object containing the last sample from the chain.

        """
        it = self.iteration
        last_sample = self.get_a_sample(it)
        return last_sample

    def get_autocorr_time(
        self, discard=0, thin=1, all_temps=False, multiply_thin=True, **kwargs
    ):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            all_temps (bool, optional): If True, calculate autocorrelation across
                all temperatures. If False, calculate autocorrelation across the minumum
                temperature chain (usually 1/T=1). (default: ``False``)
            multiply_thin (bool, optional) If True, include the thinning factor
                into the autocorrelation length. (default: ``True``)


        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            dict: autocorrelation times
                The dictionary contains autocorrelation times for all parameters
                as 1D double np.ndarrays as values with associated ``branch_names`` as
                keys.

        """
        # stopping index into temperatures
        ind = self.ntemps if all_temps else 1

        # get chain
        x = self.get_chain(discard=discard, thin=thin)
        x = {name: value[:, :ind] for name, value in x.items()}

        out = get_integrated_act(x, **kwargs)

        # apply thinning factor if desired
        thin_factor = thin if multiply_thin else 1

        return {name: values * thin_factor for name, values in out.items()}

    def get_evidence_estimate(self, discard=0, thin=1, return_error=True):
        """Get an estimate of the evidence

        Args:
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            return_error (bool, optional): If True, return the error associated
                with the log evidence estimate. (default: ``True``)

        Returns:
            double or tuple: Evidence estimate
                If requesting the error on the estimate, will receive a tuple:
                ``(logZ, dlogZ)``. Otherwise, just a double value of logZ.

        """
        # TODO: check this
        # get all the likelihood and temperature values
        logls_all = self.get_log_prob(discard=discard, thin=thin)
        betas_all = self.get_betas(discard=discard, thin=thin)

        # make sure that the betas were fixed during sampling (after burn in)
        if not (betas_all == betas_all[0]).all():
            raise ValueError(
                "Cannot compute evidence estimation if betas are allowed to vary. Use stop_adaptation kwarg in temperature settings."
            )

        # setup information
        betas = betas_all[0]
        logls = np.mean(logls_all, axis=(0, -1))

        # get log evidence and error
        logZ, dlogZ = thermodynamic_integration_log_evidence(betas, logls)

        if return_error:
            return (logZ, dlogZ)
        else:
            return logZ

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            dict: Shape of samples
                Keys are ``branch_names`` and valeus are tuples with
                shapes of individual branches: (ntemps, nwalkers, nleaves_max, ndim).

        """
        return {
            key: (self.ntemps, self.nwalkers, nleaves, ndim)
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }

    def _check_blobs(self, blobs):
        # check if the setup for blobs is correct
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def _check_rj_accepted(self, rj_accepted):
        # check fi rj_accepted is setup properly
        if not self.rj and rj_accepted is not None:
            raise ValueError("inconsistent use of rj_accepted")
        if self.rj and rj_accepted is None:
            raise ValueError("inconsistent use of rj_accepted")

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        # determine the number of entries in the chains
        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)
        {
            key: (self.ntemps, self.nwalkers, nleaves, ndim)
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }

        # temperary addition to chains
        a = {
            key: np.empty(
                (i, self.ntemps, self.nwalkers, nleaves, ndim), dtype=self.dtype
            )
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }
        # combine with original chain
        self.chain = {
            key: np.concatenate((self.chain[key], a[key]), axis=0) for key in a
        }

        # temperorary addition to inds
        a = {
            key: np.empty((i, self.ntemps, self.nwalkers, nleaves), dtype=bool)
            for key, nleaves in zip(self.branch_names, self.nleaves_max)
        }
        # combine with original inds
        self.inds = {key: np.concatenate((self.inds[key], a[key]), axis=0) for key in a}

        # temperorary addition for log likelihood
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log likelihood
        self.log_prob = np.concatenate((self.log_prob, a), axis=0)

        # temperorary addition for log prior
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log prior
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)

        # temperorary addition for betas
        a = np.empty((i, self.ntemps), dtype=self.dtype)
        # combine with original betas
        self.betas = np.concatenate((self.betas, a), axis=0)

        if blobs is not None:
            dt = np.dtype((blobs.dtype, blobs.shape[2:]))
            # temperorary addition for blobs
            a = np.empty((i, self.ntemps, self.nwalkers), dtype=dt)
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
        in_model_swaps_accepted=None,
        rj_swaps_accepted=None,
    ):
        """Check all the information going in is okay."""
        self._check_blobs(state.blobs)
        self._check_rj_accepted(rj_accepted)

        shapes = self.shape
        has_blobs = self.has_blobs()

        ntemps, nwalkers = self.ntemps, self.nwalkers

        # make sure all of the coordinate and inds dimensions are okay
        for key, shape in shapes.items():
            ntemp1, nwalker1, nleaves1, ndim1 = state.branches[key].shape
            ntemp2, nwalker2, nleaves2, ndim2 = shape

            if (ntemp1, nwalker1, ndim1) != (
                ntemp2,
                nwalker2,
                ndim2,
            ) or nleaves1 > nleaves2:
                raise ValueError(
                    "invalid coordinate dimensions for model {1} with shape {2}; expected {0}".format(
                        shape, key, state.branches[key].shape
                    )
                )

            if (ntemp1, nwalker1, nleaves1) != state.branches[key].inds.shape:
                raise ValueError(
                    "invalid inds dimensions for model {1} with shape {2}; expected {0}".format(
                        (ntemp1, nwalker1, nleaves1),
                        key,
                        state.branches[key].inds.shape,
                    )
                )

        # make sure log likelihood, log prior, blobs, accepted, rj_accepted, betas
        if state.log_prob.shape != (ntemps, nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(ntemps, nwalkers)
            )
        if state.log_prior.shape != (ntemps, nwalkers,):
            raise ValueError(
                "invalid log prior size; expected {0}".format(ntemps, nwalkers)
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and state.blobs.shape[:2] != (ntemps, nwalkers):
            raise ValueError(
                "invalid blobs size; expected {0}".format(ntemps, nwalkers)
            )
        if accepted.shape != (ntemps, nwalkers,):
            raise ValueError(
                "invalid acceptance size; expected {0}".format(ntemps, nwalkers)
            )

        if in_model_swaps_accepted is not None and in_model_swaps_accepted.shape != (
            ntemps - 1,
        ):
            raise ValueError(
                "invalid in_model_swaps_accepted size; expected {0}".format(ntemps - 1)
            )
        if self.rj:
            if rj_accepted.shape != (ntemps, nwalkers,):
                raise ValueError(
                    "invalid rj acceptance size; expected {0}".format(ntemps, nwalkers)
                )
            if rj_swaps_accepted is not None and rj_swaps_accepted.shape != (
                ntemps - 1,
            ):
                raise ValueError(
                    "invalid rj_swaps_accepted size; expected {0}".format(ntemps - 1)
                )

        if state.betas is not None and state.betas.shape != (ntemps,):
            raise ValueError("invalid beta size; expected {0}".format(ntemps))

    def save_step(
        self,
        state,
        accepted,
        rj_accepted=None,
        in_model_swaps_accepted=None,
        rj_swaps_accepted=None,
    ):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            rj_accepted (ndarray, optional): An array of the number of accepted steps
                for the reversible jump proposal for each walker.
                If :code:`self.rj` is True, then rj_accepted must be an array with
                :code:`rj_accepted.shape == accepted.shape`. If :code:`self.rj`
                is False, then rj_accepted must be None, which is the default.

        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            rj_accepted=rj_accepted,
            in_model_swaps_accepted=in_model_swaps_accepted,
            rj_swaps_accepted=rj_swaps_accepted,
        )

        # save the coordinates and inds
        for key, model in state.branches.items():
            self.inds[key][self.iteration] = model.inds
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = model.coords * model.inds[:, :, :, None]

            inds_all = np.repeat(model.inds, model.coords.shape[-1], axis=-1).reshape(
                model.inds.shape + (model.coords.shape[-1],)
            )
            coords_in[~inds_all] = self.store_missing_leaves
            self.chain[key][self.iteration] = coords_in

        # save higher level quantities
        self.log_prob[self.iteration, :, :] = state.log_prob
        self.log_prior[self.iteration, :, :] = state.log_prior
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        if state.betas is not None:
            self.betas[self.iteration, :] = state.betas

        self.accepted += accepted

        if in_model_swaps_accepted is not None:
            self.in_model_swaps_accepted += in_model_swaps_accepted
        if self.rj:
            self.rj_accepted += rj_accepted
            if rj_swaps_accepted is not None:
                self.rj_swaps_accepted += rj_swaps_accepted

        self.random_state = state.random_state
        self.iteration += 1

    def get_info(self, discard=0, thin=1):
        """Get an output info dictionary

        This dictionary could be used for diagnostics or plotting.

        Args:
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)

        Returns:
            dict: Information for diagnostics
                Dictionary that contains much of the information for diagnostic
                checks or plotting.

        """
        samples = self.get_chain(discard=discard, thin=thin)

        # add all information that would be needed from backend
        out_info = dict(samples=samples)
        out_info["thin"] = thin
        out_info["burn"] = discard

        tau = self.get_autocorr_time()

        # get log prob
        out_info["log_prob"] = self.get_log_prob(thin=thin, discard=discard)

        # get temperatures
        out_info["betas"] = self.get_betas(thin=thin, discard=discard)

        # get inds
        out_info["inds"] = self.get_inds(thin=thin, discard=discard)

        # TODO: fix self.ntemps in hdf5 backend
        out_info["shapes"] = self.shape
        out_info["ntemps"] = self.ntemps
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
                "Failed to calculate the autocorrelation length. Will not output this piece of information. \n\n Actual error: [{}]".format(
                    e
                )
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
