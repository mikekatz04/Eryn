# -*- coding: utf-8 -*-

import numpy as np

from ..utils.utility import (
    get_integrated_act,
    thermodynamic_integration_log_evidence,
    stepping_stone_log_evidence,
    psrf,
)
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
        initiailized (bool): If ``True``, backend object has been initialized.
        iteration (int): Current index within the data storage arrays.
        log_prior (3D double np.ndarray): Log of the prior values. Shape is
            (nsteps, nwalkers, ntemps).
        log_like (3D double np.ndarray): Log of the likelihood values. Shape is
            (nsteps, nwalkers, ntemps).
        move_info (dict): Dictionary containing move info.
        move_keys (list): List of keys for ``move_info``.
        nbranches (int): Number of branches.
        ndims (dict): Dimensionality of each branch.
        nleaves_max (dict): Maximum allowable leaves for each branch.
        nwalkers (int): The size of the ensemble (per temperature).
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
        branch_names=None,
        nbranches=1,
        rj=False,
        moves=None,
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble (per temperature).
            ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
            nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
                It should have the same length as the number of branches.
                If ``dict``, keys should be the branch names and values the associated maximal leaf value.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            nbranches (int, optional): Number of branches. This is only used if ``branch_names is None``.
                (default: ``1``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            moves (list, optional): List of all of the move classes input into the sampler.
                (default: ``None``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend.

        """
        # store inputs for later resets
        self.reset_args = (nwalkers, ndims)
        self.reset_kwargs = dict(
            nleaves_max=nleaves_max,
            ntemps=ntemps,
            branch_names=branch_names,
            rj=rj,
            moves=moves,
            info=info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.rj = rj

        # turn things into lists/dicts if needed
        if branch_names is not None:
            if isinstance(branch_names, str):
                branch_names = [branch_names]

            elif not isinstance(branch_names, list):
                raise ValueError("branch_names must be string or list of strings.")

        else:
            branch_names = ["model_{}".format(i) for i in range(nbranches)]

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

        self.iteration = 0

        # setup all the holder arrays
        self.accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)
        self.swaps_accepted = np.zeros((self.ntemps - 1,), dtype=self.dtype)
        if self.rj:
            self.rj_accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)

        else:
            self.rj_accepted = None

        # chains are stored in dictionaries
        self.chain = {
            name: np.empty(
                (
                    0,
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
                (0, self.ntemps, self.nwalkers, self.nleaves_max[name]), dtype=bool
            )
            for name in self.branch_names
        }

        # log likelihood and prior
        self.log_like = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)

        # temperature ladder
        self.betas = np.empty((0, self.ntemps), dtype=self.dtype)

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
                self.move_info[move] = {
                    "acceptance_fraction": np.zeros(
                        (self.ntemps, self.nwalkers), dtype=self.dtype
                    )
                }

                # update the move keys to keep proper order
                self.move_keys.append(move)

        else:
            self.move_info = None

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        return self.blobs is not None

    def get_value(self, name, thin=1, discard=0, slice_vals=None, temp_index=None, branch_names=None):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): Ignored for non-HDFBackend.
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)
            branch_names (str or list, optional): Specific branch names requested. (default: ``None``)
            
        Returns:
            dict or np.ndarray: Values requested.

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

        if temp_index is None:
            temp_index = np.arange(self.ntemps)
        else:
            assert isinstance(temp_index, int)
        
        # make sure branch_names input is a list
        if branch_names is not None:
            if isinstance(branch_names, str):
                branches_names = [branch_names]

        branch_names_in = self.branch_names if branch_names is None else branch_names
        # prepare chain for output
        if name == "chain":
            v_all = {
                key: self.chain[key][discard + thin - 1 : self.iteration : thin, temp_index]
                for key in branch_names_in
            }
            return v_all

        # prepare inds for output
        if name == "inds":
            v_all = {
                key: self.inds[key][discard + thin - 1 : self.iteration : thin, temp_index]
                for key in branch_names_in
            }
            return v_all

        # all other requests can filter through array output
        # rather than the dictionary output used above
        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin, temp_index]
        return v

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

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
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            dict: The ``inds`` associated with the MCMC samples.
                  The dictionary contains np.ndarrays of ``inds``
                  across the branches indicated which leaves were used at each step.

        """
        return self.get_value("inds", **kwargs)

    def get_nleaves(self, **kwargs):
        """Get the number of leaves for each walker

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

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
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers, nblobs]: The chain of blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_log_like(self, **kwargs):
        """Get the chain of log Likelihood values evaluated at the MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log likelihood values.

        """
        return self.get_value("log_like", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the chain of log Prior evaluated at the MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log prior values.

        """
        return self.get_value("log_prior", **kwargs)

    def get_log_posterior(self, temper: bool = False, **kwargs):
        """Get the chain of log posterior values evaluated at the MCMC samples

        Args:
            temper (bool, optional): Apply tempering to the posterior values.
                (default: ``False``)
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps, nwalkers]: The chain of log prior values.

        """
        if temper:
            betas = self.get_betas(**kwargs)

        else:
            betas = np.ones_like(self.get_betas(**kwargs))

        log_like = self.get_log_like(**kwargs)
        log_prior = self.get_log_prior(**kwargs)

        return betas[:, :, None] * log_like + log_prior

    def get_betas(self, **kwargs):
        """Get the chain of inverse temperatures

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)

        Returns:
            double np.ndarray[nsteps, ntemps]: The chain of temperatures.


        """
        return self.get_value("betas", **kwargs)

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of State to return.

        Returns:
            State: :class:`eryn.state.State` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

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

        # fill a State with quantities from the last sample in the chain
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
            betas=self.get_betas(discard=discard, thin=thin).squeeze(),
            blobs=blobs,
            random_state=self.random_state,
        )
        return sample

    def get_last_sample(self):
        """Access the most recent sample in the chain

        Returns:
            State: :class:`eryn.state.State` object containing the last sample from the chain.

        """
        it = self.iteration - 1

        # get the state from the last iteration
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
                temperature chain (usually ``T=1``). (default: ``False``)
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

        if self.ntemps > 1 or self.rj:
            raise ValueError(
                "get_autocorr_time is not well-defined for number of temperatures > 1 or when using reversible jump."
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
        """Get an estimate of the evidence

        This function gets the sample information and uses
        :func:`thermodynamic_integration_log_evidence` or
        :func:`stepping_stone_log_evidence` to compute the evidence estimate.

        Args:
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            return_error (bool, optional): If True, return the error associated
                with the log evidence estimate. (default: ``True``)
            method (string, optional): Method to compute the evidence. Available
                methods are the 'thermodynamic' and 'stepping-stone' (default: ``thermodynamic``)

        Returns:
            double or tuple: Evidence estimate
                If requesting the error on the estimate, will receive a tuple:
                ``(logZ, dlogZ)``. Otherwise, just a double value of logZ.

        """
        # get all the likelihood and temperature values
        logls_all = self.get_log_like(discard=discard, thin=thin)
        betas_all = self.get_betas(discard=discard, thin=thin)

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
            logls = np.mean(logls_all, axis=(0, -1))
            logZ, dlogZ = thermodynamic_integration_log_evidence(betas, logls)
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
        """
        The Gelman - Rubin convergence diagnostic.
        A general approach to monitoring convergence of MCMC output of multiple walkers.
        The function makes a comparison of within-chain and between-chain variances.
        A large deviation between these two variances indicates non-convergence, and
        the output [Rhat] deviates from unity.

        Based on
        a. Brooks, SP. and Gelman, A. (1998) General methods for monitoring convergence
        of iterative simulations. Journal of Computational and Graphical Statistics, 7, 434-455
        b. Gelman, A and Rubin, DB (1992) Inference from iterative simulation using multiple sequences,
        Statistical Science, 7, 457-511.

        Args:
            C (np.ndarray[nwalkers, nsamples, ndim]): The parameter traces. The MCMC chains.
            doprint (bool, optional): Flag to print the results on screen.
        discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        thin (int, optional): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
        doprint (bool, optional): Flag to print a table with the results, per temperature.

        Returns
            dict:   ``Rhat_all_branches``:
                Returns an estimate of the Gelman-Rubin convergence diagnostic ``Rhat``,
                per temperature, stored in a dictionary, per branch name.

        """
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
                print(" Model: {}".format(branch))
                print("   T \t <R̂>")
                print("  --------------")
                for temp in range(self.ntemps):
                    print(
                        "   {:01d}\t{:3.2f}".format(
                            temp, np.mean(Rhat_all_branches[branch][temp])
                        )
                    )
                print("\n")

        return Rhat_all_branches

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            dict: Shape of samples
                Keys are ``branch_names`` and values are tuples with
                shapes of individual branches: (ntemps, nwalkers, nleaves_max, ndim).

        """
        return {
            key: (self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key])
            for key in self.branch_names
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
            key: (self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key])
            for key in self.branch_names
        }

        # temperary addition to chains
        a = {
            key: np.empty(
                (i, self.ntemps, self.nwalkers, self.nleaves_max[key], self.ndims[key]),
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
                (i, self.ntemps, self.nwalkers, self.nleaves_max[key]), dtype=bool
            )
            for key in self.branch_names
        }
        # combine with original inds
        self.inds = {key: np.concatenate((self.inds[key], a[key]), axis=0) for key in a}

        # temperorary addition for log likelihood
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log likelihood
        self.log_like = np.concatenate((self.log_like, a), axis=0)

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
        swaps_accepted=None,
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

        # make sure log likelihood, log prior, blobs, accepted, rj_accepted, betas are okay
        if state.log_like.shape != (
            ntemps,
            nwalkers,
        ):
            raise ValueError(
                "invalid log probability size; expected {0}".format((ntemps, nwalkers))
            )
        if state.log_prior.shape != (
            ntemps,
            nwalkers,
        ):
            raise ValueError(
                "invalid log prior size; expected {0}".format((ntemps, nwalkers))
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and state.blobs.shape[:2] != (ntemps, nwalkers):
            raise ValueError(
                "invalid blobs size; expected {0}".format((ntemps, nwalkers))
            )
        if accepted.shape != (
            ntemps,
            nwalkers,
        ):
            raise ValueError(
                "invalid acceptance size; expected {0}".format((ntemps, nwalkers))
            )

        if swaps_accepted is not None and swaps_accepted.shape != (ntemps - 1,):
            raise ValueError(
                "invalid swaps_accepted size; expected {0}".format(ntemps - 1)
            )
        if self.rj:
            if rj_accepted.shape != (
                ntemps,
                nwalkers,
            ):
                raise ValueError(
                    "invalid rj acceptance size; expected {0}".format(
                        (ntemps, nwalkers)
                    )
                )

        if state.betas is not None and state.betas.shape != (ntemps,):
            raise ValueError("invalid beta size; expected {0}".format(ntemps))

    def get_move_info(self):
        """Get move information.

        Returns:
            dict: Keys are move names and values are dictionaries with information on the moves.

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
            swaps_accepted (ndarray, optional): 1D array with number of swaps accepted
                for the in-model step. (default: ``None``)
            moves_accepted_fraction (dict, optional): Dict of acceptance fraction arrays for all of the
                moves in the sampler. This dict must have the same keys as ``self.move_keys``.
                (default: ``None``)

        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            rj_accepted=rj_accepted,
            swaps_accepted=swaps_accepted,
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
        self.log_like[self.iteration, :, :] = state.log_like
        self.log_prior[self.iteration, :, :] = state.log_prior
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        if state.betas is not None:
            self.betas[self.iteration, :] = state.betas

        self.accepted += accepted

        if swaps_accepted is not None:
            self.swaps_accepted += swaps_accepted
        if self.rj:
            self.rj_accepted += rj_accepted

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
        out_info["log_like"] = self.get_log_like(thin=thin, discard=discard)

        # get temperatures
        out_info["betas"] = self.get_betas(thin=thin, discard=discard)

        # get inds
        out_info["inds"] = self.get_inds(thin=thin, discard=discard)

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
