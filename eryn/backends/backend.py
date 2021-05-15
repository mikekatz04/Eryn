# -*- coding: utf-8 -*-

import numpy as np

# from .. import autocorr
from ..state import State

__all__ = ["Backend"]


class Backend(object):
    """A simple default backend that stores the chain in memory"""

    def __init__(self, dtype=None):
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

    def reset(
        self, nwalkers, ndims, nleaves_max=1, ntemps=1, truth=None, branch_names=None
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        self.nwalkers = int(nwalkers)  # trees
        self.ntemps = int(ntemps)

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
            branch_names = ["model_{}".format(i) for i in range(self.nbranches)]

        self.branch_names = branch_names

        self.iteration = 0
        self.accepted = np.zeros((self.ntemps, self.nwalkers), dtype=self.dtype)

        self.chain = {
            name: np.empty(
                (0, self.ntemps, self.nwalkers, nleaves, ndim), dtype=self.dtype
            )
            for name, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }
        self.log_prob = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.inds = {
            name: np.empty((0, self.ntemps, self.nwalkers, nleaves), dtype=self.dtype)
            for name, nleaves in zip(self.branch_names, self.nleaves_max)
        }
        self.blobs = None
        self.betas = np.empty((0, self.ntemps), dtype=self.dtype)
        self.random_state = None
        self.initialized = True
        self.truth = truth

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        return self.blobs is not None

    def get_value(self, name, flat=False, thin=1, discard=0):
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if name == "blobs" and not self.has_blobs():
            return None

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

        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("chain", **kwargs)

    def get_inds(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        return self.get_value("inds", **kwargs)

    def get_nleaves(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.

        """
        inds = self.get_value("inds", **kwargs)
        nleaves = {name: np.sum(inds[name], axis=-1, dtype=int) for name in inds}
        return nleaves

    def get_blobs(self, **kwargs):
        """Get the chain of blobs for each sample in the chain

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return self.get_value("log_prob", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return self.get_value("log_prior", **kwargs)

    def get_betas(self, **kwargs):
        """ TODO: this

        TODO: make betas optional

        Get the chain of log probabilities evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log probabilities.

        """
        return self.get_value("betas", **kwargs)

    def get_last_sample(self):
        """Access the most recent sample in the chain"""
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        it = self.iteration
        blobs = self.get_blobs(discard=it - 1)
        if blobs is not None:
            blobs = blobs[0]
        return State(
            self.get_chain(discard=it - 1)[0],
            log_prob=self.get_log_prob(discard=it - 1)[0],
            log_prior=self.get_log_prior(discard=it - 1)[0],
            inds=self.get_inds(discard=it - 1)[0],
            blobs=blobs,
            random_state=self.random_state,
        )

    def get_autocorr_time(self, discard=0, thin=1, all_temps=False, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        ind = self.ntemps if all_temps else 1
        x = self.get_chain(discard=discard, thin=thin)[:, :ind]

        # TODO: fix this
        return thin * autocorr.integrated_time(x, **kwargs)

    @property
    def shape(self):
        """The dimensions of the ensemble ``(ntemps, nwalkers, ndim)``"""
        return {
            key: (self.ntemps, self.nwalkers, nleaves, ndim)
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)
        {
            key: (self.ntemps, self.nwalkers, nleaves, ndim)
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }
        a = {
            key: np.empty(
                (i, self.ntemps, self.nwalkers, nleaves, ndim), dtype=self.dtype
            )
            for key, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            )
        }
        self.chain = {
            key: np.concatenate((self.chain[key], a[key]), axis=0) for key in a
        }
        a = {
            key: np.empty((i, self.ntemps, self.nwalkers, nleaves), dtype=self.dtype)
            for key, nleaves in zip(self.branch_names, self.nleaves_max)
        }
        self.inds = {key: np.concatenate((self.inds[key], a[key]), axis=0) for key in a}
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prob = np.concatenate((self.log_prob, a), axis=0)
        a = np.empty((i, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)
        a = np.empty((i, self.ntemps), dtype=self.dtype)
        self.betas = np.concatenate((self.betas, a), axis=0)

        if blobs is not None:
            dt = np.dtype((blobs.dtype, blobs.shape[2:]))
            a = np.empty((i, self.ntemps, self.nwalkers), dtype=dt)
            if self.blobs is None:
                self.blobs = a
            else:
                self.blobs = np.concatenate((self.blobs, a), axis=0)

    def _check(self, state, accepted):
        self._check_blobs(state.blobs)
        shapes = self.shape
        has_blobs = self.has_blobs()

        ntemps, nwalkers = self.ntemps, self.nwalkers
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

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        for key, model in state.branches.items():
            self.inds[key][self.iteration] = model.inds
            # TODO: how do we want to store final info
            # right now is leaves zeros in unsed leaves
            # state retains old coordinates
            coords_in = model.coords * model.inds[:, :, :, None]
            self.chain[key][self.iteration] = coords_in

        self.log_prob[self.iteration, :, :] = state.log_prob
        self.log_prior[self.iteration, :, :] = state.log_prior
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        if state.betas is not None:
            self.betas[self.iteration, :] = state.betas

        self.accepted += accepted
        self.random_state = state.random_state
        self.iteration += 1

    def get_info(self, burn=None, thin=None):
        samples = self.get_chain()
        tau = None
        if burn is None or thin is None:
            # TODO setup autocorr
            raise NotImplementedError
            if self.thin_chain_by_ac:
                tau = self.get_autocorr_time(tol=0)
                if burn is None:
                    burn = int(2 * np.max(tau))

                if thin is None:
                    thin = int(0.5 * np.min(tau))
            else:
                if burn is None:
                    burn = 0

                if thin is None:
                    thin = 1

        if thin == 0:
            thin = 1

        if burn is None:
            burn = int(2 * np.max(tau))

        if thin is None:
            thin = int(0.5 * np.min(tau))

        samples = {name: samples[name][burn::thin] for name in samples}

        # add all information that would be needed from backend
        out_info = dict(samples=samples)
        out_info["thin"] = thin
        out_info["burn"] = burn

        # get log prob
        out_info["log_prob"] = self.get_log_prob(thin=thin, discard=burn)

        # get temperatures
        out_info["betas"] = self.get_betas(thin=thin, discard=burn)

        # get inds
        out_info["inds"] = self.get_inds(thin=thin, discard=burn)

        out_info["shapes"] = self.shape
        out_info["ntemps"] = self.ntemps
        out_info["nwalkers"] = self.nwalkers
        out_info["nbranches"] = self.nbranches
        out_info["branch names"] = self.branch_names
        out_info["ndims"] = self.ndims
        out_info["tau"] = tau

        return out_info

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
