# -*- coding: utf-8 -*-
"""Deprecated in-memory storage for the deprecated ``ParaEnsembleSampler``.

:class:`ParaBackend` stores ``ngroups`` independent single-branch ensembles with arrays of
shape ``(nsteps, ngroups, ntemps, nwalkers, ...)``. It is superseded by
:class:`~eryn.backends.Backend` with ``reset(..., nsamplers=...)``, whose storage always
carries the independent-sampler axis.
"""

import warnings

import numpy as np

from ..state import ParaState
from ..utils.utility import (
    psrf,
    stepping_stone_log_evidence,
    thermodynamic_integration_log_evidence,
)

__all__ = ["ParaBackend"]


class ParaBackend(object):
    """In-memory storage for :class:`~eryn.paraensemble.ParaEnsembleSampler` (deprecated).

    Stores ``ngroups`` independent single-branch ensembles. Coordinates of groups that are
    not running at a step are stored as ``store_missing_leaves``. Construction emits a
    ``DeprecationWarning``; use :class:`~eryn.backends.Backend` with
    ``reset(..., nsamplers=...)`` instead.

    Args:
        store_missing_leaves (float, optional): Value stored for the coordinates of groups
            that are not running. Default ``np.nan``.
        dtype (numpy.dtype, optional): Floating dtype of the stored arrays. ``None`` means
            ``np.float64``. Default ``None``.

    Attributes:
        initialized (bool): ``True`` once :meth:`reset` has run.
        iteration (int): Number of steps saved so far.
        chain (dict): ``{branch_name: array}`` of shape
            ``(nsteps, ngroups, ntemps, nwalkers, ndim)``.
        groups_running (dict): ``{branch_name: array}`` of booleans, shape
            ``(nsteps, ngroups)``.
        log_like (np.ndarray): Log-likelihood, shape ``(nsteps, ngroups, ntemps, nwalkers)``.
        log_prior (np.ndarray): Log-prior, shape ``(nsteps, ngroups, ntemps, nwalkers)``.
        betas (np.ndarray): Inverse temperatures, shape ``(nsteps, ngroups, ntemps)``.
        accepted (np.ndarray): Cumulative acceptance counts, shape
            ``(ngroups, ntemps, nwalkers)``.
        swaps_accepted (np.ndarray): Cumulative accepted swaps, shape
            ``(ngroups, ntemps - 1)``.
        random_state (object): Random state of the last saved step.
        branch_name (str): Name of the single branch.
        ndim (int): Parameters per walker.
        nwalkers (int): Walkers per temperature.
        ntemps (int): Temperatures per group.
        ngroups (int): Number of independent groups.
        reset_args (tuple): ``(ndim, nwalkers, ngroups)`` of the last :meth:`reset`.
        reset_kwargs (dict): Keyword arguments of the last :meth:`reset`.
        dtype (numpy.dtype): Storage dtype.
        store_missing_leaves (float): Value stored for non-running groups.
    """

    def __init__(self, store_missing_leaves=np.nan, dtype=None):
        warnings.warn(
            "ParaBackend is deprecated; use eryn.backends.Backend with "
            "reset(..., nsamplers=...) instead (storage always carries the "
            "sampler axis). It will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    def reset_base(self):
        """Call :meth:`reset` again with the arguments of the previous call."""
        self.reset(*self.reset_args, **self.reset_kwargs)

    def reset(
        self,
        ndim: int,
        nwalkers: int,
        ngroups: int,
        ntemps: int = 1,
        branch_name: str = "model_0",
        **info,
    ):
        """Clear the backend and allocate empty storage for ``ngroups`` ensembles.

        Args:
            ndim: Parameters per walker.
            nwalkers: Walkers per temperature.
            ngroups: Number of independent groups.
            ntemps: Temperatures per group. Default ``1``.
            branch_name: Name of the single branch. Default ``"model_0"``.
            **info: Extra key/value pairs set as attributes on the backend.
        """
        # store inputs for later resets
        self.reset_args = (ndim, nwalkers, ngroups)
        self.reset_kwargs = dict(
            ntemps=ntemps,
            branch_name=branch_name,
            **info,
        )

        # load info into class
        for key, value in info.items():
            setattr(self, key, value)

        # store all information to guide data storage
        self.nwalkers = int(nwalkers)
        self.ntemps = int(ntemps)
        self.ngroups = int(ngroups)

        self.branch_name = branch_name
        self.ndim = ndim

        self.iteration = 0

        # setup all the holder arrays
        self.accepted = np.zeros((self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.swaps_accepted = np.zeros(
            (
                self.ngroups,
                self.ntemps - 1,
            ),
            dtype=self.dtype,
        )

        # chains are stored in dictionaries
        self.chain = {
            self.branch_name: np.empty(
                (0, self.ngroups, self.ntemps, self.nwalkers, self.ndim),
                dtype=self.dtype,
            )
        }

        # inds correspond to leaves used or not
        self.groups_running = {self.branch_name: np.empty((0, self.ngroups), dtype=bool)}

        # log likelihood and prior
        self.log_like = np.empty((0, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        self.log_prior = np.empty((0, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)

        # temperature ladder
        self.betas = np.empty((0, self.ngroups, self.ntemps), dtype=self.dtype)

        self.random_state = None
        self.initialized = True

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Return one stored quantity, keeping the steps ``discard + thin - 1, ...``.

        Args:
            name (str): ``"chain"``, ``"groups_running"``, ``"log_like"``,
                ``"log_prior"``, ``"betas"`` or another stored array attribute.
            thin (int, optional): Keep every ``thin``-th step. Default ``1``.
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            slice_vals (optional): Not supported; must be ``None``. Default ``None``.

        Returns:
            np.ndarray: The selected steps (``chain`` and ``groups_running`` are returned
            for the single branch, not as a dict).

        Raises:
            ValueError: ``slice_vals`` is given.
            AttributeError: Nothing has been stored yet.
        """
        if slice_vals is not None:
            raise ValueError("slice_vals can only be used with an HDF Backend.")

        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )

        # prepare chain for output
        if name == "chain":
            v_all = self.chain[self.branch_name][discard + thin - 1 : self.iteration : thin]

            return v_all

        # prepare inds for output
        if name == "groups_running":
            v_all = self.groups_running[self.branch_name][
                discard + thin - 1 : self.iteration : thin
            ]

            return v_all

        # all other requests can filter through array output
        # rather than the dictionary output used above
        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        return v

    def get_chain(self, **kwargs):
        """Get the stored coordinates.

        Args:
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Coordinates of shape ``(nsteps, ngroups, ntemps, nwalkers, ndim)``
            (an array, not a dict).
        """
        return self.get_value("chain", **kwargs)

    def get_groups_running(self, **kwargs):
        """Get the stored running-group masks.

        Args:
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Booleans of shape ``(nsteps, ngroups)``.
        """
        return self.get_value("groups_running", **kwargs)

    def get_log_like(self, **kwargs):
        """Get the stored log-likelihood.

        Args:
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Log-likelihood of shape ``(nsteps, ngroups, ntemps, nwalkers)``.
        """
        return self.get_value("log_like", **kwargs)

    def get_log_prior(self, **kwargs):
        """Get the stored log-prior.

        Args:
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Log-prior of shape ``(nsteps, ngroups, ntemps, nwalkers)``.
        """
        return self.get_value("log_prior", **kwargs)

    # TODO check: keep/remove/document -- betas[:, :, None] misaligned if ntemps > 1; no callers
    def get_log_posterior(self, temper: bool = False, **kwargs):
        """Get the log-posterior ``beta * log_like + log_prior`` of every stored walker.

        As written, the ``(nsteps, ngroups, ntemps)`` betas are expanded as
        ``betas[:, :, None]``, which only lines up with the
        ``(nsteps, ngroups, ntemps, nwalkers)`` log-likelihood when ``ntemps == 1``.

        Args:
            temper: If ``True``, weight the log-likelihood by the stored inverse
                temperatures; otherwise use ``beta = 1``. Default ``False``.
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Log-posterior, intended shape ``(nsteps, ngroups, ntemps, nwalkers)``.
        """
        if temper:
            betas = self.get_betas(**kwargs)

        else:
            betas = np.ones_like(self.get_betas(**kwargs))

        log_like = self.get_log_like(**kwargs)
        log_prior = self.get_log_prior(**kwargs)

        return betas[:, :, None] * log_like + log_prior

    def get_betas(self, **kwargs):
        """Get the stored inverse temperatures.

        Args:
            **kwargs: ``thin`` and ``discard``, forwarded to :meth:`get_value`.

        Returns:
            np.ndarray: Inverse temperatures of shape ``(nsteps, ngroups, ntemps)``.
        """
        return self.get_value("betas", **kwargs)

    def get_a_sample(self, it):
        """Rebuild the :class:`~eryn.state.ParaState` of one stored step.

        Args:
            it (int): Stored step index, ``0 <= it < iteration``.

        Returns:
            ParaState: Coordinates, ``log_like``, ``log_prior``, ``groups_running``,
            ``betas`` of step ``it`` and the backend's latest ``random_state``.

        Raises:
            AttributeError: The backend is not initialized or nothing has been stored.
        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        # fill a ParaState with quantities from sample ``it`` in the chain;
        # each getter returns one step, so take index 0 to drop the step axis
        sample = ParaState(
            {self.branch_name: self.get_chain(discard=discard, thin=thin)[0]},
            log_like=self.get_log_like(discard=discard, thin=thin)[0],
            log_prior=self.get_log_prior(discard=discard, thin=thin)[0],
            groups_running=self.get_groups_running(discard=discard, thin=thin)[0],
            betas=self.get_betas(discard=discard, thin=thin)[0],
            random_state=self.random_state,
        )
        return sample

    def get_last_sample(self):
        """Rebuild the :class:`~eryn.state.ParaState` of the last stored step.

        Returns:
            ParaState: Equivalent to ``get_a_sample(iteration - 1)``.
        """
        it = self.iteration - 1

        # get the state from the last iteration
        last_sample = self.get_a_sample(it)
        return last_sample

    def get_evidence_estimate(
        self, discard=0, thin=1, return_error=True, method="therodynamic", **ss_kwargs
    ):
        """Estimate the log-evidence from the tempered chains (same methods as ``Backend``).

        The per-temperature inputs here keep the group axis: ``betas`` of shape
        ``(ngroups, ntemps)`` and mean log-likelihoods of shape ``(ngroups, ntemps)``,
        while :func:`~eryn.utils.utility.thermodynamic_integration_log_evidence` and
        :func:`~eryn.utils.utility.stepping_stone_log_evidence` document 1D
        ``(ntemps,)`` inputs; results for ``ngroups > 1`` are not verified.

        Args:
            discard (int, optional): Number of initial steps to drop. Default ``0``.
            thin (int, optional): Use every ``thin``-th step. Default ``1``.
            return_error (bool, optional): If ``True``, also return the error estimate.
                Default ``True``.
            method (str, optional): ``"therodynamic"`` (default; also
                ``"thermodynamic integration"``, ``"thermo"``, ``"ti"``) or
                ``"stepping-stone"`` (also ``"stepping stone"``, ``"ss"``, ``"step"``,
                ``"stone"``); case-insensitive. Default ``"therodynamic"``.
            **ss_kwargs: Forwarded to
                :func:`~eryn.utils.utility.stepping_stone_log_evidence`.

        Returns:
            float or tuple: ``logZ``, or ``(logZ, dlogZ)`` when ``return_error``.

        Raises:
            ValueError: The inverse temperatures vary over the selected steps, or an unknown
                ``method``.
        """
        # get all the likelihood and temperature values
        logls_all = self.get_log_like(discard=discard, thin=thin)
        betas_all = self.get_betas(discard=discard, thin=thin)

        # make sure that the betas were fixed during sampling (after burn in)
        if not (betas_all == betas_all[0]).all():
            raise ValueError(
                "Cannot compute evidence estimation if betas are allowed to vary. "
                "Use stop_adaptation kwarg in temperature settings."
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

    # TODO check: keep/remove/document -- broken: needs branch_names/ndims/get_inds; no callers
    def get_gelman_rubin_convergence_diagnostic(
        self, discard=0, thin=1, doprint=True, **psrf_kwargs
    ):
        """Gelman-Rubin diagnostic copied from :class:`~eryn.backends.Backend` (not working).

        It reads ``self.branch_names``, ``self.ndims`` and ``self.get_inds``, which
        :class:`ParaBackend` does not define, and indexes :meth:`get_chain` (an array) by
        branch name, so it raises before computing anything.
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
                    chains_in = chains.squeeze().transpose((1, 0, 2))
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
                    print(f"   {temp:01d}\t{np.mean(Rhat_all_branches[branch][temp]):3.2f}")
                print("\n")

        return Rhat_all_branches

    @property
    def shape(self):
        """tuple: Per-step coordinate shape ``(ngroups, ntemps, nwalkers, ndim)``."""
        return (self.ngroups, self.ntemps, self.nwalkers, self.ndim)

    def grow(self, ngrow, blobs):
        """Make room for ``ngrow`` more steps after the current :attr:`iteration`.

        Args:
            ngrow (int): Number of steps that will be saved next.
            blobs (object): Ignored (blobs are not stored).
        """

        # determine the number of entries in the chains
        i = ngrow - (len(self.chain[list(self.chain.keys())[0]]) - self.iteration)

        # temperary addition to chains
        a = {
            self.branch_name: np.empty(
                (i, self.ngroups, self.ntemps, self.nwalkers, self.ndim),
                dtype=self.dtype,
            )
        }
        # combine with original chain
        self.chain = {
            self.branch_name: np.concatenate(
                (self.chain[self.branch_name], a[self.branch_name]), axis=0
            )
        }

        # temperorary addition to groups_running
        a = {self.branch_name: np.empty((i, self.ngroups), dtype=bool)}
        # combine with original groups_running
        self.groups_running = {
            self.branch_name: np.concatenate(
                (self.groups_running[self.branch_name], a[self.branch_name]), axis=0
            )
        }

        # temperorary addition for log likelihood
        a = np.empty((i, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log likelihood
        self.log_like = np.concatenate((self.log_like, a), axis=0)

        # temperorary addition for log prior
        a = np.empty((i, self.ngroups, self.ntemps, self.nwalkers), dtype=self.dtype)
        # combine with original log prior
        self.log_prior = np.concatenate((self.log_prior, a), axis=0)

        # temperorary addition for betas
        a = np.empty((i, self.ngroups, self.ntemps), dtype=self.dtype)
        # combine with original betas
        self.betas = np.concatenate((self.betas, a), axis=0)

    def _check(
        self,
        state,
        accepted,
        swaps_accepted=None,
    ):
        """Raise ``ValueError`` if the state or acceptance arrays have the wrong shapes."""

        shape = self.shape

        ngroups, ntemps, nwalkers = self.ngroups, self.ntemps, self.nwalkers

        # make sure all of the coordinate and inds dimensions are okay
        ngroup1, ntemp1, nwalker1, ndim1 = state.branches[self.branch_name].shape
        ngroup2, ntemp2, nwalker2, ndim2 = shape

        if (ngroup1, ntemp1, nwalker1, ndim1) != (
            ngroup2,
            ntemp2,
            nwalker2,
            ndim2,
        ):
            raise ValueError(
                f"invalid coordinate dimensions for model {self.branch_name} with shape {state.branches[self.branch_name].shape}; expected {shape}"
            )

        if (ngroup1,) != state.groups_running.shape:
            raise ValueError(
                f"invalid inds dimensions for model {self.branch_name} with shape {state.groups_running.shape}; expected {(ngroup1,)}"
            )

        # make sure log likelihood, log prior, blobs, accepted, rj_accepted, betas are okay
        if state.log_like.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(
                f"invalid log probability size; expected {(ngroups, ntemps, nwalkers)}"
            )
        if state.log_prior.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(f"invalid log prior size; expected {(ngroups, ntemps, nwalkers)}")

        if accepted.shape != (
            ngroups,
            ntemps,
            nwalkers,
        ):
            raise ValueError(f"invalid acceptance size; expected {(ngroups, ntemps, nwalkers)}")

        if swaps_accepted is not None and swaps_accepted.shape != (
            ngroups,
            ntemps - 1,
        ):
            raise ValueError(f"invalid swaps_accepted size; expected {(ngroups, ntemps - 1)}")

        if state.betas is not None and state.betas.shape != (
            ngroups,
            ntemps,
        ):
            raise ValueError(f"invalid beta size; expected {(ngroups, ntemps)}")

    def save_step(
        self,
        state,
        accepted,
        swaps_accepted=None,
    ):
        """Write one step at index :attr:`iteration` and advance it.

        ``cupy`` inputs are copied to the host (``.get()``); ``numpy`` inputs are stored
        directly. Coordinates of groups that are not running are replaced by
        ``store_missing_leaves``.

        Args:
            state (ParaState): State to save; arrays of shape
                ``(ngroups, ntemps, nwalkers, ...)`` and ``groups_running`` of shape
                ``(ngroups,)``.
            accepted (np.ndarray or cupy.ndarray): Acceptance counts of this step, shape
                ``(ngroups, ntemps, nwalkers)``.
            swaps_accepted (np.ndarray or cupy.ndarray, optional): Accepted swaps, shape
                ``(ngroups, ntemps - 1)``. Default ``None``.

        Raises:
            ValueError: A shape mismatch.
        """
        # check to make sure all information in the state is okay
        self._check(
            state,
            accepted,
            swaps_accepted=swaps_accepted,
        )

        try:
            # save the coordinates and groups_running
            self.groups_running[self.branch_name][self.iteration] = state.groups_running.get()
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = (
                state.branches[self.branch_name].coords * state.groups_running[:, None, None, None]
            ).get()

            coords_in[~state.groups_running.get()] = self.store_missing_leaves
            self.chain[self.branch_name][self.iteration] = coords_in

            # save higher level quantities
            self.log_like[self.iteration, :, :] = state.log_like.get()
            self.log_prior[self.iteration, :, :] = state.log_prior.get()

            if state.betas is not None:
                self.betas[self.iteration, :] = state.betas.get()

            self.accepted += accepted.get()

            if swaps_accepted is not None:
                self.swaps_accepted += swaps_accepted.get()

        except AttributeError:
            # save the coordinates and groups_running
            self.groups_running[self.branch_name][self.iteration] = state.groups_running
            # use self.store_missing_leaves to set value for missing leaves
            # state retains old coordinates
            coords_in = (
                state.branches[self.branch_name].coords * state.groups_running[:, None, None, None]
            )

            coords_in[~state.groups_running] = self.store_missing_leaves
            self.chain[self.branch_name][self.iteration] = coords_in

            # save higher level quantities
            self.log_like[self.iteration, :, :] = state.log_like
            self.log_prior[self.iteration, :, :] = state.log_prior

            if state.betas is not None:
                self.betas[self.iteration, :] = state.betas

            self.accepted += accepted

            if swaps_accepted is not None:
                self.swaps_accepted += swaps_accepted

        self.random_state = state.random_state
        self.iteration += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
