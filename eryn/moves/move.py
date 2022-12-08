# -*- coding: utf-8 -*-

from ..state import BranchSupplimental
import numpy as np

try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    import numpy as xp

__all__ = ["Move"]


class Move(object):
    """Parent class for proposals or "moves"

    Args:
        temperature_control (:class:`tempering.TemperatureControl`, optional):
            This object controls the tempering. It is passed to the parent class
            to moves so that all proposals can share and use temperature settings.
            (default: ``None``)
        periodic (:class:`eryn.utils.PeriodicContainer, optional):
            This object holds periodic information and methods for periodic parameters. It is passed to the parent class
            to moves so that all proposals can share and use periodic information.
            (default: ``None``)
        proposal_branch_names (list or str, optional): Branch names to run with this move class.
        prevent_swaps (bool, optional): If ``True``, do not perform temperature swaps in this move.
        skip_supp_names_update (list, optional): List of names (`str`), that can be in any :class:`eryn.state.BranchSupplimental`,
            to skip when updating states (:func:`Move.update`). This is useful if a large amount of memory is stored
            in the branch supplimentals.

    Raises:
        ValueError: Incorrect inputs.

    Attributes:
        Note: All kwargs are stored as attributes.
        num_proposals (int): the number of times this move has been run. This is needed to 
            compute the acceptance fraction.

    """

    def __init__(
        self,
        temperature_control=None,
        periodic=None,
        proposal_branch_names=None,
        prevent_swaps=False,
        skip_supp_names_update=[],
    ):
        # store all information
        self.temperature_control = temperature_control
        self.periodic = periodic
        self.skip_supp_names_update = skip_supp_names_update
        self.prevent_swaps = prevent_swaps
        self.proposal_branch_names = proposal_branch_names

        # setup proposal branches properly
        if self.proposal_branch_names is not None:
            if isinstance(self.proposal_branch_names, str):
                self.proposal_branch_names = [self.proposal_branch_names]
            elif not isinstance(self.proposal_branch_names, list):
                raise ValueError("proposal_branch_names must be string or list of str.")

        self.num_proposals = 0

    @property
    def accepted(self):
        """Accepted counts for this move."""
        if self._accepted is None:
            raise ValueError(
                "accepted must be inititalized with the init_accepted function if you want to use it."
            )
        return self._accepted

    @accepted.setter
    def accepted(self, accepted):
        assert isinstance(accepted, np.ndarray)
        self._accepted = accepted

    @property
    def acceptance_fraction(self):
        """Acceptance fraction for this move."""
        return self.accepted / self.num_proposals

    @property
    def temperature_control(self):
        """Temperature controller"""
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        self._temperature_control = temperature_control

        # use the setting of the temperature control to determine which log posterior function to use
        # tempered or basic
        if temperature_control is None:
            self.compute_log_posterior = self.compute_log_posterior_basic
        else:
            self.compute_log_posterior = (
                self.temperature_control.compute_log_posterior_tempered
            )

            self.ntemps = self.temperature_control.ntemps

    def compute_log_posterior_basic(self, logl, logp):
        """Compute the log of posterior

        :math:`\log{P} = \log{L} + \log{p}`

        This method is to mesh with the tempered log posterior computation.

        Args:
            logl (np.ndarray[ntemps, nwalkers]): Log-likelihood values.
            logp (np.ndarray[ntemps, nwalkers]): Log-prior values.

        Returns:
            np.ndarray[ntemps, nwalkers]: Log-Posterior values.
        """
        return logl + logp

    def tune(self, state, accepted):
        """Tune a proposal

        This is a place holder for tuning.

        Args:
            state (:class:`eryn.state.State`): Current state of sampler.
            accepted (np.ndarray[ntemps, nwalkers]): Accepted values for last pass
                through proposal.

        """
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        This class was updated from ``emcee`` to handle the added structure
        of Eryn.

        Args:
            old_state (:class:`eryn.state.State`): State with current information.
                New information is added to this state.
            new_state (:class:`eryn.state.State`): State with information from proposed
                points.
            accepted (np.ndarray[ntemps, nwalkers]): A vector of booleans indicating
                which walkers were accepted.
            subset (np.ndarray[ntemps, nwalkers], optional): A boolean mask
                indicating which walkers were included in the subset.
                This can be used, for example, when updating only the
                primary ensemble in a :class:`RedBlueMove`.
                (default: ``None``)

        Returns:
            :class:`eryn.state.State`: ``old_state`` with accepted points added from ``new_state``.

        """
        if subset is None:
            # subset of everything
            subset = np.tile(
                np.arange(old_state.log_like.shape[1]), (old_state.log_like.shape[0], 1)
            )

        # take_along_axis is necessary to do this all in higher dimensions
        accepted_temp = np.take_along_axis(accepted, subset, axis=1)

        # new log likelihood
        old_log_likes = np.take_along_axis(old_state.log_like, subset, axis=1)
        new_log_likes = new_state.log_like
        temp_change_log_like = new_log_likes * (accepted_temp) + old_log_likes * (
            ~accepted_temp
        )

        np.put_along_axis(old_state.log_like, subset, temp_change_log_like, axis=1)

        # new log prior
        old_log_priors = np.take_along_axis(old_state.log_prior, subset, axis=1)
        new_log_priors = new_state.log_prior.copy()

        # deal with -infs
        new_log_priors[np.isinf(new_log_priors)] = 0.0

        temp_change_log_prior = new_log_priors * (accepted_temp) + old_log_priors * (
            ~accepted_temp
        )

        np.put_along_axis(old_state.log_prior, subset, temp_change_log_prior, axis=1)

        # inds
        old_inds = {
            name: np.take_along_axis(branch.inds, subset[:, :, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_inds = {name: branch.inds for name, branch in new_state.branches.items()}

        temp_change_inds = {
            name: new_inds[name] * (accepted_temp[:, :, None])
            + old_inds[name] * (~accepted_temp[:, :, None])
            for name in old_inds
        }

        [
            np.put_along_axis(
                old_state.branches[name].inds,
                subset[:, :, None],
                temp_change_inds[name],
                axis=1,
            )
            for name in new_inds
        ]

        # check for branches_supplimental
        run_branches_supplimental = False
        for name, value in old_state.branches_supplimental.items():
            if value is not None:
                run_branches_supplimental = True

        if run_branches_supplimental:
            # branch_supplimental
            temp_change_branch_supplimental = {}
            for name in old_state.branches:
                if old_state.branches[name].branch_supplimental is not None:
                    old_branch_supplimental = old_state.branches[
                        name
                    ].branch_supplimental.take_along_axis(
                        subset[:, :, None],
                        axis=1,
                        skip_names=self.skip_supp_names_update,
                    )
                    new_branch_supplimental = new_state.branches[
                        name
                    ].branch_supplimental[:]

                    tmp = {}
                    for key in old_branch_supplimental:
                        if key in self.skip_supp_names_update:
                            continue
                        accepted_temp_here = accepted_temp.copy()
                        if new_branch_supplimental[key].dtype.name != "object":
                            for _ in range(
                                new_branch_supplimental[key].ndim
                                - accepted_temp_here.ndim
                            ):
                                accepted_temp_here = np.expand_dims(
                                    accepted_temp_here, (-1,)
                                )

                        try:
                            tmp[key] = new_branch_supplimental[key] * (
                                accepted_temp_here
                            ) + old_branch_supplimental[key] * (~accepted_temp_here)
                        except TypeError:
                            # for gpus
                            tmp[key] = new_branch_supplimental[key] * (
                                xp.asarray(accepted_temp_here)
                            ) + old_branch_supplimental[key] * (
                                xp.asarray(~accepted_temp_here)
                            )

                    temp_change_branch_supplimental[name] = BranchSupplimental(
                        tmp,
                        obj_contained_shape=new_state.branches_supplimental[name].shape,
                        copy=True,
                    )

                else:
                    temp_change_branch_supplimental[name] = None

            [
                old_state.branches[name].branch_supplimental.put_along_axis(
                    subset[:, :, None],
                    temp_change_branch_supplimental[name][:],
                    axis=1,
                )
                for name in new_inds
                if temp_change_branch_supplimental[name] is not None
            ]

        # sampler level supplimental
        if old_state.supplimental is not None:

            old_suppliment = old_state.supplimental.take_along_axis(subset, axis=1)
            new_suppliment = new_state.supplimental[:]

            accepted_temp_here = accepted_temp.copy()

            temp_change_suppliment = {}
            for name in old_suppliment:
                if name in self.skip_supp_names_update:
                    continue
                if old_suppliment[name].dtype.name != "object":
                    for _ in range(old_suppliment[name].ndim - accepted_temp_here.ndim):
                        accepted_temp_here = np.expand_dims(accepted_temp_here, (-1,))
                try:
                    temp_change_suppliment[name] = new_suppliment[name] * (
                        accepted_temp_here
                    ) + old_suppliment[name] * (~accepted_temp_here)
                except TypeError:
                    temp_change_suppliment[name] = new_suppliment[name] * (
                        xp.asarray(accepted_temp_here)
                    ) + old_suppliment[name] * (xp.asarray(~accepted_temp_here))
            old_state.supplimental.put_along_axis(
                subset, temp_change_suppliment, axis=1
            )

        # coords
        old_coords = {
            name: np.take_along_axis(branch.coords, subset[:, :, None, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_coords = {
            name: branch.coords for name, branch in new_state.branches.items()
        }

        temp_change_coords = {
            name: new_coords[name] * (accepted_temp[:, :, None, None])
            + old_coords[name] * (~accepted_temp[:, :, None, None])
            for name in old_coords
        }

        [
            np.put_along_axis(
                old_state.branches[name].coords,
                subset[:, :, None, None],
                temp_change_coords[name],
                axis=1,
            )
            for name in new_coords
        ]

        # take care of blobs
        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_like, "
                    "you also need to provide the current list of "
                    "blobs at that position."
                )

            old_blobs = np.take_along_axis(old_state.blobs, subset[:, :, None], axis=1)
            new_blobs = new_state.blobs
            temp_change_blobs = new_blobs * (accepted_temp[:, :, None]) + old_blobs * (
                ~accepted_temp[:, :, None]
            )

            np.put_along_axis(
                old_state.blobs, subset[:, :, None], temp_change_blobs, axis=1
            )

        return old_state

