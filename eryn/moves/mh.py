# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from ..state import State
from .move import Move

__all__ = ["MHMove"]


class MHMove(Move):
    r"""A general Metropolis-Hastings proposal

    Concrete implementations can be made by providing a ``get_proposal`` method.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be used.

    """

    def __init__(self, **kwargs):

        Move.__init__(self, **kwargs)
        # TODO: check ndim stuff

    def get_proposal(self, branches_coords, branches_inds, random):
        """Get proposal from distribution for MH proposal

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            branches_inds (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max] representing which
                leaves are currently being used.
            random (object): Current random state object.

        Raises:
            NotImplementedError: If proposal is not implemented in a subclass.

        """

        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """

        # get all branch names for gibbs setup
        all_branch_names = list(state.branches.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # iterate through gibbs setup
        for (branch_names_run, inds_run) in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):

            # setup supplimental information
            if not np.all(
                np.asarray(list(state.branches_supplimental.values())) == None
            ):
                new_branch_supps = deepcopy(state.branches_supplimental)
            else:
                new_branch_supps = None

            if state.supplimental is not None:
                new_supps = deepcopy(state.supplimental)
            else:
                new_supps = None

            # setup information according to gibbs info
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            # if no walkers are actually being proposed
            if not at_least_one_proposal:
                continue

            # Get the move-specific proposal.
            q, factors = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                branches_inds=inds_going_for_proposal,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # Compute prior of the proposed position
            logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

            self.fix_logp_gibbs(branch_names_run, inds_run, logp, state.branches_inds)

            # Compute the lnprobs of the proposed position.
            # Can adjust supplimentals in place
            logl, new_blobs = model.compute_log_like_fn(
                q,
                inds=state.branches_inds,
                logp=logp,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # get log posterior
            logP = self.compute_log_posterior(logl, logp)

            # get previous information
            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # difference
            lnpdiff = factors + logP - prev_logP

            # draw against acceptance fraction
            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # Update the parameters
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=new_blobs,
                inds=state.branches_inds,
                supplimental=new_supps,
                branch_supplimental=new_branch_supps,
            )
            state = self.update(state, new_state, accepted)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted
