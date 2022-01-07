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

        super(MHMove, self).__init__(**kwargs)
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
        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        # Get the move-specific proposal.
        q, factors = self.get_proposal(
            state.branches_coords, state.branches_inds, model.random
        )

        # setup supplimental information
        if state.branches_supplimental is not None:
            new_branch_supps = deepcopy(state.branches_supplimental)
        else:
            new_branch_supps = None

        if state.supplimental is not None:
            # TODO: should there be a copy?
            new_supps = deepcopy(state.supplimental)
        else:   
            new_supps = None

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(q, inds=state.branches_inds)
        # Compute the lnprobs of the proposed position.
        # Can adjust supplimentals in place
        logl, new_blobs = model.compute_log_prob_fn(
            q, inds=state.branches_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps
        )

        logP = self.compute_log_posterior(logl, logp)

        prev_logl = state.log_prob

        prev_logp = state.log_prior

        # TODO: check about prior = - inf
        # takes care of tempering
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

        lnpdiff = factors + logP - prev_logP

        accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

        # Update the parameters
        new_state = State(
            q, log_prob=logl, log_prior=logp, blobs=new_blobs, inds=state.branches_inds, supplimental=new_supps, branch_supplimental=new_branch_supps
        )
        state = self.update(state, new_state, accepted)

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
