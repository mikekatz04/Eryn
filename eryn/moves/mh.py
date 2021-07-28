# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["MHMove"]


class MHMove(Move):
    r"""A general Metropolis-Hastings proposal

    Concrete implementations can be made by providing a ``get_proposal`` method.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be used.

    """

    def __init__(self, ndim=None, **kwargs):

        super(MHMove, self).__init__(**kwargs)
        self.ndim = ndim
        # TODO: check ndim stuff

    def get_proposal(self, sample, complement, random):
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
        # if self.ndim is not None and self.ndim != ndim:
        #    raise ValueError("Dimension mismatch in proposal")

        # Get the move-specific proposal.
        q, factors = self.get_proposal(
            state.branches_coords, state.branches_inds, model.random
        )

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(q, inds=state.branches_inds)
        # Compute the lnprobs of the proposed position.
        logl, new_blobs = model.compute_log_prob_fn(
            q, inds=state.branches_inds, logp=logp
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
            q, log_prob=logl, log_prior=logp, blobs=new_blobs, inds=state.branches_inds
        )
        state = self.update(state, new_state, accepted)

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
