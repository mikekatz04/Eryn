# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["MHMove"]


class MHMove(Move):
    r"""A general Metropolis-Hastings proposal

    Concrete implementations can be made by providing a ``proposal_function``
    argument that implements the proposal as described below.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be
    used.

    Args:
        proposal_function: The proposal function. It should take 2 arguments: a
            numpy-compatible random number generator and a ``(K, ndim)`` list
            of coordinate vectors. This function should return the proposed
            position and the log-ratio of the proposal probabilities
            (:math:`\ln q(x;\,x^\prime) - \ln q(x^\prime;\,x)` where
            :math:`x^\prime` is the proposed coordinate).
        ndim (Optional[int]): If this proposal is only valid for a specific
            dimension of parameter space, set that here.

    """

    def __init__(self, ndim=None, **kwargs):

        super(MHMove, self).__init__(**kwargs)
        self.ndim = ndim

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

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
        logl, new_blobs = model.compute_log_prob_fn(q, inds=state.branches_inds)

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
