# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from ..state import State
from .move import Move

__all__ = ["MHMove"]


class DelayedRejection(Move):
    r"""A general Metropolis-Hastings proposal

    Concrete implementations can be made by providing a ``get_proposal`` method.
    For standard Gaussian Metropolis moves, :class:`moves.GaussianMove` can be used.

    """

    def __init__(self, proposal, max_iter=10, **kwargs):
        self.proposal = proposal
        self.max_iter = max_iter

        super(DelayedRejection, self).__init__(**kwargs)

    
    @property
    def temperature_control(self):
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        self.proposal.temperature_control = temperature_control
        self._temperature_control = temperature_control
        if temperature_control is None:
            self.compute_log_posterior = self.compute_log_posterior_basic
        else:
            self.compute_log_posterior = (
                self.temperature_control.compute_log_posterior_tempered
            )

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

        # self.lnpdiff = factors + logP - prev_logP
        current_state = deepcopy(state)

        # Get the old coords and posterior values
        prev_logl = state.log_prob
        prev_logp = state.log_prior
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp) # takes care of tempering

        # Initialize TODO: Maybe a smarter way to do it?
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        rejind   = ~accepted # Get the rejected points
        logalpha = np.full((ntemps, nwalkers), -np.inf) 
        prev_logalpha = np.full((ntemps, nwalkers), -np.inf) 
        prev_factors  = np.ones((ntemps, nwalkers))
        nom_logalpha  = np.zeros((ntemps, nwalkers))
        dnom_logalpha = np.zeros((ntemps, nwalkers))
        driter = 0

        # Start DR loop. Check if all accepted (extreme case). Then stop.
        # TODO: As currently implemented, it proposes and computes the model for all temps and walkers
        # TODO: We should try to reduce the computation to those only rejected. Probably through state.branches_inds
        while (driter < self.max_iter) or (np.sum(accepted) == np.prod(accepted.shape)):
            
            print(" - Iter {}: Accepted = {}/{}".format(driter, np.sum(accepted), np.prod(accepted.shape)))             

            # Set betas to 0 in order to propose a new point for all walkers across all temps
            self.proposal.temperature_control.betas = np.zeros_like(state.betas)
            state, _ = self.proposal.propose(model, state) # Propose for all walkers and temps, get posterior and prior
            logl     = state.log_prob
            logp     = state.log_prior
            factors  = self.proposal.factors.copy() # copy the factors

            # Set the temps back to the original values, get tempered posterior
            self.proposal.temperature_control.betas = current_state.betas.copy()

            # Compute the logposterior for all
            logP = self.compute_log_posterior(logl, logp)
            
            # Adjust for previous acceptance rates
            nom_logalpha[rejind]  += np.log(1 - np.exp(logalpha[rejind]))
            dnom_logalpha[rejind] += np.log(1 - np.exp(prev_logalpha[rejind]))

            # Calculate the delayed rejection acceptance ratio
            logalpha[rejind] = factors[rejind] - prev_factors[rejind] \
                               + logP[rejind] - prev_logP[rejind] \
                               + nom_logalpha[rejind]  - dnom_logalpha[rejind] 

            accepted[rejind] = logalpha[rejind] > np.log(model.random.rand(ntemps, nwalkers)[rejind])
            
            rejind = ~accepted # Update the indeces of rejected points
            
            # Switch for remaining rejected cases
            prev_logalpha[rejind] = logalpha[rejind]
            prev_factors[rejind] = factors[rejind]
            prev_logl[rejind] = logl[rejind]
            prev_logp[rejind] = logp[rejind]

            # Update the parameters, update the state. TODO: Fix blobs?
            new_state = State(
                state.branches_coords, log_prob=prev_logl, log_prior=prev_logp, blobs=state.blobs, inds=state.branches_inds
            )
            state = self.update(state, new_state, accepted)

            driter += 1 # Increase iteration

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
