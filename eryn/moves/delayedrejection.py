# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from ..state import State
from .move import Move

__all__ = ["MHMove"]


class DelayedRejection(Move):
    r"""
    Delayed Rejection

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

    def calculate_log_acceptance_ratio(self, stateslist, prev_logP, model, rj_inds):
        """Calcuate the delayed rejection acceptace ratio. 

        Args: 
            stateslist (:class:`State`): a python list containing the proposed states
        
        Returns:
            logalpha: a numpy array containing the acceptance ratios per temperature and walker.
        """
        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = stateslist[0].branches[list(stateslist[0].branches.keys())[0]].shape

        driter = len(stateslist) - 1  # The stage we're in, elements in trypath - 1
        
        loga1 = 0.0  # Init
        loga2 = 0.0  

        # recursively compute past alphas
        for kk in range(0, driter - 1):
            prevla1,_,_ = self.calculate_log_acceptance_ratio(stateslist[0:(kk + 2)], prev_logP, model, rj_inds)
            prevla1[prevla1>=0] = 0.0 # Ensure we do not get NaNs # TODO: Check if correct
            loga1 = loga1 + np.log(1 - np.exp(prevla1))
            prevla2,_,_ = self.calculate_log_acceptance_ratio(stateslist[driter:driter - kk - 2:-1], prev_logP, model, rj_inds)
            prevla2[prevla2>=0] = 0.0
            loga2 = loga2 + np.log(1 - np.exp(prevla2))
            
            if np.all(loga2 == 1.0): 
                logalpha = np.ones((ntemps, nwalkers))
                return logalpha, stateslist[0].log_prob, stateslist[0].log_prior # TODO: check if index 0 is the correct

        logp = model.compute_log_prior_fn(stateslist[-1].branches_coords, inds=rj_inds) # inds=stateslist[-1].branches_inds
        logl, _ = model.compute_log_prob_fn(
                    stateslist[-1].branches_coords, inds=rj_inds, logp=logp # compute logp and fill -inf to accepted (-inf points are skipped)
        )

        # Compute the logposterior for all
        logP = self.compute_log_posterior(logl, logp)

        logprop_density_ratio = 0.0
        for kk in range(driter):
            logprop_density_ratio += self.get_log_proposal_ratio_for_iter(kk, stateslist)

        logalpha = logP - prev_logP + loga2 - loga1 + logprop_density_ratio

        return logalpha, logl, logp

    # TODO: This assumes a Gaussian proposal. We need to think how to make it work with the factors?
    def get_log_proposal_ratio_for_iter(self, iq, statespath):
        """
        Gaussian nth stage log proposal ratio. Since it's symmetric, after the third
        iteration, it depends only on the first, i-th and (i-1)th points.
        """
        stage = len(statespath) - 1 - 1  # - 1, i
        zq    = 0.0  # Not too deep into the iterations = symmetric
        if stage != iq:
            # breakpoint()
            for model in statespath[0].branches_coords:
                x1, x2, x3, x4 = self.get_state_coords(iq, stage, statespath, model)
                invCmat = self.proposal.all_proposal[model].invscale
                # TODO: Check if this does it right across dimensions. A: It does not. I need to think about it more
                zq += -0.5*((np.linalg.norm(np.dot(x4-x3, invCmat)))**2 - (np.linalg.norm(np.dot(x2-x1, invCmat)))**2)
                # zq = 0.0
        return zq

    def get_state_coords(self, iq, stage, statespath, m):
        """
        Extract coordinates from states for the given path.
        """
        x1 = statespath[0].branches_coords[m]
        x2 = statespath[iq + 1].branches_coords[m]  
        x3 = statespath[stage + 1].branches_coords[m]
        x4 = statespath[stage - iq].branches_coords[m]
        return x1, x2, x3, x4

    def get_new_state(self, model, state, first_state, rj_inds):
        """ A utility function to propose new points
        """
        self.proposal.temperature_control.betas = np.zeros_like(state.betas)
        new_state, _ = self.proposal.propose(model, state) # Propose for all walkers and temps, get posterior and prior
        logl = new_state.log_prob
        logp = new_state.log_prior

        # Set the temps back to the original values, get tempered posterior
        self.proposal.temperature_control.betas = first_state.betas.copy()

        # Update the parameters, update the state. TODO: Fix blobs?
        new_state = State(
            state.branches_coords, log_prob=logl, log_prior=logp, blobs=state.blobs, inds=rj_inds
        ) # I create a new initial state that all are accepted
        return new_state

    def propose(self, accepted, model, state, rj_inds, factors):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            accepted ():
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.
            rj_inds (): Dictionary containing the indices where the Reversible Jump
            move proposed "birth" of a model. Will we operate a Delayed Rejection type
            of move only on those cases. The keys of the dictionary are the names of the 
            models.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """

        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        initial_state = deepcopy(state)     # Get the current state and save it

        # Get the old coords and posterior values
        prev_logl = state.log_prob
        prev_logp = state.log_prior
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp) # takes care of tempering

        # Initialize TODO: Maybe a smarter way to do it?
        rejind   = ~accepted # Get the rejected points
        logalpha = np.full((ntemps, nwalkers), -np.inf) 
        driter   = 1 

        new_state = self.get_new_state(model, state, initial_state, rj_inds)

        states_path = [initial_state, new_state]  # This is the path of states that we need to track
        loga1 = np.zeros((ntemps, nwalkers, self.max_iter), dtype=bool) # Placeholder to keep the forward calculations -> speed it up a little
        
        # Start DR loop. Check if all accepted (extreme case). Then stop.
        # TODO: As currently implemented, it proposes and computes the model for all temps and walkers
        # TODO: We should try to reduce the computation to those only rejected. Probably through state.branches_inds
        while (driter < self.max_iter) and not (np.sum(accepted) == np.prod(accepted.shape)):         
            
            logalpha, logl, logp = self.calculate_log_acceptance_ratio(states_path, prev_logP, model, rj_inds) 

            accepted[rejind] = logalpha[rejind] > np.log(model.random.rand(ntemps, nwalkers)[rejind])

            rejind = ~accepted # Update the indeces of rejected points (by excluding newly accepted ones)
            
            # Switch for remaining rejected cases
            prev_logl[rejind] = logl[rejind]
            prev_logp[rejind] = logp[rejind]

            # Update the parameters, update the state. TODO: Fix blobs?
            update_state = State(
                state.branches_coords, log_prob=prev_logl, log_prior=prev_logp, blobs=state.blobs, inds=state.branches_inds
            )
            state = self.update(state, update_state, accepted)

            new_state = self.get_new_state(model, state, initial_state, rj_inds)

            states_path.append(new_state) # Add this new state to the path
            
            # print(" - Iter {}: Accepted = {}/{}".format(driter, np.sum(accepted), np.prod(accepted.shape)))    
            driter += 1 # Increase iteration
        
        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
