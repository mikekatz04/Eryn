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

    def dr_scheme(self, state, states_path, alpha_0, accepted, prev_logP, model, ntemps, nwalkers, inds=None, dr_iter=0):
        """Calcuate the delayed rejection acceptace ratio. 

        Args: 
            stateslist (:class:`State`): a python list containing the proposed states
        
        Returns:
            logalpha: a numpy array containing the acceptance ratios per temperature and walker.
        """

        # Get the chains that were rejected
        rejected = ~accepted # Get the rejected points

        # Draw a uniform random for the previously rejected points
        randU = model.random.rand(ntemps, nwalkers) # [rejected]

        # Propose a new point 
        new_state = self.get_new_state(model, state, inds) # Get a new state

        # Compute log-likelihood and log-prior
        logp = model.compute_log_prior_fn(new_state.branches_coords, inds=inds) # inds=stateslist[-1].branches_inds
        logl, _ = model.compute_log_prob_fn(new_state.branches_coords, inds=inds, logp=logp) # compute logp and fill -inf to accepted (-inf points are skipped)

        # Compute the logposterior for all
        logP = self.compute_log_posterior(logl, logp)

        # Placeholder for asymmetric proposals
        logproposal_density_ratio = 0.0

        # Compute the acceptance ratio
        lndiff  = logP - prev_logP - logproposal_density_ratio
        alpha_1 = np.exp(lndiff)
        alpha_1[alpha_1 > 1.0] = 1.0 # np.min((1, alpha))

        # update delayed rejection alpha
        dr_alpha = np.exp( lndiff + np.log(1.0 - alpha_1) - np.log(1.0 - alpha_0) )
        dr_alpha[dr_alpha > 1.0] = 1.0 # np.min((1., dr_alpha ))

        dr_alpha = np.nan_to_num(dr_alpha) # Automatically reject NaNs

        new_accepted = deepcopy(accepted)

        # Compute the acceptance probability for the rejected points only
        new_accepted[rejected] = np.logical_or(dr_alpha >= 1.0, randU < dr_alpha)[rejected]

        # print(" - Iter {}: Accepted = {}/{}".format(dr_iter, np.sum(new_accepted[rejected]), np.prod(new_accepted.shape)-np.sum(accepted))) 

        # Update state with the new accepted points
        new_state = self.update(state, new_state, new_accepted)
        breakpoint()
        # Create a new state where we keep the rejected points in order to perform DR
        # TODO: Check that when I propose a new state I do it only for the rejected +1 proposals
        new_accepted_flipped = deepcopy(new_accepted)
        new_accepted_flipped[rejected] = ~new_accepted_flipped[rejected]
        updated_state_keep_rejected = self.update(state, new_state, new_accepted_flipped)
    
        # Continue with theDR loop. Check if all accepted (extreme case). Then stop.
        if dr_iter <= self.max_iter and not (np.sum(new_accepted) == np.prod(new_accepted.shape)):
            dr_iter += 1
            states_path.append(updated_state_keep_rejected)
            new_state, new_accepted = self.dr_scheme(state, states_path, dr_alpha, new_accepted, \
                                                    prev_logP, model, ntemps, nwalkers, dr_iter=dr_iter, inds=inds)

        return new_state, new_accepted


    def get_new_state(self, model, state, inds_to_use):
        """ A utility function to propose new points
        """
        # TODO: Isntead of state.branches inds I was using rejected. Is it necessary to calculate all?
        if inds_to_use is None:
            inds_to_use = state.branches_inds

        qn, inds, _ = self.proposal.get_proposal(state.branches_coords, inds_to_use)

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(qn, inds=inds_to_use)
        # Compute the lnprobs of the proposed position. 
        logl, new_blobs = model.compute_log_prob_fn(qn, inds=inds_to_use, logp=logp)

        # Update the parameters, update the state. TODO: Fix blobs? 
        new_state = State(
            qn, log_prob=logl, log_prior=logp, blobs=new_blobs, inds=inds_to_use
        ) # I create a new initial state that all are accepted
        return new_state

    def propose(self, log_diff_0, accepted, model, state, plus_one_rj_inds, factors):
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

        # Get the old coords and posterior values
        prev_logl = state.log_prob
        prev_logp = state.log_prior
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp) # takes care of tempering
        
        # Initialize 
        alpha_0  = np.exp(log_diff_0)
        alpha_0[alpha_0 > 1.0] = 1.0 # np.min((1.0, diff_0)) 

        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        # Create a given path
        states_path = [state]

        # print(" - Before DR: Accepted = {}/{}".format(np.sum(accepted), np.prod(accepted.shape))) 

        out_state, accepted = self.dr_scheme(state, states_path, alpha_0, accepted, prev_logP, model, ntemps, nwalkers, inds=plus_one_rj_inds)       
        
        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return out_state, accepted
