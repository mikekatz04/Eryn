# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from ..state import State
from .move import Move
from ..state import BranchSupplimental

__all__ = ["MHMove"]

class DelayedRejectionContainer:
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)

        
        self.coords = []
        self.log_prob = []
        self.log_prior = []
        self.alpha = []

    def append(self, new_coords, new_log_prob, new_log_prior, new_alpha):
        self.coords.append(new_coords)
        self.log_prob.append(new_log_prob)
        self.log_prior.append(new_log_prior)
        self.alpha.append(new_alpha)


class DelayedRejection(Move):
    r"""
    Delayed Rejection

    """

    def __init__(self, proposal, max_iter=10, **kwargs):
        self.proposal = proposal
        self.max_iter = max_iter
        self.dr_container = None
        super(DelayedRejection, self).__init__(**kwargs)

    def dr_scheme(self, state, new_state, keep_rejected, model, ntemps, nwalkers, inds_for_change, inds=None, dr_iter=0):
        """Calcuate the delayed rejection acceptace ratio. 

        Args: 
            stateslist (:class:`State`): a python list containing the proposed states
        
        Returns:
            logalpha: a numpy array containing the acceptance ratios per temperature and walker.
        """

        # Find inds that satisfy rejected & at least one tree has proposed +1. TODO: Make it more efficient?
        
        # plus_one_inds['gauss']['+1'][:,:2]
        # accepted[(la[:,0],la[:,1])] --- accepted: la[accepted[(la[:,0],la[:,1])]]
        # Draw a uniform random for the previously rejected points
        randU = model.random.rand(ntemps, nwalkers) # [rejected]

        old_new_state = State(new_state, copy=True)

        # Propose a new point 
        new_state = self.get_new_state(model, new_state, keep_rejected) # Get a new state

        # Compute log-likelihood and log-prior
        logp = new_state.log_prior
        logl = new_state.log_prob

        # Compute the logposterior for all
        logP = self.compute_log_posterior(logl, logp)

        # Compute log-likelihood and log-prior
        prev_logp = old_new_state.log_prior
        prev_logl = old_new_state.log_prob

        # Compute the logposterior for all
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

        # Placeholder for asymmetric proposals
        logproposal_density_ratio = 0.0

        # Compute the acceptance ratio
        lndiff  = logP - prev_logP - logproposal_density_ratio
        alpha_1 = np.exp(lndiff)
        alpha_1[alpha_1 > 1.0] = 1.0 # np.min((1, alpha))

        # update delayed rejection alpha
        dr_alpha = np.exp(lndiff + np.log(1.0 - alpha_1) - np.log(1.0 - old_new_state.supplimental[:]["past_alpha"]) )
        dr_alpha[dr_alpha > 1.0] = 1.0 # np.min((1., dr_alpha ))
        dr_alpha = np.nan_to_num(dr_alpha) # Automatically reject NaNs

        new_state.supplimental[:] = {"alpha": dr_alpha}

        new_accepted = np.logical_or(dr_alpha >= 1.0, randU < dr_alpha)

        #new_accepted = deepcopy(accepted)
        # Compute the acceptance probability for the rejected points only
        # new_accepted[rejected] = np.logical_or(dr_alpha >= 1.0, randU < dr_alpha)[rejected]
        #for name in inds_for_change:
        #    new_accepted[plus_one_rej_inds[name][:,0],plus_one_rej_inds[name][:,1]] = \
        #        [plus_one_rej_inds[name][:,0],plus_one_rej_inds[name][:,1]]
        

        # Update state with the new accepted points
        state = self.update(state, new_state, new_accepted)
    
        # Create a new state where we keep the rejected points in order to perform DR
        # TODO: Check that when I propose a new state I do it only for the rejected +1 proposals
        #new_accepted_flipped = deepcopy(new_accepted)
        #new_accepted_flipped[rejected] = ~new_accepted_flipped[rejected]
        #updated_state_keep_rejected = self.update(state, new_state, new_accepted_flipped)
    
        #print(" - Iter {}: Accepted = {}/{}".format(dr_iter, np.sum(new_accepted[rejected]), np.prod(new_accepted.shape)-np.sum(accepted))) 


        # Continue with theDR loop. Check if all accepted (extreme case). Then stop.
        #if dr_iter <= self.max_iter and not (np.sum(new_accepted) == np.prod(new_accepted.shape)):
        #    dr_iter += 1
        #    states_path.append(updated_state_keep_rejected)
        #    new_state, new_accepted = self.dr_scheme(state, states_path, dr_alpha, new_accepted, \
        #                                            prev_logP, model, ntemps, nwalkers, dr_iter=dr_iter, inds=inds)

        return state, new_accepted, new_state


    def get_new_state(self, model, state, keep):
        """ A utility function to propose new points
        """
        qn, _, _ = self.proposal.get_proposal(state.branches_coords, inds=state.branches_inds)

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(qn, inds=state.branches_inds)
        logp[~keep] = -np.inf

        # Compute the lnprobs of the proposed position. 
        logl, new_blobs = model.compute_log_prob_fn(qn, inds=state.branches_inds, logp=logp)

        # Update the parameters, update the state. TODO: Fix blobs? 
        new_state = State(
            qn, log_prob=logl, log_prior=logp, blobs=new_blobs, inds=state.branches_inds, supplimental=state.supplimental
        ) # I create a new initial state that all are accepted
        return new_state

    def propose(self, log_diff_0, accepted, model, state, new_state, inds, inds_for_change, factors):
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

        alpha_0  = np.exp(log_diff_0)
        alpha_0[alpha_0 > 1.0] = 1.0 # np.min((1.0, diff_0)) 
        new_state.supplimental = BranchSupplimental({"past_alpha": alpha_0}, obj_contained_shape=(ntemps, nwalkers))
        
        # Get the old coords and posterior values
        prev_logl = state.log_prob
        prev_logp = state.log_prior
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp) # takes care of tempering
        
        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        dr_iter = 0
        while dr_iter <= self.max_iter and not np.all(accepted):
            rejected = ~accepted 
            plus_one_rej_inds = {}
            for name in inds_for_change:
                plus_one_inds = inds_for_change[name]['+1'][:,:2]

            plus_one_rej_inds[name] = plus_one_inds[rejected[(plus_one_inds[:,0],plus_one_inds[:,1])]]

            keep_rejected = np.unique(np.concatenate(list(plus_one_rej_inds.values())), axis=0)

            run_dr = np.zeros_like(rejected)
            run_dr[tuple(keep_rejected.T)] = True

            state, new_accepted, new_state = self.dr_scheme(state, new_state, run_dr, model, ntemps, nwalkers, inds_for_change, inds=inds)
            accepted += new_accepted

        # Create a given path
        states_path = [state]

        # print(" - Before DR: Accepted = {}/{}".format(np.sum(accepted), np.prod(accepted.shape))) 

        out_state, accepted = self.dr_scheme(state, states_path, alpha_0, accepted, prev_logP, model, ntemps, nwalkers, inds_for_change, inds=inds)       
        
        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return out_state, accepted
