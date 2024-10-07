# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from ..state import State
from .move import Move
from ..state import BranchSupplemental

__all__ = ["MHMove"]


class DelayedRejectionContainer:
    def __init__(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self, key, item)

        # Initialize
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
    Delayed Rejection scheme assuming symmetric and non-adaptive proposal distribution.
    We apply the DR algorithm only on the cases where we have rejected a +1 proposal for
    a given Reversible Jump proposal and branch.

    Refernces:

    Tierney L and Mira A, Stat. Med. 18 2507 (1999)
    Haario et al, Stat. Comput. 16:339-354 (2006)
    Mira A, Metron - International Journal of Statistics, vol. LIX, issue 3-4, 231-241 (2001)
    M. Trias, et al, https://arxiv.org/abs/0904.2207

    """

    def __init__(self, proposal, max_iter=10, **kwargs):
        self.proposal = proposal
        self.max_iter = max_iter
        self.dr_container = None
        super(DelayedRejection, self).__init__(**kwargs)

    def dr_scheme(
        self,
        state,
        new_state,
        keep_rejected,
        model,
        ntemps,
        nwalkers,
        inds_for_change,
        inds=None,
        dr_iter=0,
    ):
        """Calcuate the delayed rejection acceptace ratio.

        Args:
            stateslist (:class:`State`): a python list containing the proposed states

        Returns:
            logalpha: a numpy array containing the acceptance ratios per temperature and walker.
        """
        # Draw a uniform random for the previously rejected points
        randU = model.random.rand(
            ntemps, nwalkers
        )  # We draw for all temps x walkers but we ignore
        # previously accepted points by setting prior[rej] = - inf

        old_new_state = State(new_state, copy=True)

        # Propose a new point
        new_state, log_proposal_ratio = self.get_new_state(
            model, new_state, keep_rejected
        )  # Get a new state

        # Compute log-likelihood and log-prior
        logp = new_state.log_prior
        logl = new_state.log_like

        # Compute the logposterior for all
        logP = self.compute_log_posterior(logl, logp)

        # Compute log-likelihood and log-prior
        prev_logp = old_new_state.log_prior
        prev_logl = old_new_state.log_like

        # Compute the logposterior for all
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

        # Compute the acceptance ratio
        lndiff = logP - prev_logP + log_proposal_ratio
        alpha_1 = np.exp(lndiff)
        alpha_1[alpha_1 > 1.0] = 1.0  # np.min((1, alpha))

        # update delayed rejection alpha
        dr_alpha = np.exp(
            lndiff
            + np.log(1.0 - alpha_1)
            - np.log(1.0 - old_new_state.supplemental[:]["past_alpha"])
        )
        dr_alpha[dr_alpha > 1.0] = 1.0  # np.min((1., dr_alpha ))
        dr_alpha = np.nan_to_num(dr_alpha)  # Automatically reject NaNs

        new_state.supplemental[:] = {"alpha": dr_alpha}  # Replace current dr alpha

        new_accepted = np.logical_or(
            dr_alpha >= 1.0, randU < dr_alpha
        )  # Decide on accepted points

        # Update state with the new accepted points
        state = self.update(state, new_state, new_accepted)

        return state, new_accepted, new_state

    def get_new_state(self, model, state, keep):
        """A utility function to propose new points"""
        qn, factors = self.proposal.get_proposal(
            state.branches_coords, state.branches_inds, model.random
        )

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(qn, inds=state.branches_inds)
        logp[~keep] = -np.inf  # This trick help us compute only the indeces of interest

        # Compute the lnprobs of the proposed position.
        logl, new_blobs = model.compute_log_like_fn(
            qn, inds=state.branches_inds, logp=logp
        )

        # Update the parameters, update the state. TODO: Fix blobs?
        new_state = State(
            qn,
            log_like=logl,
            log_prior=logp,
            blobs=new_blobs,
            inds=state.branches_inds,
            supplemental=state.supplemental,
        )  # I create a new initial state that all are accepted
        return new_state, factors

    def propose(
        self,
        log_diff_0,
        accepted,
        model,
        state,
        new_state,
        inds,
        inds_for_change,
        factors,
    ):
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

        alpha_0 = np.exp(log_diff_0)
        alpha_0[alpha_0 > 1.0] = 1.0  # np.min((1.0, alpha_0))
        new_state.supplemental = BranchSupplemental(
            {"past_alpha": alpha_0}, base_shape=(ntemps, nwalkers)
        )

        # Check to make sure that the dimensions match.
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        dr_iter = 0  # Initialize

        # Begin main DR loop. Stop when we exceed the maximum iterations, or (extreme case) all proposals are accepted
        while dr_iter <= self.max_iter and not np.all(accepted):
            rejected = ~accepted  # Get rejected points

            # Get the +1 proposals that got previously rejected
            plus_one_rej_inds = {}
            for name in inds_for_change:
                plus_one_inds = inds_for_change[name]["+1"][:, :2]
            plus_one_rej_inds[name] = plus_one_inds[
                rejected[(plus_one_inds[:, 0], plus_one_inds[:, 1])]
            ]

            # Generate the indeces of the proposals that got rejected
            keep_rejected = np.unique(
                np.concatenate(list(plus_one_rej_inds.values())), axis=0
            )
            run_dr = np.zeros_like(rejected)
            run_dr[tuple(keep_rejected.T)] = True

            # Pass into the DR scheme
            state, new_accepted, new_state = self.dr_scheme(
                state,
                new_state,
                run_dr,
                model,
                ntemps,
                nwalkers,
                inds_for_change,
                inds=inds,
            )

            # Update the accepted, increment current iteration
            accepted += new_accepted
            dr_iter += 1

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
