# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Value
import numpy as np
from copy import deepcopy
from ..state import State
from .move import Move
from .delayedrejection import DelayedRejection
from .distgen import DistributionGenerate

__all__ = ["ReversibleJumpMove"]


class ReversibleJumpMove(Move):
    """
    An abstract reversible jump move from # TODO: add citations.

    Args:
        max_k (int or list of int): Maximum number(s) of leaves for each model.
        min_k (int or list of int): Minimum number(s) of leaves for each model.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)
        fix_change (int or None, optional): Fix the change in the number of leaves. Make them all 
            add a leaf or remove a leaf. This can be useful for some search functions. Options
            are ``+1`` or ``-1``. (default: ``None``)

    """

    def __init__(
        self,
        max_k,
        min_k,
        dr=None,
        dr_max_iter=5,
        tune=False,
        fix_change=None,
        **kwargs
    ):
        # super(ReversibleJumpMove, self).__init__(**kwargs)
        Move.__init__(self, is_rj=True, **kwargs)

        # setup leaf limits
        if isinstance(max_k, int):
            max_k = [max_k]

        if isinstance(max_k, int):
            min_k = [min_k]

        # store info
        self.max_k = max_k
        self.min_k = min_k
        self.tune = tune
        self.dr = dr
        self.fix_change = fix_change
        if self.fix_change not in [None, +1, -1]:
            raise ValueError("fix_change must be None, +1, or -1.")

        # Decide if DR is desirable. TODO: Now it uses the prior generator, we need to
        # think carefully if we want to use the in-model sampling proposal
        if self.dr is not None and self.dr is not False:
            if self.dr is True:  # Check if it's a boolean, then we just generate
                # from prior (kills the purpose, but yields "healther" chains)
                dr_proposal = DistributionGenerate(
                    self.priors, temperature_control=self.temperature_control
                )
            else:
                # Otherwise pass given input
                dr_proposal = self.dr

            self.dr = DelayedRejection(dr_proposal, max_iter=dr_max_iter)

    def setup(self, branches_coords):
        """Any setup for the proposal. 

        Args:
            branches_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.

        """

    def get_proposal(
        self, all_coords, all_inds, min_k_all, max_k_all, random, **kwargs
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            min_k_all (list): Minimum values of leaf ount for each model. Must have same order as ``all_cords``. 
            max_k_all (list): Maximum values of leaf ount for each model. Must have same order as ``all_cords``. 
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity. 

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        Raises:
            NotImplementedError: If this proposal is not implemented by a subclass.

        """
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def get_model_change_proposal(self, inds, random, min_k, max_k):
        """Helper function for changing the model count by 1.
        
        This helper function works with nested models where you want to add or remove
        one leaf at a time. 

        Args:
            inds (np.ndarray): ``inds`` values for this specific branch with shape 
                ``(ntemps, nwalkers, nleaves_max)``.
            random (object): Current random state of the sampler.
            min_k (int): Minimum allowable leaf count for this branch.
            max_k (int): Maximum allowable leaf count for this branch.

        Returns:
            dict: Keys are ``"+1"`` and ``"-1"``. Values are indexing information.
                    ``"+1"`` and ``"-1"`` indicate if a source is being added or removed, respectively.
                    The indexing information is a 2D array with shape ``(number changing, 3)``.
                    The length 3 is the index into each of the ``(ntemps, nwalkers, nleaves_max)``.
        
        """

        raise NotImplementedError

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        # TODO: check if temperatures are properly repeated after reset
        # this exposes anywhere in the proposal class to this information

        # Run any move-specific setup.
        self.setup(state.branches)

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        all_branch_names = list(state.branches.keys())

        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        for (branch_names_run, inds_run) in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):

            # gibbs sampling is only over branches so pick out that info
            coords_propose_in = {
                key: state.branches_coords[key] for key in branch_names_run
            }
            inds_propose_in = {
                key: state.branches_inds[key] for key in branch_names_run
            }
            branches_supp_propose_in = {
                key: state.branches_supplimental[key] for key in branch_names_run
            }

            if len(list(coords_propose_in.keys())) == 0:
                raise ValueError(
                    "Right now, no models are getting a reversible jump proposal. Check min_k and max_k or do not use rj proposal."
                )

            # get min and max leaf information
            max_k_all = []
            min_k_all = []
            for brn in branch_names_run:
                # get index into all branches to get the proper max_k and min_k
                ind_keep_here = all_branch_names.index(brn)
                max_k_all.append(self.max_k[ind_keep_here])
                min_k_all.append(self.min_k[ind_keep_here])

            self.current_model = model
            self.current_state = state
            # propose new sources and coordinates
            q, new_inds, factors = self.get_proposal(
                coords_propose_in,
                inds_propose_in,
                min_k_all,
                max_k_all,
                model.random,
                branch_supps=branches_supp_propose_in,
                supps=state.supplimental,
            )

            branches_supps_new = {
                key: item for key, item in branches_supp_propose_in.items()
            }
            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # put back any branches that were left out from Gibbs split
            for name, branch in state.branches.items():
                if name not in q:
                    q[name] = state.branches[name].coords[:].copy()
                if name not in new_inds:
                    new_inds[name] = state.branches[name].inds[:].copy()

                if name not in branches_supps_new:
                    branches_supps_new[name] = state.branches_supplimental[name]

            # fix any ordering issues
            q, new_inds, branches_supps_new = self.ensure_ordering(
                list(state.branches.keys()), q, new_inds, branches_supps_new
            )

            # TODO: check this
            edge_factors = np.zeros((ntemps, nwalkers))
            # get factors for edges
            for (name, branch), min_k, max_k in zip(
                state.branches.items(), self.min_k, self.max_k
            ):

                if name not in branch_names_run:
                    continue

                # get old and new values
                old_nleaves = branch.nleaves
                new_nleaves = new_inds[name].sum(axis=-1)

                # do not work on sources with fixed source count
                if min_k == max_k or min_k + 1 == max_k:
                    # min_k == max_k --> no rj proposal
                    # min_k + 1 == max_k --> no edge factors because it is guaranteed to be min_k or max_k
                    continue

                elif min_k > max_k:
                    raise ValueError("min_k cannot be greater than max_k.")

                else:
                    # fix proposal asymmetry at bottom of k range (kmin -> kmin + 1)
                    inds_min = np.where(old_nleaves == min_k)
                    # numerator term so +ln
                    edge_factors[inds_min] += np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range (kmax -> kmax - 1)
                    inds_max = np.where(old_nleaves == max_k)
                    # numerator term so -ln
                    edge_factors[inds_max] += np.log(1 / 2.0)

                    # fix proposal asymmetry at bottom of k range (kmin + 1 -> kmin)
                    inds_min = np.where(new_nleaves == min_k)
                    # numerator term so +ln
                    edge_factors[inds_min] -= np.log(1 / 2.0)

                    # fix proposal asymmetry at top of k range (kmax - 1 -> kmax)
                    inds_max = np.where(new_nleaves == max_k)
                    # numerator term so -ln
                    edge_factors[inds_max] -= np.log(1 / 2.0)

            factors += edge_factors

            # setup supplimental information

            if state.supplimental is not None:
                # TODO: should there be a copy?
                new_supps = deepcopy(state.supplimental)

            else:
                new_supps = None

            # for_transfer information can be taken directly from custom proposal

            # supp info

            if hasattr(self, "mt_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_supps = self.mt_supps

            if hasattr(self, "mt_branch_supps"):
                # logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
                new_branch_supps = self.mt_branch_supps

            # logp and logl

            # Compute prior of the proposed position
            if hasattr(self, "mt_lp"):
                logp = self.mt_lp.reshape(ntemps, nwalkers)

            else:
                logp = model.compute_log_prior_fn(q, inds=new_inds)

            self.fix_logp_gibbs(branch_names_run, inds_run, logp, new_inds)

            if hasattr(self, "mt_ll"):
                logl = self.mt_ll.reshape(ntemps, nwalkers)

            else:
                # Compute the ln like of the proposed position.
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=new_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=branches_supps_new,
                )

            # posterior and previous info

            logP = self.compute_log_posterior(logl, logp)

            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # acceptance fraction
            lnpdiff = factors + logP - prev_logP

            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # update with new state
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=None,
                inds=new_inds,
                supplimental=new_supps,
                branch_supplimental=branches_supps_new,
            )
            state = self.update(state, new_state, accepted)

            # apply delayed rejection to walkers that are +1
            # TODO: need to reexamine this a bit. I have a feeling that only applying
            # this to +1 may not be preserving detailed balance. You may need to
            # "simulate it" for -1 similar to what we do in multiple try
            if self.dr:
                raise NotImplementedError
                # for name, branch in state.branches.items():
                #     # We have to work with the binaries added only.
                #     # We need the a) rejected points, b) the model,
                #     # c) the current state, d) the indices where we had +1 (True),
                #     # and the e) factors.
                state, accepted = self.dr.propose(
                    lnpdiff,
                    accepted,
                    model,
                    state,
                    new_state,
                    new_inds,
                    inds_for_change,
                    factors,
                )  # model, state

            # If RJ is true we control only on the in-model step, so no need to do it here as well
            # In most cases, RJ proposal is has small acceptance rate, so in the end we end up
            # switching back what was swapped in the previous in-model step.
            # TODO: MLK: I think we should allow for swapping but no adaptation.

            if self.temperature_control is not None and not self.prevent_swaps:
                state = self.temperature_control.temper_comps(state, adapt=False)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

            return state, accepted
