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
            This is a keyword argument, nut it is required.
        min_k (int or list of int): Minimum number(s) of leaves for each model.
            This is a keyword argument, nut it is required.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)
        fix_change (int or None, optional): Fix the change in the number of leaves. Make them all
            add a leaf or remove a leaf. This can be useful for some search functions. Options
            are ``+1`` or ``-1``. (default: ``None``)

    """

    def __init__(
        self,
        max_k=None,
        min_k=None,
        dr=None,
        dr_max_iter=5,
        tune=False,
        fix_change=None,
        **kwargs
    ):
        # super(ReversibleJumpMove, self).__init__(**kwargs)
        Move.__init__(self, is_rj=True, **kwargs)

        if max_k is None or min_k is None:
            raise ValueError("Must provide min_k and max_k keyword arguments for RJ.")

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
            if self.dr is True: # Check if it's a boolean, then we just generate
                                # from prior (kills the purpose, but yields "healther" chains)
                dr_proposal = PriorGenerate(
                self.priors,
                temperature_control=self.temperature_control)
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

    def get_model_change_proposal(self, state, model):

        inds_for_change = {}
        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        assert len(self.min_k) == len(self.max_k)
        assert len(state.branches.keys()) == len(self.max_k)

        inds_choose=np.where(np.array(self.max_k)>np.array(self.min_k))[0]
        names_upd=[]
        if len(inds_choose)>0:
            ind_upd=np.random.choice(inds_choose)


            names_upd+=[list(state.branches.keys())[ind_upd]]

            if list(state.branches.keys())[ind_upd]=='qpl':
                names_upd+=['qbpl']

            elif list(state.branches.keys())[ind_upd]=='qbpl':
                names_upd+=['qpl']


            if list(state.branches.keys())[ind_upd]=='pl':
                prand=np.random.rand()
                if prand<0.5:
                    names_upd+=['bpl']
            elif list(state.branches.keys())[ind_upd]=='bpl':
                prand=np.random.rand()
                if prand<0.5:
                    names_upd+=['pl']

            # if list(state.branches.keys())[ind_upd]=='pl':
            #     names_upd+=['bpl']
            # elif list(state.branches.keys())[ind_upd]=='bpl':
            #     names_upd+=['pl']

        for (name, branch), min_k, max_k in zip(
            state.branches.items(), self.min_k, self.max_k
        ):
            if self.proposal_branch_names is not None and name not in self.proposal_branch_names:
                # skip this one
                continue

            if min_k == max_k:
                continue
            elif min_k > max_k:
                raise ValueError("min_k is greater than max_k. Not allowed.")
            elif name not in names_upd:
                continue

            nleaves = branch.nleaves
            # choose whether to add or remove
            if self.fix_change is None:
                change = model.random.choice([-1, +1], size=nleaves.shape)
            else:
                change = np.full(nleaves.shape, self.fix_change)

        Returns:
            dict: Keys are ``"+1"`` and ``"-1"``. Values are indexing information.
                    ``"+1"`` and ``"-1"`` indicate if a source is being added or removed, respectively.
                    The indexing information is a 2D array with shape ``(number changing, 3)``.
                    The length 3 is the index into each of the ``(ntemps, nwalkers, nleaves_max)``.

        """

            # setup storage for this information
            inds_for_change[name] = {}
            num_increases = np.sum(change == +1)
            inds_for_change[name]["+1"] = np.zeros((num_increases, 3), dtype=int)
            num_decreases = np.sum(change == -1)
            inds_for_change[name]["-1"] = np.zeros((num_decreases, 3), dtype=int)

            # TODO: not loop ? Is it necessary?
            # TODO: might be able to subtract new inds from old inds type of thing
            # fill the inds_for_change
            increase_i = 0
            decrease_i = 0
            for t in range(ntemps):
                for w in range(nwalkers):
                    # check if add or remove
                    change_tw = change[t][w]
                    # inds array from specific walker
                    inds_tw = branch.inds[t][w]

                    # adding
                    if change_tw == +1:
                        # find where leaves are not currently used
                        inds_false = np.where(inds_tw == False)[0]
                        # decide which spot to add
                        ind_change = model.random.choice(inds_false)
                        # put in the indexes into inds arrays
                        inds_for_change[name]["+1"][increase_i] = np.array(
                            [t, w, ind_change], dtype=int
                        )
                        # count increases
                        increase_i += 1

                        # TODO: change this so we can actually generate new coords rather than reuse old standing coords
                        # q[name][t, w, ind_change, :] = branch.coords[
                        #    t, w, ind_change, :
                        # ]

                    # removing
                    elif change_tw == -1:
                        # change_tw == -1
                        # find which leavs are used
                        inds_true = np.where(inds_tw == True)[0]
                        # choose which to remove
                        ind_change = model.random.choice(inds_true)
                        # add indexes into inds
                        if inds_for_change[name]["-1"].shape[0] > 0:
                            inds_for_change[name]["-1"][decrease_i] = np.array(
                                [t, w, ind_change], dtype=int
                            )
                            decrease_i += 1
                        # do not care currently about what we do with discarded coords, they just sit in the state
                    # model component number not changing
                    else:
                        pass
        return inds_for_change,names_upd

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """

        # TODO: keep this?
        # this exposes anywhere in the proposal class to this information

        # Run any move-specific setup.
        self.setup(state.branches)

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # TODO: check if temperatures are properly repeated after reset
        # TODO: do we want an probability that the model count will not change?
        inds_for_change, names_upd = self.get_model_change_proposal(state, model)
        coords_propose_in = state.branches_coords
        inds_propose_in = state.branches_inds
        branches_supp_propose_in = state.branches_supplimental

        if len(list(coords_propose_in.keys())) == 0:
            raise ValueError("Right now, no models are getting a reversible jump proposal. Check min_k and max_k or do not use rj proposal.")

        # propose new sources and coordinates
        q0, new_inds0, factors = self.get_proposal(
            state.branches_coords, state.branches_inds, inds_for_change, model.random, branch_supps=state.branches_supplimental, supps=state.supplimental
        )

        for name, branch in state.branches.items():
            if name not in q0:
                q0[name] = state.branches[name].coords[:].copy()
            if name not in new_inds0:
                new_inds0[name] = state.branches[name].inds[:].copy()

        q={}
        new_inds={}
        for name, branch in state.branches.items():
            q[name]=q0[name][:].copy()
            new_inds[name]=new_inds0[name][:].copy()

        if "inds_here" in q:
            temp_transfer_info = {name: q.pop(name) for name in ["ll", "lp", "inds_here"]}
        else:
            temp_transfer_info = {}

        # TODO: check this
        edge_factors = np.zeros((ntemps, nwalkers))
        # get factors for edges
        for (name, branch), min_k, max_k in zip(
            state.branches.items(), self.min_k, self.max_k
        ):
            nleaves = branch.nleaves
            nleaves_new = np.sum(new_inds[name],axis=-1)

            # do not work on sources with fixed source count
            if min_k+1 >= max_k:
                continue
            if name not in names_upd:
                continue

            # fix proposal asymmetry at bottom of k range
            inds_min = np.where(nleaves == min_k)
            # numerator term so +ln
            edge_factors[inds_min] += np.log(1 / 2.0)

            # fix proposal asymmetry at top of k range
            inds_max = np.where(nleaves == max_k)
            # numerator term so -ln
            edge_factors[inds_max] += np.log(1 / 2.0)

            # fix proposal asymmetry at bottom of k range (kmin + 1)
            inds_min = np.where((nleaves == min_k + 1) & (nleaves_new == min_k))
            # numerator term so +ln
            edge_factors[inds_min] -= np.log(1 / 2.0)

            # fix proposal asymmetry at top of k range (kmax - 1)
            inds_max = np.where((nleaves == max_k - 1) & (nleaves_new == max_k))
            # numerator term so -ln
            edge_factors[inds_max] -= np.log(1 / 2.0)

        factors += edge_factors

        # setup supplimental information

        if state.supplimental is not None:
            # TODO: should there be a copy?
            new_supps = deepcopy(state.supplimental)

        else:
            new_supps = None

        if not np.all(np.asarray(list(state.branches_supplimental.values())) == None):
            # TODO: remove this?
            new_branch_supps = deepcopy(state.branches_supplimental)
            for name in new_branch_supps:
                if new_branch_supps[name] is not None:
                    indicator_inds = (new_inds[name].astype(int) - state.branches_inds[name].astype(int)) > 0
                    new_branch_supps[name].add_objects({"inds_keep": indicator_inds})

        else:
            new_branch_supps = None
            if new_supps is not None:
                new_branch_supps = {}
                for name in new_branch_supps:
                    if new_branch_supps[name] is not None:
                        indicator_inds = (new_inds[name].astype(int) - state.branches_inds[name].astype(int)) > 0
                        new_branch_supps[name] = BranchSupplimental({"inds_keep": indicator_inds}, obj_contained_shape=new_inds[name].shape, copy=False)

        if hasattr(self, "new_supps_for_transfer"):
            #logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
            new_supps = self.new_supps_for_transfer

        if hasattr(self, "new_branch_supps_for_transfer"):
            #logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
            new_branch_supps = self.new_branch_supps_for_transfer

        # TODO: adjust this setup
        # Compute prior of the proposed position
        if hasattr(self, "ll_for_transfer"):
            #logp = self.lp_for_transfer.reshape(ntemps, nwalkers)
            logp = model.compute_log_prior_fn(q, inds=new_inds)
            logl = self.ll_for_transfer.reshape(ntemps, nwalkers)
            loglcheck, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)
            if not np.all(np.abs(logl[logl != -1e300] - loglcheck[logl != -1e300]) < 1e-5):
                breakpoint()

        else:
            logp = model.compute_log_prior_fn(q, inds=new_inds)

            # pass ll values from special likelihoods in the proposal
            # prevent ll values from being run again
            #if "inds_here" in temp_transfer_info:
            #    logp_keep = logp[temp_transfer_info["inds_here"]]
            #    logp[temp_transfer_info["inds_here"]] = -np.inf

            #if (new_branch_supps is not None or new_supps is not None) and self.adjust_supps_pre_logl_func is not None:
            #    self.adjust_supps_pre_logl_func(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)

            # Compute the lnprobs of the proposed position.
            logl, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)

            # pass ll values from special likelihoods in the proposal
            # put int he correct values
            #if "inds_here" in temp_transfer_info:
            #    logp[temp_transfer_info["inds_here"]] = logp_keep
            #    logl[temp_transfer_info["inds_here"]] = temp_transfer_info["ll"]

        logP = self.compute_log_posterior(logl, logp)

        prev_logl = state.log_prob

        prev_logp = state.log_prior

        # TODO: check about prior = - inf
        # takes care of tempering
        prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

        # TODO: fix this
        # this is where _metropolisk should come in
        lnpdiff = factors + logP - prev_logP

        accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

        # TODO: deal with blobs
        new_state = State(q, log_prob=logl, log_prior=logp, blobs=None, inds=new_inds, supplimental=new_supps, branch_supplimental=new_branch_supps)
        state = self.update(state, new_state, accepted)

        # apply delayed rejection to walkers that are +1
        # TODO: need to reexamine this a bit. I have a feeling that only applying
        # this to +1 may not be preserving detailed balance. You may need to
        # "simulate it" for -1 similar to what we do in multiple try
        if self.dr:
            # for name, branch in state.branches.items():
            #     # We have to work with the binaries added only.
            #     # We need the a) rejected points, b) the model,
            #     # c) the current state, d) the indices where we had +1 (True),
            #     # and the e) factors.
            state, accepted = self.dr.propose(
                lnpdiff, accepted, model, state, new_state, new_inds, inds_for_change, factors
            )  # model, state

        # If RJ is true we control only on the in-model step, so no need to do it here as well
        # In most cases, RJ proposal is has small acceptance rate, so in the end we end up
        # switching back what was swapped in the previous in-model step.
        # TODO: MLK: I think we should allow for swapping but no adaptation.

        if self.temperature_control is not None and not self.prevent_swaps:
             state, accepted = self.temperature_control.temper_comps(state, accepted, adapt=False)
        if np.any(state.log_prob > 1e10):
            breakpoint()

        return state, accepted
