# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Value
import numpy as np
from copy import deepcopy
from ..state import State, BranchSupplimental
from .move import Move
from .delayedrejection import DelayedRejection

__all__ = ["ProductSpaceMove"]


class ProductSpaceMove(Move):
    """
    An abstract reversible jump move from # TODO: add citations.

    Args:
        max_k (int or list of int): Maximum number(s) of leaves for each model.
        min_k (int or list of int): Minimum number(s) of leaves for each model.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)

    """

    def __init__(
        self, max_k, min_k, random_change=True, tune=False, **kwargs
    ):
        super(ProductSpaceMove, self).__init__(**kwargs)

        self.max_k = max_k
        self.min_k = min_k
        self.possible_models = list(np.arange(min_k, max_k + 1))
        self.random_change = random_change

        
    def setup(self, coords):
        pass

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            all_inds_for_change (dict): Keys are ``branch_names``. Values are
                dictionaries. These dictionaries have keys ``"+1"`` and ``"-1"``,
                indicating waklkers that are adding or removing a leafm respectively.
                The values for these dicts are ``int`` np.ndarray[..., 3]. The "..." indicates
                the number of walkers in all temperatures that fall under either adding
                or removing a leaf. The second dimension, 3, is the indexes into
                the three-dimensional arrays within ``all_inds`` of the specific leaf
                that is being added or removed from those leaves currently considered.
            random (object): Current random state of the sampler.

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

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
            state (:class:`State`): Current state of the sampler.

        Returns:
            :class:`State`: State of sampler after proposal is complete.

        """
        # TODO: check stretch proposaln works here?
        # Check that the dimensions are compatible.
        ndim_total = 0

        # Run any move-specific setup.
        self.setup(state.branches)

        if "model_indicator" not in state.branches:
            raise ValueError("When using product space jumps, must include the 'model_indicator' key in available modes.")

        ntemps, nwalkers, nleaves_max, ndim_indicator = state.branches["model_indicator"].shape

        assert nleaves_max == 1
        assert ndim_indicator == 1

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        model_indicator = state.branches["model_indicator"].coords.astype(int).squeeze().copy()
        unique_model_indicators = np.unique(model_indicator)
        
        factors = np.zeros((ntemps, nwalkers))
        model_indicator_temp = model_indicator.copy()
        if self.random_change:
            for indicator in unique_model_indicators:
                inds_keep = np.where(model_indicator_temp == indicator)
                temp_inds = self.possible_models.copy()
                temp_inds.remove(indicator)
                new_model_indicator = model.random.choice(temp_inds, len(inds_keep[0]), replace=True)
                model_indicator[inds_keep] = new_model_indicator 

        else:
            for indicator in unique_model_indicators:
                inds_keep = np.where(model_indicator_temp == indicator)

                if indicator == self.min_k:
                    # TODO: check this
                    new_model_indicator = model_indicator_temp[inds_keep] + 1
                    factors[inds_keep] = +np.log(1 / 2.0)

                elif indicator == self.max_k:
                    new_model_indicator = model_indicator_temp[inds_keep] - 1
                    factors[inds_keep] = -np.log(1 / 2.0)

                else:
                    if indicator < min_k or indicator > max_k:
                        raise ValueError("model_indicators outside min_k and max_k.")

                    new_model_indicator = model_indicator_temp[inds_keep] + model.random.choice([-1, 1], len(inds_keep[0]), replace=True)

                model_indicator[inds_keep] = new_model_indicator 
    
        q = deepcopy(state.branches_coords)
        q["model_indicator"][:] = model_indicator.astype(float)[:, :, None, None]
        
        new_inds = {name: np.ones_like(values, dtype=bool) for name, values in state.branches_inds.items()}
        unique_model_indicators = np.unique(model_indicator)
        model_names = list(q.keys())
        model_names.remove("model_indicator")
        assert model_indicator.dtype == int
        assert len(unique_model_indicators) <= len(model_names)

        # adjust indices
        for i in unique_model_indicators:
            new_inds[model_names[i]][model_indicator != i] = False

        # propose new sources and coordinates

        # setup supplimental information

        if state.supplimental is not None:
            # TODO: should there be a copy?
            new_supps = deepcopy(state.supplimental)
            
        else:   
            new_supps = None

        if not np.all(np.asarray(list(state.branches_supplimental.values())) == None):
            new_branch_supps = deepcopy(state.branches_supplimental)
            for name in new_branch_supps:
                indicator_inds = (new_inds[name].astype(int) - state.branches_inds[name].astype(int)) > 0
                new_branch_supps[name].add_objects({"inds_keep": indicator_inds})

        else:
            new_branch_supps = None
            if new_supps is not None:
                new_branch_supps = {}
                for name in new_branch_supps:
                    indicator_inds = (new_inds[name].astype(int) - state.branches_inds[name].astype(int)) > 0
                    new_branch_supps[name] = BranchSupplimental({"inds_keep": indicator_inds}, obj_contained_shape=new_inds[name].shape, copy=False)

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(q, inds=new_inds)
        
        #if (new_branch_supps is not None or new_supps is not None) and self.adjust_supps_pre_logl_func is not None:
        #    self.adjust_supps_pre_logl_func(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)

        # Compute the lnprobs of the proposed position.
        logl, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)

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

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
