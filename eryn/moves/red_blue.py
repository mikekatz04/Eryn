# -*- coding: utf-8 -*-
from abc import ABC
from copy import deepcopy
import numpy as np
import warnings

from ..state import BranchSupplimental, State
from .move import Move



__all__ = ["RedBlueMove"]

class RedBlueMove(Move, ABC):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.

    # TODO: think about this for reversible jump.

    Args:
        nsplits (int, optional): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (bool, optional): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.

        live_dangerously (bool, optional): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.
        gibbs_sampling_leaves_per (int, optional): Number of individual leaves
            to sample over during Gibbs Sampling. ``None`` means there will be
            no Gibbs sampling. Currently, if you specify anything other than ``None``,
            it will split up different branches automatically and then Gibbs sample
            over each branch indivudally. Default is ``None``.

    ``kwargs`` are passed to :class:`Move` class.

    """

    def __init__(
        self,
        nsplits=2,
        randomize_split=True,
        gibbs_sampling_leaves_per=None,
        live_dangerously=False,
        **kwargs
    ):
        super(RedBlueMove, self).__init__(**kwargs)
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

        if gibbs_sampling_leaves_per is not None:
            if (
                not isinstance(gibbs_sampling_leaves_per, int)
                or gibbs_sampling_leaves_per < 1
            ):
                raise ValueError(
                    "gibbs_sampling_leaves_per must be an integer greater than zero."
                )

        self.gibbs_sampling_leaves_per = gibbs_sampling_leaves_per

    def setup(self, coords):
        """Any setup necessary for the proposal"""
        pass

    @classmethod
    def get_proposal(self, sample, complement, random, inds=None):
        """Make a proposal

        Args:
            sample (dict): Keys are ``branch_names``. Values are
                np.ndarray[subset size, nleaves_max, ndim]. This is the subset
                whose ``coords`` are being proposed.
            complement (dict): Keys are ``branch_names``. Values are lists of other
                other np.ndarray[nwalkers - subset size, nleaves_max, ndim] from
                all other subsets. This is the compliment whose ``coords`` are
                used to form the proposal for the ``sample`` subset.
            random (object): Current random state of the sampler.
            inds (dict, optional): # TODO check this.

        Returns:
            tuple: Tuple contained proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as np.ndarray[subset size, ndim * nleaves_max]
                of new coordinates. Second entry is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

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
        # Check that the dimensions are compatible.
        ndim_total = 0
        for branch in state.branches.values():
            ntemps, nwalkers, nleaves_, ndim_ = branch.shape
            ndim_total += ndim_ * nleaves_

        if nwalkers < 2 * ndim_total and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # TODO: deal with more intensive acceptance fractions
        # Run any move-specific setup.
        self.setup(state.branches)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        all_inds = np.tile(np.arange(nwalkers), (ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            [np.random.shuffle(x) for x in inds]

        if self.gibbs_sampling_leaves_per is not None:
            # setup for gibbs sampling
            gibbs_splits = []
            for name, branch in state.branches.items():
                nleaves_max_here = branch.nleaves.max()
                num_each = np.arange(0, nleaves_max_here)

                split_inds = np.arange(
                    self.gibbs_sampling_leaves_per,
                    nleaves_max_here,
                    self.gibbs_sampling_leaves_per,
                )

                num_each_splits = np.split(num_each, split_inds)
                for each in num_each_splits:
                    gibbs_splits.append([name, each, nleaves_max_here])

        else:
            gibbs_splits = [None]

        for gs in gibbs_splits:
            accepted_here = np.zeros((ntemps, nwalkers), dtype=bool)
            for split in range(self.nsplits):
                S1 = inds == split
                num_total_here = np.sum(inds == split)
                nwalkers_here = np.sum(S1[0])

                all_inds_shaped = all_inds[S1].reshape(ntemps, nwalkers_here)
                fixed_inds_shaped = all_inds[~S1].reshape(ntemps, nwalkers_here)

                new_inds = {
                    name: np.take_along_axis(
                        state.branches[name].inds, all_inds_shaped[:, :, None], axis=1
                    )
                    for name in state.branches
                }

                temp_coords = {
                    name: np.take_along_axis(
                        state.branches[name].coords,
                        all_inds_shaped[:, :, None, None],
                        axis=1,
                    )
                    for name in state.branches
                }

                # fix values in q that are not actually being tested here
                # new_inds_adjust is used temporarily for this
                new_inds_adjust = deepcopy(new_inds)
                if gs is not None:
                    # TODO: adjust to not one model at a time?
                    # adjust new inds for gibbs sampling
                    name_keep, inds_keep, nleaves_max_here = gs
                    keep_arr = np.zeros_like(new_inds_adjust[name], dtype=bool)
                    for name in new_inds_adjust:
                        if name != name_keep:
                            # not using any of these
                            new_inds_adjust[name] = False

                        else:
                            inds_1 = np.cumsum(new_inds_adjust[name], axis=2) * (
                                new_inds_adjust[name] == True
                            )

                            for ii in inds_keep:
                                keep_inds = np.where(inds_1 == (ii + 1))
                                keep_arr[keep_inds] = True

                            new_inds_adjust[name] = keep_arr.copy()

                sets = {
                    key: [
                        np.take_along_axis(
                        state.branches[key].coords,
                        all_inds[inds == j].reshape(ntemps, nwalkers_here)[:, :, None, None],
                        axis=1,
                    )
                        for j in range(self.nsplits)
                    ]
                    for key in state.branches
                }

                s = {key: sets[key][split] for key in sets}
                c = {
                    key: sets[key][:split] + sets[key][split + 1 :] for key in sets
                }

                # Get the move-specific proposal.
                # Get the move-specific proposal.

                # need to trick stretch proposal into using the dimenionality associated
                # with Gibbs sampling if it is being used
                
                temp_inds_s = {
                    name: new_inds_adjust[name][:].reshape((ntemps, -1,) + state.branches[name].coords.shape[2:3]) for name in new_inds_adjust
                }

                temp_inds_c = {
                    name: np.take_along_axis(
                        state.branches[name].inds, all_inds[inds != split].reshape(ntemps, -1, 1), axis=1
                    )
                    for name in state.branches_inds
                }

                q_temp, factors_temp = self.get_proposal(
                    s, c, model.random, inds_s=temp_inds_s, inds_c=temp_inds_c
                )
                q = {}
                for name in q_temp:
                    q[name] = q_temp[name]  # TODO: take this out? .reshape((ntemps, -1,) + state.branches[name].coords.shape[2:])

                factors = factors_temp.reshape((ntemps, -1,))

                if "model_indicator" in q:
                    model_indicator = model.log_prob_fn.f.map_fn(q["model_indicator"].squeeze())
                    unique_model_indicators = np.unique(model_indicator)
                    model_names = list(q.keys())
                    model_names.remove("model_indicator")
                    assert model_indicator.dtype == int
                    assert len(unique_model_indicators) <= len(model_names)

                    # adjust indices
                    for i in unique_model_indicators:
                        new_inds_adjust[model_names[i]][model_indicator != i] = False

                    # adjust factors
                    if hasattr(self, "adjust_factors"):
                        ndims = np.asarray([q[key].shape[-1] for key in model_names])
                        # +1 is for the update of the model indicator
                        ndims_old = np.full_like(factors, ndims.sum() + 1)
                        ndims_new = ndims[model_indicator] + 1
                        self.adjust_factors(factors, ndims_old, ndims_new)

                for name in q:
                    q[name] = q[name] * (
                        new_inds_adjust[name][:, :, :, None]
                    ) + temp_coords[name] * (~new_inds_adjust[name][:, :, :, None])

                # Compute prior of the proposed position
                logp = model.compute_log_prior_fn(q, inds=new_inds)

                # set logp for walkers with no leaves that are being tested
                # in this gibbs run
                if gs is not None:
                    logp[np.where(np.sum(keep_arr, axis=-1) == 0)] = -np.inf

                # setup supplimental information
                if state.supplimental is not None:
                    # TODO: should there be a copy?
                    new_supps = BranchSupplimental(state.supplimental.take_along_axis(all_inds_shaped, axis=1), obj_contained_shape=(ntemps, nwalkers), copy=False)

                else:
                    new_supps = None

                # default for removing inds info from supp
                if not np.all(np.asarray(list(state.branches_supplimental.values())) == None):
                    new_branch_supps = {
                        name: state.branches[name].branch_supplimental.take_along_axis(
                            all_inds_shaped[:, :, None], axis=1
                        )
                        for name in state.branches
                    }
                    
                    new_branch_supps = {name: BranchSupplimental(new_branch_supps[name], obj_contained_shape=new_inds[name].shape, copy=False) for name in new_branch_supps}
                    for name in new_branch_supps:
                        new_branch_supps[name].add_objects({"inds_keep": new_inds_adjust[name]})

                else:
                    new_branch_supps = None
                    if new_supps is not None:
                        new_branch_supps = {name: BranchSupplimental({"inds_keep": new_inds_adjust[name]}, obj_contained_shape=new_inds[name].shape, copy=False) for name in new_branch_supps}

                # TODO: add supplimental prepare step
                #if (new_branch_supps is not None or new_supps is not None) and self.adjust_supps_pre_logl_func is not None:
                #    self.adjust_supps_pre_logl_func(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps, inds_keep=new_inds_adjust)

                # Compute the lnprobs of the proposed position.
                logl, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps)
                    
                if (new_branch_supps is not None and not np.all(np.asarray(list(new_branch_supps.values())) == None)) or new_supps is not None:
                    if new_branch_supps is not None:
                        for name in new_branch_supps:
                            new_branch_supps[name].remove_objects("inds_keep")
                    elif new_supps is not None:
                        pass  # del new_branch_supps

                # catch and warn about nans
                if np.any(np.isnan(logl)):
                    logl[np.isnan(logl)] = -1e300
                    warnings.warn("Getting Nan in likelihood computation.")

                logP = self.compute_log_posterior(logl, logp)

                prev_logl = np.take_along_axis(state.log_prob, all_inds_shaped, axis=1)

                prev_logp = np.take_along_axis(state.log_prior, all_inds_shaped, axis=1)

                # TODO: check about prior = - inf
                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

                # TODO: think about factors in tempering
                lnpdiff = factors + logP - prev_logP

                keep = lnpdiff > np.log(model.random.rand(ntemps, nwalkers_here))

                # if gibbs sampling, this will say it is accepted if
                # any of the gibbs proposals were accepted
                np.put_along_axis(
                    accepted_here, all_inds_shaped, keep, axis=1,
                )

                # readout for total
                accepted = (accepted.astype(int) + accepted_here.astype(int)).astype(
                    bool
                )

                new_state = State(
                    q, log_prob=logl, log_prior=logp, blobs=new_blobs, inds=new_inds, supplimental=new_supps, branch_supplimental=new_branch_supps
                )

                state = self.update(
                    state, new_state, accepted_here, subset=all_inds_shaped
                )

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        # make accepted move specific ?
        return state, accepted

