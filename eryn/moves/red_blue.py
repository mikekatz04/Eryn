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

        # TODO: have each move keep track of its own acceptance fraction
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
            tuple: (state, accepted)
                The first return is the state of the sampler after the move.
                The second return value is the accepted count array.

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

        all_branch_names = list(state.branches.keys())

        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        for (branch_names_run, inds_run) in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):

            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            if not at_least_one_proposal:
                continue

            breakpoint()
            accepted_here = np.zeros((ntemps, nwalkers), dtype=bool)
            for split in range(self.nsplits):
                S1 = inds == split
                num_total_here = np.sum(inds == split)
                nwalkers_here = np.sum(S1[0])

                all_inds_shaped = all_inds[S1].reshape(ntemps, nwalkers_here)
                fixed_inds_shaped = all_inds[~S1].reshape(ntemps, nwalkers_here)

                real_inds_subset = {
                    name: np.take_along_axis(
                        state.branches[name].inds, all_inds_shaped[:, :, None], axis=1,
                    )
                    for name in inds_going_for_proposal
                }

                new_inds = {
                    name: np.take_along_axis(
                        inds_going_for_proposal[name],
                        all_inds_shaped[:, :, None],
                        axis=1,
                    )
                    for name in inds_going_for_proposal
                }

                temp_coords = {
                    name: np.take_along_axis(
                        coords_going_for_proposal[name],
                        all_inds_shaped[:, :, None, None],
                        axis=1,
                    )
                    for name in coords_going_for_proposal
                }

                sets = {
                    key: [
                        np.take_along_axis(
                            state.branches[key].coords,
                            all_inds[inds == j].reshape(ntemps, nwalkers_here)[
                                :, :, None, None
                            ],
                            axis=1,
                        )
                        for j in range(self.nsplits)
                    ]
                    for key in state.branches
                }

                s = {key: sets[key][split] for key in sets}
                c = {key: sets[key][:split] + sets[key][split + 1 :] for key in sets}

                # Get the move-specific proposal.
                # Get the move-specific proposal.

                # need to trick stretch proposal into using the dimenionality associated
                # with Gibbs sampling if it is being used

                temp_inds_s = {
                    name: new_inds[name][:].reshape(
                        (ntemps, -1,) + state.branches[name].coords.shape[2:3]
                    )
                    for name in new_inds
                }

                temp_inds_c = {
                    name: np.take_along_axis(
                        state.branches[name].inds,
                        all_inds[inds != split].reshape(ntemps, -1, 1),
                        axis=1,
                    )
                    for name in state.branches_inds
                }

                gibbs_ndim = sum([inds_run_tmp.sum() for inds_run_tmp in inds_run])

                q, factors_temp = self.get_proposal(
                    s, c, model.random, gibbs_ndim=gibbs_ndim
                )

                # account for gibbs sampling
                self.cleanup_proposals_gibbs(branch_names_run, inds_run, q, state)

                # Compute prior of the proposed position
                # new_inds_prior is adjusted if product-space is used
                logp = model.compute_log_prior_fn(q, inds=real_inds_subset)

                self.fix_logp_gibbs(branch_names_run, inds_run, logp, real_inds_subset)

                # setup supplimental information
                if state.supplimental is not None:
                    # TODO: should there be a copy?
                    new_supps = BranchSupplimental(
                        state.supplimental.take_along_axis(all_inds_shaped, axis=1),
                        obj_contained_shape=(ntemps, nwalkers),
                        copy=False,
                    )

                else:
                    new_supps = None

                # default for removing inds info from supp
                if not np.all(
                    np.asarray(list(state.branches_supplimental.values())) == None
                ):
                    new_branch_supps = {
                        name: state.branches[name].branch_supplimental.take_along_axis(
                            all_inds_shaped[:, :, None], axis=1
                        )
                        for name in state.branches
                    }

                    new_branch_supps = {
                        name: BranchSupplimental(
                            new_branch_supps[name],
                            obj_contained_shape=new_inds[name].shape,
                            copy=False,
                        )
                        for name in new_branch_supps
                    }
                    for name in new_branch_supps:
                        new_branch_supps[name].add_objects(
                            {"inds_keep": new_inds[name]}
                        )

                else:
                    new_branch_supps = None
                    if new_supps is not None:
                        new_branch_supps = {
                            name: BranchSupplimental(
                                {"inds_keep": new_inds[name]},
                                obj_contained_shape=new_inds[name].shape,
                                copy=False,
                            )
                            for name in new_branch_supps
                        }

                # TODO: add supplimental prepare step
                # if (new_branch_supps is not None or new_supps is not None) and self.adjust_supps_pre_logl_func is not None:
                #    self.adjust_supps_pre_logl_func(q, inds=new_inds, logp=logp, supps=new_supps, branch_supps=new_branch_supps, inds_keep=new_inds_adjust)

                # Compute the lnprobs of the proposed position.
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=real_inds_subset,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

                if (
                    new_branch_supps is not None
                    and not np.all(np.asarray(list(new_branch_supps.values())) == None)
                ) or new_supps is not None:
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

                prev_logl = np.take_along_axis(state.log_like, all_inds_shaped, axis=1)

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
                    q,
                    log_like=logl,
                    log_prior=logp,
                    blobs=new_blobs,
                    inds=new_inds,
                    supplimental=new_supps,
                    branch_supplimental=new_branch_supps,
                )

                state = self.update(
                    state, new_state, accepted_here, subset=all_inds_shaped
                )

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted

