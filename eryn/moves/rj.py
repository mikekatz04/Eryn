# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move
from .delayedrejection import DelayedRejection

__all__ = ["RedBlueMove"]


class ReversibleJump(Move):
    """
    An abstract reversible jump move from # TODO: add citations.

    Args:
        max_k (int or list of int): Maximum number(s) of leaves for each model.
        min_k (int or list of int): Minimum number(s) of leaves for each model.
        tune (bool, optional): If True, tune proposal. (Default: ``False``)

    """

    def __init__(
        self, max_k, min_k, proposal=None, dr=None, dr_max_iter=5, tune=False, **kwargs
    ):
        super(ReversibleJump, self).__init__(**kwargs)

        if isinstance(max_k, int):
            max_k = [max_k]

        if isinstance(max_k, int):
            min_k = [min_k]

        self.max_k = max_k
        self.min_k = min_k
        self.tune = tune
        self.dr = dr

        # Decide if DR is desirable. TODO: Now it uses the prior generator, we need to
        # think carefully if we want to use the in-model sampling proposal
        if self.dr:
            if self.dr is True:
                if proposal is None:
                    raise ValueError("If dr==True, must provide proposal kwarg.")

                self.dr = DelayedRejection(proposal, max_iter=dr_max_iter)

        # TODO: add stuff here if needed like prob of birth / death

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

        ntemps, nwalkers, _, _ = state.branches[list(state.branches.keys())[0]].shape

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # TODO: do we want an probability that the model count will not change?
        inds_for_change = {}
        for (name, branch), min_k, max_k in zip(
            state.branches.items(), self.min_k, self.max_k
        ):
            nleaves = branch.nleaves
            # choose whether to add or remove
            change = model.random.choice([-1, +1], size=nleaves.shape)

            # fix edge cases
            change = (
                change * ((nleaves != min_k) & (nleaves != max_k))
                + (+1) * (nleaves == min_k)
                + (-1) * (nleaves == max_k)
            )

            # setup storage for this information
            inds_for_change[name] = {}
            num_increases = np.sum(change == +1)
            inds_for_change[name]["+1"] = np.zeros((num_increases, 3), dtype=int)
            num_decreases = np.sum(change == -1)
            inds_for_change[name]["-1"] = np.zeros((num_decreases, 3), dtype=int)

            # TODO: not loop ? Is it necessary?
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
                    else:
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

        # propose new sources and coordinates
        q, new_inds, factors = self.get_proposal(
            state.branches_coords, state.branches_inds, inds_for_change, model.random,
        )

        # TODO: check this
        edge_factors = np.zeros((ntemps, nwalkers))
        # get factors for edges
        for (name, branch), min_k, max_k in zip(
            state.branches.items(), self.min_k, self.max_k
        ):
            nleaves = branch.nleaves

            # fix proposal asymmetry at bottom of k range
            inds_min = np.where(nleaves == min_k)
            # numerator term so +ln
            edge_factors[inds_min] += np.log(1 / 2.0)

            # fix proposal asymmetry at top of k range
            inds_max = np.where(nleaves == max_k)
            # numerator term so -ln
            edge_factors[inds_max] -= np.log(1 / 2.0)

        factors += edge_factors

        # Compute prior of the proposed position
        logp = model.compute_log_prior_fn(q, inds=new_inds)
        # Compute the lnprobs of the proposed position.
        logl, new_blobs = model.compute_log_prob_fn(q, inds=new_inds, logp=logp)

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
        new_state = State(q, log_prob=logl, log_prior=logp, blobs=None, inds=new_inds)
        state = self.update(state, new_state, accepted)

        # apply delayed rejection to walkers that are +1
        if self.dr:
            # for name, branch in state.branches.items():
            #     # We have to work with the binaries added only.
            #     # We need the a) rejected points, b) the model,
            #     # c) the current state, d) the indices where we had +1 (True),
            #     # and the e) factors.
            #     # breakpoint()
            state, accepted = self.dr.propose(
                accepted, model, state, new_inds, factors
            )  # model, state

        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        return state, accepted
