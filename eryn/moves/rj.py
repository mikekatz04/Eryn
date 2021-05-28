# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["RedBlueMove"]


class ReversibleJump(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.

        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.

    """

    def __init__(self, max_k, min_k, tune=False, **kwargs):
        super(ReversibleJump, self).__init__(**kwargs)
        self.max_k = max_k
        self.min_k = min_k
        self.tune = tune

        # TODO: add stuff here if needed like prob of birth / death

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

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
            change = model.random.choice([-1, +1], size=nleaves.shape)

            change = (
                change * ((nleaves != min_k) & (nleaves != max_k))
                + (+1) * (nleaves == min_k)
                + (-1) * (nleaves == max_k)
            )

            inds_for_change[name] = {}
            num_increases = np.sum(change == +1)
            inds_for_change[name]["+1"] = np.zeros((num_increases, 3), dtype=int)
            num_decreases = np.sum(change == -1)
            inds_for_change[name]["-1"] = np.zeros((num_decreases, 3), dtype=int)

            # TODO: not loop ? Is it necessary?
            increase_i = 0
            decrease_i = 0
            for t in range(ntemps):
                for w in range(nwalkers):
                    change_tw = change[t][w]
                    inds_tw = branch.inds[t][w]

                    if change_tw == +1:
                        inds_false = np.where(inds_tw == False)[0]
                        ind_change = model.random.choice(inds_false)
                        inds_for_change[name]["+1"][increase_i] = np.array(
                            [t, w, ind_change], dtype=int
                        )
                        increase_i += 1

                        # TODO: change this so we can actually generate new coords rather than reuse old standing coords
                        # q[name][t, w, ind_change, :] = branch.coords[
                        #    t, w, ind_change, :
                        # ]

                    else:
                        # change_tw == -1
                        inds_true = np.where(inds_tw == True)[0]
                        ind_change = model.random.choice(inds_true)
                        inds_for_change[name]["-1"][decrease_i] = np.array(
                            [t, w, ind_change], dtype=int
                        )
                        decrease_i += 1
                        # do not care currently about what we do with discarded coords, they just sit in the state

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

        # TODO: deal with accepted (how to track outside the accepted parts, maybe separate rj from other proposal)
        if self.temperature_control is not None:
            state, accepted = self.temperature_control.temper_comps(state, accepted)

        # make accepted move specific ?
        return state, accepted
