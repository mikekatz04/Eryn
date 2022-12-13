# -*- coding: utf-8 -*-

import numpy as np

from .rj import ReversibleJump
from ..prior import ProbDistContainer

__all__ = ["DistributionGenerateRJ"]


class DistributionGenerateRJ(ReversibleJump):
    """Generate Revesible-Jump proposals from a distribution.

    The prior can be entered as the ``generate_dist`` to generate proposals directly from the prior.

    Args:
        generate_dist (dict): Keys are branch names and values are :class:`ProbDistContainer` objects 
            that have ``logpdf`` and ``rvs`` methods. If you 
        *args (tuple, optional): Additional arguments to pass to parent classes.
        **kwargs (dict, optional): Keyword arguments passed to parent classes.

    """

    def __init__(self, generate_dist, *args, **kwargs):

        # make sure all inputs are distribution Containers
        for key in generate_dist:
            if not isinstance(generate_dist[key], ProbDistContainer):
                raise ValueError(
                    "Distributions need to be eryn.prior.ProbDistContiner object."
                )
        self.generate_dist = generate_dist

        super(DistributionGenerateRJ, self).__init__(*args, **kwargs)

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

        """
        # prepare the output dictionaries
        q = {}
        new_inds = {}
        all_inds_for_change = {}

        # loop over the models included here
        assert len(min_k_all)
        assert len(all_coords.keys()) == len(max_k_all)
        for (name, inds), min_k, max_k in zip(all_inds.items(), min_k_all, max_k_all):
            if min_k == max_k:
                continue
            elif min_k > max_k:
                raise ValueError("min_k is greater than max_k. Not allowed.")

            # get the inds adjustment information
            all_inds_for_change[name] = self.get_model_change_proposal(
                inds, random, min_k, max_k
            )

        # loop through branches and propose new points from the prio
        for i, (name, coords, inds) in enumerate(
            zip(all_coords.keys(), all_coords.values(), all_inds.values(),)
        ):
            # if not included
            if name not in all_inds_for_change:
                continue

            # inds changing for this branch
            inds_for_change = all_inds_for_change[name]

            # put in base information
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            new_inds[name] = inds.copy()
            q[name] = coords.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # adjust inds

            # adjust deaths from True -> False
            inds_here = tuple(inds_for_change["-1"].T)
            new_inds[name][inds_here] = False

            # factor is +log q()
            current_generate_dist = self.generate_dist[name]
            factors[inds_here[:2]] += +1 * current_generate_dist.logpdf(
                q[name][inds_here]
            )

            # adjust births from False -> True
            inds_here = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_here] = True

            # add coordinates for new leaves
            current_generate_dist = self.generate_dist[name]
            inds_here = tuple(inds_for_change["+1"].T)
            num_inds_change = len(inds_here[0])

            q[name][inds_here] = current_generate_dist.rvs(size=num_inds_change)

            # factor is -log q()
            factors[inds_here[:2]] += -1 * current_generate_dist.logpdf(
                q[name][inds_here]
            )

        return q, new_inds, factors
