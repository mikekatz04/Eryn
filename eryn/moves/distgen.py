# -*- coding: utf-8 -*-

import numpy as np
from .mh import MHMove
from ..prior import ProbDistContainer

__all__ = ["DistributionGenerate"]


class DistributionGenerate(MHMove):
    """Generate proposals from a distribution

    Args:
        generate_dist (object): :class:`ProbDistContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(self, generate_dist, *args, **kwargs):

        for key in generate_dist:
            if not isinstance(generate_dist[key], ProbDistContainer):
                raise ValueError(
                    "Distributions need to be eryn.prior.ProbDistContainer object."
                )
        self.generate_dist = generate_dist
        super(DistributionGenerate, self).__init__(*args, **kwargs)

    def get_proposal(self, all_coords, all_inds, random, **kwargs):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        # set up all initial holders
        q = {}
        factors = {}
        new_inds = {}
        if all_inds is None:
            all_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in all_coords
            }

        # iterate through branches and propose new points where inds == True
        for i, (name, coords, inds) in enumerate(
            zip(all_coords.keys(), all_coords.values(), all_inds.values(),)
        ):
            # copy over previous info
            ntemps, nwalkers, _, _ = coords.shape
            q[name] = coords.copy()
            new_inds[name] = inds.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # add coordinates for new leaves
            current_generate_dist = self.generate_dist[name]
            inds_here = np.where(inds == True)
            num_inds_change = len(inds_here[0])

            old_points = coords[inds_here]

            # old points so + log(qold)
            factors[inds_here[:2]] += +1 * current_generate_dist.logpdf(old_points)

            # Draw
            new_points = current_generate_dist.rvs(size=num_inds_change)

            # new point, so -log(qnew)
            factors[inds_here[:2]] += -1 * current_generate_dist.logpdf(new_points)

            q[name][inds_here] = new_points

        return q, factors
