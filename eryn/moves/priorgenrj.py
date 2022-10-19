# -*- coding: utf-8 -*-

import numpy as np

from .rj import ReversibleJump
from ..prior import PriorContainer

__all__ = ["PriorGenerateRJ"]

class PriorGenerateRJ(ReversibleJump):
    """Generate Revesible-Jump proposals from prior

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """
    def __init__(self, priors, *args, **kwargs):

        for key in priors:
            if not isinstance(priors[key], PriorContainer):
                raise ValueError("Priors need to be eryn.priors.PriorContainer object.")
        self.priors = priors

        super(PriorGenerateRJ, self).__init__(*args, **kwargs)

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random, **kwargs):
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

        """
        q = {}
        new_inds = {}

        for i, name in enumerate(all_inds_for_change.keys()):

            coords=all_coords[name]
            inds=all_inds[name]
            inds_for_change=all_inds_for_change[name]
            
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
            current_priors = self.priors[name]
            factors[inds_here[:2]] += +1 * current_priors.logpdf(q[name][inds_here])

            # adjust births from False -> True
            inds_here = tuple(inds_for_change["+1"].T)
            new_inds[name][inds_here] = True

            # add coordinates for new leaves
            current_priors = self.priors[name]
            inds_here = tuple(inds_for_change["+1"].T)
            num_inds_change = len(inds_here[0])

            # TODO: Add the possibility of drawing from other distributions than priors (new class)
            q[name][inds_here] = current_priors.rvs(size=num_inds_change)

            # factor is -log q()
            factors[inds_here[:2]] += -1 * current_priors.logpdf(q[name][inds_here])

        return q, new_inds, factors
