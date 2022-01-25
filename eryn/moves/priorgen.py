# -*- coding: utf-8 -*-

import numpy as np

from .move import Move
from ..prior import PriorContainer

__all__ = ["PriorGenerate"]

class PriorGenerate(Move):
    """Generate proposals from prior

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(self, priors, *args, **kwargs):

        for key in priors:
            if not isinstance(priors[key], PriorContainer):
                raise ValueError("Priors need to be eryn.priors.PriorContainer object.")
        self.priors = priors
        super(PriorGenerate, self).__init__(*args, **kwargs)

    def get_proposal(self, all_coords, inds, random):
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
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
        q        = {}
        factors  = {}
        new_inds = {}
        all_inds = inds
        if all_inds is None:
            all_inds = {name: np.ones(coords.shape[:-1], dtype=bool) for name, coords in all_coords}

        for i, (name, coords, inds) in enumerate(
            zip(
                all_coords.keys(),
                all_coords.values(),
                all_inds.values(),
            )
        ):
            ntemps, nwalkers, _, _ = coords.shape
            q[name] = coords.copy()
            new_inds[name] = inds.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # add coordinates for new leaves
            current_priors  = self.priors[name]
            inds_here       = np.where(inds == True)
            num_inds_change = len(inds_here[0])
            # Draw
            q[name][inds_here] = current_priors.rvs(size=num_inds_change)

            # factor is -log q()
            # factors[inds_here[:2]] += +1 * current_priors.logpdf(q[name][inds_here])

        return q, np.zeros((ntemps, nwalkers))
