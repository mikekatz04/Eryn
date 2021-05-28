# -*- coding: utf-8 -*-

import numpy as np

from .rj import ReversibleJump

__all__ = ["PriorGenerate"]


class PriorGenerate(ReversibleJump):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """

    def __init__(self, priors, *args, **kwargs):
        self.priors = priors
        super(PriorGenerate, self).__init__(*args, **kwargs)

    def get_proposal(self, all_coords, all_inds, all_inds_for_change, random):
        # wc: walker coords
        # wi: walker inds
        q = {}
        new_inds = {}
        factors = {}

        for i, (name, coords, inds, inds_for_change) in enumerate(
            zip(
                all_coords.keys(),
                all_coords.values(),
                all_inds.values(),
                all_inds_for_change.values(),
            )
        ):
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
            num_available = inds.sum(axis=-1)
            factors[inds_here[:2]] += +1 * np.log(1 / num_available[inds_here[:2]])

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
