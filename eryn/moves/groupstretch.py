# -*- coding: utf-8 -*-
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .group import GroupMove
from .stretch import StretchMove

__all__ = ["GroupStretchMove"]


class GroupStretchMove(GroupMove):
    """Affine-Invariant Proposal

    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    Args:
        a (double, optional): The stretch scale parameter. (default: ``2.0``)

    Attributes:
        a (double): The stretch scale parameter.
    """

    def __init__(self, **kwargs):
        GroupStretchMove.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

    def adjust_factors(self, factors, ndims_old, ndims_new):
        # adjusts in place
        logzz = factors / (ndims_old - 1.0)
        factors[:] = logzz * (ndims_new - 1.0)

    def get_proposal(self, s_all, random, gibbs_ndim=None, **kwargs):
        """Generate stretch proposal

        # TODO: add log proposal from ptemcee

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            inds_s (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max] that indicate which leaves
                are currently being used in :code:`s_all`. (default: ``None``)
            inds_c (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max] that indicate which leaves
                are currently being used in :code:`s_all`. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        random_number_generator = random if not self.use_gpu else self.xp.random
        self.zz = None
        newpos = {}
        for i, name in enumerate(s_all):
            s = self.xp.asarray(s_all[name])

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape

            Ns = nwalkers
            c_temp = self.choose_c_vals(name, s)

            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            newpos[name] = self.get_new_points(
                name, s, c_temp, Ns, s.shape, i, random_number_generator
            )
        # proper factors
        factors = (ndim - 1.0) * self.xp.log(self.zz)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        if gibbs_ndim is not None:
            # adjust factors in place
            self.adjust_factors(factors, ndim, gibbs_ndim)

        return newpos, factors

