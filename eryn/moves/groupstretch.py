# -*- coding: utf-8 -*-
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .group import GroupMove

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

    def __init__(self, a=2.0, use_gpu=False, return_gpu=False, random_seed=None, **kwargs):
        self.a = a
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.use_gpu = use_gpu
        self.return_gpu = return_gpu

        super(GroupStretchMove, self).__init__(**kwargs)

    def adjust_factors(self, factors, ndims_old, ndims_new):
        # adjusts in place
        logzz = factors / (ndims_old - 1.0) 
        factors[:] = logzz * (ndims_new - 1.0)

    def get_proposal(self, s_all, random, **kwargs):
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

        newpos = {}
        for i, name in enumerate(s_all):
            if i > 0:
                # TODO: check?
                raise NotImplementedError

            s = self.xp.asarray(s_all[name])

            nwalkers, ndim_here = s.shape

            Ns, Nc = nwalkers, c.shape[-2]
            
            ndim = ndim_here
            zz = ((self.a - 1.0) * random_number_generator.rand(nwalkers) + 1) ** 2.0 / self.a

            c_temp = self.choose_c_vals(name, s, c=None)

            if self.periodic is not None:
                diff = self.periodic.distance(
                    s.reshape(nwalkers, 1, ndim_here), 
                    c_temp.reshape(nwalkers, 1, ndim_here), 
                    names=[name],
                    xp=self.xp
                )[name].reshape(nwalkers, ndim_here)
            else:
                diff = c_temp - s

            temp = c_temp - (diff) * zz[:, None]

            if self.periodic is not None:
                temp = self.periodic.wrap(temp.reshape(nwalkers, 1, ndim_here), names=[name], xp=self.xp)[name].reshape(nwalkers, ndim_here)

            newpos[name] = temp

        # proper factors
        factors = (ndim - 1.0) * self.xp.log(zz)
        return newpos, factors

