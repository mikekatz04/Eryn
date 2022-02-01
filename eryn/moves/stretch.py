# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
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

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchMove, self).__init__(**kwargs)

    def adjust_factors(self, factors, ndims_old, ndims_new):
        # adjusts in place
        logzz = factors / (ndims_old - 1.0) 
        factors[:] = logzz * (ndims_new - 1.0)

    def get_proposal(self, s_all, c_all, random, inds_s=None, inds_c=None, **kwargs):
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
        newpos = {}
        for i, name in enumerate(s_all):
            c = c_all[name]
            s = s_all[name]
            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            c = np.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            if inds_s is None:
                ndim_temp = s_all[name].shape[-1] * s_all[name].shape[-2]
            else:
                ndim_temp = inds_s[name].sum(axis=(2)) * s_all[name].shape[-1]
            
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns
                zz = ((self.a - 1.0) * random.rand(ntemps, Ns) + 1) ** 2.0 / self.a
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            rint = random.randint(Nc, size=(ntemps, Ns,))
            c_temp = np.take_along_axis(c, rint[:, :, None, None], axis=1)

            if self.periodic is not None:
                diff = self.periodic.distance(
                    s.reshape(ntemps * nwalkers, nleaves_max, ndim_here), 
                    c_temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here), 
                    names=[name]
                )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)
            else:
                diff = c_temp - s

            temp = c_temp - (diff) * zz[:, :, None, None]

            if self.periodic is not None:
                temp = self.periodic.wrap(temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here), names=[name])[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

            newpos[name] = temp

        # proper factors
        factors = (ndim - 1.0) * np.log(zz)
        return newpos, factors

