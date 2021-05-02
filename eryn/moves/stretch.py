# -*- coding: utf-8 -*-

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """
    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    :param a: (optional)
        The stretch scale parameter. (default: ``2.0``)

    """

    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super(StretchMove, self).__init__(**kwargs)

    def get_proposal(self, s_all, c_all, random, inds=None):
        newpos = {}
        for i, name in enumerate(s_all):
            c = c_all[name]
            s = s_all[name]
            c = np.concatenate(c, axis=0)
            Ns, Nc = len(s), len(c)
            # gets rid of any values of exactly zero
            if inds is None:
                ndim_temp = s_all[name].shape[-1] * s_all[name].shape[-2]
            else:
                ndim_temp = inds[name].sum(axis=(1)) * s_all[name].shape[-1]
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns
                zz = ((self.a - 1.0) * random.rand(Ns) + 1) ** 2.0 / self.a
            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")
            rint = random.randint(Nc, size=(Ns,))
            newpos[name] = c[rint] - (c[rint] - s) * zz[:, None, None]

        factors = (ndim - 1.0) * np.log(zz)
        return newpos, factors
