# -*- coding: utf-8 -*-
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    pass

import numpy as np

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """Affine-Invariant Proposal

    A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    This class was originally implemented in ``emcee``.

    Args:
        a (double, optional): The stretch scale parameter. (default: ``2.0``)
        use_gpu (bool, optional): If ``True``, use ``CuPy`` for computations. 
            Use ``NumPy`` if ``use_gpu == False``. (default: ``False``)
        return_gpu (bool, optional): If ``use_gpu == True and return_gpu == True``, 
            the returned arrays will be returned as ``CuPy`` arrays. (default: ``False``)
        random_seed (int, optional): Set the random seed in ``CuPy/NumPy`` if not ``None``.
            (default: ``None``)
        kwargs (dict, optional): Additional keyword arguments passed down through :class:`RedRedBlueMove`_.

    Attributes:
        a (double): The stretch scale parameter.
        xp (obj): ``NumPy`` or ``CuPy``.
        use_gpu (bool): Whether ``Cupy`` (``True``) is used or not (``False``). 
        return_gpu (bool): Whether the array being returned is in ``Cupy`` (``True``) 
            or ``NumPy`` (``False``).
        
    """

    def __init__(
        self, a=2.0, use_gpu=False, return_gpu=False, random_seed=None, **kwargs
    ):

        # store scale factor
        self.a = a

        # change array library based on GPU usage
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        # set the random seet of the library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.use_gpu = use_gpu
        self.return_gpu = return_gpu

        # pass kwargs up
        RedBlueMove.__init__(self, **kwargs)

        # how it was formerly
        # super(StretchMove, self).__init__(**kwargs)

    def adjust_factors(self, factors, ndims_old, ndims_new):
        """Adjust the ``factors`` based on changing dimensions. 

        ``factors`` is adjusted in place.

        Args: 
            factors (xp.ndarray): Array of ``factors`` values. It is adjusted in place.
            ndims_old (int or xp.ndarray): Old dimension. If given as an ``xp.ndarray``,
                must be broadcastable with ``factors``.
            ndims_new (int or xp.ndarray): New dimension. If given as an ``xp.ndarray``,
                must be broadcastable with ``factors``.  
        
        """
        # adjusts in place
        logzz = factors / (ndims_old - 1.0)
        factors[:] = logzz * (ndims_new - 1.0)

    def choose_c_vals(self, c, Nc, Ns, random_number_generator, **kwargs):
        rint = random_number_generator.randint(Nc, size=(Ns,))
        c_temp = self.xp.take_along_axis(c, rint[:, None, None], axis=1)
        return c_temp

    def get_new_points(
        self, name, s, c_temp, Ns, branch_shape, branch_i, random_number_generator
    ):
        nwalkers, nleaves_max, ndim_here = branch_shape

        if branch_i == 0:
            self.zz = (
                (self.a - 1.0) * random_number_generator.rand(Ns) + 1
            ) ** 2.0 / self.a

        if self.periodic is not None:
            diff = self.periodic.distance(s, c_temp, names=[name], xp=self.xp,)[name]
        else:
            diff = c_temp - s

        temp = c_temp - (diff) * self.zz[:, None, None]

        if self.periodic is not None:
            temp = self.periodic.wrap(temp, names=[name], xp=self.xp,)[name]

        if self.use_gpu and not self.return_gpu:
            temp = temp.get()

        return temp

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate stretch proposal

        # TODO: add log proposal from ptemcee

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            
        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        # needs to be set before we reach the end
        self.zz = None
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}
        for i, name in enumerate(s_all):
            s = self.xp.asarray(s_all[name])

            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            nwalkers, nleaves_max, ndim_here = s.shape
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            c_temp = self.choose_c_vals(c, Nc, Ns, random_number_generator)

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

