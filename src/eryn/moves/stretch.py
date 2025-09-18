# -*- coding: utf-8 -*-
try:
    import cupy as cp
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
        return_gpu (bool, optional): If ``use_gpu == True and return_gpu == True``,
            the returned arrays will be returned as ``CuPy`` arrays. (default: ``False``)
        kwargs (dict, optional): Additional keyword arguments passed down through :class:`RedRedBlueMove`_.

    Attributes:
        a (double): The stretch scale parameter.
        return_gpu (bool): Whether the array being returned is in ``Cupy`` (``True``)
            or ``NumPy`` (``False``).

    """

    def __init__(self, a=2.0, return_gpu=False, random_seed=None, **kwargs):
        # store scale factor
        self.a = a

        # pass kwargs up
        RedBlueMove.__init__(self, **kwargs)

        # change array library based on GPU usage

        # set the random seet of the library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

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
        if ndims_old == ndims_new:
            return
        logzz = factors / (ndims_old - 1.0)
        factors[:] = logzz * (ndims_new - 1.0)

    def choose_c_vals(self, c, Nc, Ns, ntemps, random_number_generator, **kwargs):
        """Get the compliment array

        The compliment represents the points that are used to move the actual points whose position is
        changing.

        Args:
            c (np.ndarray): Possible compliment values with shape ``(ntemps, Nc, nleaves_max, ndim)``.
            Nc (int): Length of the ``...``: the subset of walkers proposed to move now (usually nwalkers/2).
            Ns (int): Number of generation points.
            ntemps (int): Number of temperatures.
            random_number_generator (object): Random state object.
            **kwargs (ignored): Ignored here. For modularity.

        Returns:
            np.ndarray: Compliment values to use with shape ``(ntemps, Ns, nleaves_max, ndim)``.

        """

        rint = random_number_generator.randint(
            Nc,
            size=(
                ntemps,
                Ns,
            ),
        )
        c_temp = self.xp.take_along_axis(c, rint[:, :, None, None], axis=1)
        return c_temp

    def get_new_points(
        self, name, s, c_temp, Ns, branch_shape, branch_i, random_number_generator
    ):
        """Get mew points in stretch move.

        Takes compliment and uses it to get new points for those being proposed.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c (np.ndarray): Compliment to move points with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            Ns (int): Number to generate.
            branch_shape (tuple): Full branch shape.
            branch_i (int): Which branch in the order is being run now. This ensures that the
                randomly generated quantity per walker remains the same over branches.
            random_number_generator (object): Random state object.

        Returns:
            np.ndarray: New proposed points with shape ``(ntemps, Ns, nleaves_max, ndim)``.


        """

        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape

        # only for the first branch do we draw for zz
        if branch_i == 0:
            self.zz = (
                (self.a - 1.0) * random_number_generator.rand(ntemps, Ns) + 1
            ) ** 2.0 / self.a

        # get proper distance

        if self.periodic is not None:
            diff = self.periodic.distance(
                {name: s.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                {name: c_temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)
        else:
            diff = c_temp - s

        temp = c_temp - (diff) * self.zz[:, :, None, None]

        # wrap periodic values

        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
        return temp

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate stretch proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """

        # needs to be set before we reach the end
        self.zz = None
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            # get compliment possibilities
            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            # need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # get actual compliment values
            c_temp = self.choose_c_vals(c, Nc, Ns, ntemps, random_number_generator)

            # use stretch to get new proposals
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
