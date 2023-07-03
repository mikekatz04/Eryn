# -*- coding: utf-8 -*-

import numpy as np

from .mh import MHMove

__all__ = ["GaussianMove"]


class GaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    This class is heavily based on the same class in ``emcee``. 

    Args:
        cov (dict): The covariance of the proposal function. The keys are branch names and the 
            values are covariance information. This information can be provided as a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general, respectively.
        mode (str, optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn. (default: ``"vector"``)
        factor (float, optional): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode. (default: ``None``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(self, cov_all, mode="vector", factor=None, priors=None, indx_list=None, swap_walkers=None, **kwargs):

        self.all_proposal = {}
        self.indx_list = indx_list
        for name, cov in cov_all.items():
            # Parse the proposal type.
            try:
                float(cov)

            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    # A diagonal proposal was given.
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)

                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                    ndim = cov.shape[0]
                    proposal = _proposal(cov, factor, mode)#eigproposal(cov) #

                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)
            self.all_proposal[name] = proposal

        self.priors = priors
        self.swap_walkers = swap_walkers
        super(GaussianMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Get proposal from Gaussian distribution

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)

        """

        # initialize ouput
        q = {}
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords_tmp = proposal_fn(coords[inds_here], random)[0]
            new_coords = coords[inds_here].copy()

            if self.indx_list is not None:
                nw = new_coords_tmp.shape[0]
                for i in range(nw):
                    temp_ind = np.random.randint(len(self.indx_list))
                    new_coords[i,self.indx_list[temp_ind]] = new_coords_tmp[i,self.indx_list[temp_ind]]
            else:
                new_coords = new_coords_tmp.copy()
            
            
            if self.swap_walkers is not None:
                if np.random.uniform()<0.1:
                    ind_shuffle = np.arange(new_coords.shape[0])
                    np.random.shuffle(ind_shuffle)
                    new_coords = new_coords[ind_shuffle].copy()
            
            # if self.priors is not None:
            #     if np.random.uniform()>0.9:
            #         for var in range(new_coords.shape[-1]):
            #             new_coords[:,var] = self.priors[name][var].rvs(size=new_coords[:,var].shape[0])

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)
                    for name, tmp in q.items()
                },
                # xp=self.xp,
            )

            q = {
                name: tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)
                for name, tmp in q.items()
            }

        return q, np.zeros((ntemps, nwalkers))


class _isotropic_proposal(object):

    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        self.invscale = np.linalg.inv(np.linalg.cholesky(scale))
        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                ("'{0}' is not a recognized mode. " "Please select from: {1}").format(
                    mode, self.allowed_modes
                )
            )
        self.mode = mode

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):
    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))


class _proposal(_isotropic_proposal):

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale, size=len(x0)
        )

class eigproposal():
    def __init__(self, cov):
        self.w,self.v = np.linalg.eig(cov)

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        factors = rng.uniform(size=nw)
        ind = rng.randint(nd,size=nw)

        return x0 + (factors[None,:] * self.v[:,ind] / self.w[ind]).T, 1
