# -*- coding: utf-8 -*-

import numpy as np

from .mh import MHMove

__all__ = ["GaussianMove"]


def ensure_sphere_boundary(costheta, phi):
    """This function makes sure that if theta is proposed outside of the boundary 
    we need to flip phi, the ranges are (theta in 0 pi), (phi in 0 2pi)"""
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    new_theta = np.arccos(z/np.sqrt(x*x + y*y+z*z))
    new_phi = np.sign(y)*np.arccos(x/np.sqrt(x*x + y*y))
    mask = (new_phi < 0.0)
    new_phi[mask] = new_phi[mask] + 2*np.pi
    return np.cos(new_theta), new_phi


def reflect_cosines_array(cos_ins,angle_ins,rotfac=np.pi,modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
        which requires also rotating the signal by rotfac each time, then mod the angle by modfac"""
    for itrk in range(cos_ins.size):
        if cos_ins[itrk] < -1.:
            cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
            angle_ins[itrk] += rotfac
        if cos_ins[itrk] > 1.:
            cos_ins[itrk] = 1.-(cos_ins[itrk]-1.)%4
            angle_ins[itrk] += rotfac
            #if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
            if cos_ins[itrk] < -1.:
                cos_ins[itrk] = -1.+(-(cos_ins[itrk]+1.))%4
                angle_ins[itrk] += rotfac
        angle_ins[itrk] = angle_ins[itrk]%modfac
    return cos_ins,angle_ins


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

    def __init__(self, cov_all, mode="AM", factor=None, priors=None, indx_list=None, swap_walkers=None, sky_periodic=None, **kwargs):

        self.all_proposal = {}
        
        for name, cov in cov_all.items():
            # Parse the proposal type.
            try:
                float(cov)

            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    # A diagonal proposal was given.
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, "vector")

                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                    ndim = cov.shape[0]
                    
                    if mode=="Gaussian":
                        proposal = _proposal(cov, factor,"vector")
                    if mode=="AM":
                        proposal = AM_proposal(cov, factor, "vector")

                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor,  "vector")
            self.all_proposal[name] = proposal

        # priors to draw from
        self.priors = priors
        # swap walkers
        self.swap_walkers = swap_walkers
        # propose in blocks
        self.indx_list = indx_list
        
        self.sky_periodic = sky_periodic
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
            # draw from the prior 10% of the time
            new_coords_tmp = coords[inds_here].copy()
            new_coords = coords[inds_here].copy()
            
            if self.priors is not None:
                # if np.random.uniform()>0.9:
                for var in range(new_coords.shape[-1]):
                    new_coords_tmp[:,var] = self.priors[name][var].rvs(size=new_coords[:,var].shape[0])
            else:
                new_coords_tmp = proposal_fn(coords[inds_here], random)[0]
            
            # swap walkers, this helps for the search phase
            if self.indx_list is not None:
                indx_list_here = np.asarray([el[1] for el in self.indx_list if el[0]==name])
                nw = new_coords_tmp.shape[0]
                # list of numbers indicating wich group of parameters to change
                ind_to_chage = np.random.randint(len(indx_list_here),size=nw)
                new_coords[indx_list_here[ind_to_chage][:,0,:]] = new_coords_tmp[indx_list_here[ind_to_chage][:,0,:]]
            else:
                new_coords = new_coords_tmp.copy()
            
            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                nw = new_coords_tmp.shape[0]
                for temp_ind in range(len(indx_list_here)):
                    csth = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,0]
                    ph = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,1]
                    new_coords[:,indx_list_here[temp_ind][0]] = np.asarray(reflect_cosines_array(csth, ph)).T
                
            # jump in frequency
            # if np.random.uniform()>0.9:
            #     shape = new_coords[...,2].shape
            #     new_coords[...,2] += np.sign(np.random.uniform(-1,1))*np.ones(shape)*np.log10(np.random.randint(1,4,size=shape))
            #     new_coords[...,3] += np.sign(np.random.uniform(-1,1))*np.ones(shape)*np.log10(np.random.randint(1,4,size=shape))

            # swap walkers, this helps for the search phase
            if self.swap_walkers is not None:
                if np.random.uniform()>self.swap_walkers:
                    ind_shuffle = np.arange(new_coords.shape[0])
                    np.random.shuffle(ind_shuffle)
                    new_coords = new_coords[ind_shuffle].copy()
            
            

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                q[name] = self.periodic.wrap({name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)})
                q[name] = tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros((ntemps, nwalkers))


class _isotropic_proposal(object):

    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        self.svd = None
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



def propose_AM(x0, rng, svd, scale):
    """
    Adaptive Jump Proposal
    """
    new_pos = x0.copy()
    nw, nd = new_pos.shape
    U, S, v = svd

    # adjust step size
    prob = rng.random()

    # # large jump
    # if prob > 0.97:
    #     scale = 10.0

    # # small jump
    # elif prob > 0.9:
    #     scale = 0.2

    # # standard medium jump
    # else:
    #     scale = 1.0
    
    
    # go in eigen basis
    y = np.dot(U.T,x0.T).T # np.asarray([np.dot(U.T, x0[i]) for i in range(nw)])
    # choose a random parameter in the uncorrelated basis
    ind_vec = np.arange(nd)
    

    if np.random.uniform()>0.5:
        # move along one component
        np.random.shuffle(ind_vec)
        rand_j = ind_vec[:1]
    else:
        # move along all
        rand_j = ind_vec
    
    y[:,rand_j] += scale * np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(nd)
    # go back to the basis
    # if np.random.uniform()>0.5:
    new_pos = np.dot(U,y.T).T # np.asarray([np.dot(U, y[i]) for i in range(nw)]) 

    return new_pos


class AM_proposal(_isotropic_proposal):

    allowed_modes = ["vector"]
    
    def get_updated_vector(self, rng, x0):
        if self.svd is None:
            svd = np.linalg.svd(self.scale)
        else:
            svd = self.svd
        return propose_AM(x0, rng, svd, self.get_factor(rng))