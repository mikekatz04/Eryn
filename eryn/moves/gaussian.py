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

    def __init__(self, cov_all, mode="AM", factor=None, indx_list=None, sky_periodic=None, shift_value=None, **kwargs):

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
                    if mode=="DE":
                        proposal = DE_proposal(cov, factor, "vector")
                        

                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor,  "vector")
            self.all_proposal[name] = proposal

        # propose in blocks
        self.indx_list = indx_list
        # ensure sky periodicity
        self.sky_periodic = sky_periodic
        # add random shift (how often, param index as in self.indx_list, value to shift)
        self.shift_value = shift_value
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
            new_coords_tmp = coords[inds_here].copy()
            new_coords = coords[inds_here].copy()
            
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
            
            # shift
            if self.shift_value is not None:
                # first value tells how often to shift, self.shift_value[0]=1, always
                if np.random.uniform()<self.shift_value[0]:
                    indx_list_here = np.asarray([el[1] for el in self.shift_value[1] if el[0]==name])
                    nw = new_coords_tmp.shape[0]
                    # list of numbers indicating wich group of parameters to change
                    ind_to_chage = np.random.randint(len(indx_list_here),size=nw)
                    random_number = np.random.choice([-1, 1])
                    # add value
                    new_coords[indx_list_here[ind_to_chage][:,0,:]] += random_number*self.shift_value[2]

            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                nw = new_coords_tmp.shape[0]
                for temp_ind in range(len(indx_list_here)):
                    csth = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,0]
                    ph = new_coords_tmp[:,indx_list_here[temp_ind][0]][:,1]
                    new_coords[:,indx_list_here[temp_ind][0]] = np.asarray(reflect_cosines_array(csth, ph)).T
                

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
        self.chain = None
        self.invscale = np.linalg.inv(np.linalg.cholesky(scale))
        self.use_current_state = True
        self.crossover = False
        
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

def propose_AM(x0, rng, svd, scale):
    """
    Adaptive Jump Proposal.
    Single Component Adaptive Jump Proposal.
    """
    new_pos = x0.copy()
    nw, nd = new_pos.shape
    U, S, v = svd

    # adjust step size
    prob = rng.random()
    
    # go in eigen basis
    y = np.dot(U.T,x0.T).T # np.asarray([np.dot(U.T, x0[i]) for i in range(nw)])
    # choose a random parameter in the uncorrelated basis
    ind_vec = np.arange(nd)
    
    if prob>0.5:
        # move along only one uncorrelated direction SCAM
        np.random.shuffle(ind_vec)
        rand_j = ind_vec[:1]
    else:
        # move along all of them AM
        rand_j = ind_vec
    
    y[:,rand_j] += scale * np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(nd)
    
    # go back to the basis
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


def propose_DE(current_state, chain, F=0.5, CR=0.9, use_current_state=True, crossover=False):
    """
    Provides a proposal for MCMC using Differential Evolution (DE/rand/1).

    Parameters:
        current_state (numpy.ndarray): The current state of the MCMC chain. Shape: (n_walkers, n_params).
        chain (numpy.ndarray): The chain from which to take the mutant. Shape: (n_mutants, n_params).
        F (float): The differential weight (default is 0.5), in [0,2].
        CR (float): The crossover probability (default is 0.9), in [0,1].

    Returns:
        numpy.ndarray: The proposed state. Shape: (n_walkers, n_params).
    """
    n_walkers, n_params = current_state.shape

    # Randomly select three distinct indices for each walker
    indices = np.random.choice(chain.shape[0], size=(chain.shape[0], 3), replace=True)
    
    # Generate mutant vectors using DE/rand/1
    if use_current_state:
        mutant_vectors = current_state + F * (current_state[indices[:, 1]] - current_state[indices[:, 2]])
    else:
        mutant_vectors = chain[indices[:, 0]] + F * (chain[indices[:, 1]] - chain[indices[:, 2]])

    # Perform crossover with the current state to create the proposed state
    if crossover:
        crossover_mask = (np.random.rand(n_walkers, n_params) <= CR) | (np.arange(n_params) == np.random.randint(n_params, size=(n_walkers, 1)))
    else:
        # to update all
        crossover_mask = np.ones((n_walkers, n_params), dtype=bool)
    proposed_state = np.where(crossover_mask, mutant_vectors, current_state)
    

    return proposed_state

class DE_proposal(_isotropic_proposal):

    allowed_modes = ["vector"]
    
    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp( rng.uniform( -self._log_factor, 0.0 ) )
    
    def get_updated_vector(self, rng, x0):
        # get jump scale size
        prob = rng.random()

        # scaling
        if prob > 0.5:
            # random in range
            F = self.get_factor(rng)
            CR = np.random.uniform(0.5,1.0)
        else:
            # default
            F = 0.5
            CR = 0.9
        
        if self.chain is None:
            # use current state to update
            return propose_DE(x0, x0.copy(), F=F, CR=CR, use_current_state=self.use_current_state, crossover=self.crossover)
        else:
            # take from the pool
            return propose_DE(x0, self.chain, F=F, CR=CR, use_current_state=self.use_current_state, crossover=self.crossover)
