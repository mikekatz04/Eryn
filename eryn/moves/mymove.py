# -*- coding: utf-8 -*-

import numpy as np

from .mh import MHMove

__all__ = ["MyMove"]


class MyMove(MHMove):
    """A Metropolis step with a Gaussian proposal function.

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            vector, or matrix and the proposal will be assumed isotropic,
            axis-aligned, or general respectively.
        mode (Optional): Select the method used for updating parameters. This
            can be one of ``"vector"``, ``"random"``, or ``"sequential"``. The
            ``"vector"`` mode updates all dimensions simultaneously,
            ``"random"`` randomly selects a dimension and only updates that
            one, and ``"sequential"`` loops over dimensions and updates each
            one in turn.
        factor (Optional[float]): If provided the proposal will be made with a
            standard deviation uniformly selected from the range
            ``exp(U(-log(factor), log(factor))) * cov``. This is invalid for
            the ``"vector"`` mode.

    Raises:
        ValueError: If the proposal dimensions are invalid or if any of any of
            the other arguments are inconsistent.

    """

    def __init__(self, cov_all, prop_func=None, sky_periodic=None, **kwargs):

        self.all_proposal = {}
        self.sky_per = sky_periodic
        for name, cov in cov_all.items():
            # Parse the proposal type.
            if prop_func is not None:
                self.all_proposal[name] = prop_func
            else:
                self.all_proposal[name] = prop.walk

        super(MyMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, **kwargs):
        """Get proposal from Gaussian distribution

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            branches_inds (dict): Keys are ``branch_names`` and values are
                np.ndarray[nwalkers, nleaves_max] representing which
                leaves are currently being used.
            random (object): Current random state object.

        """

        q = {}
        for name, coords in zip(
            branches_coords.keys(), branches_coords.values()
        ):

            ntemps, nwalkers, nleaves_max, ndim_here = coords.shape
            proposal_fn = self.all_proposal[name]

            # copy coords
            q[name] = coords.copy()
            
            # coordinates holder by temperature
            by_temp = coords.copy()

            # if running gibbs, make sure we pass to the proposal function only the correct indeces
            running_coords = kwargs["inds_run"][0]
            if running_coords is not None:
                by_temp[:,:,running_coords] = np.asarray([proposal_fn(coords[tt,:,running_coords].T, random)[0] for tt in range(ntemps)])
            else:
                # assuming number of leaves is only one
                by_temp[:,:,0,:] = np.asarray([proposal_fn(coords[tt,:,0,:], random)[0] for tt in range(ntemps)])
            
            # by_temp_check = by_temp.copy()
            if self.sky_per is not None:
                    correct_extrinsic_array(by_temp.reshape(ntemps * nwalkers, ndim_here) , self.sky_per)
            # print("check", np.sum(by_temp_check - by_temp))

            q[name] = by_temp.copy()
            
            if self.periodic is not None:
                temp = q[name].copy()
                whatever = {name:temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)}
                temp = self.periodic.wrap(whatever)
                q[name] = temp[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

        return q, np.zeros((ntemps, nwalkers))


class MyRJMove(MHMove):
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

    def __init__(self, cov_all, factor=None, **kwargs):

        self.all_proposal = {}
        for name, cov in cov_all.items():
            # Parse the proposal type.
            self.all_proposal[name] = cov

        super(MyRJMove, self).__init__(**kwargs)

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
            new_coords, _ = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)
                for name, tmp in q.items()
            }

        return q, np.zeros((ntemps, nwalkers))



from itertools import permutations
import random

class proposal_template(object):

    def __init__(self, model, proposal, hypermod=False, indx_list=None):
        self.model = model
        self.hypermod = hypermod
        if proposal=="DE":
            self.proposal = self.DEmove
        elif proposal=="Prior":
            self.proposal = self.PriorProp
        elif proposal=="AM":
            self.proposal = self.AMMove
        elif proposal=="Normal":
            self.proposal = self.GaussianMove
        elif proposal=="Fisher":
            self.proposal = self.FisherMove

        if indx_list is not None:
            self.indx_list = indx_list
            self.split_update = True
        else:
            
            self.split_update = False
    
    def initial_sample(self):
        
        if self.hypermod:
            return self.model.initial_sample()[self.sample_list]
        else:
            return np.hstack(p.sample() for p in self.model.params)[self.sample_list]

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        self.sample_list = np.ones(nd, dtype=bool)
        if self.split_update:
            new_pos = x0.copy()

            for i in range(nw):
                q = np.random.randint(len(self.indx_list))
                self.sample_list = self.indx_list[q]
                new_pos[i,self.indx_list[q]] = self.proposal(x0[:,self.sample_list], rng)[0][i,:]
            return new_pos,  np.zeros(nw)
        else:
            return self.proposal(x0, rng)
    
    def PriorProp(self, x0, rng):
        nw, nd = x0.shape
        x = np.array([self.initial_sample() for _ in range(nw)])
        return  x, np.zeros(nw)

    def DEmove(self, x0, rng):
        """
        DE implemented as described in https://arxiv.org/pdf/1404.1267.pdf eq.(C5)
        """
        new_pos = x0.copy()
        if self.hypermod:
            xtemp = x0[:,:-1].copy()
        else:
            xtemp = x0.copy()
        
        nw, nd = xtemp.shape
        perms = list(permutations(np.arange(nw), 2))
        pairs = np.asarray(random.sample(perms,nw)).T
        if np.random.rand() > 0.5:
            gamma = 2.38 / np.sqrt(2 * nd) 
        else:
            gamma = 1.0

        f = 1e-5 * rng.randn(nw)

        if self.hypermod:
            new_pos[:,:-1] += gamma * (xtemp[pairs[0]]-xtemp[pairs[1]]) + f[:,None]
            new_pos[:,-1] += gamma * (xtemp[pairs[0]]-xtemp[pairs[1]])[:,-1]#np.array([self.model.initial_sample()[-1] for _ in range(nw)])# 
        else:
            new_pos += gamma * (xtemp[pairs[0]]-xtemp[pairs[1]]) + f[:,None]
        
        return new_pos, np.zeros(nw)

    def GaussianMove(self, x0, rng):
        """
        https://link.springer.com/content/pdf/10.1007/s11222-006-9438-0.pdf
        """
        new_pos = x0.copy()
        if self.hypermod:
            xtemp = x0[:,:-1].copy()
        else:
            xtemp = x0.copy()
        
        # x0 = x0[:-1]
        nw, nd = xtemp.shape
        mean = np.mean(xtemp, axis=0)
        eps = 1e-3 # makes sure that the covariance matrix is not singular
        cov = (np.cov(xtemp, rowvar=False) + eps * np.diag(np.ones_like(xtemp[0])) ) * 2.38**2 / nd

        if self.hypermod:
            new_pos[:,:-1] += np.random.multivariate_normal(mean*0.0,cov,size=nw)
            new_pos[:,-1] += np.random.normal(np.mean(x0[:,-1])*0.0, np.std(x0[:,-1]),size=nw)#np.array([self.model.initial_sample()[-1] for _ in range(nw)])# 
        else:
            new_pos += np.random.multivariate_normal(mean*0.0,cov, size=nw)

        return new_pos, np.zeros(nw)

    def AMMove(self, x0, rng):
        """
        Adaptive Jump Proposal
        """
        new_pos = x0.copy()
        if self.hypermod:
            xtemp = x0[:,:-1].copy()
        else:
            xtemp = x0.copy()
        
        nw, nd = xtemp.shape

        # calculate covariance and make SVD decomposition
        try:
            
            U, S, v = np.linalg.svd(np.cov(xtemp, rowvar=False))

            # adjust step size
            prob = rng.random()

            # large jump
            if prob > 0.99:
                scale = 10.0

            # small jump
            elif prob > 0.9:
                scale = 0.2

            # standard medium jump
            else:
                scale = 1.0
            
            
            # go in eigen basis
            y = np.asarray([np.dot(U.T, xtemp[i]) for i in range(nw)])
            # choose a random parameter in the uncorrelated basis
            ind_vec = np.arange(nd)
            np.random.shuffle(ind_vec)
            rand_j = ind_vec[:np.random.randint(1,nd)]
            y[:,rand_j] += np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * 2.38 / np.sqrt(nd)
            # go back to the basis
            new_pos = np.asarray([np.dot(U, y[i]) for i in range(nw)]) 

            # new_pos += np.random.normal(size=nw)[:,None] * np.sqrt(S[None,rand_j]) * U[:,rand_j]
        except:
            print("svd failed")
            new_pos = np.array([self.initial_sample() for i in range(nw)])
        return new_pos, np.zeros(nw)


    def FisherMove(self, x0, rng):
        """
        https://link.springer.com/content/pdf/10.1007/s11222-006-9438-0.pdf
        """
        new_pos = x0.copy()
        if self.hypermod:
            xtemp = x0[:,:-1].copy()
        else:
            xtemp = x0.copy()
        
        # x0 = x0[:-1]
        nw, nd = xtemp.shape
        mean = np.mean(xtemp, axis=0)
        eps = 1e-5 # makes sure that the covariance matrix is not singular
        cov = (np.cov(xtemp, rowvar=False) + eps * np.diag(np.ones_like(xtemp[0])) ) * 2.38**2 / nd
        w, v = np.linalg.eigh(FISHER)

        if self.hypermod:
            new_pos[:,:-1] += np.random.multivariate_normal(mean*0.0,cov,size=nw)
            new_pos[:,-1] += np.random.normal(np.mean(x0[:,-1])*0.0, np.std(x0[:,-1]),size=nw)#np.array([self.model.initial_sample()[-1] for _ in range(nw)])# 
        else:
            new_pos += np.random.multivariate_normal(mean*0.0,cov, size=nw)

        return new_pos, np.zeros(nw)


def get_fisher_eigenvectors(params, par_names, par_names_to_perturb, pta, epsilon=1e-4):
    """get fisher eigenvectors for a generic set of parameters the slow way"""
    try:
        dim = len(par_names_to_perturb)
        fisher = np.zeros((dim,dim))

        #lnlikelihood at specified point
        nn = pta.get_lnlikelihood(params)

        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[par_names.index(par_names_to_perturb[i])] += 2*epsilon
            paramsMM[par_names.index(par_names_to_perturb[i])] -= 2*epsilon

            #lnlikelihood at +-epsilon positions
            pp = pta.get_lnlikelihood(paramsPP)
            mm = pta.get_lnlikelihood(paramsMM)

            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

            if fisher[i,i] <= 0.:  # diagonal elements must be postive
                fisher[i,i] = 4.

        #calculate off-diagonal elements
        for i in range(dim):
            for j in range(i+1,dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPP[par_names.index(par_names_to_perturb[j])] += epsilon
                paramsMM[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsPM[par_names.index(par_names_to_perturb[i])] += epsilon
                paramsPM[par_names.index(par_names_to_perturb[j])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[i])] -= epsilon
                paramsMP[par_names.index(par_names_to_perturb[j])] += epsilon

                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)
                pm = pta.get_lnlikelihood(paramsPM)
                mp = pta.get_lnlikelihood(paramsMP)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                fisher[i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[j,i] = fisher[i,j]
                #fisher[j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)

        #Filter nans and infs and replace them with 1s
        #this will imply that we will set the eigenvalue to 100 a few lines below
        print('fisher 2')
        print(fisher)
        print('fisher determinant',np.linalg.det(fisher),np.prod(np.diagonal(fisher)))
        FISHER = np.where(np.isfinite(fisher), fisher, 1.0)
        if not np.array_equal(FISHER, fisher):
            print("Changed some nan elements in the Fisher matrix to 1.0")

        #Find eigenvalues and eigenvectors of the Fisher matrix
        w, v = np.linalg.eigh(FISHER)

        #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
        eig_limit = 4.0

        W = np.where(np.abs(w)>eig_limit, w, eig_limit)

        return (np.sqrt(1.0/np.abs(W))*v).T

    except np.linalg.LinAlgError:
        print("An Error occured in the eigenvalue calculation")
        print(par_names_to_perturb)
        print(params)
        return np.eye(len(par_names_to_perturb))*0.5


from numba import njit
@njit()
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

def correct_extrinsic_array(samples,periodic_var):
    samples[:,periodic_var[0]],samples[:,periodic_var[1]] = reflect_cosines_array(samples[:,periodic_var[0]],samples[:,periodic_var[1]],np.pi,2*np.pi)
    samples[:,periodic_var[2]],samples[:,periodic_var[3]] = reflect_cosines_array(samples[:,periodic_var[2]],samples[:,periodic_var[3]],np.pi/2,np.pi)
