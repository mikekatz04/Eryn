# -*- coding: utf-8 -*-

from multiprocessing.sharedctypes import Value
import numpy as np
import warnings
from scipy.special import logsumexp

try:
    import cupy as xp

    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as xp

    gpu_available = False

from .rj import ReversibleJump
from ..prior import PriorContainer
from ..utils.utility import groups_from_inds

___ = ["MultipleTryMove"]


class MultipleTryMove:
    """Generate multiple proposal tries.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        num_try,
        take_max_ll=False,
        return_accepted_info=False,
    ):
        # TODO: make priors optional like special generate function? 
        self.num_try = num_try
        self.take_max_ll = take_max_ll
        self.return_accepted_info = return_accepted_info

        if self.return_accepted_info:
            assert hasattr(self, "special_prior_func")

    def get_bf_log_posterior(self, ll, lp, betas=None):
        if betas is None:
            ll_temp = ll.copy()
        else:
            assert isinstance(betas, np.ndarray)
            if ll.ndim > 1:
               betas_tmp = np.expand_dims(betas, ll.ndim - 1)
            else:
                betas_tmp = betas
            ll_temp = betas_tmp * ll

        return ll_temp + lp

    def get_bf_proposal(self, coords, nwalkers, inds_reverse, inds_reverse_individual, random, args_generate=(), kwargs_generate={}, args_like=(), kwargs_like={}, args_prior=(), kwargs_prior={}, rj_info={}, betas=None):
        """Make a proposal

        Args:
            coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            inds_for_change (dict): Keys are ``branch_names``. Values are
                dictionaries. These dictionaries have keys ``"+1"`` and ``"-1"``,
                indicating waklkers that are adding or removing a leafm respectively.
                The values for these dicts are ``int`` np.ndarray[..., 3]. The "..." indicates
                the number of walkers in all temperatures that fall under either adding
                or removing a leaf. The second dimension, 3, is the indexes into
                the three-dimensional arrays within ``inds`` of the specific leaf
                that is being added or removed from those leaves currently considered.
            random (object): Current random state of the sampler.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
            
        # prep reverse info
        # must put them in the zero position
        inds_reverse_tuple = (inds_reverse, np.zeros_like(inds_reverse))

        # generate new points and get detailed balance info
        generated_points, log_proposal_pdf = self.special_generate_func(coords, nwalkers, *args_generate, random=random, size=self.num_try, fill=coords[inds_reverse], fill_inds=inds_reverse_tuple, **kwargs_generate)
        ll = self.special_like_func(generated_points, *args_like, **kwargs_like)

        if self.take_max_ll:
            # get max
            inds_keep = np.argmax(ll, axis=-1)
        
            factors = np.zeros((nwalkers,))
            return generated_points_out, ll_out, factors

        else:
            if np.any(np.isnan(ll)):
                warnings.warn("Getting nans for ll in multiple try.")
                ll[np.isnan(ll)] = -1e300

            if hasattr(self, "special_prior_func"):
                lp = self.special_prior_func(generated_points, *args_prior, **kwargs_prior)
                logP = self.get_bf_log_posterior(ll, lp, betas=betas)

            else:
                logP = ll

            log_importance_weights = logP - log_proposal_pdf

            log_sum_weights = logsumexp(log_importance_weights, axis=-1)

            log_of_probs = log_importance_weights - log_sum_weights[:, None]
            probs = np.exp(log_of_probs)

            # draw based on likelihood
            inds_keep = (probs.cumsum(1) > np.random.rand(probs.shape[0])[:,None]).argmax(1)

            inds_keep[inds_reverse] = 0

            #log_prob_factors = np.log(probs[:, ind_keep])
            inds_tuple = (np.arange(len(inds_keep)), inds_keep)
            logP_out = logP[inds_tuple]
            self.ll_out = ll[inds_tuple]
            if hasattr(self, "special_prior_func"):
                self.lp_out = lp[inds_tuple]
            generated_points_out = generated_points[inds_tuple].copy()  # theta^j
            log_proposal_pdf_out = log_proposal_pdf[inds_tuple]

            if rj_info == {}:
                # generate auxillary points
                aux_generated_points, aux_log_proposal_pdf = self.special_generate_func(generated_points_out, nwalkers, *args_generate, random=random, size=self.num_try, fill=generated_points_out, fill_inds=inds_tuple, **kwargs_generate)

                aux_ll = self.special_like_func(aux_generated_points, *args_like, **kwargs_like)

                if hasattr(self, "special_prior_func"):
                    aux_lp = self.special_prior_func(aux_generated_points)
                    aux_logP = self.get_bf_log_posterior(aux_ll, aux_lp, betas=betas)
                    
                else:
                    aux_logP = aux_ll

                aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

            else:
                assert "ll" in rj_info
                assert "lp" in rj_info

                aux_ll = rj_info["ll"]
                aux_lp = rj_info["lp"]

                if not hasattr(self, "special_aux_ll"):
                    raise ValueError("If using RJ, must have special_aux_ll attribute that gives the aux_ll for reverse proposals.")

                # sub in the old ll values for the reverse cases
                aux_ll[inds_reverse] = self.special_aux_ll

                # aux_lp[inds_reverse]  # do not need to do this because the inds reflect the removed case already. 
                aux_logP = self.get_bf_log_posterior(aux_ll, aux_lp, betas=betas)
                aux_log_proposal_pdf = np.zeros_like(aux_logP)
                aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

                # scale out
                aux_log_importance_weights = np.repeat(aux_log_importance_weights[:, None], self.num_try, axis=-1)

            aux_log_sum_weights = logsumexp(aux_log_importance_weights, axis=-1)

            aux_log_proposal_pdf = np.zeros_like(log_proposal_pdf_out)
            factors = (aux_logP - aux_log_proposal_pdf - aux_log_sum_weights) - (logP_out - log_proposal_pdf_out - log_sum_weights) - log_proposal_pdf_out

            # stop pretending, reverse factors for reverse case
            factors[inds_reverse] *= -1.0
            #if np.any(factors > 0.0):
            self.logP_out = logP_out.copy()

            #if np.any((log_sum_weights - aux_log_sum_weights) > 0.0):
            #    breakpoint()

            return generated_points_out[np.delete(np.arange(nwalkers), inds_reverse)], logP_out, factors
