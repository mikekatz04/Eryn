from multiprocessing.sharedctypes import Value
import numpy as np
import warnings
# from scipy.special import logsumexp

try:
    import cupy as xp

    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as xp

    gpu_available = False

from .rj import ReversibleJump
from ..prior import ProbDistContainer
from ..utils.utility import groups_from_inds

___ = ["MultipleTryMove"]

def logsumexp(a, axis=None, xp=None):
    if xp is None:
        xp = np
    
    max = xp.max(a, axis=axis)
    ds = a - max[:, None]

    sum_of_exp = xp.exp(ds).sum(axis=axis)
    return max + xp.log(sum_of_exp)


class MultipleTryMove:
    """Generate multiple proposal tries.
    Args:
        priors (object): :class:`ProbDistContainer` object that has ``logpdf``
            and ``rvs`` methods.
    """

    def __init__(
        self, num_try=1, independent=False, symmetric=False, xp=None, **kwargs
    ):
        # TODO: make priors optional like special generate function?
        self.num_try = num_try
        
        self.independent = independent
        self.symmetric = symmetric

        if xp is None:
            xp = np

        self.xp = xp

    def get_mt_log_posterior(self, ll, lp, betas=None):
        if betas is None:
            ll_temp = ll.copy()
        else:
            assert isinstance(betas, self.xp.ndarray)
            if ll.ndim > 1:
                betas_tmp = self.xp.expand_dims(betas, ll.ndim - 1)
            else:
                betas_tmp = betas
            ll_temp = betas_tmp * ll

        return ll_temp + lp

    def readout_adjustment(self, out_vals, all_vals_prop, aux_all_vals):
        pass

    def get_mt_proposal(
        self,
        coords,
        random,
        inds_rj_reverse=None,
        args_generate=(),
        kwargs_generate={},
        args_like=(),
        kwargs_like={},
        args_prior=(),
        kwargs_prior={},
        betas=None,
        ll_in=None,
        lp_in=None,
    ):
        """Make a proposal
        """

        # generate new points and get detailed balance info
        generated_points, log_proposal_pdf = self.special_generate_func(
            coords,
            random,
            *args_generate,
            size=self.num_try,
            **kwargs_generate
        )
        
        ll = self.special_like_func(generated_points, *args_like, **kwargs_like)

        if self.xp.any(self.xp.isnan(ll)):
            warnings.warn("Getting nans for ll in multiple try.")
            ll[self.xp.isnan(ll)] = -1e300

        lp = self.special_prior_func(
            generated_points, *args_prior, **kwargs_prior
        )

        logP = self.get_mt_log_posterior(ll, lp, betas=betas)
        
        if self.symmetric:
            log_importance_weights = logP
        else:
            log_importance_weights = logP - log_proposal_pdf

        log_sum_weights = logsumexp(log_importance_weights, axis=-1, xp=self.xp)

        log_of_probs = log_importance_weights - log_sum_weights[:, None]
        probs = self.xp.exp(log_of_probs)

        # draw based on likelihood
        inds_keep = (
            probs.cumsum(1) > self.xp.random.rand(probs.shape[0])[:, None]
        ).argmax(1)

        inds_tuple = (self.xp.arange(len(inds_keep)), inds_keep)
        lp_out = lp[inds_tuple]
        ll_out = ll[inds_tuple]
        logP_out = logP[inds_tuple]

        self.mt_lp = lp_out
        self.mt_ll = ll_out

        generated_points_out = generated_points[inds_tuple].copy()  # theta^j
        log_proposal_pdf_out = log_proposal_pdf[inds_tuple]
       
        if self.independent:
            aux_ll = ll.copy()
            aux_lp = lp.copy()
            
            aux_log_proposal_pdf_sub = self.special_generate_logpdf(coords)

            if ll_in is None:
                aux_ll_sub = self.special_generate_like(coords)

            else:
                assert ll_in.shape[0] == coords.shape[0]
                aux_ll_sub = ll_in
    
            if lp_in is None:
                aux_lp_sub = self.special_generate_prior(coords)

            else:
                assert lp_in.shape[0] == coords.shape[0]
                aux_lp_sub = lp_in

            aux_ll[inds_tuple] = aux_ll_sub
            aux_lp[inds_tuple] = aux_lp_sub

            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)
            
            aux_log_proposal_pdf = log_proposal_pdf.copy()
            aux_log_proposal_pdf[inds_tuple] = aux_log_proposal_pdf_sub

            aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

        else:
            # generate auxillary points
            aux_generated_points, aux_log_proposal_pdf = self.special_generate_func(
                generated_points_out,
                random,
                *args_generate,
                size=self.num_try,
                fill_tuple=inds_tuple,
                fill_values=generated_points_out,
                **kwargs_generate
            )
            aux_ll = self.special_like_func(
                aux_generated_points, *args_like, **kwargs_like
            )

            aux_lp = self.special_prior_func(aux_generated_points)

            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            if not self.symmetric:
                aux_log_importance_weights = aux_logP - aux_log_proposal_pdf_sub
            else:
                aux_log_importance_weights = aux_logP

        aux_logP_out = aux_logP[inds_tuple]
        aux_log_sum_weights = logsumexp(aux_log_importance_weights, axis=-1, xp=self.xp)

        aux_log_proposal_pdf_out = aux_log_proposal_pdf[inds_tuple]
        # this is setup to make clear with the math.
        # setting up factors properly means the
        # final lnpdiff will be effectively be the ratio of the sums
        # of the weights

        # IMPORTANT: logP_out must be subtracted against log_sum_weights before anything else due to -1e300s.
        factors = (
            (aux_logP_out - aux_log_sum_weights)
            - aux_log_proposal_pdf_out
            + aux_log_proposal_pdf_out
        ) - (
            (logP_out - log_sum_weights)
            - log_proposal_pdf_out
            + log_proposal_pdf_out
        )

        self.aux_logP_out = aux_logP_out
        self.logP_out = logP_out
        
        self.log_sum_weights = log_sum_weights
        self.aux_log_sum_weights = aux_log_sum_weights

        out_vals = [logP_out, ll_out, lp_out, log_proposal_pdf_out, log_sum_weights]
        all_vals_prop = [logP, ll, lp, log_proposal_pdf, log_sum_weights]
        aux_all_vals = [aux_logP, aux_ll, aux_lp, aux_log_proposal_pdf, aux_log_sum_weights]
        self.readout_adjustment(out_vals, all_vals_prop, aux_all_vals)

        return (
            generated_points_out,
            factors,
        )

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        
        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        key_in = list(branches_coords.keys())[0]

        if branches_inds is None:
            branches_inds = {}
            branches_inds[key_in] = np.ones(branches_coords[key_in].shape[:-1], dtype=bool)

        if np.any(branches_inds[key_in].sum(axis=-1) > 1):
            raise ValueError

        ntemps, nwalkers, _, _ = branches_coords[key_in].shape
        
        betas_here = np.repeat(self.temperature_control.betas[:, None], np.prod(branches_coords[key_in].shape[1:-1])).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        ll_here = np.repeat(self.current_state.log_like[:, :, None], branches_coords[key_in].shape[2], axis=-1).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]
        lp_here = np.repeat(self.current_state.log_prior[:, :, None], branches_coords[key_in].shape[2], axis=-1).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        generated_points, factors = self.get_mt_proposal(branches_coords[key_in][branches_inds[key_in]], random, betas=betas_here, ll_in=ll_here, lp_in=lp_here)

        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)
        # TODO: check this with multiple leaves gibbs ndim
        return {key_in: generated_points.reshape(ntemps, nwalkers, 1, -1)}, factors.reshape(ntemps, nwalkers)