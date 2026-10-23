"""Multiple-try Metropolis machinery (in-model and nested reversible jump).

:class:`MultipleTryMove` is a mixin that, given a subclass's generate / likelihood / prior
callables, draws ``num_try`` candidates per walker, picks one by importance weight, builds
the auxiliary (reverse) set and returns the multiple-try log factor. :class:`MultipleTryMoveRJ`
is its nested-RJ variant (one birth or death per walker). Concrete moves:
:class:`~eryn.moves.MTDistGenMove` and :class:`~eryn.moves.MTDistGenMoveRJ`. The helpers
:func:`logsumexp` and :func:`get_mt_computations` are reused by other moves.
"""
import warnings
from abc import ABC
from copy import deepcopy
from multiprocessing.sharedctypes import Value

import numpy as np

# from scipy.special import logsumexp

try:
    import cupy as cp

    gpu_available = True
except (ModuleNotFoundError, ImportError):
    import numpy as cp

    gpu_available = False

from ..prior import ProbDistContainer
from ..utils.utility import groups_from_inds
from .rj import ReversibleJumpMove

___ = ["MultipleTryMove"]


def logsumexp(a, axis=None, xp=None):
    """Numerically stable ``log(sum(exp(a), axis=axis))`` along axis 1.

    The maximum is broadcast back with ``max[:, None]``, so despite the general signature
    the reduction must be along axis 1 (``axis=-1`` or ``1`` for the 2D arrays all callers
    pass); the default ``axis=None`` fails.

    Args:
        a (xp.ndarray): Values, shape ``(n, m)``.
        axis (int, optional): Reduction axis, ``1`` / ``-1``. Default ``None``.
        xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` means ``numpy``.

    Returns:
        xp.ndarray: Log-sum-exp per row, shape ``(n,)``.
    """
    if xp is None:
        xp = np

    max = xp.max(a, axis=axis)
    ds = a - max[:, None]

    sum_of_exp = xp.exp(ds).sum(axis=axis)
    return max + xp.log(sum_of_exp)

# TODO check: keep/remove/document -- duplicate; shadowed by the identical definition below
def get_mt_computations(logP, log_proposal_pdf, symmetric=False, xp=None):

    if xp is None:
        xp = np

        # set weights based on if symmetric
    if symmetric:
        log_importance_weights = logP
    else:
        log_importance_weights = logP - log_proposal_pdf

    # get the sum of weights
    log_sum_weights = logsumexp(log_importance_weights, axis=-1, xp=xp)

    # probs = wi / sum(wi)
    log_of_probs = log_importance_weights - log_sum_weights[:, None]

    # probabilities to choose try
    probs = xp.exp(log_of_probs)

    # draw based on likelihood
    inds_keep = (
        probs.cumsum(1) > xp.random.rand(probs.shape[0])[:, None]
    ).argmax(1)

    return log_importance_weights, log_sum_weights, inds_keep


def get_mt_computations(logP, log_proposal_pdf, symmetric=False, xp=None):
    """Importance weights of the tries and the index of the chosen try per walker.

    The log weight of try ``j`` is ``logP_j - log q_j`` (or ``logP_j`` when
    ``symmetric``); one try per row is drawn with probability proportional to its weight,
    using the GLOBAL ``xp.random`` generator.

    Args:
        logP (xp.ndarray): Log posterior of every try, shape ``(nwalkers, num_try)``.
        log_proposal_pdf (xp.ndarray): Log proposal density of every try, same shape.
        symmetric (bool, optional): Ignore ``log_proposal_pdf``. Default ``False``.
        xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` means ``numpy``.

    Returns:
        tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(log_importance_weights,
        log_sum_weights, inds_keep)`` with shapes ``(nwalkers, num_try)``, ``(nwalkers,)``
        and ``(nwalkers,)`` (integer index of the chosen try).
    """

    if xp is None:
        xp = np

        # set weights based on if symmetric
    if symmetric:
        log_importance_weights = logP
    else:
        log_importance_weights = logP - log_proposal_pdf

    # get the sum of weights
    log_sum_weights = logsumexp(log_importance_weights, axis=-1, xp=xp)

    # probs = wi / sum(wi)
    log_of_probs = log_importance_weights - log_sum_weights[:, None]

    # probabilities to choose try
    probs = xp.exp(log_of_probs)

    # draw based on likelihood
    inds_keep = (probs.cumsum(1) > xp.random.rand(probs.shape[0])[:, None]).argmax(1)

    return log_importance_weights, log_sum_weights, inds_keep


class MultipleTryMove(ABC):
    """Mixin that turns a generate / likelihood / prior trio into a multiple-try proposal.

    Combine it with a :class:`~eryn.moves.Move` subclass (it is not a ``Move`` itself) and
    override :meth:`special_generate_func`, :meth:`special_like_func`,
    :meth:`special_prior_func` (and :meth:`special_generate_logpdf` for independent
    proposals). :meth:`get_proposal` reads ``self.temperature_control.betas`` (a controller
    must be attached) and ``self.current_state`` (set by the ``Move`` side's ``propose``).
    It stores ``mt_ll`` / ``mt_lp`` of the chosen tries, which
    :meth:`~eryn.moves.MHMove.propose` and :meth:`~eryn.moves.ReversibleJumpMove.propose`
    then use instead of re-evaluating the model. See :class:`~eryn.moves.MTDistGenMove`
    and :class:`~eryn.moves.MTDistGenMoveRJ`.

    Args:
        num_try (int, optional): Number of tries per walker. Default ``1``.
        independent (bool, optional): ``True`` if the proposal does not depend on the
            current point (the auxiliary set then reuses the tries). Default ``False``.
        symmetric (bool, optional): ``True`` if the proposal is symmetric (its density drops
            out of the weights). Default ``False``.
        rj (bool, optional): ``True`` for a nested reversible-jump proposal; requires
            ``independent`` and ``symmetric`` to be ``False``. Default ``False``.
        use_gpu (bool, optional): Selects :attr:`xp` (``cupy`` when truthy). Default
            ``None``.
        **kwargs: Ignored here (the ``Move`` side consumes them).

    Raises:
        ValueError: ``rj`` with ``symmetric`` or ``independent``.
    """

    def __init__(
        self,
        num_try=1,
        independent=False,
        symmetric=False,
        rj=False,
        use_gpu=None,
        **kwargs
    ):
        self.num_try = num_try

        self.independent = independent
        self.symmetric = symmetric
        self.rj = rj

        if self.rj:
            if self.symmetric or self.independent:
                raise ValueError(
                    "If rj==True, symmetric and independt must both be False."
                )

        self.use_gpu = use_gpu

    @property
    def xp(self):
        """module: ``cupy`` if ``use_gpu`` is truthy (and cupy is installed), else ``numpy``."""
        xp = cp if self.use_gpu else np
        return xp

    @classmethod
    def special_like_func(self, generated_coords, *args, inds_leaves_rj=None, **kwargs):
        """Log-likelihood of every try (implemented by subclasses).

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)`` (here
                ``nwalkers`` counts all independent walkers, ``ntemps * nwalkers`` of the
                sampler).
            *args: From ``args_like`` of :meth:`get_mt_proposal`.
            inds_leaves_rj (np.ndarray, optional): Leaf slot of the proposal per walker; only
                for ``rj``. Default ``None``.
            **kwargs: From ``kwargs_like`` of :meth:`get_mt_proposal`.

        Returns:
            np.ndarray: Log-likelihood, shape ``(nwalkers, num_try)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    @classmethod
    def special_prior_func(self, generated_coords, *args, **kwargs):
        """Log prior of every try (implemented by subclasses).

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            *args: From ``args_prior`` of :meth:`get_mt_proposal`.
            **kwargs: From ``kwargs_prior``; :meth:`get_mt_proposal` also passes
                ``inds_leaves_rj`` (leaf slot per walker, only meaningful for ``rj``).

        Returns:
            np.ndarray: Log prior, shape ``(nwalkers, num_try)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    @classmethod
    def special_generate_func(
        coords, random, size=1, *args, fill_tuple=None, fill_values=None, **kwargs
    ):
        """Draw the tries and their log proposal density (implemented by subclasses).

        Subclasses define it as an instance method ``(self, coords, random, size=1, ...)``.

        Args:
            coords (np.ndarray): Current coordinates, shape ``(nwalkers, ndim)``.
            random (object): Random generator.
            size (int, optional): Number of tries per walker (``num_try``). Default ``1``.
            *args: From ``args_generate`` of :meth:`get_mt_proposal`.
            fill_tuple (tuple, optional): ``(walker_idx, try_idx)`` of entries to overwrite
                with ``fill_values`` after drawing, used for the auxiliary set and reverse
                RJ proposals. Default ``None``.
            fill_values (np.ndarray, optional): Values for ``fill_tuple``, shape
                ``(len(fill_tuple[0]), ndim)``. Default ``None``.
            **kwargs: From ``kwargs_generate`` of :meth:`get_mt_proposal`.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tries of shape ``(nwalkers, size, ndim)`` and their
            log proposal density, shape ``(nwalkers, size)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    @classmethod
    def special_generate_logpdf(self, coords):
        """Log proposal density of given points (implemented by subclasses).

        Args:
            coords (np.ndarray): Points, shape ``(n, ndim)``.

        Returns:
            np.ndarray: Log proposal density, shape ``(n,)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """
        raise NotImplementedError

    def get_mt_log_posterior(self, ll, lp, betas=None):
        """Tempered log posterior of every try, ``betas * ll + lp``.

        Args:
            ll (xp.ndarray): Log-likelihood, shape ``(nwalkers, num_try)``.
            lp (xp.ndarray): Log prior, same shape.
            betas (xp.ndarray, optional): Inverse temperature per walker, shape
                ``(nwalkers,)``; must be an ``xp.ndarray``. ``None`` means untempered.
                Default ``None``.

        Returns:
            xp.ndarray: Log posterior, shape ``(nwalkers, num_try)``.
        """
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
        """Hook to read out intermediate values of a multiple-try proposal (no-op here).

        Called at the end of :meth:`get_mt_proposal`; override to keep any of these.

        Args:
            out_vals (list): Chosen-try values
                ``[logP_out, ll_out, lp_out, log_proposal_pdf_out, log_sum_weights]``.
            all_vals_prop (list): All-try values
                ``[logP, ll, lp, log_proposal_pdf, log_sum_weights]``.
            aux_all_vals (list): Auxiliary-set values
                ``[aux_logP, aux_ll, aux_lp, aux_log_proposal_pdf, aux_log_sum_weights]``.
        """
        pass

    def get_mt_proposal(
        self,
        coords,
        random,
        args_generate=(),
        kwargs_generate={},
        args_like=(),
        kwargs_like={},
        args_prior=(),
        kwargs_prior={},
        betas=None,
        ll_in=None,
        lp_in=None,
        inds_leaves_rj=None,
        inds_reverse_rj=None,
    ):
        """Make a multiple-try proposal.

        Here ``nwalkers`` counts all independent walkers, generally ``ntemps * nwalkers`` of
        the sampler. Draws ``num_try`` tries per walker, weights them
        (:func:`get_mt_computations`), picks one, builds the auxiliary (reverse) set
        (reusing the tries when ``independent``, the one-leaf-fewer model when ``rj``, or a
        fresh draw around the chosen point otherwise) and returns the log factor that makes
        the acceptance the ratio of the summed weights. For ``rj``, reverse (death)
        proposals have their factor negated and their ``mt_ll`` / ``mt_lp`` reset to
        ``ll_in`` / ``lp_in``.

        Side effects: sets ``mt_ll``, ``mt_lp`` (chosen tries), ``aux_logP_out``,
        ``logP_out``, ``aux_ll``, ``aux_lp``, ``log_sum_weights``, ``aux_log_sum_weights``
        and, for ``rj``, ``inds_reverse_rj`` / ``inds_forward_rj``; then calls
        :meth:`readout_adjustment`.

        Args:
            coords (np.ndarray): Current coordinates, shape ``(nwalkers, ndim)``.
            random (object): Random generator passed to :meth:`special_generate_func`.
            args_generate (tuple, optional): Extra ``*args`` for the generate function
                (override ``get_proposal`` to use them). Default ``()``.
            kwargs_generate (dict, optional): Extra ``**kwargs`` for the generate function.
                Default ``{}``.
            args_like (tuple, optional): Extra ``*args`` for the likelihood function.
                Default ``()``.
            kwargs_like (dict, optional): Extra ``**kwargs`` for the likelihood function.
                Default ``{}``.
            args_prior (tuple, optional): Extra ``*args`` for the prior function. Default
                ``()``.
            kwargs_prior (dict, optional): Extra ``**kwargs`` for the prior function.
                Default ``{}``.
            betas (xp.ndarray, optional): Inverse temperature per walker, shape
                ``(nwalkers,)``. Default ``None`` (untempered).
            ll_in (np.ndarray, optional): Log-likelihood at the current coordinates, shape
                ``(nwalkers,)``. Required for ``rj`` (nested model), where for proposed
                removals it must be the likelihood WITH THE LEAF REMOVED so every proposal
                looks like an addition. Used by ``independent`` proposals. Default ``None``.
            lp_in (np.ndarray, optional): Log prior at the current coordinates, same rules as
                ``ll_in``. Default ``None``.
            inds_leaves_rj (np.ndarray, optional): Leaf slot of the proposal per walker,
                shape ``(nwalkers,)``. Required for ``rj``. Default ``None``.
            inds_reverse_rj (np.ndarray, optional): Walker indices whose proposal is a
                removal. Required for ``rj``. Default ``None``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(generated_points, factors)``: the chosen try
            per walker, shape ``(nwalkers, ndim)``, and the log factor, shape
            ``(nwalkers,)``.

        Raises:
            ValueError: ``rj`` without ``ll_in``, ``lp_in``, ``inds_leaves_rj`` and
                ``inds_reverse_rj``.
        """

        # check if rj and make sure we have all the information in that case
        if self.rj:
            try:
                assert ll_in is not None and lp_in is not None
                assert inds_leaves_rj is not None and inds_reverse_rj is not None
            except AssertionError:
                raise ValueError(
                    "If using rj, must provide ll_in, lp_in, inds_leaves_rj, and inds_reverse_rj."
                )

            # if using reversible jump, fill first spot with values that are proposed to remove
            fill_tuple = (inds_reverse_rj, np.zeros_like(inds_reverse_rj))
            fill_values = coords[inds_reverse_rj]
        else:
            fill_tuple = None
            fill_values = None

        # generate new points and get log of the proposal probability
        generated_points, log_proposal_pdf = self.special_generate_func(
            coords,
            random,
            *args_generate,
            size=self.num_try,
            fill_values=fill_values,
            fill_tuple=fill_tuple,
            **kwargs_generate
        )

        # compute the Likelihood functions
        ll = self.special_like_func(
            generated_points, *args_like, inds_leaves_rj=inds_leaves_rj, **kwargs_like
        )

        # check for nans
        if self.xp.any(self.xp.isnan(ll)):
            warnings.warn("Getting nans for ll in multiple try.")
            ll[self.xp.isnan(ll)] = -1e300

        # compute the Prior functions
        lp = self.special_prior_func(
            generated_points, *args_prior, inds_leaves_rj=inds_leaves_rj, **kwargs_prior
        )

        # if rj, make proposal distribution for all other leaves the prior value
        # this will properly cancel the prior with the proposal for leaves that already exists
        if self.rj:
            log_proposal_pdf += lp_in[:, None]

        # get posterior distribution including tempering
        logP = self.get_mt_log_posterior(ll, lp, betas=betas)

        log_importance_weights, log_sum_weights, inds_keep = get_mt_computations(
            logP, log_proposal_pdf, symmetric=self.symmetric, xp=self.xp
        )

        # tuple of index arrays of which try chosen per walker
        inds_tuple = (self.xp.arange(len(inds_keep)), inds_keep)

        if self.rj:
            # this just ensures the cancellation of logP and aux_logP outside of proposal
            inds_tuple[1][inds_reverse_rj] = 0

        # get chosen prior, Likelihood, posterior information
        lp_out = lp[inds_tuple]
        ll_out = ll[inds_tuple]
        logP_out = logP[inds_tuple]

        # store this information for access outside of multiple try part
        self.mt_lp = lp_out
        self.mt_ll = ll_out

        # choose points and get the log of the proposal for storage
        generated_points_out = generated_points[inds_tuple].copy()  # theta^j
        log_proposal_pdf_out = log_proposal_pdf[inds_tuple]

        # prepare auxillary information based on if it is nested rj, independent, or not
        if self.independent:
            # if independent, all the tries can be repeated for the auxillary draws
            aux_ll = ll.copy()
            aux_lp = lp.copy()

            # sub in the generation pdf for the current coordinates
            aux_log_proposal_pdf_sub = self.special_generate_logpdf(coords)

            # set sub ll based on if it is provided
            if ll_in is None:
                aux_ll_sub = self.special_generate_like(coords)

            else:
                assert ll_in.shape[0] == coords.shape[0]
                aux_ll_sub = ll_in

            # set sub lp based on if it is provided
            if lp_in is None:
                aux_lp_sub = self.special_generate_prior(coords)

            else:
                assert lp_in.shape[0] == coords.shape[0]
                aux_lp_sub = lp_in

            # sub in this information from the current coordinates
            aux_ll[inds_tuple] = aux_ll_sub
            aux_lp[inds_tuple] = aux_lp_sub

            # get auxillary posterior
            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # get aux_log_proposal_pdf information
            aux_log_proposal_pdf = log_proposal_pdf.copy()
            aux_log_proposal_pdf[inds_tuple] = aux_log_proposal_pdf_sub

            # set auxillary weights
            aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

        elif self.rj:
            # in rj, set aux_ll and aux_lp to be repeats of the model with one less leaf
            aux_ll = np.repeat(ll_in[:, None], self.num_try, axis=-1)
            aux_lp = np.repeat(lp_in[:, None], self.num_try, axis=-1)

            # probability is the prior for the existing points
            aux_log_proposal_pdf = aux_lp.copy()

            # get log posterior
            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # get importance weights
            aux_log_importance_weights = aux_logP - aux_log_proposal_pdf

        else:
            # generate auxillary points based on chosen new points
            aux_generated_points, aux_log_proposal_pdf = self.special_generate_func(
                generated_points_out,
                random,
                *args_generate,
                size=self.num_try,
                fill_tuple=inds_tuple,
                fill_values=generated_points_out,
                **kwargs_generate
            )

            # get ll, lp, and lP
            aux_ll = self.special_like_func(
                aux_generated_points, *args_like, **kwargs_like
            )

            aux_lp = self.special_prior_func(aux_generated_points)

            aux_logP = self.get_mt_log_posterior(aux_ll, aux_lp, betas=betas)

            # set auxillary weights
            if not self.symmetric:
                aux_log_importance_weights = aux_logP - aux_log_proposal_pdf_sub
            else:
                aux_log_importance_weights = aux_logP

        # chosen output old Posteriors
        aux_logP_out = aux_logP[inds_tuple]
        # get sum of log weights
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
        ) - ((logP_out - log_sum_weights) - log_proposal_pdf_out + log_proposal_pdf_out)

        if self.rj:
            # adjust all information for reverese rj proposals
            factors[inds_reverse_rj] *= -1
            self.mt_ll[inds_reverse_rj] = ll_in[inds_reverse_rj]
            self.mt_lp[inds_reverse_rj] = lp_in[inds_reverse_rj]

        # store output information
        self.aux_logP_out = aux_logP_out
        self.logP_out = logP_out
        self.aux_ll = aux_ll
        self.aux_lp = aux_lp

        self.log_sum_weights = log_sum_weights
        self.aux_log_sum_weights = aux_log_sum_weights

        if self.rj:
            self.inds_reverse_rj = inds_reverse_rj
            self.inds_forward_rj = np.delete(
                np.arange(coords.shape[0]), inds_reverse_rj
            )

        # prepare to readout any information the user would like in readout_adjustment
        out_vals = [logP_out, ll_out, lp_out, log_proposal_pdf_out, log_sum_weights]
        all_vals_prop = [logP, ll, lp, log_proposal_pdf, log_sum_weights]
        aux_all_vals = [
            aux_logP,
            aux_ll,
            aux_lp,
            aux_log_proposal_pdf,
            aux_log_sum_weights,
        ]
        self.readout_adjustment(out_vals, all_vals_prop, aux_all_vals)

        return (
            generated_points_out,
            factors,
        )

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """In-model multiple-try proposal for a single branch with one leaf per walker.

        Flattens the alive leaf of every ``(temp, walker)``, tempers each with its rung's
        beta from ``self.temperature_control.betas``, passes the current
        ``log_like`` / ``log_prior`` of ``self.current_state`` as ``ll_in`` / ``lp_in`` and
        calls :meth:`get_mt_proposal`; ``mt_ll`` / ``mt_lp`` are reshaped to
        ``(ntemps, nwalkers)``.

        Args:
            branches_coords (dict): ONE branch, ``{branch_name: coords}`` of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``; every walker must have exactly one
                alive leaf.
            random (object): Sampler random state.
            branches_inds (dict, optional): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)``. ``None`` marks every leaf alive.
                Default ``None``.
            **kwargs: Ignored.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)`` with ``q[branch_name]`` of shape
            ``(ntemps, nwalkers, 1, ndim)`` and ``factors`` of shape ``(ntemps, nwalkers)``.

        Raises:
            ValueError: More than one branch, or a walker with more than one alive leaf.
        """

        # mutliple try is only made for one branch here
        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        # get main key
        key_in = list(branches_coords.keys())[0]
        self.key_in = key_in

        # get inds information
        if branches_inds is None:
            branches_inds = {}
            branches_inds[key_in] = np.ones(
                branches_coords[key_in].shape[:-1], dtype=bool
            )

        # Make sure for base proposals that there is only one leaf
        if np.any(branches_inds[key_in].sum(axis=-1) > 1):
            raise ValueError

        ntemps, nwalkers, _, _ = branches_coords[key_in].shape

        # get temperature information
        betas_here = np.repeat(
            self.temperature_control.betas[:, None],
            np.prod(branches_coords[key_in].shape[1:-1]),
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # previous Likelihoods in case proposal is independent
        ll_here = np.repeat(
            self.current_state.log_like[:, :, None],
            branches_coords[key_in].shape[2],
            axis=-1,
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # previous Priors in case proposal is independent
        lp_here = np.repeat(
            self.current_state.log_prior[:, :, None],
            branches_coords[key_in].shape[2],
            axis=-1,
        ).reshape(branches_inds[key_in].shape)[branches_inds[key_in]]

        # get mt proposal
        generated_points, factors = self.get_mt_proposal(
            branches_coords[key_in][branches_inds[key_in]],
            random,
            betas=betas_here,
            ll_in=ll_here,
            lp_in=lp_here,
        )

        # store this information for access outside
        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)

        return (
            {key_in: generated_points.reshape(ntemps, nwalkers, 1, -1)},
            factors.reshape(ntemps, nwalkers),
        )


class MultipleTryMoveRJ(MultipleTryMove):
    """Nested reversible-jump variant of :class:`MultipleTryMove`.

    Combine it with a :class:`~eryn.moves.ReversibleJumpMove` subclass that provides
    ``get_model_change_proposal`` (e.g. :class:`~eryn.moves.DistributionGenerateRJ`) and
    construct it with ``rj=True``. Births are multiple-try proposals; deaths are scored as
    the reverse of a birth into the one-leaf-fewer model. See
    :class:`~eryn.moves.MTDistGenMoveRJ`.
    """

    def get_proposal(
        self,
        branches_coords,
        branches_inds,
        nleaves_min_all,
        nleaves_max_all,
        random,
        **kwargs
    ):
        """Propose one multiple-try birth or a death per walker for a single branch.

        Uses ``get_model_change_proposal`` to choose births/deaths, evaluates the
        one-leaf-fewer model of every death walker through ``self.current_model`` (as the
        ``ll_in`` / ``lp_in`` of that walker), then calls :meth:`get_mt_proposal`. Needs
        ``self.temperature_control``, ``self.current_state`` and ``self.current_model`` (set
        by :meth:`~eryn.moves.ReversibleJumpMove.propose`).

        Args:
            branches_coords (dict): ONE branch, ``{branch_name: coords}`` of shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            branches_inds (dict): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)``; must not be ``None``.
            nleaves_min_all (dict): ``{branch_name: int}`` minimum leaf count (one entry).
            nleaves_max_all (dict): ``{branch_name: int}`` maximum leaf count (one entry).
            random (object): Sampler random state.
            **kwargs: Ignored.

        Returns:
            tuple[dict, dict, np.ndarray]: ``(q, new_inds, factors)``: coordinates with born
            leaves filled, ``inds`` with births on and deaths off, and log factors of shape
            ``(ntemps, nwalkers)``.

        Raises:
            ValueError: More than one branch, ``branches_inds`` is ``None``,
                ``nleaves_min >= nleaves_max``, or a change other than ``+1`` / ``-1``.
        """

        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        # get main key
        key_in = list(branches_coords.keys())[0]
        self.key_in = key_in

        if branches_inds is None:
            raise ValueError("In MT RJ proposal, branches_inds cannot be None.")

        ntemps, nwalkers, nleaves_max, ndim = branches_coords[key_in].shape

        # get temperature information
        betas_here = np.repeat(
            self.temperature_control.betas[:, None], nwalkers, axis=-1
        ).flatten()

        # current Likelihood and prior information
        ll_here = self.current_state.log_like.flatten()
        lp_here = self.current_state.log_prior.flatten()

        # do rj setup
        assert len(nleaves_min_all) == 1 and len(nleaves_max_all) == 1
        nleaves_min = nleaves_min_all[key_in]
        nleaves_max = nleaves_max_all[key_in]

        if nleaves_min == nleaves_max:
            raise ValueError("MT RJ proposal requires that nleaves_min != nleaves_max.")
        elif nleaves_min > nleaves_max:
            raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")

        # get the inds adjustment information
        all_inds_for_change = self.get_model_change_proposal(
            branches_inds[key_in], random, nleaves_min, nleaves_max
        )

        # preparing leaf information for going into the proposal
        inds_leaves_rj = np.zeros(ntemps * nwalkers, dtype=int)
        coords_in = np.zeros((ntemps * nwalkers, ndim))
        inds_reverse_rj = None

        # prepare proposal dictionaries
        new_inds = deepcopy(branches_inds)
        q = deepcopy(branches_coords)
        for change in all_inds_for_change.keys():
            if change not in ["+1", "-1"]:
                raise ValueError("MT RJ is only implemented for +1/-1 moves.")

            # get indicies of changing leaves
            temp_inds = all_inds_for_change[change][:, 0]
            walker_inds = all_inds_for_change[change][:, 1]
            leaf_inds = all_inds_for_change[change][:, 2]

            # leaf index to change
            inds_leaves_rj[temp_inds * nwalkers + walker_inds] = leaf_inds
            coords_in[temp_inds * nwalkers + walker_inds] = branches_coords[key_in][
                (temp_inds, walker_inds, leaf_inds)
            ]

            # adjustment of indices
            new_val = {"+1": True, "-1": False}[change]

            # adjust indices
            new_inds[key_in][(temp_inds, walker_inds, leaf_inds)] = new_val

            if change == "-1":
                # which walkers are removing
                inds_reverse_rj = temp_inds * nwalkers + walker_inds

        if inds_reverse_rj is not None:
            # setup reversal coords and inds
            # need to determine Likelihood and prior of removed binaries.
            # this goes into the multiple try proposal as previous ll and lp
            temp_reverse_coords = {}
            temp_reverse_inds = {}

            for key in self.current_state.branches:
                (
                    ntemps_tmp,
                    nwalkers_tmp,
                    nleaves_max_tmp,
                    ndim_tmp,
                ) = self.current_state.branches[key].shape

                # coords from reversal
                temp_reverse_coords[key] = self.current_state.branches[
                    key
                ].coords.reshape(ntemps_tmp * nwalkers_tmp, nleaves_max_tmp, ndim_tmp)[
                    inds_reverse_rj
                ][
                    None, :
                ]

                # which inds array to use
                inds_tmp_here = (
                    new_inds[key]
                    if key == key_in
                    else self.current_state.branches[key].inds
                )
                temp_reverse_inds[key] = inds_tmp_here.reshape(
                    ntemps * nwalkers, nleaves_max_tmp
                )[inds_reverse_rj][None, :]

            # calculate information for the reverse
            lp_reverse_here = self.current_model.compute_log_prior_fn(
                temp_reverse_coords, inds=temp_reverse_inds
            )[0]
            ll_reverse_here = self.current_model.compute_log_like_fn(
                temp_reverse_coords, inds=temp_reverse_inds, logp=lp_here
            )[0]

            # fill the here values
            ll_here[inds_reverse_rj] = ll_reverse_here
            lp_here[inds_reverse_rj] = lp_reverse_here

        # get mt proposal
        generated_points, factors = self.get_mt_proposal(
            coords_in,
            random,
            betas=betas_here,
            ll_in=ll_here,
            lp_in=lp_here,
            inds_leaves_rj=inds_leaves_rj,
            inds_reverse_rj=inds_reverse_rj,
        )

        # for reading outside
        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)

        # which walkers have information added
        inds_forward_rj = np.delete(np.arange(coords_in.shape[0]), inds_reverse_rj)

        # updated the coordinates
        temp_inds = all_inds_for_change["+1"][:, 0]
        walker_inds = all_inds_for_change["+1"][:, 1]
        leaf_inds = all_inds_for_change["+1"][:, 2]
        q[key_in][(temp_inds, walker_inds, leaf_inds)] = generated_points[
            inds_forward_rj
        ]

        return q, new_inds, factors.reshape(ntemps, nwalkers)
