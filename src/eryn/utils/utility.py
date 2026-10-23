# -*- coding: utf-8 -*-

"""Numerical helpers for the sampler and its diagnostics.

- :func:`groups_from_inds`: per-leaf group (walker) ids for the vectorised likelihood and
  prior calls.
- :func:`get_acf`, :func:`get_integrated_act`: FFT autocorrelation function and
  integrated autocorrelation time.
- :func:`thermodynamic_integration_log_evidence`,
  :func:`stepping_stone_log_evidence`: evidence estimates from a temperature ladder.
- :func:`psrf`: Gelman-Rubin potential scale reduction factor.
- ``NDArrayLike``: type alias (``np.ndarray`` at runtime) used in annotations.
"""

import warnings

import numpy as np
from scipy.special import logsumexp


def groups_from_inds(inds):
    """Flat group id of every active leaf, per branch.

    The group id of temperature ``t`` and walker ``w`` is ``t * nwalkers + w``. For each
    branch the ids of the ``True`` entries of ``inds`` are returned in C order, which is
    the order of ``coords[inds]``. The sampler uses this to tell a vectorised likelihood /
    prior which walker each leaf row belongs to.

    Args:
        inds (dict): ``{branch_name: bool array}`` marking the active leaves, each of
            shape ``(ntemps, nwalkers, nleaves_max)``.

    Returns:
        dict: ``{branch_name: int np.ndarray}``, flat, one entry per active leaf.

    Example:
        >>> import numpy as np
        >>> from eryn.utils.utility import groups_from_inds
        >>> inds = {"model_0": np.array([[[True, False], [True, True]]])}
        >>> groups_from_inds(inds)["model_0"].tolist()
        [0, 1, 1]
    """
    # prepare output
    groups = {}
    for name, inds_temp in inds.items():

        # shape information
        ntemps, nwalkers, nleaves_max = inds_temp.shape
        num_groups = ntemps * nwalkers

        # place which group each active leaf belongs to along flattened array
        group_id = np.repeat(
            np.arange(num_groups).reshape(ntemps, nwalkers)[:, :, None],
            nleaves_max,
            axis=-1,
        )

        # fill new information
        groups[name] = group_id[inds_temp]

    return groups


def get_acf(x, axis=0, fast=False):
    """Normalised autocorrelation function of a time series, via the FFT.

    The mean along ``axis`` is removed, the series is zero-padded to ``2 n`` and the ACF is
    normalised so lag 0 equals 1. Computed independently along every other axis.

    Args:
        x (array_like): The time series.
        axis (int, optional): Time axis. Default ``0``. (The normalisation broadcasts
            correctly only for ``axis=0``.)
        fast (bool, optional): If ``True``, ``n`` is the largest power of two not above the
            series length (FFT length ``2 n``, first ``n`` lags returned). The input itself
            is not cropped (the crop line is a no-op). Default ``False`` (``n`` = length).

    Returns:
        np.ndarray: ACF for lags ``0 .. n - 1`` along ``axis``; other axes as in ``x``.
    """

    x = np.atleast_1d(x)
    m = [
        slice(None),
    ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2 ** np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]


def get_integrated_act(x, axis=0, window=50, fast=False, average=True):
    """Integrated autocorrelation time with a fixed window.

    ``tau = 1 + 2 * sum(acf[1:window])`` along the time axis, with the ACF from
    :func:`get_acf` (see Sokal's notes on MCMC and sample estimators for autocorrelation
    times).

    Args:
        x (np.ndarray or dict): Time series with time on axis 0; or a dict
            ``{branch_name: chain}`` of 5-D chains
            ``(nsteps, ntemps, nwalkers, nleaves_max, ndim)`` (as from
            ``backend.get_chain()``), which are flattened to
            ``(nsteps, ntemps, nwalkers, nleaves_max * ndim)`` and concatenated on the last
            axis.
        axis (int, optional): Time axis; only ``0`` is supported. Default ``0``.
        window (int, optional): Window end: lags ``1 .. window - 1`` are summed. Default
            ``50``.
        fast (bool, optional): Forwarded to :func:`get_acf`. Default ``False``.
        average (bool, optional): If ``True``, average ``tau`` over axis 1 of the result,
            i.e. axis 2 of the input (the walkers of a ``(nsteps, ntemps, nwalkers, ...)``
            chain); needs an input of at least 3 dimensions. Default ``True``.

    Returns:
        float, np.ndarray or dict: A float for 1-D ``x``; otherwise an array of shape
        ``x.shape[1:]`` (with axis 1 averaged out when ``average``); for dict input a dict
        of per-branch arrays split along the last axis.

    Raises:
        NotImplementedError: If ``axis != 0``.
        ValueError: If ``x`` is neither a dict nor an ``np.ndarray``.

    Note:
        For dict input the split points advance by ``ndim`` per branch, not by
        ``nleaves_max * ndim``, so the per-branch split is only right when
        ``nleaves_max == 1``.
    """

    if axis != 0:
        # TODO: need to check this
        raise NotImplementedError

    # Compute the autocorrelation function.
    if isinstance(x, dict):
        is_dict = True
        ndim_total = 0
        values_out = []
        ind_breaks = []
        for name, values in x.items():
            nsteps, ntemps, nwalkers, nleaves_max, ndim = values.shape
            ndim_total += ndim
            ind_breaks.append(ndim_total)
            values_out.append(values.reshape(nsteps, ntemps, nwalkers, -1))

        x_in = np.concatenate(values_out, axis=-1)

    elif isinstance(x, np.ndarray):
        is_dict = False
        x_in = x
    else:
        raise ValueError("x must be dictionary of np.ndarrays or an np.ndarray.")

    f = get_acf(x_in, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2 * np.sum(f[1:window])

    # N-dimensional case.
    m = [
        slice(None),
    ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2 * np.sum(f[tuple(m)], axis=axis)

    if average:
        tau = np.average(tau, axis=1)

    if is_dict:
        splits = np.split(tau, ind_breaks, axis=-1)
        out = {name: split for name, split in zip(x.keys(), splits)}

    else:
        out = tau

    return out


def thermodynamic_integration_log_evidence(betas, logls):
    r"""Thermodynamic-integration estimate of the log-evidence (from ``ptemcee``).

    The evidence is the integral of the un-normalised posterior,

    .. math::
        Z \equiv \int d\theta \, l(\theta) p(\theta).

    With :math:`Z(\beta) = \int d\theta \, l^\beta(\theta) p(\theta)`,

    .. math::
        \frac{d \log Z}{d \beta}
        = \frac{1}{Z(\beta)} \int d\theta \, l^\beta p \log l
        = \left\langle \log l \right\rangle_\beta ,

    so

    .. math::
        \log Z(1) - \log Z(0)
        = \int_0^1 d\beta \left\langle \log l \right\rangle_\beta .

    The integral is evaluated with the trapezoidal rule over the ladder. If the hottest
    ``beta`` is not 0, a ``beta = 0`` point is appended with the hottest chain's mean
    log-likelihood. The error estimate is the difference from the same integral using
    every other temperature.

    Args:
        betas (np.ndarray): Inverse temperatures, shape ``(ntemps,)`` (any order; sorted
            internally).
        logls (np.ndarray): Mean log-likelihood at each ``beta``, shape ``(ntemps,)``.

    Returns:
        tuple[float, float]: ``(logZ, dlogZ)``, the log-evidence estimate and its
        ladder-discretisation error.

    Raises:
        ValueError: If ``betas`` and ``logls`` have different lengths.
    """

    # make sure they are the same length
    if len(betas) != len(logls):
        raise ValueError("Need the same number of log(L) values as temperatures.")

    # make sure they are in order
    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0.0:
        betas = np.concatenate((betas0, [0.0]))
        betas2 = np.concatenate((betas0[::2], [0.0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0.0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    # integrate by trapezoid rule (np.trapz was removed in numpy 2.0)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    logZ = -trapezoid(logls, betas)
    logZ2 = -trapezoid(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)


def stepping_stone_log_evidence(betas, logls, block_len=50, repeats=100):
    """Stepping-stone estimate of the log-evidence, with a block-bootstrap error.

    Based on Maturana-Russel et al. (2019), https://arxiv.org/abs/1810.04488, and Xie et
    al. (2011), doi:10.1093/sysbio/syq085. With the ladder sorted to ascending ``beta``,
    each adjacent pair contributes the log of the sample mean of
    ``exp((beta_{k+1} - beta_k) * logL)`` over the samples at ``beta_k``; the sum estimates
    ``log(Z(beta_max) / Z(beta_min))``, which is ``log Z`` when the ladder runs from 0 to 1.
    The samples of each temperature are pooled over steps and walkers
    (``(nsteps, ntemps, nwalkers)`` -> ``(nsteps * nwalkers, ntemps)``), and each pair term
    is normalised by ``log(n)`` once (``n`` = number of kept rows). Sample rows with any
    non-finite log-likelihood are dropped.

    The error is the standard deviation over ``repeats`` estimates, each built from one
    random row of every sliding window ``[i, i + block_len)``. If that raises
    ``ValueError``, a warning is issued and ``dlogZ`` is ``nan``.

    Args:
        betas (np.ndarray): Inverse temperatures, shape ``(ntemps,)``.
        logls (np.ndarray): Log-likelihood samples, shape ``(nsteps, ntemps, nwalkers)``
            (as from ``backend.get_log_like()``).
        block_len (int, optional): Window length of the bootstrap. Default ``50``.
        repeats (int, optional): Number of bootstrap estimates. Default ``100``.

    Returns:
        tuple[float, float]: ``(logZ, dlogZ)``.

    Raises:
        ValueError: If ``logls.shape[1] != len(betas)``.

    Note:
        Callers: :func:`~eryn.utils.plot.plot_logl_betas` and
        :meth:`~eryn.backends.Backend.get_evidence_estimate` (stepping-stone method), both
        passing the ``(nsteps, ntemps, nwalkers)`` array described above.
    """

    def calculate_stepping_stone(betas, logls):
        """Return ``(sum of pair terms, per-pair terms)`` for ``logls`` of shape ``(n, ntemps)``."""
        n = logls.shape[0]
        n_T = betas.shape[0]
        delta_betas = betas[1:] - betas[:-1]
        throwaways = np.any(~np.isfinite(logls), axis=1) # a safeguard against non-finite entries
        # each ratio is a mean over the retained samples: normalise each by log(n) once
        log_ratio = logsumexp(delta_betas * logls[~throwaways, :-1], axis=0) - np.log(n - np.sum(throwaways))
        return np.sum(log_ratio), log_ratio
    
    # make sure they are the same length
    if len(betas) != logls.shape[1]:
        raise ValueError(
            "Need the log(L).shape[1] to be the same as the number of temperatures."
        )

    # make sure they are in order
    order = np.argsort(betas)
    betas = betas[order]
    logls = logls[:, order, :]
    # Get all samples per temperature: (steps, ntemps, nwalkers) -> (steps * nwalkers, ntemps)
    logls = np.moveaxis(logls, 1, -1).reshape(-1, betas.shape[0])
    steps = logls.shape[0]  # Get number of samples

    logZ, _ = calculate_stepping_stone(betas, logls)

    # Estimate the evidence uncertainty (Maturana-Russel et. al. (2019))
    logZ_i = np.zeros(repeats)
    try:
        for i in range(repeats):
            idxs = [
                np.random.randint(i, i + block_len) for i in range(steps - block_len)
            ]
            logZ_i[i] = calculate_stepping_stone(betas, logls[idxs, :])[0]
        dlogZ = np.std(logZ_i)
    except ValueError:
        warnings.warn(
            "Warning: Failed to compute evidence uncertainty via Stepping Stone algorithm"
        )
        dlogZ = np.nan

    return logZ, dlogZ


def psrf(C, ndims, per_walker=False):
    """Gelman-Rubin potential scale reduction factor ``Rhat`` per parameter.

    Compares within-chain and between-chain variances; values well above 1 indicate
    non-convergence. With ``per_walker=False`` (default) all samples are pooled
    (``C.reshape(-1, ndims)``) and the first and last thirds are treated as two chains.
    With ``per_walker=True``, ``C`` must already be ``(nchains, nsamples, ndims)``.

    Based on Brooks & Gelman (1998), J. Comput. Graph. Stat. 7, 434-455, and Gelman &
    Rubin (1992), Stat. Sci. 7, 457-511; code adapted from
    https://joergdietrich.github.io/emcee-convergence.html.

    Args:
        C (np.ndarray): Parameter traces; any shape whose size is a multiple of ``ndims``
            when ``per_walker=False``, else ``(nchains, nsamples, ndims)``.
        ndims (int): Number of parameters (last axis).
        per_walker (bool, optional): Treat the leading axis of ``C`` as separate chains
            instead of splitting the pooled samples. Default ``False``.

    Returns:
        np.ndarray: ``Rhat``, shape ``(ndims,)``.
    """
    if not per_walker:
        # Split the complete chains into three parts and perform the
        # diagnostic on the forst and last 1/3 of the chains.
        C = C.reshape(-1, ndims)
        n = int(np.floor(C[:, 0].shape[0] / 3))
        c1 = C[0:n, :]
        c2 = C[-n:, :]
        C = np.zeros((2, c1.shape[0], c1.shape[1]))
        C = np.array([c1, c2])

    ssq = np.var(C, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    θb = np.mean(C, axis=1)
    θbb = np.mean(θb, axis=0)
    m = C.shape[0]
    nn = C.shape[1]
    B = nn / (m - 1) * np.sum((θbb - θb) ** 2, axis=0)

    var_θ = (nn - 1) / nn * W + 1 / nn * B
    R̂ = np.sqrt(var_θ / W)
    return R̂

from typing import TYPE_CHECKING, TypeAlias, Union
if TYPE_CHECKING:
    # Strict typing context: Pylance will use this branch.
    try:
        import cupy as cp # type: ignore[import]
        NDArrayLike: TypeAlias = Union[np.ndarray, cp.ndarray]
    except ImportError:
        NDArrayLike: TypeAlias = np.ndarray
else:
    # Runtime context: Python executes this branch without loading cupy.
    # Set to np.ndarray so it evaluates as a valid Type at runtime without crashing.
    # (NDArrayLike is the "numpy or cupy array" alias used in the priors' annotations.)
    NDArrayLike: TypeAlias = np.ndarray  