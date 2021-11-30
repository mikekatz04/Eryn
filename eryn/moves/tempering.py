# -*- coding: utf-8 -*-

import numpy as np
from ..state import State

__all__ = ["TemperatureControl"]

# TODO: add temperature control to existing proposal input by user


def make_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.
    :param ndim:
        The number of dimensions in the parameter space.
    :param ntemps: (optional)
        If set, the number of temperatures to generate.
    :param Tmax: (optional)
        If set, the maximum temperature for the ladder.
    Temperatures are chosen according to the following algorithm:
    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
      information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
      posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:
      * If ``Tmax = inf``, raise an exception (insufficient information).
      * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
    * If ``Tmax`` and ``ntemps`` are specified:
      * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
      * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.
    """

    if type(ndim) != int or ndim < 1:
        raise ValueError("Invalid number of dimensions specified.")
    if ntemps is None and Tmax is None:
        raise ValueError("Must specify one of ``ntemps`` and ``Tmax``.")
    if Tmax is not None and Tmax <= 1:
        raise ValueError("``Tmax`` must be greater than 1.")
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError("Invalid number of temperatures specified.")

    tstep = np.array(
        [
            25.2741,
            7.0,
            4.47502,
            3.5236,
            3.0232,
            2.71225,
            2.49879,
            2.34226,
            2.22198,
            2.12628,
            2.04807,
            1.98276,
            1.92728,
            1.87946,
            1.83774,
            1.80096,
            1.76826,
            1.73895,
            1.7125,
            1.68849,
            1.66657,
            1.64647,
            1.62795,
            1.61083,
            1.59494,
            1.58014,
            1.56632,
            1.55338,
            1.54123,
            1.5298,
            1.51901,
            1.50881,
            1.49916,
            1.49,
            1.4813,
            1.47302,
            1.46512,
            1.45759,
            1.45039,
            1.4435,
            1.4369,
            1.43056,
            1.42448,
            1.41864,
            1.41302,
            1.40761,
            1.40239,
            1.39736,
            1.3925,
            1.38781,
            1.38327,
            1.37888,
            1.37463,
            1.37051,
            1.36652,
            1.36265,
            1.35889,
            1.35524,
            1.3517,
            1.34825,
            1.3449,
            1.34164,
            1.33847,
            1.33538,
            1.33236,
            1.32943,
            1.32656,
            1.32377,
            1.32104,
            1.31838,
            1.31578,
            1.31325,
            1.31076,
            1.30834,
            1.30596,
            1.30364,
            1.30137,
            1.29915,
            1.29697,
            1.29484,
            1.29275,
            1.29071,
            1.2887,
            1.28673,
            1.2848,
            1.28291,
            1.28106,
            1.27923,
            1.27745,
            1.27569,
            1.27397,
            1.27227,
            1.27061,
            1.26898,
            1.26737,
            1.26579,
            1.26424,
            1.26271,
            1.26121,
            1.25973,
        ]
    )

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
    else:
        tstep = tstep[ndim - 1]

    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError(
                "Must specify at least one of ``ntemps" " and " "finite ``Tmax``."
            )

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with
        # infinity.
        betas = np.concatenate((betas, [0]))

    return betas


class TemperatureControl(object):
    def __init__(
        self,
        ndim,
        nwalkers,
        nleaves_max,
        ntemps=1,
        betas=None,
        Tmax=None,
        adaptive=True,
        adaptation_lag=10000,
        adaptation_time=100,
        stop_adaptation=-1,
    ):

        if betas is None:
            if ntemps == 1:
                betas = np.array([1.0])
            else:
                if len(ndim) > 1:
                    raise ValueError(
                        "If building a temp ladder, only done for one model."
                    )

                # A compromise for building a temperature ladder for the case of rj.
                # We start by assuming that the dimensionality will be defined by the number of
                # components. We take that maximum divided by two, and multiply it with the higher
                # dimensional component.
                if sum(nleaves_max) > 1:
                    betas = make_ladder(
                        int(max(ndim) * sum(nleaves_max) / 2), ntemps=ntemps, Tmax=Tmax
                    )
                else:
                    betas = make_ladder(ndim[0], ntemps=ntemps, Tmax=Tmax)

        self.nwalkers = nwalkers
        self.betas = betas
        self.ntemps = ntemps = len(betas)

        self.time = 0

        self.adaptive = adaptive
        self.adaptation_time, self.adaptation_lag = adaptation_time, adaptation_lag
        self.stop_adaptation = stop_adaptation

        # TODO: read this information out (specific to each proposal maybe)
        self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)

    def compute_log_posterior_tempered(self, logl, logp, betas=None):
        tempered_logl = self._tempered_likelihood(logl, betas=betas)
        return tempered_logl + logp

    def _tempered_likelihood(self, logl, betas=None):
        """
        Compute tempered log likelihood.  This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.
        Here, we find a singularity that demands more careful attention; we allow the likelihood to dominate the
        temperature, since wandering outside the likelihood support causes a discontinuity.
        """

        if betas is None:
            betas = self.betas

        with np.errstate(invalid="ignore"):
            loglT = logl * betas[:, None]

        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def temperature_swaps(self, x, logP, logl, logp, inds=None, blobs=None):
        """
        Perform parallel-tempering temperature swaps on the state in ``x`` with associated ``logP`` and ``logl``.
        """

        ntemps, nwalkers = self.ntemps, self.nwalkers
        self.swaps_accepted = np.empty(ntemps - 1)

        for i in range(ntemps - 1, 0, -1):
            bi = self.betas[i]
            bi1 = self.betas[i - 1]

            dbeta = bi1 - bi

            iperm = np.random.permutation(nwalkers)
            i1perm = np.random.permutation(nwalkers)

            raccept = np.log(np.random.uniform(size=nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[i - 1, i1perm])

            # How many swaps were accepted?
            sel = paccept > raccept
            self.swaps_accepted[i - 1] = np.sum(sel)

            # for x and inds, just do full copy
            x_temp = {name: np.copy(x[name]) for name in x}
            if inds is not None:
                inds_temp = {name: np.copy(inds[name]) for name in inds}

            logl_temp = np.copy(logl[i, iperm[sel]])
            logp_temp = np.copy(logp[i, iperm[sel]])
            logP_temp = np.copy(logP[i, iperm[sel]])
            if blobs is not None:
                blobs_temp = np.copy(blobs[i, iperm[sel]])

            for name in x:
                x[name][i, iperm[sel], :, :] = x[name][i - 1, i1perm[sel], :, :]
                if inds is not None:
                    inds[name][i, iperm[sel], :] = inds[name][i - 1, i1perm[sel], :]

            logl[i, iperm[sel]] = logl[i - 1, i1perm[sel]]
            logp[i, iperm[sel]] = logp[i - 1, i1perm[sel]]
            logP[i, iperm[sel]] = (
                logP[i - 1, i1perm[sel]] - dbeta * logl[i - 1, i1perm[sel]]
            )
            if blobs is not None:
                blobs[i, iperm[sel]] = blobs[i - 1, i1perm[sel]]

            for name in x:
                x[name][i - 1, i1perm[sel], :, :] = x_temp[name][i, iperm[sel], :, :]
                if inds is not None:
                    inds[name][i - 1, i1perm[sel], :] = inds_temp[name][
                        i, iperm[sel], :
                    ]

            logl[i - 1, i1perm[sel]] = logl_temp
            logp[i - 1, i1perm[sel]] = logp_temp
            logP[i - 1, i1perm[sel]] = logP_temp + dbeta * logl_temp

            if blobs is not None:
                blobs[i - 1, i1perm[sel]] = blobs_temp

        return x, logP, logl, logp, inds, blobs

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
        """
        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def temper_comps(self, state, accepted):
        logl = state.log_prob
        logp = state.log_prior
        logP = self.compute_log_posterior_tempered(logl, logp)
        x, logP, logl, logp, inds, blobs = self.temperature_swaps(
            state.branches_coords,
            logP.copy(),
            logl.copy(),
            logp.copy(),
            inds=state.branches_inds,
            blobs=state.blobs,
        )

        ratios = self.swaps_accepted / self.swaps_proposed

        if self.adaptive and self.ntemps > 1:
            if self.stop_adaptation < 0 or self.time < self.stop_adaptation:
                dbetas = self._get_ladder_adjustment(self.time, self.betas, ratios)
                self.betas += dbetas

        new_state = State(
            x,
            log_prob=logl,
            log_prior=logp,
            blobs=blobs,
            inds=inds,
            betas=self.betas,
            random_state=state.random_state,
        )

        self.time += 1

        return new_state, accepted
