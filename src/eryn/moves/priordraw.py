# -*- coding: utf-8 -*-

import numpy as np
from .mh import MHMove

__all__ = ["PriorDraw"]


class PriorDraw(MHMove):
    "A Metropolis step with a proposal distribution drawing directly from the priors."

    def __init__(self, priors, return_gpu=False, **kwargs):
        if not isinstance(priors, dict):
            self.priors = {"model_0": priors}
        else:
            self.priors = priors
        self.return_gpu = return_gpu
        super(PriorDraw, self).__init__(**kwargs)

    def _compute_log_prior(self, coords, inds=None, supps=None, branch_supps=None):
        """Calculate the vector of log-prior for the walkers.
        This is copied directly from `ensemble.py` since the Move object has no
        external knowledge of the sampler object and its attributes.

        Args:
            coords (dict): Keys are ``branch_names`` and values are
                the position np.arrays[ntemps, nwalkers, nleaves_max, ndim].
                This dictionary is created with the ``branches_coords`` attribute
                from :class:`State`.
            inds (dict, optional): Keys are ``branch_names`` and values are
                the ``inds`` np.arrays[ntemps, nwalkers, nleaves_max] that indicates
                which leaves are being used. This dictionary is created with the
                ``branches_inds`` attribute from :class:`State`.
                (default: ``None``)

        Returns:
            np.ndarray[ntemps, nwalkers]: Prior Values

        """

        # get number of temperature and walkers
        ntemps, nwalkers, _, _ = coords[list(coords.keys())[0]].shape

        if inds is None:
            # default use all sources
            inds = {
                name: np.full(coords[name].shape[:-1], True, dtype=bool)
                for name in coords
            }

        # take information out of dict and spread to x1..xn
        x_in = {}

        # flatten coordinate arrays
        for i, (name, coords_i) in enumerate(coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords_i.shape

            x_in[name] = coords_i.reshape(-1, ndim)

        prior_out = np.zeros((ntemps, nwalkers))
        for name in x_in:
            ntemps, nwalkers, nleaves_max, ndim = coords[name].shape
            prior_out_temp = (
                self.priors[name]
                .logpdf(x_in[name])
                .reshape(ntemps, nwalkers, nleaves_max)
            )

            # fix any infs / nans from binaries that are not being used (inds == False)
            prior_out_temp[~inds[name]] = 0.0

            # vectorized because everything is rectangular (no groups to indicate model difference)
            prior_out += prior_out_temp.sum(axis=-1)

        if np.any(np.isnan(prior_out)):
            raise ValueError("The prior function is returning Nan.")

        return prior_out

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):

        # initialize output
        q = {}
        if branches_inds is None:
            branches_inds = {
                name: np.ones_like(coords, dtype=bool)
                for name, coords in branches_coords.items()
            }
        for i, (name, coords) in enumerate(
            zip(branches_coords.keys(), branches_coords.values())
        ):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly (need to remain as dict)
            inds = branches_inds[name]

            # get the proposal for this branch
            inds_here = np.where(inds == True)

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # copy coords
            q[name] = coords.copy()

            # get log prior at current point
            log_prior_i = self._compute_log_prior({name: q[name]}, inds=branches_inds)

            # propose new state
            new_coords = self.priors[name].rvs(size=(ntemps, nwalkers, nleaves_max))

            # put into coords in proper location
            q[name][inds_here] = new_coords[inds_here].copy()

            # get log prior at new point
            log_prior_f = self._compute_log_prior({name: q[name]}, inds=branches_inds)
            factors += log_prior_i - log_prior_f

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, factors
