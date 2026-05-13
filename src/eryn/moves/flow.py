# -*- coding: utf-8 -*-

"""Normalizing flow-based MCMC proposal moves.

This module implements Metropolis-Hastings moves that use normalizing flows
as the proposal distribution. The normalizing flow is expected to be pre-trained
and is provided as some `flow` object to the proposal class.

The implementation relies on the `coppuccino` package for normalizing flow
operations (see https://github.com/aarondjohnson/coppuccino). Future development
could generalize this to support other flow libraries.
"""

import numpy as np
from .mh import MHMove

try:
    from coppuccino import sample, log_prob
except ImportError:
    pass

__all__ = ["FlowMove"]


class FlowMove(MHMove):
    """A Metropolis step with a proposal from a normalizing flow.

    This move generates proposals by sampling from a pre-trained normalizing
    flow model.

    Args:
        flow (object): A trained normalizing flow model compatible with the `coppuccino` package.
            Must support `sample()` and `log_prob()` functions for generating proposals
            and computing log densities.
        return_gpu (bool, optional): If ``use_gpu == True`` and
            ``return_gpu == True``, returned arrays remain on GPU. (default: ``False``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    """

    def __init__(self, flow, return_gpu=False, **kwargs):
        self.flow = flow
        self.return_gpu = return_gpu
        super(FlowMove, self).__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate proposal coordinates by sampling from normalizing flow.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and
                values are ``np.ndarray[ntemps, nwalkers, nleaves_max]``
                indicating active leaves. (default: ``None``)
            **kwargs (dict, optional): Extra keyword arguments for proposal.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, ndarray). The
                factors are a ``ndarray[ntemps, nwalkers]`` of the change in flow density
                between the current and proposed states.

        """

        # initialize output
        q = {}
        for i, (name, coords) in enumerate(zip(branches_coords.keys(), branches_coords.values())):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # get the proposal for this branch
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get flow density at current state
            flow_density_i = log_prob(self.flow, q[name][inds_here]).reshape((ntemps, nwalkers))

            # propose new state from normalizing flow
            new_coords = sample(
                self.flow, n_samples=1, rng_seed=random.randint(1e10)
            )  # arbitrary randint threshold, can change

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

            # get flow density at proposed state
            flow_density_f = log_prob(self.flow, q[name][inds_here]).reshape((ntemps, nwalkers))

            # update factors
            factors += flow_density_i - flow_density_f

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

        # if running on GPU but requested CPU returns, trasfer arrays back
        if self.use_gpu and not self.return_gpu:
            for name, arr in list(q.items()):
                q[name] = arr.get()
            factors = factors.get()

        return q, factors
