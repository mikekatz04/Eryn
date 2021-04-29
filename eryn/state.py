# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

__all__ = ["State"]


def atleast_nd(x, n):
    if not isinstance(x, np.ndarray):
        raise ValueError("Input value must be a numpy.ndarray.")

    elif x.ndim < n:
        ndim = x.ndim
        for _ in range(ndim, n):
            x = np.array([x])
    return x


def atleast_4d(x):
    return atleast_nd(x, 4)


def atleast_5d(x):
    return atleast_nd(x, 5)


class State(object):
    """The state of the ensemble during an MCMC run

    For backwards compatibility, this will unpack into ``coords, log_prob,
    (blobs), random_state`` when iterated over (where ``blobs`` will only be
    included if it exists and is not ``None``).

    Args:
        coords (ndarray[ntemps, nwalkers, nbranches, nleaves, ndim]): The current positions of the walkers
            in the parameter space.
        log_prob (ndarray[nwalkers, ndim], Optional): Log posterior
            probabilities for the  walkers at positions given by ``coords``.
        blobs (Optional): The metadata “blobs” associated with the current
            position. The value is only returned if lnpostfn returns blobs too.
        random_state (Optional): The current state of the random number
            generator.
    """

    __slots__ = "coords", "log_prob", "blobs", "betas", "random_state"

    def __init__(
        self,
        coords,
        log_prob=None,
        blobs=None,
        random_state=None,
        betas=None,
        copy=False,
    ):
        dc = deepcopy if copy else lambda x: x

        if hasattr(coords, "coords"):
            self.coords = dc(coords.coords)
            self.log_prob = dc(coords.log_prob)
            self.blobs = dc(coords.blobs)
            self.betas = dc(coords.betas)
            self.random_state = dc(coords.random_state)
            return

        self.coords = dc(atleast_5d(coords))
        self.log_prob = dc(np.atleast_2d(log_prob)) if log_prob is not None else None
        self.blobs = dc(np.atleast_3d(blobs)) if blobs is not None else None
        self.betas = dc(np.atleast_2d(betas)) if betas is not None else None
        self.random_state = dc(random_state)

    def __repr__(self):
        return "State({0}, log_prob={1}, blobs={2}, betas={3}, random_state={4})".format(
            self.coords, self.log_prob, self.blobs, self.betas, self.random_state
        )

    def __iter__(self):
        temp = (self.coords,)
        if self.log_prob is not None:
            temp += (self.log_prob,)

        if self.blobs is not None:
            temp += (self.blobs,)

        if self.betas is None:
            temp += (self.betas,)

        if self.random_state is not None:
            temp += (self.random_state,)
        return iter(temp)
