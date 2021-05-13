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


class Branch(object):
    def __init__(self, coords, inds=None):
        self.coords = coords
        self.ntemps, self.ntrees, self.nleaves_max, self.ndim = coords.shape
        self.shape = coords.shape

        if inds is None:
            self.inds = np.full((self.ntemps, self.ntrees, self.nleaves_max), True)
        elif not isinstance(inds, np.ndarray):
            raise ValueError("inds must be np.ndarray in Branch.")
        elif inds.shape != (self.ntemps, self.ntrees, self.nleaves_max):
            raise ValueError("inds has wrong shape.")
        else:
            self.inds = inds

        self.nleaves = np.sum(self.inds, axis=-1)

        if np.any(self.nleaves <= 0):
            raise ValueError("Number of leaves <= 0 not allowed.")


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

    __slots__ = "branches", "log_prob", "log_prior", "blobs", "betas", "random_state"

    def __init__(
        self,
        coords,
        log_prob=None,
        log_prior=None,
        blobs=None,
        random_state=None,
        betas=None,
        inds=None,
        copy=False,
    ):
        dc = deepcopy if copy else lambda x: x

        if hasattr(coords, "branches"):
            self.branches = dc(coords.branches)
            self.log_prob = dc(coords.log_prob)
            self.log_prior = dc(coords.log_prior)
            self.blobs = dc(coords.blobs)
            self.betas = dc(coords.betas)
            self.random_state = dc(coords.random_state)
            return

        # protect against simplifying settings
        if isinstance(coords, np.ndarray):
            # TODO: maybe adjust this later
            # assume just nwalkers provided
            if coords.ndim == 2:
                coords = coords[None, :, None, :]

            # assume (ntemps, nwalkers) provided
            if coords.ndim == 3:
                coords = coords[:, :, None, :]

            elif coords.ndim < 2 or coords.ndim > 4:
                raise ValueError(
                    "Dimension off coordinates must be between 2 and 4. coords dimension is {0}.".format(
                        coords.ndim
                    )
                )

            coords = {"model_0": atleast_4d(coords)}

        if inds is None:
            inds = {key: None for key in coords}
        elif not isinstance(inds, dict):
            raise ValueError("inds must be None or dict.")

        self.branches = {
            key: Branch(dc(temp_coords), inds=inds[key])
            for key, temp_coords in coords.items()
        }
        self.log_prob = dc(np.atleast_2d(log_prob)) if log_prob is not None else None
        self.log_prior = dc(np.atleast_2d(log_prior)) if log_prior is not None else None
        self.blobs = dc(np.atleast_3d(blobs)) if blobs is not None else None
        self.betas = dc(np.atleast_1d(betas)) if betas is not None else None
        self.random_state = dc(random_state)

    @property
    def branches_inds(self):
        return {name: branch.inds for name, branch in self.branches.items()}

    @property
    def branches_coords(self):
        return {name: branch.coords for name, branch in self.branches.items()}

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
