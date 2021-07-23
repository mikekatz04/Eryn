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
    """Special container for one branch (model)

    This class is a key component of Eryn. It this type of object
    that allows for different models to be considered simultaneously
    within an MCMC run.

    Args:
        coords (4D double np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): The coordinates
            in parameter space of all walkers.
        inds (3D bool np.ndarray[ntemps, nwalkers, nleaves_max], optional): The information
            on which leaves are used and which are not used. A value of True means the specific leaf
            was used in this step. Parameters from unused walkers are still kept. When they
            are output to the backend, the backend saves a special number (default: ``np.nan``) for all coords
            related to unused leaves at that step. If None, inds will fill with all True values.
            (default: ``None``)

    Raises:
        ValueError: ``inds`` has wrong shape or number of leaves is less than zero.

    """
    def __init__(self, coords, inds=None):

        # store branch info
        self.coords = coords
        self.ntemps, self.ntrees, self.nleaves_max, self.ndim = coords.shape
        self.shape = coords.shape

        # make sure inds is correct
        if inds is None:
            self.inds = np.full((self.ntemps, self.ntrees, self.nleaves_max), True)
        elif not isinstance(inds, np.ndarray):
            raise ValueError("inds must be np.ndarray in Branch.")
        elif inds.shape != (self.ntemps, self.ntrees, self.nleaves_max):
            raise ValueError("inds has wrong shape.")
        else:
            self.inds = inds

        # get number of leaves in each walker by summing inds along last axis
        self.nleaves = np.sum(self.inds, axis=-1)

        if np.any(self.nleaves <= 0):
            # TODO: fix this (?) for non-nested models
            raise ValueError("Number of leaves <= 0 not allowed.")


class State(object):
    """The state of the ensemble during an MCMC run

    For backwards compatibility, this will unpack into ``coords, log_prob,
    (blobs), random_state`` when iterated over (where ``blobs`` will only be
    included if it exists and is not ``None``).

    Args:
        coords (double ndarray[ntemps, nwalkers, nleaves_max, ndim], dict, or :class:`.State`): The current positions of the walkers
            in the parameter space. If dict, need to use ``branch_names`` for the keys.
        inds (bool ndarray[ntemps, nwalkers, nleaves_max] or dict, optional): The information
            on which leaves are used and which are not used. A value of True means the specific leaf
            was used in this step. If dict, need to use ``branch_names`` for the keys.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        log_prob (ndarray[ntemps, nwalkers], optional): Log likelihoods
            for the  walkers at positions given by ``coords``.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        log_prior (ndarray[ntemps, nwalkers], optional): Log priors
            for the  walkers at positions given by ``coords``.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        betas (ndarray[ntemps], optional): Temperatures in the sampler at the current step.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        blobs (ndarray[ntemps, nwalkers, nblobs], Optional): The metadata “blobs”
            associated with the current position. The value is only returned if
            lnpostfn returns blobs too.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        random_state (Optional): The current state of the random number
            generator.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        copy (bool, optional): If True, copy the the arrays in the former :class:`.State` obhect.

    Raises:
        ValueError: Dimensions of inputs or input types are incorrect.

    """

    __slots__ = "branches", "log_prob", "log_prior", "blobs", "betas", "random_state"

    def __init__(
        self,
        coords,
        inds=None,
        log_prob=None,
        log_prior=None,
        betas=None,
        blobs=None,
        random_state=None,
        copy=False,
    ):
        # decide if copying input info
        dc = deepcopy if copy else lambda x: x

        # check if coords is a State object
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
            coords = {"model_0": atleast_4d(coords)}
        elif not isinstance(coords, dict) or not isinstance(coords, State):
            raise ValueError("Input coords need to be np.ndarray, dict, or State object.")

        for name in coords:
            if coords[name].ndim == 2:
                coords[name] = coords[name][None, :, None, :]

            # assume (ntemps, nwalkers) provided
            if coords[name].ndim == 3:
                coords[name] = coords[name][:, :, None, :]

            elif coords[name].ndim < 2 or coords[name].ndim > 4:
                raise ValueError(
                    "Dimension off coordinates must be between 2 and 4. coords dimension is {0}.".format(
                        coords.ndim
                    )
                )

        # if no inds given, make sure this is clear for all Branch objects
        if inds is None:
            inds = {key: None for key in coords}
        elif not isinstance(inds, dict):
            raise ValueError("inds must be None or dict.")

        # setup all information for storage
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
        """Get the ``inds`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.inds for name, branch in self.branches.items()}

    @property
    def branches_coords(self):
        """Get the ``coords`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.coords for name, branch in self.branches.items()}

    """
    # TODO
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
    """
