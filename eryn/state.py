# -*- coding: utf-8 -*-

from copy import deepcopy

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    import numpy as np

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


class BranchSupplimental(object):
    def __init__(self, obj_info: dict, obj_contained_shape=None, copy=False):  # obj_contained, obj_contained_shape):

        self.holder = {}
        self.shape = None
        self.add_objects(obj_info, obj_contained_shape=obj_contained_shape, copy=copy)

    def add_objects(self, obj_info: dict, obj_contained_shape=None, copy=False):

        if self.shape is not None and obj_contained_shape is not None:
            if self.shape != obj_contained_shape:
                raise ValueError(f"Shape of input object ({obj_contained_shape}) not equivalent to established shape ({self.shape}).")
        elif obj_contained_shape is None and self.shape is not None:
            obj_contained_shape = self.shape
        
        dc = deepcopy if copy else (lambda x: x)
        
        for name, obj_contained in obj_info.items():
            # TODO: add cupy
            if isinstance(obj_contained, np.ndarray) and obj_contained.dtype.name == "object":
                # TODO: need to copy?
                self.holder[name] = dc(obj_contained)
                if obj_contained_shape is None:
                    obj_contained_shape = self.holder[name].shape
                    self.ndim = ndim =  len(obj_contained_shape)
                else:
                    if self.holder[name].shape != obj_contained_shape:
                        raise ValueError(f"Outer shapes of all input objects must be the same. {name} object array has shape {self.holder[name].shape}. The original shape found was {obj_contained_shape}.")

            elif obj_contained_shape is None:
                raise ValueError("If obj_contained is not an already built object array, obj_contained_shape cannot be None.")

            
            else:
                self.ndim = ndim =  len(obj_contained_shape)
                
                # xp for GPU
                if isinstance(obj_contained, np.ndarray) or isinstance(obj_contained, xp.ndarray):
                    self.holder[name] = obj_contained.copy()

                else:
                    self.holder[name] = np.empty(obj_contained_shape, dtype=object)
                    if len(obj_contained) != obj_contained_shape[0]:
                        raise ValueError("Shapes of obj_contained does not match obj_contained_shape along axis 0.")

                    if ndim > 1:
                        for i in range(obj_contained_shape[0]):
                            if len(obj_contained[i]) != obj_contained_shape[1]:
                                    raise ValueError("Shapes of obj_contained does not match obj_contained_sha along axis 1.")

                            if ndim > 2:
                                for j in range(obj_contained_shape[1]):
                                    if len(obj_contained[i][j]) != obj_contained_shape[2]:
                                        raise ValueError("Shapes of obj_contained does not match obj_contained_shape along axis 2.")
                                    
                                    for k in range(obj_contained_shape[2]):
                                        # TODO: copy?
                                        self.holder[name][i, j, k] = obj_contained[i][j][k]
                            else:
                                for j in range(obj_contained_shape[1]):
                                    self.holder[name][i, j] = obj_contained[i][j]

                    else:
                        for i in range(obj_contained_shape[0]):
                            self.holder[name][i] = obj_contained[i]

        if self.shape is None:
            self.shape = obj_contained_shape
            self.ndim = len(self.shape)

    def remove_objects(self, names):
        if not isinstance(names, list):
            if not isinstance(names, str):
                raise ValueError("names must be a string or list of strings.")

            names = [names]
        for name in names:
            self.holder.pop(name)

    @property
    def contained_objects(self):
        return list(self.holder.keys())
    
    def __contains__(self, name: str):
        return (name in self.holder)
            
    def __getitem__(self, tmp):
        return {name: values[tmp] for name, values in self.holder.items()}

    def __setitem__(self, tmp, new_value):
        for name, values in self.holder.items():
            if name not in new_value:
                continue
            
            self.holder[name][tmp] = new_value[name]

    def take_along_axis(self, indices, axis, skip_names=[]):
        out = {}
        
        for name, values in self.holder.items():
            if name in skip_names:
                continue

            indices_temp = indices.copy()
            if (isinstance(values, np.ndarray) and values.dtype.name != "object") or isinstance(values, xp.ndarray):
                for _ in range(values.ndim - indices_temp.ndim):
                    try:
                        indices_temp = np.expand_dims(np.asarray(indices_temp), (-1,))
                    except TypeError:
                        indices_temp = xp.expand_dims(xp.asarray(indices_temp), (-1,))
            try:
                out[name] = np.take_along_axis(values, indices_temp, axis)

            except TypeError:
                out[name] = xp.take_along_axis(values, indices_temp, axis)

            except (ValueError, IndexError) as e:
                breakpoint()
        return out

    def put_along_axis(self, indices, values_in, axis):
        for name, values in self.holder.items():
            if name not in values_in:
                continue
            indices_temp = indices.copy()
            if (isinstance(values, np.ndarray) and values.dtype.name != "object") or isinstance(values, xp.ndarray):
                for _ in range(values.ndim - indices_temp.ndim):
                    try:
                        indices_temp = np.expand_dims(np.asarray(indices_temp), (-1,))
                    except TypeError:
                        indices_temp = xp.expand_dims(xp.asarray(indices_temp), (-1,))

            try:
                inds0 = np.repeat(np.arange(len(indices_temp))[:, None], indices_temp.shape[1], axis=1)
            except TypeError:
                inds0 = xp.repeat(np.arange(len(indices_temp))[:, None], indices_temp.shape[1], axis=1)
            #self.xp.put_along_axis(self.holder[name], indices_temp, values_in[name], axis)
            # because cupy does not have put_along_axis
            try:
                self.holder[name][(inds0.flatten(), indices_temp.flatten())] = values_in[name].reshape((-1,) + values_in[name].shape[2:])
            except ValueError:
                breakpoint()

    @property
    def flat(self):
        out = {}
        for name, values in self.holder.items():
            if (isinstance(values, np.ndarray) and values.dtype.name != "object") or isinstance(values, xp.ndarray):
                out[name] = values.reshape((-1,) + values.shape[2:])
            else:
                out[name] = values.flatten()
        return out


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

    def __init__(self, coords, inds=None, branch_supplimental=None):

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

        if branch_supplimental is not None:
            if branch_supplimental.shape != self.inds.shape:
                raise ValueError(f"branch_supplimental shape ( {branch_supplimental.shape} ) does not match inds shape ( {self.inds.shape} ).")
            
        self.branch_supplimental = branch_supplimental
        # verify no 0 nleaves walkers
        self.nleaves

    @property
    def nleaves(self):
        """Number of leaves for each walker"""
        # get number of leaves in each walker by summing inds along last axis
        nleaves = np.sum(self.inds, axis=-1)
        return nleaves


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

    __slots__ = "branches", "log_prob", "log_prior", "blobs", "betas", "supplimental", "random_state"

    def __init__(
        self,
        coords,
        inds=None,
        branch_supplimental=None,
        supplimental=None,
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
            self.supplimental = dc(coords.supplimental)
            self.random_state = dc(coords.random_state)
            return

        # protect against simplifying settings
        if isinstance(coords, np.ndarray):
            coords = {"model_0": coords}
        elif not isinstance(coords, dict):
            raise ValueError(
                "Input coords need to be np.ndarray, dict, or State object."
            )

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

        if branch_supplimental is None:
            branch_supplimental = {key: None for key in coords}
        elif not isinstance(branch_supplimental, dict):
            raise ValueError("branch_supplimental must be None or dict.")

        # setup all information for storage
        self.branches = {
            key: Branch(dc(temp_coords), inds=inds[key], branch_supplimental=branch_supplimental[key])
            for key, temp_coords in coords.items()
        }
        self.log_prob = dc(np.atleast_2d(log_prob)) if log_prob is not None else None
        self.log_prior = dc(np.atleast_2d(log_prior)) if log_prior is not None else None
        self.blobs = dc(np.atleast_3d(blobs)) if blobs is not None else None
        self.betas = dc(np.atleast_1d(betas)) if betas is not None else None
        self.supplimental = dc(supplimental) 
        self.random_state = dc(random_state)

    @property
    def branches_inds(self):
        """Get the ``inds`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.inds for name, branch in self.branches.items()}

    @property
    def branches_coords(self):
        """Get the ``coords`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.coords for name, branch in self.branches.items()}

    @property
    def branches_supplimental(self):
        """Get the ``branch.supplimental`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.branch_supplimental for name, branch in self.branches.items()}

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
