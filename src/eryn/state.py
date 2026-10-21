# -*- coding: utf-8 -*-

import warnings
from copy import deepcopy

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

import numpy as np

__all__ = ["State"]

def return_x(x):
    return x

class BranchSupplemental(object):
    """Special object to carry information through sampler.

    The :class:`BranchSupplemental` object is a holder of information that is
    passed through the sampler. It can also be indexed similar to other quantities
    carried throughout the sampler.

    This indexing is based on the ``base_shape``. You can store many objects that have the same base
    shape and then index across all of them. For example, if you want to store individual leaf
    information, the base shape will be ``(ntemps, nwalkers, nleaves_max)``.
    If you want to store a 2D array per individual leaf, the overall shape will be
    ``(ntemps, nwalkers, nleaves_max, dim2_extra, dim1_extra)``. Another type of information is
    stored in a class object (for example). Using ``numpy`` object arrays,
    ``ntemps * nwalkers * nleaves_max`` number of class objects can be stored in the array. Then,
    using special indexing functions, information can be updated/accessed across all objects
    stored simultaneously. If you index this class, it will give you back a dictionary with
    all objects stored indexed for each leaf. So if you index (0, 0, 0) in our running example,
    you will get back a dictionary with one 2D array and one class object from the ``numpy`` object
    array.

    All of these objects are stored in ``self.holder``.

    Args:
        obj_info (dict): Initial information for storage. Keys are the names to be stored under
            and values are arrays. These arrays should have a base shape that is equivalent to
            ``base_shape``, meaning ``array.shape[:len(base_shape)] == self.base_shape``.
            The dimensions beyond the base shape can be anything.
        base_shape (tuple): Base shape for indexing. Objects stored in the supplemental object
            will have a shape that at minimum is equivalent to ``base_shape``.
        copy (bool, optional): If ``True``, copy whatever information is given in before it is stored.
            if ``False``, store directly the input information. (default: ``False``)

    Attributes:
        holder (dict): All of the objects stored for this supplemental object.


    """

    def __init__(self, obj_info: dict, base_shape: tuple, copy: bool = False):
        # store initial information
        self.holder = {}
        self.base_shape = base_shape
        self.ndim = len(self.base_shape)

        # add initial set of objects
        self.add_objects(obj_info, copy=copy)

    def add_objects(self, obj_info: dict, copy=False):
        """Add objects to the holder.

        Args:
            obj_info (dict): Information for storage. Keys are the names to be stored under
                and values are arrays. These arrays should have a base shape that is equivalent to
                ``base_shape``, meaning ``array.shape[:len(base_shape)] == self.base_shape``.
                The dimensions beyond the base shape can be anything.
            copy (bool, optional): If ``True``, copy whatever information is given in before it is stored.
                if ``False``, store directly the input information. (default: ``False``)

        Raises:
            ValueError: Shape matching issues.

        """

        # whether a copy is requested
        dc = deepcopy if copy else (return_x)

        # iterate through the dictionary of incoming objects to add
        for name, obj_contained in obj_info.items():
            if (
                isinstance(obj_contained, np.ndarray)
                and obj_contained.dtype.name == "object"
            ):
                self.holder[name] = dc(obj_contained)
                if self.base_shape is None:
                    self.base_shape = self.holder[name].shape
                    self.ndim = ndim = len(self.base_shape)
                else:
                    if self.holder[name].shape != self.base_shape:
                        raise ValueError(
                            f"Outer shapes of all input objects must be the same. {name} object array has shape {self.holder[name].shape}. The original shape found was {self.base_shape}."
                        )

            else:
                self.ndim = ndim = len(self.base_shape)

                # xp for GPU
                if isinstance(obj_contained, np.ndarray) or isinstance(
                    obj_contained, xp.ndarray
                ):
                    self.holder[name] = obj_contained.copy()

                # fill object array from list
                # adjust based on how many dimensions found
                else:
                    # objects to be stored
                    self.holder[name] = np.empty(self.base_shape, dtype=object)
                    if len(obj_contained) != self.base_shape[0]:
                        raise ValueError(
                            "Shapes of obj_contained does not match base_shape along axis 0."
                        )

                    if ndim > 1:
                        for i in range(self.base_shape[0]):
                            if len(obj_contained[i]) != self.base_shape[1]:
                                raise ValueError(
                                    "Shapes of obj_contained does not match obj_contained_sha along axis 1."
                                )

                            if ndim > 2:
                                for j in range(self.base_shape[1]):
                                    if len(obj_contained[i][j]) != self.base_shape[2]:
                                        raise ValueError(
                                            "Shapes of obj_contained does not match base_shape along axis 2."
                                        )

                                    for k in range(self.base_shape[2]):
                                        self.holder[name][i, j, k] = obj_contained[i][
                                            j
                                        ][k]
                            else:
                                for j in range(self.base_shape[1]):
                                    self.holder[name][i, j] = obj_contained[i][j]

                    else:
                        for i in range(self.base_shape[0]):
                            self.holder[name][i] = obj_contained[i]

    def remove_objects(self, names):
        """Remove objects from the holder.


        Args:
            names (str or list of str): Strings associated with information to delete.
                Please note it does not return the information.

        Raises:
            ValueError: Input issues.


        """
        # check inputs
        if not isinstance(names, list):
            if not isinstance(names, str):
                raise ValueError("names must be a string or list of strings.")

            names = [names]

        # iterate and remove items from holder
        for name in names:
            self.holder.pop(name)

    @property
    def contained_objects(self):
        """The list of keys of contained objects."""
        return list(self.holder.keys())

    def __contains__(self, name: str):
        """Check if the holder holds a specific key."""
        return name in self.holder

    def __getitem__(self, tmp):
        """Special indexing for retrieval.

        When indexing the overall class, this will return the slice of each object

        Args:
            tmp (int, np.ndarray, or slice): indexing slice of some form.

        Returns:
            dict: Keys are names of the objects contained. Values are the slices of those objects.

        """
        # slice each object contained
        return {name: values[tmp] for name, values in self.holder.items()}

    def __setitem__(self, tmp, new_value):
        """Special indexing for setting elements.

        When indexing the overall class, this will set object information.

        **Please note**: If you try to input information that is not already stored,
        it will ignore it.

        Args:
            tmp (int, np.ndarray, or slice): indexing slice of some form.

        """
        # loop through values already in holder
        for name, values in self.holder.items():
            if name not in new_value:
                continue
            # if the name is already contained, update with incoming value
            self.holder[name][tmp] = new_value[name]

    def take_along_axis(self, indices, axis: int, skip_names=[]):
        """Take information from contained arrays along an axis.

        See ```numpy.take_along_axis`` <https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html>`_.

        Args:
            indices (xp.ndarray): Indices to take along each 1d slice of arr. This must match the dimension
                of ``self.base_shape``, but other dimensions only need to broadcast against
                ``self.base_shape``.
            axis (int): The axis to take 1d slices along.
            skip_names (list of str, optional): By default, this function returns the results for
                all stored objects. This list gives the strings of objects to leave behind and
                not return.

        Returns:
            dict: Keys are names of stored objects and values are the proper array slices.


        """
        # prepare output dictionary
        out = {}

        # iterate through holder
        for name, values in self.holder.items():
            # skip names if desired
            if name in skip_names:
                continue

            indices_temp = indices.copy()
            # adjust indices properly for specific object within the holder
            if (
                isinstance(values, np.ndarray) and values.dtype.name != "object"
            ) or isinstance(values, xp.ndarray):
                # expand the dimensions of the indexing values for non-object arrays
                for _ in range(values.ndim - indices_temp.ndim):
                    if isinstance(values, np.ndarray):
                        indices_temp = np.expand_dims(np.asarray(indices_temp), (-1,))
                    elif isinstance(values, xp.ndarray):
                        indices_temp = xp.expand_dims(xp.asarray(indices_temp), (-1,))

            # store the output for either numpy or cupy
            if isinstance(values, np.ndarray):
                out[name] = np.take_along_axis(values, indices_temp, axis)

            elif isinstance(values, xp.ndarray):
                out[name] = xp.take_along_axis(values, indices_temp, axis)

        return out

    def put_along_axis(self, indices, values_in: dict, axis: int):
        """Put information information into contained arrays along an axis.

        See ```numpy.put_along_axis`` <https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html>`_.

        **Please note** this function is not implemented in ``cupy``, so this is a custom implementation
        for both ``cupy`` and ``numpy``.

        Args:
            indices (xp.ndarray): Indices to put values along each 1d slice of arr. This must match
            the dimension of ``self.base_shape``, but other dimensions only need to broadcast against
            ``self.base_shape``.
            axis (int): The axis to put 1d slices along.
            values_in (dict): Keys are the objects contained to update. Values are the arrays of these
                objects with shape and dimension that can broadcast to match that of indices.

        """
        # iterate through all objects in the holder
        for name, values in self.holder.items():
            # skip names that are not to be updated
            if name not in values_in:
                continue

            # will need to have flexibility to broadcast
            indices_temp = indices.copy()

            if (
                isinstance(values, np.ndarray) and values.dtype.name != "object"
            ) or isinstance(values, xp.ndarray):
                # prepare indices for proper broadcasting
                for _ in range(values.ndim - indices_temp.ndim):
                    if isinstance(values, np.ndarray):
                        indices_temp = np.expand_dims(np.asarray(indices_temp), (-1,))
                    elif isinstance(values, xp.ndarray):
                        indices_temp = xp.expand_dims(xp.asarray(indices_temp), (-1,))

            # prepare slicing information for entry
            if isinstance(values, np.ndarray):
                inds0 = np.repeat(
                    np.arange(len(indices_temp))[:, None], indices_temp.shape[1], axis=1
                )
            elif isinstance(values, xp.ndarray):
                inds0 = xp.repeat(
                    np.arange(len(indices_temp))[:, None], indices_temp.shape[1], axis=1
                )
            # self.xp.put_along_axis(self.holder[name], indices_temp, values_in[name], axis)
            # because cupy does not have put_along_axis
            self.holder[name][(inds0.flatten(), indices_temp.flatten())] = values_in[
                name
            ].reshape((-1,) + values_in[name].shape[2:])

    @property
    def flat(self):
        """Get flattened arrays from the stored objects.

        Here "flat" is in relation to ``self.base_shape``. Beyond ``self.base_shape``, the shape is mainted.

        """
        out = {}
        # loop through holder
        for name, values in self.holder.items():
            if (
                isinstance(values, np.ndarray) and values.dtype.name != "object"
            ) or isinstance(values, xp.ndarray):
                # need to account for higher dimensional arrays.
                out[name] = values.reshape((-1,) + values.shape[2:])
            else:
                out[name] = values.flatten()
        return out


class Branch(object):
    """Special container for one branch (model)

    This class is a key component of Eryn. It this type of object
    that allows for different models to be considered simultaneously
    within an MCMC run.

    When running many independent samplers at once (``nsamplers > 1``), the
    sampler axis is FOLDED into the temperature axis: arrays keep their 4D/3D
    shapes with leading axis ``nsamplers * ntemps_per_sampler``. ``self.ntemps``
    is always the folded axis-0 size. Use the ``*_grouped`` properties to view
    the arrays with the sampler axis unfolded.

    Args:
        coords (4D double np.ndarray[ntemps, nwalkers, nleaves_max, ndim]): The coordinates
            in parameter space of all walkers.
        inds (3D bool np.ndarray[ntemps, nwalkers, nleaves_max], optional): The information
            on which leaves are used and which are not used. A value of True means the specific leaf
            was used in this step. Parameters from unused walkers are still kept. When they
            are output to the backend, the backend saves a special number (default: ``np.nan``) for all coords
            related to unused leaves at that step. If None, inds will fill with all True values.
            (default: ``None``)
        branch_supplemental (object): :class:`BranchSupplemental` object specific to this branch. (default: ``None``)
        nsamplers (int, optional): Number of independent samplers folded into
            the leading axis. (default: ``1``)

    Raises:
        ValueError: ``inds`` has wrong shape or number of leaves is less than zero.

    """

    # class-level default so unpickled/legacy Branch objects still work
    nsamplers = 1

    def __init__(self, coords, inds=None, branch_supplemental=None, nsamplers=1):
        # store branch info
        self.coords = coords
        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords.shape
        self.shape = coords.shape

        self.nsamplers = int(nsamplers)
        if self.ntemps % self.nsamplers != 0:
            raise ValueError(
                f"Leading (folded) axis size {self.ntemps} is not divisible by nsamplers={self.nsamplers}."
            )

        # make sure inds is correct
        if inds is None:
            self.inds = np.full((self.ntemps, self.nwalkers, self.nleaves_max), True)
        elif not isinstance(inds, np.ndarray):
            raise ValueError("inds must be np.ndarray in Branch.")
        elif inds.shape != (self.ntemps, self.nwalkers, self.nleaves_max):
            raise ValueError("inds has wrong shape.")
        else:
            self.inds = inds

        if branch_supplemental is not None:
            # make sure branch_supplemental shape matches
            if branch_supplemental.base_shape != self.inds.shape:
                raise ValueError(
                    f"branch_supplemental shape ( {branch_supplemental.base_shape} ) does not match inds shape ( {self.inds.shape} )."
                )

        # store
        self.branch_supplemental = branch_supplemental

    @property
    def nleaves(self):
        """Number of leaves for each walker"""
        # get number of leaves in each walker by summing inds along last axis
        nleaves = np.sum(self.inds, axis=-1)
        return nleaves

    @property
    def ntemps_per_sampler(self):
        """Number of temperatures per independent sampler (folded axis-0 = nsamplers * ntemps_per_sampler)."""
        return self.ntemps // self.nsamplers

    @property
    def coords_grouped(self):
        """View of ``coords`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler, nwalkers, nleaves_max, ndim)``."""
        return self.coords.reshape(
            (self.nsamplers, self.ntemps_per_sampler) + self.coords.shape[1:]
        )

    @property
    def inds_grouped(self):
        """View of ``inds`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler, nwalkers, nleaves_max)``."""
        return self.inds.reshape(
            (self.nsamplers, self.ntemps_per_sampler) + self.inds.shape[1:]
        )


class State(object):
    """The state of the ensemble during an MCMC run

    Args:
        coords (double ndarray[ntemps, nwalkers, nleaves_max, ndim], dict, or :class:`.State`): The current positions of the walkers
            in the parameter space. If dict, need to use ``branch_names`` for the keys.
        inds (bool ndarray[ntemps, nwalkers, nleaves_max] or dict, optional): The information
            on which leaves are used and which are not used. A value of True means the specific leaf
            was used in this step. If dict, need to use ``branch_names`` for the keys.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        branch_supplemental (object): :class:`BranchSupplemental` object specific to this branch.
            (default: ``None``)
        log_like (ndarray[ntemps, nwalkers], optional): Log likelihoods
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
        nsamplers (int, optional): Number of independent samplers. Internally, the sampler
            axis is FOLDED into the temperature axis (leading axis size
            ``nsamplers * ntemps_per_sampler``); 5D coordinate input
            ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)`` is folded automatically
            (similarly 4D ``inds``, 3D ``log_like``/``log_prior``, 2D ``betas``, 4D ``blobs``).
            Use the ``*_grouped`` properties to view arrays with the sampler axis unfolded.
            If ``None``, inferred from 5D input or set to 1. (default: ``None``)
        samplers_running (bool np.ndarray[nsamplers], optional): Mask of which independent
            samplers are actively running. Inactive samplers are skipped in
            likelihood/prior evaluation and tempering by the sampler. If ``None``, all
            samplers run. (default: ``None``)

    Raises:
        ValueError: Dimensions of inputs or input types are incorrect.

    """

    # __slots__ = (
    #    "branches",
    #    "log_like",
    #    "log_prior",
    #    "blobs",
    #    "betas",
    #    "supplemental",
    #    "random_state",
    # )

    def __init__(
        self,
        coords,
        inds=None,
        branch_supplemental=None,
        supplemental=None,
        log_like=None,
        log_prior=None,
        betas=None,
        blobs=None,
        random_state=None,
        copy=False,
        nsamplers=None,
        samplers_running=None,
    ):
        # decide if copying input info
        dc = deepcopy if copy else return_x

        # check if coords is a State object
        if hasattr(coords, "branches"):
            self.branches = dc(coords.branches)
            self.log_like = dc(coords.log_like)
            self.log_prior = dc(coords.log_prior)
            self.blobs = dc(coords.blobs)
            self.betas = dc(coords.betas)
            self.supplemental = dc(coords.supplemental)
            self.random_state = dc(coords.random_state)

            # carry the independent-samplers information; getattr protects
            # against legacy/unpickled State objects
            nsamplers_in = getattr(coords, "nsamplers", 1)
            if nsamplers is not None and int(nsamplers) != nsamplers_in:
                raise ValueError(
                    f"nsamplers kwarg ({nsamplers}) conflicts with input state's nsamplers ({nsamplers_in})."
                )
            self.nsamplers = nsamplers_in
            if samplers_running is None:
                samplers_running = getattr(coords, "samplers_running", None)
            self.samplers_running = (
                dc(np.atleast_1d(samplers_running))
                if samplers_running is not None
                else None
            )
            return

        # protect against simplifying settings
        if isinstance(coords, np.ndarray) or isinstance(coords, xp.ndarray):
            coords = {"model_0": coords}
        elif not isinstance(coords, dict):
            raise ValueError(
                "Input coords need to be np.ndarray, dict, or State object."
            )

        # 5D coordinates carry an explicit leading samplers axis; it is FOLDED
        # into the temperature axis for internal storage
        nsamplers_inferred = None
        for name in coords:
            if coords[name].ndim == 2:
                coords[name] = coords[name][None, :, None, :]

            # assume (ntemps, nwalkers) provided
            if coords[name].ndim == 3:
                coords[name] = coords[name][:, :, None, :]

            elif coords[name].ndim == 5:
                nsamplers_here = coords[name].shape[0]
                if nsamplers_inferred is None:
                    nsamplers_inferred = nsamplers_here
                elif nsamplers_inferred != nsamplers_here:
                    raise ValueError(
                        "All 5D branch coordinates must share the same leading (nsamplers) axis size. "
                        f"Found {nsamplers_here} for branch {name} vs {nsamplers_inferred}."
                    )
                coords[name] = coords[name].reshape((-1,) + coords[name].shape[2:])

            elif coords[name].ndim < 2 or coords[name].ndim > 5:
                raise ValueError(
                    f"Dimension of coordinates must be between 2 and 5. coords dimension is {coords[name].ndim}."
                )

        if nsamplers is not None and nsamplers_inferred is not None:
            if int(nsamplers) != nsamplers_inferred:
                raise ValueError(
                    f"nsamplers kwarg ({nsamplers}) conflicts with leading axis of 5D coords ({nsamplers_inferred})."
                )
        self.nsamplers = int(
            nsamplers
            if nsamplers is not None
            else (nsamplers_inferred if nsamplers_inferred is not None else 1)
        )

        # if no inds given, make sure this is clear for all Branch objects
        if inds is None:
            inds = {key: None for key in coords}
        elif not isinstance(inds, dict):
            raise ValueError("inds must be None or dict.")

        # fold a leading samplers axis out of the remaining inputs if present
        if self.nsamplers > 1:
            inds = {
                key: (
                    value.reshape((-1,) + value.shape[2:])
                    if value is not None
                    and value.ndim == 4
                    and value.shape[0] == self.nsamplers
                    else value
                )
                for key, value in inds.items()
            }
            if log_like is not None and np.asarray(log_like).ndim == 3:
                log_like = np.asarray(log_like).reshape(-1, log_like.shape[-1])
            if log_prior is not None and np.asarray(log_prior).ndim == 3:
                log_prior = np.asarray(log_prior).reshape(-1, log_prior.shape[-1])
            if betas is not None and np.asarray(betas).ndim == 2:
                betas = np.asarray(betas).reshape(-1)
            if (
                blobs is not None
                and np.asarray(blobs).ndim >= 4
                and np.asarray(blobs).shape[0] == self.nsamplers
            ):
                blobs = np.asarray(blobs).reshape((-1,) + np.asarray(blobs).shape[2:])

        if branch_supplemental is None:
            branch_supplemental = {key: None for key in coords}
        elif isinstance(branch_supplemental, dict): # case where not all branches have supp
            for key in coords.keys() - branch_supplemental.keys():
                branch_supplemental[key] = None
        elif not isinstance(branch_supplemental, dict):
            raise ValueError("branch_supplemental must be None or dict.")

        # setup all information for storage
        self.branches = {
            key: Branch(
                dc(temp_coords),
                inds=inds[key],
                branch_supplemental=branch_supplemental[key],
                nsamplers=self.nsamplers,
            )
            for key, temp_coords in coords.items()
        }
        self.log_like = dc(np.atleast_2d(log_like)) if log_like is not None else None
        self.log_prior = dc(np.atleast_2d(log_prior)) if log_prior is not None else None
        self.blobs = dc(np.atleast_3d(blobs)) if blobs is not None else None
        self.betas = dc(np.atleast_1d(betas)) if betas is not None else None
        self.supplemental = dc(supplemental)
        self.random_state = dc(random_state)

        if samplers_running is not None:
            samplers_running = np.atleast_1d(samplers_running)
            if samplers_running.shape != (self.nsamplers,):
                raise ValueError(
                    f"samplers_running must have shape ({self.nsamplers},); got {samplers_running.shape}."
                )
            self.samplers_running = dc(samplers_running.astype(bool))
        else:
            self.samplers_running = None

    @property
    def branches_inds(self):
        """Get the ``inds`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.inds for name, branch in self.branches.items()}

    @property
    def branches_coords(self):
        """Get the ``coords`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.coords for name, branch in self.branches.items()}

    @property
    def branches_supplemental(self):
        """Get the ``branch.supplemental`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {
            name: branch.branch_supplemental for name, branch in self.branches.items()
        }

    @property
    def branch_names(self):
        """Get the branch names in this state."""
        return list(self.branches.keys())

    @property
    def ntemps_per_sampler(self):
        """Number of temperatures per independent sampler (folded axis-0 = nsamplers * ntemps_per_sampler)."""
        first_branch = self.branches[list(self.branches.keys())[0]]
        return first_branch.ntemps // self.nsamplers

    @property
    def log_like_grouped(self):
        """View of ``log_like`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler, nwalkers)``."""
        if self.log_like is None:
            return None
        return self.log_like.reshape((self.nsamplers, -1) + self.log_like.shape[1:])

    @property
    def log_prior_grouped(self):
        """View of ``log_prior`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler, nwalkers)``."""
        if self.log_prior is None:
            return None
        return self.log_prior.reshape((self.nsamplers, -1) + self.log_prior.shape[1:])

    @property
    def betas_grouped(self):
        """View of ``betas`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler)``."""
        if self.betas is None:
            return None
        return self.betas.reshape(self.nsamplers, -1)

    @property
    def blobs_grouped(self):
        """View of ``blobs`` with the sampler axis unfolded: shape ``(nsamplers, ntemps_per_sampler, nwalkers, ...)``."""
        if self.blobs is None:
            return None
        return self.blobs.reshape(
            (self.nsamplers, self.blobs.shape[0] // self.nsamplers)
            + self.blobs.shape[1:]
        )

    def copy_into_self(self, state_to_copy):
        for name in state_to_copy.__slots__:
            setattr(self, name, getattr(state_to_copy, name))

    def get_log_posterior(self, temper: bool = False):
        """Get the posterior probability

        Args:
            temper (bool, optional): If ``True``, apply tempering to the posterior computation.

        Returns:
            np.ndarray[ntemps, nwalkers]: Log of the posterior probability.

        """

        if temper:
            betas = self.betas

        else:
            betas = np.ones_like(self.betas)

        return betas[:, None] * self.log_like + self.log_prior

    """
    # TODO
    def __repr__(self):
        return "State({0}, log_like={1}, blobs={2}, betas={3}, random_state={4})".format(
            self.coords, self.log_like, self.blobs, self.betas, self.random_state
        )

    def __iter__(self):
        temp = (self.coords,)
        if self.log_like is not None:
            temp += (self.log_like,)

        if self.blobs is not None:
            temp += (self.blobs,)

        if self.betas is None:
            temp += (self.betas,)

        if self.random_state is not None:
            temp += (self.random_state,)
        return iter(temp)
    """



class ParaState(object):
    """The state of the ensemble during an MCMC run

    Args:
        coords (double ndarray[ntemps, nwalkers, nleaves_max, ndim], dict, or :class:`.State`): The current positions of the walkers
            in the parameter space. If dict, need to use ``branch_names`` for the keys.
        groups_running (bool ndarray[ntemps, nwalkers, nleaves_max] or dict, optional): The information
            on which leaves are used and which are not used. A value of True means the specific leaf
            was used in this step. If dict, need to use ``branch_names`` for the keys.
            Input should be ``None`` if a complete :class:`.State` object is input for ``coords``.
            (default: ``None``)
        log_like (ndarray[ntemps, nwalkers], optional): Log likelihoods
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

    # __slots__ = (
    #    "branches",
    #    "log_like",
    #    "log_prior",
    #    "blobs",
    #    "betas",
    #    "supplemental",
    #    "random_state",
    # )

    def __init__(
        self,
        coords,
        groups_running=None,
        branch_supplemental=None,
        supplemental=None,
        log_like=None,
        log_prior=None,
        betas=None,
        blobs=None,
        random_state=None,
        copy=False,
    ):
        warnings.warn(
            "ParaState is deprecated; use eryn.state.State with the nsamplers/"
            "samplers_running arguments instead. It will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # decide if copying input info
        dc = deepcopy if copy else return_x

        # check if coords is a State object
        if hasattr(coords, "branches"):
            self.branches = dc(coords.branches)
            self.groups_running = dc(coords.groups_running)
            self.log_like = dc(coords.log_like)
            self.log_prior = dc(coords.log_prior)
            self.blobs = dc(coords.blobs)
            self.betas = dc(coords.betas)
            self.supplemental = dc(coords.supplemental)
            # self.random_state = dc(coords.random_state)
            # TODO: check this
            self.random_state = coords.random_state
            return

        # protect against simplifying settings
        if isinstance(coords, np.ndarray) or isinstance(coords, xp.ndarray):
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
                    f"Dimension off coordinates must be between 2 and 4. coords dimension is {coords.ndim}."
                )

        if branch_supplemental is None:
            branch_supplemental = {key: None for key in coords}
        elif not isinstance(branch_supplemental, dict):
            raise ValueError("branch_supplemental must be None or dict.")

        # setup all information for storage
        self.branches = {
            key: Branch(
                dc(temp_coords),
                inds=None,
                branch_supplemental=branch_supplemental[key],
            )
            for key, temp_coords in coords.items()
        }

        self.groups_running = (
            dc(np.atleast_1d(groups_running)) if groups_running is not None else None
        )
        self.log_like = dc(np.atleast_2d(log_like)) if log_like is not None else None
        self.log_prior = dc(np.atleast_2d(log_prior)) if log_prior is not None else None
        self.blobs = dc(np.atleast_3d(blobs)) if blobs is not None else None
        self.betas = dc(np.atleast_1d(betas)) if betas is not None else None
        self.supplemental = dc(supplemental)
        self.random_state = dc(random_state)

    @property
    def branches_coords(self):
        """Get the ``coords`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {name: branch.coords for name, branch in self.branches.items()}

    @property
    def branches_supplemental(self):
        """Get the ``branch.supplemental`` from all branch objects returned as a dictionary with ``branch_names`` as keys."""
        return {
            name: branch.branch_supplemental for name, branch in self.branches.items()
        }

    @property
    def branch_names(self):
        """Get the branch names in this state."""
        return list(self.branches.keys())

    def copy_into_self(self, state_to_copy):
        for name in state_to_copy.__slots__:
            setattr(self, name, getattr(state_to_copy, name))

    def get_log_posterior(self, temper: bool = False):
        """Get the posterior probability

        Args:
            temper (bool, optional): If ``True``, apply tempering to the posterior computation.

        Returns:
            np.ndarray[ntemps, nwalkers]: Log of the posterior probability.

        """

        if temper:
            betas = self.betas

        else:
            betas = np.ones_like(self.betas)

        return betas * self.log_like + self.log_prior

    """
    # TODO
    def __repr__(self):
        return "State({0}, log_like={1}, blobs={2}, betas={3}, random_state={4})".format(
            self.coords, self.log_like, self.blobs, self.betas, self.random_state
        )

    def __iter__(self):
        temp = (self.coords,)
        if self.log_like is not None:
            temp += (self.log_like,)

        if self.blobs is not None:
            temp += (self.blobs,)

        if self.betas is None:
            temp += (self.betas,)

        if self.random_state is not None:
            temp += (self.random_state,)
        return iter(temp)
    """
