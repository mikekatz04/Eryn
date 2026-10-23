# -*- coding: utf-8 -*-
"""Containers for the sampler state at one step.

- :class:`State`: everything the sampler carries for all walkers at one step (per-branch
  coordinates and active-leaf flags, log-likelihood, log-prior, inverse temperatures,
  blobs, supplemental information, random state, independent-sampler mask).
- :class:`Branch`: the coordinates and ``inds`` of one branch (model).
- :class:`BranchSupplemental`: arbitrary side information indexed like the walkers or
  leaves, so that it moves with them through proposals and temperature swaps.
- :class:`ParaState`: deprecated state for the deprecated
  :class:`~eryn.paraensemble.ParaEnsembleSampler`.

``cupy`` is imported as ``xp`` when available (``numpy`` otherwise) so that
:class:`BranchSupplemental` can hold GPU arrays.
"""

import warnings
from copy import deepcopy

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

import numpy as np

__all__ = ["State"]

def return_x(x):
    """Return ``x`` unchanged (the no-copy counterpart of ``deepcopy``)."""
    return x

class BranchSupplemental(object):
    """Side information carried through the sampler and indexed like the walkers.

    A holder of named arrays that share a leading ``base_shape``. Indexing the object
    indexes every stored array at once, so the information follows the walkers (or leaves)
    through proposals and temperature swaps. Use base shape ``(ntemps, nwalkers)`` for
    walker-level information (``State.supplemental``) and
    ``(ntemps, nwalkers, nleaves_max)`` for leaf-level information
    (``Branch.branch_supplemental``). Dimensions beyond the base shape are free, for example
    ``(ntemps, nwalkers, nleaves_max, dim2_extra, dim1_extra)`` holds a 2D array per leaf.
    Arbitrary Python objects can be stored in ``numpy`` object arrays of exactly
    ``base_shape`` (one object per walker or leaf). Indexing returns a dict with each stored
    array indexed the same way; for example ``supp[0, 0, 0]`` in the leaf-level case gives
    the 2D array and the object of that leaf.

    Args:
        obj_info (dict): Initial content: names to arrays (``numpy`` or ``cupy``), object
            arrays, or nested lists (see :meth:`add_objects`). The leading dimensions must
            equal ``base_shape``.
        base_shape (tuple): Leading shape shared by every stored array.
        copy (bool, optional): If ``True``, deep-copy object arrays before storing them.
            Numeric ``numpy`` / ``cupy`` arrays are always copied. Default ``False``.

    Attributes:
        holder (dict): Name to stored array.
        base_shape (tuple): Shared leading shape.
        ndim (int): ``len(base_shape)``.

    Example:
        >>> import numpy as np
        >>> from eryn.state import BranchSupplemental
        >>> supp = BranchSupplemental({"snr": np.zeros((2, 4, 3))}, base_shape=(2, 4, 3))
        >>> supp[0]["snr"].shape
        (4, 3)
        >>> supp.contained_objects
        ['snr']
    """

    def __init__(self, obj_info: dict, base_shape: tuple, copy: bool = False):
        # store initial information
        self.holder = {}
        self.base_shape = base_shape
        self.ndim = len(self.base_shape)

        # add initial set of objects
        self.add_objects(obj_info, copy=copy)

    def add_objects(self, obj_info: dict, copy=False):
        """Add (or replace) named arrays in the holder.

        Three kinds of values are accepted:

        - a ``numpy`` object array: stored as is (deep-copied when ``copy``); its shape
          must equal ``base_shape``;
        - a numeric ``numpy`` or ``cupy`` array: always stored as a copy; its leading shape
          is not checked;
        - anything else (for example nested lists): copied element by element into a new
          object array of shape ``base_shape`` (supported up to 3 base dimensions); the
          lengths along the base axes are checked.

        Args:
            obj_info (dict): Name to value, as above.
            copy (bool, optional): Deep-copy object arrays before storing. Default ``False``.

        Raises:
            ValueError: An object array whose shape differs from ``base_shape``, or a list
                whose lengths do not match ``base_shape``.
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
        """Remove named arrays from the holder (they are not returned).

        Args:
            names (str or list of str): Names to remove.

        Raises:
            ValueError: ``names`` is neither a str nor a list.
            KeyError: A name is not stored.
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
        """list of str: Names of the stored arrays."""
        return list(self.holder.keys())

    def __contains__(self, name: str):
        """Check if the holder holds a specific key."""
        return name in self.holder

    def __getitem__(self, tmp):
        """Index every stored array with the same index.

        Args:
            tmp (int, slice, tuple, or np.ndarray): Any numpy-style index over the base
                axes (for example a boolean mask of shape ``base_shape``).

        Returns:
            dict: Name to ``array[tmp]`` for every stored array.
        """
        # slice each object contained
        return {name: values[tmp] for name, values in self.holder.items()}

    def __setitem__(self, tmp, new_value):
        """Set ``array[tmp] = new_value[name]`` for every stored array named in ``new_value``.

        Names in ``new_value`` that are not already stored are ignored; stored arrays
        missing from ``new_value`` are left unchanged.

        Args:
            tmp (int, slice, tuple, or np.ndarray): Any numpy-style index over the base
                axes.
            new_value (dict): Name to the values to write.
        """
        # loop through values already in holder
        for name, values in self.holder.items():
            if name not in new_value:
                continue
            # if the name is already contained, update with incoming value
            self.holder[name][tmp] = new_value[name]

    def take_along_axis(self, indices, axis: int, skip_names=[]):
        """Apply ``take_along_axis`` to every stored numeric array.

        The ``numpy`` / ``cupy`` counterpart of :func:`numpy.take_along_axis`, applied to
        each stored array; ``indices`` gains trailing unit axes to match arrays with extra
        dimensions. Used by the moves to gather the walkers taking part in a proposal.

        Args:
            indices (xp.ndarray): Integer indices with ``len(base_shape)`` dimensions that
                broadcast against ``base_shape`` (``numpy`` or ``cupy``, matching the stored
                arrays).
            axis (int): Axis to take along.
            skip_names (list of str, optional): Names to leave out of the result. Default
                ``[]``.

        Returns:
            dict: Name to the gathered array. Object arrays are gathered with
            :func:`numpy.take_along_axis` without index expansion.
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
        """Write values back into the stored arrays along the second base axis.

        A custom counterpart of :func:`numpy.put_along_axis` (``cupy`` has none), applied
        in place to every stored array named in ``values_in``. As implemented it handles 2D
        ``indices`` of shape ``(n0, k)`` and writes ``values_in[name]`` at positions
        ``(i, indices[i, j])`` of the first two axes, so it is equivalent to
        ``put_along_axis(..., axis=1)``; the ``axis`` argument is not used.

        Args:
            indices (xp.ndarray): Integer indices of shape ``(n0, k)`` into axis 1.
            values_in (dict): Name to the values to write, shape ``(n0, k, ...)`` (extra
                trailing dimensions as in the stored array). Names not stored are ignored.
            axis (int): Not used (see above).
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
        """dict: Name to the stored array with its first two axes merged.

        Numeric arrays of shape ``(n0, n1, ...)`` become ``(n0 * n1, ...)``; object arrays
        are fully flattened. For the walker-level base shape ``(ntemps, nwalkers)`` this
        gives one entry per walker in flattened ``(ntemps, nwalkers)`` order, which is how
        :meth:`~eryn.ensemble.EnsembleSampler.compute_log_like` indexes ``supps``. For a
        numeric array with a 3-axis base shape only the first two axes are merged.
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
    """Coordinates and active-leaf flags of one branch (model).

    One :class:`Branch` per model lets several models be sampled together in one run. The
    coordinates of inactive leaves are kept in ``coords``; the backend stores
    ``store_missing_leaves`` (default ``np.nan``) for them instead.

    When running many independent samplers at once (``nsamplers > 1``), the
    sampler axis is FOLDED into the temperature axis: arrays keep their 4D/3D
    shapes with leading axis ``nsamplers * ntemps_per_sampler``. ``self.ntemps``
    is always the folded axis-0 size. Use the ``*_grouped`` properties to view
    the arrays with the sampler axis unfolded.

    Args:
        coords (np.ndarray): Coordinates of all walkers, shape
            ``(ntemps, nwalkers, nleaves_max, ndim)`` (``numpy``, or ``cupy`` for the
            deprecated :class:`ParaState`). Stored without copying.
        inds (np.ndarray, optional): Boolean active-leaf flags, shape
            ``(ntemps, nwalkers, nleaves_max)``; ``True`` means the leaf is in the model at
            this step. Must be a ``numpy`` array. ``None`` marks every leaf active.
            Default ``None``.
        branch_supplemental (BranchSupplemental, optional): Leaf-level side information for
            this branch; its ``base_shape`` must equal ``inds.shape``. Default ``None``.
        nsamplers (int, optional): Number of independent samplers folded into
            the leading axis. Default ``1``.

    Raises:
        ValueError: ``inds`` is not a ``numpy`` array or has the wrong shape, the
            supplemental base shape differs from ``inds.shape``, or the leading axis is not
            divisible by ``nsamplers``.

    Attributes:
        coords (np.ndarray): Coordinates, shape ``(ntemps, nwalkers, nleaves_max, ndim)``.
        inds (np.ndarray): Active-leaf flags, shape ``(ntemps, nwalkers, nleaves_max)``.
        branch_supplemental (BranchSupplemental or None): Leaf-level side information.
        shape (tuple): ``coords.shape``.
        ntemps (int): Leading (folded) axis size.
        nwalkers (int): Number of walkers.
        nleaves_max (int): Maximum number of leaves.
        ndim (int): Parameters per leaf.
        nsamplers (int): Number of folded independent samplers.

    Example:
        >>> import numpy as np
        >>> from eryn.state import Branch
        >>> inds = np.array([[[True, False]]])  # (ntemps, nwalkers, nleaves_max)
        >>> branch = Branch(np.zeros((1, 1, 2, 3)), inds=inds)
        >>> branch.nleaves
        array([[1]])
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
        """np.ndarray: Active leaves per walker, shape ``(ntemps, nwalkers)`` (int)."""
        # get number of leaves in each walker by summing inds along last axis
        nleaves = np.sum(self.inds, axis=-1)
        return nleaves

    @property
    def ntemps_per_sampler(self):
        """int: Temperatures per independent sampler (``ntemps // nsamplers``)."""
        return self.ntemps // self.nsamplers

    @property
    def coords_grouped(self):
        """np.ndarray: ``coords`` reshaped with the sampler axis unfolded.

        Shape ``(nsamplers, ntemps_per_sampler, nwalkers, nleaves_max, ndim)``; a view of
        ``coords`` when the reshape allows one.
        """
        return self.coords.reshape(
            (self.nsamplers, self.ntemps_per_sampler) + self.coords.shape[1:]
        )

    @property
    def inds_grouped(self):
        """np.ndarray: ``inds`` reshaped with the sampler axis unfolded.

        Shape ``(nsamplers, ntemps_per_sampler, nwalkers, nleaves_max)``.
        """
        return self.inds.reshape(
            (self.nsamplers, self.ntemps_per_sampler) + self.inds.shape[1:]
        )


class State(object):
    """The state of the ensemble at one step of an MCMC run.

    Holds one :class:`Branch` per model plus the walker-level quantities. ``ntemps`` below
    is the folded leading-axis size (``nsamplers * ntemps_per_sampler``).

    Coordinates are normalized per branch on input: a single array becomes the branch
    ``"model_0"``; 2D ``(nwalkers, ndim)`` becomes ``(1, nwalkers, 1, ndim)``; 3D
    ``(ntemps, nwalkers, ndim)`` becomes ``(ntemps, nwalkers, 1, ndim)``; 4D
    ``(ntemps, nwalkers, nleaves_max, ndim)`` is kept; 5D
    ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)`` is folded. ``log_like`` and
    ``log_prior`` pass through ``np.atleast_2d``, ``betas`` through ``np.atleast_1d`` and
    ``blobs`` through ``np.atleast_3d``.

    Args:
        coords (np.ndarray, dict, or State): Walker positions: one array (single branch) or
            a dict keyed by branch name (shapes as above). An object with a ``branches``
            attribute (a :class:`State`) is copied instead: its branches, ``log_like``,
            ``log_prior``, ``blobs``, ``betas``, ``supplemental``, ``random_state``,
            ``nsamplers`` and ``samplers_running`` are taken over, and the other
            arguments are ignored except ``nsamplers`` (checked) and ``samplers_running``
            (overrides).
        inds (dict, optional): Branch name to boolean active-leaf arrays of shape
            ``(ntemps, nwalkers, nleaves_max)`` (or 4D with a leading ``nsamplers`` axis);
            a dict needs an entry for every branch. ``None`` (or a ``None`` entry) marks
            every leaf active. Default ``None``.
        branch_supplemental (dict, optional): Branch name to a leaf-level
            :class:`BranchSupplemental` (base shape ``(ntemps, nwalkers, nleaves_max)``);
            branches missing from the dict get ``None``. Default ``None``.
        supplemental (BranchSupplemental, optional): Walker-level side information, base
            shape ``(ntemps, nwalkers)``; passed to the likelihood as ``supps`` when the
            sampler's ``provide_supplemental`` is set. Default ``None``.
        log_like (np.ndarray, optional): Log-likelihood, shape ``(ntemps, nwalkers)``.
            Default ``None`` (computed by the sampler).
        log_prior (np.ndarray, optional): Log-prior, shape ``(ntemps, nwalkers)``. Default
            ``None`` (computed by the sampler).
        betas (np.ndarray, optional): Inverse temperatures, shape ``(ntemps,)``. Default
            ``None``.
        blobs (np.ndarray, optional): Extra per-walker values returned by the likelihood,
            shape ``(ntemps, nwalkers, nblobs)``. Default ``None``.
        random_state (object, optional): Random-number-generator state, as returned by
            :attr:`EnsembleSampler.random_state <eryn.ensemble.EnsembleSampler.random_state>`.
            Default ``None``.
        copy (bool, optional): If ``True``, deep-copy the inputs (or the attributes of an
            input :class:`State`); otherwise store references. Default ``False``.
        nsamplers (int, optional): Number of independent samplers. Internally, the sampler
            axis is FOLDED into the temperature axis (leading axis size
            ``nsamplers * ntemps_per_sampler``); 5D coordinate input
            ``(nsamplers, ntemps, nwalkers, nleaves_max, ndim)`` is folded automatically
            (similarly 4D ``inds``, 3D ``log_like``/``log_prior``, 2D ``betas``, 4D ``blobs``).
            Use the ``*_grouped`` properties to view arrays with the sampler axis unfolded.
            If ``None``, inferred from 5D input or set to 1. Default ``None``.
        samplers_running (np.ndarray, optional): Boolean mask of shape ``(nsamplers,)`` of
            the independent samplers that are running. Inactive samplers are skipped in
            likelihood/prior evaluation and tempering by the sampler. If ``None``, all
            samplers run. Default ``None``.

    Raises:
        ValueError: ``coords`` is not an array, dict or State; a coordinate array is not
            2D-5D; 5D branches disagree on ``nsamplers``; ``nsamplers`` conflicts with the
            input; ``inds`` or ``branch_supplemental`` is not a dict; or
            ``samplers_running`` has the wrong shape (plus the :class:`Branch` checks).

    Attributes:
        branches (dict): Branch name to :class:`Branch`.
        log_like (np.ndarray or None): Log-likelihood, shape ``(ntemps, nwalkers)``.
        log_prior (np.ndarray or None): Log-prior, shape ``(ntemps, nwalkers)``.
        betas (np.ndarray or None): Inverse temperatures, shape ``(ntemps,)``.
        blobs (np.ndarray or None): Blobs, shape ``(ntemps, nwalkers, nblobs)``.
        supplemental (BranchSupplemental or None): Walker-level side information.
        random_state (object or None): Random-number-generator state.
        nsamplers (int): Number of folded independent samplers.
        samplers_running (np.ndarray or None): Running-sampler mask, shape
            ``(nsamplers,)``.

    Note:
        When ``coords`` is a dict, its entries are replaced in place by the reshaped
        arrays (2D, 3D and 5D inputs).

    Example:
        >>> import numpy as np
        >>> from eryn.state import State
        >>> coords = np.zeros((2, 4, 1, 3))  # (ntemps, nwalkers, nleaves_max, ndim)
        >>> state = State({"model_0": coords}, log_like=np.zeros((2, 4)))
        >>> state.branches["model_0"].shape
        (2, 4, 1, 3)
        >>> state.branches_inds["model_0"].shape
        (2, 4, 1)
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
        """dict: Branch name to ``inds`` (shape ``(ntemps, nwalkers, nleaves_max)``)."""
        return {name: branch.inds for name, branch in self.branches.items()}

    @property
    def branches_coords(self):
        """dict: Branch name to ``coords`` (shape ``(ntemps, nwalkers, nleaves_max, ndim)``)."""
        return {name: branch.coords for name, branch in self.branches.items()}

    @property
    def branches_supplemental(self):
        """dict: Branch name to its ``branch_supplemental`` (or ``None``)."""
        return {
            name: branch.branch_supplemental for name, branch in self.branches.items()
        }

    @property
    def branch_names(self):
        """list of str: Branch names, in insertion order."""
        return list(self.branches.keys())

    @property
    def ntemps_per_sampler(self):
        """int: Temperatures per independent sampler (folded leading axis // ``nsamplers``)."""
        first_branch = self.branches[list(self.branches.keys())[0]]
        return first_branch.ntemps // self.nsamplers

    @property
    def log_like_grouped(self):
        """np.ndarray or None: ``log_like`` reshaped with the sampler axis unfolded.

        Shape ``(nsamplers, ntemps_per_sampler, nwalkers)``; ``None`` if ``log_like`` is.
        """
        if self.log_like is None:
            return None
        return self.log_like.reshape((self.nsamplers, -1) + self.log_like.shape[1:])

    @property
    def log_prior_grouped(self):
        """np.ndarray or None: ``log_prior`` reshaped with the sampler axis unfolded.

        Shape ``(nsamplers, ntemps_per_sampler, nwalkers)``; ``None`` if ``log_prior`` is.
        """
        if self.log_prior is None:
            return None
        return self.log_prior.reshape((self.nsamplers, -1) + self.log_prior.shape[1:])

    @property
    def betas_grouped(self):
        """np.ndarray or None: ``betas`` reshaped to ``(nsamplers, ntemps_per_sampler)``."""
        if self.betas is None:
            return None
        return self.betas.reshape(self.nsamplers, -1)

    @property
    def blobs_grouped(self):
        """np.ndarray or None: ``blobs`` reshaped with the sampler axis unfolded.

        Shape ``(nsamplers, ntemps_per_sampler, nwalkers, ...)``; ``None`` if ``blobs`` is.
        """
        if self.blobs is None:
            return None
        return self.blobs.reshape(
            (self.nsamplers, self.blobs.shape[0] // self.nsamplers)
            + self.blobs.shape[1:]
        )

    # TODO check: keep/remove/document -- broken: reads __slots__ (commented out); no callers
    def copy_into_self(self, state_to_copy):
        """Copy the ``__slots__`` attributes of ``state_to_copy`` onto this state."""
        for name in state_to_copy.__slots__:
            setattr(self, name, getattr(state_to_copy, name))

    def get_log_posterior(self, temper: bool = False):
        """Compute the log-posterior ``beta * log_like + log_prior`` of every walker.

        Args:
            temper (bool, optional): If ``True``, weight the log-likelihood by ``betas``;
                otherwise use ``beta = 1``. Default ``False``.

        Returns:
            np.ndarray: Log-posterior, shape ``(ntemps, nwalkers)``.

        Note:
            ``betas`` must be set even when ``temper`` is ``False`` (it gives the shape).
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
    """State of the deprecated :class:`~eryn.paraensemble.ParaEnsembleSampler` (deprecated).

    Holds ``ngroups`` independent single-branch ensembles. Each branch is stored in a
    :class:`Branch` whose four axes are reinterpreted as ``(ngroups, ntemps, nwalkers,
    ndim)`` (so ``Branch.ntemps`` is ``ngroups``, ``Branch.nwalkers`` is ``ntemps``, and
    so on); ``inds`` is not used. Arrays may be ``numpy`` or ``cupy``.

    Args:
        coords (np.ndarray, cupy.ndarray, dict, or ParaState): Positions, shape
            ``(ngroups, ntemps, nwalkers, ndim)``, either one array (stored as branch
            ``"model_0"``) or a dict keyed by branch name. 2D and 3D inputs are expanded as
            in :class:`State` (``[None, :, None, :]`` and ``[:, :, None, :]``). An object
            with a ``branches`` attribute is copied instead (all its attributes are taken
            over and the other arguments are ignored).
        groups_running (np.ndarray, optional): Boolean mask of shape ``(ngroups,)`` of the
            groups being sampled. Default ``None`` (the sampler then runs every group).
        branch_supplemental (dict, optional): Branch name to a
            :class:`BranchSupplemental` with base shape ``(ngroups, ntemps, nwalkers)``.
            Default ``None``.
        supplemental (object, optional): Group-level side information (stored, not used by
            the sampler). Default ``None``.
        log_like (np.ndarray, optional): Log-likelihood, shape
            ``(ngroups, ntemps, nwalkers)``. Default ``None``.
        log_prior (np.ndarray, optional): Log-prior, shape ``(ngroups, ntemps, nwalkers)``.
            Default ``None``.
        betas (np.ndarray, optional): Inverse temperatures, shape ``(ngroups, ntemps)``.
            Default ``None``.
        blobs (np.ndarray, optional): Blobs (stored, not used by the sampler). Default
            ``None``.
        random_state (object, optional): Random generator (the sampler stores the
            ``numpy.random`` / ``cupy.random`` module here). Default ``None``.
        copy (bool, optional): If ``True``, deep-copy the inputs. When copying a ParaState,
            ``random_state`` is always shared, not copied. Default ``False``.

    Raises:
        ValueError: ``coords`` is not an array, dict or state, or ``branch_supplemental`` is
            not a dict.
        AttributeError: A coordinate array with fewer than 2 or more than 4 dimensions (the
            intended ``ValueError`` message reads ``coords.ndim`` on the dict).

    Warning:
        Deprecated: construction emits a ``DeprecationWarning``. Use :class:`State` with the
        ``nsamplers`` / ``samplers_running`` arguments instead.
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
        """dict: Branch name to ``coords`` (shape ``(ngroups, ntemps, nwalkers, ndim)``)."""
        return {name: branch.coords for name, branch in self.branches.items()}

    @property
    def branches_supplemental(self):
        """dict: Branch name to its ``branch_supplemental`` (or ``None``)."""
        return {
            name: branch.branch_supplemental for name, branch in self.branches.items()
        }

    @property
    def branch_names(self):
        """list of str: Branch names, in insertion order."""
        return list(self.branches.keys())

    # TODO check: keep/remove/document -- broken: reads __slots__ (commented out); no callers
    def copy_into_self(self, state_to_copy):
        """Copy the ``__slots__`` attributes of ``state_to_copy`` onto this state."""
        for name in state_to_copy.__slots__:
            setattr(self, name, getattr(state_to_copy, name))

    # TODO check: keep/remove/document -- (ngroups, ntemps) betas do not broadcast; no callers
    def get_log_posterior(self, temper: bool = False):
        """Compute ``betas * log_like + log_prior`` (``betas = 1`` unless ``temper``).

        As written, ``betas`` of shape ``(ngroups, ntemps)`` multiplies ``log_like`` of
        shape ``(ngroups, ntemps, nwalkers)`` without a trailing axis, which does not
        broadcast in general.
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
