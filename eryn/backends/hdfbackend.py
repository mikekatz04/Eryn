# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["HDFBackend", "TempHDFBackend", "does_hdf5_support_longdouble"]

import os
from tempfile import NamedTemporaryFile

import numpy as np

from .. import __version__
from .backend import Backend


try:
    import h5py
except ImportError:
    h5py = None


def does_hdf5_support_longdouble():
    if h5py is None:
        return False
    with NamedTemporaryFile(
        prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
    ) as f:
        f.close()

        with h5py.File(f.name, "w") as hf:
            g = hf.create_group("group")
            g.create_dataset("data", data=np.ones(1, dtype=np.longdouble))
            if g["data"].dtype != np.longdouble:
                return False
        with h5py.File(f.name, "r") as hf:
            if hf["group"]["data"].dtype != np.longdouble:
                return False
    return True


class HDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str, optional): The name of the group where the chain will
            be saved. (default: ``"mcmc"``)
        read_only (bool, optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.
            (default: ``False``)
        dtype (dtype, optional): Dtype to use for data storage. If None,
            program uses np.float64. (default: ``None``)
        compression (str, optional): Compression type for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        compression_opts (int, optional): Compression level for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        store_missing_leaves (double, optional): Number to store for leaves that are not
            used in a specific step. (default: ``np.nan``)


    """

    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
        store_missing_leaves=np.nan,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")

        # store all necessary quantities
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

    @property
    def initialized(self):
        """Check if backend file has been initialized properly."""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        """Opens the h5 file in the proper mode.

        Args:
            mode (str, optional): Mode to open h5 file.

        Returns:
            H5 file object: Opened file.

        Raises:
            RuntimeError: If backend is opened for writing when it is read-only.

        """

        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )

        # open the file
        f = h5py.File(self.filename, mode)

        # get the data type and store it if it is not previously set
        if not self.dtype_set and self.name in f:
            # get the group from the file
            g = f[self.name]
            if "chain" in g:
                # get the model names in chain
                keys = list(g["chain"])

                # they all have the same dtype so use the first one
                self.dtype = g["chain"][keys[0]].dtype

                # we now have it
                self.dtype_set = True

        return f

    def reset(
        self,
        nwalkers,
        ndims,
        nleaves_max=1,
        ntemps=1,
        branch_names=None,
        rj=False,
        moves=None,
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble
            ndims (int or list of ints): The number of dimensions for each branch.
            nleaves_max (int or list of ints, optional):
                Maximum allowable leaf count for each branch. It should have
                the same length as the number of branches.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            moves (list, optional): List of all of the move classes input into the sampler.
                (default: ``None``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend. These are also added to the HDF5 file.

        """

        # open file in append mode
        with self.open("a") as f:
            # we are resetting so if self.name in the file we need to delete it
            if self.name in f:
                del f[self.name]

            # set the reset args and kwargs for next time if needed
            self.reset_args = (nwalkers, ndims)
            self.reset_kwargs = dict(
                nleaves_max=nleaves_max,
                ntemps=ntemps,
                branch_names=branch_names,
                rj=rj,
                moves=moves,
                info=info,
            )

            # store important info
            self.nwalkers = int(nwalkers)  # trees
            self.ntemps = int(ntemps)
            self.rj = rj

            # turn things into lists if needed

            if isinstance(ndims, int):
                self.ndims = np.array([ndims])
            elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
                self.ndims = np.asarray(ndims)
            else:
                raise ValueError("ndims is to be a scalar int or a list.")

            if isinstance(nleaves_max, int):
                self.nleaves_max = np.array([nleaves_max])
            elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
                self.nleaves_max = np.asarray(nleaves_max)
            else:
                raise ValueError("nleaves_max is to be a scalar int or a list.")

            if len(self.nleaves_max) != len(self.ndims):
                raise ValueError(
                    "Number of branches indicated by nleaves_max and ndims are not equivalent (nleaves_max: {}, ndims: {}).".format(
                        len(self.nleaves_max), len(self.ndims)
                    )
                )

            self.nbranches = len(self.nleaves_max)
            if branch_names is not None:
                if isinstance(branch_names, str):
                    branch_names = [branch_names]

                elif not isinstance(branch_names, list):
                    raise ValueError("branch_names must be string or list of strings.")

                elif len(branch_names) != self.nbranches:
                    raise ValueError(
                        """Number of branches indicated by nleaves_max and branch_names are not equivalent 
                        (nleaves_max: {}, branch_names: {}).""".format(
                            len(self.nleaves_max), len(branch_names)
                        )
                    )

            else:
                branch_names = ["model_{}".format(i) for i in range(self.nbranches)]

            # store all the info needed in memory and in the file

            self.branch_names = branch_names

            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nbranches"] = self.nbranches
            g.attrs["branch_names"] = self.branch_names
            g.attrs["ntemps"] = ntemps
            g.attrs["nwalkers"] = nwalkers
            g.attrs["nleaves_max"] = self.nleaves_max
            g.attrs["has_blobs"] = False
            g.attrs["rj"] = rj
            g.attrs["iteration"] = 0

            # load info into class and into file
            for key, value in info.items():
                setattr(self, key, value)
                g.attrs[key] = value

            g.attrs["ndims"] = self.ndims

            # prepare all the data sets

            g.create_dataset(
                "accepted",
                data=np.zeros((ntemps, nwalkers)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "swaps_accepted",
                data=np.zeros((ntemps - 1,)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            if self.rj:
                g.create_dataset(
                    "rj_accepted",
                    data=np.zeros((ntemps, nwalkers)),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            g.create_dataset(
                "log_like",
                (0, ntemps, nwalkers),
                maxshape=(None, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "log_prior",
                (0, ntemps, nwalkers),
                maxshape=(None, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "betas",
                (0, ntemps),
                maxshape=(None, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # setup data sets for branch-specific items

            chain = g.create_group("chain")
            inds = g.create_group("inds")

            for name, nleaves, ndim in zip(
                self.branch_names, self.nleaves_max, self.ndims
            ):
                chain.create_dataset(
                    name,
                    (0, ntemps, nwalkers, nleaves, ndim),
                    maxshape=(None, ntemps, nwalkers, nleaves, ndim),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                inds.create_dataset(
                    name,
                    (0, ntemps, nwalkers, nleaves),
                    maxshape=(None, ntemps, nwalkers, nleaves),
                    dtype=bool,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            # store move specific information
            if moves is not None:
                move_group = g.create_group("moves")
                # setup info and keys
                self.move_keys = []
                for move in moves:
                    # get out of tuple if weights are given
                    if isinstance(move, tuple):
                        move = move[0]

                    # get the name of the class instance as a string
                    move_name = move.__class__.__name__
                    single_move = move_group.create_group(move_name)

                    # prepare information dictionary
                    single_move.create_dataset(
                        "acceptance_fraction",
                        (ntemps, nwalkers),
                        maxshape=(ntemps, nwalkers),
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )

                    # update the move keys to keep proper order
                    self.move_keys.append(move_name)

            else:
                self.move_info = None

            self.blobs = None

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): If provided, slice the array directly
                from the HDF5 file with slice = ``slice_vals``. ``thin`` and ``discard`` will be 
                ignored if slice_vals is not ``None``. This is particularly useful if files are 
                very large and the user only wants a small subset of the overall array.
                (default: ``None``)

        Returns:
            dict or np.ndarray: Values requested.

        """
        # check if initialized
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        # open the file wrapped in a "with" statement
        with self.open() as f:
            # get the group that everything is stored in
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            if name == "chain":
                v_all = {key: g["chain"][key][slice_vals] for key in g["chain"]}
                return v_all

            if name == "inds":
                v_all = {key: g["inds"][key][slice_vals] for key in g["inds"]}

                return v_all

            v = g[name][slice_vals]

            return v

    def get_move_info(self):
        """Get move information.
        
        Returns:
            dict: Keys are move names and values are dictionaries with information on the moves.
        
        """
        # setup output dictionary
        move_info_out = {}
        with self.open() as f:
            g = f[self.name]

            # iterate through everything and produce a dictionary
            for move_name in g["moves"]:
                move_info_out[move_name] = {}
                for info_name in g["moves"][move_name]:
                    move_info_out[move_name][info_name] = g["moves"][move_name][
                        info_name
                    ][:]

        return move_info_out

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            dict: Shape of samples
                Keys are ``branch_names`` and values are tuples with
                shapes of individual branches: (ntemps, nwalkers, nleaves_max, ndim).

        """
        # open file wrapped in with
        with self.open() as f:
            g = f[self.name]
            return {
                key: (g.attrs["ntemps"], g.attrs["nwalkers"], nleaves, ndim)
                for key, nleaves, ndim in zip(
                    g.attrs["branch_names"], g.attrs["nleaves_max"], g.attrs["ndims"]
                )
            }

    @property
    def iteration(self):
        """Number of iterations stored in the hdf backend so far."""
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        """Number of accepted moves per walker."""
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def rj_accepted(self):
        """Number of accepted rj moves per walker."""
        with self.open() as f:
            return f[self.name]["rj_accepted"][...]

    @property
    def swaps_accepted(self):
        """Number of accepted swaps."""
        with self.open() as f:
            return f[self.name]["swaps_accepted"][...]

    @property
    def random_state(self):
        """Get the random state"""
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]

            # resize all the arrays accordingly

            ntot = g.attrs["iteration"] + ngrow
            for key in g["chain"]:
                g["chain"][key].resize(ntot, axis=0)
                g["inds"][key].resize(ntot, axis=0)

            g["log_like"].resize(ntot, axis=0)
            g["log_prior"].resize(ntot, axis=0)
            g["betas"].resize(ntot, axis=0)

            # deal with blobs
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                # if blobs have not been added yet
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    ntemps = g.attrs["ntemps"]
                    g.create_dataset(
                        "blobs",
                        (ntot, ntemps, nwalkers, blobs.shape[-1]),
                        maxshape=(None, ntemps, nwalkers, blobs.shape[-1]),
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                else:
                    # resize the blobs if they have been there
                    g["blobs"].resize(ntot, axis=0)
                    if g["blobs"].shape[1:] != blobs.shape:
                        raise ValueError(
                            "Existing blobs have shape {} but new blobs "
                            "requested with shape {}".format(
                                g["blobs"].shape[1:], blobs.shape
                            )
                        )
                g.attrs["has_blobs"] = True

    def save_step(
        self,
        state,
        accepted,
        rj_accepted=None,
        swaps_accepted=None,
        moves_accepted_fraction=None,
    ):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            rj_accepted (ndarray, optional): An array of the number of accepted steps
                for the reversible jump proposal for each walker.
                If :code:`self.rj` is True, then rj_accepted must be an array with
                :code:`rj_accepted.shape == accepted.shape`. If :code:`self.rj`
                is False, then rj_accepted must be None, which is the default.
            swaps_accepted (ndarray, optional): 1D array with number of swaps accepted
                for the in-model step. (default: ``None``)
            moves_accepted_fraction (list, optional): List of acceptance fraction arrays for all of the 
                moves in the sampler. This list must have the same length as ``self.move_keys``
                and be in the same order.
                (default: ``None``)

        """
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            iteration = g.attrs["iteration"]

            # make sure the backend has all the information needed to store everything
            for key in [
                "rj",
                "ntemps",
                "nwalkers",
                "nbranches",
                "branch_names",
                "ndims",
            ]:
                if not hasattr(self, key):
                    setattr(self, key, g.attrs[key])

            # check the inputs are okay
            self._check(
                state, accepted, rj_accepted=rj_accepted, swaps_accepted=swaps_accepted,
            )

            # branch-specific
            for name, model in state.branches.items():
                g["inds"][name][iteration] = model.inds
                # use self.store_missing_leaves to set value for missing leaves
                # state retains old coordinates
                coords_in = model.coords * model.inds[:, :, :, None]
                inds_all = np.repeat(model.inds, coords_in.shape[-1], axis=-1).reshape(
                    model.inds.shape + (coords_in.shape[-1],)
                )
                coords_in[~inds_all] = self.store_missing_leaves
                g["chain"][name][self.iteration] = coords_in

            # store everything else in the file
            g["log_like"][iteration, :] = state.log_like
            g["log_prior"][iteration, :] = state.log_prior
            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            if state.betas is not None:
                g["betas"][self.iteration, :] = state.betas
            g["accepted"][:] += accepted
            if swaps_accepted is not None:
                g["swaps_accepted"][:] += swaps_accepted
            if self.rj:
                g["rj_accepted"][:] += rj_accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1

            # moves
            if moves_accepted_fraction is not None:
                if "moves" not in g:
                    raise ValueError(
                        """moves_accepted_fraction was passed, but moves_info was not initialized. Use the moves kwarg 
                        in the reset function."""
                    )

                # make sure they are the same length
                assert len(moves_accepted_fraction) == len(self.move_keys)

                # update acceptance fractions
                for move_key, move_accepted_fraction in zip(
                    self.move_keys, moves_accepted_fraction
                ):
                    g["moves"][move_key]["acceptance_fraction"][
                        :
                    ] = move_accepted_fraction[:]


class TempHDFBackend(object):
    """Check if HDF5 is working and available."""

    def __init__(self, dtype=None, compression=None, compression_opts=None):
        self.dtype = dtype
        self.filename = None
        self.compression = compression
        self.compression_opts = compression_opts

    def __enter__(self):
        f = NamedTemporaryFile(
            prefix="emcee-temporary-hdf5", suffix=".hdf5", delete=False
        )
        f.close()
        self.filename = f.name
        return HDFBackend(
            f.name,
            "test",
            dtype=self.dtype,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

    def __exit__(self, exception_type, exception_value, traceback):
        os.remove(self.filename)
