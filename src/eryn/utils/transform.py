"""Parameter-basis transforms: :class:`TransformContainer`.

Maps coordinates from the sampler's INPUT basis (the sampled parameters) to an OUTPUT basis
(what a likelihood or waveform needs) by filling fixed parameters and applying per-parameter
and multi-parameter transforms, and back again when inverse transforms are supplied.
"""

from __future__ import annotations

from typing import Any, Callable, List, Dict, Tuple, TypeVar

try:
    import cupy as xp # type: ignore[import]

except (ModuleNotFoundError, ImportError) as e:
    pass

import numpy as np

Element = TypeVar("Element", str, int)

class TransformContainer:
    """Map coordinates between the sampling (input) basis and an output basis.

    Forward direction (:meth:`both_transforms`, ``(..., ndim) -> (..., ndim_full)``):

    1. :meth:`fill_values` places each input column at its output-basis position and
       writes the fixed ``fill_dict`` values into their columns.
    2. :meth:`transform_base_parameters` applies, IN OUTPUT-BASIS COLUMNS, the
       single-parameter transforms first and then the multi-parameter transforms (in
       insertion order).

    Inverse direction (:meth:`both_inverse_transforms`): the inverse transforms
    (multi-parameter first, in reverse insertion order, then single-parameter), then
    :meth:`unfill_values` drops the fixed columns and restores the input order.

    Keys of ``parameter_transforms``, ``inverse_parameter_transforms`` and ``fill_dict`` are
    output-basis ELEMENTS (names, or ints for an integer basis; an ``int`` key is matched as
    an element of ``output_basis``, not used as a column index). A transform key absent from
    ``output_basis`` is looked up in ``key_map``; input-basis names and fill keys are mapped
    through ``key_map`` whenever they appear in it. A single-parameter transform is
    ``f(column) -> column``; a multi-parameter transform keyed by a tuple is
    ``f(*columns) -> sequence of columns``, written back in the key's order. Each column
    passed has shape ``params.shape[:-1][::-1]`` (a row of the transposed array;
    multi-parameter transforms receive copies).

    Args:
        input_basis: Names (or ints) of the sampled parameters, in column order. Each must
            be in ``output_basis`` or in ``key_map``.
        output_basis: Names (or ints) of the output parameters, in column order.
        parameter_transforms: Forward transforms, ``{key or tuple of keys: callable}``.
            Default ``None`` (no transforms).
        inverse_parameter_transforms: Inverses, keyed like ``parameter_transforms``; each
            value must invert the matching forward entry. Needed by
            :meth:`inverse_transform_base_parameters` / :meth:`both_inverse_transforms`.
            Default ``None``.
        fill_dict: Fixed (non-sampled) output-basis parameters. A ``dict`` maps key ->
            scalar, filled identically in every row. A ``list`` of such dicts gives
            PER-LEAF fills: entry ``i`` holds the values for leaf ``i`` (all entries must
            have the same keys in the same order; the length is not checked, so the list
            must cover every leaf index passed in ``leaf_inds``, typically ``nleaves_max``
            entries), and :meth:`fill_values` / :meth:`both_transforms` then need
            ``leaf_inds``. An empty ``dict`` fixes nothing but still enables the reordering
            into the output basis (see the Note). Default ``None``.
        key_map: ``{name: output-basis name}``, typically for input-basis names that are
            not in ``output_basis`` (e.g. a sampled ``"Mc"`` written into an ``"fdot"``
            column that a multi-parameter transform then converts). The input column is
            placed at the mapped output column. Default ``None`` (empty).

    Attributes:
        input_basis (list): Input basis.
        output_basis (list): Output basis.
        ndim (int): ``len(input_basis)``.
        ndim_full (int): ``len(output_basis)``.
        test_inds (np.ndarray): Output-basis column of each input column.
        base_transforms (dict or None): ``"single_param"`` (``{int: f}``) and
            ``"mult_param"`` (``{tuple: f}``) keyed by output-basis column, or ``None``.
        base_inverse_transforms (dict or None): Same layout for the inverses.
        fill_dict (dict or None): Processed fills: ``"fill_inds"`` (output columns),
            ``"fill_values"`` (shape ``(n_fill,)``, or ``(n_leaves, n_fill)`` per leaf) and
            ``"test_inds"``; ``None`` without fills.
        n_leaf_fills (int or None): Number of per-leaf fill entries, or ``None``.
        original_parameter_transforms (dict or None): Input ``parameter_transforms``.
        original_inverse_parameter_transforms (dict or None): Input inverse transforms.
        original_fill_dict (dict, list or None): Input ``fill_dict``.

    Raises:
        ValueError: If an input-basis key is in neither ``output_basis`` nor ``key_map``, a
            transform key has an unsupported type, or ``fill_dict`` is malformed (empty
            list, non-dict entries, or per-leaf entries with different keys).
        AssertionError: If a transform key is in neither ``output_basis`` nor ``key_map``.

    Note:
        The reordering of input columns into output-basis positions happens only in
        :meth:`fill_values`, and only when ``fill_dict`` is not ``None``. Without a
        ``fill_dict``, :meth:`both_transforms` applies the output-basis-indexed transforms
        to the input array as it is, which is correct only when the two bases have the
        same order and length. Pass ``fill_dict={}`` otherwise.

    Example:
        >>> import numpy as np
        >>> from eryn.utils import TransformContainer
        >>> tc = TransformContainer(
        ...     input_basis=["a", "b"],
        ...     output_basis=["a", "b", "c"],
        ...     parameter_transforms={"a": np.square},
        ...     inverse_parameter_transforms={"a": np.sqrt},
        ...     fill_dict={"c": 5.0},
        ... )
        >>> out = tc.both_transforms(np.array([[3.0, 2.0]]))
        >>> out.tolist()
        [[9.0, 2.0, 5.0]]
        >>> tc.both_inverse_transforms(out).tolist()
        [[3.0, 2.0]]
    """

    def __init__(
        self, 
        input_basis: List[Element], 
        output_basis: List[Element], 
        parameter_transforms: Dict[Element | Tuple[Element, ...], Callable] | None = None, 
        inverse_parameter_transforms: Dict[Element | Tuple[Element, ...], Callable] | None = None,
        fill_dict: Dict[Element, float] | None = None,
        key_map: Dict[Element, Element] | None = None,
    ):
        if key_map is None:
            key_map = {}

        if key_map is None:
            key_map = {}

        # store originals
        self.input_basis, self.output_basis = input_basis, output_basis
        self.original_parameter_transforms = parameter_transforms
        self.original_inverse_parameter_transforms = inverse_parameter_transforms
        self.ndim_full = len(output_basis)
        self.ndim = len(input_basis)

        test_inds = []
        for key in input_basis:
            if key not in output_basis and key not in key_map:
                raise ValueError("All keys in input_basis must be present in output basis, or you must provide a key_map")
            key_in = key if key not in key_map else key_map[key]
            test_inds.append(output_basis.index(key_in))

        self.test_inds = test_inds = np.asarray(test_inds)
        if parameter_transforms is not None:
            self.base_transforms = self._parse_parameter_transforms(
                parameter_transforms, output_basis, key_map
            )
        else:
            self.base_transforms = None

        if inverse_parameter_transforms is not None:
            self.base_inverse_transforms = self._parse_parameter_transforms(
                inverse_parameter_transforms, output_basis, key_map
            )
        else:
            self.base_inverse_transforms = None

        self.original_fill_dict = fill_dict
        if fill_dict is not None:
            if isinstance(fill_dict, list):
                # per-leaf fills: one dict per leaf, identical keys across leaves
                if len(fill_dict) == 0:
                    raise ValueError("A per-leaf fill_dict list must not be empty.")
                fill_keys = list(fill_dict[0].keys())
                for i, leaf_fill in enumerate(fill_dict):
                    if not isinstance(leaf_fill, dict):
                        raise ValueError("Each entry of a per-leaf fill_dict list must be a dictionary.")
                    if list(leaf_fill.keys()) != fill_keys:
                        raise ValueError(
                            "All per-leaf fill dicts must share identical keys (in the same order); "
                            f"entry {i} has {list(leaf_fill.keys())} vs {fill_keys}."
                        )
                self.n_leaf_fills = len(fill_dict)
                # stacked (nleaves, n_fill) values; rows selected by leaf_inds at call time
                fill_values = np.asarray(
                    [[leaf_fill[key] for key in fill_keys] for leaf_fill in fill_dict],
                    dtype=float,
                )
            elif isinstance(fill_dict, dict):
                self.n_leaf_fills = None
                fill_keys = list(fill_dict.keys())
                fill_values = np.asarray([fill_dict[key] for key in fill_keys], dtype=float)
            else:
                raise ValueError("fill_dict must be a dictionary or a list of dictionaries.")

            self.fill_dict = {}
            # dtype=int keeps an empty fill_dict ({}) usable as an index array
            # (fill keys resolve through key_map like parameter_transforms keys)
            self.fill_dict["fill_inds"] = np.asarray(
                [
                    output_basis.index(key if key not in key_map else key_map[key])
                    for key in fill_keys
                ],
                dtype=int,
            )
            self.fill_dict["fill_values"] = fill_values
            # set up test_inds accordingly
            self.fill_dict["test_inds"] = test_inds

        else:
            self.n_leaf_fills = None
            self.fill_dict = None

    @staticmethod
    def _parse_parameter_transforms(parameter_transforms, output_basis, key_map):
        """Resolve a transforms dict keyed by basis names into output-column-keyed form.

        Keys not in ``output_basis`` are mapped through ``key_map`` (``AssertionError`` if
        absent there too).

        Returns:
            dict: ``"single_param"`` (``{int: callable}``) and ``"mult_param"``
            (``{tuple of int: callable}``).

        Raises:
            ValueError: If a key is not a ``str`` / ``int`` / ``tuple``.
        """
        # differentiate between single and multi parameter transformations
        transforms = {"single_param": {}, "mult_param": {}}

        # iterate through transforms and setup single and multiparameter transforms
        for key, item in parameter_transforms.items():
            if isinstance(key, str) or isinstance(key, int):
                if key not in output_basis:
                    assert key in key_map
                    key = key_map[key]
                key_in = output_basis.index(key)
                transforms["single_param"][key_in] = item
            elif isinstance(key, tuple):
                _tmp = []
                for i in range(len(key)):
                    key_tmp = key[i]
                    if  key_tmp not in output_basis:
                        assert key_tmp in key_map
                        key_tmp = key_map[key_tmp]
                    _tmp.append(output_basis.index(key_tmp))
                transforms["mult_param"][tuple(_tmp)] = item
            else:
                raise ValueError(
                    f"Parameter transform keys must be str (or int) or tuple of strs (or ints). {key} is neither."
                )
        return transforms

    def transform_base_parameters(
        self, params, copy=True, return_transpose=False, xp=None
    ):
        """Apply the forward transforms to output-basis columns.

        Single-parameter transforms run first, then multi-parameter transforms in insertion
        order. Without transforms the input is returned (transposed if requested) as is.

        Args:
            params (xp.ndarray): Coordinates of shape ``(..., ndim_full)`` in OUTPUT-basis
                column order (i.e. after :meth:`fill_values`).
            copy (bool, optional): If ``True``, work on a copy; if ``False``, ``params`` is
                modified in place. Default ``True``.
            return_transpose (bool, optional): If ``True``, return the transposed array
                (parameters on the first axis). Default ``False``.
            xp (module, optional): ``numpy`` or ``cupy``; not otherwise used here. Default
                ``None`` (``numpy``).

        Returns:
            xp.ndarray: Transformed coordinates, shape ``(..., ndim_full)`` (or its
            transpose).
        """

        # cupy or numpy
        if xp is None:
            xp = np

        if self.base_transforms is not None:
            params_temp = params.copy() if copy else params
            params_temp = params_temp.T
            # regular transforms
            for ind, trans_fn in self.base_transforms["single_param"].items():
                params_temp[ind] = trans_fn(params_temp[ind])

            # multi parameter transforms
            for inds, trans_fn in self.base_transforms["mult_param"].items():
                temp = trans_fn(*[params_temp[i].copy() for i in inds])
                for j, i in enumerate(inds):
                    params_temp[i] = temp[j]

            # its actually the opposite now
            if return_transpose:
                return params_temp
            else:
                return params_temp.T

        else:
            if return_transpose:
                return params.T
            else:
                return params

    def fill_values(self, params, xp=None, leaf_inds=None):
        """Expand input-basis coordinates to the output basis and write the fixed values.

        Builds a new float64 array of shape ``params.shape[:-1] + (ndim_full,)``, copies the
        input columns to their output positions (``test_inds``) and writes the fill values
        (per row from ``leaf_inds`` for a per-leaf container). Without a ``fill_dict`` the
        input is returned unchanged (no reordering).

        Args:
            params (xp.ndarray): Coordinates of shape ``(..., ndim)`` in input-basis order.
            xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` (``numpy``).
            leaf_inds (xp.ndarray, optional): Leaf index of each row, shape
                ``params.shape[:-1]``. Required for a per-leaf ``fill_dict``: row ``j`` gets
                the fill values of leaf ``leaf_inds[j]``. Ignored otherwise. Default
                ``None``.

        Returns:
            xp.ndarray: Filled coordinates, shape ``(..., ndim_full)``; ``params`` itself
            when there is no ``fill_dict``.

        Raises:
            ValueError: If a per-leaf container gets no ``leaf_inds`` or ``leaf_inds`` of
                the wrong shape.
        """
        if self.fill_dict is not None:
            if xp is None:
                xp = np

            # get shape
            shape = params.shape

            # setup new array to fill
            params_filled = xp.zeros(shape[:-1] + (self.ndim_full,))
            test_inds = xp.asarray(self.fill_dict["test_inds"])
            # special indexing to properly fill array with params
            indexing_test_inds = tuple([slice(0, temp) for temp in shape[:-1]]) + (
                test_inds,
            )

            # fill values directly from params array
            params_filled[indexing_test_inds] = params

            if self.fill_dict is not None and self.fill_dict != {} and len(self.fill_dict["fill_inds"]) > 0:
                fill_inds = xp.asarray(self.fill_dict["fill_inds"])
                # special indexing to fill fill_values
                indexing_fill_inds = tuple([slice(0, temp) for temp in shape[:-1]]) + (
                    fill_inds,
                )

                if self.n_leaf_fills is not None:
                    # per-leaf fills: select each row's fill values by its leaf index
                    if leaf_inds is None:
                        raise ValueError(
                            "This TransformContainer holds per-leaf fill values; "
                            "pass leaf_inds (shape params.shape[:-1]) to fill_values/both_transforms."
                        )
                    leaf_inds_in = xp.asarray(leaf_inds).astype(int)
                    if leaf_inds_in.shape != shape[:-1]:
                        raise ValueError(
                            f"leaf_inds shape {leaf_inds_in.shape} must match the params "
                            f"leading shape {shape[:-1]}."
                        )
                    params_filled[indexing_fill_inds] = xp.asarray(
                        self.fill_dict["fill_values"]
                    )[leaf_inds_in]
                else:
                    # add fill_values at fill_inds
                    params_filled[indexing_fill_inds] = xp.asarray(
                        self.fill_dict["fill_values"]
                    )

            return params_filled

        else:
            return params

    def both_transforms(
        self, params, copy=True, return_transpose=False, xp=None, leaf_inds=None
    ):
        """Map sampling-basis coordinates to the output basis (fill, then transform).

        Runs :meth:`fill_values` and then :meth:`transform_base_parameters`. Filling first
        means the transforms see the fixed values too (a fixed parameter can
        enter a multi-parameter transform), and every transform key refers to an
        output-basis column. There is no option to reverse this order.

        Args:
            params (xp.ndarray): Coordinates of shape ``(..., ndim)`` in input-basis order.
            copy (bool, optional): Forwarded to :meth:`transform_base_parameters`. Default
                ``True``.
            return_transpose (bool, optional): If ``True``, return the transposed array.
                Default ``False``.
            xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` (``numpy``).
            leaf_inds (xp.ndarray, optional): Per-row leaf indices, forwarded to
                :meth:`fill_values`; required for a per-leaf ``fill_dict``. Default
                ``None``.

        Returns:
            xp.ndarray: Output-basis coordinates, shape ``(..., ndim_full)`` (or its
            transpose).
        """
        # numpy or cupy
        if xp is None:
            xp = np

        # fill (and reorder into the output basis) first, then run the transforms
        temp = self.fill_values(params, xp=xp, leaf_inds=leaf_inds)
        temp = self.transform_base_parameters(
            temp, copy=copy, return_transpose=return_transpose, xp=xp
        )
        return temp

    def _check_inverse_coverage(self):
        """Verify every forward transform has a matching inverse.

        A no-op when there are no forward transforms.

        Raises:
            ValueError: A forward transform exists without a matching entry in
                ``inverse_parameter_transforms``.

        """
        if self.base_transforms is None:
            return

        if self.base_inverse_transforms is None:
            raise ValueError(
                "This TransformContainer has parameter_transforms but no "
                "inverse_parameter_transforms. Pass inverse_parameter_transforms "
                "at initialization to enable inverse transformations."
            )

        missing = []
        for ind in self.base_transforms["single_param"]:
            if ind not in self.base_inverse_transforms["single_param"]:
                missing.append(self.output_basis[ind])
        for inds in self.base_transforms["mult_param"]:
            if inds not in self.base_inverse_transforms["mult_param"]:
                missing.append(tuple(self.output_basis[i] for i in inds))

        if len(missing) > 0:
            raise ValueError(
                f"No inverse transform provided for parameter(s) {missing}. "
                "Add matching entries to inverse_parameter_transforms to enable "
                "inverse transformations."
            )

    def inverse_transform_base_parameters(
        self, params, copy=True, return_transpose=False, xp=None
    ):
        """Apply the inverse transforms, undoing :meth:`transform_base_parameters`.

        Multi-parameter inverses run first (in reverse insertion order, in case transforms
        overlap), then single-parameter inverses: the reverse of the forward order. The
        array stays in the output basis; :meth:`unfill_values` reduces it.

        Args:
            params (xp.ndarray): Output-basis coordinates, shape ``(..., ndim_full)``.
            copy (bool, optional): If ``True``, work on a copy; if ``False``, ``params`` is
                modified in place. Default ``True``.
            return_transpose (bool, optional): If ``True``, return the transposed array.
                Default ``False``.
            xp (module, optional): ``numpy`` or ``cupy``; not otherwise used here. Default
                ``None`` (``numpy``).

        Returns:
            xp.ndarray: Inverse-transformed coordinates, shape ``(..., ndim_full)`` (or its
            transpose).

        Raises:
            ValueError: If a forward transform has no matching inverse.
        """

        # cupy or numpy
        if xp is None:
            xp = np

        self._check_inverse_coverage()

        if self.base_inverse_transforms is not None:
            params_temp = params.copy() if copy else params
            params_temp = params_temp.T

            # multi parameter inverses first (reverse of the forward order,
            # in reverse insertion order in case transforms overlap)
            for inds, trans_fn in reversed(
                list(self.base_inverse_transforms["mult_param"].items())
            ):
                temp = trans_fn(*[params_temp[i].copy() for i in inds])
                for j, i in enumerate(inds):
                    params_temp[i] = temp[j]

            # single parameter inverses
            for ind, trans_fn in self.base_inverse_transforms["single_param"].items():
                params_temp[ind] = trans_fn(params_temp[ind])

            # its actually the opposite now
            if return_transpose:
                return params_temp
            else:
                return params_temp.T

        else:
            if return_transpose:
                return params.T
            else:
                return params

    def unfill_values(self, params, xp=None):
        """Reduce output-basis coordinates to the input basis (inverse of :meth:`fill_values`).

        Selects the input columns (``params[..., test_inds]``), which drops the fixed values
        and undoes the reordering. Without a ``fill_dict`` the input is returned unchanged.

        Args:
            params (xp.ndarray): Output-basis coordinates, shape ``(..., ndim_full)``.
            xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` (``numpy``).

        Returns:
            xp.ndarray: Input-basis coordinates, shape ``(..., ndim)``.
        """
        if self.fill_dict is not None:
            if xp is None:
                xp = np

            test_inds = xp.asarray(self.fill_dict["test_inds"])
            return params[..., test_inds]

        else:
            return params

    def both_inverse_transforms(
        self, params, copy=True, return_transpose=False, xp=None
    ):
        """Map output-basis coordinates back to the sampling basis (unfill after inverting).

        The inverse of :meth:`both_transforms`: runs
        :meth:`inverse_transform_base_parameters` and then :meth:`unfill_values`.

        Args:
            params (xp.ndarray): Output-basis coordinates, shape ``(..., ndim_full)``.
            copy (bool, optional): Forwarded to :meth:`inverse_transform_base_parameters`.
                Default ``True``.
            return_transpose (bool, optional): If ``True``, return the transposed result.
                Default ``False``.
            xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` (``numpy``).

        Returns:
            xp.ndarray: Input-basis coordinates, shape ``(..., ndim)`` (or its transpose).

        Raises:
            ValueError: If a forward transform has no matching inverse.
        """
        # numpy or cupy
        if xp is None:
            xp = np

        # run inverse transforms first, then remove fill values
        temp = self.inverse_transform_base_parameters(params, copy=copy, xp=xp)
        temp = self.unfill_values(temp, xp=xp)

        if return_transpose:
            return temp.T
        else:
            return temp
