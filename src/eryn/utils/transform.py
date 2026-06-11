try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    pass

import numpy as np


class TransformContainer:
    """Container for helpful transformations

    Args:
        input_basis (list): List of integers or strings representing each 
            basis element from the input basis.
        output_basis (list): List of integers or strings representing each 
            basis element for the output basis.
        parameter_transforms (dict, optional): Keys are ``int`` or ``tuple``
            of ``int`` that contain the indexes into the parameters
            that correspond to the transformation added as the Values to the
            dict. If using ``fill_values``, you must be careful with
            making sure parameter transforms properly comes before or after
            filling values. ``int`` indicate single parameter transforms. These
            are performed first. ``tuple`` of ``int`` indicates multiple
            parameter transforms. These are performed after single-parameter transforms. 
            (default: ``None``)
        fill_dict (dict, optional): Keys must contain ``'ndim_full'``, ``'fill_inds'``,
            and ``'fill_values'``. ``'ndim_full'`` is the full last dimension of the final
            array after fill_values are added. 'fill_inds' and 'fill_values' are
            np.ndarray[number of fill values] that contain the indexes and corresponding values
            for filling. (default: ``None``)
        inverse_parameter_transforms (dict, optional): Inverse transformations going
            from the output basis back to the input basis. Keyed identically to
            ``parameter_transforms`` (output-basis names / indexes, mapped through
            ``key_map`` as needed). Each value must be the functional inverse of the
            corresponding entry in ``parameter_transforms``. These are used by
            :func:`inverse_transform_base_parameters` and
            :func:`both_inverse_transforms`; if a forward transform has no matching
            inverse, calling those methods raises a ``ValueError``.
            (default: ``None``)

    Raises:
        ValueError: Input information is not correct.

    """

    def __init__(self, input_basis=None, output_basis=None, parameter_transforms=None, fill_dict=None, key_map={}, inverse_parameter_transforms=None):


        # store originals
        self.original_parameter_transforms = parameter_transforms
        self.original_inverse_parameter_transforms = inverse_parameter_transforms
        self.ndim_full = len(output_basis)
        self.ndim = len(input_basis)

        self.input_basis, self.output_basis = input_basis, output_basis

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
            if not isinstance(fill_dict, dict):
                raise ValueError("fill_dict must be a dictionary.")

            self.fill_dict = {}
            self.fill_dict["fill_inds"] = []
            self.fill_dict["fill_values"] = []
            for key in fill_dict.keys():
                self.fill_dict["fill_inds"].append(output_basis.index(key))
                self.fill_dict["fill_values"].append(fill_dict[key])

            # set up test_inds accordingly
            # dtype=int keeps an empty fill_dict ({}) usable as an index array
            self.fill_dict["test_inds"] = test_inds
            self.fill_dict["fill_inds"] = np.asarray(self.fill_dict["fill_inds"], dtype=int)
            self.fill_dict["fill_values"] = np.asarray(self.fill_dict["fill_values"], dtype=float)

        else:
            self.fill_dict = None

    @staticmethod
    def _parse_parameter_transforms(parameter_transforms, output_basis, key_map):
        """Resolve a transforms dict keyed by basis names into index-keyed form.

        Returns a dict with ``"single_param"`` (``{int: callable}``) and
        ``"mult_param"`` (``{tuple of int: callable}``) entries.
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
        """Transform the base parameters

        Args:
            params (np.ndarray[..., ndim]): Array with coordinates. This array is
                transformed according to the ``self.base_transforms`` dictionary.
            copy (bool, optional): If True, copy the input array.
                (default: ``True``)
            return_transpose (bool, optional): If True, return the transpose of the
                array. (default: ``False``)
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``) 

        Returns:
            np.ndarray[..., ndim]: Transformed ``params`` array.

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
                temp = trans_fn(*[params_temp[i] for i in inds])
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

    def fill_values(self, params, xp=None):
        """fill fixed parameters

        This also adjusts parameter order as needed between the two bases. 

        Args:
            params (np.ndarray[..., ndim]): Array with coordinates. This array is
                filled with values according to the ``self.fill_dict`` dictionary.
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``) 

        Returns:
            np.ndarray[..., ndim_full]: Filled ``params`` array.

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

            fill_inds = xp.asarray(self.fill_dict["fill_inds"])
            # special indexing to fill fill_values
            indexing_fill_inds = tuple([slice(0, temp) for temp in shape[:-1]]) + (
                fill_inds,
            )

            # add fill_values at fill_inds
            params_filled[indexing_fill_inds] = xp.asarray(
                self.fill_dict["fill_values"]
            )

            return params_filled

        else:
            return params

    def both_transforms(
        self, params, copy=True, return_transpose=False, xp=None
    ):
        """Transform the parameters and fill fixed parameters

        This fills the fixed parameters and then transforms all of them. Therefore, the user
        must be careful with the indexes input. 

        This is generally the direction recommended because fixed parameters may change
        non-fixed parameters during parameter transformations. This can be reversed
        with the ``reverse`` kwarg.

        Args:
            params (np.ndarray[..., ndim]): Array with coordinates. This array is
                transformed according to the ``self.base_transforms`` dictionary.
            copy (bool, optional): If True, copy the input array.
                (default: ``True``)
            return_transpose (bool, optional): If ``True``, return the transpose of the
                array. (default: ``False``)
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``) 

        Returns:
            np.ndarray[..., ndim]: Transformed and filleds ``params`` array.

        """
        # numpy or cupy
        if xp is None:
            xp = np

        # run transforms first
        temp = self.fill_values(params, xp=xp)
        temp = self.transform_base_parameters(
            temp, copy=copy, return_transpose=return_transpose, xp=xp
        )
        return temp

    def _check_inverse_coverage(self):
        """Verify every forward transform has a matching inverse.

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
        """Inverse-transform the base parameters (output basis -> input basis values)

        This applies the user-provided ``inverse_parameter_transforms``, undoing
        :func:`transform_base_parameters`. Multi-parameter inverses are applied
        first, then single-parameter inverses (the reverse of the forward order).

        Args:
            params (np.ndarray[..., ndim_full]): Array with coordinates in the output
                basis. This array is transformed according to the
                ``self.base_inverse_transforms`` dictionary.
            copy (bool, optional): If True, copy the input array.
                (default: ``True``)
            return_transpose (bool, optional): If True, return the transpose of the
                array. (default: ``False``)
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        Returns:
            np.ndarray[..., ndim_full]: Inverse-transformed ``params`` array.

        Raises:
            ValueError: A forward transform exists without a matching inverse.

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
                temp = trans_fn(*[params_temp[i] for i in inds])
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
        """remove fixed parameters (inverse of :func:`fill_values`)

        This selects the input-basis entries out of an output-basis array,
        dropping the filled (fixed) values and undoing any parameter reordering
        between the two bases.

        Args:
            params (np.ndarray[..., ndim_full]): Array with coordinates in the output
                basis. The fill entries given by ``self.fill_dict`` are removed.
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        Returns:
            np.ndarray[..., ndim]: ``params`` array reduced to the input basis.

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
        """Inverse-transform the parameters and remove fixed parameters

        This is the inverse of :func:`both_transforms`: it applies the inverse
        parameter transforms in the output basis and then removes the fill
        entries, returning coordinates in the input (sampling) basis.

        Args:
            params (np.ndarray[..., ndim_full]): Array with coordinates in the output
                basis.
            copy (bool, optional): If True, copy the input array.
                (default: ``True``)
            return_transpose (bool, optional): If ``True``, return the transpose of the
                array. (default: ``False``)
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        Returns:
            np.ndarray[..., ndim]: Inverse-transformed ``params`` array in the input basis.

        Raises:
            ValueError: A forward transform exists without a matching inverse.

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
