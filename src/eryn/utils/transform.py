try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    pass

import numpy as np


class TransformContainer:
    """Container for helpful transformations

    Args:
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

    Raises:
        ValueError: Input information is not correct.

    """

    def __init__(self, input_basis, output_basis, parameter_transforms=None, fill_dict=None, key_map={}):

        
        # store originals
        self.original_parameter_transforms = parameter_transforms
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
            # differentiate between single and multi parameter transformations
            self.base_transforms = {"single_param": {}, "mult_param": {}}

            # iterate through transforms and setup single and multiparameter transforms
            for key, item in parameter_transforms.items():
                if isinstance(key, str) or isinstance(key, int):
                    if key not in output_basis:
                        assert key in key_map
                        key = key_map[key]
                    key_in = output_basis.index(key)
                    self.base_transforms["single_param"][key_in] = item
                elif isinstance(key, tuple):
                    _tmp = []
                    for i in range(len(key)):
                        key_tmp = key[i]
                        if  key_tmp not in output_basis:
                            assert key_tmp in key_map
                            key_tmp = key_map[key_tmp]
                        _tmp.append(output_basis.index(key_tmp))
                    self.base_transforms["mult_param"][tuple(_tmp)] = item
                else:
                    raise ValueError(
                        "Parameter transform keys must be str (or int) or tuple of strs (or ints). {} is neither.".format(
                            key
                        )
                    )
        else:
            self.base_transforms = None

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
            self.fill_dict["test_inds"] = test_inds
            self.fill_dict["fill_inds"] = np.asarray(self.fill_dict["fill_inds"])
            self.fill_dict["fill_values"] = np.asarray(self.fill_dict["fill_values"])

        else:
            self.fill_dict = None

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
