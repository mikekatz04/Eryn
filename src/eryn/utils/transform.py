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

    def __init__(self, parameter_transforms=None, fill_dict=None):

        # store originals
        self.original_parameter_transforms = parameter_transforms
        if parameter_transforms is not None:
            # differentiate between single and multi parameter transformations
            self.base_transforms = {"single_param": {}, "mult_param": {}}

            # iterate through transforms and setup single and multiparameter transforms
            for key, item in parameter_transforms.items():
                if isinstance(key, int):
                    self.base_transforms["single_param"][key] = item
                elif isinstance(key, tuple):
                    self.base_transforms["mult_param"][key] = item
                else:
                    raise ValueError(
                        "Parameter transform keys must be int or tuple of ints. {} is neither.".format(
                            key
                        )
                    )
        else:
            self.base_transforms = None

        if fill_dict is not None:
            if not isinstance(fill_dict, dict):
                raise ValueError("fill_dict must be a dictionary.")

            self.fill_dict = fill_dict
            fill_dict_keys = list(self.fill_dict.keys())
            for key in ["ndim_full", "fill_inds", "fill_values"]:
                # check to make sure it has all necessary pieces
                if key not in fill_dict_keys:
                    raise ValueError(
                        f"If providing fill_inds, dictionary must have {key} as a key."
                    )
            # check all the inputs
            if not isinstance(fill_dict["ndim_full"], int):
                raise ValueError("fill_dict['ndim_full'] must be an int.")
            if not isinstance(fill_dict["fill_inds"], np.ndarray):
                raise ValueError("fill_dict['fill_inds'] must be an np.ndarray.")
            if not isinstance(fill_dict["fill_values"], np.ndarray):
                raise ValueError("fill_dict['fill_values'] must be an np.ndarray.")

            # set up test_inds accordingly
            self.fill_dict["test_inds"] = np.delete(
                np.arange(self.fill_dict["ndim_full"]), self.fill_dict["fill_inds"]
            )

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
            params_filled = xp.zeros(shape[:-1] + (self.fill_dict["ndim_full"],))
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
        self, params, copy=True, return_transpose=False, reverse=False, xp=None
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
            reverse (bool, optional): If ``True`` perform the filling after the transforms. This makes
                indexing easier, but removes the ability of fixed parameters to affect transforms. 
                (default: ``False``)
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``) 

        Returns:
            np.ndarray[..., ndim]: Transformed and filleds ``params`` array.

        """
        # numpy or cupy
        if xp is None:
            xp = np

        # run transforms first
        if reverse:
            temp = self.transform_base_parameters(
                params, copy=copy, return_transpose=return_transpose, xp=xp
            )
            temp = self.fill_values(temp, xp=xp)

        else:
            temp = self.fill_values(params, xp=xp)
            temp = self.transform_base_parameters(
                temp, copy=copy, return_transpose=return_transpose, xp=xp
            )
        return temp
