# -*- coding: utf-8 -*-

from re import L
import warnings

import numpy as np
from itertools import count
from copy import deepcopy

from .backends import Backend, HDFBackend
from .model import Model
from .moves import StretchMove, TemperatureControl, PriorGenerate, GaussianMove
from .pbar import get_progress_bar
from .state import State
from .prior import PriorContainer
from .utils import PlotContainer
from .utils import PeriodicContainer
from .utils.utility import groups_from_inds


__all__ = ["ProductSpaceLikelihood"]

try:
    from collections.abc import Iterable
except ImportError:
    # for py2.7, will be an Exception in 3.8
    from collections import Iterable


class ProductSpaceLikelihood(object):
    def __init__(self, likelihoods, args_list=None, kwargs_list=None, map_fn=None):
        self.num_models = len(likelihoods)
        self.likelihoods = likelihoods

        if args_list is None:
            args_list = [() for _ in range(self.num_models)]
        if kwargs_list is None:
            kwargs_list = [{} for _ in range(self.num_models)]

        self.args_list, self.kwargs_list = args_list, kwargs_list
        assert len(self.args_list) == len(self.kwargs_list) == len(self.likelihoods)

        if map_fn is None:
            map_fn = np.round
        self.map_fn = map_fn

    def __call__(self, x, groups, *args1, **kwargs1):
        # kwargs1 will overwrite kwargs2
        # args1 will add before args 2
        # last parameter is the model index
        # do not actually need groups, 
        # but using provide_groups=True is useful
        model_indicator = self.map_fn(x[-1][:, 0]).astype(int)

        unique_indicators = np.unique(model_indicator)

        ll_out = np.full(model_indicator.shape, -1e300)
        for i in unique_indicators:
            likelihood_fn = self.likelihoods[i]
            inds_model = np.where(model_indicator == i)
            x_input = x[i]  # [inds_model]
            args2 = self.args_list[i]
            kwargs2 = self.kwargs_list[i]

            args = args1 + args2
            kwargs = {**kwargs2, **kwargs1}

            ll_out[inds_model] = likelihood_fn(x_input, *args, **kwargs)

        # TODO: deal with blobs
        return ll_out


