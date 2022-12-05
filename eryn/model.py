# -*- coding: utf-8 -*-

from collections import namedtuple

__all__ = ["Model"]


Model = namedtuple(
    "Model",
    (
        "log_like_fn",
        "compute_log_like_fn",
        "compute_log_prior_fn",
        "temperature_control",
        "map_fn",
        "random",
    ),
)
