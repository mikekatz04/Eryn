# -*- coding: utf-8 -*-
"""The :class:`Model` record handed to every move.

:meth:`EnsembleSampler.get_model <eryn.ensemble.EnsembleSampler.get_model>` builds one
:class:`Model` per :meth:`~eryn.ensemble.EnsembleSampler.sample` call and passes it to each
``move.propose(model, state)``. Its fields, in order:

- ``log_like_fn``: the user likelihood wrapped with the sampler's ``args`` / ``kwargs``;
- ``compute_log_like_fn``: :meth:`~eryn.ensemble.EnsembleSampler.compute_log_like`,
  called as ``compute_log_like_fn(coords, inds=..., logp=..., supps=..., branch_supps=...)``
  and returning ``(log_like, blobs)``;
- ``compute_log_prior_fn``: :meth:`~eryn.ensemble.EnsembleSampler.compute_log_prior`,
  called as ``compute_log_prior_fn(coords, inds=...)``;
- ``temperature_control``: the sampler's
  :class:`~eryn.moves.tempering.TemperatureControl`, or ``None``;
- ``map_fn``: ``pool.map`` or the built-in ``map``;
- ``random``: the sampler's ``numpy.random.RandomState``.
"""

from collections import namedtuple

__all__ = ["Model"]


#: namedtuple ``Model(log_like_fn, compute_log_like_fn, compute_log_prior_fn,
#: temperature_control, map_fn, random)``: what a move receives from the sampler
#: (see the module docstring for the fields).
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
