# -*- coding: utf-8 -*-

"""Sampler utilities: parameter containers, runtime hooks, plotting and diagnostics.

- :class:`~eryn.utils.periodic.PeriodicContainer`: periodic distance and wrapping used
  by the proposals.
- :class:`~eryn.utils.transform.TransformContainer`: sampling basis -> likelihood basis
  (fill fixed values, apply parameter transforms) and back.
- :class:`~eryn.utils.plot.PlotContainer`: runtime diagnostic plots
  (``EnsembleSampler(plot_iterations=..., plot_generator=...)``).
- :mod:`~eryn.utils.updates` (:class:`~eryn.utils.updates.Update` and subclasses) and
  :mod:`~eryn.utils.stopping` (:class:`~eryn.utils.stopping.Stopping` and subclasses):
  the ``update_fn`` / ``stopping_fn`` hooks of
  :class:`~eryn.ensemble.EnsembleSampler`.
- :mod:`~eryn.utils.utility`: autocorrelation, evidence and convergence helpers.

The public names of :mod:`~eryn.utils.periodic`, :mod:`~eryn.utils.stopping`,
:mod:`~eryn.utils.transform`, :mod:`~eryn.utils.updates` and :mod:`~eryn.utils.utility` are
re-exported here by star imports; from :mod:`~eryn.utils.plot` only :class:`PlotContainer`
is. ``__all__`` lists only the three containers. Importing this package imports
:mod:`eryn.utils.plot`, which loads matplotlib / corner / seaborn / pandas and updates
matplotlib ``rcParams``.
"""

from .periodic import *
from .plot import PlotContainer
from .stopping import *
from .transform import *
from .updates import *
from .utility import *

__all__ = ["PlotContainer", "PeriodicContainer", "TransformContainer"]
