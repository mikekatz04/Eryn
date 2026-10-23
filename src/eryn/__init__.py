"""Eryn: ensemble MCMC with parallel tempering, reversible jump and multiple branches.

The package namespace holds only version metadata (``__version__``,
``__version_tuple__``, and ``_is_editable``, ``True`` for an editable install). Import the
components from their submodules:

- :class:`eryn.ensemble.EnsembleSampler`: the sampler;
- :mod:`eryn.state`: :class:`~eryn.state.State`, :class:`~eryn.state.Branch`,
  :class:`~eryn.state.BranchSupplemental`;
- :mod:`eryn.moves`: proposals and :class:`~eryn.moves.tempering.TemperatureControl`;
- :mod:`eryn.priors`: prior distributions and :class:`~eryn.priors.ProbDistContainer`;
- :mod:`eryn.backends`: chain storage (:class:`~eryn.backends.Backend`,
  :class:`~eryn.backends.HDFBackend`);
- :mod:`eryn.utils`: periodic and transform containers, update and stopping callbacks,
  plotting and diagnostics.

Eryn ships no compiled code; the deprecated GPU path of
:class:`~eryn.paraensemble.ParaEnsembleSampler` takes its array module from
``gpubackendtools``.

Example:
    >>> import eryn
    >>> isinstance(eryn.__version__, str)
    True
"""

# ruff: noqa: E402
try:
    from eryn._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False

__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
]
