# -*- coding: utf-8 -*-
"""Storage backends for the sampler chain (not compute backends).

- :class:`Backend`: in-memory ``numpy`` storage (the sampler's default).
- :class:`HDFBackend`: the same layout in an HDF5 file (requires ``h5py``); pass a file
  name as ``EnsembleSampler(..., backend="chain.h5")`` to use it.
- :class:`TempHDFBackend`: an :class:`HDFBackend` on a temporary file, for tests.

The deprecated :class:`~eryn.backends.parabackend.ParaBackend` (for the deprecated
:class:`~eryn.paraensemble.ParaEnsembleSampler`) is not re-exported here; import it from
:mod:`eryn.backends.parabackend`.
"""

from .backend import Backend
from .hdfbackend import HDFBackend, TempHDFBackend

__all__ = ["Backend", "HDFBackend", "TempHDFBackend", "get_test_backends"]


def get_test_backends():
    """Return the backend classes available for tests.

    Returns:
        list of type: ``[Backend]``, plus :class:`TempHDFBackend` when ``h5py`` is
        importable.
    """
    backends = [Backend]

    try:
        import h5py  # NOQA
    except ImportError:
        pass
    else:
        backends.append(TempHDFBackend)

    return backends
