# src/eryn/flows/torch/__init__.py
"""Torch-backend implementations for eryn.flows.

Requires the ``flow`` optional extra (``pip install eryn[flow]``), which
installs ``torch`` and ``zuko``.

This subpackage provides:

- :class:`WhiteningTransform` — periodic-aware per-condition whitening
  preprocessor implementing the :class:`eryn.flows.DataTransform` ABC.

``ZukoFlow`` will be added here in Task 3.
"""
from __future__ import annotations

from .transforms import WhiteningTransform

__all__ = [
    "WhiteningTransform",
]
