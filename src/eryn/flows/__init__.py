# src/eryn/flows/__init__.py
"""Normalizing-flow proposals for the Eryn ensemble MCMC sampler.

This subpackage provides a backend-agnostic foundation for flow-based MCMC
proposals:

- :class:`Flow` — abstract base class with init-arg capture for config
  round-trips and cross-process transport.
- :class:`DataTransform` / :class:`IdentityTransform` — invertible transforms
  with the log-det Jacobian term required for exact MH correction factors.
- :class:`ConditioningStrategy` / :class:`OneHotLeafConditioning` — protocol
  and default implementation for encoding discrete condition ids as context
  vectors.
- :func:`get_flow_wrapper` — lazy factory; backend implementations (torch/zuko)
  are only imported when first requested.

**Optional extra**

The PyTorch backend (``ZukoFlow``, ``WhiteningTransform``) lives in the
``eryn.flows.torch`` subpackage and requires the ``flow`` extra::

    pip install eryn[flow]

Importing this module (``eryn.flows``) never triggers a torch import.

**Executor classes** (``TrainerExecutor``, ``InlineExecutor``, ``ProcessExecutor``,
``TrainerError``) are defined in ``eryn.flows.executors`` and will be available
once that module is implemented (Task 8).
"""
from __future__ import annotations

__all__ = [
    "Flow",
    "FlowHistory",
    "FlowProposalDistribution",
    "DataTransform",
    "IdentityTransform",
    "ConditioningStrategy",
    "OneHotLeafConditioning",
    "TrainerExecutor",
    "InlineExecutor",
    "ProcessExecutor",
    "TrainerError",
    "ZukoFlow",
    "WhiteningTransform",
    "get_flow_wrapper",
]

# No module-level import of torch, scipy, sklearn, .torch, or .executors.
# All heavy imports happen lazily inside __getattr__ or get_flow_wrapper.

# ---- always-available (torch-free) names imported eagerly ----
from .base import Flow, FlowHistory, FlowProposalDistribution  # noqa: E402
from .transforms import DataTransform, IdentityTransform  # noqa: E402
from .conditioning import ConditioningStrategy, OneHotLeafConditioning  # noqa: E402

# ---- lazy names — resolved by __getattr__ on first attribute access ----
_TORCH_NAMES = {"ZukoFlow", "WhiteningTransform"}
_EXECUTOR_NAMES = {"TrainerExecutor", "InlineExecutor", "ProcessExecutor", "TrainerError"}


def get_flow_wrapper(backend: str = "zuko"):
    """Return the flow class and backend module for the given backend name.

    This is the recommended entry point for obtaining a backend-specific flow
    class.  The backend package is only imported on the first call, so
    importing ``eryn.flows`` never imports torch.

    Parameters
    ----------
    backend : str, optional
        Name of the flow backend.  Currently supported: ``"zuko"``.  Planned:
        ``"flowjax"``.  Default is ``"zuko"``.

    Returns
    -------
    flow_class : type
        The backend flow class (e.g. :class:`eryn.flows.torch.ZukoFlow`).
    backend_module : module
        The backend Python module (e.g. ``torch``).

    Raises
    ------
    ImportError
        If the backend package is not installed.  Install with
        ``pip install eryn[flow]``.
    ValueError
        If ``backend`` is not one of the known backends.

    Examples
    --------
    >>> FlowClass, torch = get_flow_wrapper("zuko")
    >>> flow = FlowClass(dims=4, ...)
    """
    _available = ["zuko"]
    if backend == "zuko":
        try:
            import torch  # noqa: F401
            from .torch.flows import ZukoFlow
            return ZukoFlow, torch
        except ImportError as exc:
            raise ImportError(
                "The 'zuko' flow backend requires torch and zuko.  "
                "Install the optional dependencies with:  pip install eryn[flow]"
            ) from exc
    else:
        raise ValueError(
            f"Unknown flow backend {backend!r}.  Available backends: {_available}.  "
            "Note: 'flowjax' is planned for a future release."
        )


def __getattr__(name: str):
    """Lazy attribute access for optional and future names."""
    if name in _TORCH_NAMES:
        try:
            if name == "ZukoFlow":
                from .torch.flows import ZukoFlow
                return ZukoFlow
            elif name == "WhiteningTransform":
                from .torch.transforms import WhiteningTransform
                return WhiteningTransform
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires the torch backend.  "
                "Install the optional dependencies with:  pip install eryn[flow]"
            ) from exc

    if name in _EXECUTOR_NAMES:
        try:
            from . import executors as _executors_mod
            return getattr(_executors_mod, name)
        except ImportError as exc:
            raise ImportError(
                f"'{name}' requires eryn.flows.executors (Task 8).  "
                "This module is not yet available."
            ) from exc

    raise AttributeError(f"module 'eryn.flows' has no attribute {name!r}")
