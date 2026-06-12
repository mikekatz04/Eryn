# src/eryn/flows/torch/flows.py
"""Torch-backed normalizing-flow implementations for eryn.flows.

Provides :class:`BaseTorchFlow` (shared device/weight/serialisation logic) and
:class:`ZukoFlow` (conditional NSF via the zuko library).

Requires the ``flow`` optional extra (``pip install eryn[flow]``), which
installs ``torch`` and ``zuko``.

The correctness contract (coords-space densities) is documented on
:class:`eryn.flows.base.Flow`.  In brief::

    log q(x) = flow_net.log_prob(z | ctx) + transform.log_abs_det_jacobian(x, z, c)

where ``z = transform.forward(x, c)``.
"""
from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import zuko
import zuko.flows

from eryn.flows.base import Flow, FlowHistory
from eryn.flows.transforms import IdentityTransform

__all__ = ["BaseTorchFlow", "ZukoFlow"]

# ---------------------------------------------------------------------------
# Default NSF hyper-parameters (ported from lisatools FlowModel)
# ---------------------------------------------------------------------------
_DEFAULT_NSF = dict(
    passes=2,
    transforms=8,
    bins=6,
    hidden_features=(256, 256),
    residual=True,
    randperm=True,
)


# ---------------------------------------------------------------------------
# BaseTorchFlow
# ---------------------------------------------------------------------------

class BaseTorchFlow(Flow):
    """Shared PyTorch backend behaviour for eryn flow wrappers.

    Manages device placement, weight hand-off (CPU tensors, detached),
    and HDF5 serialisation.  Concrete backends (e.g. :class:`ZukoFlow`)
    extend this class and set ``self.flow`` via the property setter.

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    device : str, optional
        PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).  Default is
        ``"cpu"``.
    data_transform : DataTransform or None, optional
        Invertible transform applied to samples before the flow.  Defaults to
        :class:`eryn.flows.transforms.IdentityTransform` if ``None``.
    conditioning : ConditioningStrategy or None, optional
        Strategy for encoding integer condition ids as context vectors.
    seed : int, optional
        Random seed applied once at construction via ``torch.manual_seed``.
        Zuko's ``rsample`` / ``rsample_and_log_prob`` do not accept an explicit
        generator, so global seeding at construction is the simplest approach
        that keeps reproducibility without hidden state.  Default is ``1234``.

    Attributes
    ----------
    xp : module
        Array-API namespace marker (``torch``).  Follows the aspire pattern.
    device : str
        Active device string.
    """

    # Array-API namespace marker (aspire pattern).
    xp = torch

    def __init__(
        self,
        dims: int,
        device: str = "cpu",
        data_transform=None,
        conditioning=None,
        seed: int = 1234,
    ):
        super().__init__(
            dims=dims,
            device=device,
            data_transform=data_transform,
            conditioning=conditioning,
        )
        self.seed = int(seed)
        # Global seed at construction.  Zuko distributions (flow(ctx).rsample etc.)
        # operate on the global torch RNG and do not accept an explicit Generator,
        # so seeding globally at construction is both necessary and sufficient.
        torch.manual_seed(self.seed)
        self._flow: nn.Module | None = None

    # ------------------------------------------------------------------
    # flow property / setter
    # ------------------------------------------------------------------

    @property
    def flow(self) -> nn.Module:
        """The underlying ``nn.Module`` (zuko flow or other).

        Raises
        ------
        RuntimeError
            If the flow module has not been set yet.
        """
        if self._flow is None:
            raise RuntimeError("Flow module has not been set.  Call the subclass constructor.")
        return self._flow

    @flow.setter
    def flow(self, module: nn.Module) -> None:
        """Store and move the module to ``self.device``.

        Note: ``torch.compile`` is deliberately NOT applied here.  Compiled
        modules use transformed state_dict keys (``_orig_mod.*``) that break
        weight round-trips (get_weights/set_weights, HDF5 save/load) and
        pickling.  Compilation can be applied externally after serialisation if
        desired.
        """
        self._flow = module.to(self.device)

    # ------------------------------------------------------------------
    # Weight hand-off
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Return the flow's trainable weights as CPU-resident detached tensors.

        The returned dict can be pickled without torch (values are
        :class:`torch.Tensor`, which pickle safely on their own) and can be
        passed across process boundaries.

        Returns
        -------
        dict
            ``{key: tensor}`` where each tensor is detached, on CPU, and cloned
            so that mutating the returned dict does NOT affect the live flow.
        """
        return {k: v.detach().cpu().clone() for k, v in self._flow.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        """Restore trainable weights from a dict produced by :meth:`get_weights`.

        The module is left on its current device after loading.

        Note: ``set_weights`` does NOT clear the logdet cache.  The logdet
        depends only on the data_transform (which is unchanged by this call),
        not on the flow network weights.

        Parameters
        ----------
        weights : dict
            Weight dict in the format returned by :meth:`get_weights`.
        """
        self._flow.load_state_dict(weights)
        # Ensure the module stays on the correct device (load_state_dict may
        # leave parameters on CPU if the dict came from a CPU clone).
        self._flow.to(self.device)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str) -> "BaseTorchFlow":
        """Move the flow module to ``device`` and update ``self.device``.

        Parameters
        ----------
        device : str
            Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns
        -------
        self
        """
        self.device = device
        if self._flow is not None:
            self._flow.to(device)
        return self

    # ------------------------------------------------------------------
    # HDF5 serialisation helpers (shared)
    # ------------------------------------------------------------------

    def _open_h5(self, h5_file, mode: str):
        """Return (handle, should_close) for a path string or open h5py handle."""
        if isinstance(h5_file, (str, os.PathLike)):
            return h5py.File(h5_file, mode), True
        return h5_file, False

    def save(self, h5_file, path: str = "flow") -> None:
        """Save the flow to an HDF5 file.

        Accepts either a path string (file is opened and closed) or an already
        open :class:`h5py.File` / :class:`h5py.Group`.

        Layout::

            <path>/
                config          — JSON-serialisable subset of config_dict()
                data_transform  — pickle blob (np.void scalar dataset)
                conditioning    — pickle blob (np.void scalar dataset)
                weights/
                    <key>       — one float32 dataset per state_dict tensor

        Parameters
        ----------
        h5_file : str or h5py.File
            Destination.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.
        """
        handle, should_close = self._open_h5(h5_file, "a")
        try:
            grp = handle.require_group(path)

            # --- config ---
            cfg = self.config_dict()
            serialisable = _make_serialisable(cfg)
            grp.attrs["config"] = json.dumps(serialisable)

            # --- data_transform ---
            blob = pickle.dumps(self.data_transform)
            if "data_transform" in grp:
                del grp["data_transform"]
            grp.create_dataset("data_transform", data=np.bytes_(blob))

            # --- conditioning ---
            blob = pickle.dumps(self.conditioning)
            if "conditioning" in grp:
                del grp["conditioning"]
            grp.create_dataset("conditioning", data=np.bytes_(blob))

            # --- weights ---
            wgrp = grp.require_group("weights")
            for k, v in self.get_weights().items():
                arr = v.numpy()
                if k in wgrp:
                    del wgrp[k]
                wgrp.create_dataset(k, data=arr)
        finally:
            if should_close:
                handle.close()

    @classmethod
    def load(cls, h5_file, path: str = "flow") -> "BaseTorchFlow":
        """Load a flow from an HDF5 file.

        Parameters
        ----------
        h5_file : str or h5py.File
            Source.
        path : str, optional
            HDF5 group path.  Default is ``"flow"``.

        Returns
        -------
        BaseTorchFlow
            Restored instance with weights, transform, and conditioning.
        """
        if isinstance(h5_file, (str, os.PathLike)):
            handle = h5py.File(h5_file, "r")
            should_close = True
        else:
            handle, should_close = h5_file, False

        try:
            grp = handle[path]

            # --- config ---
            cfg = json.loads(grp.attrs["config"])
            # Normalise lists back to tuples for tuple-typed kwargs (e.g. hidden_features)
            cfg = _restore_tuples(cfg)

            # --- data_transform ---
            raw = bytes(grp["data_transform"][()])
            data_transform = pickle.loads(raw)

            # --- conditioning ---
            raw = bytes(grp["conditioning"][()])
            conditioning = pickle.loads(raw)

            # --- build instance ---
            cfg["data_transform"] = data_transform
            cfg["conditioning"] = conditioning
            instance = cls(**cfg)

            # --- weights ---
            wgrp = grp["weights"]
            weights = {k: torch.tensor(np.array(v)) for k, v in wgrp.items()}
            instance.set_weights(weights)

            return instance
        finally:
            if should_close:
                handle.close()


# ---------------------------------------------------------------------------
# ZukoFlow
# ---------------------------------------------------------------------------

class ZukoFlow(BaseTorchFlow):
    """Conditional normalizing flow backed by the zuko library.

    Wraps a zuko NSF (Neural Spline Flow) or any other ``zuko.flows``
    distribution class, composes it with a :class:`eryn.flows.DataTransform`,
    and implements the unified context contract (``None`` / int / raw vector).

    Parameters
    ----------
    dims : int
        Dimensionality of the target distribution.
    flow_class : str or callable, optional
        Flow architecture.  When a string, resolved via
        ``getattr(zuko.flows, flow_class)``; valid names include
        ``"NSF"``, ``"MAF"``, ``"CNF"``, ``"FFJORD"``.  A callable is used
        directly (strings are preferred for config round-trips).  Default is
        ``"NSF"``.
    device : str, optional
        PyTorch device.  Default is ``"cpu"``.
    data_transform : DataTransform or None, optional
        Invertible preprocessing transform.
    conditioning : ConditioningStrategy or None, optional
        Strategy for encoding integer condition ids as context vectors.  If
        ``None``, the flow is unconditional (``context_dim=0``).
    seed : int, optional
        Random seed.  Default is ``1234``.
    **flow_kwargs
        Additional keyword arguments forwarded to the flow constructor.  When
        ``flow_class="NSF"`` (the default), the following defaults are applied
        and may be overridden::

            passes=2, transforms=8, bins=6,
            hidden_features=(256, 256), residual=True, randperm=True

        For other ``flow_class`` values, no default kwargs are injected (to
        avoid passing invalid arguments).

    Notes
    -----
    ``fit()`` is a stub in this commit; the training loop lands in the next
    task.  A :exc:`NotImplementedError` is raised if ``fit()`` is called.

    Examples
    --------
    >>> from eryn.flows import ZukoFlow, OneHotLeafConditioning
    >>> cond = OneHotLeafConditioning(nleaves_max=2)
    >>> flow = ZukoFlow(dims=4, conditioning=cond, seed=0,
    ...                 transforms=3, hidden_features=(64, 64), bins=5)
    >>> import numpy as np
    >>> x = np.random.randn(10, 4)
    >>> lp = flow.log_prob(x, context=0)
    >>> lp.shape
    (10,)
    """

    def __init__(
        self,
        dims: int,
        flow_class: str | Any = "NSF",
        device: str = "cpu",
        data_transform=None,
        conditioning=None,
        seed: int = 1234,
        **flow_kwargs,
    ):
        super().__init__(
            dims=dims,
            device=device,
            data_transform=data_transform,
            conditioning=conditioning,
            seed=seed,
        )
        self.flow_class = flow_class

        # --- resolve context dimension ---
        context_dim = conditioning.context_dim if conditioning is not None else 0
        self._context_dim = context_dim

        # --- resolve flow constructor ---
        FlowCls = _resolve_flow_class(flow_class)

        # --- merge defaults (NSF only) ---
        if isinstance(flow_class, str) and flow_class == "NSF":
            kw = {**_DEFAULT_NSF, **flow_kwargs}
        else:
            kw = dict(flow_kwargs)

        # --- build and register the flow via the property setter ---
        self.flow = FlowCls(features=dims, context=context_dim, **kw)

        # --- logdet cache (per condition id → float) ---
        self._logdet_cache: dict = {}

    # ------------------------------------------------------------------
    # Context resolution
    # ------------------------------------------------------------------

    def _resolve(self, context) -> tuple:
        """Resolve the unified context argument to (ctx_tensor_or_None, condition_int).

        Parameters
        ----------
        context : None, int, or array-like
            Context following the class-level contract.

        Returns
        -------
        ctx : torch.Tensor or None
            Context tensor of shape ``(context_dim,)`` on ``self.device``,
            or ``None`` for unconditional flows.
        condition : int
            Integer condition id (used for the data_transform and logdet cache).
            Returns ``0`` when context is an array-like (raw vector path).

        Raises
        ------
        ValueError
            If an int context is provided but ``self.conditioning is None``.
        ValueError
            If a non-None context is provided but ``self._context_dim == 0``.
        """
        if context is None:
            return None, 0

        if self._context_dim == 0:
            raise ValueError(
                "This flow was built with context_dim=0 (no conditioning).  "
                "Pass context=None or rebuild with a ConditioningStrategy."
            )

        # Integer condition id path (including numpy integer types)
        if isinstance(context, (int, np.integer)):
            if self.conditioning is None:
                raise ValueError(
                    "An integer context id was provided but this flow has no "
                    "conditioning strategy.  Either supply a ConditioningStrategy "
                    "at construction or pass a raw context vector."
                )
            ctx_np = self.conditioning.encode(int(context))  # float32 (context_dim,)
            ctx = torch.as_tensor(ctx_np, device=self.device)
            return ctx, int(context)

        # Raw array-like context vector path (condition=0 for data_transform)
        ctx = torch.as_tensor(
            np.asarray(context, dtype=np.float32), device=self.device
        )
        return ctx, 0

    # ------------------------------------------------------------------
    # Logdet cache
    # ------------------------------------------------------------------

    def _logdet(self, condition: int) -> float:
        """Return the cached per-condition log-det of the data_transform.

        The WhiteningTransform Jacobian is constant in x (linear+affine+shift),
        so it is computed once per condition on a probe point and cached.

        Parameters
        ----------
        condition : int
            Condition id.

        Returns
        -------
        float
            ``log|det J_forward(x, condition)|`` for any ``x``.
        """
        if condition not in self._logdet_cache:
            probe = np.zeros((1, self.dims), dtype=np.float64)
            z = self.data_transform.forward(probe, condition)
            # log_abs_det_jacobian may return a torch.Tensor or np.ndarray
            val = self.data_transform.log_abs_det_jacobian(probe, z, condition)
            if hasattr(val, "numpy"):
                val = val.numpy()
            self._logdet_cache[condition] = float(np.atleast_1d(np.asarray(val))[0])
        return self._logdet_cache[condition]

    def _clear_logdet_cache(self) -> None:
        """Invalidate the per-condition logdet cache.

        Called by Task 4's ``fit()`` after refitting the data_transform, since
        the Jacobian constant changes when new whitening statistics are
        computed.
        """
        self._logdet_cache.clear()

    # ------------------------------------------------------------------
    # log_prob
    # ------------------------------------------------------------------

    def log_prob(self, x, context=None) -> np.ndarray:
        """Return the coords-space log density at ``x``.

        Implements the correctness contract::

            log q(x) = flow_net.log_prob(z | ctx)
                       + data_transform.log_abs_det_jacobian(x, z, condition)

        Parameters
        ----------
        x : array-like, shape (N, dims)
            Sample points in coords space.
        context : None, int, or array-like, optional
            Context following the class-level contract.

        Returns
        -------
        log_prob : np.ndarray, shape (N,), dtype float64
        """
        x = np.asarray(x, dtype=np.float64)
        ctx, condition = self._resolve(context)

        # Apply data transform: coords → flow space
        z = self.data_transform.forward(x, condition)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)

        with torch.no_grad():
            dist = self._get_dist(ctx, n=len(x))
            log_flow = dist.log_prob(z_t)  # shape (N,)

        logdet = self._logdet(condition)
        return (log_flow.cpu().numpy() + logdet).astype(np.float64)

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(self, n: int, context=None) -> np.ndarray:
        """Draw ``n`` samples from the flow in coords space.

        Parameters
        ----------
        n : int
            Number of samples.
        context : None, int, or array-like, optional
            Context.

        Returns
        -------
        samples : np.ndarray, shape (n, dims), dtype float64
        """
        x, _ = self.sample_and_log_prob(n, context=context)
        return x

    # ------------------------------------------------------------------
    # sample_and_log_prob
    # ------------------------------------------------------------------

    def sample_and_log_prob(self, n: int, context=None) -> tuple:
        """Draw ``n`` samples and return their coords-space log densities.

        Uses zuko's ``rsample_and_log_prob`` for a single forward pass.

        Parameters
        ----------
        n : int
            Number of samples.
        context : None, int, or array-like, optional
            Context.

        Returns
        -------
        samples : np.ndarray, shape (n, dims), dtype float64
            Samples in coords space (inverse-transformed from flow space).
        log_prob : np.ndarray, shape (n,), dtype float64
            Coords-space log density at each sample.
        """
        ctx, condition = self._resolve(context)
        with torch.no_grad():
            dist = self._get_dist(ctx, n=n)
            z_t, log_flow = dist.rsample_and_log_prob((n,))

        z_t = z_t.reshape(n, self.dims)
        log_flow = log_flow.reshape(n)

        # Inverse-transform: flow space → coords space
        z_cpu = z_t.cpu()
        x = self.data_transform.inverse(z_cpu, condition)
        x = np.asarray(x, dtype=np.float64)

        logdet = self._logdet(condition)
        logq = (log_flow.cpu().numpy() + logdet).astype(np.float64)
        return x, logq

    # ------------------------------------------------------------------
    # fit stub
    # ------------------------------------------------------------------

    def fit(self, samples, **kwargs) -> FlowHistory:
        """Train the flow on ``samples``.

        .. note::
            Not yet implemented.  The training loop lands in the next commit
            (Task 4).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "ZukoFlow.fit is implemented in the next commit (training loop)."
        )

    # ------------------------------------------------------------------
    # Internal helper: build / cache distribution for a given context
    # ------------------------------------------------------------------

    def _get_dist(self, ctx, n: int = 1):
        """Return the zuko conditional distribution for the given context tensor.

        Parameters
        ----------
        ctx : torch.Tensor or None
            Context tensor of shape ``(context_dim,)`` or ``None``.  A single
            1-D context vector is passed directly to ``self._flow(ctx)``.  Zuko
            broadcasts a 1-D context over the sample batch dimension, so this
            works for both ``log_prob(batch)`` and
            ``rsample_and_log_prob((n,))``.
        n : int
            Unused (kept for API symmetry).

        Returns
        -------
        zuko distribution
        """
        if ctx is None:
            return self._flow()
        # Pass a 1-D context vector: zuko broadcasts it over the batch.
        # Do NOT expand to (N, context_dim) — that makes zuko interpret the
        # leading N as a batch of *different* contexts, producing (N, n, dims)
        # shaped samples instead of (n, dims).
        return self._flow(ctx)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_flow_class(flow_class):
    """Resolve a flow class from a string name or callable.

    Parameters
    ----------
    flow_class : str or callable
        If a string, resolved via ``getattr(zuko.flows, flow_class)``.

    Returns
    -------
    callable
        The flow constructor.

    Raises
    ------
    AttributeError
        If the string name is not found in ``zuko.flows``.
    """
    if callable(flow_class) and not isinstance(flow_class, str):
        return flow_class
    if isinstance(flow_class, str):
        try:
            return getattr(zuko.flows, flow_class)
        except AttributeError:
            valid = [n for n in dir(zuko.flows) if not n.startswith("_")]
            raise AttributeError(
                f"'{flow_class}' is not a valid zuko.flows class.  "
                f"Some valid names: {valid[:10]}"
            ) from None
    raise TypeError(f"flow_class must be a str or callable, got {type(flow_class)}")


def _make_serialisable(cfg: dict) -> dict:
    """Recursively convert a config dict to a JSON-serialisable form.

    Tuples become lists (JSON has no tuple type).  Non-serialisable objects
    (DataTransform, ConditioningStrategy) are dropped; they are stored separately
    as pickle blobs.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    dict
    """
    out = {}
    _skip_types = ("DataTransform", "IdentityTransform", "WhiteningTransform",
                   "ConditioningStrategy", "OneHotLeafConditioning")
    for k, v in cfg.items():
        # Skip objects that should be pickled separately
        type_name = type(v).__name__
        if type_name in _skip_types:
            continue
        if v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar_or_skip(i) for i in v]
        else:
            # Skip non-serialisable objects (e.g. DataTransform instances)
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                pass  # silently skip; restored via pickle
    return out


def _scalar_or_skip(v):
    """Convert a scalar to JSON-safe type; return it directly if already safe."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _restore_tuples(cfg: dict) -> dict:
    """Normalise config values loaded from JSON.

    Lists that were originally tuples are left as lists; ZukoFlow's
    ``__init__`` accepts both.  No structural change is needed here, but
    we strip keys whose values are ``None`` only when they represent
    objects stored as separate pickle blobs (data_transform, conditioning).

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    dict
    """
    # data_transform and conditioning are loaded separately and injected by load();
    # remove them from the JSON-derived config to avoid double-assignment.
    cfg.pop("data_transform", None)
    cfg.pop("conditioning", None)
    return cfg
