# src/eryn/flows/executors.py
"""Trainer-executor seam for online flow training during MCMC.

This module defines the interface a normalizing-flow proposal uses to train a
flow *while sampling*, without ever blocking the proposal on the training step.
It is intentionally torch-free at import time: flows are manipulated only
through the :class:`eryn.flows.base.Flow` interface, and the concrete flow class
is carried by reference (a module-level type) inside :class:`FlowSpec`.  The
heavy backend (torch/zuko) is only imported when a worker actually builds and
trains the flow.

Three pieces live here:

- :class:`FlowSpec` — a picklable ``(class, config, weights)`` snapshot of a
  flow with its data transform **frozen**, used to build an identically
  configured clone in a worker (or inline).
- :class:`TrainerExecutor` — the abstract, non-blocking training interface.
- :class:`InlineExecutor` — a synchronous reference implementation that trains a
  *cloned* flow on every Nth accepted batch and surfaces failures lazily, behind
  exactly the contract a future multiprocessing executor will implement.

Design contract
----------------
- A proposal NEVER blocks on training: :meth:`TrainerExecutor.submit` and
  :meth:`TrainerExecutor.latest_weights` are non-blocking.
- Failures surface loudly: a dead/failed trainer raises :class:`TrainerError`
  from :meth:`TrainerExecutor.latest_weights` (and, for the inline executor,
  from subsequent :meth:`submit` calls).  Callers do not swallow it.
- The data transform is FROZEN for the executor's lifetime: workers always call
  ``flow.fit(..., refit_data_transform=False)`` so the coords-latent map is
  identical on both sides of any process boundary and a weights-only handoff
  stays exact.  :meth:`FlowSpec.from_flow` therefore asserts the transform is
  already fitted.
- Everything that may cross a future process boundary is picklable;
  :meth:`FlowSpec.from_flow` eagerly round-trips ``pickle.dumps(spec)`` so
  unpicklable user objects fail fast in the parent with a clear message.
"""
from __future__ import annotations

import copy
import pickle
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TrainerError",
    "FlowSpec",
    "TrainerExecutor",
    "InlineExecutor",
]


class TrainerError(RuntimeError):
    """Raised when the background trainer has failed.

    The exception carries the child-side error message (and, for a future
    process executor, the child traceback) so the failure is visible in the
    parent.  Proposals do not catch this: a long run that silently keeps
    sampling from a frozen or broken flow is worse than a loud crash.
    """


@dataclass
class FlowSpec:
    """Picklable snapshot of a flow: ``(class, config, initial_weights)``.

    A :class:`FlowSpec` carries everything needed to rebuild an identically
    configured flow elsewhere (in a worker process, or inline) *without*
    pickling the live flow object itself.  The flow class is referenced by
    type (a module-level class pickles by reference); the configuration and the
    CPU weights are deep-copied snapshots.

    The data transform inside ``config`` is **frozen**: the worker that builds
    from this spec must train with ``refit_data_transform=False`` so that the
    coords-latent map matches the parent's exactly and a weights-only handoff
    back to the parent stays numerically exact.

    Parameters
    ----------
    flow_class : type
        Concrete :class:`~eryn.flows.base.Flow` subclass.  Pickled by reference,
        so it must be importable by its module path on both sides of any process
        boundary.
    config : dict
        Constructor keyword arguments (``flow.config_dict()``) with the
        ``device`` entry overridden for the worker.  Contains the
        ``data_transform`` / ``conditioning`` objects themselves; building from
        the spec reconstructs an equivalent flow.
    initial_weights : dict
        CPU-resident weights from ``flow.get_weights()``.

    Notes
    -----
    Construct via :meth:`from_flow` rather than directly: ``from_flow`` enforces
    the frozen-transform contract and verifies picklability eagerly.
    """

    flow_class: type
    config: dict
    initial_weights: dict

    @classmethod
    def from_flow(cls, flow, worker_device: str = "cpu") -> "FlowSpec":
        """Snapshot ``flow`` into a picklable :class:`FlowSpec`.

        Parameters
        ----------
        flow : eryn.flows.base.Flow
            Flow whose data transform is already fitted.  Its class must be
            importable by reference and its config/weights must be picklable.
        worker_device : str, optional
            Device string written into ``config["device"]`` for the rebuilt
            flow.  Default is ``"cpu"`` — training a clone on CPU is the safe
            default and keeps this snapshot device-agnostic.

        Returns
        -------
        FlowSpec
            A snapshot that has been verified to pickle cleanly.

        Raises
        ------
        ValueError
            If ``flow.data_transform`` is not fitted.  The data transform is
            frozen for the executor's lifetime, so it must be fitted *before*
            an executor is created — e.g. by calling ``flow.fit`` on warmup
            samples once (which fits the transform), or by fitting the transform
            directly.  Workers train with ``refit_data_transform=False`` and
            never re-fit it.
        TypeError
            If the assembled spec is not picklable (e.g. the data transform or
            conditioning closes over a lambda or holds a device tensor).  The
            message names the offending concept.
        """
        # Frozen-transform contract: the transform must already be fitted so the
        # coords-latent map is fixed for the executor's whole lifetime.
        if not flow.data_transform.is_fitted:
            raise ValueError(
                "FlowSpec.from_flow requires flow.data_transform.is_fitted == True. "
                "The data transform is FROZEN for the executor's lifetime (workers "
                "always fit with refit_data_transform=False), so it must be fitted "
                "before the executor is created.  Fit it first — e.g. call "
                "flow.fit(warmup_samples) once (which fits the transform), or fit "
                "the transform directly — then build the executor."
            )

        # Deep-copy config so later mutation of the live flow cannot leak in,
        # and override device for the worker / clone.
        config = copy.deepcopy(flow.config_dict())
        config["device"] = worker_device

        spec = cls(
            flow_class=type(flow),
            config=config,
            initial_weights=flow.get_weights(),
        )

        # Eagerly round-trip so unpicklable user objects fail fast in the parent
        # with a clear, actionable message rather than deep inside a worker spawn.
        try:
            pickle.dumps(spec)
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            raise TypeError(
                "FlowSpec must be picklable to cross a process boundary; "
                "check data_transform/conditioning for closures (e.g. lambdas) "
                "or device tensors.  Original error: " + repr(exc)
            ) from exc

        return spec

    def build(self):
        """Reconstruct a live flow from this spec.

        Returns
        -------
        eryn.flows.base.Flow
            A new ``flow_class`` instance built from ``config`` with
            ``initial_weights`` loaded.  Equivalent to the snapshotted flow
            (same config + weights → same densities).
        """
        flow = self.flow_class(**self.config)
        flow.set_weights(self.initial_weights)
        return flow


class TrainerExecutor(ABC):
    """Non-blocking interface for online flow training.

    Implementations train a flow asynchronously (inline-synchronously for
    :class:`InlineExecutor`, in a subprocess for the future process executor)
    and hand new weights back to the caller on request.  Every method on this
    interface is non-blocking with respect to the actual training step: a
    proposal calling :meth:`submit` / :meth:`latest_weights` never waits for a
    fit to finish.

    The executor trains a **clone** of the caller's flow (built from
    :class:`FlowSpec`), not the caller's object.  Weights flow back to the
    caller only through :meth:`latest_weights`; the caller then applies them to
    its own flow via ``flow.set_weights``.

    Failures are surfaced loudly: once training has failed,
    :meth:`latest_weights` raises :class:`TrainerError`.
    """

    @abstractmethod
    def submit(self, samples_by_condition: dict) -> bool:
        """Offer a batch of training samples to the trainer (non-blocking).

        Parameters
        ----------
        samples_by_condition : dict[int, np.ndarray]
            Mapping from condition id to an array of shape ``(N, dims)`` of
            coords-space samples to add to the trainer's buffer.

        Returns
        -------
        bool
            ``True`` if the batch was accepted, ``False`` if it was dropped or
            coalesced (e.g. the trainer is busy, or the executor is shut down).

        Empty harvests
        --------------
        Zero-length arrays carry no training data and must be dropped before
        buffering; a submit with no rows at all (an empty dict, or only empty
        arrays) returns ``False`` and is **not** counted as an accepted submit.
        An empty harvest — e.g. a sampling step with no active leaves — is a
        normal condition, not an error, and must never put the executor into a
        failed state.
        """

    @abstractmethod
    def latest_weights(self):
        """Return the newest ``(version, weights)`` seen so far (non-blocking).

        Returns
        -------
        tuple[int, dict] or None
            The newest ``(version, weights)`` observed, or ``None`` if no
            training has completed yet.  Calling this updates :attr:`version`.

        Ownership contract
        ------------------
        The returned ``weights`` dict is the executor's **private snapshot**.
        Callers may *load* it (``flow.set_weights(weights)`` copies the values
        into the module's parameters) but must **not** mutate it in place.  An
        executor must guarantee that successive snapshots are not aliased to
        live training state: mutating a previously returned dict, or training
        further, must never change a dict a caller is still holding.  A
        process-backed executor satisfies this naturally (each returned dict is
        a fresh deserialized copy); an inline executor must deep-copy the
        clone's weights when stashing them so it matches that semantics.

        Raises
        ------
        TrainerError
            If the trainer has failed.
        """

    @property
    @abstractmethod
    def version(self) -> int:
        """Newest version OBSERVED via :meth:`latest_weights` (``0`` = none yet).

        This is *poll-advanced*: it tracks the highest version a caller has
        actually pulled through :meth:`latest_weights`, NOT the trainer's
        internal "produced" counter.  A trainer may have completed a newer fit
        that no one has polled yet; until :meth:`latest_weights` returns it,
        :attr:`version` does not advance.  This is THE contract for every
        executor, including process-based ones — callers use it to decide
        whether a hot-reload is needed.
        """

    @abstractmethod
    def shutdown(self, timeout: float = 10.0) -> None:
        """Shut the executor down (idempotent).

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for an in-flight trainer to stop.  Default
            is ``10.0``.  Ignored by purely synchronous implementations.
        """

    def __enter__(self) -> "TrainerExecutor":
        """Enter the runtime context, returning ``self``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context, calling :meth:`shutdown`."""
        self.shutdown()


class InlineExecutor(TrainerExecutor):
    """Synchronous reference :class:`TrainerExecutor` that trains a cloned flow.

    On every ``train_every``-th *accepted* :meth:`submit` (once at least
    ``min_train_samples`` are buffered) the executor fits its own cloned flow on
    the concatenated buffer and stashes the new ``(version, weights)``.  Despite
    training synchronously inside :meth:`submit`, it honours the same
    non-blocking *contract* a multiprocessing executor will: the caller's flow
    is never touched, and weights only flow back through :meth:`latest_weights`.

    The clone is built via ``FlowSpec.from_flow(flow).build()`` — the executor
    does NOT train the caller's flow object.  The data transform is frozen
    (fits use ``refit_data_transform=False``), matching the cross-process
    handoff semantics where parent and child must agree on the coords-latent
    map.

    Failure semantics
    -----------------
    If ``flow.fit`` raises during a :meth:`submit`, the executor does **not**
    propagate it from that ``submit`` call directly.  Instead it records the
    failure and raises :class:`TrainerError` from the **next**
    :meth:`latest_weights` call and from every subsequent :meth:`submit`.  This
    mirrors the deferred error surfacing of a process-backed trainer, where the
    failure is only observed when the parent next polls — and it keeps the
    "submit never blocks on / fails because of training" contract uniform across
    executors.

    Parameters
    ----------
    flow : eryn.flows.base.Flow
        Template flow with a fitted data transform.  A clone is built from a
        :class:`FlowSpec` snapshot; the caller's object is never mutated.
    fit_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``flow.fit`` (e.g. ``n_epochs``,
        ``lr``).  ``refit_data_transform`` and ``verbose`` are controlled by the
        executor; supplying either raises :class:`ValueError` from ``__init__``.
        Default is ``None`` (empty).
    train_every : int, optional
        Train on every ``train_every``-th accepted submit.  Default is ``1``
        (train on every submit).
    min_train_samples : int, optional
        Minimum total buffered samples (summed over conditions) required before
        a fit runs.  Default is ``1``.  Training is always additionally guarded
        on a non-empty buffer, so this is self-documenting rather than a
        behaviour change: a fit never runs on zero rows.
    max_buffer_samples : int, optional
        Per-condition ring-buffer cap.  When a condition's buffered sample count
        exceeds this, the oldest arrays are dropped from the left — but the most
        recent array is always retained whole, so a single submit larger than
        the cap is kept in full (the effective cap is
        ``max(max_buffer_samples, one harvest)``).  Default is ``20_000``.
    seed : int or None, optional
        Reserved for parity with asynchronous executors; unused by the inline
        implementation (the clone carries the template flow's own seed).
        Default is ``None``.
    """

    def __init__(
        self,
        flow,
        fit_kwargs: dict | None = None,
        *,
        train_every: int = 1,
        min_train_samples: int = 1,
        max_buffer_samples: int = 20_000,
        seed=None,
    ):
        self._flow = FlowSpec.from_flow(flow).build()
        self._fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        # refit_data_transform and verbose are controlled by the executor (the
        # transform is frozen; verbosity is forced off).  Reject them in
        # fit_kwargs so a caller's collision fails fast rather than being
        # silently overridden at fit time.
        _reserved = {"refit_data_transform", "verbose"} & self._fit_kwargs.keys()
        if _reserved:
            raise ValueError(
                f"fit_kwargs may not contain {sorted(_reserved)}: these are "
                "controlled by the executor (the data transform is frozen and "
                "verbosity is forced off).  Remove them from fit_kwargs."
            )
        self._train_every = int(train_every)
        self._min_train_samples = int(min_train_samples)
        self._max_buffer_samples = int(max_buffer_samples)
        self._seed = seed

        # Per-condition ring buffers: condition id -> deque of (N_i, dims) arrays.
        self._buffers: dict[int, deque] = {}
        self._accepted_submits = 0
        self._trained_version = 0
        self._latest: tuple[int, dict] | None = None
        self._seen_version = 0
        self._error: BaseException | None = None
        self._shutdown = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self, condition: int) -> None:
        """Drop oldest arrays for ``condition`` until its buffer fits the cap."""
        buf = self._buffers[condition]
        total = sum(len(a) for a in buf)
        while total > self._max_buffer_samples and len(buf) > 1:
            removed = buf.popleft()
            total -= len(removed)

    def _buffered_total(self) -> int:
        """Total buffered samples summed over all conditions."""
        return sum(len(a) for buf in self._buffers.values() for a in buf)

    def _assemble(self) -> dict:
        """Concatenate per-condition buffers into ``{condition: (N, dims)}``."""
        return {
            cond: np.concatenate(list(buf), axis=0)
            for cond, buf in self._buffers.items()
            if len(buf) > 0
        }

    # ------------------------------------------------------------------
    # TrainerExecutor interface
    # ------------------------------------------------------------------

    def submit(self, samples_by_condition: dict) -> bool:
        """Append samples and, every ``train_every``-th accepted call, fit.

        Zero-length arrays are dropped before buffering; a submit that carries
        no rows at all (an empty dict, or only empty arrays) returns ``False``
        WITHOUT counting as an accepted submit and without bumping any version.
        Empty harvests — e.g. a sampling step with no active leaves — are a
        normal condition, not an error, and must not poison the executor.

        Returns ``False`` (drops the batch) after :meth:`shutdown`.  Otherwise
        returns ``True`` for any submit that buffered at least one row — the
        inline executor never coalesces or drops a non-empty live batch.

        Raises
        ------
        TrainerError
            If a previous fit failed (deferred surfacing; see class docstring).
        """
        if self._error is not None:
            raise TrainerError(
                "InlineExecutor training failed on a previous submit: "
                f"{self._error!r}"
            ) from self._error

        if self._shutdown:
            return False

        buffered_any = False
        for cond, arr in samples_by_condition.items():
            cond = int(cond)
            arr = np.asarray(arr)
            arr = arr.reshape(-1, self._flow.dims)
            if len(arr) == 0:
                # Drop empty conditions: a harvest with no rows is a normal
                # sampling condition, not data to train on.
                continue
            if cond not in self._buffers:
                self._buffers[cond] = deque()
            self._buffers[cond].append(arr)
            self._trim(cond)
            buffered_any = True

        # An empty harvest is not an accepted submit: don't count it, don't
        # train on it, and return False so callers can tell it was a no-op.
        if not buffered_any:
            return False

        self._accepted_submits += 1

        # Train on every train_every-th accepted submit once enough is buffered.
        # The _buffered_total() > 0 guard is independent of min_train_samples so
        # we never hand fit() an empty buffer (an empty fit is an error, not a
        # training step).
        if (
            self._accepted_submits % self._train_every == 0
            and self._buffered_total() > 0
            and self._buffered_total() >= self._min_train_samples
        ):
            try:
                self._flow.fit(
                    self._assemble(),
                    refit_data_transform=False,
                    verbose=False,
                    **self._fit_kwargs,
                )
            except BaseException as exc:  # noqa: BLE001 — record and defer.
                # Record and surface on the next poll/submit, mirroring a
                # process executor whose failure is only seen when next polled.
                self._error = exc
            else:
                self._trained_version += 1
                # Snapshot defensively: get_weights() may return references to the
                # clone's live (torch) parameters.  Deep-copy so the stashed
                # snapshot can never be aliased to training state — matching the
                # natural semantics of a process executor, whose returned dict is
                # always a fresh deserialized copy.
                self._latest = (
                    self._trained_version,
                    copy.deepcopy(self._flow.get_weights()),
                )

        return True

    def latest_weights(self):
        """Return the newest stashed ``(version, weights)`` (or ``None``).

        Updates :attr:`version` to the returned version.

        Raises
        ------
        TrainerError
            If a fit has failed.
        """
        if self._error is not None:
            raise TrainerError(
                f"InlineExecutor training failed: {self._error!r}"
            ) from self._error

        if self._latest is None:
            return None
        version, weights = self._latest
        self._seen_version = version
        return version, weights

    @property
    def version(self) -> int:
        """Latest version observed via :meth:`latest_weights` (``0`` = none yet)."""
        return self._seen_version

    def shutdown(self, timeout: float = 10.0) -> None:
        """Mark the executor shut down (idempotent).

        After shutdown, :meth:`submit` returns ``False`` and drops its batch.
        Synchronous, so ``timeout`` is unused.
        """
        self._shutdown = True
