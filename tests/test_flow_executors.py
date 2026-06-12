# tests/test_flow_executors.py
"""Tests for the TrainerExecutor seam: FlowSpec and InlineExecutor.

Most tests use a lightweight numpy-only fake Flow defined here, so they are
fast and torch-free.  One ZukoFlow-backed integration test (``importorskip``'d)
verifies that ``FlowSpec.from_flow(...).build()`` reproduces ``log_prob``.
"""
from __future__ import annotations

import numpy as np
import pytest

from eryn.flows import FlowSpec, InlineExecutor, TrainerError
from eryn.flows.base import Flow, FlowHistory
from eryn.flows.transforms import DataTransform, IdentityTransform


# ---------------------------------------------------------------------------
# Lightweight numpy-only fake Flow + transforms
# ---------------------------------------------------------------------------

class _FittableTransform(DataTransform):
    """Identity-like transform with a togglable ``is_fitted`` flag (picklable)."""

    def __init__(self, fitted: bool = True):
        self._fitted = bool(fitted)

    def fit(self, samples) -> None:
        self._fitted = True

    def forward(self, x, condition: int = 0):
        return np.asarray(x)

    def inverse(self, z, condition: int = 0):
        return np.asarray(z)

    def log_abs_det_jacobian(self, x, z, condition: int = 0):
        return np.zeros(len(x))

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class FakeFlow(Flow):
    """Numpy-only Flow for executor tests.

    ``fit`` bumps a counter, records how many samples it received, and stores a
    ``scale`` weight derived from the data so that ``get_weights`` /
    ``set_weights`` round-trips carry meaning (log_prob depends on ``scale``).
    """

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None,
                 scale: float = 1.0):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning)
        self.scale = float(scale)
        self.fit_count = 0
        self.last_fit_n = 0
        self.last_fit_kwargs: dict = {}

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum((x / self.scale) ** 2, axis=-1)

    def sample(self, n: int, context=None):
        return self.scale * np.random.randn(n, self.dims)

    def sample_and_log_prob(self, n: int, context=None):
        x = self.sample(n, context=context)
        return x, self.log_prob(x, context=context)

    def fit(self, samples, **kwargs):
        self.fit_count += 1
        self.last_fit_kwargs = kwargs
        if isinstance(samples, dict):
            n = sum(len(v) for v in samples.values())
            allx = np.concatenate([np.asarray(v) for v in samples.values()], axis=0)
        else:
            n = len(samples)
            allx = np.asarray(samples)
        self.last_fit_n = n
        # Derive a meaningful weight from the data so set_weights matters.
        self.scale = float(np.std(allx)) if allx.size else self.scale
        return FlowHistory()

    def get_weights(self):
        return {"scale": np.array(self.scale, dtype=np.float64)}

    def set_weights(self, weights):
        self.scale = float(np.asarray(weights["scale"]))

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


class _RaisingFlow(FakeFlow):
    """FakeFlow whose ``fit`` always raises (to exercise failure surfacing)."""

    def fit(self, samples, **kwargs):
        raise RuntimeError("boom in fit")


class _UnpicklableFlow(FakeFlow):
    """FakeFlow carrying a lambda in its config (unpicklable)."""

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None,
                 bad=None, scale: float = 1.0):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning, scale=scale)
        self.bad = bad


def _make_fake(dims: int = 2, scale: float = 1.0) -> FakeFlow:
    return FakeFlow(dims=dims, device="cpu", data_transform=_FittableTransform(True),
                    scale=scale)


# ---------------------------------------------------------------------------
# FlowSpec
# ---------------------------------------------------------------------------

def test_flowspec_from_flow_unfitted_transform_raises():
    flow = FakeFlow(dims=2, data_transform=_FittableTransform(fitted=False))
    with pytest.raises(ValueError, match="is_fitted"):
        FlowSpec.from_flow(flow)


def test_flowspec_from_flow_unpicklable_config_raises():
    flow = _UnpicklableFlow(dims=2, data_transform=_FittableTransform(True),
                            bad=lambda z: z)
    with pytest.raises(TypeError, match="picklable"):
        FlowSpec.from_flow(flow)


def test_flowspec_overrides_device():
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow, worker_device="cuda:0")
    assert spec.config["device"] == "cuda:0"
    # Default is cpu.
    spec_cpu = FlowSpec.from_flow(flow)
    assert spec_cpu.config["device"] == "cpu"


def test_flowspec_deepcopies_config():
    """Mutating the live flow after snapshot must not change the spec."""
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow)
    flow.config_dict()["dims"] = 999  # no-op, but mutate the transform too
    flow.data_transform._fitted = False
    assert spec.config["dims"] == 2


def test_flowspec_build_reconstructs():
    """build() reproduces config + weights of the snapshotted flow."""
    flow = _make_fake(scale=3.5)
    spec = FlowSpec.from_flow(flow)
    rebuilt = spec.build()
    assert isinstance(rebuilt, FakeFlow)
    assert rebuilt.dims == flow.dims
    # Weights (scale) round-trip.
    assert rebuilt.scale == pytest.approx(3.5)
    x = np.random.randn(5, 2)
    np.testing.assert_allclose(rebuilt.log_prob(x), flow.log_prob(x))


def test_flowspec_is_picklable():
    import pickle
    flow = _make_fake()
    spec = FlowSpec.from_flow(flow)
    spec2 = pickle.loads(pickle.dumps(spec))
    assert spec2.config["dims"] == spec.config["dims"]


# ---------------------------------------------------------------------------
# InlineExecutor — buffering / versioning
# ---------------------------------------------------------------------------

def test_inline_clones_flow_not_caller():
    """The executor trains a CLONE; the caller's flow is never mutated."""
    flow = _make_fake(scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.full((10, 2), 5.0)})  # would change scale on the clone
    assert flow.fit_count == 0  # caller's flow untouched
    assert flow.scale == 1.0


def test_inline_version_zero_before_min_train_samples():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=100)
    # Submit fewer than min_train_samples; no training should run.
    ex.submit({0: np.random.randn(10, 2)})
    assert ex.latest_weights() is None
    assert ex.version == 0


def test_inline_version_advances_after_enough_submits():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=20)
    ex.submit({0: np.random.randn(10, 2)})  # 10 buffered < 20: no train
    assert ex.latest_weights() is None
    ex.submit({0: np.random.randn(15, 2)})  # 25 buffered >= 20: train
    lw = ex.latest_weights()
    assert lw is not None
    version, weights = lw
    assert version == 1
    assert ex.version == 1


def test_inline_train_every_honored():
    """train_every=2 → first accepted submit does not train, second does."""
    flow = _make_fake()
    ex = InlineExecutor(flow, train_every=2, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})
    assert ex.latest_weights() is None  # 1st submit: no train
    ex.submit({0: np.random.randn(5, 2)})
    lw = ex.latest_weights()
    assert lw is not None and lw[0] == 1  # 2nd submit: trained


def test_inline_latest_weights_returns_newest():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})
    v1, _ = ex.latest_weights()
    ex.submit({0: np.random.randn(5, 2)})
    v2, _ = ex.latest_weights()
    assert v2 == v1 + 1
    assert ex.version == v2


def test_inline_weights_applied_to_sibling_change_outputs():
    """Weights from latest_weights, applied to a sibling flow, change its outputs."""
    flow = _make_fake(scale=1.0)
    ex = InlineExecutor(flow, min_train_samples=0)
    # Submit data with a clearly different spread so the trained scale differs.
    ex.submit({0: np.random.randn(500, 2) * 7.0})
    _, weights = ex.latest_weights()

    sibling = _make_fake(scale=1.0)
    x = np.full((4, 2), 2.0)
    before = sibling.log_prob(x).copy()
    sibling.set_weights(weights)
    after = sibling.log_prob(x)
    assert not np.allclose(before, after)


def test_inline_buffer_cap_trims():
    """Total samples handed to fit never exceed the per-condition cap."""
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0, max_buffer_samples=50)
    # Submit way more than the cap across several batches.
    for _ in range(10):
        ex.submit({0: np.random.randn(20, 2)})
    # The clone records the last fit's sample count; must be <= cap.
    assert ex._flow.last_fit_n <= 50


def test_inline_buffer_cap_multi_condition():
    """Cap is per-condition: two conditions can each fill to the cap."""
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0, max_buffer_samples=30)
    for _ in range(5):
        ex.submit({0: np.random.randn(20, 2), 1: np.random.randn(20, 2)})
    # Each condition trimmed to <= 30; total <= 60.
    assert ex._flow.last_fit_n <= 60


# ---------------------------------------------------------------------------
# InlineExecutor — failure surfacing
# ---------------------------------------------------------------------------

def test_inline_fit_failure_surfaces_on_latest_weights():
    flow = _RaisingFlow(dims=2, data_transform=_FittableTransform(True))
    ex = InlineExecutor(flow, min_train_samples=0)
    # submit that triggers the failing fit does NOT raise (deferred surfacing).
    ex.submit({0: np.random.randn(5, 2)})
    with pytest.raises(TrainerError, match="boom in fit"):
        ex.latest_weights()


def test_inline_fit_failure_surfaces_on_next_submit():
    flow = _RaisingFlow(dims=2, data_transform=_FittableTransform(True))
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.submit({0: np.random.randn(5, 2)})  # records failure
    with pytest.raises(TrainerError):
        ex.submit({0: np.random.randn(5, 2)})  # next submit re-raises


# ---------------------------------------------------------------------------
# InlineExecutor — shutdown / context manager
# ---------------------------------------------------------------------------

def test_inline_shutdown_idempotent_and_blocks_submit():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    ex.shutdown()
    ex.shutdown()  # idempotent
    assert ex.submit({0: np.random.randn(5, 2)}) is False


def test_inline_submit_returns_true_when_live():
    flow = _make_fake()
    ex = InlineExecutor(flow, min_train_samples=0)
    assert ex.submit({0: np.random.randn(5, 2)}) is True


def test_inline_context_manager_calls_shutdown():
    flow = _make_fake()
    with InlineExecutor(flow, min_train_samples=0) as ex:
        assert ex.submit({0: np.random.randn(5, 2)}) is True
    assert ex.submit({0: np.random.randn(5, 2)}) is False


# ---------------------------------------------------------------------------
# ZukoFlow integration — FlowSpec.from_flow(...).build() reproduces log_prob
# ---------------------------------------------------------------------------

def _make_zuko_flow():
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 3))

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: samples})

    return ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=0,
        flow_class="NSF",
        transforms=2,
        hidden_features=(32, 32),
        bins=4,
    )


def test_flowspec_zuko_build_reproduces_log_prob():
    pytest.importorskip("torch")
    flow = _make_zuko_flow()
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    spec = FlowSpec.from_flow(flow)
    rebuilt = spec.build()

    lp1 = flow.log_prob(x, context=0)
    lp2 = rebuilt.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="FlowSpec build changed log_prob")


def test_inline_zuko_trains_and_serves_weights():
    """InlineExecutor with a real ZukoFlow trains and serves usable weights."""
    pytest.importorskip("torch")
    flow = _make_zuko_flow()
    rng = np.random.default_rng(1)
    data = rng.standard_normal((300, 3))

    ex = InlineExecutor(flow, fit_kwargs=dict(n_epochs=2), min_train_samples=0)
    accepted = ex.submit({0: data})
    assert accepted is True
    lw = ex.latest_weights()
    assert lw is not None
    version, weights = lw
    assert version == 1

    # Weights are applicable to a sibling flow.
    sibling = _make_zuko_flow()
    sibling.set_weights(weights)
    x = rng.standard_normal((16, 3))
    lp = sibling.log_prob(x, context=0)
    assert lp.shape == (16,)
    assert np.all(np.isfinite(lp))
