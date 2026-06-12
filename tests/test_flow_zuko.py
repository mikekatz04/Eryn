# tests/test_flow_zuko.py
"""Tests for ZukoFlow — the zuko-backed conditional normalizing flow.

These tests port the three core tests from LISAanalysistools/tests/test_ml_flow.py
(renamed and adapted to the eryn.flows API) and add additional coverage for the
unified context contract, config round-trips, and h5 save/load.

Requires torch and zuko (``pip install eryn[flow]``).
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os

torch = pytest.importorskip("torch")
zuko = pytest.importorskip("zuko")

from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning, get_flow_wrapper


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_flow(seed: int = 0) -> tuple:
    """Build a small ZukoFlow with WhiteningTransform and OneHotLeafConditioning.

    Returns (flow, samples) where samples is a (3000, 3) float64 array used to
    fit the WhiteningTransform (flow itself is not trained — weights are random).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=3000),
        (rng.normal(2.0, 0.4, size=3000)) % (2 * np.pi),
    ]).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    wt.fit({0: samples})

    flow = ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow, samples


def _make_unconditional_flow(seed: int = 42) -> "ZukoFlow":
    """Build a small unconditional ZukoFlow (no conditioning, no data_transform)."""
    torch.manual_seed(seed)
    flow = ZukoFlow(
        dims=3,
        device="cpu",
        data_transform=None,
        conditioning=None,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow


# ---------------------------------------------------------------------------
# Ported from test_ml_flow.py — renamed FlowModel → ZukoFlow, condition→context,
# normalizer→data_transform, OneHotLeafConditioning kept.
# ---------------------------------------------------------------------------

def test_sample_shape_and_logprob_consistency():
    """sample_and_log_prob shapes and log_prob(x) == logq at sampled points (atol 1e-4)."""
    flow, _ = _make_flow()
    x, logq = flow.sample_and_log_prob(128, context=0)
    assert x.shape == (128, 3), f"expected (128, 3), got {x.shape}"
    assert logq.shape == (128,), f"expected (128,), got {logq.shape}"
    # Re-evaluate at the same points: must match sampling-time log_prob
    logq2 = flow.log_prob(x, context=0)
    np.testing.assert_allclose(logq, logq2, atol=1e-4,
                               err_msg="log_prob recomputed differs from sample_and_log_prob")


def test_logprob_is_finite_on_training_support():
    """log_prob returns finite values for all points in the training support."""
    flow, samples = _make_flow()
    lp = flow.log_prob(samples[:256], context=0)
    assert np.all(np.isfinite(lp)), f"Non-finite log-probs: {np.sum(~np.isfinite(lp))} entries"


def test_proposal_distribution_matches_flow():
    """FlowProposalDistribution.logpdf and rvs match ZukoFlow.log_prob/sample."""
    from eryn.flows import FlowProposalDistribution
    flow, _ = _make_flow()
    dist = FlowProposalDistribution(flow, condition=0)
    x = dist.rvs(64)
    assert x.shape == (64, 3)
    lp_dist = dist.logpdf(x)
    lp_flow = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_dist, lp_flow, atol=1e-5)


# ---------------------------------------------------------------------------
# Unconditional flow
# ---------------------------------------------------------------------------

def test_unconditional_log_prob_shape_and_finite():
    """Unconditional ZukoFlow (context=None) returns finite log_probs, correct shape."""
    flow = _make_unconditional_flow()
    x = np.random.default_rng(0).standard_normal((32, 3)).astype(np.float64)
    lp = flow.log_prob(x, context=None)
    assert lp.shape == (32,)
    assert np.all(np.isfinite(lp)), "Non-finite log_probs from unconditional flow"


def test_unconditional_sample_shape_and_finite():
    """Unconditional ZukoFlow sample returns correct shape and no NaNs."""
    flow = _make_unconditional_flow()
    x = flow.sample(64, context=None)
    assert x.shape == (64, 3)
    assert not np.any(np.isnan(x)), "NaNs in unconditional samples"


def test_unconditional_sample_and_log_prob_consistency():
    """Unconditional sample_and_log_prob logq matches log_prob(x) (atol 1e-4)."""
    flow = _make_unconditional_flow()
    x, logq = flow.sample_and_log_prob(64, context=None)
    logq2 = flow.log_prob(x, context=None)
    np.testing.assert_allclose(logq, logq2, atol=1e-4)


# ---------------------------------------------------------------------------
# Raw-array context
# ---------------------------------------------------------------------------

def test_raw_array_context_matches_int_context():
    """Passing conditioning.encode(0) directly gives same result as context=0 (atol 1e-5)."""
    flow, samples = _make_flow()
    raw_ctx = flow.conditioning.encode(0)  # numpy float32, shape (context_dim,)
    x = samples[:32]
    lp_int = flow.log_prob(x, context=0)
    lp_raw = flow.log_prob(x, context=raw_ctx)
    np.testing.assert_allclose(lp_int, lp_raw, atol=1e-5,
                               err_msg="int context and raw array context disagree")


def test_raw_array_context_sample_and_log_prob():
    """sample_and_log_prob with raw context vector returns valid outputs."""
    flow, _ = _make_flow()
    raw_ctx = flow.conditioning.encode(0)
    x, logq = flow.sample_and_log_prob(32, context=raw_ctx)
    assert x.shape == (32, 3)
    assert logq.shape == (32,)
    assert np.all(np.isfinite(logq))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_int_context_without_conditioning_raises():
    """Passing an int context to a flow with no conditioning raises ValueError."""
    flow = _make_unconditional_flow()
    with pytest.raises(ValueError, match="conditioning"):
        flow.log_prob(np.zeros((4, 3)), context=0)


def test_context_with_no_context_dim_raises():
    """Passing any non-None context to a flow built with context_dim=0 raises ValueError."""
    flow = _make_unconditional_flow()
    ctx_vec = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        flow.log_prob(np.zeros((4, 3)), context=ctx_vec)


# ---------------------------------------------------------------------------
# get_flow_wrapper
# ---------------------------------------------------------------------------

def test_get_flow_wrapper_returns_zukoflow_and_torch():
    """get_flow_wrapper('zuko') returns (ZukoFlow, torch)."""
    FlowCls, backend = get_flow_wrapper("zuko")
    assert FlowCls is ZukoFlow
    import torch as _torch
    assert backend is _torch


# ---------------------------------------------------------------------------
# get_weights / set_weights
# ---------------------------------------------------------------------------

def test_get_weights_cpu_and_detached():
    """get_weights() tensors are on CPU and detached; mutating them doesn't change the flow."""
    flow, _ = _make_flow()
    weights = flow.get_weights()
    assert all(isinstance(v, torch.Tensor) for v in weights.values())
    assert all(v.device.type == "cpu" for v in weights.values())

    # Mutating the returned dict must NOT change the live flow outputs
    x = np.zeros((4, 3), dtype=np.float64)
    lp_before = flow.log_prob(x, context=0)
    for v in weights.values():
        v.fill_(999.0)
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8,
                               err_msg="Mutating get_weights() output changed the flow")


def test_set_weights_restores_log_prob():
    """set_weights(get_weights()) is a no-op: log_prob is unchanged."""
    flow, _ = _make_flow()
    x = np.random.default_rng(7).standard_normal((16, 3))
    lp_before = flow.log_prob(x, context=0)
    flow.set_weights(flow.get_weights())
    lp_after = flow.log_prob(x, context=0)
    np.testing.assert_allclose(lp_before, lp_after, atol=1e-8)


# ---------------------------------------------------------------------------
# config_dict round-trip
# ---------------------------------------------------------------------------

def test_config_dict_round_trip_log_prob():
    """ZukoFlow(**flow.config_dict()) + set_weights reproduces log_prob (atol 1e-6)."""
    flow, _ = _make_flow()
    rng = np.random.default_rng(99)
    x = rng.standard_normal((64, 3))

    cfg = flow.config_dict()
    flow2 = ZukoFlow(**cfg)
    flow2.set_weights(flow.get_weights())

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-6,
                               err_msg="config_dict round-trip changed log_prob")


def test_config_dict_contains_flow_kwargs():
    """flow_kwargs like transforms, bins are flattened into config_dict() at the top level."""
    flow, _ = _make_flow()
    cfg = flow.config_dict()
    assert "transforms" in cfg, "transforms not found in config_dict()"
    assert "bins" in cfg, "bins not found in config_dict()"
    assert cfg["dims"] == 3


def test_config_dict_tuple_hidden_features_survives():
    """tuple hidden_features survives config round-trip (may come back as list; both ok)."""
    flow, _ = _make_flow()
    cfg = flow.config_dict()
    hf = cfg["hidden_features"]
    # tuple or list, both are fine — ZukoFlow accepts both
    assert tuple(hf) == (64, 64), f"hidden_features mangled: {hf}"


# ---------------------------------------------------------------------------
# HDF5 save / load round-trip
# ---------------------------------------------------------------------------

def test_h5_save_load_round_trip(tmp_path):
    """h5 save/load reproduces log_prob exactly and restores transform + conditioning."""
    flow, _ = _make_flow()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 3))

    h5_path = str(tmp_path / "flow.h5")
    flow.save(h5_path)
    flow2 = ZukoFlow.load(h5_path)

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.load(h5_path).log_prob(x, context=0)  # load again via cls method
    np.testing.assert_allclose(lp1, lp2, atol=1e-8,
                               err_msg="h5 save/load changed log_prob")

    # Conditioning must be restored
    assert flow2.conditioning is not None
    assert flow2.conditioning.context_dim == flow.conditioning.context_dim
    # data_transform must be restored and functional
    z = flow2.data_transform.forward(x, condition=0)
    assert z.shape == (64, 3)


def test_h5_save_load_with_h5py_handle(tmp_path):
    """save/load also accept an open h5py.File handle."""
    h5py = pytest.importorskip("h5py")
    flow, _ = _make_flow()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((16, 3))

    h5_path = str(tmp_path / "flow2.h5")
    with h5py.File(h5_path, "w") as f:
        flow.save(f, path="myflow")

    with h5py.File(h5_path, "r") as f:
        flow2 = ZukoFlow.load(f, path="myflow")

    lp1 = flow.log_prob(x, context=0)
    lp2 = flow2.log_prob(x, context=0)
    np.testing.assert_allclose(lp1, lp2, atol=1e-8)


# ---------------------------------------------------------------------------
# fit() stub
# ---------------------------------------------------------------------------

def test_fit_raises_not_implemented():
    """fit() raises NotImplementedError (Training loop lands in Task 4)."""
    flow, _ = _make_flow()
    with pytest.raises(NotImplementedError, match="fit"):
        flow.fit(np.zeros((10, 3)))
