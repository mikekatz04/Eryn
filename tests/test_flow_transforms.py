# tests/test_flow_transforms.py
"""Tests for the torch-backed WhiteningTransform and its helper transforms.

Guards
------
``torch`` and ``zuko`` are optional extras; the entire module is skipped when
either is absent.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from eryn.flows.torch.transforms import WhiteningTransform  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_samples(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Return (n, 3) float64 samples: two correlated Gaussians + one periodic angle."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    g = rng.multivariate_normal([2.0, -1.0], cov, size=n)
    ang = (rng.normal(1.0, 0.3, size=n)) % (2 * np.pi)  # bulk away from 0/2pi wrap
    return np.column_stack([g, ang]).astype(np.float64)


def _fitted_wt() -> tuple[WhiteningTransform, np.ndarray]:
    """Return a fitted WhiteningTransform and the training samples."""
    samples = _make_samples()
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    wt.fit({0: samples})
    return wt, samples


# ---------------------------------------------------------------------------
# is_fitted
# ---------------------------------------------------------------------------

def test_is_fitted_false_before_fit():
    """is_fitted is False before fit() is called."""
    wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 2 * np.pi)})
    assert not wt.is_fitted


def test_is_fitted_true_after_fit():
    """is_fitted is True after fit() is called."""
    wt, _ = _fitted_wt()
    assert wt.is_fitted


# ---------------------------------------------------------------------------
# forward / inverse round-trip
# ---------------------------------------------------------------------------

def test_forward_inverse_roundtrip():
    """forward then inverse reproduces original samples (periodic dim compared mod period)."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples, condition=0)
    x_back = wt.inverse(z, condition=0)
    # non-periodic dims
    assert np.allclose(x_back[:, :2], samples[:, :2], atol=1e-4)
    # periodic dim — compare modulo period
    d = np.abs((x_back[:, 2] - samples[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    assert np.max(d) < 1e-4


def test_forward_returns_float32_tensor():
    """forward() must return a float32 torch.Tensor (flow-input contract)."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples[:10], condition=0)
    assert isinstance(z, torch.Tensor)
    assert z.dtype == torch.float32


def test_forward_output_shape():
    """forward() output shape matches input shape."""
    wt, samples = _fitted_wt()
    z = wt.forward(samples[:17], condition=0)
    assert z.shape == (17, 3)


# ---------------------------------------------------------------------------
# log_abs_det_jacobian shape — regression test
# ---------------------------------------------------------------------------

def test_log_abs_det_jacobian_shape_with_periodic():
    """Regression: with a periodic dim the log-det must stay (N,), not broadcast to (N, N).

    The PartialTransform wrapping the circular shift returned a per-dim (N, k) term that
    ComposeTransform summed against the (N,) terms, silently producing an (N, N) Jacobian.
    """
    wt, samples = _fitted_wt()
    x = samples[:7]
    z = wt.forward(x, condition=0)
    ladj = wt.log_abs_det_jacobian(torch.tensor(x), z, condition=0)
    assert tuple(ladj.shape) == (7,), f"expected (7,), got {tuple(ladj.shape)}"


# ---------------------------------------------------------------------------
# log_abs_det_jacobian correctness — float64 finite-difference gate
# ---------------------------------------------------------------------------

def test_log_abs_det_jacobian_matches_numerical():
    """Analytic log|det J| must match a float64 finite-difference estimate.

    This is the headline correctness gate: catches a wrong/missing periodic-angle
    Jacobian — the silent-bias hole described in the spec.
    """
    wt, samples = _fitted_wt()
    pts = samples[:5].copy()  # bulk points, away from wrap
    eps = 1e-5

    # Compute latents in float64 directly via the internal transform.
    # ``wt.forward`` casts to float32, which would wreck the finite-difference
    # derivative below via catastrophic cancellation in (zp - zm) / 2eps.
    # The analytic log|det| is float64 and exact.
    def fwd64(x):
        return wt.transforms[0](torch.as_tensor(x, dtype=torch.float64)).numpy()

    z0 = fwd64(pts)
    analytic = wt.log_abs_det_jacobian(
        torch.as_tensor(pts), torch.as_tensor(z0), condition=0
    ).numpy()

    for k in range(pts.shape[0]):
        J = np.zeros((3, 3))
        for j in range(3):
            xp = pts[k].copy()
            xm = pts[k].copy()
            xp[j] += eps
            xm[j] -= eps
            zp = fwd64(xp[None])[0]
            zm = fwd64(xm[None])[0]
            J[:, j] = (zp - zm) / (2 * eps)
        numerical = np.linalg.slogdet(J)[1]
        assert np.isclose(analytic[k], numerical, atol=1e-3), (
            f"point {k}: analytic {analytic[k]} vs numerical {numerical}"
        )


# ---------------------------------------------------------------------------
# fit with dict of conditions
# ---------------------------------------------------------------------------

def test_fit_dict_of_conditions():
    """fit() accepts a dict[int, ndarray] and builds per-condition transforms."""
    rng = np.random.default_rng(42)
    s0 = rng.standard_normal((500, 3))
    s1 = rng.standard_normal((500, 3)) + 5.0
    wt = WhiteningTransform(ndim=3)
    wt.fit({0: s0, 1: s1})
    assert wt.is_fitted
    assert 0 in wt.transforms
    assert 1 in wt.transforms
    z0 = wt.forward(s0[:10], condition=0)
    z1 = wt.forward(s1[:10], condition=1)
    assert z0.shape == (10, 3)
    assert z1.shape == (10, 3)


def test_fit_array_uses_condition_zero():
    """fit() with a plain array stores the transform under condition 0."""
    wt = WhiteningTransform(ndim=3)
    samples = _make_samples(n=200)
    wt.fit(samples)
    assert 0 in wt.transforms


# ---------------------------------------------------------------------------
# pickle round-trip
# ---------------------------------------------------------------------------

def test_pickle_roundtrip():
    """WhiteningTransform must survive pickle.loads(pickle.dumps(wt)) identically.

    Load-bearing for Task-9 multiprocessing spawn: the worker process receives
    the transform via pickle and must produce bit-identical forward / log-det.
    """
    wt, samples = _fitted_wt()
    pts = samples[:10]
    z_before = wt.forward(pts, condition=0)
    ladj_before = wt.log_abs_det_jacobian(
        torch.as_tensor(pts), z_before, condition=0
    )

    wt2 = pickle.loads(pickle.dumps(wt))

    z_after = wt2.forward(pts, condition=0)
    ladj_after = wt2.log_abs_det_jacobian(
        torch.as_tensor(pts), z_after, condition=0
    )

    assert torch.equal(z_before, z_after), "forward output differs after pickle round-trip"
    assert torch.allclose(ladj_before, ladj_after), (
        "log_abs_det_jacobian differs after pickle round-trip"
    )


# ---------------------------------------------------------------------------
# periodic_indices / non_periodic_indices properties
# ---------------------------------------------------------------------------

def test_periodic_and_non_periodic_indices():
    """periodic_indices and non_periodic_indices partition range(ndim)."""
    wt = WhiteningTransform(ndim=5, periodic={1: (0.0, 1.0), 3: (0.0, 2 * np.pi)})
    assert sorted(wt.periodic_indices) == [1, 3]
    assert sorted(wt.non_periodic_indices) == [0, 2, 4]


# ---------------------------------------------------------------------------
# unknown condition raises ValueError
# ---------------------------------------------------------------------------

def test_forward_unknown_condition_raises():
    """forward() with an unknown condition raises ValueError."""
    wt, _ = _fitted_wt()
    with pytest.raises(ValueError, match="condition"):
        wt.forward(_make_samples(n=5), condition=99)
