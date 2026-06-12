# tests/test_flow_move.py
"""Tests for FlowMove and IndependentProposalMove.

Test layout
-----------
- ``test_independent_proposal_factor_identity`` — numpy-only (no torch).
- ``test_flow_move_factor_identity`` — requires torch; ``importorskip``'d.
- ``test_flow_move_multileaf_accumulates_factors`` — requires torch.
- ``test_flow_move_sampler_smoke`` — requires torch; 50-step EnsembleSampler smoke test.
"""
from __future__ import annotations

import numpy as np
import pytest

from eryn.moves import FlowMove, IndependentProposalMove


# ---------------------------------------------------------------------------
# Helpers shared across flow tests
# ---------------------------------------------------------------------------

def _make_flow(seed: int = 0):
    """Build a small ZukoFlow with WhiteningTransform + OneHotLeafConditioning.

    Returns the fitted (but not trained) flow.  Weights are random — that is
    fine for factor-identity tests; only the smoke test needs some acceptance.
    """
    torch = pytest.importorskip("torch")
    from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], size=2000
    ).astype(np.float64)

    cond = OneHotLeafConditioning(nleaves_max=1)
    wt = WhiteningTransform(ndim=2)
    wt.fit({0: samples})

    flow = ZukoFlow(
        dims=2,
        device="cpu",
        data_transform=wt,
        conditioning=cond,
        seed=seed,
        flow_class="NSF",
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    return flow


# ---------------------------------------------------------------------------
# IndependentProposalMove — factor identity (numpy only, no torch required)
# ---------------------------------------------------------------------------

class _GaussianDist:
    """Simple isotropic Gaussian proposal for testing (no torch, no scipy)."""

    def __init__(self, ndim: int, seed: int = 0):
        self.ndim = ndim
        self._rng = np.random.default_rng(seed)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # log N(0, I): -0.5 * sum(x**2) - ndim/2 * log(2pi)
        return -0.5 * np.sum(x ** 2, axis=-1) - 0.5 * self.ndim * np.log(2 * np.pi)

    def rvs(self, size: int) -> np.ndarray:
        return self._rng.standard_normal((size, self.ndim))


def test_independent_proposal_factor_identity():
    """factors == log q(old) - log q(new) for IndependentProposalMove (numpy only)."""
    ndim = 3
    ntemps, nwalkers, nleaves = 2, 8, 1
    rng = np.random.default_rng(42)

    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}
    dist = _GaussianDist(ndim, seed=7)
    move = IndependentProposalMove(dist, branch_name="x")

    q, factors = move.get_proposal(branches, rng)

    assert q["x"].shape == (ntemps, nwalkers, nleaves, ndim)
    assert factors.shape == (ntemps, nwalkers)

    # Recompute expected factors independently
    old = coords.reshape(-1, ndim)
    new = q["x"].reshape(-1, ndim)
    logq_old = dist.logpdf(old).reshape(ntemps, nwalkers)
    logq_new = dist.logpdf(new).reshape(ntemps, nwalkers)
    expected = logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-10,
                               err_msg="IndependentProposalMove factors != log q(old) - log q(new)")


def test_independent_proposal_multileaf_accumulates_factors():
    """Multi-leaf: factors sum per-leaf contributions (np.add.at, not last-write-wins)."""
    ndim = 3
    ntemps, nwalkers, nleaves = 1, 6, 2
    rng = np.random.default_rng(99)

    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}
    dist = _GaussianDist(ndim, seed=5)
    move = IndependentProposalMove(dist, branch_name="x")

    q, factors = move.get_proposal(branches, rng)

    # Recompute expected by summing over leaves
    expected = np.zeros((ntemps, nwalkers))
    for leaf in range(nleaves):
        old_l = coords[:, :, leaf, :].reshape(-1, ndim)
        new_l = q["x"][:, :, leaf, :].reshape(-1, ndim)
        expected += (dist.logpdf(old_l) - dist.logpdf(new_l)).reshape(ntemps, nwalkers)

    np.testing.assert_allclose(factors, expected, atol=1e-10,
                               err_msg="Multi-leaf factors not summed correctly")


def test_independent_proposal_non_target_branch_unchanged():
    """Branches other than branch_name are copied unchanged."""
    ndim = 2
    rng = np.random.default_rng(11)
    coords_x = rng.standard_normal((1, 4, 1, ndim))
    coords_y = rng.standard_normal((1, 4, 1, ndim))
    branches = {"x": coords_x, "y": coords_y}
    dist = _GaussianDist(ndim, seed=3)
    move = IndependentProposalMove(dist, branch_name="x")

    q, _ = move.get_proposal(branches, rng)
    np.testing.assert_array_equal(q["y"], coords_y,
                                  err_msg="Non-target branch 'y' was mutated")


# ---------------------------------------------------------------------------
# FlowMove — factor identity (requires torch)
# ---------------------------------------------------------------------------

def test_flow_move_factor_identity():
    """factors == log q(old) - log q(new) for FlowMove (recomputed via flow.log_prob)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=0)

    ntemps, nwalkers, nleaves, ndim = 1, 8, 1, 2
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    move = FlowMove(flow, branch_name="x")
    move.active_condition = 0

    q, factors = move.get_proposal(branches, rng)

    assert q["x"].shape == (ntemps, nwalkers, nleaves, ndim)
    assert factors.shape == (ntemps, nwalkers)

    # Recompute expected factors via flow.log_prob independently
    old = coords.reshape(-1, ndim)
    new = q["x"].reshape(-1, ndim)
    logq_old = flow.log_prob(old, context=0).reshape(ntemps, nwalkers)
    logq_new = flow.log_prob(new, context=0).reshape(ntemps, nwalkers)
    expected = logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-5,
                               err_msg="FlowMove factors != log q(old) - log q(new)")


def test_flow_move_multileaf_accumulates_factors():
    """Regression: multi-leaf factors must SUM per-leaf contributions (np.add.at)."""
    torch = pytest.importorskip("torch")
    flow = _make_flow(seed=2)

    ntemps, nwalkers, nleaves, ndim = 1, 6, 2, 2
    rng = np.random.default_rng(3)
    coords = rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
    branches = {"x": coords}

    move = FlowMove(flow, branch_name="x")
    move.active_condition = 0

    q, factors = move.get_proposal(branches, rng)

    # Recompute by summing per-leaf contributions
    expected = np.zeros((ntemps, nwalkers))
    for leaf in range(nleaves):
        old_l = coords[:, :, leaf, :].reshape(-1, ndim)
        new_l = q["x"][:, :, leaf, :].reshape(-1, ndim)
        logq_old = flow.log_prob(old_l, context=0).reshape(ntemps, nwalkers)
        logq_new = flow.log_prob(new_l, context=0).reshape(ntemps, nwalkers)
        expected += logq_old - logq_new

    np.testing.assert_allclose(factors, expected, atol=1e-5,
                               err_msg="Multi-leaf FlowMove factors not summed correctly")


# ---------------------------------------------------------------------------
# FlowMove sampler smoke test (requires torch)
# ---------------------------------------------------------------------------

def _log_like_gauss_vectorized(x):
    """Vectorized 2-D standard-normal log-likelihood (shape (N, ndim) -> (N,))."""
    x = np.atleast_2d(x)
    return -0.5 * np.sum(x ** 2, axis=-1)


def test_flow_move_sampler_smoke():
    """50-step EnsembleSampler smoke test with FlowMove on a 2-D Gaussian target.

    The flow is frozen (not trained); acceptance fraction > 0 just means the
    move is wired correctly — detailed balance is validated by the factor-identity
    test above.
    """
    torch = pytest.importorskip("torch")

    from eryn.ensemble import EnsembleSampler
    from eryn.prior import ProbDistContainer, uniform_dist
    from eryn.state import State

    ndim = 2
    ntemps = 1
    nwalkers = 16

    flow = _make_flow(seed=0)
    move = FlowMove(flow, branch_name="x")
    move.active_condition = 0

    priors = {
        "x": ProbDistContainer(
            {0: uniform_dist(-10.0, 10.0), 1: uniform_dist(-10.0, 10.0)}
        )
    }

    sampler = EnsembleSampler(
        nwalkers,
        {"x": ndim},
        _log_like_gauss_vectorized,
        priors,
        tempering_kwargs=dict(ntemps=ntemps),
        vectorize=True,
        moves=[move],
        branch_names=["x"],
    )

    rng = np.random.default_rng(42)
    start_coords = rng.standard_normal((ntemps, nwalkers, 1, ndim))
    start = State({"x": start_coords})

    sampler.run_mcmc(start, 50, burn=0, progress=False)

    chain = sampler.get_chain()["x"]
    # Shape: (nsteps, ntemps, nwalkers, nleaves_max, ndim)
    assert chain.shape[0] == 50, f"Unexpected chain length: {chain.shape[0]}"
    assert np.all(np.isfinite(chain)), "Chain contains non-finite values"

    # With ntemps=1 the sampler reports acceptance per move; at least one
    # proposal must have been accepted over 50 steps with 16 walkers.
    acceptance = sampler.acceptance_fraction
    # acceptance_fraction is (ntemps, nwalkers); check at least one acceptance
    assert np.any(acceptance > 0), (
        f"No proposals accepted in 50 steps (acceptance={acceptance}); "
        "the move is likely mis-wired."
    )
