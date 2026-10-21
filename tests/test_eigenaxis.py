#!/usr/bin/env python
# coding: utf-8
"""Tests for the eigen-axis / information-matrix proposal machinery.

Layers covered:

1. The module-level primitives in :mod:`eryn.moves.eigenaxis`
   (``prior_box_scales``, ``project_out_direction``, ``axis_prior_bounds``,
   ``eigen_axis_set``, ``draw_axis_step``) — pure linear algebra, batched
   over sources, array-module agnostic.
2. :class:`EigenAxisMove` as a unit: table lookup (shared / per-leaf /
   full-shape / branch-supplemental), the two draw modes, periodic
   wrapping, multi-leaf handling, and the symmetric ``factors == 0``
   contract.
3. :class:`EigenAxisMove` inside :class:`EnsembleSampler`: statistical
   recovery of a correlated Gaussian, and per-sampler isolation under the
   folded ``nsamplers`` axis.
"""

import unittest

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.moves import EigenAxisMove, StretchMove
from eryn.moves.eigenaxis import (
    axis_prior_bounds,
    draw_axis_step,
    eigen_axis_set,
    prior_box_scales,
    project_out_direction,
)
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import BranchSupplemental
from eryn.utils import PeriodicContainer

try:
    import cupy as cp

    HAS_CUPY = True
    try:
        cp.zeros(1)
    except Exception:
        HAS_CUPY = False
except ImportError:
    HAS_CUPY = False


def _random_spd(rng, n, ndim, scale=1.0):
    """Batch of symmetric positive-definite matrices ``(n, ndim, ndim)``."""
    a = rng.standard_normal((n, ndim, ndim))
    return scale * (a @ np.swapaxes(a, -1, -2)) + 0.5 * np.eye(ndim)


def _unit_rows(rng, n, ndim):
    t = rng.standard_normal((n, ndim))
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


class EigenAxisPrimitivesTest(unittest.TestCase):
    def test_prior_box_scales_widths_and_degenerate_columns(self):
        lo = np.array([0.0, -1.0, 3.0, 2.0, 0.0])
        hi = np.array([2.0, 1.0, 3.0, np.inf, -4.0])
        s = prior_box_scales(lo, hi)
        # plain widths
        np.testing.assert_allclose(s[:2], [2.0, 2.0])
        # zero-width, non-finite -> 1.0 (never divide by zero downstream)
        self.assertEqual(s[2], 1.0)
        self.assertEqual(s[3], 1.0)
        # reversed bounds still give a positive width
        self.assertEqual(s[4], 4.0)

    def test_project_out_direction_removes_component(self):
        rng = np.random.default_rng(3)
        info = _random_spd(rng, 4, 5)
        t = _unit_rows(rng, 4, 5)
        Fp = project_out_direction(info, t)
        # projected matrix annihilates t on both sides and stays symmetric
        Ft = np.einsum("nij,nj->ni", Fp, t)
        np.testing.assert_allclose(Ft, 0.0, atol=1e-12)
        np.testing.assert_allclose(Fp, np.swapaxes(Fp, -1, -2), atol=1e-13)

    def test_eigen_axis_set_plain_matches_eigh(self):
        rng = np.random.default_rng(5)
        info = _random_spd(rng, 3, 4)
        axes, sigmas = eigen_axis_set(info)
        evals, evecs = np.linalg.eigh(info)
        # no fiber, no ridge: the axes ARE the eigenvectors (up to sign)
        np.testing.assert_allclose(np.abs(axes), np.abs(evecs), atol=1e-12)
        # sigma_k = 1/sqrt(a_k^T F a_k) = 1/sqrt(eval_k), before clipping
        np.testing.assert_allclose(
            sigmas, np.minimum(1.0 / np.sqrt(evals), 1.0), atol=1e-12
        )

    def test_eigen_axis_set_sigma_max_clip(self):
        rng = np.random.default_rng(7)
        # genuinely soft curvature (no identity floor) so raw widths exceed
        # the clip
        a = rng.standard_normal((2, 3, 3))
        info = 1e-4 * (a @ np.swapaxes(a, -1, -2)) + 1e-10 * np.eye(3)
        _, sigmas = eigen_axis_set(info, sigma_max=2.5)
        self.assertLessEqual(sigmas.max(), 2.5 + 1e-14)
        # and with a large bound the raw curvature widths come through
        _, sigmas_free = eigen_axis_set(info, sigma_max=np.inf)
        self.assertGreater(sigmas_free.max(), 2.5)

    def test_eigen_axis_set_fiber_and_ridge_injection(self):
        rng = np.random.default_rng(11)
        n, ndim = 3, 5
        info = _random_spd(rng, n, ndim)
        t_fiber = _unit_rows(rng, n, ndim)
        ridge_raw = rng.standard_normal((n, ndim))

        axes, sigmas = eigen_axis_set(
            info, t_fiber=t_fiber, ridge_axis=ridge_raw, sigma_max=np.inf
        )

        # last column is the ridge, Gram-Schmidt'd against the fiber and
        # normalized
        expected = ridge_raw - t_fiber * (t_fiber * ridge_raw).sum(
            axis=-1, keepdims=True
        )
        expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(axes[:, :, -1], expected, atol=1e-12)

        # the other columns are eigenvectors of the fiber-projected matrix,
        # ordered so the fiber-aligned one landed last (and was overwritten):
        # every kept column must be near-orthogonal to the fiber... not
        # exactly, but the LAST pre-overwrite column had the LARGEST fiber
        # overlap, so the kept ones have smaller overlap than it did.
        Fp = project_out_direction(info, t_fiber)
        evals, evecs = np.linalg.eigh(Fp)
        ov = np.abs(np.einsum("ni,nij->nj", t_fiber, evecs))
        order = np.argsort(ov, axis=-1)
        sorted_evecs = np.take_along_axis(evecs, order[:, None, :], axis=-1)
        np.testing.assert_allclose(
            np.abs(axes[:, :, :-1]), np.abs(sorted_evecs[:, :, :-1]), atol=1e-10
        )

        # sigmas use the ORIGINAL info matrix: 1/sqrt(a^T F a) per column
        quad = np.einsum("nik,nij,njk->nk", axes, info, axes)
        np.testing.assert_allclose(sigmas, 1.0 / np.sqrt(quad), atol=1e-12)

    def test_eigen_axis_set_ridge_without_fiber(self):
        rng = np.random.default_rng(13)
        n, ndim = 2, 4
        info = _random_spd(rng, n, ndim)
        ridge_raw = rng.standard_normal((n, ndim))
        axes, _ = eigen_axis_set(info, ridge_axis=ridge_raw, sigma_max=np.inf)
        expected = ridge_raw / np.linalg.norm(ridge_raw, axis=-1, keepdims=True)
        np.testing.assert_allclose(axes[:, :, -1], expected, atol=1e-12)

    def test_axis_prior_bounds_limits_step(self):
        widths = np.array([1.0, 10.0, 1e-30])
        # one axis along the narrow column 0, one spread over 0 and 1, and
        # one axis with a negligible (< 1e-12) component on the tiny column
        axes = np.zeros((1, 3, 3))
        axes[0, :, 0] = [1.0, 0.0, 0.0]
        axes[0, :, 1] = [0.6, 0.8, 0.0]
        axes[0, :, 2] = [0.0, 1.0, 1e-13]
        out = axis_prior_bounds(axes, widths)
        np.testing.assert_allclose(out[0, 0], 1.0)
        np.testing.assert_allclose(out[0, 1], min(1.0 / 0.6, 10.0 / 0.8))
        # the 1e-13 component on the zero-width column is ignored
        np.testing.assert_allclose(out[0, 2], 10.0)

    def test_draw_axis_step_parallel_and_rng_shims(self):
        rng = np.random.default_rng(17)
        n, ndim = 64, 4
        info = _random_spd(rng, n, ndim)
        axes, sigmas = eigen_axis_set(info, sigma_max=np.inf)

        for r in (np.random.default_rng(19), np.random.RandomState(19)):
            dy, pick = draw_axis_step(axes, sigmas, r, jump_factor=2.0)
            self.assertEqual(dy.shape, (n, ndim))
            self.assertIsInstance(pick, np.ndarray)
            # every step lies exactly along its picked (orthonormal) axis
            for i in range(n):
                coeffs = axes[i].T @ dy[i]
                off = np.delete(coeffs, pick[i])
                np.testing.assert_allclose(off, 0.0, atol=1e-12)

        dy0, _ = draw_axis_step(axes, sigmas, np.random.default_rng(23),
                                jump_factor=0.0)
        np.testing.assert_allclose(dy0, 0.0)


def _shared_table(ndim, sigma=0.3, seed=29):
    rng = np.random.default_rng(seed)
    info = _random_spd(rng, 1, ndim, scale=1.0 / sigma**2)
    axes, sigmas = eigen_axis_set(info, sigma_max=np.inf)
    return axes[0], sigmas[0]


def _coords(nt, nw, nl, ndim, seed=31):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nt, nw, nl, ndim))


class EigenAxisMoveUnitTest(unittest.TestCase):
    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            EigenAxisMove(mode="not_a_mode")

    def test_is_not_a_stretch_move(self):
        # the addremove dispatch seam branches on isinstance(move, StretchMove)
        self.assertNotIsInstance(EigenAxisMove(), StretchMove)

    def test_missing_table_raises(self):
        move = EigenAxisMove()
        coords = {"model_0": _coords(2, 4, 1, 3)}
        with self.assertRaises(ValueError):
            move.get_proposal(coords, np.random.RandomState(0))

    def test_factors_zero_and_shapes(self):
        nt, nw, nl, ndim = 3, 8, 1, 4
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove({"model_0": (axes, sigmas)})
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, factors = move.get_proposal(coords, np.random.RandomState(1))
        self.assertEqual(q["model_0"].shape, (nt, nw, nl, ndim))
        self.assertEqual(factors.shape, (nt, nw))
        np.testing.assert_array_equal(factors, 0.0)
        # something actually moved
        self.assertFalse(np.array_equal(q["model_0"], coords["model_0"]))

    def test_axis_mode_step_parallel_to_one_column(self):
        nt, nw, nl, ndim = 2, 16, 1, 4
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove({"model_0": (axes, sigmas)}, mode="axis")
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, _ = move.get_proposal(coords, np.random.RandomState(2))
        dy = (q["model_0"] - coords["model_0"]).reshape(-1, ndim)
        for row in dy:
            coeffs = axes.T @ row
            # exactly one nonzero coefficient (orthonormal axes)
            self.assertEqual(np.sum(np.abs(coeffs) > 1e-12), 1)

    def test_full_mode_empirical_covariance(self):
        nt, nw, nl, ndim = 1, 4000, 1, 3
        axes, sigmas = _shared_table(ndim, sigma=0.5, seed=37)
        jf = 0.8
        move = EigenAxisMove(
            {"model_0": (axes, sigmas)}, mode="full", jump_factor=jf
        )
        coords = {"model_0": np.zeros((nt, nw, nl, ndim))}
        q, _ = move.get_proposal(coords, np.random.RandomState(3))
        dy = q["model_0"].reshape(-1, ndim)
        emp = np.cov(dy, rowvar=False)
        B = axes * sigmas[None, :]
        expected = jf**2 * (B @ B.T)
        np.testing.assert_allclose(
            emp, expected, atol=0.15 * np.max(np.abs(expected))
        )

    def test_periodic_wrap_applied(self):
        nt, nw, nl, ndim = 2, 32, 1, 2
        axes = np.eye(ndim)
        sigmas = np.array([5.0, 0.1])  # huge steps on the periodic column
        move = EigenAxisMove(
            {"model_0": (axes, sigmas)},
            periodic=PeriodicContainer({"model_0": {0: 1.0}}),
        )
        coords = {"model_0": np.full((nt, nw, nl, ndim), 0.5)}
        q, _ = move.get_proposal(coords, np.random.RandomState(4))
        p0 = q["model_0"][..., 0]
        self.assertGreaterEqual(p0.min(), 0.0)
        self.assertLess(p0.max(), 1.0)

    def test_per_leaf_table_indexing(self):
        nt, nw, nl, ndim = 2, 64, 2, 2
        axes = np.stack([np.eye(ndim), np.eye(ndim)])  # (nl, ndim, ndim)
        sigmas = np.array([[1.0, 0.0], [0.0, 1.0]])  # leaf0 -> e0, leaf1 -> e1
        move = EigenAxisMove({"model_0": (axes, sigmas)})
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, _ = move.get_proposal(coords, np.random.RandomState(5))
        dy = q["model_0"] - coords["model_0"]
        np.testing.assert_allclose(dy[:, :, 0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(dy[:, :, 1, 0], 0.0, atol=1e-14)
        # both leaves saw some real motion
        self.assertGreater(np.abs(dy[:, :, 0, 0]).max(), 0.0)
        self.assertGreater(np.abs(dy[:, :, 1, 1]).max(), 0.0)

    def test_full_shape_table_indexing(self):
        nt, nw, nl, ndim = 2, 3, 2, 2
        axes = np.broadcast_to(
            np.eye(ndim), (nt, nw, nl, ndim, ndim)
        ).copy()
        sigmas = np.zeros((nt, nw, nl, ndim))
        sigmas[..., 0] = 1.0
        sigmas[1, :, :, :] = 0.0  # freeze the second temperature entirely
        move = EigenAxisMove({"model_0": (axes, sigmas)})
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, _ = move.get_proposal(coords, np.random.RandomState(6))
        dy = q["model_0"] - coords["model_0"]
        np.testing.assert_allclose(dy[1], 0.0, atol=1e-14)
        self.assertGreater(np.abs(dy[0]).max(), 0.0)

    def test_multi_leaf_partial_inds_dead_leaves_untouched(self):
        nt, nw, nl, ndim = 2, 8, 3, 3
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove({"model_0": (axes, sigmas)})
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        inds = np.ones((nt, nw, nl), dtype=bool)
        inds[:, ::2, 2] = False  # leaf 2 dead on even walkers
        q, factors = move.get_proposal(
            coords, np.random.RandomState(7), branches_inds={"model_0": inds}
        )
        dy = q["model_0"] - coords["model_0"]
        # dead leaves byte-identical, every alive leaf moved
        np.testing.assert_array_equal(dy[~inds], 0.0)
        self.assertTrue(np.all(np.abs(dy[inds]).max(axis=-1) > 0))
        self.assertEqual(factors.shape, (nt, nw))
        np.testing.assert_array_equal(factors, 0.0)

    def test_multi_leaf_independent_axis_picks(self):
        # leaves of the SAME walker must draw independent axes -- folding
        # the leaf axis into the walker axis (\"pretend 1 leaf\") would give
        # every leaf of a walker the same picked axis.
        nt, nw, nl, ndim = 1, 32, 2, 6
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove({"model_0": (axes, sigmas)})
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, _ = move.get_proposal(coords, np.random.RandomState(8))
        dy = q["model_0"] - coords["model_0"]
        picks = np.zeros((nw, nl), dtype=int)
        for w in range(nw):
            for leaf in range(nl):
                coeffs = axes.T @ dy[0, w, leaf]
                picks[w, leaf] = int(np.argmax(np.abs(coeffs)))
        # with 6 axes and 32 walkers, identical picks across the two leaves
        # of every walker is a ~1e-25 probability event
        self.assertTrue(np.any(picks[:, 0] != picks[:, 1]))

    def test_branch_supps_channel(self):
        nt, nw, nl, ndim = 2, 16, 2, 2
        axes5 = np.broadcast_to(
            np.eye(ndim), (nt, nw, nl, ndim, ndim)
        ).copy()
        sigmas4 = np.zeros((nt, nw, nl, ndim))
        sigmas4[:, :, 0, 0] = 1.0  # leaf0 -> e0 only
        sigmas4[:, :, 1, 1] = 1.0  # leaf1 -> e1 only
        supp = BranchSupplemental(
            {"eigen_axes": axes5, "eigen_sigmas": sigmas4},
            base_shape=(nt, nw, nl),
        )
        move = EigenAxisMove()  # no table: must fall back to the supps
        coords = {"model_0": _coords(nt, nw, nl, ndim)}
        q, _ = move.get_proposal(
            coords,
            np.random.RandomState(9),
            branch_supps={"model_0": supp},
        )
        dy = q["model_0"] - coords["model_0"]
        np.testing.assert_allclose(dy[:, :, 0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(dy[:, :, 1, 0], 0.0, atol=1e-14)
        self.assertGreater(np.abs(dy).max(), 0.0)

    def test_set_axes_updates_table(self):
        ndim = 3
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove()
        move.set_axes("model_0", axes, sigmas)
        coords = {"model_0": _coords(1, 4, 1, ndim)}
        q, _ = move.get_proposal(coords, np.random.RandomState(10))
        self.assertFalse(np.array_equal(q["model_0"], coords["model_0"]))
        # zero-step table swap: freezes the proposal
        move.set_axes("model_0", axes, np.zeros(ndim))
        q2, _ = move.get_proposal(coords, np.random.RandomState(11))
        np.testing.assert_array_equal(q2["model_0"], coords["model_0"])

    @unittest.skipUnless(HAS_CUPY, "cupy not available")
    def test_gpu_arrays_roundtrip(self):
        ndim = 3
        axes, sigmas = _shared_table(ndim)
        move = EigenAxisMove(
            {"model_0": (cp.asarray(axes), cp.asarray(sigmas))}
        )
        coords = {"model_0": cp.asarray(_coords(2, 4, 1, ndim))}
        q, factors = move.get_proposal(coords, np.random.RandomState(12))
        self.assertIsInstance(q["model_0"], cp.ndarray)
        self.assertEqual(q["model_0"].shape, (2, 4, 1, ndim))


class EigenAxisMoveInErynTest(unittest.TestCase):
    """Statistical recovery of a correlated Gaussian through the sampler."""

    def _run(self, mode, jump_factor, nsteps, seed):
        np.random.seed(seed)
        ndim = 3
        n_walkers = 32

        mu = np.array([0.5, -1.0, 2.0])
        L = np.array(
            [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.3, -0.2, 0.4]]
        )
        cov = L @ L.T
        inv_cov = np.linalg.inv(cov)

        def log_like(x):
            diff = np.atleast_2d(x) - mu
            return float(
                -0.5 * np.einsum("ni,ij,nj->n", diff, inv_cov, diff)[0]
            )

        lims = 12.0
        priors = ProbDistContainer(
            {i: uniform_dist(-lims + mu[i], lims + mu[i]) for i in range(ndim)}
        )

        axes, sigmas = eigen_axis_set(inv_cov[None], sigma_max=np.inf)
        move = EigenAxisMove(
            {"model_0": (axes[0], sigmas[0])},
            mode=mode,
            jump_factor=jump_factor,
        )

        ensemble = EnsembleSampler(
            n_walkers, ndim, log_like, priors, moves=move, vectorize=False
        )
        coords = mu + np.random.randn(n_walkers, ndim)
        ensemble.run_mcmc(coords, nsteps, burn=200, progress=False)

        chain = ensemble.get_chain()["model_0"].reshape(-1, ndim)
        self.assertTrue(np.allclose(chain.mean(axis=0), mu, atol=0.25))
        emp_cov = np.cov(chain, rowvar=False)
        self.assertTrue(
            np.allclose(emp_cov, cov, atol=0.3 * np.max(np.abs(cov)))
        )

    def test_correlated_gaussian_recovery_axis_mode(self):
        self._run("axis", 1.0, 900, seed=61)

    def test_correlated_gaussian_recovery_full_mode(self):
        self._run("full", 0.7, 700, seed=67)


class EigenAxisIsolationTest(unittest.TestCase):
    """Disjoint per-sampler prior bands stay disjoint under the move."""

    def test_folded_axis_isolation(self):
        nsamplers, ntemps, nwalkers = 3, 4, 16

        class UnitCubeTransform:
            def __init__(self, lo, hi):
                self.lo, self.hi = lo, hi

            def transform_to_prior_basis(self, coords, running_idx):
                lo = self.lo[running_idx][:, None, None]
                hi = self.hi[running_idx][:, None, None]
                coords[..., 0] = (coords[..., 0] - lo) / (hi - lo)

            def adjust_logp(self, logp, running_idx):
                logp -= np.log(
                    (self.hi - self.lo)[running_idx][:, None, None]
                )

        lo = np.array([0.0, 10.0, 20.0])
        hi = np.array([1.0, 12.0, 24.0])
        priors = ProbDistContainer(
            {0: uniform_dist(0.0, 1.0), 1: uniform_dist(-5.0, 5.0)}
        )

        np.random.seed(71)
        move = EigenAxisMove(
            {"model_0": (np.eye(2), np.array([0.05, 0.5]))}
        )
        sampler = EnsembleSampler(
            nwalkers,
            2,
            lambda x: 0.0,
            priors,
            tempering_kwargs=dict(ntemps=ntemps),
            nsamplers=nsamplers,
            moves=move,
            prior_transform_fn=UnitCubeTransform(lo, hi),
        )
        self.assertEqual(sampler.moves[0].nsamplers, nsamplers)
        self.assertEqual(sampler.moves[0].ntemps_per_sampler, ntemps)

        c0 = lo[:, None, None] + np.random.uniform(
            0, 1, size=(nsamplers, ntemps, nwalkers)
        ) * (hi - lo)[:, None, None]
        c1 = np.random.uniform(-5, 5, size=(nsamplers, ntemps, nwalkers))
        coords = np.stack([c0, c1], axis=-1)
        sampler.run_mcmc(coords, 20, progress=False)

        param0 = sampler.get_chain()["model_0"][..., 0, 0]
        for s in range(nsamplers):
            self.assertGreaterEqual(param0[:, s].min(), lo[s])
            self.assertLessEqual(param0[:, s].max(), hi[s])


if __name__ == "__main__":
    unittest.main()
