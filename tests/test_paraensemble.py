import unittest

import numpy as np
from gpubackendtools import has_backend

from eryn.backends.parabackend import ParaBackend
from eryn.paraensemble import (
    _CUDA_BACKEND_PRIORITY,
    ParaEnsembleSampler,
    shuffle_along_axis,
)
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import ParaState
from eryn.utils import PeriodicContainer

_CUDA_AVAILABLE = any(has_backend(name) for name in _CUDA_BACKEND_PRIORITY)

NDIM = 3
NWALKERS = 10
NGROUPS = 4
NTEMPS = 4
PRIOR_LIM = 5.0


def gaussian_log_like(x, mu, invcov):
    diff = x - mu
    return -0.5 * np.einsum("...i,...ij,...j->...", diff, invcov, diff)


class IdentityPriorTransform:
    """Pass-through implementation of the prior_transform_fn contract."""

    def transform_to_prior_basis(self, coords, groups_running):
        return

    def adjust_logp(self, logp, groups_running):
        return logp


class UnitCubeTransform:
    """Per-group [lo, hi] -> [0, 1] map on dim 0.

    Mirrors the contract of the LISAanalysistools ``PriorTransformFn``
    (gbspecialstretch), including handling ``groups_running=None``.
    """

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def transform_to_prior_basis(self, coords, groups_running):
        if groups_running is None:
            groups_running = np.arange(len(self.lo))
        lo = self.lo[groups_running][:, None, None]
        hi = self.hi[groups_running][:, None, None]
        coords[:, :, :, 0] = (coords[:, :, :, 0] - lo) / (hi - lo)

    def adjust_logp(self, logp, groups_running):
        if groups_running is None:
            groups_running = np.arange(len(self.lo))
        lo = self.lo[groups_running]
        hi = self.hi[groups_running]
        logp[:] += np.log(1.0 / (hi - lo))[:, None, None]
        return logp


def make_sampler(
    ndim=NDIM,
    nwalkers=NWALKERS,
    ngroups=NGROUPS,
    ntemps=NTEMPS,
    **kwargs,
):
    means = np.zeros(ndim)
    invcov = np.eye(ndim)

    name = kwargs.pop("name", "gauss")
    priors = kwargs.pop(
        "priors",
        {name: ProbDistContainer({i: uniform_dist(-PRIOR_LIM, PRIOR_LIM) for i in range(ndim)})},
    )

    kwargs.setdefault("tempering_kwargs", dict(ntemps=ntemps) if ntemps > 1 else None)
    kwargs.setdefault("args", [means, invcov])

    return ParaEnsembleSampler(
        ndim,
        nwalkers,
        ngroups,
        kwargs.pop("log_like_fn", gaussian_log_like),
        priors,
        name=name,
        **kwargs,
    )


def make_state(sampler, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(
        -PRIOR_LIM,
        PRIOR_LIM,
        size=(sampler.ngroups, sampler.ntemps, sampler.nwalkers, sampler.ndim),
    )
    return ParaState(
        {sampler.name: coords},
        groups_running=np.ones(sampler.ngroups, dtype=bool),
    )


class ShuffleAlongAxisTest(unittest.TestCase):
    def test_rows_are_permutations(self):
        np.random.seed(7)
        a = np.tile(np.arange(12), (5, 1))
        out = shuffle_along_axis(a, -1)
        self.assertEqual(out.shape, a.shape)
        for row in out:
            np.testing.assert_array_equal(np.sort(row), np.arange(12))
        # at least one row should actually move
        self.assertTrue((out != a).any())


class ParaEnsembleSamplerBasicTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_tempered_run_shapes_and_storage(self):
        nsteps = 12
        sampler = make_sampler()
        state = make_state(sampler)
        out = sampler.run_mcmc(state, nsteps, burn=3, thin_by=2)

        self.assertIsInstance(out, ParaState)
        self.assertEqual(sampler.backend.iteration, nsteps)

        chain = sampler.get_chain()
        self.assertEqual(chain.shape, (nsteps, NGROUPS, NTEMPS, NWALKERS, NDIM))
        self.assertTrue(np.isfinite(chain).all())

        log_like = sampler.get_log_like()
        log_prior = sampler.get_log_prior()
        self.assertEqual(log_like.shape, (nsteps, NGROUPS, NTEMPS, NWALKERS))
        self.assertEqual(log_prior.shape, (nsteps, NGROUPS, NTEMPS, NWALKERS))
        self.assertTrue(np.isfinite(log_like).all())
        self.assertTrue(np.isfinite(log_prior).all())

        betas = sampler.get_betas()
        self.assertEqual(betas.shape, (nsteps, NGROUPS, NTEMPS))
        # cold chain stays at beta = 1
        np.testing.assert_allclose(betas[:, :, 0], 1.0)

        # acceptance bookkeeping
        accepted_fraction = sampler.backend.accepted / sampler.backend.iteration
        self.assertTrue((accepted_fraction > 0).any())
        self.assertTrue((accepted_fraction <= 1).all())
        self.assertEqual(sampler.backend.swaps_accepted.shape, (NGROUPS, NTEMPS - 1))
        self.assertTrue((sampler.backend.swaps_accepted >= 0).all())

        # stored log_prior/log_like must be consistent with the stored chain
        flat = chain[-1].reshape(-1, NDIM)
        expected_logl = gaussian_log_like(flat, np.zeros(NDIM), np.eye(NDIM))
        np.testing.assert_allclose(log_like[-1].flatten(), expected_logl, rtol=1e-12)

    def test_no_tempering_run(self):
        nsteps = 8
        sampler = make_sampler(ntemps=1)
        self.assertEqual(sampler.ntemps, 1)
        state = make_state(sampler)
        sampler.run_mcmc(state, nsteps)

        chain = sampler.get_chain()
        self.assertEqual(chain.shape, (nsteps, NGROUPS, 1, NWALKERS, NDIM))
        np.testing.assert_allclose(sampler.get_betas(), 1.0)

    def test_recovers_gaussian_moments(self):
        sampler = make_sampler(ndim=2, nwalkers=16, ngroups=2, ntemps=2)
        state = make_state(sampler)
        sampler.run_mcmc(state, 250, burn=50)

        # cold chain only
        samples = sampler.get_chain()[:, :, 0].reshape(-1, 2)
        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.25)
        np.testing.assert_allclose(samples.std(axis=0), 1.0, atol=0.25)

    def test_run_mcmc_continuation(self):
        sampler = make_sampler()
        state = make_state(sampler)
        sampler.run_mcmc(state, 4)
        self.assertEqual(sampler.backend.iteration, 4)

        # continue from the previous state
        sampler.run_mcmc(None, 3)
        self.assertEqual(sampler.backend.iteration, 7)

    def test_run_mcmc_none_without_history_raises(self):
        sampler = make_sampler()
        with self.assertRaises(ValueError):
            sampler.run_mcmc(None, 2)

    def test_groups_running_mask(self):
        nsteps = 6
        sampler = make_sampler()
        state = make_state(sampler)
        groups_running = np.ones(NGROUPS, dtype=bool)
        groups_running[1] = False
        state.groups_running = groups_running
        coords_initial = state.branches[sampler.name].coords.copy()

        out = sampler.run_mcmc(state, nsteps)

        # the switched-off group never moves
        np.testing.assert_array_equal(out.branches[sampler.name].coords[1], coords_initial[1])
        # running groups do move
        self.assertTrue((out.branches[sampler.name].coords[0] != coords_initial[0]).any())

        # backend stores NaN for non-running groups, finite values otherwise
        chain = sampler.get_chain()
        self.assertTrue(np.isnan(chain[:, 1]).all())
        self.assertTrue(np.isfinite(chain[:, 0]).all())
        self.assertTrue(np.isfinite(chain[:, 2:]).all())

    def test_default_groups_running(self):
        # groups_running=None in the initial state means all groups run
        sampler = make_sampler()
        state = make_state(sampler)
        state.groups_running = None
        sampler.run_mcmc(state, 2)
        self.assertTrue(np.isfinite(sampler.get_chain()).all())

    def test_update_and_stopping_fns(self):
        update_calls = []

        def update_fn(i, state, sampler_in):
            update_calls.append((i, sampler_in))

        sampler = make_sampler(update_fn=update_fn, update_iterations=2)
        state = make_state(sampler)
        sampler.run_mcmc(state, 6)
        self.assertEqual(len(update_calls), 3)
        self.assertIs(update_calls[0][1], sampler)

        stop_calls = []

        def stopping_fn(i, state, sampler_in):
            stop_calls.append(i)
            return True

        sampler = make_sampler(stopping_fn=stopping_fn, stopping_iterations=1)
        state = make_state(sampler)
        sampler.run_mcmc(state, 10)
        self.assertEqual(len(stop_calls), 1)
        # stopped after the first stored iteration
        self.assertEqual(sampler.backend.iteration, 1)

    def test_periodic_container_setup(self):
        periodic = {"gauss": {2: 2 * np.pi}}
        sampler = make_sampler(periodic=periodic)
        self.assertIsInstance(sampler.periodic, PeriodicContainer)

        container = PeriodicContainer(periodic)
        sampler = make_sampler(periodic=container)
        self.assertIs(sampler.periodic, container)

        state = make_state(sampler)
        sampler.run_mcmc(state, 2)
        self.assertTrue(np.isfinite(sampler.get_chain()).all())


class ParaEnsembleSamplerPriorTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_compute_log_prior_no_transform(self):
        # regression: prior_transform_fn=None used to raise AttributeError
        sampler = make_sampler()
        state = make_state(sampler)
        logp = sampler.compute_log_prior(state.branches_coords)
        self.assertEqual(logp.shape, (NGROUPS, NTEMPS, NWALKERS))
        expected = np.log((1.0 / (2 * PRIOR_LIM)) ** NDIM)
        np.testing.assert_allclose(logp, expected, rtol=1e-12)

    def test_identity_transform_matches_no_transform(self):
        sampler_plain = make_sampler()
        sampler_ident = make_sampler(prior_transform_fn=IdentityPriorTransform())
        state = make_state(sampler_plain)
        np.testing.assert_allclose(
            sampler_plain.compute_log_prior(state.branches_coords),
            sampler_ident.compute_log_prior(state.branches_coords),
        )

    def test_unit_cube_transform(self):
        # per-group bounds on dim 0; prior on dim 0 defined on the unit cube
        lo = np.array([10.0, 20.0, 30.0, 40.0])
        hi = lo + np.array([2.0, 4.0, 8.0, 16.0])
        priors = {
            "gauss": ProbDistContainer(
                {
                    0: uniform_dist(0.0, 1.0),
                    **{i: uniform_dist(-PRIOR_LIM, PRIOR_LIM) for i in range(1, NDIM)},
                }
            )
        }
        sampler = make_sampler(priors=priors, prior_transform_fn=UnitCubeTransform(lo, hi))

        rng = np.random.default_rng(3)
        coords = rng.uniform(-PRIOR_LIM, PRIOR_LIM, size=(NGROUPS, NTEMPS, NWALKERS, NDIM))
        # put dim 0 inside each group's physical band
        u = rng.uniform(size=(NGROUPS, NTEMPS, NWALKERS))
        coords[:, :, :, 0] = lo[:, None, None] + u * (hi - lo)[:, None, None]
        coords_before = coords.copy()

        # LAT consumption pattern: groups_running=None
        logp = sampler.compute_log_prior({"gauss": coords})

        expected = (
            np.log((1.0 / (2 * PRIOR_LIM)) ** (NDIM - 1)) + np.log(1.0 / (hi - lo))[:, None, None]
        )
        np.testing.assert_allclose(logp, np.broadcast_to(expected, logp.shape))

        # input coordinates must not be mutated by the transform
        np.testing.assert_array_equal(coords, coords_before)

        # explicit groups_running subset
        logp_sub = sampler.compute_log_prior(
            {"gauss": coords[2:]}, groups_running=np.arange(NGROUPS)[2:]
        )
        np.testing.assert_allclose(logp_sub, logp[2:])

    def test_sampling_with_transform_stays_in_band(self):
        lo = np.full(NGROUPS, 10.0)
        hi = np.full(NGROUPS, 12.0)
        priors = {
            "gauss": ProbDistContainer(
                {
                    0: uniform_dist(0.0, 1.0),
                    **{i: uniform_dist(-PRIOR_LIM, PRIOR_LIM) for i in range(1, NDIM)},
                }
            )
        }

        # likelihood only looks at dims 1+, so dim 0 samples its prior band
        def log_like_fn(x, mu, invcov):
            return gaussian_log_like(x[:, 1:], mu, invcov)

        sampler = make_sampler(
            priors=priors,
            prior_transform_fn=UnitCubeTransform(lo, hi),
            log_like_fn=log_like_fn,
            args=[np.zeros(NDIM - 1), np.eye(NDIM - 1)],
        )

        state = make_state(sampler)
        coords = state.branches[sampler.name].coords
        rng = np.random.default_rng(5)
        coords[:, :, :, 0] = (
            lo[:, None, None, None].squeeze(-1)
            + rng.uniform(size=(NGROUPS, NTEMPS, NWALKERS)) * (hi - lo)[:, None, None]
        )

        sampler.run_mcmc(state, 10)

        chain_dim0 = sampler.get_chain()[..., 0]
        self.assertTrue((chain_dim0 >= lo[0]).all())
        self.assertTrue((chain_dim0 <= hi[0]).all())


class ParaEnsembleSamplerGibbsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_fixed_dimensions_stay_fixed(self):
        # regression: fixed (non-Gibbs) dims used to be overwritten with
        # the complement walkers' values, so they migrated between walkers
        gibbs = np.array([True, True, False])

        def log_like_fn(x, mu, invcov):
            return gaussian_log_like(x[:, :2], mu, invcov)

        sampler = make_sampler(
            ntemps=1,  # temperature swaps legitimately exchange walkers
            gibbs_sampling_setup=gibbs,
            log_like_fn=log_like_fn,
            args=[np.zeros(2), np.eye(2)],
        )

        state = make_state(sampler)
        coords = state.branches[sampler.name].coords
        # give every walker a unique, identifiable value in the fixed dim
        fixed_vals = np.linspace(-4.0, 4.0, NGROUPS * NWALKERS).reshape(NGROUPS, 1, NWALKERS)
        coords[:, :, :, 2] = fixed_vals

        sampler.run_mcmc(state, 10)

        chain = sampler.get_chain()
        # moving dims actually moved
        self.assertTrue((chain[-1, :, :, :, :2] != chain[0, :, :, :, :2]).any())
        # fixed dim is bit-identical for every walker at every step
        np.testing.assert_array_equal(
            chain[..., 2], np.broadcast_to(fixed_vals, chain[..., 2].shape)
        )

    def test_gibbs_validation(self):
        with self.assertRaises(AssertionError):
            make_sampler(gibbs_sampling_setup=np.ones(NDIM + 1, dtype=bool))
        with self.assertRaises(AssertionError):
            make_sampler(gibbs_sampling_setup=np.ones(NDIM, dtype=int))


class ParaEnsembleSamplerValidationTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_odd_nwalkers_raises(self):
        with self.assertRaises(ValueError):
            make_sampler(nwalkers=11)

    def test_priors_missing_branch_raises(self):
        priors = {"other_name": ProbDistContainer({0: uniform_dist(-1.0, 1.0)})}
        with self.assertRaises(ValueError):
            make_sampler(priors=priors)

    def test_bad_periodic_raises(self):
        with self.assertRaises(ValueError):
            make_sampler(periodic="not_a_dict")

    @unittest.skipIf(_CUDA_AVAILABLE, "CUDA backend installed; gpu request is valid")
    def test_gpu_without_cuda_backend_raises(self):
        with self.assertRaises(ValueError):
            make_sampler(gpu=0)

    def test_wrong_coords_shape_raises(self):
        sampler = make_sampler()
        coords = np.zeros((NGROUPS, NTEMPS, NWALKERS + 2, NDIM))
        state = ParaState({"gauss": coords}, groups_running=np.ones(NGROUPS, dtype=bool))
        with self.assertRaisesRegex(ValueError, "incompatible input dimensions"):
            sampler.run_mcmc(state, 2)

    def test_wrong_betas_shape_raises(self):
        sampler = make_sampler()
        state = make_state(sampler)
        state.betas = np.ones((NGROUPS, NTEMPS + 1))
        with self.assertRaisesRegex(ValueError, "betas"):
            sampler.run_mcmc(state, 2)

    def test_out_of_prior_initial_state_raises(self):
        sampler = make_sampler()
        state = make_state(sampler)
        state.branches[sampler.name].coords[0, 0, 0, 0] = 100.0  # outside prior
        with self.assertRaisesRegex(ValueError, "log_prior"):
            sampler.run_mcmc(state, 2)

    def test_invalid_thin_by_raises(self):
        sampler = make_sampler()
        state = make_state(sampler)
        with self.assertRaisesRegex(ValueError, "thinning"):
            sampler.run_mcmc(state, 2, thin_by=0)

    def test_infinite_iterations_requires_store_false(self):
        sampler = make_sampler()
        state = make_state(sampler)
        gen = sampler.sample(state, iterations=None, store=True)
        with self.assertRaisesRegex(ValueError, "store"):
            next(gen)

    def test_external_backend_shape_check(self):
        backend = ParaBackend()
        backend.reset(NDIM, NWALKERS, NGROUPS, ntemps=NTEMPS, branch_name="gauss")
        sampler = make_sampler(backend=backend)
        self.assertIs(sampler.backend, backend)

        bad_backend = ParaBackend()
        bad_backend.reset(NDIM, NWALKERS, NGROUPS + 1, ntemps=NTEMPS, branch_name="gauss")
        with self.assertRaises(AssertionError):
            make_sampler(backend=bad_backend)


class ParaEnsembleSamplerLikelihoodTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_nan_log_like_replaced(self):
        def nan_log_like(x, mu, invcov):
            out = gaussian_log_like(x, mu, invcov)
            out[::2] = np.nan
            return out

        sampler = make_sampler(log_like_fn=nan_log_like)
        state = make_state(sampler)
        logl = sampler.compute_log_like(state.branches_coords)
        self.assertFalse(np.isnan(logl).any())
        self.assertTrue((logl[logl < -1e299] == -1e300).all())

    def test_out_of_prior_points_skipped(self):
        seen_counts = []

        def counting_log_like(x, mu, invcov):
            seen_counts.append(len(x))
            return gaussian_log_like(x, mu, invcov)

        sampler = make_sampler(log_like_fn=counting_log_like)
        state = make_state(sampler)
        logp = sampler.compute_log_prior(state.branches_coords)
        logp[0, 0, :2] = -np.inf  # mark two points out of prior

        logl = sampler.compute_log_like(state.branches_coords, logp=logp)

        total = NGROUPS * NTEMPS * NWALKERS
        self.assertEqual(seen_counts, [total - 2])
        np.testing.assert_array_equal(logl[0, 0, :2], -1e300)
        self.assertTrue(np.isfinite(logl.flatten()[2:]).all())


class ParaEnsembleSamplerTemperingTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_stop_adaptation_freezes_ladder(self):
        sampler = make_sampler(tempering_kwargs=dict(ntemps=NTEMPS, stop_adaptation=0))
        state = make_state(sampler)
        sampler.run_mcmc(state, 15)
        betas = sampler.get_betas()
        np.testing.assert_array_equal(betas, np.broadcast_to(betas[0], betas.shape))

    def test_adaptive_ladder_moves(self):
        sampler = make_sampler(tempering_kwargs=dict(ntemps=NTEMPS))
        state = make_state(sampler)
        sampler.run_mcmc(state, 30)
        betas = sampler.get_betas()
        # middle rungs adapt; the cold rung is pinned at beta = 1
        self.assertTrue((betas[-1, :, 1:-1] != betas[0, :, 1:-1]).any())
        np.testing.assert_allclose(betas[:, :, 0], 1.0)

    def test_swap_bookkeeping(self):
        sampler = make_sampler()
        state = make_state(sampler)
        sampler.run_mcmc(state, 10)
        self.assertEqual(sampler.swaps_accepted.shape, (NGROUPS, NTEMPS - 1))
        self.assertTrue((sampler.swaps_accepted >= 0).all())
        self.assertTrue((sampler.swaps_accepted <= NWALKERS).all())


class ComputeBackendTest(unittest.TestCase):
    """ParaEnsembleSampler sources its array module from GBT backends."""

    def setUp(self):
        np.random.seed(42)

    def test_default_backend_is_cpu(self):
        sampler = make_sampler()
        self.assertEqual(sampler.compute_backend.name, "gbt_cpu")
        self.assertIs(sampler.xp, np)
        self.assertFalse(sampler.use_gpu)
        self.assertIsNone(sampler.gpu)

    def test_force_backend_cpu_names(self):
        for name in ("cpu", "gbt_cpu"):
            sampler = make_sampler(force_backend=name)
            self.assertEqual(sampler.compute_backend.name, "gbt_cpu")
            self.assertIs(sampler.xp, np)
            self.assertFalse(sampler.use_gpu)

    def test_force_backend_cpu_full_run(self):
        sampler = make_sampler(force_backend="cpu")
        state = make_state(sampler)
        sampler.run_mcmc(state, 3)
        self.assertTrue(np.isfinite(sampler.get_chain()).all())

    def test_jax_backend_rejected(self):
        # in-place array mutation is incompatible with jax
        for name in ("jax", "gbt_jax"):
            with self.assertRaisesRegex(ValueError, "jax"):
                make_sampler(force_backend=name)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            make_sampler(force_backend="not_a_backend")

    def test_non_string_force_backend_rejected(self):
        with self.assertRaises(ValueError):
            make_sampler(force_backend=12)

    def test_conflicting_force_backend_and_gpu(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            make_sampler(force_backend="cpu", gpu=0)

    @unittest.skipIf(_CUDA_AVAILABLE, "CUDA backend installed; switch would succeed")
    def test_add_gpu_index_without_cuda_raises(self):
        sampler = make_sampler()
        with self.assertRaises(ValueError):
            sampler.add_gpu_index(0)


class ParaBackendSampleRetrievalTest(unittest.TestCase):
    """Tests for ParaBackend.get_a_sample / get_last_sample.

    Regression: these used to call the nonexistent ``get_inds`` and build
    a malformed (5-D coords, dict groups_running, squeezed betas) ParaState.
    """

    def setUp(self):
        np.random.seed(42)

    def test_get_last_sample_matches_storage(self):
        sampler = make_sampler()
        state = make_state(sampler)
        sampler.run_mcmc(state, 6)

        last = sampler.get_last_sample()
        self.assertIsInstance(last, ParaState)

        chain = sampler.get_chain()
        np.testing.assert_array_equal(last.branches[sampler.name].coords, chain[-1])
        np.testing.assert_array_equal(last.log_like, sampler.get_log_like()[-1])
        np.testing.assert_array_equal(last.log_prior, sampler.get_log_prior()[-1])
        np.testing.assert_array_equal(last.betas, sampler.get_betas()[-1])
        self.assertEqual(last.groups_running.shape, (NGROUPS,))
        self.assertTrue(last.groups_running.all())

    def test_get_a_sample_indexing(self):
        sampler = make_sampler()
        state = make_state(sampler)
        sampler.run_mcmc(state, 6)

        chain = sampler.get_chain()
        for it in (0, 2, 5):
            sample = sampler.backend.get_a_sample(it)
            np.testing.assert_array_equal(sample.branches[sampler.name].coords, chain[it])
            np.testing.assert_array_equal(sample.log_like, sampler.get_log_like()[it])

    def test_untempered_betas_shape_preserved(self):
        # regression: betas used to be ``.squeeze()``-ed, collapsing the
        # ntemps=1 axis (and ngroups=1 axes) out of the restored state
        sampler = make_sampler(ntemps=1)
        state = make_state(sampler)
        sampler.run_mcmc(state, 4)

        last = sampler.get_last_sample()
        self.assertEqual(last.betas.shape, (NGROUPS, 1))

    def test_round_trip_continuation(self):
        sampler = make_sampler()
        state = make_state(sampler)
        sampler.run_mcmc(state, 5)

        # restart a fresh sampler from the stored last sample
        last = sampler.get_last_sample()
        sampler2 = make_sampler()
        sampler2.run_mcmc(last, 3)
        self.assertEqual(sampler2.backend.iteration, 3)
        self.assertTrue(np.isfinite(sampler2.get_chain()).all())

    def test_get_a_sample_before_run_raises(self):
        sampler = make_sampler()
        with self.assertRaises(AttributeError):
            sampler.backend.get_a_sample(0)
        with self.assertRaises(AttributeError):
            sampler.get_last_sample()

    def test_reset_base_restores_info(self):
        backend = ParaBackend()
        backend.reset(
            NDIM,
            NWALKERS,
            NGROUPS,
            ntemps=NTEMPS,
            branch_name="gauss",
            extra_tag="hello",
        )
        self.assertEqual(backend.extra_tag, "hello")

        backend.reset_base()
        self.assertEqual(backend.iteration, 0)
        self.assertEqual(backend.extra_tag, "hello")
        # regression: **info used to be re-wrapped into an ``info`` attribute
        self.assertFalse(hasattr(backend, "info"))


if __name__ == "__main__":
    unittest.main()
