"""Tests for the independent-samplers (``nsamplers``) axis on EnsembleSampler.

This is the replacement for the deprecated ParaEnsembleSampler: many fully
independent parallel-tempered ensembles advance simultaneously with batched
likelihood calls. Internally the sampler axis is FOLDED into the temperature
axis (leading axis size ``nsamplers * ntemps``); storage always carries the
sampler axis and getters squeeze it when ``nsamplers == 1``.
"""

import os
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from eryn.backends import Backend, HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import GaussianMove, GroupStretchMove, TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State

NDIM = 3
NWALKERS = 16
NSAMPLERS = 3
NTEMPS = 4


def gaussian_log_like(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()


def make_priors(ndim=NDIM, lim=5.0):
    return ProbDistContainer({i: uniform_dist(-lim, lim) for i in range(ndim)})


def make_sampler(
    nsamplers=NSAMPLERS,
    ntemps=NTEMPS,
    nwalkers=NWALKERS,
    ndim=NDIM,
    backend=None,
    tempered=True,
    **kwargs,
):
    priors = make_priors(ndim)
    tempering_kwargs = dict(ntemps=ntemps) if tempered else {}
    sampler = EnsembleSampler(
        nwalkers,
        ndim,
        gaussian_log_like,
        priors,
        args=[np.zeros(ndim), np.eye(ndim)],
        tempering_kwargs=tempering_kwargs,
        nsamplers=nsamplers,
        backend=backend,
        **kwargs,
    )
    return sampler, priors


class UnitCubeTransform:
    """prior_transform_fn mapping per-sampler bands onto priors on [0, 1] (param 0)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def transform_to_prior_basis(self, coords, running_idx):
        lo = self.lo[running_idx][:, None, None]
        hi = self.hi[running_idx][:, None, None]
        coords[..., 0] = (coords[..., 0] - lo) / (hi - lo)

    def adjust_logp(self, logp, running_idx):
        logp -= np.log((self.hi - self.lo)[running_idx][:, None, None])


class NSamplersSingleSamplerCompatTest(unittest.TestCase):
    """nsamplers == 1 must look exactly like the legacy single-sampler layout."""

    def test_shapes_and_determinism(self):
        nsteps = 10

        chains = []
        for _ in range(2):
            np.random.seed(42)
            sampler, priors = make_sampler(nsamplers=1)
            coords = priors.rvs(size=(NTEMPS, NWALKERS))
            sampler.run_mcmc(coords, nsteps, burn=3, progress=False)
            chains.append(sampler.get_chain()["model_0"])

            self.assertEqual(
                sampler.get_chain()["model_0"].shape,
                (nsteps, NTEMPS, NWALKERS, 1, NDIM),
            )
            self.assertEqual(sampler.get_log_like().shape, (nsteps, NTEMPS, NWALKERS))
            self.assertEqual(sampler.get_betas().shape, (nsteps, NTEMPS))
            self.assertEqual(sampler.backend.accepted.shape, (NTEMPS, NWALKERS))
            self.assertEqual(sampler.backend.swaps_accepted.shape, (NTEMPS - 1,))
            self.assertEqual(sampler.acceptance_fraction.shape, (NTEMPS, NWALKERS))
            self.assertEqual(
                sampler.backend.shape["model_0"], (NTEMPS, NWALKERS, 1, NDIM)
            )
            self.assertEqual(
                sampler.get_log_posterior().shape, (nsteps, NTEMPS, NWALKERS)
            )

            last = sampler.get_last_sample()
            self.assertEqual(last.nsamplers, 1)
            self.assertEqual(
                last.branches["model_0"].shape, (NTEMPS, NWALKERS, 1, NDIM)
            )
            self.assertEqual(last.betas.shape, (NTEMPS,))

        # same seed -> bitwise identical chains (protects the RNG stream)
        self.assertTrue(np.array_equal(chains[0], chains[1]))

    def test_hdf_roundtrip_and_restart(self):
        nsteps = 8
        fp = tempfile.mktemp(suffix=".h5")
        try:
            np.random.seed(3)
            sampler, priors = make_sampler(nsamplers=1, backend=fp)
            coords = priors.rvs(size=(NTEMPS, NWALKERS))
            sampler.run_mcmc(coords, nsteps, progress=False)

            self.assertEqual(sampler.backend.nsamplers, 1)
            self.assertEqual(
                sampler.get_chain()["model_0"].shape,
                (nsteps, NTEMPS, NWALKERS, 1, NDIM),
            )
            self.assertEqual(sampler.backend.accepted.shape, (NTEMPS, NWALKERS))

            # restart from the file with a fresh sampler object
            sampler2, _ = make_sampler(nsamplers=1, backend=HDFBackend(fp))
            sampler2.run_mcmc(None, 3, progress=False)
            self.assertEqual(sampler2.backend.iteration, nsteps + 3)
        finally:
            if os.path.exists(fp):
                os.remove(fp)


class NSamplersMultiTest(unittest.TestCase):
    def test_shapes_and_storage(self):
        nsteps = 12
        np.random.seed(5)
        sampler, priors = make_sampler()
        coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
        sampler.run_mcmc(coords, nsteps, burn=3, progress=False)

        chain = sampler.get_chain()["model_0"]
        self.assertEqual(chain.shape, (nsteps, NSAMPLERS, NTEMPS, NWALKERS, 1, NDIM))
        self.assertEqual(
            sampler.get_log_like().shape, (nsteps, NSAMPLERS, NTEMPS, NWALKERS)
        )
        self.assertEqual(sampler.get_betas().shape, (nsteps, NSAMPLERS, NTEMPS))
        self.assertEqual(
            sampler.get_log_posterior().shape, (nsteps, NSAMPLERS, NTEMPS, NWALKERS)
        )
        self.assertEqual(
            sampler.backend.accepted.shape, (NSAMPLERS, NTEMPS, NWALKERS)
        )
        self.assertEqual(
            sampler.backend.swaps_accepted.shape, (NSAMPLERS, NTEMPS - 1)
        )
        self.assertEqual(
            sampler.backend.shape["model_0"],
            (NSAMPLERS, NTEMPS, NWALKERS, 1, NDIM),
        )
        self.assertTrue(
            (sampler.backend.get_samplers_running() == True).all()  # noqa: E712
        )

        # each sampler's cold chain stays at beta = 1
        self.assertTrue(np.allclose(sampler.get_betas()[:, :, 0], 1.0))

        # index kwargs
        self.assertEqual(
            sampler.get_chain(sampler_index=1)["model_0"].shape,
            (nsteps, NTEMPS, NWALKERS, 1, NDIM),
        )
        self.assertEqual(
            sampler.get_chain(temp_index=0)["model_0"].shape,
            (nsteps, NSAMPLERS, NWALKERS, 1, NDIM),
        )

        # independent samplers produce different chains
        self.assertFalse(np.allclose(chain[:, 0], chain[:, 1]))

        # folded state reconstruction + grouped views
        last = sampler.get_last_sample()
        self.assertEqual(last.nsamplers, NSAMPLERS)
        branch = last.branches["model_0"]
        self.assertEqual(branch.shape, (NSAMPLERS * NTEMPS, NWALKERS, 1, NDIM))
        self.assertEqual(
            branch.coords_grouped.shape, (NSAMPLERS, NTEMPS, NWALKERS, 1, NDIM)
        )
        self.assertTrue(np.shares_memory(branch.coords_grouped, branch.coords))
        self.assertEqual(last.betas.shape, (NSAMPLERS * NTEMPS,))
        self.assertEqual(last.betas_grouped.shape, (NSAMPLERS, NTEMPS))
        self.assertTrue(
            np.array_equal(last.log_like_grouped.reshape(-1, NWALKERS), last.log_like)
        )

        # restart continues seamlessly
        sampler.run_mcmc(None, 2, progress=False)
        self.assertEqual(sampler.backend.iteration, nsteps + 2)

    def test_hdf_roundtrip(self):
        nsteps = 6
        fp = tempfile.mktemp(suffix=".h5")
        try:
            np.random.seed(9)
            sampler, priors = make_sampler(backend=fp)
            coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
            sampler.run_mcmc(coords, nsteps, progress=False)

            self.assertEqual(sampler.backend.nsamplers, NSAMPLERS)
            self.assertEqual(
                sampler.get_chain()["model_0"].shape,
                (nsteps, NSAMPLERS, NTEMPS, NWALKERS, 1, NDIM),
            )
            self.assertEqual(
                sampler.get_chain(sampler_index=2)["model_0"].shape,
                (nsteps, NTEMPS, NWALKERS, 1, NDIM),
            )
            self.assertEqual(
                sampler.backend.get_samplers_running().shape, (nsteps, NSAMPLERS)
            )

            with h5py.File(fp, "r") as f:
                self.assertEqual(f["mcmc"].attrs["nsamplers"], NSAMPLERS)
                self.assertEqual(
                    f["mcmc"]["chain"]["model_0"].shape,
                    (nsteps, NSAMPLERS, NTEMPS, NWALKERS, 1, NDIM),
                )

            # restart from file
            sampler2, _ = make_sampler(backend=HDFBackend(fp))
            sampler2.run_mcmc(None, 2, progress=False)
            self.assertEqual(sampler2.backend.iteration, nsteps + 2)
        finally:
            if os.path.exists(fp):
                os.remove(fp)

    def test_untempered_multi(self):
        # nsamplers > 1 with no tempering: folded axis is just nsamplers
        nsteps = 8
        np.random.seed(21)
        sampler, priors = make_sampler(tempered=False, nsamplers=NSAMPLERS)
        coords = priors.rvs(size=(NSAMPLERS, 1, NWALKERS))
        sampler.run_mcmc(coords, nsteps, progress=False)
        self.assertEqual(
            sampler.get_chain()["model_0"].shape,
            (nsteps, NSAMPLERS, 1, NWALKERS, 1, NDIM),
        )
        self.assertEqual(sampler.backend.accepted.shape, (NSAMPLERS, 1, NWALKERS))


class LegacyHDFFileTest(unittest.TestCase):
    """Files written before the sampler axis existed keep working (read + ns=1 restart)."""

    @staticmethod
    def _make_legacy_file(fp, nsteps=6):
        """Run a small ns=1 sampler, then rewrite its file into the legacy layout."""
        np.random.seed(17)
        sampler, priors = make_sampler(nsamplers=1, backend=fp)
        coords = priors.rvs(size=(NTEMPS, NWALKERS))
        sampler.run_mcmc(coords, nsteps, progress=False)

        with h5py.File(fp, "a") as f:
            g = f["mcmc"]
            del g.attrs["nsamplers"]
            del g["samplers_running"]
            for name, squeeze_axis in [
                ("log_like", 1),
                ("log_prior", 1),
                ("betas", 1),
                ("accepted", 0),
                ("swaps_accepted", 0),
            ]:
                data = g[name][...]
                data = np.squeeze(data, axis=squeeze_axis)
                del g[name]
                if name in ["accepted", "swaps_accepted"]:
                    g.create_dataset(name, data=data)
                else:
                    g.create_dataset(
                        name, data=data, maxshape=(None,) + data.shape[1:]
                    )
            for key in list(g["chain"]):
                for grp in ["chain", "inds"]:
                    data = np.squeeze(g[grp][key][...], axis=1)
                    del g[grp][key]
                    g[grp].create_dataset(
                        key, data=data, maxshape=(None,) + data.shape[1:]
                    )
        return nsteps

    def test_read_and_restart(self):
        fp = tempfile.mktemp(suffix=".h5")
        try:
            nsteps = self._make_legacy_file(fp)
            backend = HDFBackend(fp)

            self.assertEqual(backend.nsamplers, 1)
            self.assertFalse(backend._has_sampler_axis)
            self.assertEqual(
                backend.get_chain()["model_0"].shape,
                (nsteps, NTEMPS, NWALKERS, 1, NDIM),
            )
            self.assertEqual(backend.get_betas().shape, (nsteps, NTEMPS))
            self.assertEqual(backend.accepted.shape, (NTEMPS, NWALKERS))
            self.assertIsNone(backend.get_samplers_running())

            last = backend.get_last_sample()
            self.assertEqual(last.nsamplers, 1)

            # ns=1 restart on the legacy file keeps working, in the legacy layout
            sampler, _ = make_sampler(nsamplers=1, backend=HDFBackend(fp))
            sampler.run_mcmc(None, 2, progress=False)
            self.assertEqual(sampler.backend.iteration, nsteps + 2)
            with h5py.File(fp, "r") as f:
                self.assertEqual(
                    f["mcmc"]["chain"]["model_0"].shape[1:],
                    (NTEMPS, NWALKERS, 1, NDIM),
                )

            # ns>1 restart from a legacy file must raise
            with self.assertRaises(ValueError):
                make_sampler(nsamplers=NSAMPLERS, backend=HDFBackend(fp))
        finally:
            if os.path.exists(fp):
                os.remove(fp)


class SwapIsolationTest(unittest.TestCase):
    def test_swaps_stay_within_sampler(self):
        # tag every element with its sampler id; after many swap rounds no
        # element may have crossed to another sampler's rows
        np.random.seed(31)
        nsamplers, ntemps, nwalkers = 3, 4, 8
        tc = TemperatureControl(2, nwalkers, ntemps=ntemps, nsamplers=nsamplers)

        sampler_ids = np.repeat(np.arange(nsamplers), ntemps)
        x = {
            "model_0": np.tile(
                sampler_ids[:, None, None, None].astype(float),
                (1, nwalkers, 1, 2),
            )
        }
        for _ in range(50):
            logl = np.random.randn(nsamplers * ntemps, nwalkers)
            logp = np.zeros_like(logl)
            logP = tc.compute_log_posterior_tempered(logl, logp)
            (x, logP, logl, logp, _, _, _, _) = tc.temperature_swaps(
                x, logP, logl, logp
            )
            self.assertTrue(
                np.array_equal(
                    x["model_0"][:, :, 0, 0], np.tile(sampler_ids[:, None], (1, nwalkers))
                ),
                "coordinates crossed sampler boundaries in temperature swaps",
            )

    def test_frozen_samplers_untouched_by_swaps(self):
        np.random.seed(32)
        nsamplers, ntemps, nwalkers = 3, 4, 8
        tc = TemperatureControl(2, nwalkers, ntemps=ntemps, nsamplers=nsamplers)
        running = np.array([True, False, True])

        x = {"model_0": np.random.randn(nsamplers * ntemps, nwalkers, 1, 2)}
        frozen_before = x["model_0"][ntemps : 2 * ntemps].copy()
        # frozen samplers carry the fill value in logl: without masking this
        # would auto-accept every swap
        logl = np.random.randn(nsamplers * ntemps, nwalkers)
        logl[ntemps : 2 * ntemps] = -1e300
        logp = np.zeros_like(logl)
        logP = tc.compute_log_posterior_tempered(logl, logp)
        for _ in range(20):
            (x, logP, logl, logp, _, _, _, _) = tc.temperature_swaps(
                x, logP, logl, logp, samplers_running=running
            )
        self.assertTrue(
            np.array_equal(x["model_0"][ntemps : 2 * ntemps], frozen_before)
        )
        # swap counters for the frozen sampler stay zero
        self.assertTrue(np.all(tc.swaps_accepted[1] == 0))


class LadderAdaptationTest(unittest.TestCase):
    def test_per_sampler_adaptation_matches_single(self):
        # adaptation is deterministic given swaps_accepted: each sampler's
        # ladder must adapt exactly as a standalone single-sampler control
        nsamplers, ntemps, nwalkers = 3, 5, 10
        tc = TemperatureControl(2, nwalkers, ntemps=ntemps, nsamplers=nsamplers)
        swaps = np.array(
            [np.linspace(1, 8, ntemps - 1) * (s + 1) for s in range(nsamplers)]
        )
        tc.swaps_accepted = swaps.copy()
        tc.adapt_temps()

        for s in range(nsamplers):
            single = TemperatureControl(2, nwalkers, ntemps=ntemps)
            single.swaps_accepted = swaps[s].copy()
            single.adapt_temps()
            self.assertTrue(
                np.allclose(tc.betas_grouped[s], single.betas),
                f"sampler {s} ladder deviates from the single-sampler control",
            )

    def test_frozen_ladders_do_not_adapt(self):
        nsamplers, ntemps, nwalkers = 2, 4, 10
        tc = TemperatureControl(2, nwalkers, ntemps=ntemps, nsamplers=nsamplers)
        betas0 = tc.betas_grouped.copy()
        tc.swaps_accepted = np.tile(np.linspace(1, 8, ntemps - 1), (nsamplers, 1))
        tc.adapt_temps(samplers_running=np.array([True, False]))
        self.assertFalse(np.allclose(tc.betas_grouped[0], betas0[0]))
        self.assertTrue(np.array_equal(tc.betas_grouped[1], betas0[1]))


class SamplersRunningTest(unittest.TestCase):
    def test_freeze_mid_run(self):
        nsteps = 12
        freeze_after = 5
        np.random.seed(13)

        def update_fn(i, state, sampler):
            if i + 1 >= freeze_after:
                state.samplers_running[1] = False

        sampler, priors = make_sampler(
            update_fn=update_fn, update_iterations=1
        )
        coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
        state = State(
            {"model_0": coords[:, :, :, None, :]},
            nsamplers=NSAMPLERS,
            samplers_running=np.ones(NSAMPLERS, dtype=bool),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sampler.run_mcmc(state, nsteps, progress=False)

        sr = sampler.backend.get_samplers_running()
        self.assertTrue(sr[:freeze_after, 1].all())
        self.assertFalse(sr[-1, 1])

        chain = sampler.get_chain()["model_0"]
        ll = sampler.get_log_like()
        betas = sampler.get_betas()
        frozen_steps = np.where(~sr[:, 1])[0]
        first_frozen = frozen_steps[0]
        # frozen sampler: coords and betas bit-identical, logl filled
        self.assertTrue(
            np.all(chain[first_frozen:, 1] == chain[first_frozen, 1])
        )
        self.assertTrue(
            np.array_equal(
                betas[first_frozen:, 1],
                np.tile(betas[first_frozen, 1], (len(frozen_steps), 1)),
            )
        )
        # frozen mid-run: the last finite likelihoods are retained, unchanged
        self.assertTrue(np.all(ll[first_frozen:, 1] == ll[first_frozen, 1]))
        # running samplers keep moving
        self.assertFalse(np.all(chain[first_frozen:, 0] == chain[first_frozen, 0]))

    def test_frozen_rows_skip_likelihood(self):
        rows_seen = []

        def counting_log_like(x, mu, invcov):
            rows_seen.append(1)
            diff = x - mu
            return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()

        np.random.seed(14)
        priors = make_priors()
        sampler = EnsembleSampler(
            NWALKERS,
            NDIM,
            counting_log_like,
            priors,
            args=[np.zeros(NDIM), np.eye(NDIM)],
            tempering_kwargs=dict(ntemps=NTEMPS),
            nsamplers=NSAMPLERS,
        )
        coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
        state = State(
            {"model_0": coords[:, :, :, None, :]},
            nsamplers=NSAMPLERS,
            samplers_running=np.array([True, False, True]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sampler.run_mcmc(state, 3, progress=False)

        # per-step proposals evaluate at most the running rows' walkers
        # (2 of 3 samplers): total calls must be well below the all-running count
        max_all_running = (3 + 1) * NSAMPLERS * NTEMPS * NWALKERS
        expected_max = (3 + 1) * 2 * NTEMPS * NWALKERS
        self.assertLessEqual(len(rows_seen), expected_max)
        self.assertLess(len(rows_seen), max_all_running)


class PriorTransformTest(unittest.TestCase):
    def test_identity_transform_matches_no_transform(self):
        class IdentityTransform:
            def transform_to_prior_basis(self, coords, running_idx):
                return

            def adjust_logp(self, logp, running_idx):
                return logp

        chains = []
        for transform in [None, IdentityTransform()]:
            np.random.seed(19)
            sampler, priors = make_sampler(prior_transform_fn=transform)
            coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
            sampler.run_mcmc(coords, 8, progress=False)
            chains.append(sampler.get_chain()["model_0"])
        self.assertTrue(np.array_equal(chains[0], chains[1]))

    def test_per_sampler_bands(self):
        lo = np.array([0.0, 10.0, 20.0])
        hi = np.array([1.0, 12.0, 24.0])
        priors = ProbDistContainer(
            {0: uniform_dist(0.0, 1.0), 1: uniform_dist(-5.0, 5.0)}
        )

        np.random.seed(23)
        sampler = EnsembleSampler(
            NWALKERS,
            2,
            lambda x: 0.0,
            priors,
            tempering_kwargs=dict(ntemps=NTEMPS),
            nsamplers=NSAMPLERS,
            prior_transform_fn=UnitCubeTransform(lo, hi),
        )
        c0 = lo[:, None, None] + np.random.uniform(
            0, 1, size=(NSAMPLERS, NTEMPS, NWALKERS)
        ) * (hi - lo)[:, None, None]
        c1 = np.random.uniform(-5, 5, size=(NSAMPLERS, NTEMPS, NWALKERS))
        coords = np.stack([c0, c1], axis=-1)
        sampler.run_mcmc(coords, 25, burn=5, progress=False)

        param0 = sampler.get_chain()["model_0"][..., 0, 0]
        for s in range(NSAMPLERS):
            self.assertGreaterEqual(param0[:, s].min(), lo[s])
            self.assertLessEqual(param0[:, s].max(), hi[s])

    def test_guards(self):
        priors = make_priors()
        transform = UnitCubeTransform(np.zeros(NSAMPLERS), np.ones(NSAMPLERS))
        common = dict(
            args=[np.zeros(NDIM), np.eye(NDIM)],
            tempering_kwargs=dict(ntemps=NTEMPS),
            nsamplers=NSAMPLERS,
            prior_transform_fn=transform,
        )
        with self.assertRaises(ValueError):
            EnsembleSampler(
                NWALKERS, NDIM, gaussian_log_like, priors, nleaves_max=2, **common
            )
        with self.assertRaises(ValueError):
            EnsembleSampler(
                NWALKERS,
                NDIM,
                gaussian_log_like,
                priors,
                rj_moves=True,
                nleaves_min=0,
                **common,
            )
        # plotting guard, independent of prior transform
        with self.assertRaises(ValueError):
            EnsembleSampler(
                NWALKERS,
                NDIM,
                gaussian_log_like,
                priors,
                args=[np.zeros(NDIM), np.eye(NDIM)],
                tempering_kwargs=dict(ntemps=NTEMPS),
                nsamplers=NSAMPLERS,
                plot_iterations=5,
            )


class GroupStretchIsolationTest(unittest.TestCase):
    def test_same_row_friends_stay_isolated(self):
        # a GroupStretchMove whose friends come from the same folded row is
        # per-sampler independent by construction; with disjoint per-sampler
        # prior bands, any cross-sampler leakage (moves or swaps) would show
        # up as out-of-band samples
        class SameRowFriends(GroupStretchMove):
            def setup_friends(self, branches):
                self.friends = {
                    name: b.coords.copy() for name, b in branches.items()
                }

            def find_friends(self, name, s, s_inds=None, branch_supps=None):
                nt, nw = s.shape[:2]
                friend_idx = np.random.randint(
                    self.friends[name].shape[1], size=(nt, nw)
                )
                rows = np.repeat(np.arange(nt)[:, None], nw, axis=1)
                return self.friends[name][rows, friend_idx]

            def fix_friends(self, branches):
                return

        lo = np.array([0.0, 10.0, 20.0])
        hi = np.array([1.0, 12.0, 24.0])
        priors = ProbDistContainer(
            {0: uniform_dist(0.0, 1.0), 1: uniform_dist(-5.0, 5.0)}
        )

        np.random.seed(29)
        sampler = EnsembleSampler(
            NWALKERS,
            2,
            lambda x: 0.0,
            priors,
            tempering_kwargs=dict(ntemps=NTEMPS),
            nsamplers=NSAMPLERS,
            moves=SameRowFriends(nfriends=NWALKERS, n_iter_update=5),
            prior_transform_fn=UnitCubeTransform(lo, hi),
        )
        # sampler structure is exposed on the move
        move = sampler.moves[0]
        self.assertEqual(move.nsamplers, NSAMPLERS)
        self.assertEqual(move.ntemps_per_sampler, NTEMPS)
        self.assertTrue(
            np.array_equal(
                move.sampler_id_rows, np.repeat(np.arange(NSAMPLERS), NTEMPS)
            )
        )

        c0 = lo[:, None, None] + np.random.uniform(
            0, 1, size=(NSAMPLERS, NTEMPS, NWALKERS)
        ) * (hi - lo)[:, None, None]
        c1 = np.random.uniform(-5, 5, size=(NSAMPLERS, NTEMPS, NWALKERS))
        coords = np.stack([c0, c1], axis=-1)
        sampler.run_mcmc(coords, 20, progress=False)

        param0 = sampler.get_chain()["model_0"][..., 0, 0]
        for s in range(NSAMPLERS):
            self.assertGreaterEqual(param0[:, s].min(), lo[s])
            self.assertLessEqual(param0[:, s].max(), hi[s])


class RJWithNSamplersTest(unittest.TestCase):
    def test_rj_runs_and_stores(self):
        nleaves_max = 3
        ndim = 2
        nsteps = 8
        np.random.seed(41)

        def flat_like(params_list):
            return 0.0

        priors = ProbDistContainer(
            {0: uniform_dist(-5, 5), 1: uniform_dist(-5, 5)}
        )
        sampler = EnsembleSampler(
            NWALKERS,
            ndim,
            flat_like,
            priors,
            tempering_kwargs=dict(ntemps=NTEMPS),
            nsamplers=NSAMPLERS,
            nleaves_max=nleaves_max,
            nleaves_min=0,
            moves=GaussianMove({"model_0": np.eye(ndim) * 0.04}),
            rj_moves=True,
            fill_zero_leaves_val=0.0,
        )
        coords = priors.rvs(
            size=(NSAMPLERS, NTEMPS, NWALKERS, nleaves_max)
        ).reshape(NSAMPLERS, NTEMPS, NWALKERS, nleaves_max, ndim)
        inds = np.random.rand(NSAMPLERS, NTEMPS, NWALKERS, nleaves_max) < 0.5
        state = State(
            {"model_0": coords}, inds={"model_0": inds}, nsamplers=NSAMPLERS
        )
        sampler.run_mcmc(state, nsteps, progress=False)

        stored_inds = sampler.get_inds()["model_0"]
        self.assertEqual(
            stored_inds.shape, (nsteps, NSAMPLERS, NTEMPS, NWALKERS, nleaves_max)
        )
        nleaves = sampler.get_nleaves()["model_0"]
        self.assertEqual(nleaves.shape, (nsteps, NSAMPLERS, NTEMPS, NWALKERS))
        self.assertTrue(nleaves.min() >= 0 and nleaves.max() <= nleaves_max)
        self.assertEqual(
            sampler.backend.rj_accepted.shape, (NSAMPLERS, NTEMPS, NWALKERS)
        )


class StatisticalSanityTest(unittest.TestCase):
    def test_independent_samplers_agree(self):
        # two identical Gaussian targets: per-sampler posterior moments must
        # agree with each other and with a standalone single-sampler run
        nsteps, nwalkers, ndim = 400, 32, 2
        np.random.seed(101)
        sampler, priors = make_sampler(
            nsamplers=2, ntemps=1, nwalkers=nwalkers, ndim=ndim, tempered=False
        )
        coords = priors.rvs(size=(2, 1, nwalkers))
        sampler.run_mcmc(coords, nsteps, burn=100, progress=False)
        chain = sampler.get_chain(discard=100)["model_0"]

        means = chain.reshape(-1, 2, nwalkers * 1, ndim).mean(axis=(0, 2))
        stds = chain.reshape(-1, 2, nwalkers * 1, ndim).std(axis=(0, 2))

        np.random.seed(202)
        single, priors1 = make_sampler(
            nsamplers=1, ntemps=1, nwalkers=nwalkers, ndim=ndim, tempered=False
        )
        single.run_mcmc(priors1.rvs(size=(1, nwalkers)), nsteps, burn=100, progress=False)
        chain1 = single.get_chain(discard=100)["model_0"]
        mean1 = chain1.reshape(-1, ndim).mean(axis=0)
        std1 = chain1.reshape(-1, ndim).std(axis=0)

        for s in range(2):
            self.assertTrue(np.allclose(means[s], mean1, atol=0.15))
            self.assertTrue(np.allclose(stds[s], std1, atol=0.15))


class DiagnosticsTest(unittest.TestCase):
    def test_evidence_and_guards(self):
        nsteps = 10
        np.random.seed(51)
        sampler, priors = make_sampler()
        # fixed ladders for the evidence estimate
        sampler.temperature_control.adaptive = False
        coords = priors.rvs(size=(NSAMPLERS, NTEMPS, NWALKERS))
        sampler.run_mcmc(coords, nsteps, progress=False)

        logZ, dlogZ = sampler.backend.get_evidence_estimate()
        self.assertEqual(logZ.shape, (NSAMPLERS,))
        self.assertEqual(dlogZ.shape, (NSAMPLERS,))
        self.assertTrue(np.isfinite(logZ).all())

        with self.assertRaises(ValueError):
            sampler.backend.get_autocorr_time()
        with self.assertRaises(NotImplementedError):
            sampler.backend.get_gelman_rubin_convergence_diagnostic(doprint=False)


if __name__ == "__main__":
    unittest.main()
