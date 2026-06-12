# tests/test_flow_online_training.py
"""End-to-end slow gate for the ONLINE flow-training loop.

This is the integration counterpart to ``test_flow_detailed_balance.py``: where
that test froze a trained flow and checked detailed balance, this one drives the
*whole* online machinery in anger and proves it holds together:

    EnsembleSampler  --harvest cold chain-->  ProcessExecutor (spawned trainer)
          ^                                            |
          |  hot-load versioned weights (poll)  <------/

A :class:`FlowMove` (mixed 30/70 with a plain :class:`StretchMove`, so the chain
stays ergodic while the flow learns) harvests the cold chain into a spawned
:class:`ProcessExecutor`, which trains a clone in a *second process* and hands
back versioned weights that the move hot-loads mid-run.  The single test asserts
three properties:

  (a) weights advance + hot-load   — executor reaches version >= 2 and the move
                                      loads them (loaded_version >= 2);
  (b) posterior correct            — the pooled cold chain matches ground-truth
                                      empirical moments (including the periodic
                                      dim's circular moments);
  (c) clean shutdown, no zombie    — the trainer process exits with exitcode 0
                                      (graceful — neither terminate nor kill
                                      fired), a second shutdown is a no-op, and
                                      no zombie is left in active_children().

It also asserts NO ``TrainerError`` escaped: the test passing means none was
raised, and a final wrapped ``latest_weights()`` poll fails with a clear message
if the trainer died.

spawn re-imports this module in each child as a NON-main module, so everything
at module level here is import-safe (only defs + constants).  Child startup is
slow (~2-5 s), so every wait is a wall-clock DEADLINE loop, never a tight sleep.
"""
import multiprocessing
import time

import numpy as np
import pytest

# Module-level guards: the whole file requires the optional 'flow' extra.
pytest.importorskip("torch")
pytest.importorskip("zuko")

import torch  # noqa: E402

from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.moves import FlowMove, StretchMove  # noqa: E402
from eryn.prior import ProbDistContainer, uniform_dist  # noqa: E402
from eryn.state import State  # noqa: E402
from eryn.utils import PeriodicContainer  # noqa: E402

from eryn.flows import (  # noqa: E402
    OneHotLeafConditioning,
    ProcessExecutor,
    WhiteningTransform,
    ZukoFlow,
)
from eryn.flows.benchmark import banana_periodic_target  # noqa: E402
from eryn.flows.executors import TrainerError  # noqa: E402

PERIOD = 2 * np.pi

# Trimmed budget (vs the example's defaults) so the whole loop runs in ~3 min.
NWALKERS = 48
NSTEPS = 800
BURN = 200
WARMUP_STEPS = 250
WARMUP_BURN = 80
HARVEST_EVERY = 5
SEED = 20240611

# Generous wall-clock deadline for "wait until the trainer produced something".
# Child startup is ~2-5 s on macOS CPU; we never tight-loop.
_POLL_DEADLINE_S = 60.0


# ---------------------------------------------------------------------------
# Circular statistics for the periodic dim — same formulas as
# test_flow_detailed_balance.py (a bias here would be a periodic-Jacobian bug).
# ---------------------------------------------------------------------------
def _circular_mean(ang: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * ang))) % PERIOD)


def _circular_std(ang: np.ndarray) -> float:
    R = np.abs(np.mean(np.exp(1j * ang)))
    return float(np.sqrt(-2.0 * np.log(R)))


def _build_sampler(moves, log_prob, ndim, bounds, seed):
    """Seeded, single-temperature EnsembleSampler for branch 'x'."""
    sampler = EnsembleSampler(
        NWALKERS,
        {"x": ndim},
        log_prob,
        {"x": ProbDistContainer(
            {d: uniform_dist(*bounds[d]) for d in range(ndim)}
        )},
        tempering_kwargs=dict(ntemps=1),
        vectorize=True,
        periodic=PeriodicContainer({"x": {ndim - 1: PERIOD}}),
        moves=moves,
        branch_names=["x"],
    )
    # WHY seed via the setter: EnsembleSampler builds its OWN *unseeded*
    # RandomState (np.random.seed never reaches it), so without this the
    # accept/reject draws — and therefore version timing and the posterior —
    # would be nondeterministic and the gate would flake.
    sampler.random_state = np.random.RandomState(seed).get_state()
    return sampler


def _cold_chain(sampler, ndim):
    """Flatten the post-run cold chain to (nsteps*nwalkers, ndim).

    Chain axes are (step, temp, walker, leaf, dim); ntemps=1 and single-leaf,
    so dropping temp=0 and leaf=0 is exact.
    """
    return sampler.get_chain()["x"][:, 0, :, 0, :].reshape(-1, ndim)


@pytest.mark.slow
def test_online_training_full_loop():
    # WHY save/restore the global RNG states: this test reseeds the global numpy
    # and torch RNGs (flow init + flow.sample read the global torch RNG), and
    # leaking a reseeded global state would make unrelated tests in the suite
    # order-sensitive.  Mirror the detailed-balance test's hygiene exactly.
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        # Seed every global source: numpy (warmup start draws + start selection)
        # and torch (flow weight init + flow.sample).  Each sampler additionally
        # seeds its own RandomState in _build_sampler.
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # --- target: banana (2 Gaussian dims) + 1 periodic dim => ndim = 3 ---
        log_prob, sampler_fn, ndim, periodic = banana_periodic_target(ndim_gauss=2)
        periodic_dim = ndim - 1

        # ground truth: large exact draw → empirical moments are the reference.
        truth = sampler_fn(200_000, seed=SEED + 1)
        truth_mean = np.array([
            _circular_mean(truth[:, d]) if d == periodic_dim
            else float(truth[:, d].mean())
            for d in range(ndim)
        ])
        truth_std = np.array([
            _circular_std(truth[:, d]) if d == periodic_dim
            else float(truth[:, d].std())
            for d in range(ndim)
        ])

        # generous box prior over the support; periodic dim fixed to (0, 2pi).
        lo = truth.min(0) - 2.0
        hi = truth.max(0) + 2.0
        bounds = {d: (float(lo[d]), float(hi[d])) for d in range(ndim)}
        bounds[periodic_dim] = (0.0, PERIOD)

        # --- warmup: short stretch-only run to seed the flow + fit the transform
        warmup_sampler = _build_sampler(
            [StretchMove()], log_prob, ndim, bounds, SEED
        )
        warmup_sampler.run_mcmc(
            State({"x": sampler_fn(NWALKERS, seed=SEED).reshape(
                1, NWALKERS, 1, ndim)}),
            WARMUP_STEPS, burn=WARMUP_BURN, progress=False,
        )
        warmup = _cold_chain(warmup_sampler, ndim)

        # --- flow + frozen whitening transform (fit here freezes the map) ---
        wt = WhiteningTransform(ndim=ndim, periodic={periodic_dim: (0.0, PERIOD)})
        flow = ZukoFlow(
            dims=ndim,
            device="cpu",
            conditioning=OneHotLeafConditioning(nleaves_max=1),
            data_transform=wt,
            seed=SEED,
            transforms=3,
            hidden_features=(64, 64),
            bins=5,
        )
        flow.fit(warmup, n_epochs=10, batch_size=512,
                 validation_fraction=0.1, seed=SEED)

        proc = None  # captured inside the context for the post-exit assertions
        with ProcessExecutor(
            flow,
            epochs_per_round=10,
            min_train_samples=1500,
            torch_num_threads=2,
            seed=SEED,
        ) as ex:
            flow_move = FlowMove(
                flow, "x", executor=ex, harvest_every=HARVEST_EVERY
            )
            sampler = _build_sampler(
                [(StretchMove(), 0.7), (flow_move, 0.3)],
                log_prob, ndim, bounds, SEED,
            )
            start = State({"x": warmup[
                np.random.RandomState(SEED).choice(
                    len(warmup), NWALKERS, replace=False)
            ].reshape(1, NWALKERS, 1, ndim)})
            sampler.run_mcmc(start, NSTEPS, burn=BURN, progress=False)

            # ----------------------------------------------------------------
            # (a) weights advance + hot-load.
            # The spawned trainer starts slowly (~2-5 s), so the main run may
            # finish before 2 fit rounds land.  Drive the chain forward in short
            # continuation segments (which keep harvesting + polling) on a
            # wall-clock deadline until the executor reports version >= 2.
            # ----------------------------------------------------------------
            t0 = time.monotonic()
            while ex.version < 2 and time.monotonic() < t0 + _POLL_DEADLINE_S:
                sampler.run_mcmc(
                    sampler.get_last_sample(), 50, burn=0, progress=False
                )
            assert ex.version >= 2, (
                f"trainer never reached version 2 within {_POLL_DEADLINE_S}s "
                f"(reached version {ex.version}); the spawned trainer is too "
                "slow at these budgets — report timings rather than relaxing."
            )

            # A few more steps so the move's setup() polls and hot-loads the
            # newest weights into its own flow.
            sampler.run_mcmc(
                sampler.get_last_sample(), 30, burn=0, progress=False
            )
            assert flow_move.loaded_version >= 2, (
                f"FlowMove hot-load lagged: loaded_version="
                f"{flow_move.loaded_version}, executor version={ex.version}."
            )

            # Explicit final poll, wrapped: if the trainer died, surface it as a
            # clear assertion rather than a bare TrainerError from teardown.
            try:
                ex.latest_weights()
            except TrainerError as exc:  # pragma: no cover - failure path
                pytest.fail(f"trainer raised TrainerError during the run: {exc}")

            # ----------------------------------------------------------------
            # (b) posterior correct — pooled post-burn cold chain of the ONLINE
            # run vs ground-truth empirical moments.
            # ----------------------------------------------------------------
            online = _cold_chain(sampler, ndim)
            for d in range(ndim):
                if d == periodic_dim:
                    om, os_ = _circular_mean(online[:, d]), _circular_std(online[:, d])
                    dmean = abs(((om - truth_mean[d] + np.pi) % PERIOD) - np.pi)
                else:
                    om, os_ = float(online[:, d].mean()), float(online[:, d].std())
                    dmean = abs(om - truth_mean[d])
                dstd = abs(os_ - truth_std[d])
                assert dmean < 0.15, (
                    f"dim {d} mean off: online={om:.4f} truth={truth_mean[d]:.4f} "
                    f"|delta|={dmean:.4f} >= 0.15"
                )
                assert dstd < 0.15, (
                    f"dim {d} std off: online={os_:.4f} truth={truth_std[d]:.4f} "
                    f"|delta|={dstd:.4f} >= 0.15"
                )

            # capture the live process handle for the clean-shutdown checks
            proc = ex._process
        # <-- context manager exit: graceful shutdown of the trainer process.

        # --------------------------------------------------------------------
        # (c) clean shutdown, no zombie.
        # exitcode 0 means the worker stopped on the sentinel/stop-event and
        # neither terminate() (-15) nor kill() (-9) had to fire.
        # --------------------------------------------------------------------
        assert proc is not None
        assert not proc.is_alive(), "trainer process still alive after shutdown"
        assert proc.exitcode == 0, (
            f"trainer did not exit gracefully: exitcode={proc.exitcode} "
            "(0 expected; a negative code means terminate/kill had to fire)"
        )
        ex.shutdown()  # idempotent second call: must be a no-op, no raise
        assert proc not in multiprocessing.active_children(), (
            "trainer left a zombie in multiprocessing.active_children()"
        )
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
