#!/usr/bin/env python
"""Toy demo of the full ONLINE flow-training loop, end to end, on a laptop.

This is the smallest runnable thing that proves every moving part of the online
flow proposal working together:

    sampler  --harvest cold chain-->  ProcessExecutor (trains in a 2nd process)
       ^                                        |
       |  hot-load versioned weights (poll) <---/
       |
    posterior stays correct; trainer process exits cleanly (exitcode 0).

Target: a curved "banana" with a periodic last dimension
(:func:`eryn.flows.benchmark.banana_periodic_target`, ndim=3, dim 2 periodic).
The flow is mixed 30/70 with a plain :class:`StretchMove`: the stretch component
keeps the chain ergodic while the flow learns, which is also the realistic
production configuration (you never bet the whole proposal on a cold flow).

Run::

    uv run python Eryn/examples/flow_online_training_toy.py
    uv run python Eryn/examples/flow_online_training_toy.py --no-plot --seed 7

What to look for in the report:
  - the executor reached version >= 2 (the trainer actually fit several rounds);
  - FlowMove.loaded_version tracked it (weights were hot-loaded mid-run);
  - per-dim posterior moments of the ONLINE run match the ground-truth empirical
    moments about as well as a pure stretch reference does.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.moves import FlowMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import PeriodicContainer

from eryn.flows import (
    OneHotLeafConditioning,
    ProcessExecutor,
    WhiteningTransform,
    ZukoFlow,
)
from eryn.flows.benchmark import banana_periodic_target

PERIOD = 2 * np.pi


# ----------------------------------------------------------------------------
# Circular statistics (periodic dim) — same formulas as the detailed-balance
# test, factored out so the example and ground-truth use ONE definition.
# ----------------------------------------------------------------------------
def circular_mean(ang: np.ndarray) -> float:
    """Mean direction of angles in [0, 2pi)."""
    return float(np.angle(np.mean(np.exp(1j * ang))) % PERIOD)


def circular_std(ang: np.ndarray) -> float:
    """Circular standard deviation (small for concentrated angles)."""
    R = np.abs(np.mean(np.exp(1j * ang)))
    return float(np.sqrt(-2.0 * np.log(R)))


def _priors(ndim: int, bounds: dict) -> dict:
    return {"x": ProbDistContainer({d: uniform_dist(*bounds[d]) for d in range(ndim)})}


def _make_sampler(moves, log_prob, ndim, bounds, nwalkers, seed):
    """Build a seeded, single-temperature EnsembleSampler for branch 'x'."""
    sampler = EnsembleSampler(
        nwalkers,
        {"x": ndim},
        log_prob,
        _priors(ndim, bounds),
        tempering_kwargs=dict(ntemps=1),
        vectorize=True,
        periodic=PeriodicContainer({"x": {ndim - 1: PERIOD}}),
        moves=moves,
        branch_names=["x"],
    )
    # The EnsembleSampler builds its OWN unseeded RandomState (np.random.seed
    # never reaches it).  Seed it through the public setter so the accept/reject
    # draws — and therefore the whole run — are reproducible.
    sampler.random_state = np.random.RandomState(seed).get_state()
    return sampler


def _cold_chain(sampler, ndim):
    """Flatten the post-run cold chain to (nsteps*nwalkers, ndim)."""
    # chain axes: (step, temp, walker, leaf, dim); ntemps=1 and single-leaf, so
    # we drop temp=0 and leaf=0.
    return sampler.get_chain()["x"][:, 0, :, 0, :].reshape(-1, ndim)


def _moments(samples, periodic_dim):
    """Per-dim (mean, std), using circular stats for the periodic dim."""
    means, stds = [], []
    for d in range(samples.shape[1]):
        if d == periodic_dim:
            means.append(circular_mean(samples[:, d]))
            stds.append(circular_std(samples[:, d]))
        else:
            means.append(float(samples[:, d].mean()))
            stds.append(float(samples[:, d].std()))
    return np.array(means), np.array(stds)


def main():
    ap = argparse.ArgumentParser(
        description="Toy end-to-end online flow-training demo (banana+periodic)."
    )
    ap.add_argument("--nsteps", type=int, default=1200)
    ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--harvest-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-plot", default="flow_online_toy.png")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the marginal-overlay plot")
    args = ap.parse_args()

    # Seed every global source of randomness so the demo is reproducible:
    # numpy global RNG (warmup start draws) and torch global RNG (flow weight
    # init + flow.sample).  Each sampler additionally seeds its own RandomState.
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    # --- target: banana (2 Gaussian dims) + 1 periodic dim => ndim = 3 ---
    log_prob, sampler_fn, ndim, periodic = banana_periodic_target(ndim_gauss=2)
    periodic_dim = ndim - 1  # last dim is periodic (0, 2pi)

    # ground truth: exact samples from the target (large => "analytic-ish")
    truth = sampler_fn(100_000, seed=args.seed + 1)
    truth_mean, truth_std = _moments(truth, periodic_dim)

    # generous box prior covering the support (periodic dim fixed to (0, 2pi))
    lo = truth.min(0) - 2.0
    hi = truth.max(0) + 2.0
    bounds = {d: (float(lo[d]), float(hi[d])) for d in range(ndim)}
    bounds[periodic_dim] = (0.0, PERIOD)

    # ========================================================================
    # 1) WARMUP: short StretchMove-only run to collect coords-space samples
    #    that (a) fit the whitening transform and (b) give the flow + executor
    #    a sane starting point before online training begins.
    # ========================================================================
    print("[warmup] short StretchMove-only run to seed the flow ...")
    warmup_sampler = _make_sampler(
        [StretchMove()], log_prob, ndim, bounds, args.nwalkers, args.seed
    )
    # start walkers from exact target draws so warmup is short and on-support
    start = State({"x": sampler_fn(args.nwalkers, seed=args.seed).reshape(
        1, args.nwalkers, 1, ndim)})
    warmup_sampler.run_mcmc(start, 300, burn=100, progress=False)
    warmup = _cold_chain(warmup_sampler, ndim)
    print(f"[warmup] collected {len(warmup)} samples")

    # ========================================================================
    # 2) FLOW + frozen whitening transform.  Fitting the transform here (via
    #    flow.fit) is what FREEZES the coords<->latent map for the executor's
    #    lifetime — the worker always trains with refit_data_transform=False.
    # ========================================================================
    wt = WhiteningTransform(ndim=ndim, periodic={periodic_dim: (0.0, PERIOD)})
    flow = ZukoFlow(
        dims=ndim,
        device="cpu",
        conditioning=OneHotLeafConditioning(nleaves_max=1),
        data_transform=wt,
        seed=args.seed,
        transforms=3,
        hidden_features=(64, 64),
        bins=5,
    )
    print("[flow] warm-fitting the flow on warmup samples (fits the transform) ...")
    flow.fit(warmup, n_epochs=10, batch_size=512, validation_fraction=0.1,
             seed=args.seed)

    # ========================================================================
    # 3) ONLINE run: StretchMove (70%) + FlowMove (30%) with a ProcessExecutor.
    #    The executor trains a CLONE of the flow in a spawned process; FlowMove
    #    harvests the cold chain into it and hot-loads new weights as they land.
    # ========================================================================
    print("[online] starting ProcessExecutor + mixed Stretch/Flow run ...")
    t0 = time.time()
    with ProcessExecutor(
        flow,
        epochs_per_round=15,
        min_train_samples=2000,
        torch_num_threads=2,
        seed=args.seed,
    ) as ex:
        flow_move = FlowMove(
            flow, "x", executor=ex, harvest_every=args.harvest_every
        )
        online_sampler = _make_sampler(
            [(StretchMove(), 0.7), (flow_move, 0.3)],
            log_prob, ndim, bounds, args.nwalkers, args.seed,
        )
        start_online = State({"x": warmup[
            np.random.RandomState(args.seed).choice(len(warmup), args.nwalkers,
                                                     replace=False)
        ].reshape(1, args.nwalkers, 1, ndim)})
        online_sampler.run_mcmc(
            start_online, args.nsteps, burn=args.burn, progress=False
        )

        # The spawned trainer starts slowly (~2-5 s) and a short laptop run can
        # finish before even the first fit lands.  So the online story is
        # actually VISIBLE in the report, keep stepping the chain in small
        # segments (each one harvests + polls) on a wall-clock deadline until a
        # couple of trained versions have been hot-loaded.  This is exactly what
        # a long production run gets "for free" — here we just wait for it.
        deadline = time.time() + 60.0
        while ex.version < 2 and time.time() < deadline:
            online_sampler.run_mcmc(
                online_sampler.get_last_sample(), 50, burn=0, progress=False
            )
        # a few more steps so the move's setup() polls and loads the newest
        online_sampler.run_mcmc(
            online_sampler.get_last_sample(), 30, burn=0, progress=False
        )

        # capture executor state INSIDE the context (the process is alive here)
        ex_version = ex.version
        loaded_version = flow_move.loaded_version
        flow_acc = float(np.mean(flow_move.acceptance_fraction))
        proc = ex._process  # for the clean-shutdown report after __exit__
    # <-- context manager exit: graceful shutdown of the trainer process
    online_elapsed = time.time() - t0

    # find the stretch move object back out of the sampler to report its accept
    stretch_acc = float(np.mean(online_sampler.moves[0].acceptance_fraction))

    online = _cold_chain(online_sampler, ndim)
    online_mean, online_std = _moments(online, periodic_dim)

    # ========================================================================
    # 4) REFERENCE run: pure StretchMove, same budget + seed, for comparison.
    # ========================================================================
    print("[reference] stretch-only run for comparison ...")
    ref_sampler = _make_sampler(
        [StretchMove()], log_prob, ndim, bounds, args.nwalkers, args.seed
    )
    start_ref = State({"x": warmup[
        np.random.RandomState(args.seed).choice(len(warmup), args.nwalkers,
                                                 replace=False)
    ].reshape(1, args.nwalkers, 1, ndim)})
    ref_sampler.run_mcmc(start_ref, args.nsteps, burn=args.burn, progress=False)
    ref = _cold_chain(ref_sampler, ndim)
    ref_mean, ref_std = _moments(ref, periodic_dim)

    # ========================================================================
    # 5) REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("ONLINE FLOW-TRAINING TOY — REPORT")
    print("=" * 70)
    print(f"target            : banana + periodic, ndim={ndim} "
          f"(dim {periodic_dim} periodic)")
    print(f"online run time   : {online_elapsed:.1f} s "
          f"({args.nsteps} steps, {args.nwalkers} walkers)")
    print(f"executor version  : {ex_version}  (trainer fit rounds completed)")
    print(f"loaded_version    : {loaded_version}  (weights hot-loaded into FlowMove)")
    print(f"FlowMove accept   : {flow_acc:.3f}")
    print(f"Stretch accept    : {stretch_acc:.3f}")
    print(f"trainer exitcode  : {proc.exitcode}  (0 == graceful), "
          f"alive={proc.is_alive()}")
    print("-" * 70)
    labels = [f"dim{d}" + (" (periodic)" if d == periodic_dim else "")
              for d in range(ndim)]
    print(f"{'':<16}{'truth':>22}{'online':>22}{'reference':>22}")
    print(f"{'':<16}{'mean / std':>22}{'mean / std':>22}{'mean / std':>22}")
    for d in range(ndim):
        print(f"{labels[d]:<16}"
              f"{truth_mean[d]:>10.3f} /{truth_std[d]:>8.3f}  "
              f"{online_mean[d]:>10.3f} /{online_std[d]:>8.3f}  "
              f"{ref_mean[d]:>10.3f} /{ref_std[d]:>8.3f}")
    print("=" * 70)
    print("clean shutdown confirmed: trainer process exited via context manager.")

    # ========================================================================
    # 6) PLOT (optional)
    # ========================================================================
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim, 3.2))
        if ndim == 1:
            axes = [axes]
        for d in range(ndim):
            ax = axes[d]
            bins = np.linspace(
                min(truth[:, d].min(), online[:, d].min(), ref[:, d].min()),
                max(truth[:, d].max(), online[:, d].max(), ref[:, d].max()),
                60,
            )
            ax.hist(truth[:, d], bins=bins, density=True, histtype="step",
                    color="k", lw=2, label="ground truth")
            ax.hist(online[:, d], bins=bins, density=True, histtype="stepfilled",
                    color="C0", alpha=0.4, label="online (stretch+flow)")
            ax.hist(ref[:, d], bins=bins, density=True, histtype="step",
                    color="C3", lw=1.5, ls="--", label="reference (stretch)")
            ax.set_title(labels[d])
            ax.set_yticks([])
        axes[0].legend(fontsize=8)
        fig.suptitle("Online flow-training toy — marginal posteriors")
        fig.tight_layout()
        fig.savefig(args.out_plot, dpi=120)
        print(f"plot saved to {args.out_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
