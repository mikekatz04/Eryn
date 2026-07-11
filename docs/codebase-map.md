_Last mapped: 1a1260a + nsamplers axis (2026-07-10) · regenerate when structure changes_

# Eryn codebase map

## 1. What this is

Eryn (`src/eryn/`, package `eryn`) is a general-purpose MCMC sampler library:
ensemble (affine-invariant, emcee-style) sampling with optional parallel
tempering, reversible-jump (trans-dimensional / variable-source-count)
moves, delayed rejection, multiple-try moves, and a vectorized NUTS
implementation. It is the sampling engine LISAanalysistools' global-fit
layer runs on. Unlike the other LISA Analysis Tools repos it ships **no compiled code
of its own** — it is pure Python/NumPy, with an optional CuPy path in a
few move/state classes and a GPU compute backend borrowed from
`gpubackendtools` (GBT) for `ParaEnsembleSampler` only. There is no
CUDA-kernel layer to document here; the interesting architecture is the
sampler/move/backend/state object model.

## 2. Layout

| Path | Role |
|---|---|
| `src/eryn/ensemble.py` | `EnsembleSampler` — the main sampler class, one-branch or multi-branch, RJ-capable, single-process (`pool`-parallel likelihood calls). `nsamplers=N` runs N fully independent ensembles at once with batched likelihood calls (see §5). |
| `src/eryn/paraensemble.py` | **DEPRECATED** `ParaEnsembleSampler` — superseded by `EnsembleSampler(nsamplers=...)`; warns on construction, kept until the LAT call site migrates. |
| `src/eryn/state.py` | `State` (per-iteration snapshot; carries `nsamplers`/`samplers_running`, sampler axis folded into the temperature axis), `Branch` (per-model coords+inds), `BranchSupplemental` (arbitrary side-channel data indexed like the ensemble), deprecated `ParaState`. |
| `src/eryn/model.py` | `Model` — a namedtuple bundling `log_like_fn`, `compute_log_like_fn`, `compute_log_prior_fn`, `temperature_control`, `map_fn`, `random`; passed into every `Move.propose`. |
| `src/eryn/prior.py` | `ProbDistContainer` (aggregates per-parameter distributions into one prior object with `logpdf`/`rvs`), plus `UniformDistribution`/`MappedUniformDistribution`/`log_uniform` helpers. |
| `src/eryn/moves/` | The proposal ("move") hierarchy — see §3. |
| `src/eryn/backends/` | `Backend` (in-memory), `HDFBackend`/`TempHDFBackend` (h5py-backed), `ParaBackend` (in-memory, `ParaEnsembleSampler`-shaped). These store **sampler chain state**, not compute backend. |
| `src/eryn/utils/` | `PeriodicContainer` (periodic-parameter wrap/distance), `TransformContainer` (parameter reparam pre-prior/pre-likelihood), `PlotContainer` + `plot.py` (corner/trace/tempering-ridge/RJ diagnostic plots), `updates.py` (`AdjustStretchProposalScale` and other post-step callbacks), `stopping.py` (`Stopping`/`AutoCorrelationStop` callbacks), `utility.py` (autocorrelation, PSRF, evidence estimators, `groups_from_inds`). |
| `src/eryn/pbar.py` | `tqdm` progress-bar wrapper (`get_progress_bar`), no-op fallback. |
| `tests/test_eryn.py`, `test_nuts.py`, `test_paraensemble.py` | Unit tests; `test_paraensemble.py` is the most current/rigorous (has regression tests documenting fixed bugs — see §7). |
| `examples/` | Tutorial scripts. **`two_models_swap_test.py` is stale** — imports `BasicSymmetricModelSwapRJMove`, which is commented out of `moves/__init__.py` and whose source file (`basicmodelswaprj.py`) does not exist in this checkout. |

## 3. Core abstractions

**`EnsembleSampler`** (`ensemble.py`) owns: `nwalkers`, per-branch `ndims`/
`nleaves_max`/`nleaves_min`, the `priors` dict (branch → `ProbDistContainer`),
a list of in-model `moves` (+ weights) and optional `rj_moves` (+ weights),
an optional `TemperatureControl`, and a `Backend`. `sample()` is a generator:
each iteration picks a random in-model move (repeated `num_repeats_in_model`
times), then — if RJ is enabled — a random RJ move (repeated
`num_repeats_rj` times), then saves a `State` to the backend every
`thin_by` proposals. `compute_log_prior`/`compute_log_like` do the
branch-dict ↔ flat-array marshalling into the user's `log_like_fn`
(vectorized or per-group, with/without RJ "groups", with/without
supplementals — see the long `__init__` docstring for the exact call
signatures).

**Independent samplers (`nsamplers`) — the folded representation.**
`EnsembleSampler(nsamplers=N)` runs N fully independent parallel-tempered
ensembles simultaneously with one batched likelihood call per proposal
(the GPU-throughput pattern `ParaEnsembleSampler` existed for). Internally
the sampler axis is **folded into the temperature axis**: all `State`/
`Branch` arrays keep their usual 4D/2D shapes with leading axis
`ntemps_eff = nsamplers * ntemps` (row `s * ntemps + t` = temperature `t`
of sampler `s`), so every move works unchanged — proposals just see more
"temperatures". The only sampler-aware components are `TemperatureControl`
(per-sampler betas ladders stored flat, swaps only within a sampler's own
rows, per-sampler ladder adaptation, vectorized across samplers), the
`compute_log_prior` wrapper (`samplers_running` row masking +
`prior_transform_fn` per-sampler prior-basis transforms), and the backends
(storage always carries the sampler axis after the step axis; getters
squeeze it when `nsamplers == 1` so single-sampler users see the legacy
shapes everywhere). `State.samplers_running` (bool `(nsamplers,)`) freezes
converged samplers: their rows get `-inf` log-prior, which skips the user
likelihood, always-rejects proposals, and excludes them from swaps and
adaptation.

**`ParaEnsembleSampler`** (`paraensemble.py`) — **deprecated**, replaced
by the above. It re-implemented a hard-coded stretch + tempering loop for
a `(ngroups, ntemps, nwalkers, ndim)` shape contract (fixed single branch,
no RJ). It warns on construction and will be removed once
`lisatools/globalfit/moves/gbspecialstretch.py` migrates.

**`Move` hierarchy** (`moves/`, base class `move.py:Move`):
- `Move` — shared machinery: Gibbs-sampling split bookkeeping
  (`gibbs_sampling_setup_iterator`, `setup_proposals`,
  `cleanup_proposals_gibbs`, `fix_logp_gibbs`), tempered-vs-basic log
  posterior dispatch, and `update()` — the accept/reject state-merge and
  single most load-bearing method in the package: it uses
  `np.take_along_axis`/`put_along_axis` to merge only the accepted
  walkers' coords/inds/logl/logp/blobs/supplementals back into `State`.
- `RedBlueMove` (`red_blue.py`) — abstract Goodman & Weare red/blue split
  ensemble move (from `emcee`); `StretchMove` (`stretch.py`) implements
  `get_proposal` with the affine stretch. `GroupMove`/`GroupStretchMove`
  (`group.py`/`groupstretch.py`) are the RJ-safe cousin: the complement
  ("friends") is a **stationary** snapshot refreshed every `n_iter_update`
  iterations rather than the live ensemble, because a live-ensemble
  complement breaks detailed balance under RJ.
- `MHMove` (`mh.py`) → `GaussianMove` (`gaussian.py`), `DistributionGenerate`
  (`distgen.py`, in-model draw from an arbitrary prior-like distribution).
- `ReversibleJumpMove` (`rj.py`) — abstract trans-dimensional move;
  `DistributionGenerateRJ` (`distgenrj.py`) is the concrete default (birth/
  death proposals drawn from a `ProbDistContainer`, usually the prior),
  handling `nleaves_min`/`nleaves_max` edge-case proposal-asymmetry
  factors and (partially-implemented) delayed rejection on births.
- `MultipleTryMove` (`multipletry.py`) — abstract multiple-try framework
  (importance-weighted choice among `num_try` proposals);
  `MTDistGenMove`/`MTDistGenMoveRJ` (`mtdistgen.py`/`mtdistgenrj.py`) are
  the concrete in-model / RJ multiple-try distribution-generate moves.
- `DelayedRejection` (`delayedrejection.py`) — fallback-proposal wrapper
  for a rejected RJ birth (wired via `ReversibleJumpMove.__init__(dr=...)`;
  the `propose()`-time DR branch in `rj.py` currently
  `raise NotImplementedError`).
- `CombineMove` (`combine.py`) — runs a list of moves in sequence within
  one proposal slot, exposing a list-shaped `accepted`.
- `TemperatureControl` (`tempering.py`) — not a `Move` subclass, but every
  move holds a reference to it. Builds the beta ladder (`make_ladder`,
  ptemcee-style), computes the tempered log posterior, performs
  neighbor-temperature swaps (`temperature_swaps`/`temper_comps`, with an
  optional "fancy swap" mode that recomputes the likelihood post-swap
  instead of exchanging cached values), and adapts the ladder
  (`adapt_temps`, arXiv:1501.05823 dynamics).
- `NUTSSampler`/`NUTSMove` (`nuts.py`) — standalone vectorized
  Hoffman & Gelman (2014) NUTS (`NUTSSampler`, works on raw
  gradient/log-posterior callables, not tied to `State`) plus a thin
  `MHMove` subclass (`NUTSMove`) that slots one NUTS step into the
  regular Eryn move schedule.

**`Backend` / `HDFBackend`** (`backends/`) — **sampler chain-state
storage**, unrelated to the GPU/CPU/JAX "compute backend" concept used
elsewhere across LISA Analysis Tools (`force_backend=`). `Backend` holds `chain`,
`inds`, `log_like`, `log_prior`, `betas`, `blobs`, `accepted`,
`rj_accepted`, `swaps_accepted`, and per-move acceptance fractions as
in-memory NumPy arrays keyed by branch name; `grow()` pre-extends
storage, `save_step()` appends one iteration, `get_a_sample`/
`get_last_sample` reconstruct a `State`. `HDFBackend` is a drop-in
h5py-backed mirror (same surface, properties read live from the open
file; retry/backoff around `BlockingIOError`/lock contention).
`ParaBackend` (`backends/parabackend.py`) is the `ParaState`-shaped
counterpart for `ParaEnsembleSampler` (known gap — see §6).

**`State`/`ParaState`** (`state.py`) wrap `Branch` objects (`coords`
`(ntemps, nwalkers, nleaves_max, ndim)` + `inds` `(ntemps, nwalkers,
nleaves_max)` bool + optional `BranchSupplemental`) plus `log_like`,
`log_prior`, `betas`, `blobs`, `supplemental`, `random_state`. `ParaState`
drops `inds`/`nleaves_max` (fixed dimension) and adds a top-level
`groups_running` bool mask over the leading `ngroups` axis instead.
`BranchSupplemental` is a dict-of-arrays container indexable like the
ensemble (`base_shape` = `(ntemps, nwalkers[, nleaves_max])`), with
`take_along_axis`/`put_along_axis` used by `Move.update` and the
temperature-swap code to move side-channel data (caches, per-leaf
metadata, GPU-resident arrays) in lockstep with accepted moves.

**`ProbDistContainer`** (`prior.py`) aggregates a dict of
`{index-or-str-key: distribution}` into one object with `logpdf`/`rvs`
that operates over the full parameter vector; distributions can be int-,
tuple-, or string-keyed (string keys populate `key_order`, used to
validate priors ↔ backend consistency across restarts).

**Branches/leaves relationship sketch**

```
EnsembleSampler
 ├─ priors: {branch_name: ProbDistContainer}
 ├─ moves: [Move, ...]            (in-model; RedBlueMove/StretchMove/GroupStretchMove/...)
 ├─ rj_moves: [ReversibleJumpMove, ...]   (optional, trans-dimensional)
 ├─ temperature_control: TemperatureControl (optional)
 └─ backend: Backend | HDFBackend
       stores → State (per iteration)
                 ├─ branches: {branch_name: Branch(coords, inds, branch_supplemental)}
                 ├─ log_like, log_prior, betas, blobs
                 └─ supplemental: BranchSupplemental
Move.propose(model: Model, state: State) -> (new_state, accepted)
```

## 4. Public API / entry points

```python
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.moves import StretchMove, DistributionGenerateRJ, TemperatureControl
from eryn.backends import HDFBackend

sampler = EnsembleSampler(nwalkers, ndims, log_like_fn, priors,
                           branch_names=[...], nleaves_max={...},
                           rj_moves=True, tempering_kwargs={"ntemps": 10},
                           backend=HDFBackend("chain.h5"))
state = sampler.run_mcmc(initial_state, nsteps, burn=1000)
sampler.get_chain(discard=..., thin=...)   # -> {branch_name: ndarray}
```

`EnsembleSampler.sample(initial_state, iterations, ...)` is the underlying
generator `run_mcmc` iterates; call it directly for per-iteration control
(diagnostics between steps, custom stopping). For many independent
ensembles at once, add `nsamplers=N` (see §5). The deprecated
`ParaEnsembleSampler` (`eryn.paraensemble`, state via
`eryn.state.ParaState`) is the old path for the same job. Convenience
re-exports: `eryn.backends.get_test_backends()` (used by test suites to
parametrize over in-memory vs HDF5), `eryn.utils.PlotContainer` /
`eryn.utils.plot.*` for post-hoc diagnostic plots, `eryn.utils.utility.*`
for autocorrelation/evidence/PSRF.

## 5. Sampler mechanics

**Ensemble + tempering.** Each `sample()` iteration draws a random move
from the weighted `moves` list and calls `move.propose(model, state)`.
Red/blue-family moves (`StretchMove`, `GroupStretchMove`) split walkers
into `nsplits` (default 2) sub-ensembles and propose each half against a
complement built from the others (live ensemble for `StretchMove`, a
periodically-refreshed stationary snapshot for `GroupStretchMove`/RJ-safe
moves). Acceptance uses the tempered log posterior
(`TemperatureControl.compute_log_posterior_tempered`, `logl*beta + logp`)
when `temperature_control` is set. After the in-model (and RJ) moves for
an iteration, `TemperatureControl.temper_comps` performs adjacent-rung
swaps top-down (`temperature_swaps`) and, if `adaptive=True`, nudges the
beta ladder (`adapt_temps`, hyperbolic-decay dynamics from
arXiv:1501.05823) — unless `stop_adaptation` has been reached.

**Reversible jump / branches-leaves-nleaves.** Each branch has a fixed
`nleaves_max` (array capacity) and a per-iteration `inds` bool mask
marking which slots are "alive" — the trans-dimensional state; dead slots
keep stale coordinates in memory but are excluded from prior/likelihood/
plots. `ReversibleJumpMove.propose` calls `get_model_change_proposal`
(concrete in `DistributionGenerateRJ`) to pick, per walker, one leaf to
add or remove (`fix_change=None` picks randomly between +1/-1, clamped at
`nleaves_min`/`nleaves_max`), draws new-leaf coordinates from
`generate_dist` (typically the prior), and applies proposal-asymmetry
`edge_factors` (`log(1/2)` terms) at the min/max boundaries where the
+1/-1 choice isn't symmetric. Plain `StretchMove` alongside RJ is
discouraged (constructor warning) since its live-ensemble complement
ignores the per-walker active-dimension mask; `GroupStretchMove` with a
stationary friends group is the recommended in-model move when RJ is on.

**Gibbs sampling.** Any `Move` can be constructed with
`gibbs_sampling_setup` (str / tuple / dict / list) to restrict which
branches/leaves/parameters a given proposal call touches; `Move` iterates
the configured splits via `gibbs_sampling_setup_iterator` and
`cleanup_proposals_gibbs` restores untouched parameters after each split.
RJ moves only allow branch-level (not parameter-level) Gibbs splits.

**Independent samplers (`nsamplers > 1`).** User-facing: pass
`nsamplers=N` to `EnsembleSampler`, give initial coordinates shaped
`(N, ntemps, nwalkers, nleaves_max, ndim)` (para-style 4D
`(N, ntemps, nwalkers, ndim)` also accepted when `nleaves_max == 1`), and
every stored/returned quantity gains a sampler axis right after the step
axis (`get_chain()[name] -> (nsteps, N, ntemps, nwalkers, nleaves_max,
ndim)`; `sampler_index=` on all getters parallels `temp_index=`). With
`nsamplers=1` (default) nothing changes anywhere, bitwise — the seeded RNG
stream is preserved. Freezing: set/replace `state.samplers_running` (e.g.
inside `update_fn`); frozen samplers' rows hold fill values (`-1e300`
logl, `-inf` logp) and their coords/betas stay bit-identical.
Per-sampler prior bounds: `prior_transform_fn` (same contract as para's:
in-place `transform_to_prior_basis(coords, running_idx)` +
`adjust_logp(logp, running_idx)`), requires `nleaves_max == 1` and no
RJ/multiple-try/`provide_groups`/`all_models_together`. Diagnostics for
`nsamplers > 1`: `get_evidence_estimate` returns per-sampler arrays;
autocorrelation/Gelman-Rubin/plotting raise — compute per sampler via
`get_chain(sampler_index=...)`. There is no MPI code here — the
parallelism is intra-process array-batching (SIMD/GPU); MPI-style
multi-process coordination lives in LAT's `lisatools.globalfit.engine`,
which drives multiple sampler instances across ranks itself.

## 6. Cross-repo dependencies

**Eryn → gpubackendtools (GBT).** `paraensemble.py` imports
`get_backend`/`get_first_backend`/`GPUBACKENDTOOLSException` from
`gpubackendtools` — Eryn's *only* cross-repo import within LISA Analysis Tools, and its only
source of `cupy`/GPU array support for `ParaEnsembleSampler` (hard
dependency in `pyproject.toml`, annotated "GBT backend registry:
numpy/cupy access for paraensemble"). This is a **different "backend"
than the sampler-state `Backend`/`HDFBackend` classes in
`eryn.backends`** — GBT's `Backend` objects resolve `.xp` (numpy vs cupy)
and device selection per the LISA Analysis Tools–wide "backend chosen at construction,
not per-method-kwarg" rule (`force_backend="cpu"|"cuda12x"`,
`gpu=<index>`); `eryn.backends.Backend` is unrelated MCMC-chain
storage/serialization. Elsewhere in Eryn (`state.py`, `move.py`,
`stretch.py`, `prior.py`), GPU support is a local
`try: import cupy as cp/xp except: import numpy` fallback with a
`use_gpu`/`use_cupy` flag — no GBT dependency there.

**Consumers of Eryn.** LISAanalysistools' `lisatools/globalfit/` package
is the dominant downstream: `state.py`, `run.py`, `recipe.py`,
`psdglobal.py`, `postprocessing.py`, `pipeline.py`, and
`moves/{psdmove,mbhspecialmove,globalfitmove}.py` import
`eryn.ensemble.EnsembleSampler`, `eryn.state.{State,Branch,
BranchSupplemental}`, `eryn.moves.{Move,RedBlueMove,StretchMove,
CombineMove,TemperatureControl,make_ladder}`, `eryn.prior
.ProbDistContainer`, `eryn.backends.HDFBackend`, and
`eryn.utils.{TransformContainer,PeriodicContainer,get_integrated_act,
PlotContainer}`. `lisatools/globalfit/moves/gbspecialstretch.py` and
`lisatools/globalfit/stock/erebor/gb.py` are the `ParaEnsembleSampler`
consumers (GB global-fit in-model step), passing `force_backend` through
from their own settings. No other LISA Analysis Tools repo (GBGPU, BBHx, FEW,
GPUBackendTools, lisa-on-gpu) imports Eryn directly.

## 7. Non-obvious invariants / gotchas

- **The `nsamplers` axis is FOLDED internally**: inside `State`/moves, the
  leading axis of every array is `nsamplers * ntemps` (`Branch.ntemps` and
  `Move.ntemps` are that folded size; per-sampler values live in
  `ntemps_per_sampler`). The sampler axis only exists at the storage/user
  boundary (backends unfold on write, squeeze on read when
  `nsamplers == 1`). `TemperatureControl.betas` is likewise flat
  `(nsamplers * ntemps,)` with a `betas_grouped` reshape view.
- **`GroupMove`/`GroupStretchMove` friends must be per-sampler
  independent** when `nsamplers > 1`: `find_friends`/`setup_friends`
  receive folded arrays, and pooling candidates across rows (e.g. a global
  friends catalog over all temperatures) silently correlates the
  supposedly independent ensembles. Row-wise selection is automatically
  safe; use `move.sampler_id_rows`/`branch.coords_grouped` to restrict any
  cross-row pooling to same-sampler rows.
- **Frozen samplers must be excluded from temperature swaps** (they are —
  `temperature_swaps(samplers_running=...)`): their rungs all hold
  `-1e300` logl, so an unmasked swap would always accept (`paccept = 0`)
  and permute frozen coordinates. Relatedly, masked rows produce one-time
  `inf - inf` RuntimeWarnings in move acceptance math (`nan` → reject) —
  harmless.
- **Legacy HDF files (no `nsamplers` attr) read/write in the old layout**:
  restart from them works only with `nsamplers == 1`; `sampler_index=` and
  `get_samplers_running` are unavailable on them. New files always store
  the sampler axis (rank-6 chain datasets) — raw-h5py consumers must index
  accordingly.
- **`ParaBackend.get_a_sample`/`get_last_sample`** were previously broken
  (malformed 5-D coords, dict `groups_running`, squeezed `betas`) — now
  **fixed and regression-tested** (`tests/test_paraensemble.py::
  ParaBackendSampleRetrievalTest`). The MEMORY.md note "`ParaBackend
  .get_a_sample` still broken" is **stale** as of this checkout.
- **`ParaBackend.get_gelman_rubin_convergence_diagnostic` is still
  broken**, however: it calls `self.get_inds(...)` (parabackend.py:506),
  but `ParaBackend` only defines `get_groups_running` — `get_inds`
  doesn't exist (no RJ/leaves concept in the fixed-dimension
  `ParaEnsembleSampler`). Calling it raises `AttributeError`.
- **`examples/two_models_swap_test.py` is stale**: imports
  `BasicSymmetricModelSwapRJMove`, whose import line is commented out in
  `moves/__init__.py` and whose source file does not exist in this
  checkout (same for DE/DE-snooker/KDE/Walk moves referenced there).
- **RJ + `StretchMove` is a documented foot-gun, not a hard error**: the
  constructor only `warnings.warn`s if a plain `StretchMove` is mixed
  with RJ — it runs but is "most likely very inefficient" since the
  live-ensemble complement ignores the per-walker active-dimension mask.
  Use `GroupStretchMove` instead.
- **Delayed rejection on RJ births is wired but not implemented**:
  `ReversibleJumpMove.__init__` builds a `DelayedRejection` wrapper when
  `dr=` is set, but `rj.py`'s `propose()` immediately
  `raise NotImplementedError(...)` on that branch — don't pass `dr_moves`.
- **`store_missing_leaves` (default `np.nan`) only overwrites dead-leaf
  coords on write**, not in the live `State` — `Branch.coords` for
  inactive leaves keeps its last real value in memory (warm-restart-able)
  but the backend/HDF5 file always gets `store_missing_leaves` there.
- **Backend `key_order`/move-key mismatches are hard restart-blockers**:
  `EnsembleSampler.__init__` raises `ValueError` if
  `key_order != backend.key_order`, or if `track_moves=True` and the set
  of move keys changed since the backend was written — intentional
  "start a fresh backend" guards, not bugs.
- **Tempering `fancy_swap` recomputes the likelihood on every swap**
  (`TemperatureControl.perform_fancy_swap_acceptance_fraction` calls
  `compute_log_like`) — far more expensive than the default cached-value
  exchange; opt in only if the coordinate-value swap can't cheaply carry
  the likelihood along.
- **`ParaBackend.save_step` branches on cupy vs numpy via
  `try/except AttributeError`** (on `.get()`) rather than an explicit
  `xp` check — don't assume one array type when subclassing.

## 8. Where to look for X

| Want to... | Start in |
|---|---|
| Understand the main sampling loop / how a move gets chosen and applied | `src/eryn/ensemble.py` (`EnsembleSampler.sample`) |
| Add/modify an in-model proposal | `src/eryn/moves/move.py` (base), `src/eryn/moves/red_blue.py` + `stretch.py` (affine-invariant family) |
| Add/modify a reversible-jump (trans-dimensional) proposal | `src/eryn/moves/rj.py` (base), `src/eryn/moves/distgenrj.py` (concrete default) |
| Change how accept/reject merges state (coords/inds/blobs/supplementals) | `src/eryn/moves/move.py` (`Move.update`) |
| Parallel tempering ladder, swaps, adaptation | `src/eryn/moves/tempering.py` (`TemperatureControl`) |
| Multiple-try proposals | `src/eryn/moves/multipletry.py`, `mtdistgen.py`, `mtdistgenrj.py` |
| NUTS / gradient-based sampling | `src/eryn/moves/nuts.py` |
| Priors (define/compose/sample) | `src/eryn/prior.py` (`ProbDistContainer`) |
| What gets stored per iteration / chain array shapes | `src/eryn/state.py` (`State`, `Branch`, `BranchSupplemental`) |
| In-memory vs HDF5 chain storage | `src/eryn/backends/backend.py`, `hdfbackend.py` |
| Run many independent ensembles at once (GPU-batched likelihoods) | `EnsembleSampler(nsamplers=...)` in `src/eryn/ensemble.py`; shapes/freezing in `src/eryn/state.py`; per-sampler tempering in `src/eryn/moves/tempering.py`; usage in `tests/test_nsamplers.py` |
| `ParaEnsembleSampler`-specific storage (deprecated) | `src/eryn/backends/parabackend.py` (note §7 gotcha) |
| GPU-batched multi-ensemble sampling, deprecated para path | `src/eryn/paraensemble.py` (use `nsamplers=` instead) |
| Periodic-parameter wrap/distance math | `src/eryn/utils/periodic.py` |
| Parameter reparameterization before prior/likelihood | `src/eryn/utils/transform.py` |
| Autocorrelation / evidence (thermodynamic, stepping-stone) / Gelman-Rubin | `src/eryn/utils/utility.py` |
| Post-hoc diagnostic plots (corner, trace, tempering ridge, RJ leaves) | `src/eryn/utils/plot.py`, `PlotContainer` |
| Stopping criteria / mid-run callbacks | `src/eryn/utils/stopping.py`, `src/eryn/utils/updates.py` |
| Existing usage patterns for multi-branch + RJ | `tests/test_eryn.py` (`test_rj_multiple_branches`, `test_gibbs_sampling`) |
| Existing usage patterns for `ParaEnsembleSampler` | `tests/test_paraensemble.py` |
