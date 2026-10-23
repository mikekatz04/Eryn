_Last mapped: e448a17 (dev) + the 2026-10-19 docstring pass · regenerate when structure changes_

# Eryn codebase map

## 1. What this is

Eryn (`src/eryn/`, package `eryn`) is a general-purpose MCMC sampler library:
ensemble (affine-invariant, emcee-style) sampling with optional parallel
tempering, reversible-jump (trans-dimensional / variable-source-count)
moves, multiple-try moves, eigen-axis and ridge-Gibbs moves, and a
vectorized NUTS implementation. Delayed rejection is wired into the
reversible-jump move but not functional (see §7). It is the sampling engine
LISAanalysistools' global-fit layer runs on. Unlike the other LISA Analysis
Tools repos it ships **no compiled code of its own**: it is pure
Python/NumPy, with an optional CuPy path in several classes and a GPU
compute backend borrowed from `gpubackendtools` (GBT) for the deprecated
`ParaEnsembleSampler` only. There is no CUDA-kernel layer to document here;
the interesting architecture is the sampler/move/backend/state object model.

## 2. Layout

| Path | Role |
|---|---|
| `src/eryn/ensemble.py` | `EnsembleSampler`, the main sampler class: one or many branches, RJ-capable, single-process (`pool`-parallel likelihood calls when not vectorized). `nsamplers=N` runs N fully independent ensembles at once with batched likelihood calls (see §5). Also `walkers_independent`. |
| `src/eryn/paraensemble.py` | **DEPRECATED** `ParaEnsembleSampler` (subclass of `EnsembleSampler` that does not call its `__init__`) and `shuffle_along_axis`; superseded by `EnsembleSampler(nsamplers=...)`; warns on construction, kept until the LAT call sites migrate (§6). |
| `src/eryn/state.py` | `State` (per-iteration snapshot; carries `nsamplers`/`samplers_running`, sampler axis folded into the temperature axis), `Branch` (per-model coords + inds), `BranchSupplemental` (arbitrary side-channel data indexed like the ensemble), deprecated `ParaState`. |
| `src/eryn/model.py` | `Model`, a namedtuple bundling `log_like_fn`, `compute_log_like_fn`, `compute_log_prior_fn`, `temperature_control`, `map_fn`, `random`; built by `EnsembleSampler.get_model` once per `sample()` call and passed into every `Move.propose`. |
| `src/eryn/priors/` | The prior subpackage: `base.py` (`AbstractPrior`, `Prior`, `PriorException`), `analytical.py` (`UniformDistribution`, `DeltaFunction`, `Gaussian`/`Normal`, `LogNormal`/`LogGaussian`, `Exponential`, and the transform-backed `PowerLaw`, `LogUniform`, `Log10Uniform`, `CosineUniform`, `SineUniform`, `UniformInVolume`, `InverseUniform`), `discrete.py` (`DiscreteUniform`, `CategoricalDistribution`, `PoissonDistribution`), `joint.py` (`JointPrior`, `MultivariateGaussian`), `probdist.py` (`ProbDistContainer`, the per-branch prior the sampler takes). |
| `src/eryn/prior.py` | **DEPRECATED** shim (module-level `DeprecationWarning`): re-exports `ProbDistContainer`, `UniformDistribution`, `LogUniform` and keeps `uniform_dist`, `log_uniform`, `MappedUniformDistribution` as wrappers. Still the import path of most LAT prior builds. |
| `src/eryn/moves/` | The proposal ("move") hierarchy and `TemperatureControl`; see §3. |
| `src/eryn/backends/` | `Backend` (in-memory), `HDFBackend` (h5py-backed), `TempHDFBackend` (context manager giving an `HDFBackend` on a temporary file, for tests), `get_test_backends`, and the deprecated `ParaBackend` (`backends/parabackend.py`, not re-exported). These store **sampler chain state**, not a compute backend. |
| `src/eryn/utils/` | `PeriodicContainer` (periodic wrap/distance), `TransformContainer` (sampling basis -> likelihood basis: fill fixed values, parameter transforms, optional inverses), `PlotContainer` + `plot.py` (corner/trace/tempering/RJ/advanced diagnostic plots), `updates.py` (`Update`, `CompositeUpdate`, `UpdateStep`, `AdjustStretchProposalScale`), `stopping.py` (`Stopping`, `SearchConvergeStopping`; an `AutoCorrelationStop` sketch is commented out inside a string), `utility.py` (`groups_from_inds`, `get_acf`, `get_integrated_act`, evidence estimators, `psrf`). Importing `eryn.utils` imports `plot.py` (matplotlib, corner, pandas, seaborn). |
| `src/eryn/pbar.py` | `tqdm` progress-bar wrapper (`get_progress_bar`), no-op fallback. |
| `tests/` | `test_eryn.py`, `test_nsamplers.py`, `test_paraensemble.py`, `test_nuts.py`, `test_eigenaxis.py`, `test_fancy_swap_pairing.py`, `test_priors.py`. CI runs `python -m unittest discover` (`.github/workflows/run-tests.yml`). `test_paraensemble.py` has regression tests for fixed `ParaBackend` bugs (§7). |
| `examples/` | `Eryn_tutorial.ipynb` (copied into `docs/source/` by `docs/source/conf.py` at docs build), `eryn_group_moves.ipynb` (same file as `docs/source/tutorial/Group_moves.ipynb`), `more_tutorials.ipynb`, plotting scripts, `developing_delayed_rejection*.py` (dev scripts for the non-functional delayed rejection). **`two_models_swap_test.py` is stale**: it imports `BasicSymmetricModelSwapRJMove`, which is commented out of `moves/__init__.py` and whose source file does not exist. |
| `docs/source/` | Sphinx sources: `index.rst`, `readme.md` (includes the repo `README.md`), API pages `user/{ensemble,state,backend,moves,prior,temper,utils}.rst`, `general/todos.rst` (code TODOs and known issues). `docs/doctrees/` and `docs/_build/` are build output. |

## 3. Core abstractions

**`EnsembleSampler`** (`ensemble.py`) owns: `nwalkers`, per-branch `ndims`/
`nleaves_max`/`nleaves_min`, the `priors` dict (branch -> `ProbDistContainer`),
a list of in-model `moves` (+ weights) and optional `rj_moves` (+ weights),
an optional `TemperatureControl`, and a `Backend`. `sample()` is a generator:
each proposal iteration runs `num_repeats_in_model` in-model proposals (a
move drawn at random each time), then, if RJ is enabled, `num_repeats_rj`
RJ proposals; every `thin_by` proposal iterations it saves the `State` to
the backend (`store=True`) and yields it. `run_mcmc` wraps it (burn-in,
stopping function, runtime plots). The default in-model move is
`StretchMove(a=2.0)`; `rj_moves=True` (or `"together"`,
`"iterate_branches"`, `"separate_branches"`) builds
`DistributionGenerateRJ` moves from the priors; any `rj_moves` other than
`None` requires `moves`. `compute_log_prior`/`compute_log_like` do the
branch-dict <-> flat-array marshalling into the user's `log_like_fn`
(vectorized or per walker, with or without RJ "groups", with or without
supplementals; see the class docstring for the exact call forms).

**Independent samplers (`nsamplers`): the folded representation.**
`EnsembleSampler(nsamplers=N)` runs N fully independent parallel-tempered
ensembles simultaneously with one batched likelihood call per proposal
(the GPU-throughput pattern `ParaEnsembleSampler` existed for). Internally
the sampler axis is **folded into the temperature axis**: all `State`/
`Branch` arrays keep their usual shapes with leading axis
`ntemps_eff = nsamplers * ntemps` (row `s * ntemps + t` = temperature `t`
of sampler `s`), so every move works unchanged; proposals just see more
"temperatures". The only sampler-aware components are `TemperatureControl`
(per-sampler betas ladders stored flat, swaps only within a sampler's own
rows, per-sampler ladder adaptation, vectorized across samplers), the
`compute_log_prior` wrapper (`samplers_running` row masking +
`prior_transform_fn` per-sampler prior-basis transforms), and the backends
(storage always carries the sampler axis after the step axis; getters
squeeze it when `nsamplers == 1` so single-sampler users see the legacy
shapes everywhere). `State.samplers_running` (bool `(nsamplers,)`) freezes
converged samplers: their rows get `-inf` log-prior, which skips the user
likelihood, always rejects proposals, and excludes them from swaps and
adaptation.

**`ParaEnsembleSampler`** (`paraensemble.py`): **deprecated**, replaced
by the above. It re-implements a hard-coded stretch + tempering loop for
a `(ngroups, ntemps, nwalkers, ndim)` shape contract (fixed single branch,
no RJ), stores into `ParaBackend` and works on `ParaState`. It warns on
construction.

**`Move` hierarchy** (`moves/`, base class `move.py:Move`):
- `Move`: shared machinery. Gibbs-sampling split bookkeeping
  (`gibbs_sampling_setup_iterator`, `setup_proposals`,
  `cleanup_proposals_gibbs`, `fix_logp_gibbs`), tempered-vs-basic log
  posterior dispatch, and `update()`, the accept/reject state merge and
  the single most load-bearing method in the package: it uses
  `np.take_along_axis`/`put_along_axis` to merge only the accepted
  walkers' coords/inds/logl/logp/blobs/supplementals back into `State`.
- `RedBlueMove` (`red_blue.py`): abstract Goodman & Weare red/blue split
  ensemble move (from `emcee`); `StretchMove` (`stretch.py`) implements
  `get_proposal` with the affine stretch. `GroupMove`/`GroupStretchMove`
  (`group.py`/`groupstretch.py`) are the RJ-safe cousins: the complement
  ("friends") is a **stationary** set rebuilt every `n_iter_update`
  calls rather than the live ensemble; under RJ the live complement
  ignores which leaves each walker has active, and the sampler warns that
  a plain `StretchMove` there is "most likely very inefficient".
- `MHMove` (`mh.py`) -> `GaussianMove` (`gaussian.py`), `DistributionGenerate`
  (`distgen.py`, in-model draw from a prior-like distribution),
  `EigenAxisMove` (`eigenaxis.py`, jumps along caller-supplied
  information-matrix axes; helpers `eigen_axis_set`, `prior_box_scales`,
  `axis_prior_bounds`, `project_out_direction`, `draw_axis_step`), and
  `NUTSMove` (`nuts.py`).
- `RidgeGibbsMove` (`ridgegibbs.py`): likelihood-free Gibbs resampling
  along an exact 1-D likelihood degeneracy supplied by a duck-typed
  `fiber_map` (zero likelihood evaluations).
- `ReversibleJumpMove` (`rj.py`): abstract trans-dimensional move;
  `DistributionGenerateRJ` (`distgenrj.py`) is the concrete default (birth/
  death proposals drawn from a `ProbDistContainer`, usually the prior),
  with the `nleaves_min`/`nleaves_max` edge factors applied in
  `ReversibleJumpMove.propose`.
- `MultipleTryMove` (`multipletry.py`): abstract multiple-try mixin
  (importance-weighted choice among `num_try` proposals) and its nested-RJ
  variant `MultipleTryMoveRJ`; `MTDistGenMove`/`MTDistGenMoveRJ`
  (`mtdistgen.py`/`mtdistgenrj.py`) are the concrete in-model / RJ
  multiple-try distribution-generate moves. Module helpers `logsumexp` and
  `get_mt_computations` (defined twice; the second definition is the live
  one).
- `DelayedRejection` (`delayedrejection.py`): fallback-proposal wrapper
  for a rejected RJ birth, built by `ReversibleJumpMove.__init__(dr=...)`;
  the `propose()`-time DR branch in `rj.py` raises `NotImplementedError`,
  so it never runs. Not exported from `eryn.moves`.
- `CombineMove` (`combine.py`): runs a list of moves in sequence within
  one proposal slot; each sub-move does its own accept/reject and swaps.
- `TemperatureControl` (`tempering.py`): not a `Move` subclass; a tempered
  move holds a reference to one (untempered moves hold `None`). Builds the
  beta ladder (`make_ladder`, ptemcee-style), computes the tempered log
  posterior, performs neighbour-rung swaps (`temperature_swaps` /
  `temper_comps`, with an optional "fancy swap" mode that recomputes the
  likelihood for the swapped coordinates instead of exchanging cached
  values), and adapts the ladder (`adapt_temps`, arXiv:1501.05823
  dynamics).
- `NUTSSampler`/`NUTSMove` (`nuts.py`): standalone vectorized
  Hoffman & Gelman (2014) NUTS (`NUTSSampler`, works on raw
  gradient/log-posterior callables, not tied to `State`) plus a thin
  `MHMove` subclass (`NUTSMove`) that slots one NUTS step into the
  regular Eryn move schedule.

**`Backend` / `HDFBackend`** (`backends/`): **sampler chain-state
storage**, unrelated to the GPU/CPU/JAX "compute backend" concept used
elsewhere across LISA Analysis Tools (`force_backend=`). `Backend` holds `chain`,
`inds`, `log_like`, `log_prior`, `betas`, `blobs`, `accepted`,
`rj_accepted`, `swaps_accepted`, and per-move acceptance fractions as
in-memory NumPy arrays keyed by branch name; `grow()` pre-extends
storage, `save_step()` appends one iteration, `get_a_sample`/
`get_last_sample` reconstruct a `State`, and the getters share the
`get_value` selection keywords (`discard`, `thin`, `temp_index`,
`sampler_index`, `slice_vals`, `branch_names`). `HDFBackend` is an
h5py-backed mirror (same surface, properties read live from the open
file; opening retries on HDF5 lock contention, reads retry on `OSError`).
`ParaBackend` (`backends/parabackend.py`) is the `ParaState`-shaped
counterpart for `ParaEnsembleSampler`.

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

**`ProbDistContainer`** (`priors/probdist.py`) aggregates a dict of
`{key: distribution}` into one object with `logpdf`/`rvs` over the full
parameter vector. Keys are `int` column indices, `str` names (next free
column, in insertion order; they populate `key_order`, which the sampler
checks against a resumed backend), or tuples of either for multi-column
components; int and str keys cannot be mixed. Conditional components
(priors whose `logpdf`/`rvs` take another column as an argument) are
resolved from keyword arguments or other columns. The container calls the
plain `rvs`/`logpdf` of each component, i.e. the SAMPLING variable: a
`LogUniform` column is drawn and scored in `ln x`, a `PowerLaw` column in
`x**(alpha + 1)` (the `*_physical` methods give the physical variable).

**Branches/leaves relationship sketch**

```
EnsembleSampler
 ├─ priors: {branch_name: ProbDistContainer}
 ├─ moves: [Move, ...]            (in-model; StretchMove/GroupStretchMove/GaussianMove/...)
 ├─ rj_moves: [ReversibleJumpMove, ...]   (optional, trans-dimensional)
 ├─ temperature_control: TemperatureControl (optional)
 └─ backend: Backend | HDFBackend
       stores → State (per stored step)
                 ├─ branches: {branch_name: Branch(coords, inds, branch_supplemental)}
                 ├─ log_like, log_prior, betas, blobs
                 └─ supplemental: BranchSupplemental
Move.propose(model: Model, state: State) -> (new_state, accepted)
```

## 4. Public API / entry points

```python
import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.priors import ProbDistContainer, UniformDistribution
from eryn.moves import GaussianMove, DistributionGenerateRJ, TemperatureControl
from eryn.backends import HDFBackend

priors = {"gauss": ProbDistContainer({"amp": UniformDistribution(0.0, 10.0),
                                      "mean": UniformDistribution(-1.0, 1.0),
                                      "sigma": UniformDistribution(0.01, 0.2)})}
sampler = EnsembleSampler(nwalkers, 3, log_like_fn, priors,
                          branch_names=["gauss"], nleaves_max={"gauss": 8},
                          nleaves_min={"gauss": 0},
                          moves=GaussianMove({"gauss": 1e-4 * np.eye(3)}),
                          rj_moves=True, tempering_kwargs={"ntemps": 10},
                          backend=HDFBackend("chain.h5"))
state = sampler.run_mcmc(initial_state, nsteps, burn=1000)
sampler.get_chain(discard=..., thin=...)   # -> {branch_name: ndarray}
```

`rj_moves` (anything but `None`) requires `moves`. `EnsembleSampler.sample(initial_state,
iterations, ...)` is the underlying generator `run_mcmc` iterates; call it
directly for per-iteration control. For many independent ensembles at once,
add `nsamplers=N` (see §5). The deprecated `ParaEnsembleSampler`
(`eryn.paraensemble`, state via `eryn.state.ParaState`) is the old path for
the same job. Other entry points: `eryn.backends.get_test_backends()` (used
by test suites to parametrize over in-memory vs HDF5),
`eryn.utils.PlotContainer` / `eryn.utils.plot.*` for diagnostic plots,
`eryn.utils.utility.*` for autocorrelation/evidence/PSRF, `eryn.utils`
`Update`/`Stopping` subclasses for the `update_fn`/`stopping_fn` hooks.

## 5. Sampler mechanics

**Ensemble + tempering.** Each proposal iteration draws a random move
from the weighted `moves` list and calls `move.propose(model, state)`.
Red/blue-family moves (`StretchMove`) split walkers into `nsplits`
(default 2) sub-ensembles and propose each part against the live
complement built from the others; group moves (`GroupStretchMove`) propose
all walkers at once against a stationary friends set rebuilt every
`n_iter_update` calls (no sub-ensemble split). Acceptance
uses the tempered log posterior
(`TemperatureControl.compute_log_posterior_tempered`, `logl*beta + logp`)
when `temperature_control` is set. At the end of `propose`, the stock
tempered MH, red-blue, group and NUTS moves call
`TemperatureControl.temper_comps` (`RidgeGibbsMove` makes no swaps), which proposes
adjacent-rung swaps from the hottest pair down (`temperature_swaps`) and,
if `adaptive=True`, nudges the beta ladder (`adapt_temps`, hyperbolic-decay
dynamics from arXiv:1501.05823) until `stop_adaptation` is reached.
Reversible-jump moves call `temper_comps(state, adapt=False)`: they swap
but do not adapt (and skip swaps with `prevent_swaps=True`). The swap
counts stored in the backend come from the last in-model move.

**Hooks.** `update_fn(i, state, sampler)` runs inside `sample()` every
`update_iterations` proposal iterations (also during burn-in);
`stopping_fn(i, state, sampler)` and runtime plots
(`plot_generator.produce_plots`) run in `run_mcmc` every
`stopping_iterations` / `plot_iterations` yielded states.

**Reversible jump / branches-leaves-nleaves.** Each branch has a fixed
`nleaves_max` (array capacity) and a per-iteration `inds` bool mask
marking which slots are "alive" (the trans-dimensional state); dead slots
keep stale coordinates in memory but are excluded from prior/likelihood/
plots. `ReversibleJumpMove.propose` calls `get_model_change_proposal`
(concrete in `DistributionGenerateRJ`) to pick, per walker, one leaf to
add or remove (`fix_change=None` picks randomly between +1/-1, forced at
`nleaves_min`/`nleaves_max`), draws new-leaf coordinates from
`generate_dist` (typically the prior), and applies proposal-asymmetry
`edge_factors` (`log(1/2)` terms) at the min/max boundaries where the
+1/-1 choice is not symmetric. Plain `StretchMove` alongside RJ is
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
`nsamplers=1` (default) nothing changes anywhere, bitwise; the seeded RNG
stream is preserved. Freezing: set/replace `state.samplers_running` (e.g.
inside `update_fn`); frozen samplers' rows hold fill values (`-1e300`
logl, `-inf` logp) and their coords/betas stay bit-identical.
Per-sampler prior bounds: `prior_transform_fn` (same contract as para's:
in-place `transform_to_prior_basis(coords, running_idx)` +
`adjust_logp(logp, running_idx)`), requires `nleaves_max == 1` and no
RJ/multiple-try/`provide_groups`/`all_models_together`. Diagnostics for
`nsamplers > 1`: `get_evidence_estimate` returns per-sampler arrays;
`get_autocorr_time` raises `ValueError` and
`get_gelman_rubin_convergence_diagnostic` raises `NotImplementedError`;
runtime plotting is rejected at construction. Compute per sampler via
`get_chain(sampler_index=...)`. There is no MPI code here: the parallelism
is intra-process array batching (SIMD/GPU). Multi-process coordination
lives in LAT's `lisatools.globalfit` (`run.py`, `communication/`).

## 6. Cross-repo dependencies

**Eryn -> gpubackendtools (GBT).** `paraensemble.py` imports
`get_backend`/`get_first_backend` and `GPUBACKENDTOOLSException` from
`gpubackendtools`: Eryn's *only* cross-repo import within LISA Analysis
Tools, and the source of `cupy`/GPU array support for
`ParaEnsembleSampler` (hard dependency in `pyproject.toml`, annotated "GBT
backend registry: numpy/cupy access for paraensemble"). This is a
**different "backend" than the sampler-state `Backend`/`HDFBackend`
classes in `eryn.backends`**: GBT's `Backend` objects resolve `.xp` (numpy
vs cupy) and device selection per the LISA Analysis Tools-wide "backend
chosen at construction, not per-method-kwarg" rule
(`force_backend="cpu"|"cuda12x"`, `gpu=<index>`); `eryn.backends.Backend`
is unrelated MCMC-chain storage. Elsewhere in Eryn (`state.py`,
`moves/move.py`, `moves/stretch.py`, `moves/groupstretch.py`, `moves/eigenaxis.py`,
`moves/multipletry.py`, `priors/*`, `utils/periodic.py`,
`utils/transform.py`), GPU support is a local `try: import cupy` fallback
with a `use_gpu`/`use_cupy` flag; no GBT dependency there.

**Consumers of Eryn.** LISAanalysistools is the dominant downstream; the
9-month and 12-month global-fit runs drive Eryn as a thin shell:
- `globalfit/engine.py`: `GlobalFitEngine(EnsembleSampler)`; the head runs
  `run_mcmc -> sample` with one recipe-built `GFCombineMove`, `Recipe` as
  `stopping_fn`, `GFHDFBackend` as backend. In the stock fits the engine
  runs the cold chain only (engine `ntemps` pinned to 1; `erebor.blank` is
  the exception); module tempering uses per-leaf / per-band
  `TemperatureControl`s owned by LAT moves.
- `globalfit/state.py` (`GFState`, `ModuleSubState` subclass `State`),
  `globalfit/hdfbackend.py` (`GFHDFBackend`, `ModuleSubBackend` subclass
  `HDFBackend`), `globalfit/run.py` (`State`, `BranchSupplemental`,
  `PeriodicContainer`, and the saver's `PlotContainer`, `plots=["base"]`
  when the engine is untempered),
  `globalfit/recipe.py` (`TemperatureControl`, `make_ladder`; it relies on
  `CombineMove.accepted` raising `AttributeError`, see §7).
- `globalfit/moves/`: `GFCombineMove(CombineMove)`, `GBSpecialBase`
  (`GroupStretchMove`), `PSDMove` (`StretchMove`), `addremovemove.py` and
  `walkerfanout.py` (`TemperatureControl` swaps / pooled adaptation),
  `eigen_refresh.py` (eigenaxis helpers), `mbhspecialmove.py`,
  `functionmove.py` (`Move`).
- `sampling/`: `moves/eigenjumpmix.py` (`EigenAxisMove`,
  `draw_axis_step`), `moves/walkerdiffjump.py` and `moves/skymodehop.py`
  (`MHMove`), `ridge_fiber.py` (`RidgeGibbsMove`), `prior.py`
  (`UniformDistribution`, `multipletry.logsumexp`), `snrladder.py`
  (`make_ladder`), `fstat_proposal.py`/`gmm.py` (`ProbDistContainer`).
- `globalfit/priors/*`, `globalfit/stock/*`: `eryn.prior.uniform_dist`,
  `ProbDistContainer`, `TransformContainer`, `Move`, `StretchMove`,
  `EigenAxisMove`, `make_ladder`.
- `globalfit/moves/gbbands.py` (`Branch`, `TransformContainer`),
  `globalfit/generatefuncs.py` (`HDFBackend`), `globalfit/mbhsearch.py`
  (legacy standalone `EnsembleSampler` MBH search, not used by the global fit),
  `sampling/likelihood.py` (`Branch`, `BranchSupplemental`),
  `sampling/stopping.py` (`Stopping`,
  `thermodynamic_integration_log_evidence`), `sampling/utility.py`
  (`State`, `BranchSupplemental`, `TransformContainer`, `groups_from_inds`).
- `communication/walkerslice.py` (`BranchSupplemental`),
  `globalfit/postprocessing.py` (`HDFBackend`, `get_integrated_act`,
  `TransformContainer`), and `diagnostic.py`, `analysiscontainer.py`,
  `sources/utils.py` (`TransformContainer`).
- `ParaEnsembleSampler`/`ParaState` consumers: `globalfit/moves/gbspecialstretch.py`
  (`GBSpecialRJSerialSearchMCMC`, not in a default recipe) and
  `globalfit/warmstart/para_search.py` (behind `GB_PARA_SEARCH`, off by
  default).

Other repos: BBHx `bbhx/mbhphentax.py` imports
`eryn.utils.transform.TransformContainer`; GBGPU and BBHx list `eryn` in
their optional `sampling` extra. The FEW, GPUBackendTools and lisa-on-gpu
packages do not import Eryn (the lisa-on-gpu dev script `dev_wavelet.py`
does).

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
- **Frozen samplers must be excluded from temperature swaps** (they are:
  `temperature_swaps(samplers_running=...)`): their rungs all hold
  `-1e300` logl, so an unmasked swap would always accept (`paccept = 0`)
  and permute frozen coordinates. Relatedly, masked rows produce one-time
  `inf - inf` RuntimeWarnings in move acceptance math (`nan` -> reject);
  harmless.
- **Legacy HDF files (no `nsamplers` attr) read/write in the old layout**:
  restart from them works only with `nsamplers == 1`; `sampler_index=` and
  `get_samplers_running` are unavailable on them. New files always store
  the sampler axis (rank-6 chain datasets); raw-h5py consumers must index
  accordingly.
- **`ParaBackend.get_a_sample`/`get_last_sample`** were previously broken
  (malformed 5-D coords, dict `groups_running`, squeezed `betas`); they are
  fixed and regression-tested (`tests/test_paraensemble.py::
  ParaBackendSampleRetrievalTest`).
- **`ParaBackend.get_gelman_rubin_convergence_diagnostic` is broken**: it
  reads `self.branch_names`, `self.ndims` and `self.get_inds`, which
  `ParaBackend` does not have, so it raises `AttributeError`.
- **Delayed rejection is wired but not implemented**:
  `ReversibleJumpMove.__init__` builds a `DelayedRejection` wrapper when
  `dr=` is set (`EnsembleSampler(dr_moves=...)` passes it to the default RJ
  move), but `rj.py`'s `propose()` raises `NotImplementedError` on that
  branch; do not pass `dr_moves`.
- **`CombineMove.accepted` getter raises `AttributeError`** (it reads a
  `_accepted` attribute nothing sets). LAT `globalfit/recipe.py` relies on
  this: it tests `hasattr(combined, "accepted")` before setting the
  counters, so changing the getter to raise anything else breaks every
  LAT stage build. The `accepted` setter and the acceptance-fraction
  properties do not unwrap `(move, weight)` tuples; pass bare moves.
- **Resuming an untempered run fails**: with the default
  `tempering_kwargs={}` the sampler has `temperature_control = None` and the
  state carries no `betas`, so the backends never write them; `get_a_sample`
  still returns a `State` with the stored (zero or uninitialized) betas, and
  `sample()` then assigns them to the `None` temperature control
  (`AttributeError`). This applies to `run_mcmc(None, ...)` on a resumed
  backend and to passing `get_last_sample()` explicitly. Any non-empty
  `tempering_kwargs`, even `{"ntemps": 1}`, builds a `TemperatureControl`
  (`betas = [1.0]`) that `sample()` copies onto the state, so a run STARTED
  that way writes `betas` and resumes. Resuming an existing untempered file
  with such a sampler does not raise but loads the never-written betas rows
  (HDF5 zeros, or `np.empty` memory) into the ladder. LAT's global fit is
  NOT affected: `run.py`
  builds `GlobalFitEngine` with `tempering_kwargs={"ntemps": self.ntemps}`
  (a one-rung control in the stock fits) and resumes from
  `GFHDFBackend.get_last_sample()`, whose `get_a_sample` wraps Eryn's.
- **`EnsembleSampler` getters copy `Backend` docstrings at import**
  (`get_chain.__doc__ = Backend.get_chain.__doc__`, ...); `get_log_like`
  is assigned the `get_log_prior` docstring by mistake.
- **`sample(skip_initial_state_check=False)` raises `AttributeError`** (it
  reads `state.coords`, which `State` does not define). The default `True`
  skips the check.
- **Blobs are limited to one column**: `compute_log_like` reshapes the
  blobs to `(ntemps, nwalkers)`.
- **RJ + `StretchMove` is a documented foot-gun, not a hard error**: the
  constructor only `warnings.warn`s if a plain `StretchMove` is mixed
  with RJ; it runs but is "most likely very inefficient" since the
  live-ensemble complement ignores the per-walker active-dimension mask.
  Use `GroupStretchMove` instead.
- **`store_missing_leaves` (default `np.nan`) only overwrites dead-leaf
  coords on write**, not in the live `State`: `Branch.coords` for
  inactive leaves keeps its last real value in memory (warm-restart-able)
  but the backend (in-memory or HDF5) always gets `store_missing_leaves`
  there.
- **Backend `key_order`/move-key mismatches are hard restart-blockers**:
  `EnsembleSampler.__init__` raises `ValueError` if
  `key_order != backend.key_order`, or if `track_moves=True` and the set
  of move keys changed since the backend was written; intentional
  "start a fresh backend" guards, not bugs.
- **Priors work in the sampling variable**: `ProbDistContainer` calls the
  plain `rvs`/`logpdf`, so `LogUniform`, `PowerLaw`, `CosineUniform` and
  the other transform-backed priors are drawn and scored in their
  transformed variable; the likelihood (or a `TransformContainer`) must map
  back. The `keys=` argument of `ProbDistContainer.logpdf`/`rvs` is matched
  against column indices, not names.
- **`TransformContainer` reorders inputs only when `fill_dict` is given**;
  with input and output bases of different order and nothing to fill, pass
  `fill_dict={}`.
- **`psrf` returns only `Rhat`** (shape `(ndims,)`), not `(Rhat, neff)`.
- **`stepping_stone_log_evidence`** pools each temperature's samples over
  steps and walkers and normalises each pair term by `log(n)` (the
  regrouping and normalisation bug was fixed in `d46b5aa`). It is used by
  `Backend.get_evidence_estimate(method="stepping-stone")` and
  `plot_logl_betas`; the LAT stock fits do not reach it (engine `ntemps`
  = 1, plots `["base"]`).
- **`AdjustStretchProposalScale` raises `AttributeError` on every call**
  with an `EnsembleSampler` (it reads `sampler._moves`).
- **Tempering `fancy_swap` recomputes the likelihood on every swap**
  (`TemperatureControl.perform_fancy_swap_acceptance_fraction` calls
  `compute_log_like`); far more expensive than the default cached-value
  exchange; opt in only if the likelihood depends on the walker slot
  (e.g. a per-walker residual) rather than only on the coordinates.
- **`ParaBackend.save_step` branches on cupy vs numpy via
  `try/except AttributeError`** (on `.get()`) rather than an explicit
  `xp` check; do not assume one array type when subclassing.

## 8. Where to look for X

| Want to... | Start in |
|---|---|
| Understand the main sampling loop / how a move gets chosen and applied | `src/eryn/ensemble.py` (`EnsembleSampler.sample`) |
| Add/modify an in-model proposal | `src/eryn/moves/move.py` (base), `src/eryn/moves/mh.py` (MH guide), `src/eryn/moves/red_blue.py` + `stretch.py` (affine-invariant family) |
| Information-matrix (eigen-axis) proposals | `src/eryn/moves/eigenaxis.py` (`EigenAxisMove` + helpers); usage in `tests/test_eigenaxis.py` |
| Gibbs moves along exact likelihood degeneracies | `src/eryn/moves/ridgegibbs.py` (`RidgeGibbsMove`) |
| Add/modify a reversible-jump (trans-dimensional) proposal | `src/eryn/moves/rj.py` (base), `src/eryn/moves/distgenrj.py` (concrete default) |
| Change how accept/reject merges state (coords/inds/blobs/supplementals) | `src/eryn/moves/move.py` (`Move.update`) |
| Parallel tempering ladder, swaps, adaptation | `src/eryn/moves/tempering.py` (`TemperatureControl`) |
| Multiple-try proposals | `src/eryn/moves/multipletry.py`, `mtdistgen.py`, `mtdistgenrj.py` |
| NUTS / gradient-based sampling | `src/eryn/moves/nuts.py` |
| Priors (define/compose/sample) | `src/eryn/priors/probdist.py` (`ProbDistContainer`), `src/eryn/priors/analytical.py` (distributions); `tests/test_priors.py` |
| What gets stored per iteration / chain array shapes | `src/eryn/state.py` (`State`, `Branch`, `BranchSupplemental`) |
| In-memory vs HDF5 chain storage | `src/eryn/backends/backend.py`, `hdfbackend.py` |
| Run many independent ensembles at once (GPU-batched likelihoods) | `EnsembleSampler(nsamplers=...)` in `src/eryn/ensemble.py`; shapes/freezing in `src/eryn/state.py`; per-sampler tempering in `src/eryn/moves/tempering.py`; usage in `tests/test_nsamplers.py` |
| `ParaEnsembleSampler`-specific storage (deprecated) | `src/eryn/backends/parabackend.py` (note §7 gotcha) |
| GPU-batched multi-ensemble sampling, deprecated para path | `src/eryn/paraensemble.py` (use `nsamplers=` instead) |
| Periodic-parameter wrap/distance math | `src/eryn/utils/periodic.py` |
| Parameter reparameterization before prior/likelihood | `src/eryn/utils/transform.py` |
| Autocorrelation / evidence (thermodynamic, stepping-stone) / Gelman-Rubin | `src/eryn/utils/utility.py` |
| Diagnostic plots (corner, trace, tempering ridge, RJ leaves) | `src/eryn/utils/plot.py`, `PlotContainer` |
| Stopping criteria / mid-run callbacks | `src/eryn/utils/stopping.py`, `src/eryn/utils/updates.py` |
| Existing usage patterns for multi-branch + RJ | `tests/test_eryn.py` (`test_rj_multiple_branches`, `test_gibbs_sampling`) |
| Existing usage patterns for `ParaEnsembleSampler` | `tests/test_paraensemble.py` |
| Known defects and code TODOs | `docs/source/general/todos.rst`, plus §7 above |
