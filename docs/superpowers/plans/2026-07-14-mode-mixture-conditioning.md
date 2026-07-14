# Mode-Mixture Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ConditionalFlowMove` proposals work on multimodal leaves by estimating each leaf's mode structure from the training buffer and proposing from an exact mixture density q(x|leaf) = Σ_m w_m q(x|leaf, m).

**Architecture:** A new `ModeMixtureFlow(ZukoFlow)` wrapper clusters each leaf's buffer into K ≤ kmax "mode" components (GMM + BIC, in a standardized cos/sin-embedded space), trains the underlying conditional NSF on **composite integer conditions** `cid = leaf * kmax + slot` (so per-condition whitening becomes per-*island* whitening and `ZukoFlow.fit`/`WhiteningTransform`/executors need zero changes), and exposes the same `sample_and_log_prob(n, context=leaf)` / `log_prob(x, context=leaf)` API the move already uses — internally drawing the component from the buffer weights and always evaluating the **mixture** log-density at both proposed and current points, which is what keeps Metropolis-Hastings exact regardless of clustering quality.

**Tech Stack:** Python 3.12, torch + zuko (via existing `ZukoFlow`), scikit-learn 1.9 (`GaussianMixture`), scipy (`linear_sum_assignment`), pytest. Repo: `/data/asantini/globalfit/erebor_org_setup/Eryn`, branch `feat-conditionalflow`.

## Global Constraints

- Python interpreter is ALWAYS `/data/asantini/globalfit/erebor_org_setup/.venv/bin/python`; every test command is prefixed `CUDA_VISIBLE_DEVICES=""` (CPU only — GPUs 4/5/6 carry a live production run; GPU 7 is reserved by the orchestrator).
- Never read or write anything under `/data/asantini/globalfit/erebor_org_setup/mojito_runs/`.
- `ZukoFlow`, `WhiteningTransform`, `ConditionalFlowMove`, and `eryn/flows/executors.py` must NOT be modified — the whole feature lives in new files plus `conditioning.py`. (`ZukoFlow.fit` coerces condition ids with `int(cond_id)` at `src/eryn/flows/torch/flows.py:972` — composite integer ids are the designed-for path.)
- MH exactness invariant: the density used in the move's factors must be the mixture `logsumexp_m [log w_m + logq(x | cid(leaf,m))]` evaluated identically for proposed and current points, with the same `base_scale`. Clustering quality may affect fit quality, never validity.
- `base_scale` (temperature-scaled base, commit c434068) must pass through every wrapper method unchanged; `base_scale=None` default paths stay bit-identical.
- The wrap-cut contract is pinned: never `allclose`-test rows that cross a periodic wrap cut; assert median round-trip / bulk agreement instead.
- Commit messages end with the two trailer lines:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01QqDRxEhoubyngcWnhZTLX9`

## File Structure

- `src/eryn/flows/conditioning.py` (modify): add `LeafModeConditioning` — composite-id → concat(leaf one-hot, mode one-hot).
- `src/eryn/flows/modes.py` (create): buffer clustering — `estimate_modes()` + `ModeState`. Pure numpy/sklearn/scipy; no torch.
- `src/eryn/flows/torch/mixture.py` (create): `ModeMixtureFlow(ZukoFlow)` — fit-time clustering, mixture sampling/density, snapshot & HDF5 persistence of the mixture state.
- `src/eryn/flows/__init__.py` (modify): export the two new public names.
- `tests/test_leaf_mode_conditioning.py`, `tests/test_estimate_modes.py`, `tests/test_mode_mixture_flow.py` (create).
- Downstream (lisa-analysis-tools, separate task): `mojito_input/emri_mbh_psd_settings.py` MBH flow block; harness validation.

---

### Task 1: `LeafModeConditioning`

**Files:**
- Modify: `src/eryn/flows/conditioning.py` (append after `OneHotLeafConditioning`; add to `__all__`)
- Modify: `src/eryn/flows/__init__.py` (export)
- Test: `tests/test_leaf_mode_conditioning.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LeafModeConditioning(nleaves_max: int, kmax: int)` with `context_dim = nleaves_max + kmax`, `encode(cid: int) -> np.ndarray(float32, (context_dim,))` where `leaf, slot = divmod(cid, kmax)`, and `assign(...)` raising `NotImplementedError`. Later tasks rely on exactly this composite-id convention.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_leaf_mode_conditioning.py
import numpy as np
import pytest

from eryn.flows.conditioning import ConditioningStrategy, LeafModeConditioning


def test_encode_concatenates_leaf_and_mode_onehots():
    cond = LeafModeConditioning(nleaves_max=6, kmax=8)
    assert cond.context_dim == 14
    ctx = cond.encode(3 * 8 + 5)  # leaf 3, slot 5
    assert ctx.shape == (14,) and ctx.dtype == np.float32
    assert ctx.sum() == 2.0
    assert ctx[3] == 1.0            # leaf one-hot block [0:6)
    assert ctx[6 + 5] == 1.0        # mode one-hot block [6:14)


def test_encode_rejects_out_of_range():
    cond = LeafModeConditioning(nleaves_max=2, kmax=4)
    with pytest.raises(ValueError):
        cond.encode(2 * 4)          # leaf 2 out of range
    with pytest.raises(ValueError):
        cond.encode(-1)


def test_protocol_and_assign():
    cond = LeafModeConditioning(nleaves_max=2, kmax=4)
    assert isinstance(cond, ConditioningStrategy)
    with pytest.raises(NotImplementedError):
        cond.assign(np.zeros(3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/asantini/globalfit/erebor_org_setup/Eryn && CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_leaf_mode_conditioning.py -q`
Expected: FAIL / error — `ImportError: cannot import name 'LeafModeConditioning'`.

- [ ] **Step 3: Implement**

Append to `src/eryn/flows/conditioning.py` (mirror `OneHotLeafConditioning`'s docstring style) and add `"LeafModeConditioning"` to `__all__`:

```python
class LeafModeConditioning:
    """Composite (leaf, mode-slot) one-hot conditioning.

    Condition ids are composite integers ``cid = leaf * kmax + slot`` with
    ``leaf in [0, nleaves_max)`` and ``slot in [0, kmax)``.  ``encode``
    returns the concatenation of the leaf one-hot (length ``nleaves_max``)
    and the mode-slot one-hot (length ``kmax``).  Used by
    :class:`eryn.flows.torch.mixture.ModeMixtureFlow`, which estimates the
    mode slots from the training buffer each round.
    """

    def __init__(self, nleaves_max: int, kmax: int):
        self.nleaves_max = int(nleaves_max)
        self.kmax = int(kmax)
        self.context_dim = self.nleaves_max + self.kmax

    def encode(self, condition_id: int) -> np.ndarray:
        cid = int(condition_id)
        leaf, slot = divmod(cid, self.kmax)
        if not (0 <= leaf < self.nleaves_max) or cid < 0:
            raise ValueError(
                f"composite condition_id {cid} out of range "
                f"[0, {self.nleaves_max * self.kmax})"
            )
        vec = np.zeros(self.context_dim, dtype=np.float32)
        vec[leaf] = 1.0
        vec[self.nleaves_max + slot] = 1.0
        return vec

    def assign(self, coords_summary: np.ndarray) -> int:
        raise NotImplementedError(
            "LeafModeConditioning.assign is not used: ModeMixtureFlow "
            "marginalizes over mode slots instead of assigning points."
        )
```

In `src/eryn/flows/__init__.py`, add `LeafModeConditioning` next to the existing `OneHotLeafConditioning` import/export.

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2. Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/eryn/flows/conditioning.py src/eryn/flows/__init__.py tests/test_leaf_mode_conditioning.py
git commit -m "feat: LeafModeConditioning for composite (leaf, mode) conditions"
```
(with the two trailer lines from Global Constraints)

---

### Task 2: `estimate_modes` clustering utility

**Files:**
- Create: `src/eryn/flows/modes.py`
- Modify: `src/eryn/flows/__init__.py` (export `estimate_modes`, `ModeState`)
- Test: `tests/test_estimate_modes.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used verbatim by Task 3):

```python
@dataclasses.dataclass
class ModeState:
    slots: list[int]              # populated slot ids, sorted
    weights: dict[int, float]     # slot -> weight, floored + renormalized, sums to 1
    centers: dict[int, np.ndarray]  # slot -> center in the embedded space
    labels: np.ndarray            # (N,) slot id per input row
    embed_mean: np.ndarray        # (D_emb,) standardization used for embedding
    embed_std: np.ndarray         # (D_emb,)

def embed(x, periodic) -> np.ndarray            # (N, D_emb) raw embedding
def estimate_modes(x, periodic, kmax, *, prev: ModeState | None = None,
                   floor: float = 0.02, min_rows: int = 25,
                   seed: int = 0) -> ModeState
```

`periodic` is the same `{dim: (low, high)}` dict the flows use. Embedding: periodic dim d with period T=high−low is replaced by two columns (cos(2πx_d/T), sin(2πx_d/T)); non-periodic dims pass through; then every column is standardized ((x−mean)/std, std floored at 1e-12). K is chosen by BIC over K=1..kmax (`GaussianMixture(n_components=K, covariance_type="full", reg_covar=1e-6, random_state=seed)`). Components with fewer than `min_rows` member rows are dissolved: their rows are re-labeled to the nearest surviving center (guaranteeing every surviving slot has ≥ min_rows or K collapses to 1). Slot ids are made **stable across rounds**: when `prev` is given, new centers are matched to `prev.centers` with `scipy.optimize.linear_sum_assignment` on the pairwise distance matrix (computed after re-standardizing prev centers into the new embed frame via `prev.embed_mean/std`); matched components inherit the previous slot id, unmatched ones take the lowest free slot in `[0, kmax)`. Weights are the label fractions, floored at `floor` and renormalized.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_estimate_modes.py
import numpy as np

from eryn.flows.modes import ModeState, embed, estimate_modes

PER = {1: (0.0, 2 * np.pi)}          # dim 1 periodic


def _three_islands(rng, n=900):
    """3 well-separated islands in (linear, angle) space; island 2 straddles the wrap."""
    c = [(-5.0, 1.0), (5.0, 3.0), (0.0, 6.2)]
    xs = []
    for i, (a, b) in enumerate(c):
        x = np.column_stack([
            a + 0.1 * rng.standard_normal(n // 3),
            (b + 0.05 * rng.standard_normal(n // 3)) % (2 * np.pi),
        ])
        xs.append(x)
    return np.concatenate(xs), np.repeat([0, 1, 2], n // 3)


def test_embed_shapes_and_wrap_continuity():
    x = np.array([[1.0, 0.01], [1.0, 2 * np.pi - 0.01]])
    e = embed(x, PER)
    assert e.shape == (2, 3)                     # 1 linear + cos + sin
    assert np.linalg.norm(e[0] - e[1]) < 0.1     # wrap-adjacent points embed close


def test_finds_three_islands_and_weights_sum_to_one():
    rng = np.random.default_rng(0)
    x, true = _three_islands(rng)
    st = estimate_modes(x, PER, kmax=8, seed=0)
    assert len(st.slots) == 3
    assert abs(sum(st.weights.values()) - 1.0) < 1e-12
    assert all(w >= 0.02 for w in st.weights.values())
    # labels must partition the data consistently with the true islands
    for t in range(3):
        lab = st.labels[true == t]
        assert (lab == lab[0]).mean() > 0.99


def test_unimodal_collapses_to_one_slot():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((600, 2)) * [1.0, 0.1] + [0.0, 3.0]
    st = estimate_modes(x, PER, kmax=8, seed=0)
    assert st.slots == [0] and st.weights[0] == 1.0
    assert (st.labels == 0).all()


def test_slot_ids_stable_across_rounds():
    rng = np.random.default_rng(2)
    x1, _ = _three_islands(rng)
    st1 = estimate_modes(x1, PER, kmax=8, seed=0)
    x2, _ = _three_islands(rng)          # fresh draw, same islands
    st2 = estimate_modes(x2, PER, kmax=8, prev=st1, seed=1)
    assert set(st2.slots) == set(st1.slots)
    for s in st1.slots:                  # matched slots point at the same island
        d = np.linalg.norm(st1.centers[s] - st2.centers[s])
        assert d < 1.0


def test_tiny_components_are_dissolved():
    rng = np.random.default_rng(3)
    x, _ = _three_islands(rng, n=900)
    x = np.concatenate([x, [[20.0, 1.0]] * 3])   # 3-row spur, below min_rows
    st = estimate_modes(x, PER, kmax=8, min_rows=25, seed=0)
    assert len(st.slots) == 3                    # spur absorbed, not a slot
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_estimate_modes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eryn.flows.modes'`.

- [ ] **Step 3: Implement `src/eryn/flows/modes.py`**

```python
"""Buffer-driven mode estimation for mixture flow proposals.

``estimate_modes`` clusters one leaf's training buffer into K <= kmax
components (GMM + BIC in a standardized cos/sin-embedded space).  The result
shapes the proposal ONLY — MH exactness never depends on the clustering
(see ModeMixtureFlow: the proposal density is the exact mixture).
Selection is deliberately split-biased: under-splitting recreates NSF
island-bridging (the failure this feature removes); over-splitting merely
tiles an island across two components.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

__all__ = ["ModeState", "embed", "estimate_modes"]


@dataclasses.dataclass
class ModeState:
    slots: list[int]
    weights: dict[int, float]
    centers: dict[int, np.ndarray]
    labels: np.ndarray
    embed_mean: np.ndarray
    embed_std: np.ndarray


def embed(x: np.ndarray, periodic: dict) -> np.ndarray:
    """Map (N, ndim) coords to a wrap-free embedding: periodic dim ->
    (cos, sin) pair on its own period; non-periodic dims pass through."""
    x = np.asarray(x, dtype=np.float64)
    cols = []
    for d in range(x.shape[1]):
        if d in periodic:
            low, high = periodic[d]
            ang = 2.0 * np.pi * (x[:, d] - low) / (high - low)
            cols.append(np.cos(ang))
            cols.append(np.sin(ang))
        else:
            cols.append(x[:, d])
    return np.column_stack(cols)


def _standardize(e: np.ndarray):
    mean = e.mean(axis=0)
    std = np.maximum(e.std(axis=0), 1e-12)
    return (e - mean) / std, mean, std


def estimate_modes(
    x: np.ndarray,
    periodic: dict,
    kmax: int,
    *,
    prev: ModeState | None = None,
    floor: float = 0.02,
    min_rows: int = 25,
    seed: int = 0,
) -> ModeState:
    e_raw = embed(x, periodic)
    z, mean, std = _standardize(e_raw)
    n = z.shape[0]

    # --- BIC scan (warm-started at the previous K when available) ---
    best = None
    for k in range(1, int(kmax) + 1):
        kwargs = dict(
            n_components=k, covariance_type="full",
            reg_covar=1e-6, random_state=seed,
        )
        if prev is not None and k == len(prev.slots):
            prev_c = np.stack([prev.centers[s] for s in prev.slots])
            # re-standardize previous centers into the new embed frame
            kwargs["means_init"] = (prev_c * prev.embed_std + prev.embed_mean - mean) / std
        gm = GaussianMixture(**kwargs).fit(z)
        bic = gm.bic(z)
        if best is None or bic < best[0]:
            best = (bic, gm)
    gm = best[1]
    labels = gm.predict(z)

    # --- dissolve tiny components into their nearest surviving center ---
    while True:
        keep = [c for c in np.unique(labels) if (labels == c).sum() >= min_rows]
        if not keep:
            labels = np.zeros(n, dtype=int)
            keep = [0]
            break
        drop = [c for c in np.unique(labels) if c not in keep]
        if not drop:
            break
        centers_keep = {c: z[labels == c].mean(axis=0) for c in keep}
        for c in drop:
            rows = labels == c
            dists = np.stack([
                np.linalg.norm(z[rows] - centers_keep[k2], axis=1) for k2 in keep
            ])
            labels[rows] = np.asarray(keep)[np.argmin(dists, axis=0)]

    comps = sorted(np.unique(labels))
    centers = {c: z[labels == c].mean(axis=0) for c in comps}

    # --- stable slot ids: match to prev by linear_sum_assignment ---
    slot_of = {}
    used = set()
    if prev is not None and prev.slots:
        prev_c = np.stack([prev.centers[s] for s in prev.slots])
        prev_c = (prev_c * prev.embed_std + prev.embed_mean - mean) / std
        new_c = np.stack([centers[c] for c in comps])
        cost = np.linalg.norm(new_c[:, None, :] - prev_c[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            slot_of[comps[i]] = prev.slots[j]
            used.add(prev.slots[j])
    free = iter([s for s in range(int(kmax)) if s not in used])
    for c in comps:
        if c not in slot_of:
            slot_of[c] = next(free)

    out_labels = np.array([slot_of[c] for c in labels], dtype=int)
    slots = sorted(slot_of[c] for c in comps)
    w = np.array([(out_labels == s).mean() for s in slots])
    w = np.maximum(w, floor)
    w = w / w.sum()
    return ModeState(
        slots=slots,
        weights={s: float(wi) for s, wi in zip(slots, w)},
        centers={slot_of[c]: centers[c] for c in comps},
        labels=out_labels,
        embed_mean=mean,
        embed_std=std,
    )
```

Export `estimate_modes`, `ModeState` from `src/eryn/flows/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: same command as Step 2. Expected: `5 passed`. If `test_finds_three_islands_and_weights_sum_to_one` flakes on K, the BIC scan is under-fitting: check `reg_covar` and that standardization ran — do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/eryn/flows/modes.py src/eryn/flows/__init__.py tests/test_estimate_modes.py
git commit -m "feat: buffer-driven mode estimation (GMM+BIC, stable slots) for mixture proposals"
```
(with trailers)

---

### Task 3: `ModeMixtureFlow` — fit path + snapshot/persistence

**Files:**
- Create: `src/eryn/flows/torch/mixture.py`
- Modify: `src/eryn/flows/__init__.py` (export `ModeMixtureFlow`)
- Test: `tests/test_mode_mixture_flow.py` (fit/snapshot tests; Task 4 adds sampling tests to the same file)

**Interfaces:**
- Consumes: `LeafModeConditioning` (Task 1), `estimate_modes`/`ModeState` (Task 2), `ZukoFlow` (unchanged).
- Produces (relied on by Task 4 and downstream):

```python
class ModeMixtureFlow(ZukoFlow):
    def __init__(self, dims, nleaves_max, kmax=8, mode_floor=0.02,
                 min_rows_per_component=25, cluster_seed=0,
                 periodic=None, conditioning=None, **zuko_kwargs): ...
    # self.mode_state : dict[int leaf, ModeState] (empty before first fit)
    def _cid(self, leaf: int, slot: int) -> int   # leaf * kmax + slot
    def fit(self, samples, **fit_kwargs)          # samples: {leaf: (N, dims)}
    def get_snapshot(self) -> dict                # adds "mixture_state"
    def set_weights(self, obj) -> None            # installs "mixture_state" if present
    def save(self, h5_file, path="flow")          # persists mixture_state blob
    @classmethod load(...)                        # restores it
```

Construction details: `periodic` is the same `{dim: (low, high)}` dict given to `WhiteningTransform` (the wrapper needs it for `embed`); if `conditioning is None` build `LeafModeConditioning(nleaves_max, kmax)` internally (the `conditioning=None` escape hatch exists because `ZukoFlow.load` reconstructs with the unpickled conditioning). The constructor must register `nleaves_max`, `kmax`, `mode_floor`, `min_rows_per_component`, `cluster_seed`, and `periodic` in whatever config mapping `ZukoFlow.save()` serializes to `grp.attrs["config"]` — **read `ZukoFlow.__init__`/`save` first to find that mechanism** (`ZukoFlow.load` does `cls(**cfg)`, so `ModeMixtureFlow.load` must round-trip; note JSON: serialize `periodic` as `{str(dim): [low, high]}` and convert back in `__init__`).

`fit(samples, **fit_kwargs)`:
1. For each leaf, `st = estimate_modes(rows, self._periodic, self.kmax, prev=self.mode_state.get(leaf), floor=self.mode_floor, min_rows=self.min_rows_per_component, seed=self.cluster_seed)`; store `self.mode_state[leaf] = st`.
2. Build the composite dict `{self._cid(leaf, s): rows[st.labels == s] for each populated slot s}`. **Preserve row order within each component** (boolean masking already does) — `val_split="temporal"` depends on it.
3. `return super().fit(composite_dict, **fit_kwargs)`.

Snapshot atomicity: `get_snapshot()` = `d = super().get_snapshot(); d["mixture_state"] = copy.deepcopy(self.mode_state); return d`. `set_weights(obj)`: if `isinstance(obj, dict) and "net" in obj and "mixture_state" in obj`, install `self.mode_state = obj.pop-free copy` **before** calling `super().set_weights(obj)` (do not mutate the caller's dict — read the "net" sentinel contract in `BaseTorchFlow.set_weights`, src/eryn/flows/torch/flows.py:162). `save`/`load`: pickle `self.mode_state` into a `"mixture_state"` dataset next to the existing `"data_transform"` blob; on load, restore it after construction (absent dataset → empty dict, i.e. old checkpoints load fine).

- [ ] **Step 1: Write the failing tests** (append to a new `tests/test_mode_mixture_flow.py`)

```python
# tests/test_mode_mixture_flow.py
import numpy as np
import pytest

from eryn.flows import ModeMixtureFlow, WhiteningTransform

PER = {2: (0.0, 2 * np.pi)}
DIMS = 3
FIT_KW = dict(n_epochs=3, batch_size=256, lr=1e-3, patience=10,
              validation_fraction=0.15, val_split="temporal", verbose=False)


def _make_flow(kmax=4, seed=7):
    return ModeMixtureFlow(
        dims=DIMS, nleaves_max=2, kmax=kmax, cluster_seed=0,
        periodic=PER, flow_class="NSF", device="cpu",
        data_transform=WhiteningTransform(ndim=DIMS, periodic=PER, shared=False),
        seed=seed, transforms=2, hidden_features=(16, 16), bins=4,
    )


def _bimodal_leaf(rng, n=600):
    a = np.column_stack([rng.normal(-4, 0.1, n // 2), rng.normal(0, 0.1, n // 2),
                         rng.vonmises(0.5, 20, n // 2) % (2 * np.pi)])
    b = np.column_stack([rng.normal(4, 0.1, n // 2), rng.normal(2, 0.1, n // 2),
                         rng.vonmises(3.0, 20, n // 2) % (2 * np.pi)])
    return np.concatenate([a, b])


def _unimodal_leaf(rng, n=600):
    return np.column_stack([rng.normal(0, 0.5, n), rng.normal(-1, 0.2, n),
                            rng.vonmises(1.0, 20, n) % (2 * np.pi)])


def test_fit_builds_composite_conditions_and_mode_state():
    rng = np.random.default_rng(0)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    assert set(fl.mode_state) == {0, 1}
    assert len(fl.mode_state[0].slots) == 2
    assert fl.mode_state[1].slots == [0]
    # per-island whitening: the transform was fitted per composite id
    for s in fl.mode_state[0].slots:
        cid = fl._cid(0, s)
        z = fl.data_transform.forward(_bimodal_leaf(rng)[:5], cid)
        assert np.asarray(z).shape == (5, DIMS)


def test_snapshot_roundtrip_carries_mixture_state():
    rng = np.random.default_rng(1)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    snap = fl.get_snapshot()
    assert "mixture_state" in snap and "net" in snap
    fl2 = _make_flow(seed=8)                     # different init
    fl2.set_weights(snap)
    assert set(fl2.mode_state[0].weights) == set(fl.mode_state[0].weights)
    x, lq = fl2.sample_and_log_prob(8, context=0)
    assert x.shape == (8, DIMS) and np.isfinite(lq).all()


def test_h5_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    p = str(tmp_path / "mix.h5")
    fl.save(p)
    fl2 = ModeMixtureFlow.load(p)
    assert set(fl2.mode_state) == {0, 1}
    x = rng.standard_normal((4, DIMS)); x[:, 2] = np.abs(x[:, 2]) % (2 * np.pi)
    np.testing.assert_allclose(fl2.log_prob(x, context=0), fl.log_prob(x, context=0),
                               rtol=0, atol=1e-6)
```

(`test_snapshot_roundtrip...` and `test_h5_save_load...` exercise `sample_and_log_prob`/`log_prob` with a bare leaf context — implemented in Task 4; for this task's commit, implement fit/snapshot/persistence AND the Task 4 methods' signatures as thin `raise NotImplementedError` ONLY IF you split the commits; otherwise implement Tasks 3+4 back-to-back and run the full file before the first commit. Recommended: implement Task 3, run only `test_fit_builds_composite_conditions_and_mode_state`, commit; then Task 4.)

- [ ] **Step 2: Run the fit test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_mode_mixture_flow.py::test_fit_builds_composite_conditions_and_mode_state -q`
Expected: FAIL — `ImportError: cannot import name 'ModeMixtureFlow'`.

- [ ] **Step 3: Implement fit/snapshot/persistence in `src/eryn/flows/torch/mixture.py`** per the Interfaces block above. Read first: `ZukoFlow.__init__` (config capture), `BaseTorchFlow.get_snapshot`, `set_weights` (flows.py:162), `save`/`load` (flows.py:240–360). Keep the module torch-free except through `ZukoFlow`.

- [ ] **Step 4: Run the fit test to verify it passes**

Run: same as Step 2. Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/eryn/flows/torch/mixture.py src/eryn/flows/__init__.py tests/test_mode_mixture_flow.py
git commit -m "feat: ModeMixtureFlow fit path — per-island conditions, snapshot + h5 persistence"
```
(with trailers)

---

### Task 4: `ModeMixtureFlow` — mixture sampling and density

**Files:**
- Modify: `src/eryn/flows/torch/mixture.py`
- Test: `tests/test_mode_mixture_flow.py` (append)

**Interfaces:**
- Consumes: Task 3's class and `mode_state`.
- Produces: `sample_and_log_prob(n, context=leaf, base_scale=None)` and `log_prob(x, context=leaf, base_scale=None)` where `context` is a **bare leaf id**; both return the exact mixture density. `ConditionalFlowMove` needs no changes.

Semantics (implement exactly):

```python
def log_prob(self, x, context=None, base_scale=None):
    leaf = int(context)
    st = self.mode_state.get(leaf)
    if st is None:
        raise RuntimeError(f"ModeMixtureFlow: no mode_state for leaf {leaf}; fit first.")
    comps = np.stack([
        np.log(st.weights[s])
        + super(ModeMixtureFlow, self).log_prob(x, context=self._cid(leaf, s),
                                                base_scale=base_scale)
        for s in st.slots
    ])                                   # (K, N)
    return logsumexp(comps, axis=0)      # scipy.special.logsumexp

def sample_and_log_prob(self, n, context=None, base_scale=None):
    leaf = int(context)
    st = self.mode_state[leaf]           # same guard as log_prob
    counts = self._rng.multinomial(n, [st.weights[s] for s in st.slots])
    xs = [super(ModeMixtureFlow, self).sample_and_log_prob(
              int(c), context=self._cid(leaf, s), base_scale=base_scale)[0]
          for s, c in zip(st.slots, counts) if c > 0]
    x = np.concatenate(xs, axis=0)
    x = x[self._rng.permutation(n)]      # break component<->row-position correlation
    return x, self.log_prob(x, context=leaf, base_scale=base_scale)
```

Notes the implementer must respect:
- `self._rng = np.random.default_rng(cluster_seed + 1)` created in `__init__` (component choice and shuffle only; the underlying flow keeps its own torch seeding).
- The **shuffle is mandatory**: the move maps returned rows positionally onto (temperature, walker) rows; unshuffled component-grouped draws would correlate mode with temperature.
- The reported density is `log_prob` of the mixture — deliberately NOT the per-component value from the inner `rsample_and_log_prob` — so proposed-point and current-point densities in the MH factors come from the identical code path. (Within a component these differ only on rare wrap-cut crossers — the pinned contract.)
- `logsumexp` from `scipy.special`.
- K=1 leaves: `log_prob` reduces to `log w_0 (=0) + component log_prob` — plain ZukoFlow behavior.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_mode_mixture_flow.py`)

```python
def test_kone_reduces_to_plain_component_density():
    rng = np.random.default_rng(3)
    fl = _make_flow()
    fl.fit({0: _unimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(64, context=0)
    cid = fl._cid(0, fl.mode_state[0].slots[0])
    from eryn.flows import ZukoFlow
    lp_component = ZukoFlow.log_prob(fl, x, context=cid)   # bypass mixture wrapper
    np.testing.assert_allclose(lq, lp_component, rtol=0, atol=1e-10)


def test_mixture_density_normalization_importance_identity():
    # E_{x ~ q_mix}[ q_component0(x) / q_mix(x) ] must equal w_0-weighted ratio ~ 1
    # simpler exact invariant: E_{x ~ q_mix}[ exp(lq_mix(x) - lq_mix(x)) ] == 1;
    # the load-bearing check is sample/log_prob consistency on the bulk:
    rng = np.random.default_rng(4)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(512, context=0)
    lp = fl.log_prob(x, context=0)
    diff = np.abs(lp - lq)
    assert np.median(diff) < 1e-10          # identical code path by construction
    # both islands are actually proposed
    assert (x[:, 0] < 0).any() and (x[:, 0] > 0).any()


def test_mixture_covers_both_modes_with_expected_rates():
    rng = np.random.default_rng(5)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, _ = fl.sample_and_log_prob(2000, context=0)
    frac = (x[:, 0] > 0).mean()
    assert 0.35 < frac < 0.65               # weights ~0.5/0.5


def test_base_scale_passes_through():
    rng = np.random.default_rng(6)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(256, context=0, base_scale=1.5)
    lp = fl.log_prob(x, context=0, base_scale=1.5)
    assert np.median(np.abs(lp - lq)) < 1e-10
    x1, _ = fl.sample_and_log_prob(256, context=0, base_scale=None)
    assert x[:, 0].std() > 0                # smoke: scaled draws exist and are finite
    assert np.isfinite(lq).all()


def test_move_contract_smoke():
    """ConditionalFlowMove-style usage: leaf context, factors finite."""
    rng = np.random.default_rng(7)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    old = _bimodal_leaf(rng)[:24]
    new, lq_new = fl.sample_and_log_prob(24, context=0)
    factors = fl.log_prob(old, context=0) - lq_new
    assert np.isfinite(factors).all()
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_mode_mixture_flow.py -q`
Expected: Task 3's fit test passes; the new ones FAIL (`NotImplementedError` or missing methods).

- [ ] **Step 3: Implement** the two methods per the Semantics block.

- [ ] **Step 4: Run the whole file + regression neighbors**

Run: `CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_mode_mixture_flow.py tests/test_flow_base_scale.py tests/test_flow_move.py tests/test_whitening_periodic_cholesky.py -q`
Expected: all pass (7 mixture + 7 base_scale + 18 move + 10 whitening).

- [ ] **Step 5: Commit**

```bash
git add src/eryn/flows/torch/mixture.py tests/test_mode_mixture_flow.py
git commit -m "feat: ModeMixtureFlow mixture sampling/density (exact MH marginalization)"
```
(with trailers)

---

### Task 5: Executor integration test

**Files:**
- Test: `tests/test_mode_mixture_executor.py` (create; no source changes expected)

**Interfaces:**
- Consumes: `ModeMixtureFlow` (Tasks 3–4), existing `ProcessExecutor`/`SerialExecutor` from `eryn.flows` (read `tests/test_flow_executors.py` — or the closest existing executor test file — first and mirror its setup/teardown conventions exactly).

Purpose: prove the executor round-trip ships the mixture state — worker fits on submitted per-leaf buffers, `latest_weights()` snapshot installs into a parent-side `ModeMixtureFlow` via `set_weights`, and the parent can then propose with bare leaf contexts. Use the serial/in-process executor variant if one exists (no spawned process needed to prove the contract); if only `ProcessExecutor` exists, use `worker_device="cpu"`, `torch_num_threads=1`, tiny epochs, and make teardown call its shutdown method.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mode_mixture_executor.py
import numpy as np

from eryn.flows import ModeMixtureFlow, WhiteningTransform
# executor import: mirror the existing executor test file's import exactly

PER = {2: (0.0, 2 * np.pi)}


def _flow(seed):
    return ModeMixtureFlow(
        dims=3, nleaves_max=1, kmax=4, cluster_seed=0, periodic=PER,
        flow_class="NSF", device="cpu",
        data_transform=WhiteningTransform(ndim=3, periodic=PER, shared=False),
        seed=seed, transforms=2, hidden_features=(16, 16), bins=4,
    )


def test_executor_ships_mixture_state(tmp_path):
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(-4, 0.1, 400), rng.normal(0, 0.1, 400),
                         rng.vonmises(0.5, 20, 400) % (2 * np.pi)])
    b = np.column_stack([rng.normal(4, 0.1, 400), rng.normal(2, 0.1, 400),
                         rng.vonmises(3.0, 20, 400) % (2 * np.pi)])
    buffer = {0: np.concatenate([a, b])}

    # construct the executor exactly as the existing executor tests do,
    # with fit_kwargs=dict(n_epochs=3, batch_size=256, validation_fraction=0.2,
    # val_split="temporal", verbose=False), min_train_samples=100
    ...  # executor = <mirrored construction>(_flow(1), ...)
    # submit buffer, wait for one round (mirror the existing test's wait idiom)

    version, snap = executor.latest_weights()
    assert version >= 1 and "mixture_state" in snap

    parent = _flow(2)
    parent.set_weights(snap)
    assert len(parent.mode_state[0].slots) == 2
    x, lq = parent.sample_and_log_prob(32, context=0)
    assert np.isfinite(lq).all() and (x[:, 0] < 0).any() and (x[:, 0] > 0).any()
```

(The `...` is a *mirroring instruction*, not a placeholder for invented API: copy the construction/wait/teardown lines from the existing executor test file verbatim, substituting the flow and buffer. Name that file in your report.)

- [ ] **Step 2: Run to verify it fails** — before Tasks 3–4 land it fails on import; after, it must pass without touching `executors.py`. If it fails because the executor strips unknown snapshot keys, STOP and report BLOCKED with the offending executor lines — do not patch the executor unilaterally.

- [ ] **Step 3–4: Make it pass; run the executor regression file too**

Run: `CUDA_VISIBLE_DEVICES="" /data/asantini/globalfit/erebor_org_setup/.venv/bin/python -m pytest tests/test_mode_mixture_executor.py <existing executor test file> -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mode_mixture_executor.py
git commit -m "test: executor round-trip ships ModeMixtureFlow mixture state"
```
(with trailers)

---

### Task 6: Settings wiring (lisa-analysis-tools) + docs

**Files:**
- Modify: `/data/asantini/globalfit/erebor_org_setup/lisa-analysis-tools/mojito_input/emri_mbh_psd_settings.py` (MBH flow block only, ~lines 495–520)
- Modify: `/data/asantini/globalfit/erebor_org_setup/lisa-analysis-tools/tasks/todo_flow_mbh_proposal.md` (mark Step 5 implemented)

**Interfaces:**
- Consumes: `ModeMixtureFlow`, `LeafModeConditioning` (importable from `eryn.flows`).

Replace the MBH `ZukoFlow(...)` construction with:

```python
    flow = ModeMixtureFlow(
        dims=len(input_basis),
        nleaves_max=nleaves_max_mbh,
        kmax=8,                    # sky-mode lattice size; over-splitting is cheap
        mode_floor=0.02,
        cluster_seed=general_set.random_seed,
        periodic=flow_periodic,
        flow_class="NSF",
        device="cpu",  # proposal-side net; training runs on the executor's GPU
        data_transform=WhiteningTransform(
            ndim=len(input_basis), periodic=flow_periodic, shared=False,
            periodic_in_cholesky=True,
        ),
        seed=general_set.random_seed,
        transforms=8,
        hidden_features=(128, 128, 128),
        bins=8,
    )
```

with the import added next to the existing `from eryn.flows import ...` line, a comment explaining per-island whitening + exact-mixture MH (cite `tasks/todo_flow_mbh_proposal.md` Step 5), and the now-redundant `conditioning=OneHotLeafConditioning(...)` line removed **for the MBH block only** — EMRI stays on plain `ZukoFlow` (unimodal; measured 0.50 already). Verify with `python -m py_compile` and an import smoke (`python -c "import emri_mbh_psd_settings"` is NOT possible without the data — instead: `python -c "from eryn.flows import ModeMixtureFlow"` plus `py_compile` on the settings file). Note in the settings comment that `min_train_samples` interacts with per-component `min_rows_per_component=25`: with 6000-row buffers and ≤8 components this is comfortable.

- [ ] Commit (LAT repo):

```bash
git add mojito_input/emri_mbh_psd_settings.py tasks/todo_flow_mbh_proposal.md
git commit -m "settings: MBH flow -> ModeMixtureFlow (per-island whitening, exact mixture MH)"
```
(with trailers)

---

### Task 7: Offline harness validation (measurement gate — controller-run)

**Files:**
- Modify: `/data/asantini/globalfit/erebor_org_setup/lisa-analysis-tools/scripts/diagnostics/flow_proposal_harness/train_offline_flows.py` (add a `mixture` candidate: `ModeMixtureFlow` with `pcov=True`, `train_noise=0`, `window=168`)
- Run: `train_offline_flows.py` then `score_offline_flows.py` on the reserved GPU, against a fresh backend snapshot (scratch-dir copies; see the harness README).

**Acceptance criteria (from `tasks/todo_flow_mbh_proposal.md`):**
- Multimodal MBH leaves (3/4/5): exact-MH implied acceptance ≥ 0.08 each (vs 0.005–0.012 for pcov-without-mixture).
- Unimodal leaves (0/1/2): within noise of the Step-3 numbers (no regression below ~0.15 on L1/L2).
- `score_offline_flows.py` needs one adaptation: `ModeMixtureFlow.load` for the mixture candidate (the scorer's manual CPU-load path must dispatch on a class tag stored in the checkpoint, or simply try `ModeMixtureFlow.load` first for files named `*_mixture*.h5`).

This task is a measurement, not code to merge: report the score table and update `tasks/todo_flow_mbh_proposal.md` Step 5 with the numbers. If the criteria fail, the fallback knobs (in order) are: `kmax` 8→12, `min_rows_per_component` 25→50, and per-component `bins` 8→12 — each is one config line in the harness; re-measure before touching feature code.

---

## Self-Review (done at planning time)

- **Spec coverage:** buffer-estimated modes (T2), K_max slots + stable ids (T2), per-island whitening via composite conditions (T3), exact mixture density both points (T4), weight floors (T2), snapshot/persistence atomicity (T3), executor compatibility (T5), base_scale composition (T4), settings + measurement gate (T6–7). Deliberately deferred (documented in the plan body): proposing modes with zero buffer rows (other moves + PT repopulate; floor applies over populated slots only — deviation from the earlier "floor all lattice modes" sketch, chosen because an unpopulated component has no fitted whitening to propose from).
- **Placeholder scan:** the only ellipsis is Task 5's executor-construction mirroring instruction, which names its source precisely.
- **Type consistency:** composite id `cid = leaf * kmax + slot` defined once (T1) and used identically in T3/T4 via `self._cid`; `ModeState` fields used in T3/T4 match the T2 dataclass; `base_scale` keyword name matches commit c434068.
