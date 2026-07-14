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

__all__ = ["ModeState", "embed", "estimate_modes"]

# Covariance floor for the BIC scan.  A handful of numerical safety margin over
# sklearn's default (1e-6) so per-component covariance estimates stay
# well-conditioned; deliberately NOT relied on to control over-splitting (see
# `_SEPARATION_SIGMAS` below) since no fixed floor does that robustly (see next
# comment).
_REG_COVAR = 1e-3

# In-sample BIC for a full-covariance GaussianMixture is anti-conservative near
# model-order boundaries: its log-likelihood is unbounded as a component's
# covariance shrinks toward singular, so the Schwarz parameter-count penalty
# alone does not reliably stop BIC from favoring K == kmax (verified empirically:
# with reg_covar at sklearn's default, BIC decreases ~monotonically out to
# K=kmax on both a well-separated multi-island buffer and a genuinely unimodal
# one -- raising reg_covar alone cannot fix this for both regimes at once, since
# the floor that suppresses spurious splitting of a unimodal blob is large enough
# to also merge genuinely-separated islands whose within-island spread is small
# relative to their separation).
#
# Fix: after BIC picks a K, collapse any two surviving components whose centers
# are not separated by at least `_SEPARATION_SIGMAS` standard deviations, measured
# along the axis connecting the two centers (not each component's total/isotropic
# spread, which can be inflated by variance in a direction unrelated to what
# separates them, e.g. a wide nuisance angle) -- the standard "c-separation"
# criterion for telling a genuine extra mode from an oversplit single one
# (Dasgupta 1999).  This is scale-adaptive (uses each fit's own component spread,
# not a global constant) and is what actually makes K-selection robust: swept
# over independent RNG draws of both fixtures (40+ seeds each) at reg_covar in
# {1e-6, 1e-3, 1e-2, 1e-1} crossed with sigmas in {3, 4, 5} -- 0
# misclassifications in every combination tried.  Do not remove without
# re-running that sweep.
_SEPARATION_SIGMAS = 4.0


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
    try:
        from scipy.optimize import linear_sum_assignment  # lazy — scipy is optional
        from sklearn.mixture import GaussianMixture  # lazy — sklearn is optional
    except ImportError as exc:
        raise ImportError(
            "estimate_modes requires scipy and scikit-learn; "
            "pip install scipy scikit-learn"
        ) from exc

    e_raw = embed(x, periodic)
    z, mean, std = _standardize(e_raw)
    n = z.shape[0]

    # --- BIC scan (warm-started at the previous K when available) ---
    best = None
    for k in range(1, int(kmax) + 1):
        kwargs = dict(
            n_components=k, covariance_type="full",
            reg_covar=_REG_COVAR, random_state=seed,
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

    # --- merge components that are not meaningfully separated (c-separation) ---
    # Separation is measured along the axis connecting the two centers (not the
    # isotropic/total spread of each component): a pair of components can have
    # large spread in a direction unrelated to what separates them (e.g. a wide
    # nuisance angle) without that spread saying anything about whether the two
    # centers are actually distinct modes.  Projecting onto the connecting axis
    # keeps the criterion sensitive to genuine separation regardless of spread in
    # orthogonal directions.
    while True:
        comps = sorted(np.unique(labels))
        if len(comps) <= 1:
            break
        centers_now = {c: z[labels == c].mean(axis=0) for c in comps}
        pair, ratio = None, None
        for i, c1 in enumerate(comps):
            for c2 in comps[i + 1:]:
                delta = centers_now[c2] - centers_now[c1]
                d = np.linalg.norm(delta)
                if d < 1e-12:
                    r = 0.0
                else:
                    axis = delta / d
                    p1 = (z[labels == c1] - centers_now[c1]) @ axis
                    p2 = (z[labels == c2] - centers_now[c2]) @ axis
                    s1 = float(np.std(p1)) if (labels == c1).sum() > 1 else 0.0
                    s2 = float(np.std(p2)) if (labels == c2).sum() > 1 else 0.0
                    r = d / max(s1, s2, 1e-6)
                if ratio is None or r < ratio:
                    ratio, pair = r, (c1, c2)
        if ratio is not None and ratio < _SEPARATION_SIGMAS:
            labels[labels == pair[1]] = pair[0]
        else:
            break

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
