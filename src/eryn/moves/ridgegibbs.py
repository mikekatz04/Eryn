# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["RidgeGibbsMove"]


def _take_invariants(K, index):
    """Index opaque per-row invariants ``K`` down to a row subset.

    ``K`` may be a dict of 1D arrays, an ``(n, k)`` (or 1D) array, or a
    tuple/list of arrays -- anything whose leading axis is the row axis.
    """
    if isinstance(K, dict):
        return {key: np.asarray(value)[index] for key, value in K.items()}
    if isinstance(K, (tuple, list)):
        return type(K)(np.asarray(value)[index] for value in K)
    return np.asarray(K)[index]


class RidgeGibbsMove(Move):
    r"""Gibbs-style resampling along an exact 1-D likelihood degeneracy (a "fiber").

    For models with an EXACT likelihood invariance along a 1-D curve in a
    leaf's parameter space, this move resamples the position along that curve
    (the fiber) while holding the fiber's invariants -- and therefore the
    likelihood -- fixed. Because the likelihood is invariant by construction,
    the move makes ZERO likelihood evaluations: ``state.log_like`` is copied
    through untouched, and the per-leaf acceptance probability is

    .. math::

        \log\alpha = \left[\log p(x') - \log p(x)\right]
            + \left[\mu(t') - \mu(t)\right],

    where :math:`\log p` is the per-leaf prior density in the sampling basis
    and :math:`\mu(t) = ` ``fiber_map.log_fiber_measure(t, K)`` is the log of
    the fiber-measure (Jacobian) factor of the reparameterization to
    ``(t, invariants)``. The proposal ``t' ~ Uniform[t_lo, t_hi]`` uses an
    interval that depends only on the invariants, so it is identical forward
    and reverse and cancels from the acceptance ratio except for the measure
    term. There is no likelihood term, so acceptance is
    temperature-independent; the move performs no temperature swaps.

    Validity assumptions (stated, not checked):

    * Per-leaf fibers are disjoint coordinate blocks: resampling leaf ``i``
      touches only leaf ``i``'s coordinates, so per-leaf accepts with
      independent uniforms are a valid product of independent MH kernels.
    * The prior is per-leaf separable and ``per_leaf_log_prior`` is the SAME
      density (same basis, same normalization behavior under differencing)
      the sampler's prior uses -- accepted per-leaf deltas are summed into
      ``state.log_prior`` per walker in place of a full recomputation.

    Args:
        branch_name (str): Branch whose leaves are resampled.
        fiber_map (object): Duck-typed fiber parameterization with vectorized
            methods over ``(n, ndim)`` coordinate rows:

            * ``to_fiber(coords) -> (t, K)``: ``t`` is the ``(n,)`` fiber
              coordinate; ``K`` holds opaque per-row invariants (dict of 1D
              arrays, ``(n, k)`` array, or tuple of arrays -- leading axis is
              the row axis).
            * ``from_fiber(t, K, coords) -> (n, ndim)``: new coordinate rows;
              ``coords`` is passed so untouched columns carry through.
            * ``fiber_bounds(K) -> (t_lo, t_hi)``: ``(n,)`` each. An empty or
              degenerate interval (``t_hi <= t_lo``, or non-finite bounds)
              marks the row as a no-op skip.
            * ``log_fiber_measure(t, K) -> (n,)``: log of the fiber-measure
              factor; only its ``t``-dependence must be correct (constants
              cancel in the acceptance difference).
        per_leaf_log_prior (callable): ``rows (n, ndim) -> (n,)`` log prior
            density per leaf in the sampling basis (e.g. a
            :class:`eryn.priors.ProbDistContainer.logpdf`). Required --
            per-leaf acceptance is the point of the move; per-walker totals
            cannot be used.
        leaf_fraction (float, optional): Probability with which each alive
            leaf is included in a given call (random subset, like the RJ
            flip fraction). (default: ``1.0``)
        **kwargs: Passed to :class:`Move`.

    Raises:
        ValueError: Incorrect inputs.

    """

    def __init__(
        self, branch_name, fiber_map, per_leaf_log_prior, leaf_fraction=1.0, **kwargs
    ):
        Move.__init__(self, **kwargs)

        if not isinstance(branch_name, str):
            raise ValueError("branch_name must be a string.")
        for method in ("to_fiber", "from_fiber", "fiber_bounds", "log_fiber_measure"):
            if not callable(getattr(fiber_map, method, None)):
                raise ValueError(
                    f"fiber_map must provide a callable ``{method}`` method."
                )
        if not callable(per_leaf_log_prior):
            raise ValueError(
                "per_leaf_log_prior is required and must be callable "
                "(per-leaf acceptance is the point of this move)."
            )
        if not 0.0 < leaf_fraction <= 1.0:
            raise ValueError("leaf_fraction must be in (0, 1].")

        self.branch_name = branch_name
        self.fiber_map = fiber_map
        self.per_leaf_log_prior = per_leaf_log_prior
        self.leaf_fraction = leaf_fraction

    def propose(self, model, state):
        """Resample fiber coordinates for a random subset of alive leaves.

        All temperatures and walkers are handled at once: alive leaves of
        ``self.branch_name`` are flattened over ``(temp, walker, leaf)``,
        subsampled by ``leaf_fraction``, and resampled independently.
        ``state.log_like`` is copied through UNCHANGED (zero likelihood
        calls -- that is the move's purpose); ``state.log_prior`` is updated
        by the summed accepted per-leaf prior deltas.

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler information.
                Only ``model.random`` is used (no likelihood or prior calls
                through the model).
            state (:class:`eryn.state.State`): Current state of the sampler.

        Returns:
            tuple: ``(state, accepted)``
                ``accepted`` is the ``(ntemps, nwalkers)`` boolean array of
                walkers with at least one accepted leaf (bookkeeping
                convention of the other moves).

        """
        ntemps, nwalkers = state.log_like.shape
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)
        self.num_proposals += 1

        branch = state.branches[self.branch_name]

        # flatten (temp, walker, leaf) over alive leaves
        temp_inds, walker_inds, leaf_inds = np.where(branch.inds)
        if len(temp_inds) == 0:
            self._accumulate_accepted(accepted)
            return state, accepted

        # random leaf subset (independent of coordinates -> symmetric)
        if self.leaf_fraction < 1.0:
            keep = model.random.rand(len(temp_inds)) < self.leaf_fraction
            temp_inds = temp_inds[keep]
            walker_inds = walker_inds[keep]
            leaf_inds = leaf_inds[keep]
            if len(temp_inds) == 0:
                self._accumulate_accepted(accepted)
                return state, accepted

        # host-side copy of the sampled rows (the move is cheap; device
        # arrays come down via .get())
        rows = branch.coords[temp_inds, walker_inds, leaf_inds]
        if hasattr(rows, "get"):  # CuPy -> host
            rows = rows.get()
        rows = np.array(rows, dtype=np.float64, copy=True)

        t_curr, invariants = self.fiber_map.to_fiber(rows)
        t_lo, t_hi = self.fiber_map.fiber_bounds(invariants)

        # empty / degenerate / non-finite intervals are no-op skips
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(t_lo) & np.isfinite(t_hi) & (t_hi > t_lo)
        if not np.any(valid):
            self._accumulate_accepted(accepted)
            return state, accepted

        temp_inds = temp_inds[valid]
        walker_inds = walker_inds[valid]
        leaf_inds = leaf_inds[valid]
        rows = rows[valid]
        t_curr = np.asarray(t_curr)[valid]
        t_lo = t_lo[valid]
        t_hi = t_hi[valid]
        invariants = _take_invariants(invariants, valid)
        num = len(temp_inds)

        # t' ~ Uniform[t_lo, t_hi]; the interval depends only on the
        # invariants, so the uniform proposal density cancels forward/reverse
        t_new = t_lo + (t_hi - t_lo) * model.random.rand(num)
        new_rows = self.fiber_map.from_fiber(t_new, invariants, rows)

        # per-leaf acceptance: prior ratio x fiber-measure ratio only
        # (no likelihood, no beta)
        logp_old = np.asarray(self.per_leaf_log_prior(rows))
        logp_new = np.asarray(self.per_leaf_log_prior(new_rows))
        dlogp = logp_new - logp_old
        log_alpha = dlogp + (
            np.asarray(self.fiber_map.log_fiber_measure(t_new, invariants))
            - np.asarray(self.fiber_map.log_fiber_measure(t_curr, invariants))
        )

        # independent per-leaf accepts (valid: disjoint leaf blocks +
        # separable prior); NaN (e.g. -inf - -inf) rejects
        accept_leaf = np.zeros(num, dtype=bool)
        with np.errstate(invalid="ignore"):
            finite = ~np.isnan(log_alpha)
        accept_leaf[finite] = np.log(model.random.rand(finite.sum())) < log_alpha[
            finite
        ]

        if not np.any(accept_leaf):
            self._accumulate_accepted(accepted)
            return state, accepted

        # type(state), not State: the LISA global-fit engine subclasses State
        # (band_info / sub-state extras) and a plain-State rebuild would
        # silently drop them.
        new_state = type(state)(state, copy=True)

        acc_t = temp_inds[accept_leaf]
        acc_w = walker_inds[accept_leaf]
        acc_l = leaf_inds[accept_leaf]

        new_state.branches[self.branch_name].coords[acc_t, acc_w, acc_l] = new_rows[
            accept_leaf
        ]

        # log_like untouched by construction (already copied through in the
        # new State); log_prior gains the summed accepted per-leaf deltas
        if new_state.log_prior is not None:
            np.add.at(new_state.log_prior, (acc_t, acc_w), dlogp[accept_leaf])

        accepted[acc_t, acc_w] = True
        self._accumulate_accepted(accepted)

        return new_state, accepted

    def _accumulate_accepted(self, accepted):
        """Add to the per-move acceptance counters if the sampler set them up."""
        if getattr(self, "_accepted", None) is not None:
            self.accepted += accepted
