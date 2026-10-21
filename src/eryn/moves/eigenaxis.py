# -*- coding: utf-8 -*-
"""Eigen-axis / information-matrix proposal machinery.

The primitives here build per-source proposal *axes* (principal axes of an
information matrix, optionally with analytic-manifold directions injected)
and per-axis 1-sigma widths, and :class:`EigenAxisMove` turns them into a
Metropolis-Hastings proposal: jump along ONE uniformly chosen axis per leaf
(``mode="axis"``), or jointly along all of them (``mode="full"``).

Two structural invariants keep the proposal symmetric (``factors == 0``):

1. The axis/width tables are FROZEN between explicit updates
   (:meth:`EigenAxisMove.set_axes` or a refreshed branch supplemental).
   Nothing in the draw depends on the current point, so
   ``q(x -> y) == q(y -> x)`` exactly. State-dependence belongs in an
   explicit coordinate change that pays its log-Jacobian — never in the
   step size.
2. The move never *computes* axes from the ensemble. Callers own the
   information-matrix computation and hand the result in — which also
   keeps the move safe under the folded ``nsamplers`` axis (a per-row
   local jump cannot leak between independent samplers).

This is deliberately the same split as :class:`eryn.moves.RidgeGibbsMove`
and its duck-typed ``fiber_map``: the generic sampling machinery lives
here; the physics that builds fibers, ridges, and information matrices
lives with the model.
"""

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

from .mh import MHMove

__all__ = [
    "EigenAxisMove",
    "prior_box_scales",
    "project_out_direction",
    "axis_prior_bounds",
    "eigen_axis_set",
    "draw_axis_step",
]

#: Branch-supplemental keys :class:`EigenAxisMove` reads when no explicit
#: table is set for a branch.
EIGEN_AXES_SUPP_KEY = "eigen_axes"
EIGEN_SIGMAS_SUPP_KEY = "eigen_sigmas"


def _array_module(arr):
    """``cupy`` if ``arr`` lives on a GPU, else ``numpy``."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def prior_box_scales(lo, hi):
    """Per-column whitening scales = prior box widths.

    An information matrix eigendecomposed with a RELATIVE floor is not
    scale invariant: in raw sampling units, *which* directions fall under
    the floor is decided by the unit choice rather than by curvature.
    Whitening to the prior box makes every coordinate O(1) so the spectrum
    reflects real anisotropy.

    Degenerate columns (a fixed / per-leaf-filled parameter has zero prior
    width) keep a scale of 1.0 rather than dividing by zero.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    s = np.abs(hi - lo)
    s[~np.isfinite(s) | (s <= 0.0)] = 1.0
    return s


def project_out_direction(info, t):
    """``P F P`` with ``P = I - t t^T``, batched over sources.

    Removes one direction from the information matrix. Written out rather
    than materialising ``P`` so the cost stays O(ndim^2) per source.
    """
    xp = _array_module(info)
    Ft = xp.einsum("nij,nj->ni", info, t)
    tFt = xp.einsum("ni,ni->n", t, Ft)
    out = (info
           - t[:, :, None] * Ft[:, None, :]
           - Ft[:, :, None] * t[:, None, :]
           + t[:, :, None] * t[:, None, :] * tFt[:, None, None])
    return 0.5 * (out + xp.swapaxes(out, -1, -2))


def axis_prior_bounds(axes, widths):
    """Largest sensible 1-sigma step along each axis, from the prior box.

    For unit axis ``a`` the step that just leaves the box is
    ``min_i (width_i / |a_i|)`` over the components it actually moves. This
    is the scale-correct way to bound a step WITHOUT re-expressing the
    information matrix in whitened coordinates: a bare ``sigma_max = 1``
    only means "one prior width" if the coordinates were whitened first.
    Bounding per axis achieves the same end -- prior-aware, unit-correct
    step sizes -- and touches nothing else.

    ``widths`` is the per-column prior box width from
    :func:`prior_box_scales`. Components below ``1e-12`` of the axis
    norm are ignored so a direction that barely touches a narrow parameter
    is not bounded by it.
    """
    xp = _array_module(axes)
    aa = xp.abs(axes)
    big = aa > 1e-12
    ratio = xp.where(big, widths[None, :, None] / xp.where(big, aa,
                                                           xp.ones_like(aa)),
                     xp.full(aa.shape, xp.inf))
    return ratio.min(axis=1)


def eigen_axis_set(info, t_fiber=None, ridge_axis=None, sigma_max=1.0):
    """Per-source proposal axes and their own 1-sigma widths.

    Returns ``(axes, sigmas)`` with ``axes`` shaped ``(n, ndim, ndim)``
    (column ``k`` is axis ``k``) and ``sigmas`` shaped ``(n, ndim)``.

    Args:
        info: ``(n, ndim, ndim)`` information matrices (one per source).
        t_fiber: Optional ``(n, ndim)`` unit tangents of an exactly (or
            nearly) likelihood-flat direction. When given, that direction
            is projected out before the eigendecomposition and the columns
            are ordered by |overlap with the fiber| so the fiber-aligned
            eigenvector lands LAST — where it is either overwritten by
            ``ridge_axis`` or left as the (bounded) near-flat direction.
        ridge_axis: Optional ``(n, ndim)`` analytic-manifold direction to
            install in the last column (raw; it is orthogonalised against
            ``t_fiber`` when one is given, then normalised). Use this for
            model-specific ridge/manifold axes the information matrix
            cannot be trusted to produce itself.
        sigma_max: Bound on any single axis width — a genuinely flat
            direction otherwise yields ``1/sqrt(~0)``. In prior-box-
            whitened coordinates the natural bound is 1.0 = one prior
            width; it binds only on near-null axes.

    ``sigma_k = 1 / sqrt(a_k^T F a_k)`` uses the ORIGINAL information
    matrix, so each axis is scaled by its own curvature. A 1-D move pays
    no ``d``-dimensional cost penalty, which is why no relative
    eigen-floor is needed: that floor exists only because a joint draw
    must share one global scale.
    """
    xp = _array_module(info)
    if t_fiber is not None:
        Fp = project_out_direction(info, t_fiber)
        evals, evecs = xp.linalg.eigh(Fp)
        # Order columns by |overlap with the fiber| so the fiber-aligned
        # eigenvector lands last.
        ov = xp.abs(xp.einsum("ni,nij->nj", t_fiber, evecs))
        order = xp.argsort(ov, axis=-1)
        axes = xp.take_along_axis(evecs, order[:, None, :], axis=-1)
    else:
        _, axes = xp.linalg.eigh(info)
    if ridge_axis is not None:
        ridge = ridge_axis
        if t_fiber is not None:
            ridge = ridge - t_fiber * (t_fiber * ridge).sum(axis=-1,
                                                            keepdims=True)
        rn = xp.sqrt((ridge * ridge).sum(axis=-1, keepdims=True))
        ridge = ridge / xp.where(rn > 0, rn, xp.ones_like(rn))
        axes[:, :, -1] = ridge
    quad = xp.einsum("nik,nij,njk->nk", axes, info, axes)
    sigmas = 1.0 / xp.sqrt(xp.maximum(quad, 1e-300))
    return axes, xp.minimum(sigmas, float(sigma_max))


def draw_axis_step(axes, sigmas, rng, jump_factor=1.0):
    """Draw a 1-D Gaussian step along ONE uniformly chosen axis per source.

    Cost-neutral against a joint draw (still one likelihood call per
    repeat), but each direction is scaled by its own width and reports its
    own acceptance. The proposal is symmetric along a fixed axis, so the
    Metropolis-Hastings factor stays zero -- the axis set must be built
    from a FIXED information matrix and held constant while it is in use,
    so the basis does not depend on the current point.

    ``rng`` is a host random generator (``numpy.random`` ``Generator`` or
    legacy ``RandomState`` — e.g. Eryn's ``model.random``); the picks and
    variates are drawn on the host and moved to the axes' array module.

    Returns ``(dy, picked_axis)``; ``picked_axis`` is host numpy for
    per-axis acceptance counters.
    """
    xp = _array_module(axes)
    n, _, naxes = axes.shape
    if hasattr(rng, "integers"):
        pick = np.asarray(rng.integers(naxes, size=n))
        z = np.asarray(rng.standard_normal(n))
    else:                                   # legacy RandomState / cupy
        pick = np.asarray(rng.randint(0, naxes, n))
        z = np.asarray(rng.randn(n))
    pick_x = pick if xp is np else xp.asarray(pick)
    z_x = z if xp is np else xp.asarray(z)
    rows = xp.arange(n)
    a = axes[rows, :, pick_x]
    s = sigmas[rows, pick_x]
    return (float(jump_factor) * s * z_x)[:, None] * a, pick


class EigenAxisMove(MHMove):
    """Metropolis move that jumps along supplied eigen/manifold axes.

    The move OWNS no information-matrix computation: per-branch axes and
    per-axis 1-sigma widths are handed in (constructor / :meth:`set_axes`
    / an ``"eigen_axes"``/``"eigen_sigmas"`` branch supplemental) and used
    as-is. Between updates the tables are frozen, which is what makes the
    proposal exactly symmetric (``factors == 0``).

    Args:
        axes_and_sigmas (dict, optional): ``{branch_name: (axes, sigmas)}``.
            Accepted shapes per branch (``ndim`` = parameter count):

            * shared: ``axes (ndim, ndim)``, ``sigmas (ndim,)`` — one table
              for every temperature/walker/leaf;
            * per-leaf: ``axes (nleaves_max, ndim, ndim)``,
              ``sigmas (nleaves_max, ndim)``;
            * full: ``axes (ntemps, nwalkers, nleaves_max, ndim, ndim)``,
              ``sigmas (ntemps, nwalkers, nleaves_max, ndim)`` (with the
              leading axis FOLDED as ``nsamplers * ntemps`` when running
              independent samplers).

            Column ``k`` of ``axes`` is axis ``k``; ``sigmas[..., k]`` is
            its 1-sigma step. Branches with no table fall back to the
            branch supplemental keys above; neither present raises.
        mode (str, optional): ``"axis"`` draws along ONE uniformly picked
            axis per leaf (default); ``"full"`` draws jointly,
            ``dy = jump_factor * (axes * sigmas) @ z``.
        jump_factor (float, optional): Fixed scalar multiplier on every
            step. (default: ``1.0``)
        **kwargs: Forwarded to the parent :class:`Move` (``periodic``,
            ``temperature_control``, ``gibbs_sampling_setup``, ...).
            ``skip_supp_names_update`` defaults to the two eigen
            supplemental keys so frozen tables are not merged
            arithmetically on accept/reject.
    """

    _MODES = ("axis", "full")

    def __init__(self, axes_and_sigmas=None, mode="axis", jump_factor=1.0,
                 **kwargs):
        if mode not in self._MODES:
            raise ValueError(
                f"mode must be one of {self._MODES}, got {mode!r}."
            )
        self.mode = mode
        self.jump_factor = float(jump_factor)
        self._tables = {}
        if axes_and_sigmas is not None:
            for name, (axes, sigmas) in axes_and_sigmas.items():
                self.set_axes(name, axes, sigmas)
        kwargs.setdefault(
            "skip_supp_names_update",
            [EIGEN_AXES_SUPP_KEY, EIGEN_SIGMAS_SUPP_KEY],
        )
        super().__init__(**kwargs)

    def set_axes(self, name, axes, sigmas):
        """Install (or replace) the axis table for one branch.

        See the class docstring for the accepted shapes. The table is used
        as-is until the next call — the freeze between calls is what keeps
        ``factors == 0`` honest, so refresh it only at proposal-block
        boundaries, never inside a repeat sweep.
        """
        xp = _array_module(axes)
        axes = xp.asarray(axes)
        sigmas = _array_module(sigmas).asarray(sigmas)
        if axes.ndim not in (2, 3, 5):
            raise ValueError(
                "axes must be (ndim, ndim), (nleaves_max, ndim, ndim) or "
                f"(ntemps, nwalkers, nleaves_max, ndim, ndim); got shape "
                f"{axes.shape}."
            )
        if axes.shape[-1] != axes.shape[-2]:
            raise ValueError(f"axes must be square in the last two "
                             f"dimensions; got shape {axes.shape}.")
        if sigmas.ndim != axes.ndim - 1 or sigmas.shape != axes.shape[:-1]:
            raise ValueError(
                f"sigmas shape {sigmas.shape} does not match axes shape "
                f"{axes.shape} (expected {axes.shape[:-1]})."
            )
        self._tables[name] = (axes, sigmas)

    def _rows_for(self, name, inds_here, nrows, ndim, branch_supps, xp):
        """Resolve per-row ``(axes, sigmas)`` for the alive leaves."""
        table = self._tables.get(name)
        if table is not None:
            axes, sigmas = table
            axes = xp.asarray(axes)
            sigmas = xp.asarray(sigmas)
            if axes.shape[-1] != ndim:
                raise ValueError(
                    f"axes table for branch '{name}' has ndim "
                    f"{axes.shape[-1]}, coords have ndim {ndim}."
                )
            if axes.ndim == 2:
                return (xp.broadcast_to(axes, (nrows, ndim, ndim)),
                        xp.broadcast_to(sigmas, (nrows, ndim)))
            if axes.ndim == 3:
                leaf_idx = inds_here[2]
                return axes[leaf_idx], sigmas[leaf_idx]
            return axes[inds_here], sigmas[inds_here]

        supp = None if branch_supps is None else branch_supps.get(name)
        if supp is not None and EIGEN_AXES_SUPP_KEY in supp.holder \
                and EIGEN_SIGMAS_SUPP_KEY in supp.holder:
            sub = supp[inds_here]
            return sub[EIGEN_AXES_SUPP_KEY], sub[EIGEN_SIGMAS_SUPP_KEY]

        raise ValueError(
            f"No eigen axes available for branch '{name}': provide them "
            "via the constructor / set_axes, or as 'eigen_axes' + "
            "'eigen_sigmas' entries in the branch supplemental."
        )

    def get_proposal(self, branches_coords, random, branches_inds=None,
                     supps=None, branch_supps=None, **kwargs):
        """Propose a symmetric eigen-axis jump for every alive leaf.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim].
            random (object): Host random state (``model.random``).
            branches_inds (dict, optional): Keys are ``branch_names`` and
                values are np.ndarray[ntemps, nwalkers, nleaves_max]
                marking alive leaves. ``None`` treats every leaf as alive.
            supps (ignored): For signature compatibility.
            branch_supps (dict, optional): Per-branch
                :class:`eryn.state.BranchSupplemental`; used as the axis
                source for branches with no explicit table.
            **kwargs (ignored): For signature compatibility.

        Returns:
            tuple: ``(q, factors)`` — proposed coordinates dict and
            ``np.zeros((ntemps, nwalkers))`` (the draw is symmetric).

        Every alive leaf is one row: leaves of the same walker draw their
        axes INDEPENDENTLY, and dead leaves are returned byte-identical.
        The leaf axis is never folded into the walker/temperature axes.
        """
        q = {}
        for name, coords in branches_coords.items():
            xp = _array_module(coords)
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            inds_here = _array_module(inds).where(inds == True)  # noqa: E712

            axes_rows, sigmas_rows = self._rows_for(
                name, inds_here, len(inds_here[0]), ndim, branch_supps, xp
            )

            if self.mode == "axis":
                dy, _ = draw_axis_step(
                    axes_rows, sigmas_rows, random,
                    jump_factor=self.jump_factor,
                )
            else:
                nrows = len(inds_here[0])
                if hasattr(random, "standard_normal"):
                    z = np.asarray(random.standard_normal((nrows, ndim)))
                else:
                    z = np.asarray(random.randn(nrows, ndim))
                z_x = z if xp is np else xp.asarray(z)
                dy = self.jump_factor * xp.einsum(
                    "nik,nk->ni", axes_rows * sigmas_rows[:, None, :], z_x
                )

            new_coords = coords.copy()
            new_coords[inds_here] = coords[inds_here] + dy
            q[name] = new_coords

        # handle periodic parameters (not automatic anywhere upstream)
        if self.periodic is not None:
            wrapped = self.periodic.wrap(
                {
                    name: tmp.reshape((-1,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )
            q = {
                name: wrapped[name].reshape(q[name].shape)
                for name in q
            }

        return q, np.zeros((ntemps, nwalkers))
