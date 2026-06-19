# -*- coding: utf-8 -*-
"""No-U-Turn Sampler (NUTS).

This module provides two pieces:

* :class:`NUTSSampler` is a standalone, vectorized implementation of the
  Efficient NUTS algorithm of Hoffman & Gelman (2014). It wraps a
  user-supplied gradient (and log-posterior) function and can produce
  samples on its own, independent of the rest of Eryn.

* :class:`NUTSMove` is a thin Eryn proposal that uses :class:`NUTSSampler`
  internally so the same NUTS step can be combined with the rest of Eryn's
  machinery (tempering, gibbs ordering, ensemble bookkeeping, ...).
"""

import numpy as np

from ..state import State
from .mh import MHMove

__all__ = ["NUTSSampler", "NUTSMove"]


class NUTSSampler(object):
    r"""Vectorized No-U-Turn Sampler.

    Implements the Efficient NUTS algorithm (Hoffman & Gelman 2014,
    Algorithm 3) using leapfrog integration of Hamiltonian dynamics with
    automatic trajectory-length selection via the no-U-turn criterion.

    The sampler is vectorized: a single call advances ``N`` walkers
    simultaneously through the same logical step. The user-supplied
    gradient (and log-posterior) functions must therefore accept input
    of shape ``(N, ndim)`` and return arrays of shape ``(N, ndim)`` /
    ``(N,)`` respectively.

    Args:
        grad_log_posterior_fn (callable): Gradient of the log-posterior.
            Signature: ``grad_fn(x) -> ndarray[N, ndim]`` for input
            ``x`` of shape ``(N, ndim)``. Vectorized: all walkers in
            one call.
        log_posterior_fn (callable): Log-posterior. Signature:
            ``log_fn(x) -> ndarray[N]`` for input ``x`` of shape
            ``(N, ndim)``. Vectorized.
        ndim (int): Dimensionality of the parameter space.
        metric (None, str, np.ndarray, or callable, optional): Metric /
            mass-matrix specification.

            * ``None`` / ``"flat"`` / ``"identity"`` / ``"cartesian"``:
              identity mass matrix (flat / Cartesian metric, default).
            * ``np.ndarray`` of shape ``(ndim, ndim)``: a constant
              symmetric positive-definite mass matrix ``M``. The
              kinetic energy is ``0.5 * r^T M^{-1} r``, momenta are
              drawn from ``N(0, M)`` and the velocity used in the
              U-turn check is ``v = M^{-1} r``.
            * callable ``metric_fn(x)`` -> ``ndarray[N, ndim, ndim]``:
              a *position-dependent* metric. The metric is evaluated
              once at the start of each NUTS step (so a constant /
              piecewise-constant mass matrix per trajectory) — this
              keeps the leapfrog symplectic without requiring the more
              expensive Riemannian-NUTS generalized leapfrog.

            (default: ``None``)
        step_size (float, optional): Leapfrog step size.
            (default: ``0.1``)
        max_tree_depth (int, optional): Maximum tree-doubling depth.
            The trajectory length is at most ``2**max_tree_depth``
            leapfrog steps. (default: ``10``)
        max_delta_energy (float, optional): Energy-divergence
            threshold. A trajectory branch is marked as divergent when
            ``H_new - H0 > max_delta_energy``. (default: ``1000.0``)
        adapt_step_size (bool, optional): If ``True``, adapt the step
            size by primal-dual averaging for the first ``n_adapt``
            calls to :meth:`step`. (default: ``False``)
        target_accept (float, optional): Target Metropolis-style
            acceptance for step-size adaptation. (default: ``0.8``)
        n_adapt (int, optional): Number of steps over which to adapt
            the step size before freezing it. (default: ``100``)
        random (np.random.RandomState or np.random.Generator, optional):
            Random source. If ``None``, uses ``np.random``.
            (default: ``None``)

    Attributes:
        step_size (float): Current leapfrog step size.
        last_alpha (float or None): Average acceptance probability of
            the last NUTS step (over all walkers). Useful for tuning.
        last_tree_depth (int or None): Depth reached on the last step.
    """

    def __init__(
        self,
        grad_log_posterior_fn,
        log_posterior_fn,
        ndim,
        metric=None,
        step_size=0.1,
        max_tree_depth=10,
        max_delta_energy=1000.0,
        adapt_step_size=False,
        target_accept=0.8,
        n_adapt=100,
        random=None,
    ):
        self.grad_log_posterior_fn = grad_log_posterior_fn
        self.log_posterior_fn = log_posterior_fn
        self.ndim = int(ndim)

        self.step_size = float(step_size)
        self.max_tree_depth = int(max_tree_depth)
        self.max_delta_energy = float(max_delta_energy)

        self._set_metric(metric)

        # dual-averaging adaptation parameters (Hoffman & Gelman 2014)
        self.adapt_step_size = bool(adapt_step_size)
        self.target_accept = float(target_accept)
        self.n_adapt = int(n_adapt)
        self._gamma = 0.05
        self._t0 = 10.0
        self._kappa = 0.75
        self._mu = np.log(10.0 * self.step_size)
        self._h_bar = 0.0
        self._log_eps_bar = 0.0
        self._adapt_count = 0

        # diagnostics from the most recent call
        self.last_alpha = None
        self.last_tree_depth = None

        self.random = random if random is not None else np.random

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _set_metric(self, metric):
        if metric is None or (
            isinstance(metric, str)
            and metric.lower() in ("flat", "identity", "cartesian", "euclidean")
        ):
            self._flat_metric = True
            self._metric_fn = None
            self._mass_matrix = None
            self._inv_mass_matrix = None
            self._mass_chol = None
            return

        if callable(metric):
            self._flat_metric = False
            self._metric_fn = metric
            self._mass_matrix = None
            self._inv_mass_matrix = None
            self._mass_chol = None
            return

        mass = np.atleast_2d(np.asarray(metric, dtype=float))
        if mass.shape != (self.ndim, self.ndim):
            raise ValueError(
                "metric ndarray must have shape (ndim, ndim); "
                f"got {mass.shape} for ndim={self.ndim}."
            )
        self._flat_metric = False
        self._metric_fn = None
        self._mass_matrix = mass
        self._mass_chol = np.linalg.cholesky(mass)
        self._inv_mass_matrix = np.linalg.inv(mass)

    def _refresh_position_metric(self, x):
        """If the metric is position-dependent, freeze it at ``x``."""
        if self._metric_fn is None:
            return
        M = np.asarray(self._metric_fn(x), dtype=float)
        if M.ndim == 2:
            # broadcast a single (ndim, ndim) over all walkers
            M = np.broadcast_to(M, (x.shape[0], self.ndim, self.ndim)).copy()
        if M.shape != (x.shape[0], self.ndim, self.ndim):
            raise ValueError(
                "metric_fn must return an array of shape (N, ndim, ndim); "
                f"got {M.shape}."
            )
        self._mass_matrix = M
        self._mass_chol = np.linalg.cholesky(M)
        self._inv_mass_matrix = np.linalg.inv(M)

    def _sample_momentum(self, shape):
        z = self.random.randn(*shape)
        if self._flat_metric:
            return z
        if self._mass_chol.ndim == 2:
            return z @ self._mass_chol.T
        # per-walker chol
        return np.einsum("nij,nj->ni", self._mass_chol, z)

    def _kinetic_energy(self, r):
        if self._flat_metric:
            return 0.5 * np.sum(r * r, axis=-1)
        if self._inv_mass_matrix.ndim == 2:
            invM = self._inv_mass_matrix
            return 0.5 * np.einsum("ni,ij,nj->n", r, invM, r)
        return 0.5 * np.einsum("ni,nij,nj->n", r, self._inv_mass_matrix, r)

    def _velocity(self, r):
        if self._flat_metric:
            return r
        if self._inv_mass_matrix.ndim == 2:
            return r @ self._inv_mass_matrix.T
        return np.einsum("nij,nj->ni", self._inv_mass_matrix, r)

    # ------------------------------------------------------------------
    # Leapfrog
    # ------------------------------------------------------------------

    def _safe_grad(self, x):
        g = self.grad_log_posterior_fn(x)
        return np.where(np.isfinite(g), g, 0.0)

    def _leapfrog(self, x, r, eps_signed):
        """Single leapfrog step with per-walker signed step size.

        Args:
            x: ``(N, ndim)`` positions.
            r: ``(N, ndim)`` momenta.
            eps_signed: ``(N,)`` signed step sizes (sign sets direction).
        """
        grad = self._safe_grad(x)
        eps = eps_signed[..., None]
        r_half = r + 0.5 * eps * grad
        x_new = x + eps * self._velocity(r_half)
        grad_new = self._safe_grad(x_new)
        r_new = r_half + 0.5 * eps * grad_new
        return x_new, r_new

    # ------------------------------------------------------------------
    # NUTS tree
    # ------------------------------------------------------------------

    def _hamiltonian(self, x, r):
        logp = self.log_posterior_fn(x)
        K = self._kinetic_energy(r)
        H = -logp + K
        # treat -inf log-posterior / nan as divergent
        H = np.where(np.isnan(H), np.inf, H)
        return H, logp

    def _build_tree(self, x, r, log_u, v, j, eps, H0):
        """Recursive doubling tree builder (vectorized over walkers).

        Args:
            x, r: ``(N, ndim)`` positions and momenta.
            log_u: ``(N,)`` slice variable.
            v: ``(N,)`` ``±1`` direction per walker.
            j: current tree depth.
            eps: scalar leapfrog step size.
            H0: ``(N,)`` initial Hamiltonian.

        Returns:
            dict: ``x_m``, ``r_m``, ``x_p``, ``r_p`` (subtree endpoints),
            ``x_s`` (proposed sample within the subtree), ``n`` (number
            of "in slice" points), ``s`` (subtree alive flag),
            ``alpha``, ``n_alpha`` (for step-size adaptation).
        """
        if j == 0:
            eps_signed = v * eps
            x_new, r_new = self._leapfrog(x, r, eps_signed)
            H_new, _ = self._hamiltonian(x_new, r_new)

            # Slice / multinomial weight: log w = -H_new
            n = (log_u <= -H_new).astype(np.int64)
            s = (log_u < self.max_delta_energy - H_new).astype(np.int64)

            delta_H = H0 - H_new
            # alpha = min(1, exp(delta_H)). Clip to prevent overflow.
            alpha = np.exp(np.minimum(delta_H, 0.0))
            # When delta_H >= 0 this is 1; when delta_H < 0 this is < 1
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            n_alpha = np.ones_like(alpha)

            return {
                "x_m": x_new, "r_m": r_new,
                "x_p": x_new, "r_p": r_new,
                "x_s": x_new,
                "n": n, "s": s,
                "alpha": alpha, "n_alpha": n_alpha,
            }

        # depth > 0: build two subtrees and combine
        t1 = self._build_tree(x, r, log_u, v, j - 1, eps, H0)

        v_pos = (v > 0)[:, None]
        x_start = np.where(v_pos, t1["x_p"], t1["x_m"])
        r_start = np.where(v_pos, t1["r_p"], t1["r_m"])
        t2 = self._build_tree(x_start, r_start, log_u, v, j - 1, eps, H0)

        # combine endpoints
        x_m = np.where(v_pos, t1["x_m"], t2["x_m"])
        r_m = np.where(v_pos, t1["r_m"], t2["r_m"])
        x_p = np.where(v_pos, t2["x_p"], t1["x_p"])
        r_p = np.where(v_pos, t2["r_p"], t1["r_p"])

        # progressive sampling: with probability n2 / (n1 + n2), keep t2 sample
        n1, n2 = t1["n"], t2["n"]
        n_sum = n1 + n2
        prob = np.where(
            n_sum > 0,
            n2.astype(float) / np.maximum(n_sum, 1).astype(float),
            0.0,
        )
        accept = self.random.uniform(size=v.shape) < prob
        x_s = np.where(accept[:, None], t2["x_s"], t1["x_s"])

        # U-turn check using velocities for non-identity metrics
        diff = x_p - x_m
        v_m = self._velocity(r_m)
        v_p = self._velocity(r_p)
        u_ok = ((np.sum(diff * v_m, axis=-1) >= 0.0)
                & (np.sum(diff * v_p, axis=-1) >= 0.0))
        s = t1["s"] * t2["s"] * u_ok.astype(np.int64)
        n = n1 + n2

        alpha = t1["alpha"] + t2["alpha"]
        n_alpha = t1["n_alpha"] + t2["n_alpha"]

        return {
            "x_m": x_m, "r_m": r_m,
            "x_p": x_p, "r_p": r_p,
            "x_s": x_s,
            "n": n, "s": s,
            "alpha": alpha, "n_alpha": n_alpha,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, x):
        """Take a single NUTS step for each of the ``N`` walkers in ``x``.

        Args:
            x (np.ndarray): Starting positions, shape ``(N, ndim)``.

        Returns:
            tuple: ``(x_new, accepted)`` where ``x_new`` has shape
            ``(N, ndim)`` and ``accepted`` is a boolean array of shape
            ``(N,)`` flagging walkers whose position actually changed.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        N, D = x.shape
        if D != self.ndim:
            raise ValueError(
                f"x has ndim={D} but sampler configured with ndim={self.ndim}"
            )

        self._refresh_position_metric(x)

        r0 = self._sample_momentum((N, D))
        H0, _ = self._hamiltonian(x, r0)

        log_u = np.log(self.random.uniform(size=N)) - H0

        x_m, r_m = x.copy(), r0.copy()
        x_p, r_p = x.copy(), r0.copy()
        x_s = x.copy()
        n = np.ones(N, dtype=np.int64)
        s = np.ones(N, dtype=np.int64)

        alpha_sum = np.zeros(N)
        n_alpha_sum = np.zeros(N)

        depth_reached = 0
        for j in range(self.max_tree_depth):
            if not np.any(s):
                break
            depth_reached = j + 1
            v = (self.random.randint(0, 2, size=N).astype(float) * 2.0 - 1.0)

            v_pos = (v > 0)[:, None]
            x_start = np.where(v_pos, x_p, x_m)
            r_start = np.where(v_pos, r_p, r_m)

            tree = self._build_tree(x_start, r_start, log_u, v, j, self.step_size, H0)

            # progressive sampling against the running sample
            prob = np.where(
                n > 0,
                np.minimum(1.0, tree["n"].astype(float) / np.maximum(n, 1).astype(float)),
                0.0,
            )
            prob = prob * tree["s"].astype(float) * s.astype(float)
            accept = self.random.uniform(size=N) < prob
            x_s = np.where(accept[:, None], tree["x_s"], x_s)

            # extend the appropriate endpoint
            x_m = np.where(v_pos, x_m, tree["x_m"])
            r_m = np.where(v_pos, r_m, tree["r_m"])
            x_p = np.where(v_pos, tree["x_p"], x_p)
            r_p = np.where(v_pos, tree["r_p"], r_p)

            n = n + tree["n"]

            diff = x_p - x_m
            v_m_vel = self._velocity(r_m)
            v_p_vel = self._velocity(r_p)
            u_ok = ((np.sum(diff * v_m_vel, axis=-1) >= 0.0)
                    & (np.sum(diff * v_p_vel, axis=-1) >= 0.0))
            s = s * tree["s"] * u_ok.astype(np.int64)

            alpha_sum = alpha_sum + tree["alpha"]
            n_alpha_sum = n_alpha_sum + tree["n_alpha"]

        # diagnostics
        alpha_walker = np.where(
            n_alpha_sum > 0,
            alpha_sum / np.maximum(n_alpha_sum, 1.0),
            0.0,
        )
        self.last_alpha = float(np.mean(alpha_walker))
        self.last_tree_depth = int(depth_reached)

        # step-size adaptation
        if self.adapt_step_size:
            if self._adapt_count < self.n_adapt:
                self._update_dual_averaging(self.last_alpha)
                self._adapt_count += 1
            elif self._adapt_count == self.n_adapt:
                self.step_size = float(np.exp(self._log_eps_bar))
                self._adapt_count += 1

        accepted = np.any(x_s != x, axis=-1)
        return x_s, accepted

    def _update_dual_averaging(self, mean_alpha):
        m = self._adapt_count + 1
        eta1 = 1.0 / (m + self._t0)
        self._h_bar = (1.0 - eta1) * self._h_bar + eta1 * (
            self.target_accept - mean_alpha
        )
        log_eps = self._mu - np.sqrt(m) / self._gamma * self._h_bar
        eta2 = m ** (-self._kappa)
        self._log_eps_bar = eta2 * log_eps + (1.0 - eta2) * self._log_eps_bar
        self.step_size = float(np.exp(log_eps))

    def sample(self, x0, n_samples, thin=1, store_diagnostics=False):
        """Draw ``n_samples`` NUTS samples for each walker in ``x0``.

        Args:
            x0 (np.ndarray): Initial positions, shape ``(N, ndim)``.
            n_samples (int): Number of post-thinning samples per walker.
            thin (int, optional): Keep every ``thin``-th sample.
                (default: ``1``)
            store_diagnostics (bool, optional): If ``True``, also record
                per-iteration mean acceptance and tree depth.
                (default: ``False``)

        Returns:
            np.ndarray (or tuple): Chain of shape
            ``(n_samples, N, ndim)``. If ``store_diagnostics`` is set,
            returns ``(chain, diagnostics_dict)``.
        """
        x = np.atleast_2d(np.asarray(x0, dtype=float)).copy()
        N, D = x.shape
        chain = np.empty((n_samples, N, D), dtype=float)

        diag = {"alpha": [], "tree_depth": [], "accepted": []} if store_diagnostics else None

        sample_i = 0
        step_i = 0
        while sample_i < n_samples:
            x, accepted = self.step(x)
            if store_diagnostics:
                diag["alpha"].append(self.last_alpha)
                diag["tree_depth"].append(self.last_tree_depth)
                diag["accepted"].append(accepted.copy())
            step_i += 1
            if step_i % thin == 0:
                chain[sample_i] = x
                sample_i += 1

        if store_diagnostics:
            diag["alpha"] = np.asarray(diag["alpha"])
            diag["tree_depth"] = np.asarray(diag["tree_depth"])
            diag["accepted"] = np.asarray(diag["accepted"])
            return chain, diag
        return chain


class NUTSMove(MHMove):
    r"""Eryn proposal that takes a NUTS step on every call.

    This is a thin wrapper around :class:`NUTSSampler` that plugs the
    NUTS step into Eryn's :class:`~eryn.ensemble.EnsembleSampler`. NUTS
    handles its own acceptance via slice sampling, so Eryn's MH accept
    test is bypassed: a walker is recorded as "accepted" whenever NUTS
    actually moved it.

    **Tempering.** When the move has a
    :class:`~eryn.moves.tempering.TemperatureControl` attached, NUTS
    runs against the *tempered* posterior

    .. math::

        \log P_\beta(x) = \beta\, \log\mathcal{L}(x) + \log\pi(x),

    so the gradient of the target is

    .. math::

        \nabla \log P_\beta(x) = \beta\, \nabla \log\mathcal{L}(x)
                                + \nabla \log\pi(x).

    The inverse-temperature ladder is read directly from
    ``self.temperature_control.betas`` and broadcast to a per-walker
    array of shape ``(ntemps * nwalkers,)``. The user therefore
    supplies the **untempered** likelihood gradient (and optionally the
    log-prior gradient); the move applies the per-walker ``beta``
    automatically.

    Args:
        grad_log_like_fn (callable): Gradient of the **untempered**
            log-likelihood. Signature: ``grad_fn(x) -> (N, ndim)`` for
            input ``x`` of shape ``(N, ndim)``. Vectorized across
            walkers.
        ndim (int): Number of parameters per walker.
        grad_log_prior_fn (callable, optional): Gradient of the
            log-prior. Signature: ``grad_fn(x) -> (N, ndim)``. If
            ``None``, the prior is assumed flat within its support
            (gradient zero) — appropriate for uniform priors.
            (default: ``None``)
        metric (None, str, np.ndarray, or callable, optional): See
            :class:`NUTSSampler`. (default: ``None``)
        scale (np.ndarray, optional): Convenience knob for per-parameter
            scaling. If provided, builds ``metric = diag(1 / scale**2)``
            so the natural NUTS step in dimension ``i`` is ``scale[i]``.
            Mutually exclusive with ``metric``. Required for problems
            whose parameters span very different scales (e.g.
            ``amp ~ 1e-22`` vs ``phi0 ~ 1``) — otherwise the leapfrog
            either nudges large-scale params by orders of magnitude or
            fails to move small-scale params. (default: ``None``)
        step_size (float, optional): Leapfrog step size. (default:
            ``0.1``)
        max_tree_depth (int, optional): Maximum tree doubling depth.
            (default: ``10``)
        max_delta_energy (float, optional): Energy divergence
            threshold. (default: ``1000.0``)
        adapt_step_size (bool, optional): Whether to dual-average adapt
            the step size during the first ``n_adapt`` proposals.
            (default: ``False``)
        target_accept (float, optional): Target acceptance for the
            adaptation. (default: ``0.8``)
        n_adapt (int, optional): Number of proposals over which the
            step size is adapted. (default: ``100``)
        **kwargs: Passed to :class:`MHMove`.

    Notes:
        The current implementation supports problems with a single
        branch and ``nleaves_max == 1``.
    """

    def __init__(
        self,
        grad_log_like_fn,
        ndim,
        grad_log_prior_fn=None,
        metric=None,
        scale=None,
        step_size=0.1,
        max_tree_depth=10,
        max_delta_energy=1000.0,
        adapt_step_size=False,
        target_accept=0.8,
        n_adapt=100,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ndim_per_walker = int(ndim)
        self.grad_log_like_fn = grad_log_like_fn
        self.grad_log_prior_fn = grad_log_prior_fn

        # ``scale`` is a convenience: build a diagonal mass matrix so the
        # natural step in dimension i is ``scale[i]``. Mass matrix
        # M = diag(1 / scale**2); kinetic energy K = 0.5 * sum(r_i^2 * scale_i^2),
        # so the velocity contribution to the position update is
        # ``scale_i^2 * r_i`` and a typical r_i ~ 1/scale_i (momentum
        # drawn from N(0, M)) gives a step ~ scale_i.
        if scale is not None:
            if metric is not None:
                raise ValueError("Pass either ``metric`` OR ``scale``, not both.")
            scale_arr = np.asarray(scale, dtype=float)
            if scale_arr.shape != (self.ndim_per_walker,):
                raise ValueError(
                    f"scale must have shape (ndim,) = ({self.ndim_per_walker},); got {scale_arr.shape}."
                )
            metric = np.diag(1.0 / scale_arr**2)

        # Internal NUTSSampler. The gradient and log-posterior functions
        # are patched on every ``propose`` call so they can fold in the
        # current temperature ladder and the model's likelihood / prior
        # callables.
        self._nuts = NUTSSampler(
            grad_log_posterior_fn=lambda x: np.zeros_like(x),  # patched below
            log_posterior_fn=lambda x: np.zeros(x.shape[0]),  # patched below
            ndim=self.ndim_per_walker,
            metric=metric,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            max_delta_energy=max_delta_energy,
            adapt_step_size=adapt_step_size,
            target_accept=target_accept,
            n_adapt=n_adapt,
        )

    @property
    def step_size(self):
        return self._nuts.step_size

    @step_size.setter
    def step_size(self, value):
        self._nuts.step_size = float(value)

    def _make_tempered_callables(self, model, state, ntemps, nwalkers, ndim, branch_name):
        """Build NUTS' ``grad_log_posterior_fn`` and ``log_posterior_fn``.

        The inverse-temperature vector ``beta`` of shape ``(ntemps,)``
        is taken from ``self.temperature_control.betas`` (or ``[1.0]``
        if untempered) and broadcast to a per-walker array
        ``betas_flat`` of shape ``(ntemps * nwalkers,)``. The
        likelihood gradient is multiplied by ``betas_flat``; the prior
        gradient is left untempered.
        """
        N = ntemps * nwalkers

        if self.temperature_control is not None:
            betas_t = np.asarray(self.temperature_control.betas, dtype=float)
        else:
            betas_t = np.ones(ntemps, dtype=float)
        betas_flat = np.repeat(betas_t, nwalkers)  # (ntemps*nwalkers,)

        grad_like_fn = self.grad_log_like_fn
        grad_prior_fn = self.grad_log_prior_fn

        def grad_log_posterior_fn(x):
            grad_like = np.asarray(grad_like_fn(x))
            if grad_like.shape != (N, ndim):
                raise ValueError(
                    f"grad_log_like_fn must return shape (N, ndim) = ({N}, {ndim}); "
                    f"got {grad_like.shape}."
                )
            g = betas_flat[:, None] * grad_like
            if grad_prior_fn is not None:
                g = g + np.asarray(grad_prior_fn(x))
            return g

        def log_post_fn(x):
            xr = x.reshape(ntemps, nwalkers, 1, ndim)
            q_dict = {branch_name: xr}
            logp = model.compute_log_prior_fn(q_dict, inds=state.branches_inds)
            logl, _ = model.compute_log_like_fn(
                q_dict, inds=state.branches_inds, logp=logp
            )
            # compute_log_posterior already applies beta when a
            # TemperatureControl is attached.
            logP = self.compute_log_posterior(logl, logp)
            logP = np.where(np.isfinite(logP), logP, -np.inf)
            return logP.reshape(N)

        return grad_log_posterior_fn, log_post_fn

    def propose(self, model, state):
        """Take a NUTS step using ``model`` to evaluate the posterior.

        Args:
            model (:class:`eryn.model.Model`): Carrier of the
                log-likelihood / log-prior callables.
            state (:class:`eryn.state.State`): Current sampler state.

        Returns:
            tuple: ``(new_state, accepted)``.
        """
        self.setup(state.branches_coords)

        all_branch_names = list(state.branches.keys())
        if len(all_branch_names) > 1:
            raise NotImplementedError(
                "NUTSMove currently supports only a single branch."
            )
        branch_name = all_branch_names[0]
        branch = state.branches[branch_name]
        coords = branch.coords
        ntemps, nwalkers, nleaves_max, ndim = coords.shape
        if nleaves_max != 1:
            raise NotImplementedError(
                "NUTSMove currently supports only nleaves_max == 1."
            )
        if ndim != self.ndim_per_walker:
            raise ValueError(
                f"NUTSMove configured with ndim={self.ndim_per_walker} but branch has ndim={ndim}."
            )

        N = ntemps * nwalkers
        x_flat = coords.reshape(N, ndim).copy()

        grad_log_post_fn, log_post_fn = self._make_tempered_callables(
            model, state, ntemps, nwalkers, ndim, branch_name
        )
        self._nuts.grad_log_posterior_fn = grad_log_post_fn
        self._nuts.log_posterior_fn = log_post_fn
        self._nuts.random = model.random

        x_new_flat, moved_flat = self._nuts.step(x_flat)
        x_new = x_new_flat.reshape(ntemps, nwalkers, 1, ndim)
        moved = moved_flat.reshape(ntemps, nwalkers)

        # evaluate log-like / log-prior at the new state
        q_dict = {branch_name: x_new}
        new_logp = model.compute_log_prior_fn(q_dict, inds=state.branches_inds)
        new_logl, new_blobs = model.compute_log_like_fn(
            q_dict, inds=state.branches_inds, logp=new_logp
        )

        new_state = State(
            q_dict,
            log_like=new_logl,
            log_prior=new_logp,
            blobs=new_blobs,
            inds=state.branches_inds,
        )

        accepted = moved
        state = self.update(state, new_state, accepted)

        self.accepted += accepted
        self.num_proposals += 1

        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted
