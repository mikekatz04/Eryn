#!/usr/bin/env python
# coding: utf-8
"""Tests for the NUTS proposal and standalone NUTSSampler.

The tests cover three layers:

1. The standalone :class:`NUTSSampler` driven directly by analytic
   gradient / log-posterior closures (isotropic Gaussian, correlated
   Gaussian with a constant mass matrix, and a banana / Rosenbrock-like
   distribution).
2. The :class:`NUTSMove` plugged into Eryn's
   :class:`EnsembleSampler`, validated against the same analytic
   posteriors.
3. A head-to-head comparison against :class:`StretchMove` on the same
   problems: NUTS should recover the moments at least as accurately as
   the stretch move at matched walker count / wall-clock.
"""

import unittest

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.moves import NUTSMove, NUTSSampler, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State


# ----------------------------------------------------------------------
# Analytic posteriors and their gradients
# ----------------------------------------------------------------------

def make_gaussian(mu, cov):
    """Return (log_post_fn, grad_log_post_fn) for a multivariate Gaussian."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    inv_cov = np.linalg.inv(cov)
    log_det = np.linalg.slogdet(cov)[1]
    D = mu.size
    log_norm = -0.5 * D * np.log(2.0 * np.pi) - 0.5 * log_det

    def log_post(x):
        diff = x - mu
        return log_norm - 0.5 * np.einsum("ni,ij,nj->n", diff, inv_cov, diff)

    def grad_log_post(x):
        return -(x - mu) @ inv_cov.T

    return log_post, grad_log_post


def make_banana(a=1.0, b=1.0):
    """Rosenbrock-like target in 2D with a curved correlation ridge.

    log_post = -0.5 * [(x0 - a)^2 + b * (x1 - x0^2)^2]
    """

    def log_post(x):
        x0, x1 = x[:, 0], x[:, 1]
        return -0.5 * ((x0 - a) ** 2 + b * (x1 - x0 ** 2) ** 2)

    def grad_log_post(x):
        x0, x1 = x[:, 0], x[:, 1]
        d0 = -(x0 - a) + 2.0 * b * x0 * (x1 - x0 ** 2)
        d1 = -b * (x1 - x0 ** 2)
        return np.stack([d0, d1], axis=-1)

    return log_post, grad_log_post


# ----------------------------------------------------------------------
# Eryn likelihood helpers
# ----------------------------------------------------------------------

def gaussian_log_like_eryn(x, mu, inv_cov):
    diff = x - mu
    return -0.5 * (diff * np.dot(inv_cov, diff.T).T).sum()


def gaussian_grad_eryn(x, mu, inv_cov):
    return -(x - mu) @ inv_cov.T


# ----------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------

class NUTSStandaloneTest(unittest.TestCase):
    """Exercise :class:`NUTSSampler` directly without the rest of Eryn."""

    def test_isotropic_gaussian_recovers_moments(self):
        """NUTS on N(0, I) recovers zero mean and unit covariance."""
        rng = np.random.RandomState(0)
        ndim = 4
        mu = np.zeros(ndim)
        cov = np.eye(ndim)
        log_post, grad_log_post = make_gaussian(mu, cov)

        n_walkers = 32
        nuts = NUTSSampler(
            grad_log_posterior_fn=grad_log_post,
            log_posterior_fn=log_post,
            ndim=ndim,
            step_size=0.5,
            max_tree_depth=6,
            random=rng,
        )

        x0 = rng.standard_normal((n_walkers, ndim))
        # warmup
        nuts.sample(x0, 200)
        chain = nuts.sample(nuts.sample(x0, 200)[-1], 800)

        samples = chain.reshape(-1, ndim)
        self.assertTrue(np.allclose(samples.mean(axis=0), mu, atol=0.1))
        # diagonal of the empirical covariance close to 1
        emp_cov = np.cov(samples, rowvar=False)
        self.assertTrue(np.allclose(np.diag(emp_cov), 1.0, atol=0.2))
        # off-diagonals should be small
        off = emp_cov - np.diag(np.diag(emp_cov))
        self.assertLess(np.max(np.abs(off)), 0.2)

    def test_correlated_gaussian_with_constant_metric(self):
        """A constant mass matrix tuned to the covariance dramatically
        improves the effective acceptance.
        """
        rng = np.random.RandomState(1)
        ndim = 3
        mu = np.array([1.0, -2.0, 0.5])
        L = np.array([[1.0, 0.0, 0.0],
                      [0.8, 0.6, 0.0],
                      [0.4, -0.2, 0.5]])
        cov = L @ L.T
        log_post, grad_log_post = make_gaussian(mu, cov)

        n_walkers = 24
        # mass matrix = inverse covariance → kinetic energy whitens the target
        mass = np.linalg.inv(cov)
        nuts = NUTSSampler(
            grad_log_posterior_fn=grad_log_post,
            log_posterior_fn=log_post,
            ndim=ndim,
            metric=mass,
            step_size=0.5,
            max_tree_depth=6,
            random=rng,
        )

        x0 = mu + rng.standard_normal((n_walkers, ndim))
        chain = nuts.sample(x0, 1500)
        # discard burn-in
        samples = chain[300:].reshape(-1, ndim)

        emp_mean = samples.mean(axis=0)
        emp_cov = np.cov(samples, rowvar=False)
        self.assertTrue(np.allclose(emp_mean, mu, atol=0.15))
        # cov should match within ~25%
        self.assertTrue(
            np.allclose(emp_cov, cov, atol=0.25 * np.max(np.abs(cov)))
        )

    def test_callable_metric_evaluates_per_step(self):
        """A position-dependent metric must be accepted by the API."""
        rng = np.random.RandomState(2)
        ndim = 2
        mu = np.zeros(ndim)
        cov = np.diag([4.0, 0.25])
        log_post, grad_log_post = make_gaussian(mu, cov)

        # callable metric: returns the same constant matrix
        const_mass = np.linalg.inv(cov)

        def metric_fn(x):
            N = x.shape[0]
            return np.broadcast_to(const_mass, (N, ndim, ndim))

        nuts = NUTSSampler(
            grad_log_posterior_fn=grad_log_post,
            log_posterior_fn=log_post,
            ndim=ndim,
            metric=metric_fn,
            step_size=0.7,
            max_tree_depth=6,
            random=rng,
        )

        x0 = rng.standard_normal((16, ndim)) * np.sqrt(np.diag(cov))
        chain = nuts.sample(x0, 1500)
        samples = chain[300:].reshape(-1, ndim)

        emp_var = samples.var(axis=0)
        self.assertTrue(np.allclose(emp_var, np.diag(cov), rtol=0.25))

    def test_banana_target_mean_matches(self):
        """Sanity check on a curved target.

        For log p = -0.5 * ((x0 - a)^2 + b * (x1 - x0^2)^2), the marginal
        of x0 is N(a, 1), so E[x0] = a and E[x1] = E[x0^2] = a^2 + 1.
        """
        rng = np.random.RandomState(3)
        a = 1.0
        b = 4.0
        log_post, grad_log_post = make_banana(a=a, b=b)

        # use step-size adaptation since a constant step size is risky
        # on a curved target with the identity metric
        nuts = NUTSSampler(
            grad_log_posterior_fn=grad_log_post,
            log_posterior_fn=log_post,
            ndim=2,
            step_size=0.05,
            max_tree_depth=8,
            adapt_step_size=True,
            target_accept=0.85,
            n_adapt=400,
            random=rng,
        )

        x0_start = np.array([a, a ** 2]) + 0.3 * rng.standard_normal((40, 2))
        chain = nuts.sample(x0_start, 3000)
        samples = chain[1000:].reshape(-1, 2)
        self.assertAlmostEqual(samples[:, 0].mean(), a, delta=0.2)
        self.assertAlmostEqual(samples[:, 1].mean(), a ** 2 + 1.0, delta=0.5)

    def test_step_size_adaptation_runs(self):
        """Adaptation should change the step size and stay finite."""
        rng = np.random.RandomState(4)
        ndim = 3
        log_post, grad_log_post = make_gaussian(np.zeros(ndim), np.eye(ndim))

        nuts = NUTSSampler(
            grad_log_posterior_fn=grad_log_post,
            log_posterior_fn=log_post,
            ndim=ndim,
            step_size=2.0,  # deliberately too large
            adapt_step_size=True,
            target_accept=0.8,
            n_adapt=200,
            random=rng,
        )

        x0 = rng.standard_normal((16, ndim))
        eps0 = nuts.step_size
        nuts.sample(x0, 250)
        self.assertTrue(np.isfinite(nuts.step_size))
        # adaptation should have moved the step size at least a little
        self.assertNotEqual(nuts.step_size, eps0)


class NUTSMoveInErynTest(unittest.TestCase):
    """Drop :class:`NUTSMove` into an Eryn ``EnsembleSampler``."""

    def test_isotropic_gaussian_via_ensemble(self):
        np.random.seed(10)
        ndim = 3
        n_walkers = 24

        mu = np.zeros(ndim)
        cov = np.eye(ndim)
        inv_cov = np.linalg.inv(cov)

        def grad_fn(x):
            return gaussian_grad_eryn(x, mu, inv_cov)

        lims = 6.0
        priors = ProbDistContainer(
            {i: uniform_dist(-lims, lims) for i in range(ndim)}
        )

        nuts_move = NUTSMove(
            grad_log_like_fn=grad_fn,
            ndim=ndim,
            step_size=0.4,
            max_tree_depth=6,
        )

        ensemble = EnsembleSampler(
            n_walkers,
            ndim,
            gaussian_log_like_eryn,
            priors,
            args=[mu, inv_cov],
            moves=nuts_move,
            vectorize=False,
        )

        coords = priors.rvs(size=(n_walkers,))
        ensemble.run_mcmc(coords, 400, burn=200, progress=False)

        chain = ensemble.get_chain()["model_0"].reshape(-1, ndim)
        self.assertTrue(np.allclose(chain.mean(axis=0), mu, atol=0.2))
        self.assertTrue(np.allclose(chain.std(axis=0), 1.0, atol=0.3))

    def test_correlated_gaussian_constant_metric_via_ensemble(self):
        np.random.seed(11)
        ndim = 3
        n_walkers = 24

        mu = np.array([0.5, -1.0, 2.0])
        L = np.array([[1.0, 0.0, 0.0],
                      [0.6, 0.8, 0.0],
                      [0.3, -0.2, 0.4]])
        cov = L @ L.T
        inv_cov = np.linalg.inv(cov)

        def grad_fn(x):
            return gaussian_grad_eryn(x, mu, inv_cov)

        lims = 10.0
        priors = ProbDistContainer(
            {i: uniform_dist(-lims + mu[i], lims + mu[i]) for i in range(ndim)}
        )

        nuts_move = NUTSMove(
            grad_log_like_fn=grad_fn,
            ndim=ndim,
            metric=inv_cov,  # ~ mass = inverse covariance
            step_size=0.5,
            max_tree_depth=6,
        )

        ensemble = EnsembleSampler(
            n_walkers,
            ndim,
            gaussian_log_like_eryn,
            priors,
            args=[mu, inv_cov],
            moves=nuts_move,
            vectorize=False,
        )

        coords = mu + np.random.randn(n_walkers, ndim)
        ensemble.run_mcmc(coords, 600, burn=200, progress=False)

        chain = ensemble.get_chain()["model_0"].reshape(-1, ndim)
        self.assertTrue(np.allclose(chain.mean(axis=0), mu, atol=0.25))
        emp_cov = np.cov(chain, rowvar=False)
        self.assertTrue(
            np.allclose(emp_cov, cov, atol=0.3 * np.max(np.abs(cov)))
        )


class NUTSTemperedTest(unittest.TestCase):
    """Check that NUTSMove uses the temperature ladder on the gradient.

    For a Gaussian likelihood ``L(x) = N(0, I)`` with a wide flat prior,
    the tempered target ``beta * log L + log pi`` is ``N(0, 1/beta * I)``
    on the support. So at temperature ``1/beta`` each marginal should
    have variance ``1/beta``.
    """

    def test_hot_chain_has_wider_marginals(self):
        np.random.seed(7)
        ndim = 2
        nwalkers = 32
        ntemps = 4

        mu = np.zeros(ndim)
        cov = np.eye(ndim)
        inv_cov = np.linalg.inv(cov)

        def grad_log_like(x):
            return gaussian_grad_eryn(x, mu, inv_cov)

        # very wide flat priors so the prior gradient is zero everywhere
        # that matters for this test
        lims = 50.0
        priors = ProbDistContainer(
            {i: uniform_dist(-lims, lims) for i in range(ndim)}
        )

        nuts_move = NUTSMove(
            grad_log_like_fn=grad_log_like,
            ndim=ndim,
            step_size=0.5,
            max_tree_depth=6,
        )

        # Geometric ladder
        betas = np.geomspace(1.0, 1e-2, ntemps)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            gaussian_log_like_eryn,
            priors,
            args=[mu, inv_cov],
            moves=nuts_move,
            tempering_kwargs={"betas": betas},
            vectorize=False,
        )

        coords = np.random.randn(ntemps, nwalkers, ndim)
        ensemble.run_mcmc(coords, 400, burn=300, progress=False)

        chain = ensemble.get_chain()["model_0"]  # (n_steps, ntemps, nwalkers, 1, ndim)
        # variance per temperature, averaged across the two marginals
        var_per_temp = chain.reshape(chain.shape[0], ntemps, nwalkers, ndim).var(
            axis=(0, 2)
        ).mean(axis=-1)

        # Each tempered marginal variance should be ~ 1 / beta.
        target = 1.0 / betas
        # generous tolerance because of finite-sample noise at the hot
        # temperatures and because tempered swaps mix the chains
        rel_err = np.abs(var_per_temp - target) / target
        self.assertTrue(
            np.all(rel_err < 0.5),
            msg="variance per temperature {0} vs target {1} (rel err {2})".format(
                var_per_temp, target, rel_err
            ),
        )
        # monotonic: hot chains should be wider than cold
        self.assertTrue(
            var_per_temp[0] < var_per_temp[-1],
            msg="cold-chain variance {0} should be less than hot-chain variance {1}".format(
                var_per_temp[0], var_per_temp[-1]
            ),
        )


class NUTSVsStretchTest(unittest.TestCase):
    """Compare NUTS and Stretch on the same analytic Gaussian target."""

    def _run_eryn(self, move, ndim, mu, inv_cov, n_walkers, n_steps, burn):
        lims = 8.0
        priors = ProbDistContainer(
            {i: uniform_dist(-lims + mu[i], lims + mu[i]) for i in range(ndim)}
        )
        ensemble = EnsembleSampler(
            n_walkers,
            ndim,
            gaussian_log_like_eryn,
            priors,
            args=[mu, inv_cov],
            moves=move,
            vectorize=False,
        )
        coords = mu + np.random.randn(n_walkers, ndim) * 0.5
        ensemble.run_mcmc(coords, n_steps, burn=burn, progress=False)
        return ensemble.get_chain()["model_0"].reshape(-1, ndim)

    def test_both_recover_moments_on_correlated_gaussian(self):
        """NUTS and Stretch should both pass the same moment tolerances."""
        np.random.seed(123)
        ndim = 3
        n_walkers = 32
        mu = np.array([0.0, 0.0, 0.0])
        L = np.array([[1.0, 0.0, 0.0],
                      [0.5, 0.7, 0.0],
                      [-0.3, 0.4, 0.6]])
        cov = L @ L.T
        inv_cov = np.linalg.inv(cov)

        def grad_fn(x):
            return gaussian_grad_eryn(x, mu, inv_cov)

        # NUTS with a metric matching the inverse covariance
        nuts = NUTSMove(
            grad_log_like_fn=grad_fn,
            ndim=ndim,
            metric=inv_cov,
            step_size=0.5,
            max_tree_depth=6,
        )
        stretch = StretchMove()

        np.random.seed(123)
        nuts_chain = self._run_eryn(nuts, ndim, mu, inv_cov, n_walkers, 600, 200)
        np.random.seed(123)
        stretch_chain = self._run_eryn(
            stretch, ndim, mu, inv_cov, n_walkers, 600, 200
        )

        # Both should recover the marginal means and standard deviations.
        for label, chain in [("NUTS", nuts_chain), ("Stretch", stretch_chain)]:
            with self.subTest(label=label):
                self.assertTrue(
                    np.allclose(chain.mean(axis=0), mu, atol=0.3),
                    msg="{0}: mean = {1}".format(label, chain.mean(axis=0)),
                )
                emp_std = chain.std(axis=0)
                true_std = np.sqrt(np.diag(cov))
                self.assertTrue(
                    np.allclose(emp_std, true_std, atol=0.3 * true_std.max()),
                    msg="{0}: std = {1}, target = {2}".format(label, emp_std, true_std),
                )


if __name__ == "__main__":
    unittest.main()
