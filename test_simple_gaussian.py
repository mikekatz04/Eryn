from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
from eryn.stopping import AutoCorrelationStop
import numpy as np


def log_prob_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum(axis=1)


def log_prob_fn_wrap(x, *args):
    shape = x.shape[:-1]
    ndim = x.shape[-1]
    x_temp = x.reshape(-1, ndim).copy()
    temp = log_prob_fn(x_temp, *args)
    out = temp.reshape(shape)
    return out


def test_no_temps():
    ndim = 5
    nwalkers = 100

    np.random.seed(42)
    means = np.zeros(ndim)  # np.random.rand(ndim)

    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    cov *= 10000.0
    invcov = np.linalg.inv(cov)

    # cov = np.diag(np.ones(5)) * 10.0

    p0 = np.random.randn(nwalkers, ndim)

    lims = 2.0
    priors = {i: uniform_dist(-lims, lims) for i in range(ndim)}

    coords = np.zeros((nwalkers, ndim))

    for ind, dist in priors.items():
        coords[:, ind] = dist.rvs(size=(nwalkers,))

    log_prob = log_prob_fn(coords, means, cov)
    check = log_prob_fn(means[None, :], means, cov)

    blobs = None  # np.random.randn(ntemps, nwalkers, 3)

    state = State(coords, log_prob=log_prob, blobs=blobs)

    stopping_fn = AutoCorrelationStop(verbose=True)

    ensemble = EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn,
        priors,
        args=[means, cov],
        stopping_iterations=1000,
        stopping_fn=stopping_fn,
    )

    nsteps = 50000
    ensemble.run_mcmc(state, nsteps, burn=1000, progress=True, thin_by=1)

    check = ensemble.get_chain()["model_0"].reshape(-1, ndim)
    return check


def test_with_temps():
    ndim = 5
    ntemps = 30
    nwalkers = 100

    np.random.seed(42)
    means = np.zeros(ndim)  # np.random.rand(ndim)

    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    cov *= 10000.0
    invcov = np.linalg.inv(cov)

    # cov = np.diag(np.ones(5)) * 10.0

    p0 = np.random.randn(ntemps, nwalkers, ndim)

    lims = 2.0
    priors = {i: uniform_dist(-lims, lims) for i in range(ndim)}

    coords = np.zeros((ntemps, nwalkers, ndim))

    for ind, dist in priors.items():
        coords[:, :, ind] = dist.rvs(size=(ntemps, nwalkers,))

    log_prob = log_prob_fn_wrap(coords, means, cov)
    check = log_prob_fn_wrap(means[None, None, :], means, cov)

    blobs = None  # np.random.randn(ntemps, nwalkers, 3)

    state = State(coords, log_prob=log_prob, blobs=blobs)

    burn = 1000
    stopping_fn = AutoCorrelationStop(verbose=True)

    ensemble = EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn_wrap,
        priors,
        args=[means, cov],
        tempering_kwargs={"Tmax": np.inf, "ntemps": ntemps, "stop_adaptation": burn},
        plot_iterations=1000,
        stopping_fn=stopping_fn,
        stopping_iterations=1000,
    )

    nsteps = 50000
    ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=1)

    check = ensemble.get_chain()["model_0"][:, 0, :].reshape(-1, ndim)

    # check_ac1 = ensemble.backend.get_autocorr_time(average=True, all_temps=True)
    # check_ac = ensemble.backend.get_autocorr_time()
    # evidence = ensemble.backend.get_evidence_estimate(return_error=True)
    return check


if __name__ == "__main__":
    check_temps = test_with_temps()
    check_no_temps = test_no_temps()

    import corner

    ndim = 5
    fig = corner.corner(
        check_temps,
        range=[0.9999 for _ in range(ndim)],
        levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)),
        bins=30,
        plot_density=False,
    )
    corner.corner(
        check_no_temps,
        range=[0.9999 for _ in range(ndim)],
        levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)),
        bins=30,
        plot_density=False,
        fig=fig,
    )
    import matplotlib.pyplot as plt

    plt.show()

    plt.close()
    breakpoint()
