import matplotlib.pyplot as plt

import numpy as np

from eryn.paraensemble import ParaEnsembleSampler

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import ParaState

import corner


# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    # return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()
    out = -0.5 * np.einsum("...i,...ij,...j->...", diff, invcov, diff)
    return out


if __name__ == "__main__":
    ngroups = 105
    ndim = 5
    ntemps = 10
    nwalkers = 32

    groups_running = np.ones(ngroups, dtype=bool)

    # set up problem

    means = np.zeros(ndim)  # np.random.rand(ndim)

    # define covariance matrix
    cov = np.diag(np.ones(ndim))
    invcov = np.linalg.inv(cov)

    lims = 5.0
    priors_in = {
        i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
    }
    priors = {"gauss": ProbDistContainer(priors_in)}

    # fill kwargs dictionary
    tempering_kwargs = dict(ntemps=ntemps)

    # randomize throughout prior
    coords = priors["gauss"].rvs(
        size=(
            ngroups,
            ntemps,
            nwalkers,
        )
    )

    # initialize sampler
    ensemble_pt = ParaEnsembleSampler(
        ndim,
        nwalkers,
        ngroups,
        log_like_fn,
        priors,
        tempering_kwargs=tempering_kwargs,
        args=[means, cov],
        kwargs={},
        gpu=None,
        periodic=None,
        backend=None,
        update_fn=None,
        update_iterations=-1,
        stopping_fn=None,
        stopping_iterations=-1,
        name="gauss",
    )

    nsteps = 500
    # burn for 1000 steps
    burn = 1000
    # thin by 5
    thin_by = 25

    state = ParaState({"gauss": coords}, groups_running=groups_running)

    ensemble_pt.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)

    # ll = ensemble_pt.backend.get_log_like()

    samples = (
        ensemble_pt.get_chain()[:, :, 0]
        .transpose(1, 0, 2, 3)
        .reshape(ngroups, -1, ndim)
    )

    fig = None
    for i in range(samples.shape[0])[:10]:
        fig = corner.corner(
            samples[i],
            plot_datapoints=False,
            plot_density=False,
            color=f"C{i}",
            levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2),
            fig=fig,
        )
    breakpoint()
