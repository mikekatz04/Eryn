from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer, PlotContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds

import matplotlib.pyplot as plt
import numpy as np

import sys

# set random seed
np.random.seed(42)

import corner

# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()

def main():
    ndim = 5
    nwalkers = 50
    ntemps = 10

    # mean
    means = np.zeros(ndim)  # np.random.rand(ndim)

    # define covariance matrix
    cov = np.diag(np.ones(ndim))
    invcov = np.linalg.inv(cov)

    # set prior limits
    lims = 50.0
    priors_in = {i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)}
    priors = ProbDistContainer(priors_in)

    tempering_kwargs=dict(ntemps=ntemps)

    # randomize throughout prior
    coords = priors.rvs(size=(ntemps, nwalkers,))

    plotter = PlotContainer(
        plots='all',
        truths=dict(model_0=np.full(ndim, 0.0)),
        overlay_covariance=dict(model_0=cov),
        parent_folder='diagnostic',
        tempering_palette="icefire",
        discard=0.1
    )

    # initialize sampler
    ensemble_pt = EnsembleSampler(
        nwalkers,
        ndim,
        log_like_fn,
        priors,
        args=[means, cov],
        tempering_kwargs=tempering_kwargs,
        plot_generator=plotter,
        plot_iterations=100
    )

    nsteps = 1000
    burn = 0
    # thin by 5
    thin_by = 1
    ensemble_pt.run_mcmc(coords, nsteps, burn=0, progress=True, thin_by=thin_by)

if __name__ == "__main__":
    main()