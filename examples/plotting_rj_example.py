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

def gaussian_pulse(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x

def combine_gaussians(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template

def log_like_fn_gauss_pulse(params, t, data, sigma):
    
    template = combine_gaussians(t, params)
    
    ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll


def main():
    nwalkers = 30
    ntemps = 20
    ndim = 3
    nleaves_max = 8
    nleaves_min = 0

    branch_names = ["gauss"]

    # define time stream
    num = 500
    t = np.linspace(-1, 1, num)

    gauss_inj_params = np.asarray([
        [3.3, -0.2, 0.1],
        [2.6, -0.1, 0.1],
        [3.4, 0.0, 0.1],
        [2.9, 0.3, 0.1],
    ])

    # combine gaussians
    injection = combine_gaussians(t, gauss_inj_params)

    # set noise level
    sigma = 2.0

    # produce full data
    y = injection + sigma * np.random.randn(len(injection))

    priors = {
    "gauss": {
        "amp": uniform_dist(2.5, 3.5),          # amplitude
        "mean": uniform_dist(t.min(), t.max()),  # mean 
        "sigma": uniform_dist(0.01, 0.21),        # sigma
    },
    }

    # for the Gaussian Move, will be explained later
    factor = 0.00001
    cov = {"gauss": np.diag(np.ones(ndim)) * factor}

    moves = GaussianMove(cov)

    plotter = PlotContainer(
        plots='all',
        #labels=None,
        truths=dict(gauss=gauss_inj_params),
        overlay_covariance=None,
        parent_folder='diagnostic_rj',
        tempering_palette="icefire",
        discard=0.1
    )

    # initialize sampler
    ensemble = EnsembleSampler(
        nwalkers,
        ndim,
        log_like_fn_gauss_pulse,
        priors,
        args=[t, y, sigma],
        tempering_kwargs=dict(ntemps=ntemps),
        nbranches=len(branch_names),
        branch_names=branch_names,
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        moves=moves,
        rj_moves=True,  # basic generation of new leaves from the prior
        plot_generator=plotter,
        plot_iterations=100
    )

    coords = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max, ndim))}

    # this is the sigma for the multivariate Gaussian that sets starting points
    # We need it to be very small to assume we are passed the search phase
    # we will verify this is with likelihood calculations
    sig1 = 0.0001

    # setup initial walkers to be the correct count (it will spread out)
    for nn in range(nleaves_max):
        if nn >= len(gauss_inj_params):
            # not going to add parameters for these unused leaves
            continue
            
        coords["gauss"][:, :, nn] = np.random.multivariate_normal(gauss_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)) 

    # make sure to start near the proper setup
    inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

    # turn False -> True for any binary in the sampler
    inds['gauss'][:, :, :len(gauss_inj_params)] = True

    log_prior = ensemble.compute_log_prior(coords, inds=inds)
    log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

    # make sure it is reasonably close to the maximum which this is
    # will not be zero due to noise
    print("Log-likelihood:\n", log_like)
    print("\nLog-prior:\n", log_prior)

    # setup starting state
    state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

    nsteps = 2000
    burn = 0
    # thin by 5
    thin_by = 1
    ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)

if __name__ == "__main__":
    main()