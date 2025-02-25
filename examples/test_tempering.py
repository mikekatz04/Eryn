import os

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds

import matplotlib.pyplot as plt
import numpy as np

# set random seed
np.random.seed(42)

import corner

# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()


if __name__ == '__main__':
    ndim = 5

    # mean
    means = np.zeros(ndim)  # np.random.rand(ndim)

    # define covariance matrix
    cov = np.diag(np.ones(ndim))
    invcov = np.linalg.inv(cov)

    # set prior limits
    lims = 5.0
    priors_in = {i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)}
    priors = ProbDistContainer(priors_in)

    # setup sampler
    nwalkers = 100
    ntemps = 10
    Tmax = np.inf

    #fill kwargs dictionary
    betas = np.array([0.0])
    tempering_kwargs=dict(
                            #betas=betas,
                            Tmax=Tmax,
                            ntemps=ntemps,
                        )

    # randomize throughout prior
    coords = priors.rvs(size=(ntemps, nwalkers,))

    cov_all = {'model_0': np.eye(ndim) * 1e4}

    proposal = GaussianMove(cov_all)

    # initialize sampler
    ensemble_pt = EnsembleSampler(
        nwalkers,
        ndim,
        log_like_fn,
        priors,
        args=[means, cov],
        moves=proposal,
        tempering_kwargs=tempering_kwargs
    )

    nsteps = 3000
    # burn for 1000 steps
    burn = 1000
    # thin by 5
    thin_by = 5
    ensemble_pt.run_mcmc(coords, nsteps, burn=burn, progress=True, thin_by=thin_by)

    #! diagnostic stuff

    discard, thin = 0, 1
    os.makedirs('tmp', exist_ok=True)

    for temp in range(ntemps):
        samples = ensemble_pt.get_chain(discard=discard, thin=thin)['model_0'][:, temp].reshape(-1, ndim)

        fig = corner.corner(samples, truths=np.full(ndim, 0.0), weights=np.ones(samples.shape[0])/samples.shape[0])
        corner.corner(coords[temp], color='r', weights=np.ones(coords[temp].shape[0])/coords[temp].shape[0], fig=fig)
        plt.legend(['Posterior', 'Truth', 'Starting points'], bbox_to_anchor=(1.02, 1.6), fontsize=13)

        plt.savefig('tmp/corner_temp_{}.png'.format(temp))
        plt.close(fig)

    # plot traces
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        for temp in range(ntemps):
            axes[i].plot(ensemble_pt.get_chain(discard=discard, thin=thin)['model_0'][:, temp, :, :, i].reshape(-1))
        axes[i].set_ylabel('x{}'.format(i))
    plt.savefig('tmp/traces.png')
    plt.close(fig)

    breakpoint()
    