import numpy as np

from eryn.paraensemble import ParaEnsembleSampler

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import ParaState

# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()


if __name__ == "__main__":
    ngroups = 15
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

    nsteps = 50
    # burn for 1000 steps
    burn = 10
    # thin by 5
    thin_by = 1

    state = ParaState({"gauss": coords}, groups_running=groups_running)
    
    ensemble_pt.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)

    for temp in range(ntemps):
        samples = ensemble_pt.get_chain()["model_0"][:, temp].reshape(-1, ndim)

    ll = ensemble_pt.backend.get_log_like()
