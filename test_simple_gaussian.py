from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
import numpy as np


def log_prob_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum(axis=1)


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

state2 = State(state)

backend = Backend()

backend.reset(
    nwalkers, ndim,
)

# backend.grow(100, blobs)

ensemble = EnsembleSampler(nwalkers, ndim, log_prob_fn, priors, args=[means, cov],)

nsteps = 50000
ensemble.run_mcmc(state, nsteps, burn=1000, progress=True, thin=5)

testing = ensemble.get_nleaves()

import matplotlib.pyplot as plt

check = ensemble.get_chain()["model_0"].reshape(-1, ndim)[0::]
import corner

fig = corner.corner(
    check[0::20],
    range=[0.999 for _ in range(ndim)],
    levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)),
    bins=30,
)
plt.show()

plt.close()
breakpoint()
