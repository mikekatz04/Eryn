from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
import numpy as np


def gaussian(x, a, b, c):
    f_x = a[:, None] * np.exp(-((x[None, :] - b[:, None]) ** 2) / (2 * c[:, None] ** 2))
    return f_x


def gaussian_flat(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x


def log_prob_fn(x1, group1, t, data, inds=None, fill_inds=[], fill_values=None):

    # gauss
    if len(fill_inds) > 0:
        raise NotImplementedError
        if fill_values is None:
            raise ValueError
        ndim = x1.shape[-1]
        ndim_total = ndim + len(fill_inds[0])
        base = x1.shape[:-1]
        x1_in = np.zeros(base + (ndim_total,))
        test_inds = np.delete(np.arange(ndim_total), fill_inds[0])
        x1_in[:, :, :, test_inds] = x1
        x1_in[:, :, :, fill_inds[0]] = fill_values[0]

    else:
        x1_in = x1

    a = x1_in[:, 0]
    b = x1_in[:, 1]
    c = x1_in[:, 2]

    gauss_out = gaussian(t, a, b, c)

    num_groups = group1.max() + 1

    template = np.zeros((num_groups, len(t)))
    for i in range(num_groups):
        inds1 = np.where(group1 == i)

        template[i] += gauss_out[inds1].sum(axis=0)
    # breakpoint()
    ll = - 0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1) / len(t)
    return ll


nwalkers = 50
ntemps = 10
nbranches = 2
ndims = [3]
nleaves_max = [12]

branch_names = ["gauss"]

num = 500
t = np.linspace(-10, 10, num)

gauss_inj_params = [
    [3.0, 2.0, 0.25],
    [4.0, -2.0, 0.25],
    [2.0, 0.0, 0.4],
    [3.5, -5.0, 0.25],
    [5.0, 5.0, 0.25],
    [3.0, 8.0, 0.25],
    [3.0, -8.0, 0.25],
]

injection = np.zeros(num)

for pars in gauss_inj_params:
    injection += gaussian_flat(t, *pars)

sigma = 0.01
y = injection + sigma * np.random.randn(len(injection))

import matplotlib.pyplot as plt

#plt.plot(t, y, label="data", color="lightskyblue")
#plt.plot(t, injection, label="injection", color="crimson")
#plt.show()
#plt.close()

priors = {
    "gauss": {
        0: uniform_dist(1.0, 6.0),
        1: uniform_dist(t.min(), t.max()),
        2: uniform_dist(0.1, 1.0),
    },
}

coords = {
    name: np.zeros((ntemps, nwalkers, nleaf, ndim))
    for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names)
}

for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names):
    temp = priors[name]
    for ind, dist in temp.items():
        coords[name][:, :, :, ind] = dist.rvs(size=(ntemps, nwalkers, nleaf))

# Do the actual true vals in
# coords['gauss'][0,0] = np.asarray(gauss_inj_params)


# inds = None
inds = {
     name: np.random.randint(0, high=2, size=(ntemps, nwalkers, nleaf), dtype=bool)
     for nleaf, name in zip(nleaves_max, branch_names)
}
#inds = {
#    name: np.full((ntemps, nwalkers, nleaf), True, dtype=bool)
#    for nleaf, name in zip(nleaves_max, branch_names)
#}

for name, inds_temp in inds.items():
    inds_fix = np.where(np.sum(inds_temp, axis=-1) == 0)

    for ind1, ind2 in zip(inds_fix[0], inds_fix[1]):
        inds_temp[ind1, ind2, 0] = True

groups = {
    name: np.arange(coords[name].shape[0] * coords[name].shape[1]).reshape(
        coords[name].shape[:2]
    )[:, :, None]
    for name in coords
}

groups = {
    name: np.repeat(groups[name], coords[name].shape[2], axis=-1) for name in groups
}

coords_in = {name: coords[name][inds[name]] for name in coords}
groups_in = {name: groups[name][inds[name]] for name in groups}

log_prob = log_prob_fn(
    coords_in["gauss"],
    groups_in["gauss"],
    t,
    y,
    fill_inds=[],
    fill_values=None,
)

log_prob = log_prob.reshape(ntemps, nwalkers)
betas = np.linspace(1.0, 0.0, ntemps)

blobs = None  # np.random.randn(ntemps, nwalkers, 3)

state = State(coords, log_prob=log_prob, betas=betas, blobs=blobs, inds=inds)

state2 = State(state)

backend = Backend()

backend.reset(
    nwalkers,
    ndims,
    nleaves_max=nleaves_max,
    ntemps=ntemps,
    truth=None,
    branch_names=branch_names,
    rj=True,
)

factor = 0.001
cov = {"gauss": np.diag(np.ones(3)) * factor}

# backend.grow(100, blobs)

ensemble = EnsembleSampler(
    nwalkers,
    ndims,  # assumes ndim_max
    log_prob_fn,
    priors,
    args=[t, y],
    tempering_kwargs=dict(betas=betas),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    provide_groups=True,
    cov=cov,
    plot_iterations=-1,
    rj_moves=True,
)

nsteps = 2000
ensemble.run_mcmc(state, nsteps, burn=2500, progress=True, thin_by=5)

check = ensemble.backend.get_autocorr_time(average=True, all_temps=True)
# breakpoint()
testing = ensemble.get_nleaves()

import matplotlib.pyplot as plt

check = ensemble.get_chain()["gauss"][:, 0, :, :, 1].flatten()
check = check[check != 0.0]
plt.hist(check, bins=30)
plt.show()

plt.close()

bns = (
    np.arange(1, nleaves_max[0] + 2) - 0.5
)  # Justto make it pretty and center the bins
plt.hist(testing["gauss"][:, 0].flatten() + 1, bins=bns)
plt.xticks(np.arange(1, nleaves_max[0] + 1))
plt.xlabel("# of peaks in the data")
plt.show()
plt.close()
breakpoint()
