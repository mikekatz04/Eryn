from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
import numpy as np


def gaussian(x, a, b, c):
    f_x = a[:, None] * np.exp(-((x[None, :] - b[:, None]) ** 2) / (2 * c[:, None] ** 2))
    return f_x


def sine(x, A, f):
    return A[:, None] * np.sin(2 * np.pi * f[:, None] * t[None, :])


def gaussian_flat(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x


def sine_flat(x, A, f):
    return A * np.sin(2 * np.pi * f * t)


def log_prob_fn(
    x1, x2, group1, group2, data, inds=None, fill_inds=[], fill_values=None
):

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

    # sine
    if len(fill_inds) > 0:
        if fill_values is None:
            raise ValueError
        ndim = x2.shape[-1]
        ndim_total = ndim + len(fill_inds[1])
        base = x2.shape[:-1]
        x2_in = np.zeros(base + (ndim_total,))
        test_inds = np.delete(np.arange(ndim_total), fill_inds[1])
        x2_in[:, :, :, test_inds] = x2
        x2_in[:, :, :, fill_inds[1]] = fill_values[1]

    else:
        x2_in = x2

    a = x1_in[:, 0]
    b = x1_in[:, 1]
    c = x1_in[:, 2]

    gauss_out = gaussian(t, a, b, c)

    A = x2_in[:, 0]
    f = x2_in[:, 1]

    sine_out = sine(t, A, f)

    num_groups = group1.max() + 1

    template = np.zeros((num_groups, len(t)))
    for i in range(num_groups):
        inds1 = np.where(group1 == i)
        inds2 = np.where(group2 == i)

        template[i] += gauss_out[inds1].sum(axis=0) + sine_out[inds2].sum(axis=0)

    ll = -np.sum((template - data) ** 2 / data ** 2, axis=-1)  # /np.sqrt(len(t))
    blobs = np.random.randn(*(ll.shape + (5,)))
    out = np.concatenate([np.expand_dims(ll, axis=-1), blobs], axis=-1)
    return out


nwalkers = 50
ntemps = 4
nbranches = 2
ndims = [3, 2]
nleaves_max = [5, 4]

branch_names = ["gauss", "sine"]

num = 100
t = np.linspace(-5, 5, num)

gauss_inj_params = [[10.0, 1.0, 0.25], [10.0, -1.0, 0.25]]
injection = np.zeros(num)

sine_inj_params = [[1.0, 35.2342342], [1.0, 487.21391123]]

for pars in gauss_inj_params:
    injection += gaussian_flat(t, *pars)

for pars in sine_inj_params:
    injection += sine_flat(t, *pars)

# import matplotlib.pyplot as plt
# plt.plot(t, injection)
# plt.show()
# breakpoint()

priors = {
    "gauss": {
        0: uniform_dist(9.0, 11.0),
        1: uniform_dist(t.min(), t.max()),
        2: uniform_dist(0.1, 1.0),
    },
    "sine": {0: uniform_dist(0.1, 2.0), 1: uniform_dist(10.0, 1000.0)},
}

coords = {
    name: np.zeros((ntemps, nwalkers, nleaf, ndim))
    for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names)
}

for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names):
    temp = priors[name]
    for ind, dist in temp.items():
        coords[name][:, :, :, ind] = dist.rvs(size=(ntemps, nwalkers, nleaf))

inds = {
    name: np.random.randint(0, high=2, size=(ntemps, nwalkers, nleaf), dtype=bool)
    for nleaf, name in zip(nleaves_max, branch_names)
}

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

temp = log_prob_fn(
    coords_in["gauss"],
    coords_in["sine"],
    groups_in["gauss"],
    groups_in["sine"],
    t,
    injection,
    fill_inds=[],
    fill_values=None,
)

log_prob = temp[:, 0]
blobs = temp[:, 1:]
log_prob = log_prob.reshape(ntemps, nwalkers)
blobs = blobs.reshape(ntemps, nwalkers, -1)
betas = np.linspace(1.0, 0.0, ntemps)

state = State(coords, log_prob=log_prob, betas=betas, blobs=blobs, inds=inds)

state2 = State(state)

from eryn.backends import HDFBackend

backend = HDFBackend("testing_full.h5")

factor = 0.01
cov = {"gauss": np.diag(np.ones(3)) * factor, "sine": np.diag(np.ones(2)) * factor}

# backend.grow(100, blobs)

ensemble = EnsembleSampler(
    nwalkers,
    ndims,  # assumes ndim_max
    log_prob_fn,
    priors,
    args=[t, injection],
    tempering_kwargs=dict(betas=betas),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    provide_groups=True,
    cov=cov,
    plot_iterations=-1,
    rj=True,
    backend=backend,
)

nsteps = 10000
ensemble.run_mcmc(state, nsteps, burn=1000, progress=True, thin_by=5)

check = ensemble.backend.get_autocorr_time(average=True, all_temps=True)

testing = ensemble.get_nleaves()

import matplotlib.pyplot as plt

check = ensemble.get_chain()["sine"][:, 0, :, :, 1].flatten()
check = check[check != 0.0]
plt.hist(check, bins=30)
plt.show()

plt.close()

plt.hist(testing["sine"][:, 0].flatten(), bins=30)
plt.hist(testing["gauss"][:, 0].flatten(), bins=30)
plt.show()
plt.close()
breakpoint()
