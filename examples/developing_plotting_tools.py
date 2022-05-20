## Load standard modules
import numpy as np
import sys

sys.path.append('../')
from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
from eryn.moves import GaussianMove
from eryn.utils import PlotContainer

# seed our random number generator, so we have reproducible data
np.random.seed(sum([ord(v) for v in 'gaussians']))

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
    ll = - 0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)  #  / len(t)
    return ll

branch_names = ["gauss"]

num = 300
t = np.linspace(-1, 1, num)

gauss_inj_params = [
    [1.0, -0.3, 0.1],
    [1.0, -0.0, 0.1],
    [1.0, 0.3, 0.2],
]

injection = np.zeros(num)

for pars in gauss_inj_params:
    injection += gaussian_flat(t, *pars)

sigma = 0.3
y = injection + sigma * np.random.randn(len(injection))

nwalkers    = 20
ntemps      = 10
ndims       = [3]
nleaves_max = [10]

priors = {
    "gauss": {
        0: uniform_dist(0.1, 5.),
        1: uniform_dist(t.min(), t.max()),
        2: uniform_dist(0.01, 0.3),
    },
}

coords = {
    name: np.zeros((ntemps, nwalkers, nleaf, ndim))
    for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names)
}

sig1 = 0.0000001
for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names):
    for nn in range(nleaf):
        if nn >= len(gauss_inj_params):
            nn = np.random.randint(low=0, high=3)
        coords[name][:, :, nn] = np.random.multivariate_normal(gauss_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers))  


inds = {
     name: np.random.randint(low=0, high=1, size=(ntemps, nwalkers, nleaf), dtype=bool)
     for nleaf, name in zip(nleaves_max, branch_names)
}

inds['gauss'][:, :, :3] = True

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
betas    = np.linspace(1.0, 0.0, ntemps)

blobs = None  # np.random.randn(ntemps, nwalkers, 3)

state   = State(coords, log_prob=log_prob, betas=betas, blobs=blobs, inds=inds)
state2  = State(state)
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

factor = 0.0001
cov    = {"gauss": np.diag(np.ones(3)) * factor}
moves  = GaussianMove(cov)

nsteps = 400
burnin = 100

ensemble = EnsembleSampler(
    nwalkers,
    ndims,  # assumes ndim_max
    log_prob_fn,
    priors,
    args=[t, y],
    tempering_kwargs=dict(betas=betas, stop_adaptation=-1), # dict(ntemps=ntemps),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    provide_groups=True,
    plot_iterations=-1,
    moves=moves,
    rj_moves=True,
)

ensemble.run_mcmc(state, nsteps, burn=burnin, progress=True, thin_by=1)

mcmcchains = ensemble.get_chain()

print(mcmcchains.keys(), mcmcchains['gauss'])

# (400, 10, 20, 10, 3)
# (nsamples, ntemps, nwalkers, modelmax, ndim)

# breakpoint()

plot = PlotContainer(backend=ensemble.backend)

# Define some parameter names for testing
paramnames = [r'$A$',r'$\mu$',r'$\sigma$']

plot.generate_corner(burn=burnin, labels=paramnames)

plot.generate_parameter_chains_per_temperature(burn=burnin, labels=paramnames)

plot.generate_parameter_chains_per_temperature_per_walker(burn=burnin, labels=paramnames)

plot.generate_parameter_chains(burn=burnin, labels=paramnames)

plot.generate_posterior_chains(burn=burnin)

plot.generate_temperature_chains(onefig=True)

plot.generate_leaves_chains(burn=burnin, labels=paramnames)

plot.generate_k_per_temperature_chains(burn=burnin)

plot.generate_k_per_tree_chains(burn=burnin)