## Load standard modules
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex']=True
import matplotlib.pyplot as plt
import sys
from spectral import window

sys.path.append('../')
from eryn.state import State
from eryn.backends.backend import Backend
from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist
from eryn.moves import GaussianMove
from eryn.utils import PlotContainer
from eryn.backends import HDFBackend

import numpy as np

# seed our random number generator, so we have reproducible data
np.random.seed(sum([ord(v) for v in 'srgerag']))

DOPLOT=False

def gaussian(x, a, b, c):
    f_x = a[:, None] * np.exp(-((x[None, :] - b[:, None]) ** 2) / (2 * c[:, None] ** 2))
    return f_x


def gaussian_flat(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x

def log_prob_fn(x, groups, t, data, inds=None, fill_inds=[], fill_values=None):

    x1, x2 = x
    group1, group2 = groups
    a = x1[:, 0]
    b = x1[:, 1]
    c = x1[:, 2]
    n = len(t)

    gauss_out  = gaussian(t, a, b, c)
    num_groups = int(group1.max() + 1)
    template   = np.zeros((num_groups, len(t)))
    for i in range(num_groups):
        inds1 = np.where(group1 == i)
        template[i] += gauss_out[inds1].sum(axis=0)

    sig = np.atleast_2d(x2)[:,0]
    llh = - 0.5 * ( np.sum(((template - data)) ** 2, axis=-1) )
    llh *= 1/sig**2
    llh += - n*np.log(sig) - .5 * n * np.log(2.*np.pi)
    return llh




num = 500
t = np.linspace(-5, 5, num)

gauss_inj_params = [
    [1.0, -4, 0.3],
    [1.0, -2, 0.3],
    [1.0, -0.0, 0.3],
    [1.0, 2, 0.3],
    [1.0, 4, 0.3],
]

injection = np.zeros(num)

for pars in gauss_inj_params:
    injection += gaussian_flat(t, *pars)

sigma = [[0.3]]
y = injection + sigma[0][0] * np.random.randn(len(injection))

if DOPLOT:
    plt.figure(figsize=(8,6))
    plt.plot(t, y, label="data", color="lightskyblue")
    plt.plot(t, injection, label="injection", color="crimson")
    plt.xlabel(r'$x$')
    plt.ylabel(r'Amplitude')
    plt.show()
    plt.close()


nwalkers     = 20
ntemps       = 10
ndims        = [3, 1]
nleaves_max  = [10, 1]
branch_names = ["gauss", "noise"]

priors = {
    "gauss": {
        0: uniform_dist(0.1, 5.),
        1: uniform_dist(t.min(), t.max()),
        2: uniform_dist(0.01, 0.5),
    },
    "noise": {
        0: uniform_dist(0.00001, 2.),
    }
}

coords = {
    name: np.zeros((ntemps, nwalkers, nleaf, ndim))
    for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names)
}

sig1 = 0.0000001
for nleaf, ndim, name in zip(nleaves_max, ndims, branch_names):
    for nn in range(nleaf):
        if name == "gauss":
            if nn >= len(gauss_inj_params):
                nn = np.random.randint(low=0, high=3)
            coords[name][:, :, nn] = np.random.multivariate_normal(gauss_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers))
        else:
            coords[name][:, :, nn] = np.random.multivariate_normal(sigma[nn], np.diag(np.ones(1) * sig1), size=(ntemps, nwalkers))  


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

# We need to write down this in the help/tutorial pages:
# Likelihood for many model types needs coords_1, coords_2, ...,
# and then groups_1, groups_2, ... . Took me a while to figure it out
log_prob = log_prob_fn(
    [coords_in["gauss"], coords_in["noise"]],
    [groups_in["gauss"],groups_in["noise"]],
    t,
    y,
    fill_inds=[],
    fill_values=None,
)

log_prob = log_prob.reshape(ntemps, nwalkers)
betas = np.linspace(1.0, 0.0, ntemps)

factor = 0.0001
cov    = {"gauss": np.diag(np.ones(3)) * factor, 
          "noise": np.diag(np.ones(1)) * factor}
moves  = GaussianMove(cov)

usedr = False
# usedr = GaussianMove(cov)
dr_miter = 5

if usedr:
    fname = "output_DR{}_I{}".format(True, dr_miter)
else:
    fname = "output_DR{}".format(usedr)

backend = HDFBackend(fname + ".h5")

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
    plot_iterations=-1,
    moves=moves,
    rj_moves=True,
    dr_moves=usedr,
    dr_max_iter=dr_miter,
    backend=backend,
)

nsteps = 10000
burnin = 2000

state = State(coords, log_prob=log_prob, betas=betas, blobs=None, inds=inds)

ensemble.run_mcmc(state, nsteps, burn=burnin, progress=True, thin_by=1)

check = ensemble.backend.get_autocorr_time(average=True, all_temps=True)

# Define some parameter names for testing
paramnames = [r'$A$',r'$\mu$',r'$\sigma$',r'$\sigma_n$']

plot = PlotContainer(backend=ensemble.backend, fp=fname)

plot.generate_corner(labels=paramnames)

plot.generate_k_per_temperature_chains()

plot.generate_temperature_chains()

plot.generate_parameter_chains_per_temperature()