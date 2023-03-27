import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import uniform_dist
from eryn.moves import GaussianMove, BasicSymmetricModelSwapRJMove

# make the plots look a bit nicer with some defaults
import matplotlib as mpl
import matplotlib.pyplot as plt
rcparams = {}
rcparams["axes.linewidth"] = 0.5
rcparams["font.family"] = "serif"
rcparams["font.size"] = 22
rcparams["legend.fontsize"] = 16
rcparams["mathtext.fontset"] = "stix"
rcparams["text.usetex"] = True 
mpl.rcParams.update(rcparams) # update plot parameters

# set random seed
np.random.seed(42)

from chainconsumer import ChainConsumer
from scipy.stats import cauchy

# Define the injection signals. In this simple example case they are Gaussian and Cauchy pulses
def gaussian_pulse(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x

def cauchy_pulse(x, a, b, c):
    f_x = a * cauchy.pdf(x, loc=b, scale=c) / 3
    return f_x

def combine_gaussians(t, params):
    template = np.zeros_like(t)
    params = np.atleast_2d(params)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template

def combine_cauchys(t, params):
    template = np.zeros_like(t)
    params = np.atleast_2d(params)
    for param in params:
        template += cauchy_pulse(t, *param)  # *params -> a, b, c
    return template

# Define the log-likelihood. Input parameters are in a list, where 
# the "inactive" model is set to None. We assume that both models require
# the same number and the same type of parameters.  
def log_like_fn_gauss_pulse(params, t, data, sigma):
        
    if params[0] is not None:
        template = combine_gaussians(t, params[0])
    if params[1] is not None:
        template = combine_cauchys(t, params[1])
    
    ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll

# Set up the sampler
nwalkers = 20
ntemps = 8
ndim = [3, 3]
nleaves_max = [1, 1]
nleaves_min = [0, 0]

branch_names = ["gauss", "cauchy"]

# define time stream
num = 500
t = np.linspace(-1, 1, num)

inj_params = [[3.4, 0.0, 0.1],]

# combine gaussians
injection = combine_gaussians(t, np.asarray(inj_params))

# set noise level
sigma = 2.0

# produce full data
y = injection + sigma * np.random.randn(len(injection))

plt.figure(figsize=(10,5))
plt.plot(t, y, label="data", color="lightskyblue")
plt.plot(t, injection, label="Injection (Gaussian)", color="crimson")
plt.plot(t, combine_cauchys(t, np.asarray(inj_params)), label="Evaluation of Cauchy at injected parameters", color="orange")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()

# Start setting up the coordinates to feed Eryn with
coords = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max[0], ndim[0])),
          "cauchy": np.zeros((ntemps, nwalkers, nleaves_max[1], ndim[1]))}

# this is the sigma for the injected signals that sets starting points
# We need it to be very small to assume we are passed the search phase
# we will verify this is with likelihood calculations
sig1 = 0.0001

# setup initial walkers to be the correct count (it will spread out)
for nn in range(nleaves_max[0]):
    if nn >= len(inj_params):
        # not going to add parameters for these unused leaves
        continue
        
    coords["gauss"][:, :, nn] = np.random.multivariate_normal(inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)) 
    coords["cauchy"][:, :, nn] = np.random.multivariate_normal(inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers))

# make sure to start near the proper setup
inds_choices = np.random.randint(low=0, high=2, size=(ntemps, nwalkers, 1))

inds = {"gauss": np.array(inds_choices, dtype=bool),
        "cauchy": ~np.array(inds_choices, dtype=bool)}

# describes priors for all leaves independently
priors = {
    "gauss": {
        0: uniform_dist(2.5, 3.5),  # amplitude
        1: uniform_dist(t.min(), t.max()),  # mean 
        2: uniform_dist(0.01, 0.21),  # sigma
    },
    "cauchy": {
        0: uniform_dist(2.5, 3.5),  # amplitude
        1: uniform_dist(t.min(), t.max()),  # mean 
        2: uniform_dist(0.01, 0.21),  # sigma
    },
}

# for the Gaussian Move, will be explained later
factor = 0.00001
cov = {"gauss": np.diag(np.ones(ndim[0])) * factor,
       "cauchy": np.diag(np.ones(ndim[1])) * factor}

moves = GaussianMove(cov) # In-model step proposal

rjmoves = BasicSymmetricModelSwapRJMove(nleaves_max, nleaves_min) # Outer-model step proposal

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
    rj_moves=rjmoves,  # basic generation of new leaves from the prior
)

log_prior = ensemble.compute_log_prior(coords, inds=inds)
log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

# setup starting state
state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

nsteps = 2000
last_sample = ensemble.run_mcmc(state, nsteps, burn=1000, progress=True, thin_by=1)

last_sample.branches["gauss"].nleaves
last_sample.branches["cauchy"].nleaves

print(f'max ll: {ensemble.get_log_like().max()}')

# END