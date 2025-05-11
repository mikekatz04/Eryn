import matplotlib.pyplot as plt

import numpy as np

import cupy as cp

from eryn.paraensemble import ParaEnsembleSampler

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import ParaState

import corner


# Gaussian likelihood
def log_like_fn(x, mu, invcov, use_gpu=False):

    xp = cp if use_gpu else np

    diff = x - mu
    # return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()
    out = -0.5 * xp.einsum("...i,...ij,...j->...", diff, invcov, diff)
    return out  # xp.zeros(x.shape[0])


class PriorTransformFn:
    def __init__(self, f_min, f_max, fdot_min, fdot_max):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = f_min, f_max, fdot_min, fdot_max

    def __call__(self, logp, groups_running):
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1. / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1. / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]
        return logp


if __name__ == "__main__":
    ngroups = 1000
    ndim = 5
    ntemps = 10
    nwalkers = 100
    use_gpu = True

    xp = cp if use_gpu else np
    gpu = 6 if use_gpu else None

    if use_gpu:
        cp.cuda.runtime.setDevice(gpu)

    prior_transform_fn = PriorTransformFn(xp.full(ngroups, 1.0), xp.full(ngroups, 1.1), xp.full(ngroups, 1e-21), xp.full(ngroups, 1e-13))

    groups_running = xp.ones(ngroups, dtype=bool)

    # set up problem

    means = xp.zeros(ndim)  # np.random.rand(ndim)

    # define covariance matrix
    cov = xp.diag(xp.ones(ndim))
    invcov = xp.linalg.inv(cov)

    lims = 5.0
    priors_in = {
        i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
    }
    priors = {"gauss": ProbDistContainer(priors_in, use_cupy=use_gpu)}

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
        kwargs={"use_gpu": use_gpu},
        gpu=gpu,
        periodic=None,
        backend=None,
        update_fn=None,
        update_iterations=-1,
        stopping_fn=None,
        stopping_iterations=-1,
        name="gauss",
        prior_transform_fn=prior_transform_fn
    )

    nsteps = 500
    # burn for 1000 steps
    burn = 10
    # thin by 5
    thin_by = 25

    state = ParaState({"gauss": coords}, groups_running=groups_running)

    ensemble_pt.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)

    # ll = ensemble_pt.backend.get_log_like()

    samples = (
        ensemble_pt.get_chain()[:, :, 0]
        .transpose(1, 0, 2, 3)
        .reshape(ngroups, -1, ndim)
    )

    fig = None
    for i in range(samples.shape[0])[:10]:
        fig = corner.corner(
            samples[i],
            plot_datapoints=False,
            plot_density=False,
            color=f"C{i}",
            levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2),
            fig=fig,
        )
    fig.savefig("check_para.png")
    breakpoint()
