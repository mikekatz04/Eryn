#!/usr/bin/env python
# coding: utf-8

# In[14]:


from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds

import unittest
import matplotlib.pyplot as plt
import numpy as np

# set random seed
np.random.seed(42)

import corner

# Gaussian likelihood
def log_like_fn(x, mu, invcov):
    diff = x - mu
    return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()


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


def gaussian_pulse(x, a, b, c):
    f_x = a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    return f_x


def combine_gaussians(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += gaussian_pulse(t, *param)  # *params -> a, b, c
    return template


def sine(x, a, b, c):
    f_x = a * np.sin(2 * np.pi * b * x + c)
    return f_x


def combine_sine(t, params):
    template = np.zeros_like(t)
    for param in params:
        template += sine(t, *param)  # *params -> a, b, c
    return template


def log_like_fn_gauss_and_sine(params_both, t, data, sigma):

    params_gauss, params_sine = params_both
    template = np.zeros_like(t)

    if params_gauss is not None:
        template += combine_gaussians(t, params_gauss)

    if params_sine is not None:
        template += combine_sine(t, params_sine)

    ll = -0.5 * np.sum(((template - data) / sigma) ** 2, axis=-1)
    return ll


class WaveformTest(unittest.TestCase):
    def test_base(self):

        ndim = 5
        nwalkers = 100

        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        lims = 5.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        ensemble = EnsembleSampler(
            nwalkers, ndim, log_like_fn, priors, args=[means, invcov],
        )

        coords = priors.rvs(size=(nwalkers,))

        # check log_like
        log_like = np.asarray(
            [log_like_fn(coords[i], means, invcov) for i in range(nwalkers)]
        )

        # check log_prior
        log_prior = np.asarray([priors.logpdf(coords[i]) for i in range(nwalkers)])

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

        samples = ensemble.get_chain()["model_0"].reshape(-1, ndim)

        ll = ensemble.backend.get_log_like()
        lp = ensemble.backend.get_log_prior()

        samples = ensemble.get_chain()

        ensemble.backend.shape

        last_state = ensemble.backend.get_last_sample()

        last_state.branches

        last_state.branches["model_0"].coords

    def test_pt(self):
        # set up problem
        ndim = 5
        nwalkers = 100
        ntemps = 10

        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        lims = 5.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        # fill kwargs dictionary
        tempering_kwargs = dict(ntemps=ntemps)

        # randomize throughout prior
        coords = priors.rvs(size=(ntemps, nwalkers,))

        # initialize sampler
        ensemble_pt = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, cov],
            tempering_kwargs=tempering_kwargs,
        )

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        ensemble_pt.run_mcmc(coords, nsteps, burn=burn, progress=False, thin_by=thin_by)

        for temp in range(ntemps):
            samples = ensemble_pt.get_chain()["model_0"][:, temp].reshape(-1, ndim)

        ll = ensemble_pt.backend.get_log_like()

    def test_rj(self):

        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = 8
        nleaves_min = 0

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

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

            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        moves = GaussianMove(cov)

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
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 20
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        last_sample.branches["gauss"].nleaves

        nleaves = ensemble.get_nleaves()["gauss"]
        bns = (
            np.arange(1, nleaves_max + 2) - 0.5
        )  # Just to make it pretty and center the bins

        samples = ensemble.get_chain()["gauss"][:, 0].reshape(-1, ndim)

        # same as ensemble.get_chain()['gauss'][ensemble.get_inds()['gauss']]
        samples = samples[~np.isnan(samples[:, 0])]

        means = np.asarray(gauss_inj_params)[:, 1]

    def test_rj_multiple_branches(self):

        nwalkers = 20
        ntemps = 8
        ndims = [3, 3]
        nleaves_max = [8, 4]
        nleaves_min = [0, 0]

        branch_names = ["gauss", "sine"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        sine_inj_params = [
            [1.3, 10.1, 1.0],
            [0.8, 4.6, 1.2],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))
        injection += combine_sine(t, np.asarray(sine_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max[0], ndims[0])),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max[1], ndims[1])),
        }

        # make sure to start near the proper setup
        inds = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max[0]), dtype=bool),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max[1]), dtype=bool),
        }

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        # start with gaussians
        for nn in range(nleaves_max[0]):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )
            inds["gauss"][:, :, nn] = True

        # next do sine waves
        for nn in range(nleaves_max[1]):
            if nn >= len(sine_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["sine"][:, :, nn] = np.random.multivariate_normal(
                sine_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)
            )
            inds["sine"][:, :, nn] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
            "sine": {
                0: uniform_dist(0.5, 1.5),  # amplitude
                1: uniform_dist(1.0, 20.0),  # mean
                2: uniform_dist(0.0, 2 * np.pi),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {
            "gauss": np.diag(np.ones(ndims[0])) * factor,
            "sine": np.diag(np.ones(ndims[1])) * factor,
        }

        moves = GaussianMove(cov)

        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn_gauss_and_sine,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves=True,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        state.branches

        nsteps = 50
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        np.array(
            [
                last_sample.branches["gauss"].nleaves[0],
                last_sample.branches["sine"].nleaves[0],
            ]
        ).T

        nleaves_gauss = ensemble.get_nleaves()["gauss"]
        nleaves_sine = ensemble.get_nleaves()["sine"]

        samples = ensemble.get_chain()["gauss"][:, 0].reshape(-1, ndims[0])

        # same as ensemble.get_chain()['gauss'][ensemble.get_inds()['gauss']]
        samples = samples[~np.isnan(samples[:, 0])]

        means = np.asarray(gauss_inj_params)[:, 1]

    def test_gibbs_sampling(self):

        nwalkers = 20
        ntemps = 8
        ndims = [3, 3]
        nleaves_max = [8, 2]  # same min and max means no changing
        nleaves_min = [0, 2]

        branch_names = ["gauss", "sine"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        sine_inj_params = [
            [1.3, 10.1, 1.0],
            [0.8, 4.6, 1.2],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))
        injection += combine_sine(t, np.asarray(sine_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

        coords = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max[0], ndims[0])),
            "sine": np.zeros((ntemps, nwalkers, nleaves_max[1], ndims[1])),
        }

        # make sure to start near the proper setup
        inds = {
            "gauss": np.zeros((ntemps, nwalkers, nleaves_max[0]), dtype=bool),
            "sine": np.ones((ntemps, nwalkers, nleaves_max[1]), dtype=bool),
        }

        # this is the sigma for the multivariate Gaussian that sets starting points
        # We need it to be very small to assume we are passed the search phase
        # we will verify this is with likelihood calculations
        sig1 = 0.0001

        # setup initial walkers to be the correct count (it will spread out)
        # start with gaussians
        for nn in range(nleaves_max[0]):
            if nn >= len(gauss_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )
            inds["gauss"][:, :, nn] = True

        # next do sine waves
        for nn in range(nleaves_max[1]):
            if nn >= len(sine_inj_params):
                # not going to add parameters for these unused leaves
                continue
            coords["sine"][:, :, nn] = np.random.multivariate_normal(
                sine_inj_params[nn], np.diag(np.ones(3) * sig1), size=(ntemps, nwalkers)
            )
            # inds["sine"][:, :, nn] = True  # already True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
            "sine": {
                0: uniform_dist(0.5, 1.5),  # amplitude
                1: uniform_dist(1.0, 20.0),  # mean
                2: uniform_dist(0.0, 2 * np.pi),  # sigma
            },
        }

        # for the Gaussian Move
        factor = 0.00001
        cov = {
            "gauss": np.diag(np.ones(ndims[0])) * factor,
        }

        # pass boolean array of shape (nleaves_max[0], ndims[0])
        gibbs_sampling_gauss = [
            ("gauss", np.zeros((nleaves_max[0], ndims[0]), dtype=bool))
            for _ in range(nleaves_max[0])
        ]

        for i in range(nleaves_max[0]):
            gibbs_sampling_gauss[i][-1][i] = True

        gauss_move = GaussianMove(cov, gibbs_sampling_setup=gibbs_sampling_gauss)

        gibbs_sampling_sine = [
            ("sine", np.zeros((nleaves_max[1], ndims[1]), dtype=bool))
            for _ in range(2 * nleaves_max[1])
        ]
        for i in range(nleaves_max[1]):
            for j in range(2):
                if j == 0:
                    gibbs_sampling_sine[2 * i + j][-1][i, :2] = True
                else:
                    gibbs_sampling_sine[2 * i + j][-1][i, 2:] = True

        sine_move = StretchMove(
            live_dangerously=True, gibbs_sampling_setup=gibbs_sampling_sine
        )

        move = CombineMove([gauss_move, sine_move])

        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn_gauss_and_sine,
            priors,
            args=[t, y, sigma],
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=move,
            rj_moves=True,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 50
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

    def test_utilities(self):
        def transform1(x, y):
            return x * y, y / x

        # this will do transform lambda x, y: (x**2, y**2) before transform1
        parameter_transforms = {
            0: lambda x: np.log(x),
            (1, 2): lambda x, y: (x ** 2, y ** 2),
            (0, 2): transform1,
        }

        fill_dict = {
            "ndim_full": 6,  # full dimensionality after values are added
            "fill_inds": np.array([2, 3, 5]),  # indexes for fill values in final array
            "fill_values": np.array([0.0, 1.0, -1.0]),  # associated values for filling
        }

        tc = TransformContainer(
            parameter_transforms=parameter_transforms, fill_dict=fill_dict
        )

        x = np.random.uniform(0.1, 4.0, size=(40, 3))

        # can copy and transpose values if needed
        out = tc.transform_base_parameters(x, copy=True, return_transpose=False)

        def lnprob(x1, group1, x2, group2, transform_containers):
            x = [x1, x2]
            for i, (x_i, transform) in enumerate(zip([x1, x2], transform_containers)):
                temp = transform.transform_base_parameters(
                    x_i, copy=True, return_transpose=False
                )
                x[i] = transform.fill_values(temp)

            ## do more in the likelihood here with transformed information

        # setup transforms for x1
        parameter_transforms1 = {0: lambda x: np.log(x)}

        # setup transforms for x2
        parameter_transforms2 = {(1, 2): lambda x, y: (x ** 2, y ** 2)}

        # fill dict for x1
        fill_dict1 = {
            "ndim_full": 6,  # full dimensionality after values are added
            "fill_inds": np.array([2, 3, 5]),  # indexes for fill values in final array
            "fill_values": np.array([0.0, 1.0, -1.0]),  # associated values for filling
        }

        # fill dict for x2
        fill_dict2 = {
            "ndim_full": 5,  # full dimensionality after values are added
            "fill_inds": np.array([1]),  # indexes for fill values in final array
            "fill_values": np.array([-1.0]),  # associated values for filling
        }

        tcs = [
            TransformContainer(
                parameter_transforms=parameter_transforms1, fill_dict=fill_dict1
            ),
            TransformContainer(
                parameter_transforms=parameter_transforms2, fill_dict=fill_dict2
            ),
        ]

        num = 40
        x1 = np.random.uniform(0.1, 4.0, size=(num, 3))
        x2 = np.random.uniform(0.1, 4.0, size=(num, 4))

        group1 = np.arange(num)
        group2 = np.arange(num)

        # it can be added via args or kwargs in the ensemble sampler
        lnprob(x1, group1, x2, group2, tcs)

        # ### Periodic Container

        from eryn.utils import PeriodicContainer

        periodic = PeriodicContainer({"sine": {2: 2 * np.pi}})
        ntemps, nwalkers, nleaves_max, ndim = (10, 100, 2, 3)

        params_before_1 = {
            "sine": np.random.uniform(
                0, 7.0, size=(ntemps * nwalkers, nleaves_max, ndim)
            )
        }
        params_before_2 = {
            "sine": np.random.uniform(
                0, 7.0, size=(ntemps * nwalkers, nleaves_max, ndim)
            )
        }

        distance = periodic.distance(params_before_1, params_before_2)

        # the max distance should be near half the period

        params_after_1 = periodic.wrap(params_before_1)

        # max after wrapping should be near the period

        # ### Stopping & Update Functions
        from eryn.utils import SearchConvergeStopping

        ndim = 5
        nwalkers = 100

        # mean
        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        # set prior limits
        lims = 50.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        stop = SearchConvergeStopping(n_iters=5, diff=0.01, verbose=False)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, invcov],
            stopping_fn=stop,
            stopping_iterations=5,
        )

        # starting positions
        # randomize throughout prior
        coords = priors.rvs(size=(nwalkers,))

        # check log_like
        log_like = np.asarray(
            [log_like_fn(coords[i], means, invcov) for i in range(nwalkers)]
        )

        # check log_prior
        log_prior = np.asarray([priors.logpdf(coords[i]) for i in range(nwalkers)])

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1
        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

    def test_group_stretch(self):
        from eryn.moves import GroupStretchMove

        class MeanGaussianGroupMove(GroupStretchMove):
            def __init__(self, **kwargs):
                GroupStretchMove.__init__(self, **kwargs)

            def setup_friends(self, branches):
                self.friends = branches["gauss"].coords[branches["gauss"].inds]
                self.means = self.friends[:, 1]

            def find_friends(self, name, s, s_inds=None):
                friends = np.zeros_like(s)
                means_here = s[s_inds][:, 1]
                dist = np.abs(means_here[:, None] - self.means[None, :])
                dist_inds_sort = np.argsort(dist, axis=-1)
                inds_choice = np.random.randint(
                    0, self.nfriends, size=means_here.shape[0]
                )
                keep = dist_inds_sort[(np.arange(inds_choice.shape[0]), inds_choice)]

                friends[s_inds] = self.friends[keep]
                return friends

        # set random seed
        np.random.seed(42)

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

        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = 8
        nleaves_min = 0

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

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

            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": {
                0: uniform_dist(2.5, 3.5),  # amplitude
                1: uniform_dist(t.min(), t.max()),  # mean
                2: uniform_dist(0.01, 0.21),  # sigma
            },
        }

        # for the Gaussian Move, will be explained later
        # factor = 0.00001
        # cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        # moves = GaussianMove(cov)

        moves = MeanGaussianGroupMove(nfriends=nwalkers)

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
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 20
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        nleaves = ensemble.get_nleaves()["gauss"][:, 0].flatten()

    def test_mt(self):
        # Gaussian likelihood
        def log_like_fn(x, mu, invcov):
            diff = x - mu
            return -0.5 * (diff * np.dot(invcov, diff.T).T).sum()

        ndim = 5
        nwalkers = 100

        # mean
        means = np.zeros(ndim)  # np.random.rand(ndim)

        # define covariance matrix
        cov = np.diag(np.ones(ndim))
        invcov = np.linalg.inv(cov)

        # set prior limits
        lims = 5.0
        priors_in = {
            i: uniform_dist(-lims + means[i], lims + means[i]) for i in range(ndim)
        }
        priors = ProbDistContainer(priors_in)

        nwalkers = 20
        ntemps = 10
        nleaves_max = 1

        from eryn.moves import MTDistGenMove

        mt_prior = MTDistGenMove(priors, num_try=25, independent=True)

        ensemble = EnsembleSampler(
            nwalkers,
            ndim,
            log_like_fn,
            priors,
            args=[means, invcov],
            moves=mt_prior,
            tempering_kwargs={"ntemps": ntemps},
        )

        # starting positions
        # randomize throughout prior
        coords = priors.rvs(size=(ntemps, nwalkers, 1))

        nsteps = 50
        # burn for 1000 steps
        burn = 10
        # thin by 5
        thin_by = 1

        out = ensemble.run_mcmc(
            coords, nsteps, burn=burn, progress=False, thin_by=thin_by
        )

        samples_out = ensemble.get_chain()["model_0"][:, 0].reshape(-1, ndim)

    def test_mt_rj(self):
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

        nwalkers = 20
        ntemps = 8
        ndim = 3
        nleaves_max = 8
        nleaves_min = 0

        branch_names = ["gauss"]

        # define time stream
        num = 500
        t = np.linspace(-1, 1, num)

        gauss_inj_params = [
            [3.3, -0.2, 0.1],
            [2.6, -0.1, 0.1],
            [3.4, 0.0, 0.1],
            [2.9, 0.3, 0.1],
        ]

        # combine gaussians
        injection = combine_gaussians(t, np.asarray(gauss_inj_params))

        # set noise level
        sigma = 2.0

        # produce full data
        y = injection + sigma * np.random.randn(len(injection))

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

            coords["gauss"][:, :, nn] = np.random.multivariate_normal(
                gauss_inj_params[nn],
                np.diag(np.ones(3) * sig1),
                size=(ntemps, nwalkers),
            )

        # make sure to start near the proper setup
        inds = {"gauss": np.zeros((ntemps, nwalkers, nleaves_max), dtype=bool)}

        # turn False -> True for any binary in the sampler
        inds["gauss"][:, :, : len(gauss_inj_params)] = True

        # describes priors for all leaves independently
        priors = {
            "gauss": ProbDistContainer(
                {
                    0: uniform_dist(2.5, 3.5),  # amplitude
                    1: uniform_dist(t.min(), t.max()),  # mean
                    2: uniform_dist(0.01, 0.21),  # sigma
                }
            )
        }

        # for the Gaussian Move, will be explained later
        factor = 0.00001
        cov = {"gauss": np.diag(np.ones(ndim)) * factor}

        moves = GaussianMove(cov)

        from eryn.moves import MTDistGenMoveRJ

        mt_rj_prior = MTDistGenMoveRJ(
            priors, max_k=[nleaves_max], min_k=[nleaves_min], num_try=25, rj=True
        )

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
            rj_moves=mt_rj_prior,  # basic generation of new leaves from the prior
        )

        log_prior = ensemble.compute_log_prior(coords, inds=inds)
        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

        # make sure it is reasonably close to the maximum which this is
        # will not be zero due to noise

        # setup starting state
        state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        nsteps = 20
        last_sample = ensemble.run_mcmc(
            state, nsteps, burn=10, progress=False, thin_by=1
        )

        nleaves = ensemble.get_nleaves()["gauss"]
        bns = (
            np.arange(1, nleaves_max + 2) - 0.5
        )  # Just to make it pretty and center the bins

