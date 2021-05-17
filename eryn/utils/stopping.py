# -*- coding: utf-8 -*-

from abc import ABC

import numpy as np


class Stopping(ABC, object):
    @classmethod
    def __call__(self, iter, last_sample, sampler):
        raise NotImplementedError


class AutoCorrelationStop(Stopping):
    def __init__(self, autocorr_multiplier=50, verbose=False):
        self.autocorr_multiplier = autocorr_multiplier
        self.verbose = verbose

        self.time = 0

    def __call__(self, iter, last_sample, sampler):

        tau = sampler.backend.get_autocorr_time(multiply_thin=False)

        if self.time > 0:
            # backend iteration
            iteration = sampler.backend.iteration

            finish = []

            for name, values in tau.items():
                converged = np.all(tau[name] * self.autocorr_multiplier < iteration)
                converged &= np.all(
                    np.abs(self.old_tau[name] - tau[name]) / tau[name] < 0.01
                )

                finish.append(converged)

            stop = True if np.all(finish) else False
            if self.verbose:
                print(
                    "\ntau:",
                    tau,
                    "\nIteration:",
                    iteration,
                    "\nAutocorrelation multiplier:",
                    self.autocorr_multiplier,
                    "\nStopping:",
                    stop,
                    "\n",
                )

        else:
            stop = False

        self.old_tau = tau
        self.time += 1
        return stop
