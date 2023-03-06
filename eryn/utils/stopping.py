# -*- coding: utf-8 -*-

from abc import ABC

import numpy as np


class Stopping(ABC, object):
    """Base class for stopping.
    
    Stopping checks are only performed every ``thin_by`` iterations.
    
    """

    @classmethod
    def __call__(self, iter, last_sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: Value of ``stop``. If ``True``, stop sampling.
            
        """
        raise NotImplementedError


class SearchConvergeStopping(Stopping):
    """Stopping function based on a convergence to a maximunm Likelihood.

    Stopping checks are only performed every ``thin_by`` iterations.
    Therefore, the iterations of stopping checks are really every
    ``sampler iterations * thin_by``.  

    All arguments are stored as attributes.

    Args:
        n_iters (int, optional): Number of iterative stopping checks that need to pass
            in order to stop the sampler. (default: ``30``)
        diff (float, optional): Change in the Likelihood needed to fail the stopping check. In other words,
            if the new maximum Likelihood is more than ``diff`` greater than the old, all iterative checks 
            reset. (default: 0.1). 
        start_iteration (int, optional): Iteration of sampler to start checking to stop. (default: 0)
        verbose (bool, optional): If ``True``, print information. (default: ``False``)

    Attributes:
        iters_consecutive (int): Number of consecutive passes of the stopping check.
        past_like_best (float): Previous best Likelihood. The initial value is ``-np.inf``.
    
    """

    def __init__(self, n_iters=30, diff=0.1, start_iteration=0, verbose=False):

        # store all the relevant information
        self.n_iters = n_iters

        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

        # initialize important info
        self.iters_consecutive = 0
        self.past_like_best = -np.inf

    def __call__(self, iter, sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: Value of ``stop``. If ``True``, stop sampling.
            
        """

        # if we have not reached the start iteration return
        if iter < self.start_iteration:
            return False

        # get best Likelihood so far
        like_best = sampler.get_log_like(discard=self.start_iteration).max()

        # compare to last
        # if it is less than diff change it passes
        if np.abs(like_best - self.past_like_best) < self.diff:
            self.iters_consecutive += 1

        else:
            # if it fails reset iters consecutive
            self.iters_consecutive = 0

            # store new best
            self.past_like_best = like_best

        # print information
        if self.verbose:
            print(
                f"\nITERS CONSECUTIVE: {self.iters_consecutive}",
                f"Previous best LL: {self.past_like_best}",
                f"Current best LL: {like_best}\n",
            )

        if self.iters_consecutive >= self.n_iters:
            # if we have passes the number of iters necessary, return True and reset
            self.iters_consecutive = 0
            return True

        else:
            return False


"""
class AutoCorrelationStop(Stopping):
    # TODO: check and doc this
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
"""
