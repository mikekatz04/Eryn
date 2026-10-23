# -*- coding: utf-8 -*-

"""Sampler stopping hooks (``EnsembleSampler(stopping_fn=..., stopping_iterations=...)``).

:meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` calls ``stopping_fn(i, state, sampler)``
after every ``stopping_iterations``-th stored step and ends the run when it returns
``True``. :class:`Stopping` is the base class; :class:`SearchConvergeStopping` stops once
the best log-likelihood has stopped improving.
"""

from abc import ABC

import numpy as np


class Stopping(ABC, object):
    """Base class for stopping criteria.

    Subclasses implement ``__call__(self, iter, last_sample, sampler) -> bool``. The sampler
    only calls it every ``stopping_iterations`` stored steps, and each stored step is
    ``thin_by`` sampler iterations, so checks happen every
    ``thin_by * stopping_iterations`` iterations.
    """

    @classmethod
    def __call__(self, iter, last_sample, sampler):
        """Decide whether to stop sampling (override in subclasses).

        Args:
            iter (int): Step counter passed by the sampler.
            last_sample (State): Last state of the sampler (:class:`eryn.state.State`).
            sampler (EnsembleSampler): The sampler (:class:`eryn.ensemble.EnsembleSampler`).

        Returns:
            bool: ``True`` to stop sampling.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError


class SearchConvergeStopping(Stopping):
    """Stop once the maximum log-likelihood has stopped changing.

    At each check the best log-likelihood stored so far (all temperatures and walkers,
    after discarding the first ``start_iteration`` stored samples) is compared with the
    previous best. A change below ``diff`` counts one pass; a change of ``diff`` or more
    resets the count and records the new best. After ``n_iters`` consecutive passes the
    call returns ``True`` and the count resets.

    Checks happen every ``stopping_iterations`` stored steps, i.e. every
    ``thin_by * stopping_iterations`` sampler iterations. All arguments are stored as
    attributes.

    Args:
        n_iters (int, optional): Consecutive passing checks needed to stop. Default
            ``30``.
        diff (float, optional): Log-likelihood change that fails a check. Default ``0.1``.
        start_iteration (int, optional): Calls with ``iter < start_iteration`` return
            ``False``; also the ``discard`` used when reading the log-likelihood. Default
            ``0``.
        verbose (bool, optional): If ``True``, print the count and both maxima at every
            check. Default ``False``.

    Attributes:
        iters_consecutive (int): Number of consecutive passes so far.
        past_like_best (float): Best log-likelihood at the last reset; starts at
            ``-np.inf``.

    Example:
        .. code-block:: python

            stop = SearchConvergeStopping(n_iters=30, diff=1.0, start_iteration=0)
            sampler = EnsembleSampler(..., stopping_fn=stop, stopping_iterations=1)
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
        """Run one convergence check.

        Args:
            iter (int): Step counter passed by the sampler.
            sample (State): Last state of the sampler (unused).
            sampler (EnsembleSampler): The sampler; ``sampler.get_log_like`` is read.

        Returns:
            bool: ``True`` to stop sampling.
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
