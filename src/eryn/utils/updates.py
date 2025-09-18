# -*- coding: utf-8 -*-

from abc import ABC

import numpy as np


class Update(ABC, object):
    """Update the sampler."""

    @classmethod
    def __call__(self, iter, last_sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).

        """
        raise NotImplementedError


class AdjustStretchProposalScale(Update):
    def __init__(
        self,
        target_acceptance=0.22,
        supression_factor=0.1,
        max_change=0.5,
        verbose=False,
    ):
        """Adjusted scale for stretch proposal based on cold chain acceptance rate"""
        self.target_acceptance = target_acceptance
        self.verbose = verbose
        self.max_change, self.supression_factor = max_change, supression_factor

        self.time = 0

    def __call__(self, iter, last_sample, sampler):

        mean_af = 0.0
        change = 1.0
        if self.time > 0:
            # cold chain -> 0
            mean_af = np.mean(
                (sampler.backend.accepted[:, 0] - self.previously_accepted)
                / (sampler.backend.iteration - self.previous_iter)
            )

            if mean_af > self.target_acceptance:
                factor = self.supression_factor * (mean_af / self.target_acceptance)
                if factor > self.max_change:
                    factor = self.max_change
                change = 1 + self.supression_factor * factor

            else:
                factor = self.supression_factor * (self.target_acceptance / mean_af)
                if factor > self.max_change:
                    factor = self.max_change
                change = 1 - factor

            sampler._moves[0].a *= change

        self.previously_accepted = sampler.backend.accepted[:, 0].copy()
        print(
            self.previously_accepted, "\n", mean_af, change, "\n", sampler._moves[0].a
        )
        self.previous_iter = sampler.backend.iteration
        self.time += 1
