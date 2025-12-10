# -*- coding: utf-8 -*-

from abc import ABC
import dataclasses

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

class CompositeUpdate(Update):
    """A composite update that chains multiple Update objects together."""
    
    def __init__(self, updates: list):
        """
        Args:
            updates (list): List of Update objects to chain together.
        """
        self._updates = updates
    
    def __call__(self, iter, last_sample, sampler):
        """Call all chained updates in sequence."""
        for update in self._updates:
            update(iter, last_sample, sampler)
    
    def __add__(self, other):
        """Concatenate with another Update or CompositeUpdate."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(self._updates + other._updates)
        elif isinstance(other, Update):
            return CompositeUpdate(self._updates + [other])
        else:
            return NotImplemented
    
    def __radd__(self, other):
        """Support other + self."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(other._updates + self._updates)
        elif isinstance(other, Update):
            return CompositeUpdate([other] + self._updates)
        else:
            return NotImplemented
    
    def __repr__(self):
        return f"CompositeUpdate({self._updates})"


@dataclasses.dataclass
class UpdateStep(Update):
    """Base class for chainable update steps."""
    nsteps: int = 100
    increment: int = 1
    increment_every: int = 500
    stop: int = None
    
    def __add__(self, other):
        """Concatenate with another Update or CompositeUpdate."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate([self] + other._updates)
        elif isinstance(other, Update):
            return CompositeUpdate([self, other])
        else:
            return NotImplemented
    
    def __radd__(self, other):
        """Support other + self."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(other._updates + [self])
        elif isinstance(other, Update):
            return CompositeUpdate([other, self])
        else:
            return NotImplemented
    
    def check_step(self, iteration):
        """Check if the update should be applied at this iteration.
        
        The diagnostic frequency decreases over time. The interval between
        diagnostics is multiplied by `increment` every `increment_every` steps.
        
        Example with nsteps=100, increment=2, increment_every=500:
            - iterations 0-499: check every 100 steps (but not at 0)
            - iterations 500-999: check every 200 steps
            - iterations 1000-1499: check every 400 steps
            - etc.
        """
        if iteration == 0:
            return False
        
        exponent = iteration // self.increment_every
        interval = self.nsteps * (self.increment ** exponent)
        
        if self.stop is not None and iteration >= self.stop:
            return False
        
        return (iteration % interval == 0)

    def update(self, iteration, last_sample, sampler):
        """Override this method in subclasses to define the update behavior."""
        raise NotImplementedError("Subclasses must implement update() method.")
    
    def __call__(self, iteration, last_sample, sampler):
        """Call the update if the step condition is met."""
        if self.check_step(iteration):
            print(f'Calling {self.__class__.__name__} at iteration {iteration}')
            self.update(iteration, last_sample, sampler)
        
   

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
