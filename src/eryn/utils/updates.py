# -*- coding: utf-8 -*-

"""Sampler update hooks (``EnsembleSampler(update_fn=..., update_iterations=...)``).

:meth:`~eryn.ensemble.EnsembleSampler.sample` calls ``update_fn(i, state, sampler)`` after
every ``update_iterations``-th proposal iteration (counted per proposal, independent of
``thin_by``; ``i`` is the 0-based proposal counter of the current ``sample`` call, which
also runs during a ``run_mcmc`` burn-in). :class:`Update` is the base class,
:class:`UpdateStep` adds its own decreasing cadence, :class:`CompositeUpdate` chains
several updates (``update_a + update_b``), and :class:`AdjustStretchProposalScale` tunes
the stretch-move scale.
"""

import dataclasses
from abc import ABC

import numpy as np


class Update(ABC, object):
    """Base class for sampler update hooks.

    Subclasses implement ``__call__(self, iter, last_sample, sampler)`` and may change the
    sampler in place (move settings, temperatures, ...). The return value is ignored.
    """

    @classmethod
    def __call__(self, iter, last_sample, sampler):
        """Apply the update (override in subclasses).

        Args:
            iter (int): Step counter passed by the sampler.
            last_sample (State): Last state of the sampler (:class:`eryn.state.State`).
            sampler (EnsembleSampler): The sampler (:class:`eryn.ensemble.EnsembleSampler`).

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

class CompositeUpdate(Update):
    """Chain several :class:`Update` objects; calling it calls each in order.

    Usually built with ``+`` (``UpdateStep.__add__`` / ``CompositeUpdate.__add__``).

    Args:
        updates (list): Update objects (any callables taking
            ``(iter, last_sample, sampler)``), called in list order.

    Example:
        With ``MyUpdateA`` and ``MyUpdateB`` subclasses of :class:`UpdateStep`:

        .. code-block:: python

            update_fn = MyUpdateA(nsteps=100) + MyUpdateB(nsteps=500)
            sampler = EnsembleSampler(..., update_fn=update_fn, update_iterations=1)
    """
    
    def __init__(self, updates: list):
        """Store the update list (arguments are documented on the class)."""
        self._updates = updates
    
    def __call__(self, iter, last_sample, sampler):
        """Call every chained update with ``(iter, last_sample, sampler)``, in order."""
        for update in self._updates:
            update(iter, last_sample, sampler)
    
    def __add__(self, other):
        """Return a new :class:`CompositeUpdate` with ``other`` appended.

        Raises ``NotImplementedError`` (rather than returning ``NotImplemented``) when
        ``other`` is not an :class:`Update`.
        """
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(self._updates + other._updates)
        elif isinstance(other, Update):
            return CompositeUpdate(self._updates + [other])
        else:
            raise NotImplementedError
    
    def __radd__(self, other):
        """Return a new :class:`CompositeUpdate` with ``other`` prepended (``other + self``)."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(other._updates + self._updates)
        elif isinstance(other, Update):
            return CompositeUpdate([other] + self._updates)
        else:
            raise NotImplementedError
    
    def __repr__(self):
        return f"CompositeUpdate({self._updates})"


@dataclasses.dataclass
class UpdateStep(Update):
    """Base class for chainable updates with their own (growing) cadence.

    A dataclass: subclasses add fields and implement :meth:`update`. Calling the object
    runs :meth:`update` only when :meth:`check_step` allows it, so it is typically passed
    with ``update_iterations=1``. Combine instances with ``+`` into a
    :class:`CompositeUpdate`.

    Args:
        nsteps (int, optional): Base interval between updates. Default ``100``.
        increment (int, optional): Factor applied to the interval every
            ``increment_every`` iterations. Default ``1`` (fixed interval).
        increment_every (int, optional): Iterations between interval increases. Default
            ``500``.
        stop (int, optional): Iteration from which updates stop. Default ``None`` (never).

    Attributes:
        nsteps (int): Base interval.
        increment (int): Interval growth factor.
        increment_every (int): Iterations between interval increases.
        stop (int or None): Iteration from which updates stop.

    Example:
        .. code-block:: python

            @dataclasses.dataclass
            class PrintAcceptance(UpdateStep):
                def update(self, iteration, last_sample, sampler):
                    print(sampler.acceptance_fraction)

            sampler = EnsembleSampler(
                ..., update_fn=PrintAcceptance(nsteps=50), update_iterations=1
            )
    """
    nsteps: int = 100
    increment: int = 1
    increment_every: int = 500
    stop: int = None
    
    def __add__(self, other):
        """Return a :class:`CompositeUpdate` of ``self`` followed by ``other``."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate([self] + other._updates)
        elif isinstance(other, Update):
            return CompositeUpdate([self, other])
        else:
            return NotImplemented
    
    def __radd__(self, other):
        """Return a :class:`CompositeUpdate` of ``other`` followed by ``self``."""
        if isinstance(other, CompositeUpdate):
            return CompositeUpdate(other._updates + [self])
        elif isinstance(other, Update):
            return CompositeUpdate([other, self])
        else:
            return NotImplemented
    
    def check_step(self, iteration):
        """Whether to run the update at ``iteration``.

        The interval is ``nsteps * increment**(iteration // increment_every)``; the update
        runs when ``iteration`` is a positive multiple of it and below ``stop``. With
        ``nsteps=100, increment=2, increment_every=500``: iterations 100-400 every 100,
        600 and 800 in 500-999, 1200 in 1000-1499, and so on.

        Args:
            iteration (int): Step counter passed by the sampler.

        Returns:
            bool: ``True`` to run the update. Always ``False`` at ``iteration == 0``.
        """
        if iteration == 0:
            return False
        
        exponent = iteration // self.increment_every
        interval = self.nsteps * (self.increment ** exponent)
        
        if self.stop is not None and iteration >= self.stop:
            return False
        
        return (iteration % interval == 0)

    def update(self, iteration, last_sample, sampler):
        """The update itself (override in subclasses).

        Args:
            iteration (int): Step counter passed by the sampler.
            last_sample (State): Last state of the sampler.
            sampler (EnsembleSampler): The sampler.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("Subclasses must implement update() method.")
    
    def __call__(self, iteration, last_sample, sampler):
        """Run :meth:`update` when :meth:`check_step` allows it (and print a line saying so)."""
        if self.check_step(iteration):
            print(f'Calling {self.__class__.__name__} at iteration {iteration}')
            self.update(iteration, last_sample, sampler)
        
   

class AdjustStretchProposalScale(Update):
    """Rescale the first move's stretch parameter ``a`` towards a target acceptance.

    The first call only records the counters. Each later call computes the mean acceptance
    fraction ``af`` since the previous call from ``sampler.backend.accepted[:, 0]`` and
    multiplies ``sampler._moves[0].a`` (a :class:`~eryn.moves.StretchMove`-like move) by
    ``change``:

    .. code-block:: text

        af >  target:  f = min(supression_factor * af / target, max_change)
                       change = 1 + supression_factor * f
        af <= target:  f = min(supression_factor * target / af, max_change)
                       change = 1 - f

    Every call prints the counters, the acceptance, the change factor and the new ``a``.

    Args:
        target_acceptance (float, optional): Target acceptance fraction. Default ``0.22``.
        supression_factor (float, optional): Damping factor (spelling as in the code).
            Default ``0.1``.
        max_change (float, optional): Cap on the factor in the formulas above. Default
            ``0.5``.
        verbose (bool, optional): Stored; the printing happens regardless. Default
            ``False``.

    Note:
        Two mismatches with the current sampler: ``backend.accepted`` has shape
        ``(ntemps, nwalkers)``, so ``accepted[:, 0]`` is walker 0 at every temperature, not
        the cold chain the code comment describes; and
        :class:`~eryn.ensemble.EnsembleSampler` stores its moves as ``moves``, not
        ``_moves``, so every call (the final print reads ``sampler._moves[0].a``) raises
        ``AttributeError`` unless the sampler object provides ``_moves``.
    """

    def __init__(
        self,
        target_acceptance=0.22,
        supression_factor=0.1,
        max_change=0.5,
        verbose=False,
    ):
        """Store the settings (arguments are documented on the class)."""
        self.target_acceptance = target_acceptance
        self.verbose = verbose
        self.max_change, self.supression_factor = max_change, supression_factor

        self.time = 0

    def __call__(self, iter, last_sample, sampler):
        """Update ``sampler._moves[0].a`` as described on the class.

        Args:
            iter (int): Step counter passed by the sampler (unused).
            last_sample (State): Last state of the sampler (unused).
            sampler (EnsembleSampler): The sampler; its backend counters are read and its
                first move is modified.
        """

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
