# -*- coding: utf-8 -*-
"""Run several moves in sequence as one proposal slot (:class:`CombineMove`)."""

import numpy as np
import tqdm

from ..state import BranchSupplemental
from .move import Move

__all__ = ["CombineMove"]


class CombineMove(Move):
    """Move that runs a fixed list of moves in order, as one proposal.

    :meth:`propose` calls each sub-move's ``propose`` in list order, threading the state
    through. Each sub-move does its own accept/reject and its own temperature swaps (if it
    has a controller), so the combined move does no extra swapping. Acceptance bookkeeping
    lives on the sub-moves (see :attr:`acceptance_fraction`).

    Args:
        moves (list): Moves to run, in order. :meth:`propose` and the
            ``temperature_control`` / ``periodic`` setters unwrap ``(move, weight)`` tuples
            (weights ignored), but the :attr:`accepted` setter and the acceptance-fraction
            properties do not, so the sampler's counter initialisation fails on tuples;
            pass bare moves.
        *args: Forwarded to :class:`~eryn.moves.Move`.
        verbose (bool, optional): Wrap the loop over sub-moves in a ``tqdm`` progress bar
            (useful when debugging). Default ``False``.
        share_temperature_control (bool, optional): If ``True``, setting a non-``None``
            ``temperature_control`` on the combined move (including through the
            constructor or the sampler) also sets it on every sub-move. Default ``True``.
        **kwargs: Forwarded to :class:`~eryn.moves.Move`. Its ``periodic`` is always
            propagated to the sub-moves, including the ``None`` default at construction
            (which replaces any ``periodic`` already set on a sub-move until the sampler
            fills in its own).

    Attributes:
        moves (list): The sub-moves (or ``(move, weight)`` tuples) in run order.
        verbose (bool): See ``verbose``.
        share_temperature_control (bool): See ``share_temperature_control``.
    """

    def __init__(self, moves, *args, verbose=False, share_temperature_control=True, **kwargs):
        # store moves
        self.moves = moves
        self.verbose = verbose
        self.share_temperature_control = share_temperature_control
        Move.__init__(self, *args, **kwargs)

    @property
    def accepted(self):
        """list[np.ndarray]: Accepted counts of each sub-move, in order.

        Setting it copies the given ``np.ndarray`` onto every sub-move (this is how
        :class:`~eryn.ensemble.EnsembleSampler` initialises the counters). Reading it checks
        ``self._accepted``, which this class never sets, so the getter raises
        ``AttributeError`` unless a subclass sets ``_accepted``. LISA Analysis Tools' stage
        builder relies on that: it tests ``hasattr(combined, "accepted")`` and then sets
        the counters, so making the getter raise anything else breaks that check.
        """
        if self._accepted is None:
            raise ValueError(
                "accepted must be inititalized with the init_accepted function if you want to use it."
            )
        # this retrieves the accepted arrays from the individual moves
        accepted_out = [move.accepted for move in self.moves]
        return accepted_out

    @accepted.setter
    def accepted(self, accepted):
        # set the accepted arrays for all moves
        assert isinstance(accepted, np.ndarray)
        for move in self.moves:
            move.accepted = accepted.copy()

    @property
    def acceptance_fraction(self):
        """np.ndarray: Sub-move acceptance fractions averaged over the sub-moves,
        shape ``(ntemps, nwalkers)``."""
        acceptance_fraction_out = np.mean(
            [move.acceptance_fraction for move in self.moves], axis=0
        )
        return acceptance_fraction_out

    @property
    def acceptance_fraction_separate(self):
        """list[np.ndarray]: Acceptance fraction of each sub-move, in order."""
        acceptance_fraction_out = [move.acceptance_fraction for move in self.moves]
        return acceptance_fraction_out

    @property
    def temperature_control(self):
        """:class:`~eryn.moves.tempering.TemperatureControl` or None: Temperature controller.

        A non-``None`` value is also set on every sub-move when
        ``share_temperature_control`` is ``True``.
        """
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        # when setting the temperature control object
        # need to apply it to each move
        for i, move in enumerate(self.moves):
            # if weights were provided with moves, remove move class
            if isinstance(move, tuple):
                move = move[0]
            # set temperature control for each move
            if temperature_control is not None and self.share_temperature_control:
                move.temperature_control = temperature_control

        # main temperature control here for reference
        self._temperature_control = temperature_control

    @property
    def periodic(self):
        """:class:`~eryn.utils.PeriodicContainer` or None: Periodic-parameter information;
        setting it also sets it on every sub-move."""
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        # when setting the periodic parameters
        # need to apply it to each move
        for i, move in enumerate(self.moves):
            # if weights were provided with moves, remove move class
            if isinstance(move, tuple):
                move = move[0]
            move.periodic = periodic
        self._periodic = periodic

    def propose(self, model, state):
        """Run every sub-move's ``propose`` in order.

        Args:
            model (:class:`eryn.model.Model`): Sampler callables and ``random``.
            state (:class:`eryn.state.State`): Current state, passed through the sub-moves.

        Returns:
            tuple[:class:`eryn.state.State`, np.ndarray]: ``(state, accepted)``: the state
            after the last sub-move and the element-wise sum of the sub-moves' ``accepted``
            arrays, shape ``(ntemps, nwalkers)`` (an OR when they are boolean, since the
            first array's dtype is kept).
        """

        # prepare for verbosity if needed
        iterator = enumerate(self.moves)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)

        # we will set this inside the loop during the first iteration
        accepted_out = None
        for i, move in iterator:
            # get move out of tuple
            if isinstance(move, tuple):
                move = move[0]

            # run move
            state, accepted = move.propose(model, state)

            # set (first iteration) or add (after first iteration) to accepted_out
            if accepted_out is None:
                accepted_out = accepted.copy()
            else:
                accepted_out += accepted

        return state, accepted_out
