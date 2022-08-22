# -*- coding: utf-8 -*-

from ..state import BranchSupplimental
from . import Move
import numpy as np
import tqdm
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    import numpy as xp

__all__ = ["CombineMove"]


class CombineMove(Move):
    """Parent class for proposals or "moves"

    Args:
        temperature_control (:class:`tempering.TemperatureControl`, optional):
            This object controls the tempering. It is passes to the parent class
            to moves so that all proposals can share and use temperature settings.
            (default: ``None``)
        # TODO: update

    """

    def __init__(self, moves, verbose=False, *args, **kwargs):
        self.moves = moves
        self.verbose = verbose
        self.accepted = [None for _ in moves]
        Move.__init__(self, *args, **kwargs)

    @property
    def temperature_control(self):
        return self._temperature_control

    @temperature_control.setter
    def temperature_control(self, temperature_control):
        for i, move in enumerate(self.moves):
            if isinstance(move, tuple):
                move = move[0]
            move.temperature_control = temperature_control
        self._temperature_control = temperature_control

    def propose(self, model, state):
        # TODO: add probabilistic draw just like outside
        iterator = enumerate(self.moves)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)

        accepted_out = None
        for i, move in iterator:
            if isinstance(move, tuple):
                move = move[0]
            state, accepted = move.propose(model, state)

            if accepted_out is None:
                accepted_out = accepted.copy()
            else:
                accepted_out += accepted

            if self.accepted[i] is None:
                self.accepted[i] = accepted.astype(int)
            else:
                self.accepted[i] += accepted.astype(int)
        return state, accepted_out
