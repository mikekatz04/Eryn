# -*- coding: utf-8 -*-
"""Proposal distributions ("moves") for :class:`~eryn.ensemble.EnsembleSampler`.

Base classes: :class:`Move` (shared bookkeeping and :meth:`Move.update`),
:class:`MHMove`, :class:`RedBlueMove`, :class:`GroupMove`, :class:`ReversibleJumpMove` and
the :class:`MultipleTryMove` mixin. Concrete moves: :class:`StretchMove` (the sampler
default), :class:`GroupStretchMove`, :class:`GaussianMove`, :class:`EigenAxisMove`,
:class:`DistributionGenerate`, :class:`DistributionGenerateRJ` (the default RJ move),
:class:`MTDistGenMove`, :class:`MTDistGenMoveRJ`, :class:`RidgeGibbsMove`,
:class:`NUTSMove` (with the standalone :class:`NUTSSampler`) and :class:`CombineMove`.
Tempering lives in :class:`TemperatureControl` (:mod:`eryn.moves.tempering`).
"""

# from .de import DEMove
# from .de_snooker import DESnookerMove
from .combine import CombineMove
from .distgen import DistributionGenerate
from .distgenrj import DistributionGenerateRJ
from .eigenaxis import EigenAxisMove
from .gaussian import GaussianMove
from .group import GroupMove
from .groupstretch import GroupStretchMove

# from .kde import KDEMove
from .mh import MHMove
from .move import Move

# from .basicmodelswaprj import BasicSymmetricModelSwapRJMove
from .mtdistgen import MTDistGenMove
from .mtdistgenrj import MTDistGenMoveRJ
from .multipletry import MultipleTryMove

# from .walk import WalkMove
from .nuts import NUTSMove, NUTSSampler
from .red_blue import RedBlueMove
from .ridgegibbs import RidgeGibbsMove
from .rj import ReversibleJumpMove
from .stretch import StretchMove
from .tempering import TemperatureControl

__all__ = [
    "Move",
    "MHMove",
    "EigenAxisMove",
    "GaussianMove",
    "RedBlueMove",
    "StretchMove",
    "DistributionGenerateRJ",
    "DistributionGenerate",
    "TemperatureControl",
    "ReversibleJumpMove",
    "MultipleTryMove",
    "GroupMove",
    "GroupStretchMove",
    "RidgeGibbsMove",
    "CombineMove",
    "NUTSMove",
    "NUTSSampler",
]
