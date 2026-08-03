"""Tests for walking nested move trees and reporting per-move acceptance."""
from __future__ import annotations

import numpy as np
import pytest

from eryn.moves import CombineMove, Move
from eryn.utils.utility import walk_moves


class LeafA(Move):
    """Minimal concrete move; only the acceptance counters are exercised."""

    def propose(self, model, state):  # pragma: no cover - never called
        raise NotImplementedError


class LeafB(LeafA):
    """A second leaf class, so paths can be told apart by name."""


def test_sub_moves_defaults_to_empty():
    assert LeafA().sub_moves == []


def test_combine_move_sub_moves_unwraps_weight_tuples():
    a, b = LeafA(), LeafB()
    combine = CombineMove([(a, 0.5), (b, 0.5)])
    assert combine.sub_moves == [a, b]


def test_walk_moves_flat_uses_class_names():
    a, b = LeafA(), LeafB()
    assert [path for path, _ in walk_moves([a, b])] == ["LeafA", "LeafB"]


def test_walk_moves_disambiguates_repeated_class_at_same_level():
    a, b = LeafA(), LeafA()
    assert [path for path, _ in walk_moves([a, b])] == ["LeafA_0", "LeafA_1"]


def test_walk_moves_composes_nested_paths():
    leaf, inner_a, inner_b = LeafA(), LeafA(), LeafB()
    inner = CombineMove([inner_a, inner_b])
    outer = CombineMove([leaf, inner])

    assert [path for path, _ in walk_moves([outer])] == [
        "CombineMove",
        "CombineMove/LeafA",
        "CombineMove/CombineMove",
        "CombineMove/CombineMove/LeafA",
        "CombineMove/CombineMove/LeafB",
    ]


def test_walk_moves_yields_the_move_objects():
    leaf = LeafA()
    outer = CombineMove([leaf])
    found = dict(walk_moves([outer]))
    assert found["CombineMove/LeafA"] is leaf
    assert found["CombineMove"] is outer


def test_walk_moves_reports_a_shared_instance_once():
    shared = LeafA()
    outer = CombineMove([shared, CombineMove([shared])])
    paths = [path for path, _ in walk_moves([outer])]
    assert paths.count("CombineMove/LeafA") == 1
    assert "CombineMove/CombineMove/LeafA" not in paths


def test_walk_moves_tolerates_a_move_without_the_hook():
    class NotAMove:
        pass

    assert [path for path, _ in walk_moves([NotAMove()])] == ["NotAMove"]
