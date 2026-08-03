"""Tests for walking nested move trees and reporting per-move acceptance."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from eryn.moves import CombineMove, Move
from eryn.utils.plot import PlotContainer, move_acceptance_rates, move_counters
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


def test_walk_moves_reports_a_repeated_sibling_once():
    shared = LeafA()
    paths = [path for path, _ in walk_moves([shared, shared])]
    assert paths == ["LeafA"]


def test_walk_moves_tolerates_a_move_without_the_hook():
    class NotAMove:
        pass

    assert [path for path, _ in walk_moves([NotAMove()])] == ["NotAMove"]


# cumulative counters over 4 plot calls, shape (nsteps, ntemps=1, nwalkers=2).
# Between call 1 and 2 the move was never drawn: num_proposals does not move.
COUNTS = np.array([[[1.0, 2.0]], [[3.0, 5.0]], [[3.0, 5.0]], [[7.0, 9.0]]])
NUM_PROPOSALS = np.array([10.0, 20.0, 20.0, 40.0])


def test_cumulative_rate_is_the_plain_ratio():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="cumulative")
    expected = np.array([[[0.1, 0.2]], [[0.15, 0.25]], [[0.15, 0.25]], [[0.175, 0.225]]])
    np.testing.assert_allclose(rates, expected)


def test_interval_rate_differences_the_counters():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
    # first row is measured from the start of the run; rows 1 and 3 are the
    # increments (2/10, 3/10) and (4/20, 4/20)
    np.testing.assert_allclose(rates[0], [[0.1, 0.2]])
    np.testing.assert_allclose(rates[1], [[0.2, 0.3]])
    np.testing.assert_allclose(rates[3], [[0.2, 0.2]])


def test_interval_rate_is_nan_when_the_move_was_not_drawn():
    rates = move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
    # NaN, not 0.0 -- the line must break rather than read as a real collapse
    assert np.all(np.isnan(rates[2]))


def test_no_divide_by_zero_warning_is_emitted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="interval")
        move_acceptance_rates(COUNTS, np.zeros(4), mode="cumulative")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="interval"):
        move_acceptance_rates(COUNTS, NUM_PROPOSALS, mode="sideways")


def _leaf_with_counters(accepted_value, num_proposals):
    move = LeafA()
    move.accepted = np.full((2, 3), float(accepted_value))
    move.num_proposals = num_proposals
    return move


def test_move_counters_reads_a_leafs_own_counters():
    move = _leaf_with_counters(4.0, 10)
    accepted, num_proposals = move_counters(move)
    np.testing.assert_allclose(accepted, np.full((2, 3), 4.0))
    assert num_proposals == 10.0


def test_move_counters_returns_none_when_uninitialised():
    assert move_counters(LeafA()) is None


def test_move_counters_pools_a_combine_move_from_its_children():
    # CombineMove.accepted raises AttributeError (its setter never stores
    # _accepted) and its num_proposals is never incremented, so it must be
    # pooled rather than read directly.
    combine = CombineMove([_leaf_with_counters(4.0, 10), _leaf_with_counters(6.0, 30)])
    accepted, num_proposals = move_counters(combine)
    np.testing.assert_allclose(accepted, np.full((2, 3), 10.0))
    assert num_proposals == 40.0


def test_move_counters_pooling_weights_by_proposal_count():
    # 4/10 and 6/30 pool to 10/40 = 0.25, not the mean of 0.4 and 0.2 = 0.3
    combine = CombineMove([_leaf_with_counters(4.0, 10), _leaf_with_counters(6.0, 30)])
    accepted, num_proposals = move_counters(combine)
    assert accepted[0, 0] / num_proposals == pytest.approx(0.25)


class _FakeBackend:
    def __init__(self, iteration):
        self.iteration = iteration


def test_collect_move_acceptance_records_every_node(tmp_path):
    leaf = _leaf_with_counters(2.0, 10)
    combine = CombineMove([leaf])
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))

    container._collect_move_acceptance([combine])

    assert set(container.move_accepted) == {"CombineMove", "CombineMove/LeafA"}
    assert container.move_steps["CombineMove/LeafA"] == [100]
    assert container.move_num_proposals["CombineMove/LeafA"] == [10.0]


def test_collect_move_acceptance_keeps_per_path_steps(tmp_path):
    ready = _leaf_with_counters(2.0, 10)
    late = LeafA()
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))

    container._collect_move_acceptance([ready, late])
    # the second node's counters only become valid on the next call
    late.accepted = np.full((2, 3), 1.0)
    late.num_proposals = 5
    container.backend = _FakeBackend(200)
    container._collect_move_acceptance([ready, late])

    # the late starter must not be plotted against the earlier step
    assert container.move_steps["LeafA_0"] == [100, 200]
    assert container.move_steps["LeafA_1"] == [200]


def test_move_acceptance_fractions_property_is_cumulative(tmp_path):
    leaf = _leaf_with_counters(2.0, 10)
    container = PlotContainer(backend=_FakeBackend(100), parent_folder=str(tmp_path))
    container._collect_move_acceptance([leaf])

    fractions = container.move_acceptance_fractions
    assert fractions["LeafA"].shape == (1, 2, 3)
    np.testing.assert_allclose(fractions["LeafA"][0], np.full((2, 3), 0.2))


def test_move_counters_counts_a_shared_instance_once():
    shared = _leaf_with_counters(4.0, 10)
    outer = CombineMove([shared, CombineMove([shared])])
    accepted, num_proposals = move_counters(outer)
    np.testing.assert_allclose(accepted, np.full((2, 3), 4.0))
    assert num_proposals == 10.0


def test_move_counters_returns_none_when_no_child_has_counters():
    combine = CombineMove([LeafA(), LeafA()])
    assert move_counters(combine) is None
