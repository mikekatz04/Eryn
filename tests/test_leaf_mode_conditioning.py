import numpy as np
import pytest

from eryn.flows.conditioning import ConditioningStrategy, LeafModeConditioning


def test_encode_concatenates_leaf_and_mode_onehots():
    cond = LeafModeConditioning(nleaves_max=6, kmax=8)
    assert cond.context_dim == 14
    ctx = cond.encode(3 * 8 + 5)  # leaf 3, slot 5
    assert ctx.shape == (14,) and ctx.dtype == np.float32
    assert ctx.sum() == 2.0
    assert ctx[3] == 1.0            # leaf one-hot block [0:6)
    assert ctx[6 + 5] == 1.0        # mode one-hot block [6:14)


def test_encode_rejects_out_of_range():
    cond = LeafModeConditioning(nleaves_max=2, kmax=4)
    with pytest.raises(ValueError):
        cond.encode(2 * 4)          # leaf 2 out of range
    with pytest.raises(ValueError):
        cond.encode(-1)


def test_protocol_and_assign():
    cond = LeafModeConditioning(nleaves_max=2, kmax=4)
    assert isinstance(cond, ConditioningStrategy)
    with pytest.raises(NotImplementedError):
        cond.assign(np.zeros(3))
