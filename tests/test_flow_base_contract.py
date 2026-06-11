# tests/test_flow_base_contract.py
"""Tests for the torch-free Flow ABC and FlowProposalDistribution.

NOTE: ZukoFlow contract tests are added in T3.
"""
from __future__ import annotations

import numpy as np
import pytest

from eryn.flows.base import Flow, FlowHistory, FlowProposalDistribution
from eryn.flows.transforms import IdentityTransform


# ---------------------------------------------------------------------------
# Minimal concrete Flow subclass for testing (no torch, standard normal)
# ---------------------------------------------------------------------------

class GaussianFlow(Flow):
    """Standard-normal flow for testing the ABC contract."""

    def __init__(self, dims: int, device=None, data_transform=None, conditioning=None):
        super().__init__(dims=dims, device=device, data_transform=data_transform,
                         conditioning=conditioning)

    def log_prob(self, x, context=None):
        x = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum(x ** 2, axis=-1) - 0.5 * x.shape[-1] * np.log(2 * np.pi)

    def sample(self, n: int, context=None):
        return np.random.randn(n, self.dims)

    def sample_and_log_prob(self, n: int, context=None):
        x = self.sample(n, context=context)
        lp = self.log_prob(x, context=context)
        return x, lp

    def fit(self, samples, **kwargs):
        return FlowHistory()

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

    def save(self, h5_file, path="flow"):
        pass

    @classmethod
    def load(cls, h5_file, path="flow"):
        return cls(dims=1)


# ---------------------------------------------------------------------------
# Flow ABC tests
# ---------------------------------------------------------------------------

def test_gaussian_flow_instantiates():
    f = GaussianFlow(dims=3)
    assert f.dims == 3


def test_flow_default_data_transform():
    """If no data_transform given, defaults to IdentityTransform."""
    f = GaussianFlow(dims=3)
    assert isinstance(f.data_transform, IdentityTransform)


def test_flow_custom_data_transform():
    tr = IdentityTransform()
    f = GaussianFlow(dims=3, data_transform=tr)
    assert f.data_transform is tr


def test_flow_device_stored():
    f = GaussianFlow(dims=3, device="cpu")
    assert f.device == "cpu"


def test_flow_conditioning_none():
    f = GaussianFlow(dims=3)
    assert f.conditioning is None


def test_config_dict_contains_dims():
    f = GaussianFlow(dims=5)
    cfg = f.config_dict()
    assert cfg["dims"] == 5


def test_config_dict_contains_defaults():
    """config_dict must include default values (device, data_transform, conditioning)."""
    f = GaussianFlow(dims=2)
    cfg = f.config_dict()
    assert "device" in cfg
    assert "data_transform" in cfg
    assert "conditioning" in cfg


def test_config_dict_excludes_self():
    f = GaussianFlow(dims=2)
    cfg = f.config_dict()
    assert "self" not in cfg


def test_init_args_excludes_self():
    f = GaussianFlow(dims=2)
    assert "self" not in f._init_args


def test_config_dict_round_trip():
    """config_dict() produces kwargs that reconstruct an equivalent instance."""
    f = GaussianFlow(dims=7)
    cfg = f.config_dict()
    f2 = GaussianFlow(**cfg)
    assert f2.dims == f.dims
    assert f2.device == f.device
    assert f2.conditioning == f.conditioning


def test_flow_cannot_instantiate_abc():
    """Flow ABC itself cannot be instantiated."""
    with pytest.raises(TypeError):
        Flow(dims=3)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FlowHistory tests
# ---------------------------------------------------------------------------

def test_flow_history_default():
    h = FlowHistory()
    assert h.training_loss == []
    assert h.validation_loss == []


def test_flow_history_fields():
    h = FlowHistory(training_loss=[1.0, 0.8], validation_loss=[1.1, 0.9])
    assert h.training_loss == [1.0, 0.8]
    assert h.validation_loss == [1.1, 0.9]


# ---------------------------------------------------------------------------
# FlowProposalDistribution tests
# ---------------------------------------------------------------------------

def test_fpd_logpdf_shape():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    x = np.random.randn(10, 3)
    lp = fpd.logpdf(x)
    assert lp.shape == (10,)


def test_fpd_rvs_int_size():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs(5)
    assert samples.shape == (5, 3)


def test_fpd_rvs_tuple_size():
    """rvs accepts a tuple with a single element (eryn DistributionGenerate compat)."""
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    samples = fpd.rvs((7,))
    assert samples.shape == (7, 3)


def test_fpd_logpdf_finite():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=0)
    x = np.zeros((5, 3))
    lp = fpd.logpdf(x)
    assert np.all(np.isfinite(lp))


def test_fpd_condition_stored():
    f = GaussianFlow(dims=3)
    fpd = FlowProposalDistribution(f, condition=2)
    assert fpd.condition == 2
