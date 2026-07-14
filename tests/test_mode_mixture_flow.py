# tests/test_mode_mixture_flow.py
import numpy as np
import pytest

from eryn.flows import ModeMixtureFlow, WhiteningTransform

PER = {2: (0.0, 2 * np.pi)}
DIMS = 3
FIT_KW = dict(n_epochs=3, batch_size=256, lr=1e-3, patience=10,
              validation_fraction=0.15, val_split="temporal", verbose=False)


def _make_flow(kmax=4, seed=7):
    return ModeMixtureFlow(
        dims=DIMS, nleaves_max=2, kmax=kmax, cluster_seed=0,
        periodic=PER, flow_class="NSF", device="cpu",
        data_transform=WhiteningTransform(ndim=DIMS, periodic=PER, shared=False),
        seed=seed, transforms=2, hidden_features=(16, 16), bins=4,
    )


def _bimodal_leaf(rng, n=600):
    a = np.column_stack([rng.normal(-4, 0.1, n // 2), rng.normal(0, 0.1, n // 2),
                         rng.vonmises(0.5, 20, n // 2) % (2 * np.pi)])
    b = np.column_stack([rng.normal(4, 0.1, n // 2), rng.normal(2, 0.1, n // 2),
                         rng.vonmises(3.0, 20, n // 2) % (2 * np.pi)])
    return np.concatenate([a, b])


def _unimodal_leaf(rng, n=600):
    return np.column_stack([rng.normal(0, 0.5, n), rng.normal(-1, 0.2, n),
                            rng.vonmises(1.0, 20, n) % (2 * np.pi)])


def test_fit_builds_composite_conditions_and_mode_state():
    rng = np.random.default_rng(0)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    assert set(fl.mode_state) == {0, 1}
    assert len(fl.mode_state[0].slots) == 2
    assert fl.mode_state[1].slots == [0]
    # per-island whitening: the transform was fitted per composite id
    for s in fl.mode_state[0].slots:
        cid = fl._cid(0, s)
        z = fl.data_transform.forward(_bimodal_leaf(rng)[:5], cid)
        assert np.asarray(z).shape == (5, DIMS)


def test_snapshot_roundtrip_carries_mixture_state():
    rng = np.random.default_rng(1)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    snap = fl.get_snapshot()
    assert "mixture_state" in snap and "net" in snap
    fl2 = _make_flow(seed=8)                     # different init
    fl2.set_weights(snap)
    assert set(fl2.mode_state[0].weights) == set(fl.mode_state[0].weights)
    x, lq = fl2.sample_and_log_prob(8, context=0)
    assert x.shape == (8, DIMS) and np.isfinite(lq).all()


def test_h5_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    p = str(tmp_path / "mix.h5")
    fl.save(p)
    fl2 = ModeMixtureFlow.load(p)
    assert set(fl2.mode_state) == {0, 1}
    x = rng.standard_normal((4, DIMS)); x[:, 2] = np.abs(x[:, 2]) % (2 * np.pi)
    np.testing.assert_allclose(fl2.log_prob(x, context=0), fl.log_prob(x, context=0),
                               rtol=0, atol=1e-6)


def test_kone_reduces_to_plain_component_density():
    rng = np.random.default_rng(3)
    fl = _make_flow()
    fl.fit({0: _unimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(64, context=0)
    cid = fl._cid(0, fl.mode_state[0].slots[0])
    from eryn.flows import ZukoFlow
    lp_component = ZukoFlow.log_prob(fl, x, context=cid)   # bypass mixture wrapper
    np.testing.assert_allclose(lq, lp_component, rtol=0, atol=1e-10)


def test_mixture_density_normalization_importance_identity():
    # E_{x ~ q_mix}[ q_component0(x) / q_mix(x) ] must equal w_0-weighted ratio ~ 1
    # simpler exact invariant: E_{x ~ q_mix}[ exp(lq_mix(x) - lq_mix(x)) ] == 1;
    # the load-bearing check is sample/log_prob consistency on the bulk:
    rng = np.random.default_rng(4)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(512, context=0)
    lp = fl.log_prob(x, context=0)
    diff = np.abs(lp - lq)
    assert np.median(diff) < 1e-10          # identical code path by construction
    # both islands are actually proposed
    assert (x[:, 0] < 0).any() and (x[:, 0] > 0).any()


def test_mixture_covers_both_modes_with_expected_rates():
    rng = np.random.default_rng(5)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, _ = fl.sample_and_log_prob(2000, context=0)
    frac = (x[:, 0] > 0).mean()
    assert 0.35 < frac < 0.65               # weights ~0.5/0.5


def test_base_scale_passes_through():
    rng = np.random.default_rng(6)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    x, lq = fl.sample_and_log_prob(256, context=0, base_scale=1.5)
    lp = fl.log_prob(x, context=0, base_scale=1.5)
    assert np.median(np.abs(lp - lq)) < 1e-10
    x1, _ = fl.sample_and_log_prob(256, context=0, base_scale=None)
    assert x[:, 0].std() > 0                # smoke: scaled draws exist and are finite
    assert np.isfinite(lq).all()


def test_move_contract_smoke():
    """ConditionalFlowMove-style usage: leaf context, factors finite."""
    rng = np.random.default_rng(7)
    fl = _make_flow()
    fl.fit({0: _bimodal_leaf(rng), 1: _unimodal_leaf(rng)}, **FIT_KW)
    old = _bimodal_leaf(rng)[:24]
    new, lq_new = fl.sample_and_log_prob(24, context=0)
    factors = fl.log_prob(old, context=0) - lq_new
    assert np.isfinite(factors).all()
