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
