import numpy as np

from eryn.flows.modes import ModeState, embed, estimate_modes

PER = {1: (0.0, 2 * np.pi)}          # dim 1 periodic


def _three_islands(rng, n=900):
    """3 well-separated islands in (linear, angle) space; island 2 straddles the wrap."""
    c = [(-5.0, 1.0), (5.0, 3.0), (0.0, 6.2)]
    xs = []
    for i, (a, b) in enumerate(c):
        x = np.column_stack([
            a + 0.1 * rng.standard_normal(n // 3),
            (b + 0.05 * rng.standard_normal(n // 3)) % (2 * np.pi),
        ])
        xs.append(x)
    return np.concatenate(xs), np.repeat([0, 1, 2], n // 3)


def test_embed_shapes_and_wrap_continuity():
    x = np.array([[1.0, 0.01], [1.0, 2 * np.pi - 0.01]])
    e = embed(x, PER)
    assert e.shape == (2, 3)                     # 1 linear + cos + sin
    assert np.linalg.norm(e[0] - e[1]) < 0.1     # wrap-adjacent points embed close


def test_finds_three_islands_and_weights_sum_to_one():
    rng = np.random.default_rng(0)
    x, true = _three_islands(rng)
    st = estimate_modes(x, PER, kmax=8, seed=0)
    assert len(st.slots) == 3
    assert abs(sum(st.weights.values()) - 1.0) < 1e-12
    assert all(w >= 0.02 for w in st.weights.values())
    # labels must partition the data consistently with the true islands
    for t in range(3):
        lab = st.labels[true == t]
        assert (lab == lab[0]).mean() > 0.99


def test_unimodal_collapses_to_one_slot():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((600, 2)) * [1.0, 0.1] + [0.0, 3.0]
    st = estimate_modes(x, PER, kmax=8, seed=0)
    assert st.slots == [0] and st.weights[0] == 1.0
    assert (st.labels == 0).all()


def test_slot_ids_stable_across_rounds():
    rng = np.random.default_rng(2)
    x1, _ = _three_islands(rng)
    st1 = estimate_modes(x1, PER, kmax=8, seed=0)
    x2, _ = _three_islands(rng)          # fresh draw, same islands
    st2 = estimate_modes(x2, PER, kmax=8, prev=st1, seed=1)
    assert set(st2.slots) == set(st1.slots)
    for s in st1.slots:                  # matched slots point at the same island
        d = np.linalg.norm(st1.centers[s] - st2.centers[s])
        assert d < 1.0


def test_tiny_components_are_dissolved():
    rng = np.random.default_rng(3)
    x, _ = _three_islands(rng, n=900)
    x = np.concatenate([x, [[20.0, 1.0]] * 3])   # 3-row spur, below min_rows
    st = estimate_modes(x, PER, kmax=8, min_rows=25, seed=0)
    assert len(st.slots) == 3                    # spur absorbed, not a slot
