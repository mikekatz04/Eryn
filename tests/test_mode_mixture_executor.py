# tests/test_mode_mixture_executor.py
"""Executor integration test for ModeMixtureFlow (mode-mixture plan, Task 5).

Proves the trainer-executor round-trip ships ModeMixtureFlow's mixture state
end to end: a worker-side clone fits per-leaf buffers (clustering into mode
slots), ``latest_weights()`` hands back a self-contained snapshot carrying
``"mixture_state"`` alongside ``"net"``/``"data_transform"``, and a parent-side
ModeMixtureFlow installs that snapshot via ``set_weights`` and can then propose
with bare leaf contexts.

Uses :class:`~eryn.flows.InlineExecutor` — the serial, in-process
``TrainerExecutor`` implementation (see ``tests/test_flow_executors.py``,
whose ``test_inline_zuko_trains_and_serves_weights`` this test's
construct/submit/latest_weights sequence mirrors) — rather than
:class:`~eryn.flows.ProcessExecutor`: it proves the same snapshot contract
without spawning a child process, and it trains synchronously inside
``submit()`` so there is no polling/wait loop to mirror.  It holds no OS
resources (no subprocess, no thread), so — matching every InlineExecutor test
in ``test_flow_executors.py`` — no ``shutdown()``/teardown call is needed.
"""
import numpy as np

from eryn.flows import ModeMixtureFlow, WhiteningTransform, InlineExecutor

PER = {2: (0.0, 2 * np.pi)}


def _flow(seed):
    return ModeMixtureFlow(
        dims=3, nleaves_max=1, kmax=4, cluster_seed=0, periodic=PER,
        flow_class="NSF", device="cpu",
        data_transform=WhiteningTransform(ndim=3, periodic=PER, shared=False),
        seed=seed, transforms=2, hidden_features=(16, 16), bins=4,
    )


def test_executor_ships_mixture_state():
    rng = np.random.default_rng(0)
    a = np.column_stack([rng.normal(-4, 0.1, 400), rng.normal(0, 0.1, 400),
                         rng.vonmises(0.5, 20, 400) % (2 * np.pi)])
    b = np.column_stack([rng.normal(4, 0.1, 400), rng.normal(2, 0.1, 400),
                         rng.vonmises(3.0, 20, 400) % (2 * np.pi)])
    buffer = {0: np.concatenate([a, b])}

    # Construction/submit mirrors test_inline_zuko_trains_and_serves_weights in
    # tests/test_flow_executors.py: build the InlineExecutor around the
    # template flow, submit once (InlineExecutor trains synchronously inside
    # submit() — no wait/poll loop needed), then read latest_weights().
    # NOTE: fit_kwargs deliberately omits `verbose` (present in this brief's
    # pseudocode and in test_mode_mixture_flow.py's FIT_KW): InlineExecutor's
    # __init__ rejects `verbose`/`refit_data_transform` in fit_kwargs with a
    # ValueError (executors.py) because it controls both itself.
    executor = InlineExecutor(
        _flow(1),
        fit_kwargs=dict(n_epochs=3, batch_size=256, validation_fraction=0.2,
                        val_split="temporal"),
        min_train_samples=100,
    )
    accepted = executor.submit(buffer)
    assert accepted is True

    version, snap = executor.latest_weights()
    assert version >= 1 and "mixture_state" in snap

    parent = _flow(2)
    parent.set_weights(snap)
    assert len(parent.mode_state[0].slots) == 2
    x, lq = parent.sample_and_log_prob(32, context=0)
    assert np.isfinite(lq).all() and (x[:, 0] < 0).any() and (x[:, 0] > 0).any()
