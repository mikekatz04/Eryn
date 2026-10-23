State
------------------------

The :class:`eryn.state.State` carries the information around for all walkers at a given step. The individual branches are carried in :class:`eryn.state.Branch` class objects. A utility provided for state objects is the :class:`eryn.state.BranchSupplemental` object. This object can carry information around the sampler while being indexed and moved around just like other sampler information. See the tutorial for more information.

Shapes inside a state: each branch holds ``coords`` of shape
``(ntemps, nwalkers, nleaves_max, ndim)`` and a boolean ``inds`` array of shape
``(ntemps, nwalkers, nleaves_max)`` that marks the active leaves; ``log_like`` and
``log_prior`` have shape ``(ntemps, nwalkers)`` and ``betas`` has shape ``(ntemps,)``. With
``nsamplers > 1`` independent samplers the sampler axis is folded into the leading axis
(size ``nsamplers * ntemps``). Coordinates of inactive leaves keep their last values in
the state; the backends store ``store_missing_leaves`` (``np.nan`` by default) for them
instead.

The stock moves update the state object yielded by
:meth:`~eryn.ensemble.EnsembleSampler.sample` in place on later iterations; copy it
(``State(state, copy=True)``) to keep a snapshot.

.. automodule:: eryn.state

.. autoclass:: eryn.state.State
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.state.Branch
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.state.BranchSupplemental
    :members:
    :show-inheritance:
    :inherited-members:

Deprecated: ParaState
~~~~~~~~~~~~~~~~~~~~~

State of the deprecated :class:`~eryn.paraensemble.ParaEnsembleSampler`. Use
:class:`~eryn.state.State` with ``nsamplers`` instead.

.. autoclass:: eryn.state.ParaState
    :members:
    :show-inheritance:
