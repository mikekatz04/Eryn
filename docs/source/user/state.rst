State
------------------------

The :class:`eryn.state.State` carries the information around for all walkers at a given step. The individual branches are carried in :class:`eryn.state.Branch` class objects. A utility provided for state objects is the :class:`eryn.state.BranchSupplimental` object. This object can carry information around the sampler while being indexed and moved around just like other sampler information. See the tutorial for more information.

.. autoclass:: eryn.state.State
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.state.Branch
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.state.BranchSupplimental
    :members:
    :show-inheritance:
    :inherited-members:
