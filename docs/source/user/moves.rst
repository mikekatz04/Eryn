Moves
------------------------

Proposal distributions for MCMC.

Every move subclasses :class:`eryn.moves.Move` and implements
``propose(model, state) -> (state, accepted)``. ``model`` is the
:class:`~eryn.model.Model` namedtuple of sampler callables (see :doc:`utils`), ``state`` the
current :class:`~eryn.state.State`, and ``accepted`` a ``(ntemps, nwalkers)`` array that
the sampler adds to its acceptance counter. Most moves follow the three-tier layout from
``emcee``: the :class:`~eryn.moves.Move` base (Gibbs splits, tempered posterior,
:meth:`~eryn.moves.Move.update`), a "guide" class that implements ``propose``
(:class:`~eryn.moves.MHMove`, :class:`~eryn.moves.RedBlueMove`,
:class:`~eryn.moves.GroupMove`, :class:`~eryn.moves.ReversibleJumpMove`), and a concrete
move that implements ``get_proposal``.

Tempering and moves: the stock MH, red-blue, group and NUTS moves call
:meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` at the end of ``propose``
when their ``temperature_control`` is set, so neighbouring rungs are swapped (and the
ladder adapted) after every proposal. :class:`~eryn.moves.ReversibleJumpMove` swaps
without adapting the ladder (and skips the swaps with ``prevent_swaps=True``).
:class:`~eryn.moves.CombineMove` does no swapping of its own; each sub-move swaps.
:class:`~eryn.moves.RidgeGibbsMove` makes no swaps. Untempered moves hold
``temperature_control = None``.

.. automodule:: eryn.moves

Move Parent Class
~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.moves.Move
    :members:
    :show-inheritance:
    :inherited-members:


Metropolis-Hastings Moves
~~~~~~~~~~~~~~~~~~~~~~~~~

MH Parent Class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.MHMove
    :members:
    :show-inheritance:
    :inherited-members:

Implemented MH Moves
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.GaussianMove
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.moves.EigenAxisMove
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.moves.DistributionGenerate
    :members:
    :show-inheritance:
    :inherited-members:

Eigen-Axis Helper Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The axis and width tables of :class:`~eryn.moves.EigenAxisMove` are built by the caller
with these functions (module :mod:`eryn.moves.eigenaxis`); they follow the input's array
module (``numpy`` or ``cupy``).

.. autofunction:: eryn.moves.eigenaxis.eigen_axis_set

.. autofunction:: eryn.moves.eigenaxis.prior_box_scales

.. autofunction:: eryn.moves.eigenaxis.axis_prior_bounds

.. autofunction:: eryn.moves.eigenaxis.project_out_direction

.. autofunction:: eryn.moves.eigenaxis.draw_axis_step

Red-Blue Moves
~~~~~~~~~~~~~~~~~~

Red-Blue Parent Class
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: eryn.moves.RedBlueMove
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Red-Blue Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.StretchMove
    :members:
    :show-inheritance:
    :inherited-members:

Group Moves
~~~~~~~~~~~~~~

Group moves use a stationary set of "friends", rebuilt every ``n_iter_update`` calls, in
place of the live complementary ensemble. This keeps the stretch proposal usable with
reversible jump; :class:`~eryn.ensemble.EnsembleSampler` warns when a plain
:class:`~eryn.moves.StretchMove` is combined with reversible jump.

Group Parent Class
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: eryn.moves.GroupMove
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Group Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.GroupStretchMove
    :members:
    :show-inheritance:
    :inherited-members:


Reversible Jump Moves
~~~~~~~~~~~~~~~~~~~~~~~

Reversible Jump Parent Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: eryn.moves.ReversibleJumpMove
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Reversible Jump Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.DistributionGenerateRJ
    :members:
    :show-inheritance:
    :inherited-members:

Delayed Rejection (not functional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~eryn.moves.delayedrejection.DelayedRejection` is built by
:class:`~eryn.moves.ReversibleJumpMove` when ``dr`` is set (and by
``EnsembleSampler(dr_moves=...)``), but the delayed-rejection branch of
:meth:`~eryn.moves.ReversibleJumpMove.propose` raises ``NotImplementedError`` before it is
used. It is not exported from :mod:`eryn.moves`.

.. autoclass:: eryn.moves.delayedrejection.DelayedRejection
    :members:
    :show-inheritance:

Multiple Try Moves
~~~~~~~~~~~~~~~~~~~~~~~

Multiple Try Parent Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: eryn.moves.MultipleTryMove
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.moves.multipletry.MultipleTryMoveRJ
    :members:
    :show-inheritance:

Implemented Multiple Try Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.MTDistGenMove
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Multiple Try RJ Moves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.moves.MTDistGenMoveRJ
    :members:
    :show-inheritance:
    :inherited-members:

Multiple Try Helper Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: eryn.moves.multipletry.logsumexp

.. autofunction:: eryn.moves.multipletry.get_mt_computations


Gibbs Moves Along Likelihood Degeneracies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.moves.RidgeGibbsMove
    :members:
    :show-inheritance:
    :inherited-members:


No-U-Turn Sampler
~~~~~~~~~~~~~~~~~

:class:`~eryn.moves.NUTSSampler` is a standalone vectorized NUTS (Hoffman & Gelman 2014)
that works on user-supplied gradient and log-posterior callables;
:class:`~eryn.moves.NUTSMove` plugs one NUTS step into the regular move schedule.

.. autoclass:: eryn.moves.NUTSSampler
    :members:
    :show-inheritance:

.. autoclass:: eryn.moves.NUTSMove
    :members:
    :show-inheritance:
    :inherited-members:


Utility Moves
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.moves.CombineMove
    :members:
    :show-inheritance:
    :inherited-members:
