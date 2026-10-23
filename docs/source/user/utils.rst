Utilities
------------------------

Utility functions and classes.

.. automodule:: eryn.utils

Periodic Container
~~~~~~~~~~~~~~~~~~~~

The sampler builds one :class:`~eryn.utils.PeriodicContainer` from its ``periodic``
argument (``{branch_name: {parameter: period}}``) and hands it to every move that has no
``periodic`` of its own.
:meth:`~eryn.utils.PeriodicContainer.wrap` maps values into ``[0, period)`` in place and
:meth:`~eryn.utils.PeriodicContainer.distance` gives the shortest signed separation.
Parameter names need ``key_order``; a plain ``periodic`` dict given to the sampler is
wrapped without it, so it must use column indices (pass a ready
:class:`~eryn.utils.PeriodicContainer` to use names).

.. autoclass:: eryn.utils.PeriodicContainer
    :members:
    :show-inheritance:
    :inherited-members:


TransformContainer
~~~~~~~~~~~~~~~~~~~

:class:`~eryn.utils.TransformContainer` maps coordinates from the sampling (input) basis to
an output basis: :meth:`~eryn.utils.TransformContainer.both_transforms` first fills the
fixed ``fill_dict`` values and places each input column at its output position, then
applies the parameter transforms (keyed by output-basis names). Inputs are reordered into
the output basis only when ``fill_dict`` is given; pass ``fill_dict={}`` when the two bases
differ in order but nothing is fixed. Inverse transforms are optional.

.. autoclass:: eryn.utils.TransformContainer
    :members:
    :show-inheritance:
    :inherited-members:

PlotContainer
~~~~~~~~~~~~~~~~~~~

Runtime diagnostic plots: with ``plot_iterations > 0``,
:meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` calls
:meth:`~eryn.utils.PlotContainer.produce_plots` every ``plot_iterations`` yielded states. If
no ``plot_generator`` is given, the sampler builds a container with
``plots=["base", "rj"]``, ``discard=0.2`` and output folder ``plot_folder`` (default
``"./runtime_plots"``). Runtime plotting is not supported with ``nsamplers > 1``.

.. autoclass:: eryn.utils.PlotContainer
    :members:
    :show-inheritance:
    :inherited-members:

Plotting Functions
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: eryn.utils.plot
    :members: produce_base_plots, produce_tempering_plots, produce_advanced_plots, produce_rj_plots, setup_plotting, save_or_show, cov_ellipse, overlay_fim_covariance, cornerplot, traceplot, plot_loglikelihood, tempering_ridgeplot, plot_swap_acceptance, plot_logl_betas, plot_betas_evolution, plot_leaves, plot_leaves_evolution, plot_acceptance_fraction, plot_tempered_acceptance_fraction, plot_act_evolution

Update functions
~~~~~~~~~~~~~~~~~~~

An ``update_fn`` is called as ``update_fn(i, state, sampler)`` inside
:meth:`~eryn.ensemble.EnsembleSampler.sample` every ``update_iterations`` proposal
iterations (independent of ``thin_by``, and also during burn-in).

Update Base Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.Update
    :members:
    :show-inheritance:
    :inherited-members:

Composite Update Base Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.CompositeUpdate
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.utils.UpdateStep
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Update Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As written, :class:`~eryn.utils.AdjustStretchProposalScale` raises ``AttributeError`` on
every call with an :class:`~eryn.ensemble.EnsembleSampler`, because it reads
``sampler._moves`` (the sampler attribute is ``moves``); see the Note in its docstring.

.. autoclass:: eryn.utils.AdjustStretchProposalScale
    :members:
    :show-inheritance:
    :inherited-members:

Stopping functions
~~~~~~~~~~~~~~~~~~~

A ``stopping_fn`` is called as ``stopping_fn(i, state, sampler)`` by
:meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` every ``stopping_iterations`` yielded
states (``thin_by * stopping_iterations`` proposal iterations); a truthy return ends the
run. It is not called during burn-in or by :meth:`~eryn.ensemble.EnsembleSampler.sample`.

Stopping Base Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.Stopping
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Stopping Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.SearchConvergeStopping
    :members:
    :show-inheritance:
    :inherited-members:


Sampler Model Container
~~~~~~~~~~~~~~~~~~~~~~~~~

The sampler model container (``eryn.model.Model``) is a named tuple that carries the
sampler's callables into every proposal. After initializing the
:class:`eryn.ensemble.EnsembleSampler` object, it can be obtained with the
:meth:`eryn.ensemble.EnsembleSampler.get_model` method; each field is an attribute, e.g.
``model.compute_log_like_fn``.

.. automodule:: eryn.model

.. autodata:: eryn.model.Model
    :no-value:

Progress Bar
~~~~~~~~~~~~

.. autofunction:: eryn.pbar.get_progress_bar

Other Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Autocorrelation, evidence and convergence helpers. Note that
:func:`~eryn.utils.utility.psrf` returns only ``Rhat`` (shape ``(ndims,)``).

.. automodule:: eryn.utils.utility
    :members:

