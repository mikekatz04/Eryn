Utilities
------------------------

Utility functions and classes.

Periodic Container
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.utils.PeriodicContainer
    :members:
    :show-inheritance:
    :inherited-members:

PlotContainer
~~~~~~~~~~~~~~~

.. autoclass:: eryn.utils.PlotContainer
    :members:
    :show-inheritance:
    :inherited-members:

TransformContainer
~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.utils.TransformContainer
    :members:
    :show-inheritance:
    :inherited-members:

Update functions
~~~~~~~~~~~~~~~~~~~

Update Base Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.Update
    :members:
    :show-inheritance:
    :inherited-members:

Implemented Update Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: eryn.utils.AdjustStretchProposalScale
    :members:
    :show-inheritance:
    :inherited-members:

Stopping functions
~~~~~~~~~~~~~~~~~~~

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

The sampler model container (``eryn.model.Model``) is a named tuple that carries around some of the most important objects in the sampler. These are then passed into proposals for usage. The model container has keys: ``["log_like_fn", "compute_log_like_fn", "compute_log_prior_fn", "temperature_control", "map_fn", "random"]``. These correspond, respectively, to the log Likelihood function in the form of the function wrapper with ``ensemble.py``; the log Likelihood function from the sampler; the log prior function from the sampler; the temperature controller; the map function where ``pool`` objects can be found; and the random generator. After initializing the :class:`eryn.ensemble.EnsembleSampler` object, the model container tuple can be accessed with the :func:`eryn.ensemble.EnsembleSampler.get_model` method. If you store this in a variable ``model``, you can access each member as an attribute, e.g. ``model.compute_log_like_fn``. 

Other Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: eryn.utils.utility
    :members:

