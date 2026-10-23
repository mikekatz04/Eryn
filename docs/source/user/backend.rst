Backends
------------------------

Backends are used to store sampler information as it runs. They hold the MCMC chain
(sampler state per stored step), not a compute backend: choosing numpy or cupy is a
separate concept.

.. automodule:: eryn.backends

* Stored per step: the coordinates and ``inds`` of every branch, ``log_like``,
  ``log_prior``, ``betas`` (only when the state carries them, i.e. when the sampler has a
  temperature control; with the default ``tempering_kwargs={}`` they are not written), blobs
  and the running-sampler mask. Coordinates of inactive leaves are stored as
  ``store_missing_leaves`` (``np.nan`` by default).
* Kept as running totals: the in-model, reversible-jump and swap acceptance counts
  (``accepted``, ``rj_accepted``, ``swaps_accepted``); with ``track_moves=True`` the
  latest acceptance fraction of every move.
* The getters (:meth:`~eryn.backends.Backend.get_chain`,
  :meth:`~eryn.backends.Backend.get_log_like`, ...) share the selection keywords of
  :meth:`~eryn.backends.Backend.get_value` (``discard``, ``thin``, ``temp_index``,
  ``sampler_index``, ``slice_vals``, ``branch_names``).
* A sampler built with an already-initialized backend checks it against its configuration
  and resumes from the last stored sample (see the Note on
  :class:`~eryn.ensemble.EnsembleSampler`).

Regular Backend
~~~~~~~~~~~~~~~~

.. autoclass:: eryn.backends.Backend
    :members:
    :show-inheritance:
    :inherited-members:


HDF Backend
~~~~~~~~~~~~~~~~

.. autoclass:: eryn.backends.HDFBackend
    :members:
    :show-inheritance:
    :inherited-members:

Test Helpers
~~~~~~~~~~~~

.. autoclass:: eryn.backends.TempHDFBackend
    :members:
    :show-inheritance:

.. autofunction:: eryn.backends.get_test_backends

Deprecated: ParaBackend
~~~~~~~~~~~~~~~~~~~~~~~

In-memory storage for the deprecated :class:`~eryn.paraensemble.ParaEnsembleSampler`.
Use :class:`~eryn.backends.Backend` with ``reset(..., nsamplers=...)`` instead.

.. autoclass:: eryn.backends.parabackend.ParaBackend
    :members:
    :show-inheritance:
