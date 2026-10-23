Ensemble
------------------------

:class:`eryn.ensemble.EnsembleSampler` is the main driver and container for MCMC runs. It
parses the branches (models), priors, in-model and reversible-jump moves, the tempering
setup and the storage backend, then advances the ensemble with
:meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` (or the generator
:meth:`~eryn.ensemble.EnsembleSampler.sample`).

How one run proceeds:

* One sampler iteration (a "proposal iteration") runs ``num_repeats_in_model`` in-model
  proposals, each with a move drawn at random from ``moves`` (with ``weights``), and then,
  when reversible jump is on, ``num_repeats_rj`` proposals drawn from ``rj_moves``. The
  default in-model move is ``StretchMove(a=2.0)``. ``rj_moves=True`` (or one of the strings
  ``"together"``, ``"iterate_branches"``, ``"separate_branches"``) builds
  :class:`~eryn.moves.DistributionGenerateRJ` moves that draw births from the priors; the
  default ``rj_moves=None`` means no reversible jump. Any ``rj_moves`` other than ``None``
  requires ``moves`` to be given as well.
* The stock tempered moves swap neighbouring temperature rungs at the end of their own
  ``propose`` (see :doc:`temper` and :doc:`moves`); reversible-jump moves swap without
  adapting the ladder.
* Every ``thin_by`` proposal iterations the state is saved to the backend (``store=True``)
  and yielded, so ``run_mcmc(state, nsteps, thin_by=k)`` makes ``nsteps * k`` proposal
  iterations and stores ``nsteps`` states. ``burn`` proposal iterations run first, with
  ``store=False`` and ``thin_by=1``.
* ``update_fn`` runs inside :meth:`~eryn.ensemble.EnsembleSampler.sample` every
  ``update_iterations`` proposal iterations (also during burn-in). ``stopping_fn`` and the
  runtime plots run in :meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` every
  ``stopping_iterations`` / ``plot_iterations`` yielded states.
* :meth:`~eryn.ensemble.EnsembleSampler.get_chain` returns ``{branch_name: array}`` with
  arrays of shape ``(nsteps, ntemps, nwalkers, nleaves_max, ndim)``. With ``nsamplers > 1``
  independent samplers a sampler axis follows the step axis.

Known limitations of the current code (each is described in the docstrings below):

* ``dr_moves`` (delayed rejection) builds its wrapper, but the delayed-rejection branch of
  :meth:`~eryn.moves.ReversibleJumpMove.propose` raises ``NotImplementedError`` on the
  first reversible-jump proposal, so only ``None`` / ``False`` can run.
* Resuming an untempered run (the default ``tempering_kwargs={}``, so
  ``temperature_control`` is ``None``) from an initialized backend
  (``run_mcmc(None, ...)``, or passing ``get_last_sample()``) raises ``AttributeError``.
  A run STARTED with ``tempering_kwargs=dict(ntemps=1)`` has a one-rung
  :class:`~eryn.moves.tempering.TemperatureControl` (``betas = [1.0]``), stores ``betas``
  and resumes. Do not resume an existing untempered file that way: its ``betas`` rows were
  never written (zeros or uninitialized memory) and would be loaded into the ladder.
* ``sample(skip_initial_state_check=False)`` raises ``AttributeError``.
* A likelihood may return at most one blob column.
* The sampler getters (``get_chain``, ``get_log_prior``, ...) copy their docstrings from
  :class:`~eryn.backends.Backend` at import. ``EnsembleSampler.get_log_like`` receives the
  ``get_log_prior`` text by mistake; see :meth:`eryn.backends.Backend.get_log_like` for the
  correct description (it returns the log-likelihood).

.. automodule:: eryn.ensemble

Ensemble Sampler
~~~~~~~~~~~~~~~~

.. autoclass:: eryn.ensemble.EnsembleSampler
    :members:
    :show-inheritance:
    :inherited-members:

Helper Functions
~~~~~~~~~~~~~~~~

.. autofunction:: eryn.ensemble.walkers_independent

Deprecated: ParaEnsembleSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~eryn.paraensemble.ParaEnsembleSampler` runs ``ngroups`` independent single-branch
ensembles with one batched likelihood call per proposal. It is deprecated (it warns on
construction) and superseded by ``EnsembleSampler(nsamplers=...)``. It stores into
:class:`~eryn.backends.parabackend.ParaBackend` and works on
:class:`~eryn.state.ParaState` objects (see :doc:`backend` and :doc:`state`). It takes its
array module (``numpy`` or ``cupy``) from ``gpubackendtools``.

.. autoclass:: eryn.paraensemble.ParaEnsembleSampler
    :members:
    :show-inheritance:

.. autofunction:: eryn.paraensemble.shuffle_along_axis
