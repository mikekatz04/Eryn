TODOs and Known Issues
======================

Code Projects
--------------------

* Implement Product-Space MCMC for direct model comparison.
    * Use supplimental objects to carry log-Likelihood for each model.
    * Perform tempering within each model.
* Produce image for tree metaphor.

Code TODOs
-----------

* :class:`eryn.moves.StretchMove`: add log proposal option used in ptemcee with a comparison.
* :mod:`eryn.moves.group`: Combine with red-blue where the stationary distribution is split in two according to two groups of walkers. Will guarantee detailed balance always.

Known Issues
------------

Defects found while documenting the current code (a selection, not a complete list); most are also described in the docstrings of the objects named.

* Delayed rejection (``EnsembleSampler(dr_moves=...)``, ``ReversibleJumpMove(dr=...)``) is not implemented: :meth:`eryn.moves.ReversibleJumpMove.propose` raises ``NotImplementedError`` when it is enabled.
* :class:`eryn.ensemble.EnsembleSampler`: resuming an untempered run (default ``tempering_kwargs={}``) from an initialized backend raises ``AttributeError`` (a run started with ``tempering_kwargs=dict(ntemps=1)`` stores ``betas`` and resumes); ``sample(skip_initial_state_check=False)`` raises ``AttributeError``; a likelihood can return at most one blob column; ``EnsembleSampler.get_log_like`` shows the docstring of ``get_log_prior``; a bare ``ProbDistContainer`` prior is stored under ``"model_0"``, so a single branch with another name needs ``priors={name: container}``.
* :meth:`eryn.backends.Backend.get_info` raises ``ValueError`` for tempered, reversible-jump or ``nsamplers > 1`` runs (it calls ``get_autocorr_time`` unguarded).
* :meth:`eryn.backends.Backend.get_value` (and the getters): a single string ``branch_names`` is not wrapped in a list; pass a list.
* :meth:`eryn.backends.HDFBackend.save_step`: a lock error that reaches its retry handler raises ``NameError`` (the message formats an undefined name), so that outer retry loop never retries; :meth:`eryn.backends.HDFBackend.open` keeps its own retries.
* :class:`eryn.moves.ReversibleJumpMove`: the ``tune`` attribute shadows :meth:`eryn.moves.Move.tune`, so ``sample(tune=True)`` with reversible-jump moves raises ``TypeError``.
* :class:`eryn.utils.AdjustStretchProposalScale` raises ``AttributeError`` on every call with an :class:`eryn.ensemble.EnsembleSampler` (it reads ``sampler._moves``; the attribute is ``moves``).
* :class:`eryn.moves.GaussianMove`: the diagonal form (a 1D vector of more than one variance) fails at construction; pass ``np.diag(variances)``.
* :class:`eryn.moves.CombineMove`: reading ``accepted`` raises ``AttributeError`` (LISA Analysis Tools relies on that through ``hasattr``; read the docstring before changing it); pass bare moves, not ``(move, weight)`` tuples.
* :class:`eryn.moves.DistributionGenerate`: with more than one active leaf per walker only one leaf's proposal term is kept.
* :class:`eryn.backends.parabackend.ParaBackend`: ``get_gelman_rubin_convergence_diagnostic`` raises ``AttributeError``.
