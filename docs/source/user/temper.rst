Parallel Tempering
------------------------

The tempering is controlled from :class:`eryn.moves.tempering.TemperatureControl`.

.. automodule:: eryn.moves.tempering

* Pass ``tempering_kwargs`` to :class:`~eryn.ensemble.EnsembleSampler` (for example
  ``dict(ntemps=10)`` or ``dict(betas=betas)``); they are the keyword arguments of
  :class:`~eryn.moves.tempering.TemperatureControl`. Without ``betas`` the ladder comes from
  :func:`~eryn.moves.tempering.make_ladder` with the effective dimension
  ``sum(nleaves_max[b] * ndims[b])`` over branches. The default ``{}`` means no tempering
  (``ntemps = 1``, ``temperature_control = None``).
* The tempered log posterior of rung ``t`` is :math:`\beta_t \log L + \log p`.
* The stock moves with a temperature controller call
  :meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` at the end of their
  ``propose``: neighbouring rungs are proposed for swapping from the hottest pair down to
  the coldest pair, accepted with
  :math:`\log\alpha = (\beta_{i-1} - \beta_i)(\log L_i - \log L_{i-1})`, and the ladder is
  then adapted. Reversible-jump moves swap without adapting;
  :class:`~eryn.moves.RidgeGibbsMove` does not swap.
* Adaptation follows arXiv:1501.05823 (``adaptive``, ``adaptation_lag``,
  ``adaptation_time``); ``stop_adaptation >= 0`` stops it after that many adaptations
  (for example ``stop_adaptation = burn``).
* ``fancy_swap=True`` in :meth:`~eryn.moves.tempering.TemperatureControl.temperature_swaps`
  recomputes the likelihood of the swapped coordinates instead of exchanging the cached
  values; it is much more expensive and is used by moves that manage their own swaps.
* With ``nsamplers > 1`` each independent sampler has its own ladder; the ladders are folded
  into one flat ``betas`` array of length ``nsamplers * ntemps``, and swaps and adaptation
  stay within each sampler.

.. autoclass:: eryn.moves.tempering.TemperatureControl
    :members:
    :show-inheritance:
    :inherited-members:

.. autofunction:: eryn.moves.tempering.make_ladder
