Code Projects
--------------------

* Implement Product-Space MCMC for direct model comparison. 
    * Use supplimental objects to carry log-Likelihood for each model. 
    * Perform tempering within each model. 
* Plotting Module to automatically produce a variety of plots.
* Produce image for tree metaphor.

Code TODOs
-----------

* :class:`eryn.moves.StretchMove`: add log proposal option used in ptemcee with a comparison.
* :class:`eryn.moves.tempering`: add stepping-stone integration.
* :class:`eryn.moves.group`: Combine with red-blue where the stationary distribution is split in two according to two groups of walkers. Will guarantee detailed balance always.
