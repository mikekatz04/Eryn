Priors
------------------------

Prior distributions live in the :mod:`eryn.priors` subpackage. The sampler takes one
:class:`~eryn.priors.ProbDistContainer` per branch
(``priors={branch_name: ProbDistContainer(...)}``, or a plain
``{branch_name: {parameter key: distribution}}`` dict that the sampler wraps). The container
sums independent component priors, each owning one column or a tuple of columns of the
``(N, ndim)`` coordinate array. :meth:`~eryn.priors.ProbDistContainer.rvs` draws samples
(for starting points, and for the births of
:class:`~eryn.moves.DistributionGenerateRJ` when it draws from the priors).

Sampling space and physical space: the plain ``rvs`` / ``logpdf`` / ``pdf`` / ``cdf``
methods and the ``minimum`` / ``maximum`` / ``width`` attributes of every distribution work
in the SAMPLING variable; the ``*_physical`` methods map to the PHYSICAL variable. The two
coincide for :class:`~eryn.priors.UniformDistribution`, :class:`~eryn.priors.Gaussian` and
the other priors without a transform. For the transform-backed priors
(:class:`~eryn.priors.LogUniform`, :class:`~eryn.priors.Log10Uniform`,
:class:`~eryn.priors.PowerLaw`, :class:`~eryn.priors.CosineUniform`,
:class:`~eryn.priors.SineUniform`, :class:`~eryn.priors.UniformInVolume`,
:class:`~eryn.priors.InverseUniform`) they differ, and the container calls the plain
methods: a column holding ``LogUniform(1.0, 100.0)`` is drawn and scored in ``ln x`` on
``[0, ln 100]``. The likelihood, or a :class:`~eryn.utils.TransformContainer`, must map such
a column back to ``x``.

The legacy module :mod:`eryn.prior` (``uniform_dist``, ``log_uniform``,
``MappedUniformDistribution``) is a deprecated shim over :mod:`eryn.priors`; importing it
emits a ``DeprecationWarning``.

.. automodule:: eryn.priors

Probability Distribution Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.priors.ProbDistContainer
    :members:
    :show-inheritance:

Base Classes
~~~~~~~~~~~~

.. autoclass:: eryn.priors.base.AbstractPrior
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.Prior
    :members:
    :show-inheritance:

.. autoexception:: eryn.priors.PriorException
    :show-inheritance:

Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~

Sampling space equal to physical space:

.. autoclass:: eryn.priors.UniformDistribution
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.DeltaFunction
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.Gaussian
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.Normal
    :show-inheritance:

.. autoclass:: eryn.priors.LogNormal
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.LogGaussian
    :show-inheritance:

.. autoclass:: eryn.priors.Exponential
    :members:
    :show-inheritance:

Uniform in a transformed sampling variable:

.. autoclass:: eryn.priors.LogUniform
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.Log10Uniform
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.PowerLaw
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.UniformInVolume
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.InverseUniform
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.CosineUniform
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.SineUniform
    :members:
    :show-inheritance:

Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.priors.DiscreteUniform
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.CategoricalDistribution
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.PoissonDistribution
    :members:
    :show-inheritance:

Joint Distributions
~~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.priors.JointPrior
    :members:
    :show-inheritance:

.. autoclass:: eryn.priors.MultivariateGaussian
    :members:
    :show-inheritance:

Deprecated Module eryn.prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kept for backward compatibility; it also re-exports ``ProbDistContainer``,
``UniformDistribution`` and ``LogUniform`` from :mod:`eryn.priors`. New code should import
from :mod:`eryn.priors`.

.. autofunction:: eryn.prior.uniform_dist

.. autofunction:: eryn.prior.log_uniform

.. autoclass:: eryn.prior.MappedUniformDistribution
    :members:
    :show-inheritance:
