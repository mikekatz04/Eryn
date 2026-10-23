# -*- coding: utf-8 -*-

"""Prior distributions for the Eryn sampler.

The subpackage holds the prior base classes (:class:`~eryn.priors.base.Prior`,
:class:`~eryn.priors.joint.JointPrior`), the 1-D analytical priors
(:mod:`eryn.priors.analytical`), the discrete priors (:mod:`eryn.priors.discrete`), the
multivariate Gaussian, and :class:`~eryn.priors.probdist.ProbDistContainer`, the object the
sampler takes as a branch prior (``priors={branch_name: ProbDistContainer(...)}``).

Every distribution has two coordinate spaces. The plain ``rvs`` / ``logpdf`` / ``pdf`` /
``cdf`` methods and the ``minimum`` / ``maximum`` / ``width`` attributes work in the SAMPLING
space; the ``*_physical`` methods map through the prior's ``forward_transform`` /
``inverse_transform`` / ``log_jacobian`` to the PHYSICAL space. For
:class:`~eryn.priors.analytical.UniformDistribution`, :class:`~eryn.priors.analytical.Gaussian`
and the other priors without transforms the two spaces coincide. For the transform-backed
priors (:class:`~eryn.priors.analytical.LogUniform`,
:class:`~eryn.priors.analytical.Log10Uniform`, :class:`~eryn.priors.analytical.PowerLaw`,
:class:`~eryn.priors.analytical.CosineUniform`, :class:`~eryn.priors.analytical.SineUniform`,
:class:`~eryn.priors.analytical.UniformInVolume`,
:class:`~eryn.priors.analytical.InverseUniform`) they differ, and
:class:`~eryn.priors.probdist.ProbDistContainer` calls the plain sampling-space methods.

The legacy module :mod:`eryn.prior` (``uniform_dist``, ``log_uniform``,
``MappedUniformDistribution``) is a deprecated shim over this subpackage.

Example:
    >>> from eryn.priors import ProbDistContainer, UniformDistribution
    >>> priors = ProbDistContainer({
    ...     "a": UniformDistribution(0.0, 1.0),
    ...     "b": UniformDistribution(-2.0, 2.0),
    ... })
    >>> priors.rvs(size=4).shape
    (4, 2)
"""

from .base import Prior, PriorException
from .joint import JointPrior, MultivariateGaussian
from .analytical import (
    UniformDistribution,
    DeltaFunction,
    PowerLaw,
    LogUniform,
    CosineUniform,
    SineUniform,
    Log10Uniform,
    UniformInVolume,
    InverseUniform,
    Gaussian,
    Normal,
    LogNormal,
    LogGaussian,
    Exponential,
)
from .discrete import (
    DiscreteUniform, 
    CategoricalDistribution, 
    PoissonDistribution,
)

# the container that turns a dict of component priors into a branch prior
from .probdist import ProbDistContainer

__all__ = [
    # Base
    "Prior",
    "PriorException",
    
    # Joint
    "JointPrior",
    "MultivariateGaussian",
    
    # Analytical / 1D Continuous
    "UniformDistribution",
    "DeltaFunction",
    "PowerLaw",
    "LogUniform",
    "CosineUniform",
    "SineUniform",
    "Log10Uniform",
    "UniformInVolume",
    "InverseUniform",
    "Gaussian",
    "Normal",
    "LogNormal",
    "LogGaussian",
    "Exponential",
    
    # Discrete
    "DiscreteUniform",
    "CategoricalDistribution",
    "PoissonDistribution",
    
    # Containers
    "ProbDistContainer",
]