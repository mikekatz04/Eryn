# -*- coding: utf-8 -*-

"""
Prior infrastructure for the Eryn sampler.

This subpackage contains base classes, analytical priors, discrete priors, 
joint N-dimensional priors, and the container classes required to manage them 
across complex, trans-dimensional sampler configurations.
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

# We will create this container next
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