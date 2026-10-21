"""
Deprecated prior module. 

Maintained for backward compatibility. Please update your imports to use 
the new `eryn.priors` subpackage.
"""

import warnings

from .priors import UniformDistribution, LogUniform, ProbDistContainer

warnings.warn(
    "The 'eryn.prior' module is deprecated and will be removed in a future release. "
    "Please use the new 'eryn.priors' subpackage (e.g., 'from eryn.priors import UniformDistribution').",
    DeprecationWarning,
    stacklevel=2,
)

def uniform_dist(min, max, use_cupy=False, return_gpu=False):
    """Deprecated wrapper for UniformDistribution."""
    warnings.warn(
        "eryn.prior.uniform_dist is deprecated. Use eryn.priors.analytical.UniformDistribution directly.", 
        DeprecationWarning, 
        stacklevel=2
    )
    return UniformDistribution(
        minimum=min, 
        maximum=max, 
        use_cupy=use_cupy, 
        return_gpu=return_gpu
    )

def log_uniform(min, max):
    """Deprecated wrapper for LogUniform."""
    warnings.warn(
        "eryn.prior.log_uniform is deprecated. Use eryn.priors.analytical.LogUniform directly.", 
        DeprecationWarning, 
        stacklevel=2
    )
    return LogUniform(minimum=min, maximum=max)

class MappedUniformDistribution(UniformDistribution):
    """
    Deprecated. Maps uniform distribution to 0 to 1.
    
    In the old infrastructure, this effectively returned a logpdf of 0.0 inside 
    the bounds and -inf outside (i.e. an unnormalized uniform prior).
    """
    def __init__(self, min, max, use_cupy=False, return_gpu=False):
        warnings.warn(
            "eryn.prior.MappedUniformDistribution is deprecated. "
            "If you need an unnormalized prior, consider subclassing the new UniformDistribution.", 
            DeprecationWarning, 
            stacklevel=2
        )
        super().__init__(
            minimum=min, 
            maximum=max, 
            use_cupy=use_cupy, 
            return_gpu=return_gpu
        )
        self.logpdf_val = 0.0