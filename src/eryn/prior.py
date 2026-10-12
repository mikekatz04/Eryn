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

class GenericFunctionalDistribution:
    """
    Generic distribution class, based on user-provided callable functions for the probability density function (pdf) and the inverse cumulative density function (cdf).

    Args:
        pdf_func (callable): A callable function that computes the pdf of the distribution.
        inv_cdf_func (callable): A callable function that computes the inverse cdf of the distribution.
        min_val (float): The minimum value of the distribution's support.
        max_val (float): The maximum value of the distribution's support.
        clip (bool, optional): If ``True``, clip the samples to the range [min_val, max_val]. If ``False``, samples outside this range will be returned as is. (default: ``True``)
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy. The two callable functions must be compatible with the chosen array library. (default: ``False``)
        return_gpu (bool, optional): If ``True``, return CuPy array. If ``False``, return Numpy array. (default: ``False``)
    """

    def __init__(self, pdf_func, inv_cdf_func, min_val, max_val, clip=True, use_cupy=False, return_gpu=False):
        self.pdf_func = pdf_func
        self.inv_cdf_func = inv_cdf_func
        self.min_val = min_val
        self.max_val = max_val
        self.clip = clip
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

    @property
    def xp(self):
        """Numpy or Cupy"""
        xp = np if not self.use_cupy else cp
        return xp

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        rand_unif = self.xp.random.rand(*size)

        out = self.inv_cdf_func(rand_unif)

        if self.clip:
            out = self.xp.clip(out, self.min_val, self.max_val)

        if self.use_cupy and not self.return_gpu:
            return out.get()

        return out

    def pdf(self, x):
        out = self.pdf_func(x)
        if self.use_cupy and not self.return_gpu:
            return out.get()

        return out

    def logpdf(self, x):
        out = self.xp.zeros_like(x)
        out[(x >= self.min_val) & (x <= self.max_val)] = self.xp.log(self.pdf_func(x[(x >= self.min_val) & (x <= self.max_val)]))
        out[(x < self.min_val) | (x > self.max_val)] = -np.inf
        if self.use_cupy and not self.return_gpu:
            return out.get()
            
        return out

    def copy(self):
        return deepcopy(self)