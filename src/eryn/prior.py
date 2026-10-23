"""Deprecated prior module, kept for backward compatibility.

Importing it emits a ``DeprecationWarning``. It re-exports
:class:`~eryn.priors.ProbDistContainer`, :class:`~eryn.priors.UniformDistribution` and
:class:`~eryn.priors.LogUniform` from :mod:`eryn.priors`, and keeps the old helpers
:func:`uniform_dist`, :func:`log_uniform` and :class:`MappedUniformDistribution` as thin
wrappers over the new classes. New code should import from :mod:`eryn.priors`.
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
    """Build a uniform prior on ``[min, max]`` (deprecated wrapper).

    Emits a ``DeprecationWarning`` and returns
    ``UniformDistribution(minimum=min, maximum=max, use_cupy=use_cupy,
    return_gpu=return_gpu)``, whose ``logpdf`` is ``-log(max - min)`` inside the bounds
    (inclusive) and ``-inf`` outside.

    Args:
        min (float): Lower bound.
        max (float): Upper bound.
        use_cupy (bool, optional): Evaluate with ``cupy``. Default ``False``.
        return_gpu (bool, optional): Return ``cupy`` arrays when ``use_cupy``. Default
            ``False``.

    Returns:
        UniformDistribution: The prior (see :class:`eryn.priors.UniformDistribution`).

    Example:
        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore", DeprecationWarning)
        ...     from eryn.prior import uniform_dist
        ...     dist = uniform_dist(0.0, 2.0)
        >>> float(dist.logpdf(1.0))
        -0.6931471805599453
    """
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
    """Build a log-uniform prior between ``min`` and ``max`` (deprecated wrapper).

    Emits a ``DeprecationWarning`` and returns ``LogUniform(minimum=min, maximum=max)``:
    uniform in ``ln x`` on ``[ln(min), ln(max)]`` in the sampling basis (see
    :class:`eryn.priors.LogUniform` for how it maps to the physical ``x``).

    Args:
        min (float): Lower bound in ``x``; must be positive.
        max (float): Upper bound in ``x``.

    Returns:
        LogUniform: The prior.

    Raises:
        ValueError: ``min <= 0`` or ``max <= min``.
    """
    warnings.warn(
        "eryn.prior.log_uniform is deprecated. Use eryn.priors.analytical.LogUniform directly.", 
        DeprecationWarning, 
        stacklevel=2
    )
    return LogUniform(minimum=min, maximum=max)

class MappedUniformDistribution(UniformDistribution):
    """Unnormalized uniform prior on ``[min, max]`` (deprecated).

    A :class:`~eryn.priors.UniformDistribution` whose ``logpdf`` is ``0.0`` inside the
    bounds and ``-inf`` outside. Construction emits a ``DeprecationWarning``.

    Args:
        min (float): Lower bound.
        max (float): Upper bound.
        use_cupy (bool, optional): Evaluate with ``cupy``. Default ``False``.
        return_gpu (bool, optional): Return ``cupy`` arrays when ``use_cupy``. Default
            ``False``.
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