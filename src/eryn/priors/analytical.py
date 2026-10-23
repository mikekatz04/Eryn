"""Closed-form 1-D priors.

Two families:

- Priors whose sampling and physical spaces coincide: :class:`UniformDistribution`,
  :class:`DeltaFunction`, :class:`Gaussian` (alias :class:`Normal`), :class:`LogNormal`
  (alias :class:`LogGaussian`) and :class:`Exponential`.
- Uniform priors in a transformed SAMPLING variable ``u``, with the physical variable
  ``x`` reached through the ``*_physical`` methods: :class:`PowerLaw`
  (``u = x**(alpha + 1)``), :class:`LogUniform` (``u = ln x``), :class:`Log10Uniform`
  (``u = log10 x``), :class:`CosineUniform` (``u = cos x``), :class:`SineUniform`
  (``u = sin x``), :class:`UniformInVolume` (``u = x**3``) and :class:`InverseUniform`
  (``u = 1/x``).

For the second family ``rvs`` / ``logpdf`` / ``pdf`` / ``cdf`` and ``minimum`` /
``maximum`` / ``width`` refer to ``u``. :class:`~eryn.priors.probdist.ProbDistContainer`
calls the plain methods, so a container column holding, e.g., a :class:`LogUniform` is
sampled and scored in ``ln x``; the likelihood (or a
:class:`~eryn.utils.TransformContainer`) must map it back to ``x``.

All random draws use the global ``numpy.random`` (or ``cupy.random``) state.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import ArrayLike

from ..utils.utility import NDArrayLike
from .base import Prior

# margin kept inside [-1, 1] by CosineUniform / SineUniform so arccos / arcsin and the
# log-Jacobian stay finite at the edges
ANGLE_SAFE = 1e-12


class UniformDistribution(Prior):
    """Uniform prior on ``[minimum, maximum]``.

    The bounds are inclusive in :meth:`logpdf` / :meth:`pdf`. This is the distribution
    behind the legacy ``eryn.prior.uniform_dist(min, max)`` and the base class of the
    transform-backed priors in this module (which pass their transformed bounds here).

    Args:
        minimum: Lower bound.
        maximum: Upper bound; must exceed ``minimum``.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior` (``name_phys``,
            ``latex_label``, ``unit``, ``boundary``, ...).

    Attributes:
        pdf_val (float): ``1 / width``.
        logpdf_val (float): ``log(1 / width)``.

    Raises:
        ValueError: If ``maximum <= minimum``.

    Example:
        >>> from eryn.priors import UniformDistribution
        >>> dist = UniformDistribution(0.0, 2.0)
        >>> dist.rvs(5).shape
        (5,)
        >>> float(dist.logpdf(1.0)), float(dist.logpdf(3.0))
        (-0.6931471805599453, -inf)
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=minimum,
            maximum=maximum,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.pdf_val = 1.0 / self.width 
        self.logpdf_val = self.xp.log(self.pdf_val)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw uniform samples in ``[minimum, maximum)``.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 samples of shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)

        samples = self.xp.random.uniform(self.minimum, self.maximum, size=size)
        return self._to_device(samples)

    def pdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Density: ``1 / width`` inside ``[minimum, maximum]``, 0 outside.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Density, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.pdf_val * ((x_arr >= self.minimum) & (x_arr <= self.maximum))
        return self._to_device(out)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Log-density: ``-log(width)`` inside ``[minimum, maximum]``, ``-inf`` outside.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 log-density, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)

        mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
        out[mask] = self.logpdf_val

        return self._to_device(out)
    
    
class DeltaFunction(Prior):
    """Dirac delta prior at ``peak``.

    :meth:`rvs` always returns ``peak``; :meth:`logpdf` is ``+inf`` at exactly ``peak`` and
    ``-inf`` elsewhere (so a :class:`~eryn.priors.probdist.ProbDistContainer` sum is
    ``+inf`` for points at the peak); :meth:`cdf` is the unit step at ``peak``.
    ``minimum == maximum == peak``.

    Args:
        peak: Location of the delta function.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        peak (float): Location of the delta function.
    """

    def __init__(
        self,
        peak: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=peak,
            maximum=peak,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.peak = peak

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Return an array of shape ``size`` filled with ``peak`` (float64).

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Array of ``peak`` values.
        """
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.full(size, self.peak, dtype=self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """``+inf`` where ``x == peak``, ``-inf`` elsewhere.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Log-density, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr == self.peak, self.xp.inf, -self.xp.inf)
        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        """Unit step: 1 where ``x >= peak``, 0 elsewhere.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: CDF values, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.where(x_arr >= self.peak, 1.0, 0.0)
        return self._to_device(out)


class PowerLaw(UniformDistribution):
    """Power-law prior, ``p(x)`` proportional to ``x**alpha`` on ``[minimum, maximum]``.

    SAMPLING space: uniform in ``u = x**(alpha + 1)``; the stored ``minimum`` / ``maximum``
    are the ``u`` bounds (``min`` / ``max`` of ``minimum**(alpha + 1)`` and
    ``maximum**(alpha + 1)``, so a negative ``alpha + 1`` is handled). PHYSICAL space:
    ``x`` via :meth:`rvs_physical` / :meth:`logpdf_physical`, with
    ``log|du/dx| = log|alpha + 1| + alpha * log(x)``.

    Args:
        alpha: Power-law index; ``-1`` is not allowed (use :class:`LogUniform`).
        minimum: Lower PHYSICAL bound.
        maximum: Upper PHYSICAL bound. (The ``u`` bounds are sorted, so swapped bounds are
            accepted; equal bounds raise.)
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`UniformDistribution`.

    Attributes:
        alpha (float): Power-law index.

    Raises:
        ValueError: If ``alpha == -1``, or ``minimum < 0`` with a non-integer
            ``alpha + 1``, or the transformed bounds coincide.

    Example:
        >>> from eryn.priors import PowerLaw
        >>> dist = PowerLaw(alpha=2.0, minimum=1.0, maximum=2.0)
        >>> dist.minimum, dist.maximum
        (1.0, 8.0)
        >>> x = dist.rvs_physical(100)
        >>> bool(((x >= 1.0) & (x <= 2.0)).all())
        True
        >>> round(float(dist.pdf_physical(1.5)), 6)
        0.964286
    """

    def __init__(
        self,
        alpha: float,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if alpha == -1:
            raise ValueError("For alpha=-1, use the LogUniform class instead.")
        if minimum < 0 and (alpha + 1) % 1 != 0:
            raise ValueError("Fractional powers of negative numbers are undefined.")

        self.alpha = alpha
        self._power = alpha + 1.0
        self._inv_power = 1.0 / self._power
        self._log_abs_power = np.log(np.abs(self._power))

        u_bound_1 = minimum ** self._power
        u_bound_2 = maximum ** self._power

        super().__init__(
            minimum=min(u_bound_1, u_bound_2),
            maximum=max(u_bound_1, u_bound_2),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map physical ``x`` to ``u = x**(alpha + 1)``."""
        return x_phys ** self._power

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map ``u`` back to ``x = u**(1 / (alpha + 1))``."""
        return u_samp ** self._inv_power

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """``log|du/dx| = log|alpha + 1| + alpha * log(x)`` (``x <= 0`` treated as 1)."""
        x_phys = self.xp.asarray(x_phys)
        valid_x = self.xp.where(x_phys > 0, x_phys, 1.0)
        out = self._log_abs_power + self.alpha * self.xp.log(valid_x)
        return self._to_device(out)
    

class LogUniform(UniformDistribution):
    """Log-uniform prior, ``p(x)`` proportional to ``1/x`` on ``[minimum, maximum]``.

    SAMPLING space: uniform in ``u = ln x`` on ``[ln(minimum), ln(maximum)]``; the stored
    ``minimum`` / ``maximum`` / ``width`` are these ln-space values, and :meth:`rvs` /
    :meth:`logpdf` take and return ``u``. PHYSICAL space: ``x`` via :meth:`rvs_physical` /
    :meth:`logpdf_physical`, with ``log|du/dx| = -ln x``.

    Args:
        minimum: Lower PHYSICAL bound; must be ``> 0``.
        maximum: Upper PHYSICAL bound; must exceed ``minimum``.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`UniformDistribution`.

    Raises:
        ValueError: If ``minimum <= 0`` or ``maximum <= minimum``.

    Warning:
        :class:`~eryn.priors.probdist.ProbDistContainer` calls the plain ``rvs`` /
        ``logpdf``, so a container column holding this prior is drawn and scored in
        ``ln x``. Code that reads that column must treat it as ``ln x``.

    Example:
        >>> from eryn.priors import LogUniform
        >>> dist = LogUniform(1.0, 100.0)
        >>> dist.minimum, round(dist.maximum, 4)
        (0.0, 4.6052)
        >>> x = dist.rvs_physical(100)
        >>> bool(((x >= 1.0) & (x <= 100.0)).all())
        True
        >>> round(float(dist.pdf_physical(10.0)), 6)
        0.021715
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("LogUniform minimum value must be strictly positive.")

        super().__init__(
            minimum=float(np.log(minimum)),
            maximum=float(np.log(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map physical ``x`` to ``u = ln x``."""
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(self.xp.log(x_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map ``u`` back to ``x = exp(u)``."""
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.exp(u_samp))

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """``log|du/dx| = -ln x``."""
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(-self.xp.log(x_phys))


class CosineUniform(UniformDistribution):
    """Uniform-in-cosine prior for a polar angle (e.g. an inclination ``iota``).

    SAMPLING space: uniform in ``u = cos(iota)`` on
    ``[-1 + ANGLE_SAFE, 1 - ANGLE_SAFE]`` (``ANGLE_SAFE = 1e-12``). PHYSICAL space:
    ``iota`` in ``(0, pi)`` with ``p(iota)`` proportional to ``sin(iota)``, via
    :meth:`rvs_physical` (``arccos``) and :meth:`logpdf_physical`
    (``log|du/diota| = log(sin(iota))``). The bounds are fixed; there are no
    ``minimum`` / ``maximum`` arguments.

    Args:
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`UniformDistribution`.

    Example:
        >>> from eryn.priors import CosineUniform
        >>> iota = CosineUniform().rvs_physical(100)
        >>> bool(((iota > 0.0) & (iota < 3.1416)).all())
        True
    """

    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            minimum=-1.0 + ANGLE_SAFE,
            maximum=1.0 - ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, iota_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map the angle to ``u = cos(iota)``."""
        iota_phys = self.xp.asarray(iota_phys)
        return self._to_device(self.xp.cos(iota_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map ``u`` back to ``iota = arccos(u)``."""
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.arccos(u_samp))

    def _jacobian(self, iota_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """``log|du/diota| = log(sin(iota))``."""
        iota_phys = self.xp.asarray(iota_phys)
        return self._to_device(self.xp.log(self.xp.sin(iota_phys)))


class SineUniform(UniformDistribution):
    """Uniform-in-sine prior for a latitude-like angle (e.g. a declination ``beta``).

    SAMPLING space: uniform in ``u = sin(beta)`` on
    ``[-1 + ANGLE_SAFE, 1 - ANGLE_SAFE]`` (``ANGLE_SAFE = 1e-12``). PHYSICAL space:
    ``beta`` in ``(-pi/2, pi/2)`` with ``p(beta)`` proportional to ``cos(beta)``, via
    :meth:`rvs_physical` (``arcsin``) and :meth:`logpdf_physical`
    (``log|du/dbeta| = log(cos(beta))``). The bounds are fixed; there are no
    ``minimum`` / ``maximum`` arguments.

    Args:
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`UniformDistribution`.
    """

    def __init__(
        self,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            minimum=-1.0 + ANGLE_SAFE,
            maximum=1.0 - ANGLE_SAFE,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, beta_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map the angle to ``u = sin(beta)``."""
        beta_phys = self.xp.asarray(beta_phys)
        return self._to_device(self.xp.sin(beta_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map ``u`` back to ``beta = arcsin(u)``."""
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.arcsin(u_samp))

    def _jacobian(self, beta_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """``log|du/dbeta| = log(cos(beta))``."""
        beta_phys = self.xp.asarray(beta_phys)
        return self._to_device(self.xp.log(self.xp.cos(beta_phys)))
    

class Log10Uniform(UniformDistribution):
    """Log-uniform prior sampled in ``u = log10 x``.

    SAMPLING space: uniform in ``u = log10 x`` on ``[log10(minimum), log10(maximum)]``
    (the stored ``minimum`` / ``maximum`` / ``width``). PHYSICAL space: ``x`` with ``p(x)``
    proportional to ``1/x`` on ``[minimum, maximum]``, via :meth:`rvs_physical` and
    :meth:`logpdf_physical` (``log|du/dx| = -ln x - ln(ln 10)``). The same
    :class:`~eryn.priors.probdist.ProbDistContainer` caveat as :class:`LogUniform` applies
    (container columns are ``log10 x``).

    Args:
        minimum: Lower PHYSICAL bound; must be ``> 0``.
        maximum: Upper PHYSICAL bound; must exceed ``minimum``.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`UniformDistribution`.

    Raises:
        ValueError: If ``minimum <= 0`` or ``maximum <= minimum``.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("Log10Uniform minimum value must be strictly positive.")

        super().__init__(
            minimum=float(np.log10(minimum)),
            maximum=float(np.log10(maximum)),
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self._log_log10 = np.log(np.log(10.0))

        self.forward_transform = self._forward
        self.inverse_transform = self._inverse
        self.log_jacobian = self._jacobian

    def _forward(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map physical ``x`` to ``u = log10 x``."""
        x_phys = self.xp.asarray(x_phys)
        return self._to_device(self.xp.log10(x_phys))

    def _inverse(self, u_samp: NDArrayLike, **kwargs) -> NDArrayLike:
        """Map ``u`` back to ``x = 10**u``."""
        u_samp = self.xp.asarray(u_samp)
        return self._to_device(self.xp.power(10.0, u_samp))

    def _jacobian(self, x_phys: NDArrayLike, **kwargs) -> NDArrayLike:
        """``log|du/dx| = -ln x - ln(ln 10)``."""
        x_phys = self.xp.asarray(x_phys)
        out = -self.xp.log(x_phys) - self._log_log10
        return self._to_device(out)
    
    
class UniformInVolume(PowerLaw):
    """Uniform-in-volume prior for a distance ``d``: :class:`PowerLaw` with ``alpha = 2``.

    SAMPLING space: uniform in ``u = d**3`` on ``[minimum**3, maximum**3]``. PHYSICAL
    space: ``d`` with ``p(d)`` proportional to ``d**2`` on ``[minimum, maximum]``.

    Args:
        minimum: Lower PHYSICAL bound; must be ``>= 0``.
        maximum: Upper PHYSICAL bound. (The ``u`` bounds are sorted, so swapped bounds are
            accepted; equal bounds raise.)
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`PowerLaw`.

    Raises:
        ValueError: If ``minimum < 0`` or ``maximum == minimum``.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum < 0:
            raise ValueError("Distance cannot be negative.")
            
        super().__init__(
            alpha=2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )


class InverseUniform(PowerLaw):
    """Uniform-in-``1/x`` prior: :class:`PowerLaw` with ``alpha = -2``.

    SAMPLING space: uniform in ``u = 1/x`` on ``[1/maximum, 1/minimum]``. PHYSICAL space:
    ``x`` with ``p(x)`` proportional to ``x**-2`` on ``[minimum, maximum]``.

    Args:
        minimum: Lower PHYSICAL bound; must be ``> 0``.
        maximum: Upper PHYSICAL bound. (The ``u`` bounds are sorted, so swapped bounds are
            accepted; equal bounds raise.)
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`PowerLaw`.

    Raises:
        ValueError: If ``minimum <= 0`` or ``maximum == minimum``.
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        if minimum <= 0:
            raise ValueError("InverseUniform minimum value must be strictly positive.")
            
        super().__init__(
            alpha=-2.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )


class Gaussian(Prior):
    """Gaussian (normal) prior with mean ``mu`` and standard deviation ``sigma``.

    The support is unbounded (``minimum = -inf``, ``maximum = inf``), so the generic
    numerical :meth:`~eryn.priors.base.Prior.cdf` raises ``ValueError`` for this class.

    Args:
        mu: Mean.
        sigma: Standard deviation (not validated).
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        mu (float): Mean.
        sigma (float): Standard deviation.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        self.mu = mu
        self.sigma = sigma
        self._log_norm = 0.5 * np.log(2.0 * np.pi * self.sigma**2)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw normal samples.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Samples of shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Normalised Gaussian log-density.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: ``-0.5 * ((x - mu) / sigma)**2 - 0.5 * log(2 pi sigma**2)``, shape of
            ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = -0.5 * ((x_arr - self.mu) / self.sigma) ** 2 - self._log_norm
        return self._to_device(out)


class Normal(Gaussian):
    """Alias of :class:`Gaussian` (same constructor and methods)."""
    pass


class LogNormal(Prior):
    """Log-normal prior: ``ln x`` is normal with mean ``mu`` and standard deviation ``sigma``.

    Defined on ``x > 0`` (``minimum = 0``, ``maximum = inf``); :meth:`logpdf` is ``-inf``
    for ``x <= 0``. Sampling and physical spaces coincide (no transform).

    Args:
        mu: Mean of ``ln x``.
        sigma: Standard deviation of ``ln x``; must be ``> 0``.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        mu (float): Mean of ``ln x``.
        sigma (float): Standard deviation of ``ln x``.

    Raises:
        ValueError: If ``sigma <= 0``.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if sigma <= 0.0:
            raise ValueError("Standard deviation sigma must be positive.")
        self.mu = mu
        self.sigma = sigma
        self._log_sqrt_2pi = np.log(np.sqrt(2.0 * np.pi))

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw log-normal samples (``x``, not ``ln x``).

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Positive samples of shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Normalised log-normal log-density in ``x``; ``-inf`` for ``x <= 0``.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 log-density, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr > self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 1.0)
        log_x = self.xp.log(valid_x)
        
        log_prob = (
            -0.5 * ((log_x - self.mu) / self.sigma) ** 2
            - self.xp.log(valid_x * self.sigma)
            - self._log_sqrt_2pi
        )
        out[mask] = log_prob[mask]
        
        return self._to_device(out)


class LogGaussian(LogNormal):
    """Alias of :class:`LogNormal` (same constructor and methods)."""
    pass


class Exponential(Prior):
    """Exponential prior with scale (mean) ``mu`` on ``x >= 0``.

    ``p(x) = exp(-x / mu) / mu`` for ``x >= 0`` (``minimum = 0``, ``maximum = inf``).

    Args:
        mu: Scale (mean); must be ``> 0``.
        name: Name of the prior. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        mu (float): Scale (mean).

    Raises:
        ValueError: If ``mu <= 0``.
    """

    def __init__(
        self,
        mu: float,
        name: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            minimum=0.0,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs,
        )
        if mu <= 0.0:
            raise ValueError("Exponential mean 'mu' must be positive.")
        self.mu = mu
        self._log_mu = np.log(self.mu)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw exponential samples.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Non-negative samples of shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)
        samples = self.xp.random.exponential(scale=self.mu, size=size)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Log-density ``-x / mu - log(mu)`` for ``x >= 0``, ``-inf`` below.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 log-density, shape of ``x``.

        Note:
            As written, an array input with SOME negative entries raises a numpy
            ``ValueError`` from the masked assignment (the right-hand side keeps the full
            shape) instead of returning ``-inf`` there. Scalars and all-non-negative arrays
            work.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -self.xp.inf, dtype=self.xp.float64)
        mask = x_arr >= self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = -(valid_x / self.mu) - self._log_mu
        
        return self._to_device(out)

    def cdf(self, x: ArrayLike, **kwargs: Any) -> NDArrayLike:
        """Closed-form CDF ``1 - exp(-x / mu)`` for ``x >= 0``, 0 below.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 CDF values, shape of ``x``.

        Note:
            Same masked-assignment limitation as :meth:`logpdf` for arrays with some
            negative entries.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.zeros_like(x_arr, dtype=self.xp.float64)
        mask = x_arr >= self.minimum
        
        valid_x = self.xp.where(mask, x_arr, 0.0)
        out[mask] = 1.0 - self.xp.exp(-valid_x / self.mu)
        
        return self._to_device(out)