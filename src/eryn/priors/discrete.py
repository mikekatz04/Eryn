"""Discrete priors over integer values.

:class:`DiscreteUniform` (uniform over ``minimum .. maximum`` inclusive),
:class:`CategoricalDistribution` (uniform over ``0 .. n_categories - 1``) and
:class:`PoissonDistribution`. Samples are returned as float64 arrays holding integer values,
so they fit in the sampler's float coordinate arrays; ``logpdf`` (alias ``logpmf`` on
:class:`DiscreteUniform`) is ``-inf`` at non-integer values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..utils.utility import NDArrayLike
from .base import Prior


class DiscreteUniform(Prior):
    """Uniform prior over the integers ``minimum, minimum + 1, ..., maximum`` (inclusive).

    Useful for discrete states such as a model index. ``n_states = maximum - minimum + 1``
    and every state has probability ``1 / n_states``.

    Args:
        minimum: Lowest state; a Python ``int`` (numpy integers fail the type check).
        maximum: Highest state; a Python ``int`` strictly greater than ``minimum``.
        name: Name of the prior. Default ``None``.
        name_phys: Physical parameter name. Default ``None``.
        latex_label: LaTeX label. Default ``None``.
        unit: Unit string. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        min_int (int): Lowest state.
        max_int (int): Highest state.
        n_states (int): Number of states.
        minimum (float): ``float(minimum)``.
        maximum (float): ``float(maximum)``.

    Raises:
        ValueError: If a bound is not an ``int`` or ``minimum >= maximum``.

    Example:
        >>> import numpy as np
        >>> from eryn.priors import DiscreteUniform
        >>> dist = DiscreteUniform(1, 4)
        >>> bool(np.all(np.isin(dist.rvs(20), [1.0, 2.0, 3.0, 4.0])))
        True
        >>> bool(dist.logpdf(2.0) == -np.log(4)), float(dist.logpdf(2.5))
        (True, -inf)
    """

    def __init__(
        self,
        minimum: int,
        maximum: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(minimum, int) or not isinstance(maximum, int):
            raise ValueError(f"DiscreteUniform requires integer bounds. Got min: {minimum}, max: {maximum}")
        if minimum >= maximum:
            raise ValueError(f"maximum ({maximum}) must be strictly greater than minimum ({minimum}).")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            unit=unit,
            minimum=float(minimum),
            maximum=float(maximum),
            check_range_nonzero=True,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.min_int = minimum
        self.max_int = maximum
        self.n_states = self.max_int - self.min_int + 1
        self._logpdf_val = -np.log(self.n_states)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw states uniformly from ``min_int .. max_int``.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 array of integer values, shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)
            
        samples = self.xp.random.randint(self.min_int, self.max_int + 1, size=size)
        samples = samples.astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Log-probability ``-log(n_states)`` at integer states in range, ``-inf`` elsewhere.

        Also available as ``logpmf``.

        Args:
            x: Value(s); non-integer values get ``-inf``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 log-probabilities, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        
        mask = (x_arr >= self.min_int) & (x_arr <= self.max_int) & (self.xp.round(x_arr) == x_arr)
        out[mask] = self._logpdf_val
        return self._to_device(out)

    logpmf = logpdf

    def cdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Step CDF: ``clip(floor(x) - min_int + 1, 0, n_states) / n_states``.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: CDF values in ``[0, 1]``, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        states_below = self.xp.floor(x_arr) - self.min_int + 1
        states_below = self.xp.clip(states_below, 0, self.n_states)
        out = states_below / self.n_states
        return self._to_device(out)


class CategoricalDistribution(DiscreteUniform):
    """Uniform categorical prior over ``0, 1, ..., n_categories - 1``.

    A :class:`DiscreteUniform` with ``minimum = 0`` and ``maximum = n_categories - 1``.

    Args:
        n_categories: Number of categories; a Python ``int`` ``>= 2``.
        name: Name of the prior. Default ``None``.
        name_phys: Physical parameter name. Default ``None``.
        latex_label: LaTeX label. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`DiscreteUniform` (e.g. ``unit``).

    Raises:
        ValueError: If ``n_categories`` is not an ``int`` or is ``< 2``.
    """

    def __init__(
        self,
        n_categories: int,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if not isinstance(n_categories, int) or n_categories < 2:
            raise ValueError(f"Categorical prior requires n_categories >= 2. Got {n_categories}.")

        super().__init__(
            minimum=0,
            maximum=n_categories - 1,
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )


class PoissonDistribution(Prior):
    """Poisson prior for non-negative integer counts.

    ``p(k) = lam**k * exp(-lam) / k!`` for integer ``k >= 0`` (``minimum = 0``,
    ``maximum = inf``). ``log k!`` is evaluated with ``gammaln`` from ``scipy.special``
    (numpy) or ``cupyx.scipy.special`` (``use_cupy``), resolved once at construction.

    Args:
        lam: Rate (mean); must be ``> 0``.
        name: Name of the prior. Default ``None``.
        name_phys: Physical parameter name. Default ``None``.
        latex_label: LaTeX label. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        **kwargs: Forwarded to :class:`~eryn.priors.base.Prior`.

    Attributes:
        lam (float): Rate (mean).

    Raises:
        ValueError: If ``lam <= 0``.
        ImportError: If ``use_cupy`` and ``cupyx.scipy.special`` cannot be imported.
    """

    def __init__(
        self,
        lam: float,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        if lam <= 0:
            raise ValueError(f"Poisson rate parameter 'lam' must be strictly positive. Got {lam}.")

        super().__init__(
            name=name,
            name_phys=name_phys,
            latex_label=latex_label,
            minimum=0.0,
            maximum=np.inf,
            check_range_nonzero=False,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            **kwargs
        )

        self.lam = lam
        self._log_lam = np.log(self.lam)

        # Resolve log-gamma function once at init
        if self.use_cupy:
            try:
                from cupyx.scipy.special import gammaln # type: ignore[import]
                self._gammaln = gammaln
            except ImportError:
                raise ImportError(
                    "The Poisson prior requires 'cupyx' for the gammaln function "
                    "when use_cupy=True. Please install cupyx."
                )
        else:
            from scipy.special import gammaln
            self._gammaln = gammaln

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        """Draw Poisson counts.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 array of non-negative integer values, shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)
            
        samples = self.xp.random.poisson(lam=self.lam, size=size).astype(self.xp.float64)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Log-probability ``k log(lam) - lam - log(k!)`` at integers ``k >= 0``, else ``-inf``.

        Args:
            x: Value(s).
            **kwargs: Ignored.

        Returns:
            NDArrayLike: float64 log-probabilities, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        out = self.xp.full_like(x_arr, -np.inf, dtype=self.xp.float64)
        
        mask = (x_arr >= 0) & (self.xp.round(x_arr) == x_arr)
        valid_x = self.xp.where(mask, x_arr, 0.0)
        
        log_prob = valid_x * self._log_lam - self.lam - self._gammaln(valid_x + 1.0)
        
        out[mask] = log_prob[mask]
        return self._to_device(out)