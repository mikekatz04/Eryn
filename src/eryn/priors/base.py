"""Base classes for Eryn priors.

:class:`AbstractPrior` holds what every prior shares: the numpy / cupy backend switch
(``use_cupy`` / ``return_gpu`` and the :attr:`AbstractPrior.xp` module), the conditional
dependency names inferred from the subclass ``logpdf`` / ``rvs`` signatures
(``required_variables``, read by :class:`~eryn.priors.probdist.ProbDistContainer`), and a
metadata ``__repr__``. :class:`Prior` is the 1-D base: domain bounds, boundary label, LaTeX
label and unit, and the optional transforms between the SAMPLING space (where ``rvs`` /
``logpdf`` live) and the PHYSICAL space (where the ``*_physical`` methods live).
:class:`PriorException` is an exported exception class for user code (the priors here do not
raise it).
"""

from __future__ import annotations

import inspect
import warnings
from copy import deepcopy
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike

try:
    import cupy as _CUPY  # type: ignore[import]
except ImportError:
    _CUPY = None


class PriorException(Exception):
    """General base class for prior exceptions.

    Exported from :mod:`eryn.priors` for user code; the priors in this package raise
    ``ValueError`` / ``NotImplementedError`` rather than this class.
    """
    pass


class AbstractPrior:
    """Base class for shared prior functionality (backend, dependencies, metadata).

    On construction the subclass's own ``logpdf`` and ``rvs`` signatures are inspected
    (``_infer_dependencies``): every named parameter other than ``self``, ``x``,
    ``size``, ``rng``, ``random_state``, ``args`` and ``kwargs`` is a conditional variable,
    stored in ``required_variables``. :class:`~eryn.priors.probdist.ProbDistContainer` then
    feeds those variables from its keyword arguments or from the other columns of the
    input array.

    Args:
        use_cupy: If ``True``, compute with CuPy (``xp`` is ``cupy``). Default ``False``.
        return_gpu: If ``True`` (and ``use_cupy``), return CuPy arrays; if ``False``,
            results are moved to the host with ``.get()``. Default ``False``.

    Attributes:
        use_cupy (bool): Backend flag; :attr:`xp` is derived from it.
        return_gpu (bool): Whether results stay on the device.
        required_variables (tuple[str, ...]): Sorted conditional-variable names inferred
            from the subclass ``logpdf`` / ``rvs`` signatures. Only methods defined
            directly on the concrete class are inspected (inherited ones are skipped).

    Raises:
        ImportError: If ``use_cupy`` is ``True`` and CuPy is not installed.
    """

    def __init__(self, use_cupy: bool = False, return_gpu: bool = False):
        if use_cupy and _CUPY is None:
            raise ImportError("use_cupy is True, but CuPy is not installed.")
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

        self._infer_dependencies()

    @property
    def xp(self):
        """module: ``numpy`` or ``cupy``, resolved from ``use_cupy``.

        ``use_cupy`` is kept as a flag so the prior stays deepcopy / pickle safe (storing
        the module as an instance attribute breaks ``copy.deepcopy`` of settings trees that
        contain priors). CuPy is imported once at module import and cached in a
        module-level global, so this is a cheap lookup.
        """
        return _CUPY if self.use_cupy else np

    def _infer_dependencies(self):
        """Infer conditional variables from the subclass signatures and cache repr keys.

        Inspects ``logpdf`` and ``rvs`` (only when defined directly on the concrete class)
        and stores every non-reserved named parameter in ``required_variables``. Also caches
        the ``__init__`` parameter names in ``_repr_keys`` for :meth:`__repr__`.
        """
        deps = set()
        reserved_kwargs = {"self", "x", "size", "rng", "random_state", "kwargs", "args"}

        for method_name in ("logpdf", "rvs"):
            if method_name not in self.__class__.__dict__:
                continue

            method = getattr(self, method_name)
            sig = inspect.signature(method)

            for name, param in sig.parameters.items():
                if name in reserved_kwargs:
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                deps.add(name)

        self.required_variables: tuple[str, ...] = tuple(sorted(deps))

        # Cache __init__ arguments for fast __repr__ evaluation
        sig_init = inspect.signature(self.__class__.__init__)
        self._repr_keys = [
            k for k in sig_init.parameters.keys() 
            if k not in ("self", "kwargs", "args")
        ]

    def _to_device(self, array: NDArrayLike) -> NDArrayLike:
        """Move a result to the host when ``use_cupy`` is set and ``return_gpu`` is not.

        Args:
            array: The result to format.

        Returns:
            NDArrayLike: ``array.get()`` (or ``np.asarray(array)``) when ``use_cupy`` is set
            and ``return_gpu`` is not; otherwise ``array`` unchanged.
        """
        if self.use_cupy and not self.return_gpu:
            if hasattr(array, "get"):
                return getattr(array, "get")()
            return np.asarray(array)
        return array

    def __repr__(self) -> str:
        """String representation of the prior for metadata serialization.

        Lists ``ClassName(key=value, ...)`` for every ``__init__`` parameter that is stored
        as an attribute of the same name with a ``str`` / ``int`` / ``float`` / ``tuple`` /
        ``list`` / ``bool`` value; other parameters are omitted.

        Returns:
            str: The representation.
        """
        class_name = self.__class__.__name__
        args = []
        for key in getattr(self, "_repr_keys", []):
            if hasattr(self, key):
                val = getattr(self, key)
                if isinstance(val, str):
                    args.append(f"{key}='{val}'")
                elif isinstance(val, (int, float, tuple, list, bool)):
                    args.append(f"{key}={val}")
                
        args_str = ", ".join(args)
        return f"{class_name}({args_str})"

    def copy(self) -> AbstractPrior:
        """Return a deep copy of the prior.

        Returns:
            AbstractPrior: Deep copy of self.
        """
        return deepcopy(self)


class Prior(AbstractPrior):
    """Base class for 1-D priors.

    A subclass implements ``rvs(size=..., **kwargs)`` and ``logpdf(x, **kwargs)`` in the
    SAMPLING space. Extra named parameters in those signatures (for example
    ``logpdf(self, x, a=0.0, **kwargs)``) declare conditional variables (see
    :class:`AbstractPrior`). When ``forward_transform`` / ``inverse_transform`` /
    ``log_jacobian`` are given, the ``*_physical`` methods evaluate the same distribution in
    the PHYSICAL space:

    .. code-block:: text

        rvs_physical     = inverse_transform(rvs(...))
        logpdf_physical  = logpdf(forward_transform(x_phys)) + log_jacobian(x_phys)

    Args:
        name: Name associated with the prior. Default ``None``.
        name_phys: Name of the associated physical parameter. Default ``None`` (use
            ``name``).
        latex_label: LaTeX label. Default ``None``: look ``name`` up in the class-level
            ``_default_latex_labels`` dict, else use ``str(name)``.
        unit: LaTeX string with the units. Default ``None``.
        minimum: Lower bound of the SAMPLING-space domain. Default ``-np.inf``.
        maximum: Upper bound of the SAMPLING-space domain. Default ``np.inf``.
        boundary: Boundary-condition label: ``None``, ``"periodic"`` or ``"reflective"``.
            Stored as metadata only; Eryn's proposals take periodicity from
            :class:`~eryn.utils.PeriodicContainer`, not from this field. Default ``None``.
        check_range_nonzero: If ``True``, require ``maximum > minimum``. Default ``True``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        forward_transform: Callable ``f(x_phys, **kwargs) -> x_samp`` from the physical to
            the sampling space. Default ``None``.
        inverse_transform: Callable ``g(x_samp, **kwargs) -> x_phys`` from the sampling to
            the physical space. Default ``None``.
        log_jacobian: Callable ``h(x_phys, **kwargs)`` returning
            ``log|d x_samp / d x_phys|`` at the physical value. Default ``None``.

    Attributes:
        name (str or None): Prior name.
        name_phys (str or None): Physical parameter name.
        boundary (str or None): Boundary-condition label.
        forward_transform (callable or None): Physical -> sampling map.
        inverse_transform (callable or None): Sampling -> physical map.
        log_jacobian (callable or None): ``log|d x_samp / d x_phys|``.

    Raises:
        ValueError: If ``check_range_nonzero`` and ``maximum <= minimum``, or ``boundary`` is
            not one of the supported labels.
        ImportError: If ``use_cupy`` is ``True`` and CuPy is not installed.

    Example:
        A conditional prior whose support follows another column ``a``:

        >>> import numpy as np
        >>> from eryn.priors import Prior, ProbDistContainer, UniformDistribution
        >>> class Shifted(Prior):
        ...     def rvs(self, size=(1,), a=0.0, **kwargs):
        ...         return np.asarray(a) + np.random.uniform(0.0, 1.0, size=size)
        ...     def logpdf(self, x, a=0.0, **kwargs):
        ...         d = np.asarray(x) - np.asarray(a)
        ...         return np.where((d >= 0.0) & (d <= 1.0), 0.0, -np.inf)
        >>> Shifted().required_variables
        ('a',)
        >>> priors = ProbDistContainer({"a": UniformDistribution(0.0, 5.0), "b": Shifted()})
        >>> x = priors.rvs(size=3)
        >>> bool(np.all(np.isfinite(priors.logpdf(x))))
        True
    """

    _default_latex_labels = {}

    def __init__(
        self,
        name: str | None = None,
        name_phys: str | None = None,
        latex_label: str | None = None,
        unit: str | None = None,
        minimum: float = -np.inf,
        maximum: float = np.inf,
        boundary: str | None = None,
        check_range_nonzero: bool = True,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform: Callable[..., NDArrayLike] | None = None,
        inverse_transform: Callable[..., NDArrayLike] | None = None,
        log_jacobian: Callable[..., NDArrayLike] | None = None,
    ):
        """Build the 1-D prior (arguments are documented on the class)."""
        super().__init__(use_cupy=use_cupy, return_gpu=return_gpu)

        if check_range_nonzero and maximum <= minimum:
            raise ValueError(
                f"maximum {maximum} <= minimum {minimum} "
                f"for {type(self).__name__} prior on {name}"
            )

        self.name = name
        self.name_phys = name_phys if name_phys is not None else name
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        self.latex_label = latex_label
        self.boundary = boundary
        
        if self.boundary not in [None, "periodic", "reflective"]:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary}' for prior '{self.name}'. "
                f"Supported values are None, 'periodic', or 'reflective'."
            )

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.log_jacobian = log_jacobian

    def rvs(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draw samples in the SAMPLING space (override in subclasses).

        The base implementation only validates ``size`` and then raises.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Conditional variables (and any extra options) for subclasses.

        Returns:
            NDArrayLike: Samples of shape ``size`` (in subclasses).

        Raises:
            ValueError: If ``size`` is not an ``int`` or a tuple.
            NotImplementedError: Always, in the base class.
        """
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        if isinstance(size, int):
            size = (size,)
        
        raise NotImplementedError("rvs method is implemented in subclass")

    def rvs_physical(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draw samples in the PHYSICAL space: ``inverse_transform(rvs(size))``.

        Without an ``inverse_transform`` this returns the :meth:`rvs` samples unchanged.

        Args:
            size: Output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Forwarded to :meth:`rvs` and to ``inverse_transform``.

        Returns:
            NDArrayLike: Physical-space samples of shape ``size``.
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform is None:
            return self._to_device(samples)
        
        output = self.inverse_transform(samples, **kwargs)
        return self._to_device(output)
    
    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Log-density in the SAMPLING space (override in subclasses).

        Args:
            x: Sampling-space value(s).
            **kwargs: Conditional variables (and any extra options) for subclasses.

        Returns:
            NDArrayLike: Log-density with the shape of ``x`` (in subclasses).

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("logpdf method is implemented in subclass")
    
    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Log-density in the PHYSICAL space.

        Computes ``logpdf(forward_transform(x_phys)) + log_jacobian(x_phys)``. If either
        ``forward_transform`` or ``log_jacobian`` is ``None``, returns ``logpdf(x_phys)``.

        Args:
            x_phys: Physical-space value(s).
            **kwargs: Forwarded to ``forward_transform``, :meth:`logpdf` and
                ``log_jacobian``.

        Returns:
            NDArrayLike: Physical-space log-density, shape of ``x_phys``.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = logp_samp + log_J
        return self._to_device(output)
    
    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Density in the SAMPLING space: ``exp(logpdf(x))``.

        Args:
            x: Sampling-space value(s).
            **kwargs: Forwarded to :meth:`logpdf`.

        Returns:
            NDArrayLike: Density, shape of ``x``.
        """
        output = self.xp.exp(self.logpdf(x, **kwargs))
        return self._to_device(output)
    
    def pdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Density in the PHYSICAL space.

        Computes ``pdf(forward_transform(x_phys)) * exp(log_jacobian(x_phys))``; falls back
        to ``pdf(x_phys)`` when either transform piece is ``None``.

        Args:
            x_phys: Physical-space value(s).
            **kwargs: Forwarded to ``forward_transform``, :meth:`pdf` and ``log_jacobian``.

        Returns:
            NDArrayLike: Physical-space density, shape of ``x_phys``.
        """
        if self.forward_transform is None or self.log_jacobian is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_J = self.log_jacobian(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        n_points: int = 1000,
        **kwargs
    ) -> NDArrayLike:
        """CDF in the SAMPLING space by numerical integration of :meth:`pdf`.

        Integrates :meth:`pdf` with the trapezoidal rule on ``n_points`` evenly spaced points
        over ``[minimum, maximum]``, normalises the cumulative sum to end at 1 (when it is
        positive) and linearly interpolates at ``x`` (0 below the grid, 1 above).
        Subclasses with a closed form override this.

        Args:
            x: Sampling-space value(s).
            n_points: Number of grid points. Default ``1000``.
            **kwargs: Forwarded to :meth:`pdf`.

        Returns:
            NDArrayLike: CDF values with the shape of ``x``; a Python ``float`` when ``x`` is
            a Python ``int`` / ``float`` and ``return_gpu`` is ``False`` (with ``use_cupy``
            that float is then wrapped as a 0-d numpy array).

        Raises:
            ValueError: If ``minimum`` or ``maximum`` is infinite.
        """
        if self.xp.any(self.xp.isinf([self.minimum, self.maximum])):
            raise ValueError(
                "Unable to use the generic CDF calculation for priors with "
                "infinite support."
            )
        
        x_arr = self.xp.asarray(x)
        grid, dx = self.xp.linspace(self.minimum, self.maximum, n_points, retstep=True)

        pdf_grid = self.pdf(grid, **kwargs) 
        cdf_grid = self.xp.zeros_like(pdf_grid)
        # Numerical integration using the trapezoidal rule
        cdf_grid[1:] = 0.5 * dx * self.xp.cumsum(pdf_grid[:-1] + pdf_grid[1:])
        
        if cdf_grid[-1] > 0:
            cdf_grid /= cdf_grid[-1]
        
        output = self.xp.interp(x_arr, grid, cdf_grid, left=0.0, right=1.0)        

        if isinstance(x, (int, float)) and not self.return_gpu:
            output = float(output)
        return self._to_device(output)

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        fallback_samples: int = 10_000,
        **kwargs
    ) -> NDArrayLike:
        """Boolean mask of ``minimum <= x <= maximum`` (SAMPLING-space bounds).

        If ``minimum`` or ``maximum`` has been set to ``None``, the bounds are instead
        estimated from ``fallback_samples`` draws of :meth:`rvs`, with a warning. (The
        constructor defaults are ``-inf`` / ``inf``, which take the direct path.)

        Args:
            x: Sampling-space value(s).
            fallback_samples: Number of draws for the bound estimate. Default ``10_000``.
            **kwargs: Forwarded to :meth:`rvs` on the fallback path.

        Returns:
            NDArrayLike: Boolean mask, shape of ``x``.
        """
        x_arr = self.xp.asarray(x)
        if self.minimum is not None and self.maximum is not None:
            mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
            return self._to_device(mask)
        
        warnings.warn(
            "Prior range not defined, using samples to estimate bounds. "
            "This may result in an inaccurate estimate.",
            stacklevel=2
        )
        samples = self.xp.asarray(self.rvs(size=fallback_samples, **kwargs))
        minimum, maximum = samples.min(), samples.max()
        mask = (x_arr >= minimum) & (x_arr <= maximum)
        return self._to_device(mask)

    @property
    def latex_label(self) -> str:
        """LaTeX label; setting ``None`` uses the class default for ``name``, else ``str(name)``."""
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: str | None = None):
        if latex_label is not None:
            self.__latex_label = latex_label
        elif self.name in self._default_latex_labels:
            self.__latex_label = self._default_latex_labels[self.name]
        else:
            self.__latex_label = str(self.name)

    @property
    def unit(self) -> str | None:
        """LaTeX unit string, or ``None``."""
        return self.__unit

    @unit.setter
    def unit(self, unit: str | None = None):
        self.__unit = unit

    @property
    def latex_label_with_unit(self) -> str:
        """str: Latex label combined with unit string."""
        if self.unit is not None:
            return f"{self.latex_label} [{self.unit}]"
        return self.latex_label

    @property
    def minimum(self) -> float:
        """Lower bound of the SAMPLING-space domain."""
        return self._minimum

    @minimum.setter
    def minimum(self, minimum: float):
        self._minimum = minimum

    @property
    def maximum(self) -> float:
        """Upper bound of the SAMPLING-space domain."""
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: float):
        self._maximum = maximum

    @property
    def width(self) -> float:
        """``maximum - minimum`` (SAMPLING-space width)."""
        return self.maximum - self.minimum