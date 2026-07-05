"""The base Prior classes.

Provides a functional, stateless base Prior class supporting GPU acceleration,
automatic conditional dependency tracking via method signatures, and 
seamless transformations between physical and sampling parameter spaces.
"""

from __future__ import annotations

import inspect
import warnings
from copy import deepcopy
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike


class PriorException(Exception):
    """General base class for all prior exceptions"""
    pass


class AbstractPrior:
    """Base class for shared Prior functionality.

    Handles device resolution, automatic dependency inference, and metadata strings.

    Args:
        use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration.
            (default: ``False``)
        return_gpu (bool, optional): If ``True``, returns GPU arrays when 
            ``use_cupy`` is ``True``. (default: ``False``)

    """

    def __init__(self, use_cupy: bool = False, return_gpu: bool = False):
        self.use_cupy = use_cupy
        self.return_gpu = return_gpu

        if self.use_cupy:
            try:
                import cupy as cp # type: ignore[import]
                self.xp = cp
            except ImportError:
                raise ImportError("use_cupy is True, but CuPy is not installed.")
        else:
            self.xp = np

        self._infer_dependencies()

    def _infer_dependencies(self):
        """Automatically infer dependencies from subclass signatures.
        
        Evaluates ``logpdf`` and ``rvs`` to determine required conditional 
        parameters and caches the class initialization keys for fast ``__repr__`` 
        evaluation.
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
        """Safely move arrays back to CPU if requested.

        Args:
            array (NDArrayLike): The array to format.

        Returns:
            NDArrayLike: CPU or GPU array depending on instance settings.
        """
        if self.use_cupy and not self.return_gpu:
            if hasattr(array, "get"):
                return getattr(array, "get")()
            return np.asarray(array)
        return array

    def __repr__(self) -> str:
        """String representation of the prior for metadata serialization.

        Returns:
            str: Evaluatable string representation of the object.
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
        """Main 1D prior constructor.

        Args:
            name (str, optional): Name associated with prior. (default: ``None``)
            name_phys (str, optional): Name of the associated physical parameter. 
                Defaults to name if not provided. (default: ``None``)
            latex_label (str, optional): Latex label associated with prior. (default: ``None``)
            unit (str, optional): A Latex string describing the units. (default: ``None``)
            minimum (float, optional): Minimum of the domain. (default: ``-np.inf``)
            maximum (float, optional): Maximum of the domain. (default: ``np.inf``)
            boundary (str, optional): Boundary condition (None, 'periodic', 'reflective'). (default: ``None``)
            check_range_nonzero (bool, optional): If ``True``, check that the range is > 0. (default: ``True``)
            use_cupy (bool, optional): If ``True``, uses CuPy for acceleration. (default: ``False``)
            return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
            forward_transform (callable, optional): Transform from physical to sampling space. (default: ``None``)
            inverse_transform (callable, optional): Transform from sampling to physical space. (default: ``None``)
            log_jacobian (callable, optional): Log Jacobian of the physical to sampling transform. (default: ``None``)
                
        Raises:
            ValueError: If ``check_range_nonzero`` is True and maximum <= minimum, or boundary is invalid.
        """
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
        """Draws a random variable(s) from the prior distribution.

        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments for conditional priors.

        Returns:
            NDArrayLike: A random sample drawn from the prior distribution.

        Raises:
            ValueError: If size is not an integer or a tuple of integers.
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
        """Draw a random sample in the physical parameter space. 
        
        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments to pass to the rvs method.
            
        Returns:
            NDArrayLike: A random sample in the physical parameter space.
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
        """The log probability density at the given value(s).

        Args:
            x (ArrayLike): Value(s) to calculate logpdf.
            **kwargs: Additional keyword arguments for conditional priors.

        Returns:
            NDArrayLike: The log probability density.
        """
        raise NotImplementedError("logpdf method is implemented in subclass")
    
    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Log probability density evaluated in the physical parameter space. 
        
        Args:
            x_phys (ArrayLike): Value(s) in physical parameter space.
            **kwargs: Additional keyword arguments.
                
        Returns:
            NDArrayLike: Log probability density in physical space.
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
        """Probability density of the prior distribution.

        Args:
            x (ArrayLike): Value(s) to calculate density.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: The probability density.
        """
        output = self.xp.exp(self.logpdf(x, **kwargs))
        return self._to_device(output)
    
    def pdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Probability density evaluated in the physical parameter space.
        
        Args:
            x_phys (ArrayLike): Value(s) in physical space.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Probability density in physical space.
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
        """Generic method to calculate CDF numerically.

        Args:
            x (ArrayLike): Value(s) to calculate the CDF at.
            n_points (int, optional): Grid points for numerical integration. (default: ``1000``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: CDF values.

        Raises:
            ValueError: If minimum or maximum is infinite.
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
        """Check if values fall within the prior support.

        Args:
            x (ArrayLike): Values to evaluate.
            fallback_samples (int, optional): Number of samples to draw to estimate
                bounds if minimum/maximum are undefined. (default: ``10_000``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Boolean mask indicating inclusion in prior range.
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
        return self._minimum

    @minimum.setter
    def minimum(self, minimum: float):
        self._minimum = minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: float):
        self._maximum = maximum

    @property
    def width(self) -> float:
        return self.maximum - self.minimum