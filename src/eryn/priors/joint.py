from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike

from .base import AbstractPrior


class JointPrior(AbstractPrior):
    _default_latex_labels = {}

    def __init__(
        self,
        names: Sequence[str],
        names_phys: Sequence[str] | None = None,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        boundaries: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        forward_transform_nd: Callable[..., NDArrayLike] | None = None,
        inverse_transform_nd: Callable[..., NDArrayLike] | None = None,
        log_jacobian_nd: Callable[..., NDArrayLike] | None = None,
    ):
        """Main joint prior constructor for N-dimensional parameter spaces.

        Args:
            names (list of str): Names associated with the parameters.
            names_phys (list of str, optional): Physical parameter names. (default: ``None``)
            latex_labels (list of str, optional): Latex labels for parameters. (default: ``None``)
            units (list of str, optional): Strings describing parameter units. (default: ``None``)
            boundaries (list of str, optional): Boundary conditions for each parameter. (default: ``None``)
            use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration. (default: ``False``)
            return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
            forward_transform_nd (callable, optional): Transform from physical to sampling space. (default: ``None``)
            inverse_transform_nd (callable, optional): Transform from sampling to physical space. (default: ``None``)
            log_jacobian_nd (callable, optional): Log-determinant of the Jacobian matrix. (default: ``None``)
                
        Raises:
            ValueError: If names sequence is empty, or lengths of provided metadata lists mismatch.
        """
        super().__init__(use_cupy=use_cupy, return_gpu=return_gpu)

        if not names:
            raise ValueError("Joint priors require a sequence of parameter names.")

        self.names = tuple(names)
        self.num_vars = len(self.names)

        if names_phys is not None:
            if len(names_phys) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} names_phys, got {len(names_phys)}")
            self.names_phys = tuple(
                np if np is not None else n for n, np in zip(self.names, names_phys)
            )
        else:
            self.names_phys = self.names
            
        if boundaries is not None:
            if len(boundaries) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} boundaries, got {len(boundaries)}")
            for boundary in boundaries:
                if boundary not in [None, "periodic", "reflective"]:
                    raise ValueError(
                        f"Invalid boundary condition '{boundary}'. "
                        "Supported values are None, 'periodic', or 'reflective'."
                    )
            self.boundaries = tuple(boundaries)
        else:
            self.boundaries = tuple(None for _ in range(self.num_vars))

        self.latex_label = latex_labels
        self.unit = units

        self.forward_transform_nd = forward_transform_nd
        self.inverse_transform_nd = inverse_transform_nd
        self.log_jacobian_nd = log_jacobian_nd

    def rvs(
        self, 
        size: int | tuple[int, ...] = (1,), 
        **kwargs
    ) -> NDArrayLike:
        """Draws random variable(s) from the joint prior distribution.

        Args:
            size (int or tuple of int, optional): Size of the sample. (default: ``(1,)``)
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Random sample drawn from the prior distribution.

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
            **kwargs: Additional keyword arguments.
            
        Returns:
            NDArrayLike: A random sample in the physical parameter space.
        """
        samples = self.rvs(size=size, **kwargs)
        if self.inverse_transform_nd is None:
            return self._to_device(samples)
        
        output = self.inverse_transform_nd(samples, **kwargs)
        return self._to_device(output)

    def logpdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """The log probability density of the joint prior.

        Args:
            x (ArrayLike): The N-dimensional value(s) to evaluate.
            **kwargs: Additional keyword arguments.

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
            x_phys (ArrayLike): Value(s) in the physical space.
            **kwargs: Additional keyword arguments.
                
        Returns:
            NDArrayLike: Log probability density in physical space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.logpdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        logp_samp = self.logpdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = logp_samp + log_det_J
        return self._to_device(output)

    def pdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Probability density of the joint prior distribution.

        Args:
            x (ArrayLike): N-dimensional value(s) to evaluate.
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
            x_phys (ArrayLike): N-dimensional value(s) in physical space.
            **kwargs: Additional keyword arguments.

        Returns:
            NDArrayLike: Probability density in physical space.
        """
        if self.forward_transform_nd is None or self.log_jacobian_nd is None:
            return self._to_device(self.pdf(x_phys, **kwargs))

        x_samp = self.forward_transform_nd(x_phys, **kwargs)
        pdf_samp = self.pdf(x_samp, **kwargs)
        log_det_J = self.log_jacobian_nd(x_phys, **kwargs)
        
        output = pdf_samp * self.xp.exp(log_det_J)
        return self._to_device(output)

    def cdf(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """CDF numerical integration for N-dimensional JointPriors.

        Raises:
            NotImplementedError: Mathematically undefined for general N-dimensions.
        """
        raise NotImplementedError(
            "Generic CDF numerical integration is mathematically undefined "
            "for N-dimensional JointPriors."
        )

    def is_in_prior_range(
        self, 
        x: ArrayLike,
        **kwargs
    ) -> NDArrayLike:
        """Boundary checking for N-dimensional bounds.

        Raises:
            NotImplementedError: Must be implemented by specific JointPrior subclass.
        """
        raise NotImplementedError(
            "Boundary checking must be implemented per JointPrior subclass."
        )

    @property
    def latex_label(self) -> tuple[str, ...]:
        return self.__latex_label

    @latex_label.setter
    def latex_label(self, latex_label: Sequence[str | None] | None = None):
        if latex_label is not None:
            if len(latex_label) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} latex_label, got {len(latex_label)}")
            
            labels = []
            for name, label in zip(self.names, latex_label):
                if label is not None:
                    labels.append(label)
                elif name in self._default_latex_labels:
                    labels.append(self._default_latex_labels[name])
                else:
                    labels.append(str(name))
            self.__latex_label = tuple(labels)
        else:
            self.__latex_label = tuple(
                self._default_latex_labels.get(name, str(name)) for name in self.names
            )

    @property
    def unit(self) -> tuple[str | None, ...]:
        return self.__unit

    @unit.setter
    def unit(self, unit: Sequence[str | None] | None = None):
        if unit is not None:
            if len(unit) != self.num_vars:
                raise ValueError(f"Expected {self.num_vars} units, got {len(unit)}")
            self.__unit = tuple(unit)
        else:
            self.__unit = tuple(None for _ in self.names)

    @property
    def latex_label_with_unit(self) -> tuple[str, ...]:
        """tuple: Strings of latex labels combined with units."""
        return tuple(
            f"{label} [{unit}]" if unit is not None else label
            for label, unit in zip(self.latex_label, self.unit)
        )
    
    
class MultivariateGaussian(JointPrior):
    """N-dimensional Multivariate Gaussian Prior.

    Args:
        names (list of str): Names associated with parameters.
        mu (ArrayLike): Mean vector.
        cov (ArrayLike): Covariance matrix.
        latex_labels (list of str, optional): Latex labels for parameters. (default: ``None``)
        units (list of str, optional): Strings describing parameter units. (default: ``None``)
        use_cupy (bool, optional): If ``True``, uses CuPy for GPU acceleration. (default: ``False``)
        return_gpu (bool, optional): If ``True``, returns GPU arrays. (default: ``False``)
    """

    def __init__(
        self,
        names: Sequence[str],
        mu: ArrayLike,
        cov: ArrayLike,
        latex_labels: Sequence[str | None] | None = None,
        units: Sequence[str | None] | None = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        super().__init__(
            names=names, 
            latex_labels=latex_labels,
            units=units,
            use_cupy=use_cupy, 
            return_gpu=return_gpu
        )

        self.mu = self.xp.asarray(mu, dtype=self.xp.float64)
        self.cov = self.xp.asarray(cov, dtype=self.xp.float64)

        if self.mu.shape != (self.num_vars,):
            raise ValueError(f"'mu' must have shape ({self.num_vars},).")
        if self.cov.shape != (self.num_vars, self.num_vars):
            raise ValueError(f"'cov' must have shape ({self.num_vars}, {self.num_vars}).")

        if not self.xp.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix must be symmetric.")

        try:
            self.prec = self.xp.linalg.inv(self.cov)
        except self.xp.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular and cannot be inverted.")

        sign, logdet = self.xp.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite.")
            
        self._log_norm = -0.5 * (self.num_vars * self.xp.log(2.0 * self.xp.pi) + logdet)
        self.cholesky_lower = self.xp.linalg.cholesky(self.cov)

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        z = self.xp.random.normal(0.0, 1.0, size=(*size, self.num_vars))
        samples = self.mu + self.xp.einsum("ij,...j->...i", self.cholesky_lower, z)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != self.num_vars:
            raise ValueError(
                f"Expected last dimension of x to be {self.num_vars}, got {x_arr.shape[-1]}"
            )

        diff = x_arr - self.mu
        quadratic_term = self.xp.einsum("...i,ij,...j->...", diff, self.prec, diff)
        out = -0.5 * quadratic_term + self._log_norm
        return self._to_device(out)