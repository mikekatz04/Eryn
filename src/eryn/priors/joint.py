"""N-dimensional (joint) priors.

:class:`JointPrior` is the base for priors over several parameters at once (per-parameter
metadata plus optional N-D transforms between the sampling and physical spaces);
:class:`MultivariateGaussian` is the implemented example. Use one in a
:class:`~eryn.priors.probdist.ProbDistContainer` under a tuple key, e.g.
``{("a", "b"): MultivariateGaussian(...)}``: the container then passes it arrays of shape
``(N, k)`` with ``k`` the tuple length.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from ..utils.utility import NDArrayLike

from .base import AbstractPrior


class JointPrior(AbstractPrior):
    """Base class for N-dimensional priors.

    A subclass implements ``rvs(size=..., **kwargs)`` returning shape
    ``size + (num_vars,)`` and ``logpdf(x, **kwargs)`` taking ``x`` of shape
    ``(..., num_vars)`` and returning shape ``x.shape[:-1]``, both in the SAMPLING space.
    The ``*_physical`` methods map through ``forward_transform_nd`` /
    ``inverse_transform_nd`` / ``log_jacobian_nd`` the same way as
    :class:`~eryn.priors.base.Prior` does in 1-D. The column layout inside a
    :class:`~eryn.priors.probdist.ProbDistContainer` comes from the container key, not from
    ``names``.

    Args:
        names: Parameter names; must be non-empty. Sets ``num_vars``.
        names_phys: Physical parameter names; a ``None`` entry falls back to the matching
            ``names`` entry. Default ``None`` (use ``names``).
        latex_labels: LaTeX labels; a ``None`` entry falls back to
            ``_default_latex_labels[name]`` and then ``str(name)``. Default ``None``.
        units: Unit strings, one per parameter. Default ``None`` (all ``None``).
        boundaries: Boundary-condition labels (``None``, ``"periodic"``, ``"reflective"``),
            one per parameter; metadata only. Default ``None`` (all ``None``).
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.
        forward_transform_nd: Callable ``f(x_phys, **kwargs) -> x_samp`` on
            ``(..., num_vars)`` arrays. Default ``None``.
        inverse_transform_nd: Callable ``g(x_samp, **kwargs) -> x_phys``. Default ``None``.
        log_jacobian_nd: Callable ``h(x_phys, **kwargs)`` returning
            ``log|det(d x_samp / d x_phys)|``. Default ``None``.

    Attributes:
        names (tuple[str, ...]): Parameter names.
        num_vars (int): Number of parameters.
        names_phys (tuple[str, ...]): Physical parameter names.
        boundaries (tuple): Boundary-condition labels.
        forward_transform_nd (callable or None): Physical -> sampling map.
        inverse_transform_nd (callable or None): Sampling -> physical map.
        log_jacobian_nd (callable or None): Log-determinant of the forward Jacobian.

    Raises:
        ValueError: If ``names`` is empty, a metadata sequence has the wrong length, or a
            boundary label is unsupported.
        ImportError: If ``use_cupy`` is ``True`` and CuPy is not installed.
    """

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
        """Build the joint prior (arguments are documented on the class)."""
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
        """Draw samples in the SAMPLING space (override in subclasses).

        The base implementation only validates ``size`` and then raises.

        Args:
            size: Leading output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Conditional variables (and any extra options) for subclasses.

        Returns:
            NDArrayLike: Samples of shape ``size + (num_vars,)`` (in subclasses).

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
        """Draw samples in the PHYSICAL space: ``inverse_transform_nd(rvs(size))``.

        Without ``inverse_transform_nd`` this returns the :meth:`rvs` samples unchanged.

        Args:
            size: Leading output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Forwarded to :meth:`rvs` and ``inverse_transform_nd``.

        Returns:
            NDArrayLike: Physical-space samples of shape ``size + (num_vars,)``.
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
        """Joint log-density in the SAMPLING space (override in subclasses).

        Args:
            x: Values of shape ``(..., num_vars)``.
            **kwargs: Conditional variables (and any extra options) for subclasses.

        Returns:
            NDArrayLike: Log-density of shape ``x.shape[:-1]`` (in subclasses).

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("logpdf method is implemented in subclass")

    def logpdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Joint log-density in the PHYSICAL space.

        Computes ``logpdf(forward_transform_nd(x_phys)) + log_jacobian_nd(x_phys)``; falls
        back to ``logpdf(x_phys)`` when either piece is ``None``.

        Args:
            x_phys: Physical values of shape ``(..., num_vars)``.
            **kwargs: Forwarded to the transform, :meth:`logpdf` and the Jacobian.

        Returns:
            NDArrayLike: Log-density of shape ``x_phys.shape[:-1]``.
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
        """Joint density in the SAMPLING space: ``exp(logpdf(x))``.

        Args:
            x: Values of shape ``(..., num_vars)``.
            **kwargs: Forwarded to :meth:`logpdf`.

        Returns:
            NDArrayLike: Density of shape ``x.shape[:-1]``.
        """
        output = self.xp.exp(self.logpdf(x, **kwargs))
        return self._to_device(output)

    def pdf_physical(
        self, 
        x_phys: ArrayLike, 
        **kwargs
    ) -> NDArrayLike:
        """Joint density in the PHYSICAL space.

        Computes ``pdf(forward_transform_nd(x_phys)) * exp(log_jacobian_nd(x_phys))``; falls
        back to ``pdf(x_phys)`` when either piece is ``None``.

        Args:
            x_phys: Physical values of shape ``(..., num_vars)``.
            **kwargs: Forwarded to the transform, :meth:`pdf` and the Jacobian.

        Returns:
            NDArrayLike: Density of shape ``x_phys.shape[:-1]``.
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
        """Not available for joint priors.

        Args:
            x: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always; there is no generic N-D CDF.
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
        """Support check for N-D values (override in subclasses).

        Args:
            x: Values of shape ``(..., num_vars)``.
            **kwargs: Subclass options.

        Raises:
            NotImplementedError: Always, in the base class (and in
                :class:`MultivariateGaussian`, which does not override it).
        """
        raise NotImplementedError(
            "Boundary checking must be implemented per JointPrior subclass."
        )

    @property
    def latex_label(self) -> tuple[str, ...]:
        """Per-parameter LaTeX labels (``None`` entries resolved on assignment)."""
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
        """Per-parameter unit strings (``None`` where unset)."""
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
    """Multivariate Gaussian prior over ``len(names)`` parameters.

    At construction ``mu`` and ``cov`` are checked for shape, ``cov`` for symmetry,
    invertibility and a positive determinant, and then Cholesky-factorised; the precision
    matrix, log-normalisation and lower Cholesky factor are cached. No transforms
    (sampling and physical spaces coincide).

    Args:
        names: Parameter names; ``num_vars = len(names)``.
        mu: Mean vector, shape ``(num_vars,)``.
        cov: Covariance matrix, shape ``(num_vars, num_vars)``.
        latex_labels: LaTeX labels (see :class:`JointPrior`). Default ``None``.
        units: Unit strings. Default ``None``.
        use_cupy: Compute with CuPy. Default ``False``.
        return_gpu: Return CuPy arrays (when ``use_cupy``). Default ``False``.

    Attributes:
        mu (xp.ndarray): Mean, float64, shape ``(num_vars,)``.
        cov (xp.ndarray): Covariance, float64, shape ``(num_vars, num_vars)``.
        prec (xp.ndarray): Inverse covariance.
        cholesky_lower (xp.ndarray): Lower Cholesky factor of ``cov``.

    Raises:
        ValueError: If ``mu`` / ``cov`` have the wrong shape, ``cov`` is not symmetric, is
            singular, or has a non-positive determinant.
        numpy.linalg.LinAlgError: From the Cholesky factorisation when ``cov`` is not
            positive definite but its determinant is positive (an even number of negative
            eigenvalues passes the determinant test).

    Example:
        >>> import numpy as np
        >>> from eryn.priors import MultivariateGaussian, ProbDistContainer
        >>> mvn = MultivariateGaussian(["a", "b"], mu=[0.0, 1.0], cov=np.eye(2))
        >>> mvn.rvs(4).shape
        (4, 2)
        >>> priors = ProbDistContainer({("a", "b"): mvn})
        >>> priors.logpdf(priors.rvs(3)).shape
        (3,)
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
        """Draw ``mu + L z`` with ``z`` standard normal and ``L`` the Cholesky factor.

        Args:
            size: Leading output shape; an ``int`` means ``(size,)``. Default ``(1,)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Samples of shape ``size + (num_vars,)``.
        """
        if isinstance(size, int):
            size = (size,)

        z = self.xp.random.normal(0.0, 1.0, size=(*size, self.num_vars))
        samples = self.mu + self.xp.einsum("ij,...j->...i", self.cholesky_lower, z)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        """Normalised multivariate-normal log-density.

        Args:
            x: Values of shape ``(..., num_vars)``.
            **kwargs: Ignored.

        Returns:
            NDArrayLike: Log-density of shape ``x.shape[:-1]``.

        Raises:
            ValueError: If the last axis of ``x`` is not ``num_vars`` long.
        """
        x_arr = self.xp.asarray(x)
        
        if x_arr.shape[-1] != self.num_vars:
            raise ValueError(
                f"Expected last dimension of x to be {self.num_vars}, got {x_arr.shape[-1]}"
            )

        diff = x_arr - self.mu
        quadratic_term = self.xp.einsum("...i,ij,...j->...", diff, self.prec, diff)
        out = -0.5 * quadratic_term + self._log_norm
        return self._to_device(out)