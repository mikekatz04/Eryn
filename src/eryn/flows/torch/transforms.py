# src/eryn/flows/torch/transforms.py
"""Torch-backed data transforms for normalizing-flow proposals.

Requires the ``flow`` extra (``pip install eryn[flow]``), which pulls in
``torch`` and ``zuko``.

Public API
----------
WhiteningTransform
    Periodic-aware per-condition whitening preprocessor implementing the
    :class:`eryn.flows.DataTransform` ABC.

Internal helpers (public for testing / direct use)
---------------------------------------------------
unwrap, wrap
    Angle-range utilities.
LogTransform, CircularShiftTransform, LinearMatrixTransform, PartialTransform
    Composable :class:`torch.distributions.Transform` subclasses used by
    :class:`WhiteningTransform` internally.
"""
from __future__ import annotations

import numpy as np
import torch
from zuko.transforms import AffineTransform, ComposeTransform, LULinearTransform  # noqa: F401

from eryn.flows.transforms import DataTransform

__all__ = [
    "WhiteningTransform",
    "unwrap",
    "wrap",
    "LogTransform",
    "CircularShiftTransform",
    "LinearMatrixTransform",
    "PartialTransform",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def unwrap(x: torch.Tensor, left: float, right: float) -> torch.Tensor:
    """Unwrap a tensor of angles to the range ``[left, right)``.

    Parameters
    ----------
    x : torch.Tensor, shape (N,)
        Tensor of angles to unwrap.
    left : float
        Left boundary of the target range.
    right : float
        Right boundary of the target range.

    Returns
    -------
    torch.Tensor, shape (N,)
        Unwrapped angles in ``[left, right)``.
    """
    period = right - left
    mid = (left + right) / 2
    return ((x - mid) % period) + mid


def wrap(x: torch.Tensor, left: float, right: float) -> torch.Tensor:
    """Wrap a tensor of angles to the range ``[left, right)``.

    Parameters
    ----------
    x : torch.Tensor, shape (N,)
        Tensor of angles to wrap.
    left : float
        Left boundary of the target range.
    right : float
        Right boundary of the target range.

    Returns
    -------
    torch.Tensor, shape (N,)
        Wrapped angles in ``[left, right)``.
    """
    period = right - left
    return ((x - left) % period) + left


# ---------------------------------------------------------------------------
# Composable torch.distributions.Transform subclasses
# ---------------------------------------------------------------------------

class PartialTransform(torch.distributions.Transform):
    """Apply a transform to one or more specific dimension(s) only.

    Parameters
    ----------
    transform : torch.distributions.Transform
        The transform to apply to the selected dimensions.
    dimensions : list of int
        Indices of the dimensions to which ``transform`` is applied.
    """

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True

    def __init__(self, transform, dimensions: list[int]):
        super().__init__()
        self.transform = transform
        self.dimensions = dimensions

    def _call(self, x):
        y = x.clone()
        y[..., self.dimensions] = self.transform(x[..., self.dimensions])
        return y

    def _inverse(self, y):
        x = y.clone()
        x[..., self.dimensions] = self.transform.inv(y[..., self.dimensions])
        return x

    def log_abs_det_jacobian(self, x, y):
        # The wrapped transform acts elementwise on the selected dims, returning a
        # per-dimension log|det| of shape (..., len(dimensions)). Sum over those dims so
        # this contribution matches the (batch,) convention of the other transforms in the
        # ComposeTransform. Without the sum, a (N, k) term broadcasts against the (N,)
        # terms into an (N, N) Jacobian -> silently wrong log-prob (MCMC bias) for periodic dims.
        return self.transform.log_abs_det_jacobian(
            x[..., self.dimensions], y[..., self.dimensions]
        ).sum(dim=-1)


class LogTransform(torch.distributions.Transform):
    """Elementwise natural logarithm transform.

    Maps positive reals to reals: ``y = log(x)``.

    Parameters
    ----------
    None.
    """

    domain = torch.distributions.constraints.positive
    codomain = torch.distributions.constraints.real
    bijective = True

    def __init__(self):
        super().__init__()

    def _call(self, x):
        return torch.log(x)

    def _inverse(self, y):
        return torch.exp(y)

    def log_abs_det_jacobian(self, x, y):
        return -torch.log(x)


class CircularShiftTransform(torch.distributions.Transform):
    """Circular shift of a periodic variable to center the bulk at zero.

    Parameters
    ----------
    shift : torch.Tensor
        Scalar tensor; the circular mean of the component, in the same units
        as the input.
    period : float
        Full period of the circular variable (e.g. ``2 * pi``).
    """

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    bijective = True

    def __init__(self, shift: torch.Tensor, period: float):
        super().__init__()
        self.shift = shift
        self.period = period
        self.half_period = period / 2.0

    def _call(self, x):
        # Shift data relative to circular mean and recenter at 0
        return ((x - self.shift + self.half_period) % self.period) - self.half_period

    def _inverse(self, y):
        # Revert shift to original periodic bounds
        return (y + self.shift) % self.period

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


class LinearMatrixTransform(torch.distributions.Transform):
    """Invertible linear map ``y = x @ M``.

    Parameters
    ----------
    matrix : torch.Tensor, shape (d, d)
        Square invertible matrix defining the linear transformation.
    """

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True

    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix = matrix
        self.inv_matrix = torch.linalg.inv(matrix)
        self._log_det = torch.linalg.slogdet(matrix)[1]

    def _call(self, x):
        return x @ self.matrix

    def _inverse(self, y):
        return y @ self.inv_matrix

    def log_abs_det_jacobian(self, x, y):
        # Return a tensor of shape matching the batch dimensions
        return self._log_det.expand(x.shape[:-1])


# ---------------------------------------------------------------------------
# WhiteningTransform — DataTransform implementation
# ---------------------------------------------------------------------------

class WhiteningTransform(DataTransform):
    """Periodic-aware per-condition whitening preprocessor.

    Builds a per-condition (per-leaf) sequence of invertible transforms from
    training samples via :meth:`fit`.  Periodic components are circular-shifted
    to their circular mean before centering and block-diagonal whitening
    (Cholesky whitening for the non-periodic block, marginal-std scaling for
    the periodic block).

    The forward pass operates in float64 internally (CPU) and returns a
    float32 tensor; the log-absolute-determinant Jacobian is returned as a
    per-sample scalar of shape ``(N,)``.

    Parameters
    ----------
    ndim : int
        Number of input dimensions.
    periodic : dict of {int: (float, float)}, optional
        Mapping from dimension index to ``(left, right)`` boundary of that
        periodic component (e.g. ``{2: (0.0, 2 * pi)}``).  Default is ``None``
        (no periodic dimensions).

    Examples
    --------
    >>> import numpy as np
    >>> wt = WhiteningTransform(ndim=3, periodic={2: (0.0, 6.283185307179586)})
    >>> samples = np.random.default_rng(0).standard_normal((100, 3))
    >>> wt.fit(samples)
    >>> wt.is_fitted
    True
    >>> z = wt.forward(samples)
    >>> z.dtype
    torch.float32
    """

    def __init__(
        self,
        ndim: int,
        periodic: dict[int, tuple[float, float]] | None = None,
    ):
        self.ndim = ndim
        self.periodic = periodic or {}
        self._set_indices()
        self._transforms: dict[int, ComposeTransform] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_indices(self) -> None:
        """Compute and cache the periodic / non-periodic index lists."""
        self._periodic_indices: list[int] = list(self.periodic.keys())
        self._non_periodic_indices: list[int] = [
            i for i in range(self.ndim) if i not in self._periodic_indices
        ]

    @property
    def eps(self) -> float:
        """Regularization added to the covariance matrix for Cholesky stability.

        Returns
        -------
        float
            ``1e-8``.
        """
        return 1e-8

    # ------------------------------------------------------------------
    # Public index properties
    # ------------------------------------------------------------------

    @property
    def periodic_indices(self) -> list[int]:
        """Indices of the periodic components.

        Returns
        -------
        list of int
        """
        return self._periodic_indices

    @property
    def non_periodic_indices(self) -> list[int]:
        """Indices of the non-periodic components.

        Returns
        -------
        list of int
        """
        return self._non_periodic_indices

    @property
    def transforms(self) -> dict[int, ComposeTransform]:
        """Per-condition :class:`zuko.transforms.ComposeTransform` objects.

        Returns
        -------
        dict of {int: ComposeTransform}
            Keyed by condition id.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._transforms is None:
            raise RuntimeError(
                "WhiteningTransform has not been fitted yet.  Call fit(samples) first."
            )
        return self._transforms

    # ------------------------------------------------------------------
    # DataTransform ABC
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully.

        Returns
        -------
        bool
        """
        return self._transforms is not None

    def fit(self, samples: np.ndarray | dict[int, np.ndarray]) -> None:
        """Fit per-condition whitening transforms from training samples.

        Parameters
        ----------
        samples : np.ndarray, shape (N, ndim) or dict of {int: np.ndarray}
            Training samples.  When a plain array is provided it is treated as
            condition ``0``.  When a dict is provided, keys are condition ids
            and each value is an ``(N_i, ndim)`` array.

        Returns
        -------
        None
        """
        self._transforms = {}

        if not isinstance(samples, dict):
            samples = {0: samples}

        for condition, cond_samples in samples.items():
            transforms_list: list = []

            if len(self.periodic_indices) > 0:
                # For the periodic components, circular-shift so the bulk
                # does not straddle the period boundary.
                periodic_transform = self._build_periodic_transform(cond_samples)
                samples_unwrapped = periodic_transform(
                    torch.tensor(cond_samples, dtype=torch.float64)
                )
                transforms_list.append(periodic_transform)
            else:
                samples_unwrapped = torch.tensor(cond_samples, dtype=torch.float64)

            # 1. Center the data (mean = 0)
            mean = torch.mean(samples_unwrapped, dim=0)
            centering_transform = AffineTransform(
                (-mean).to(dtype=torch.float64),
                torch.ones_like(mean, dtype=torch.float64),
            )
            transforms_list.append(centering_transform)

            centered_samples = samples_unwrapped - mean

            # 2. Block-diagonal whitening: joint Cholesky for non-periodic
            # components, marginal-std scaling for periodic components.
            # Periodic posteriors are often bimodal (due to symmetries), so
            # including them in the joint Cholesky inflates the conditional
            # variance of the periodic block, producing large (~±10) latent
            # values.  The flow learns any remaining cross-correlations.
            matrix = self._build_whitening_matrix(centered_samples)
            matrix_transform = LinearMatrixTransform(matrix)
            transforms_list.append(matrix_transform)

            self._transforms[condition] = ComposeTransform(transforms_list)

    def forward(self, x: np.ndarray, condition: int = 0) -> torch.Tensor:
        """Map coords-space samples to flow-space latents.

        Operates in float64 internally and returns a float32 tensor.

        Parameters
        ----------
        x : np.ndarray, shape (N, ndim)
            Samples in coords space.
        condition : int, optional
            Condition id selecting per-condition whitening statistics.
            Default is ``0``.

        Returns
        -------
        z : torch.Tensor, shape (N, ndim), dtype float32
            Whitened latent samples on CPU.

        Raises
        ------
        ValueError
            If ``condition`` has no fitted transform.
        """
        if condition not in self.transforms:
            raise ValueError(
                f"Condition {condition} not found in transforms.  "
                f"Available conditions: {list(self.transforms.keys())}"
            )
        return self.transforms[condition](
            torch.tensor(x, dtype=torch.float64)
        ).to(dtype=torch.float32)

    def inverse(self, z: torch.Tensor, condition: int = 0) -> np.ndarray:
        """Map flow-space latents back to coords space.

        Parameters
        ----------
        z : torch.Tensor, shape (N, ndim)
            Samples in flow (latent) space.
        condition : int, optional
            Condition id.  Default is ``0``.

        Returns
        -------
        x : np.ndarray, shape (N, ndim)
            Reconstructed samples in coords space.

        Raises
        ------
        ValueError
            If ``condition`` has no fitted transform.
        """
        if condition not in self.transforms:
            raise ValueError(
                f"Condition {condition} not found in transforms.  "
                f"Available conditions: {list(self.transforms.keys())}"
            )
        return self.transforms[condition].inv(z.to(dtype=torch.float64)).numpy()

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        condition: int = 0,
    ) -> torch.Tensor:
        """Log absolute determinant of the Jacobian of ``forward``.

        Returns a per-sample scalar of shape ``(N,)``.

        Parameters
        ----------
        x : torch.Tensor or array-like, shape (N, ndim)
            Input samples in coords space (pre-transform).
        z : torch.Tensor or array-like, shape (N, ndim)
            Output samples in flow space (post-transform).
        condition : int, optional
            Condition id.  Default is ``0``.

        Returns
        -------
        log_det : torch.Tensor, shape (N,)
            Per-sample ``log |det J_forward(x)|``.

        Raises
        ------
        ValueError
            If ``condition`` has no fitted transform.
        """
        if condition not in self.transforms:
            raise ValueError(
                f"Condition {condition} not found in transforms.  "
                f"Available conditions: {list(self.transforms.keys())}"
            )
        return self.transforms[condition].log_abs_det_jacobian(x, z)

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _build_whitening_matrix(self, centered_samples: torch.Tensor) -> torch.Tensor:
        """Build block-diagonal whitening matrix.

        Cholesky whitening for the non-periodic block; marginal-std scaling for
        the periodic block.

        Parameters
        ----------
        centered_samples : torch.Tensor, shape (N, ndim), dtype float64
            Mean-centred training samples.

        Returns
        -------
        torch.Tensor, shape (ndim, ndim), dtype float64
            Whitening matrix.
        """
        n = centered_samples.shape[1]
        np_idx = self.non_periodic_indices
        p_idx = self.periodic_indices

        matrix = torch.zeros(n, n, dtype=torch.float64)

        # Cholesky whitening for the non-periodic block
        if len(np_idx) > 0:
            np_samples = centered_samples[:, np_idx]
            cov_np = torch.cov(np_samples.T)
            cov_reg = cov_np + self.eps * torch.eye(len(np_idx), dtype=torch.float64)
            lower = torch.linalg.cholesky(cov_reg)
            inv_np = torch.linalg.inv(lower).T  # upper triangular whitening matrix
            for i, row in enumerate(np_idx):
                for j, col in enumerate(np_idx):
                    matrix[row, col] = inv_np[i, j]

        # Marginal std scaling for periodic components — avoids Cholesky
        # amplification when the posterior is bimodal across the period boundary
        for i in p_idx:
            std_i = centered_samples[:, i].std()
            matrix[i, i] = 1.0 / (std_i + self.eps)

        return matrix

    def _build_periodic_transform(self, samples: np.ndarray) -> ComposeTransform:
        """Build a circular-shift transform for all periodic components.

        Parameters
        ----------
        samples : np.ndarray, shape (N, ndim)
            Training samples for the current condition.

        Returns
        -------
        ComposeTransform
            Composed :class:`PartialTransform` wrappers, one per periodic
            component.
        """
        periodic_transforms = []
        for i, bounds in self.periodic.items():
            period = bounds[1] - bounds[0]
            # Compute circular mean to find the bulk
            # Scale to 2*pi for exact trig functions
            phase = torch.tensor(samples[:, i], dtype=torch.float64) * (
                2 * np.pi / period
            )
            mean_phase = torch.atan2(
                torch.mean(torch.sin(phase)), torch.mean(torch.cos(phase))
            )
            shift = mean_phase * (period / (2 * np.pi))

            circular_shift = CircularShiftTransform(shift, period)
            periodic_transforms.append(PartialTransform(circular_shift, dimensions=[i]))

        return ComposeTransform(periodic_transforms)
