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
LogTransform, CircularShiftTransform, LinearMatrixTransform, PartialTransform
    Composable :class:`torch.distributions.Transform` subclasses used by
    :class:`WhiteningTransform` internally.
"""
from __future__ import annotations

import numpy as np
import torch
from zuko.transforms import AffineTransform, ComposeTransform

from eryn.flows.transforms import DataTransform

__all__ = [
    "WhiteningTransform",
    "LogTransform",
    "CircularShiftTransform",
    "LinearMatrixTransform",
    "PartialTransform",
]


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
        Scalar tensor; the angle mapped to zero, in the same units as the
        input.  The wrap cut (the discontinuity of the forward map) sits at
        ``shift ± period / 2``.
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
        # Forward (coords -> recentered): wrap (x - shift) into the half-open
        # fundamental window [-T/2, T/2).  This is a true bijection on the
        # circle and is robust to ANY real input x (e.g. an MH-proposed angle
        # anywhere on the line), which is why the modulo must stay: log_prob is
        # called on external coords that need canonicalising into the bulk.
        return ((x - self.shift + self.half_period) % self.period) - self.half_period

    def _inverse(self, y):
        # Inverse (recentered -> coords): undo the shift and canonicalise into
        # the original periodic bounds [0, T).  Round-trip identities:
        #   * _inverse(_call(x)) == x  (mod T)  for every real x.
        #   * _call(_inverse(y)) == wrap_{[-T/2, T/2)}(y).  This equals y iff
        #     y already lies in [-T/2, T/2); for y outside that window the
        #     forward modulo necessarily folds it back by +/- T.
        #
        # The second identity is the crux of the periodic sample/log_prob
        # contract.  After the downstream affine whitening the flow models the
        # periodic latent as an UNBOUNDED real coordinate, so it places a small
        # amount of mass outside scale * [-T/2, T/2).  For such a draw the
        # coords-space density is the WRAPPED density (a sum over the latent
        # aliases z + k * scale * T); the single-image value reported by
        # rsample_and_log_prob can differ from log_prob(inverse(z)) by up to a
        # few nats at the wrap boundary.  No choice of output window for this
        # inverse removes that gap (the forward modulo is the obstruction), and
        # dropping the modulo here would break the periodicity of log_prob and
        # bias the ConditionalFlowMove Hastings factor far more severely -- so the canonical
        # [0, T) representative is kept deliberately.  See
        # test_circular_shift_round_trip_consistency for the pinned invariants.
        return (y + self.shift) % self.period

    def log_abs_det_jacobian(self, x, y):
        # Circular shift + wrap is volume-preserving: unit Jacobian, log-det 0.
        # Returned per-element (shape of x); PartialTransform sums over the
        # selected dims to keep the (N,) convention of the composed transform.
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
    so that the wrap cut lands in a low-density region (see ``periodic_cut``)
    before centering and block-diagonal whitening (Cholesky whitening for the
    non-periodic block, marginal-std scaling for the periodic block).

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
    shared : bool, optional
        Whitening granularity.  Default ``False`` fits **one map per condition**
        (per leaf): each condition is centered on its own mean and scaled by its
        own covariance — the right choice when the conditions are different
        sources in the same parameter space (different means) and the set of
        conditions is **fixed** (e.g. non-RJ runs), so every condition is present
        by fit time.  ``True`` fits a **single map on the pooled samples** of all
        conditions and applies it regardless of the ``condition`` argument; the
        flow's conditioning (not the whitening) then carries per-condition
        identity.  Shared mode does not center each condition, but it is robust
        when a condition is first seen *after* fitting (e.g. an RJ run where
        leaves are added/removed) — an unseen condition reuses the shared map
        instead of raising.
    periodic_cut : str, optional
        Where :meth:`fit` places the wrap cut (the discontinuity of the
        circular shift) of each periodic component.

        * ``"antimode"`` (default): at the minimum of a smoothed circular
          histogram of the training samples — the empirically emptiest
          region.  For a unimodal angle this coincides with the antipode of
          the bulk (the legacy behaviour); for **multimodal** angles (e.g.
          antipodal sky-position modes) it lands between the modes, whereas
          the circular mean of antipodal modes is noise-dominated and can put
          the cut *on* a mode, splitting it across the latent boundary.
        * ``"circular_mean"``: legacy behaviour — the cut sits at the
          antipode of the circular mean of the samples.

    Notes
    -----
    Under a fit-once-then-freeze trainer (the executor's lazy fit), ``shared=False``
    requires every condition to be present in the first ``fit`` call: a condition
    that first appears in a later frozen round has no map and will raise.  This
    holds automatically for fixed leaves; use ``shared=True`` when the active set
    of conditions can change over the run.

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
        shared: bool = False,
        periodic_cut: str = "antimode",
    ):
        self.ndim = ndim
        self.periodic = periodic or {}
        self.shared = bool(shared)
        if periodic_cut not in ("antimode", "circular_mean"):
            raise ValueError(
                f"periodic_cut must be 'antimode' or 'circular_mean', "
                f"got {periodic_cut!r}."
            )
        self.periodic_cut = periodic_cut
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
        """Fit the whitening transform(s) from training samples.

        Parameters
        ----------
        samples : np.ndarray, shape (N, ndim) or dict of {int: np.ndarray}
            Training samples.  When a plain array is provided it is treated as
            condition ``0``.  When a dict is provided, keys are condition ids
            and each value is an ``(N_i, ndim)`` array.

        Returns
        -------
        None

        Notes
        -----
        With ``shared=True`` all conditions' samples are pooled into a single
        whitening map applied regardless of the ``condition`` argument — robust
        when a condition (e.g. a leaf) appears only after fitting.  With
        ``shared=False`` (default) one map is fit per condition.
        """
        if self.shared:
            if isinstance(samples, dict):
                pooled = np.concatenate(
                    [np.asarray(v) for v in samples.values()], axis=0
                )
            else:
                pooled = np.asarray(samples)
            # one shared map, stored under key 0; _transform_for ignores condition
            self._transforms = {0: self._fit_one(pooled)}
            return

        self._transforms = {}
        if not isinstance(samples, dict):
            samples = {0: samples}
        for condition, cond_samples in samples.items():
            self._transforms[condition] = self._fit_one(cond_samples)

    def _fit_one(self, cond_samples: np.ndarray) -> ComposeTransform:
        """Build one whitening ``ComposeTransform`` from a sample array."""
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

        return ComposeTransform(transforms_list)

    def _transform_for(self, condition: int) -> ComposeTransform:
        """Return the fitted transform serving ``condition``.

        In shared mode the single pooled map serves every condition (so an
        unseen condition never raises).  Otherwise the per-condition map is
        looked up and a missing condition raises ``ValueError``.
        """
        if self._transforms is None:
            raise RuntimeError(
                "WhiteningTransform has not been fitted yet.  Call fit(samples) first."
            )
        if self.shared:
            return self._transforms[0]
        if condition not in self._transforms:
            raise ValueError(
                f"Condition {condition} not found in transforms.  "
                f"Available conditions: {list(self._transforms.keys())}"
            )
        return self._transforms[condition]

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
        return self._transform_for(condition)(
            torch.as_tensor(x, dtype=torch.float64)
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
        return self._transform_for(condition).inv(
            torch.as_tensor(z, dtype=torch.float64)
        ).numpy()

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
        x = torch.as_tensor(x, dtype=torch.float64)
        z = torch.as_tensor(z, dtype=torch.float64)
        return self._transform_for(condition).log_abs_det_jacobian(x, z)

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

    @staticmethod
    def _antimode_cut_shift(
        values: np.ndarray, lo: float, period: float, nbins: int = 64
    ) -> float:
        """Return the circular shift placing the wrap cut at the empirical antimode.

        A circular histogram of the samples is smoothed with a short
        triangular kernel (so a single noisy empty bin inside a mode cannot
        win) and the cut is placed at the midpoint of the **longest circular
        run of minimal-count bins** — the middle of the emptiest arc.  (A bare
        argmin would pick the first empty bin, i.e. the *edge* of a wide empty
        arc; the midpoint reproduces the antipode for a unimodal angle.)  The
        returned shift is the angle mapped to zero by
        :class:`CircularShiftTransform`, whose cut sits at
        ``shift + period / 2``.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Samples of one periodic component (any real values; wrapped
            internally).
        lo : float
            Left boundary of the periodic component.
        period : float
            Full period of the component.
        nbins : int, optional
            Maximum number of histogram bins; reduced for small sample
            counts.  Default is ``64``.

        Returns
        -------
        float
            Shift, in the same units as the input.
        """
        v = (np.asarray(values, dtype=np.float64) - lo) % period
        nbins = int(min(nbins, max(8, v.size // 10)))
        counts, edges = np.histogram(v, bins=nbins, range=(0.0, period))
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel /= kernel.sum()
        smooth = sum(
            w * np.roll(counts.astype(np.float64), k)
            for k, w in zip(range(-2, 3), kernel)
        )

        # Midpoint of the longest circular run of minimal bins.  The doubled
        # array makes wrap-around runs contiguous; runs are capped at nbins
        # and must start in the first copy so no run is counted twice.
        is_min = smooth <= smooth.min() + 1e-12
        best_idx = int(np.argmin(smooth))
        best_len = 0
        run = 0
        for j, flag in enumerate(np.concatenate([is_min, is_min])):
            if not flag:
                run = 0
                continue
            run += 1
            start = j - run + 1
            length = min(run, nbins)
            if start < nbins and length > best_len:
                best_len = length
                best_idx = (start + (length - 1) // 2) % nbins

        cut = lo + edges[best_idx] + 0.5 * period / nbins
        return cut - 0.5 * period

    def _build_periodic_transform(self, samples: np.ndarray) -> ComposeTransform:
        """Build a circular-shift transform for all periodic components.

        The shift determines where the wrap cut lands; see the class-level
        ``periodic_cut`` parameter for the two placement strategies.

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
        # getattr: transforms pickled before periodic_cut existed lack the
        # attribute; they only reach this method if re-fit, so default them
        # to the current default rather than crashing.
        cut_mode = getattr(self, "periodic_cut", "antimode")

        periodic_transforms = []
        for i, bounds in self.periodic.items():
            period = bounds[1] - bounds[0]

            if cut_mode == "antimode":
                shift = torch.tensor(
                    self._antimode_cut_shift(samples[:, i], bounds[0], period),
                    dtype=torch.float64,
                )
            else:
                # Legacy: circular mean of the bulk (cut at its antipode).
                # Scale to 2*pi for exact trig functions.
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
