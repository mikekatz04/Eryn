# -*- coding: utf-8 -*-
"""Gaussian random-walk Metropolis move (:class:`GaussianMove`)."""

import numpy as np

from .mh import MHMove

__all__ = ["GaussianMove"]


class GaussianMove(MHMove):
    """Metropolis step with a symmetric Gaussian random-walk proposal.

    Heavily based on the class of the same name in ``emcee``. Every alive leaf of a branch
    is perturbed independently with that branch's proposal; the proposal is symmetric, so
    the returned factors are zero. Values are wrapped when ``periodic`` is set.

    Args:
        cov_all (dict): ``{branch_name: cov}``. ``cov`` is a scalar variance (isotropic
            proposal), a length-``ndim`` vector of variances (axis-aligned) or an
            ``(ndim, ndim)`` covariance matrix (general).
        mode (str, optional): ``"vector"`` updates all parameters at once, ``"random"`` one
            randomly chosen parameter per leaf, ``"sequential"`` one parameter per call,
            cycling through them. A full covariance matrix only supports ``"vector"``.
            Default ``"vector"``.
        factor (float, optional): If given (``>= 1``), the step is scaled by
            ``exp(U(-log(factor), log(factor)))``, one draw per call. Default ``None``.
        **kwargs: Forwarded to :class:`~eryn.moves.MHMove` / :class:`~eryn.moves.Move`.

    Raises:
        ValueError: ``cov`` has an invalid shape, ``factor < 1``, or ``mode`` is not
            supported for that covariance form.

    Warning:
        The diagonal form (a 1D variance vector with more than one entry; unrelated to
        ``mode="vector"``) currently fails at construction: the helper computes
        ``np.linalg.cholesky`` of the 1D scale vector, which raises
        ``numpy.linalg.LinAlgError``. Pass ``np.diag(variances)`` instead.

    Example:
        >>> import numpy as np
        >>> from eryn.moves import GaussianMove
        >>> move = GaussianMove({"model_0": 0.01 * np.eye(2)})
        >>> sorted(move.all_proposal)
        ['model_0']
    """

    def __init__(self, cov_all, mode="vector", factor=None, **kwargs):
        self.all_proposal = {}
        for name, cov in cov_all.items():
            # Parse the proposal type.
            try:
                float(cov)

            except TypeError:
                cov = np.atleast_1d(cov)
                if len(cov.shape) == 1:
                    # A diagonal proposal was given.
                    ndim = len(cov)
                    proposal = _diagonal_proposal(np.sqrt(cov), factor, mode)

                elif len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]:
                    # The full, square covariance matrix was given.
                    ndim = cov.shape[0]
                    proposal = _proposal(cov, factor, mode)

                else:
                    raise ValueError("Invalid proposal scale dimensions")

            else:
                # This was a scalar proposal.
                ndim = None
                proposal = _isotropic_proposal(np.sqrt(cov), factor, mode)
            self.all_proposal[name] = proposal

        super().__init__(**kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Draw a Gaussian step for every selected leaf.

        Args:
            branches_coords (dict): ``{branch_name: coords}``, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (object): Random state with ``randn`` / ``randint`` / ``uniform`` /
                ``multivariate_normal`` (``model.random``).
            branches_inds (dict, optional): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)``; only these leaves move. ``None`` moves
                every leaf. Default ``None``.
            **kwargs: Ignored.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: proposed coordinates with the shapes
            of ``branches_coords`` and zeros of shape ``(ntemps, nwalkers)``.
        """

        # initialize ouput
        q = {}
        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, _ = proposal_fn(coords[inds_here], random)

            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        return q, np.zeros((ntemps, nwalkers))


class _isotropic_proposal(object):
    """Isotropic Gaussian step ``x0 + f * scale * N(0, 1)`` for rows ``(n, ndim)``.

    Called as ``proposal(x0, rng) -> (x_new, zeros(n))``; ``mode`` restricts the update to
    one random (``"random"``) or cycling (``"sequential"``) parameter per row.
    """

    allowed_modes = ["vector", "random", "sequential"]

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale
        
        if isinstance(scale, float):
            self.invscale = 1. / scale
        else:
            self.invscale = np.linalg.inv(np.linalg.cholesky(scale))

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                (f"'{mode}' is not a recognized mode. " f"Please select from: {self.allowed_modes}")
            )
        self.mode = mode

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _diagonal_proposal(_isotropic_proposal):
    """Axis-aligned Gaussian step with a per-parameter ``scale`` vector."""

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))


class _proposal(_isotropic_proposal):
    """Correlated Gaussian step with full covariance ``scale``; ``"vector"`` mode only."""

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * rng.multivariate_normal(
            np.zeros(len(self.scale)), self.scale, size=len(x0)
        )
