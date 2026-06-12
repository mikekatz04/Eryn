# -*- coding: utf-8 -*-
"""Independent-Metropolis move backed by an eryn.flows Flow.

This module is intentionally torch-free at import time: ``eryn.flows.base``
(where ``FlowProposalDistribution`` lives) is pure NumPy, but
``eryn.flows.torch`` (where the concrete backends live) may import torch.
``FlowProposalDistribution`` is therefore imported lazily inside
``get_proposal`` rather than at module level, ensuring that
``import eryn.moves`` never triggers a torch import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mh import MHMove

if TYPE_CHECKING:
    # Only for static type-checkers; never executed at runtime.
    from eryn.flows.base import Flow, FlowProposalDistribution  # noqa: F401

__all__ = ["FlowMove"]


class FlowMove(MHMove):
    """Independent-Metropolis proposal drawing from a conditional normalizing flow.

    Proposes new coordinates for a single named branch by sampling from a
    :class:`eryn.flows.Flow` conditioned on :attr:`active_condition`.  The
    Hastings factor convention matches :class:`eryn.moves.DistributionGenerate`::

        factors += +log q(x_old) - log q(x_new)

    A poor or stale flow only lowers acceptance — it never biases the posterior.
    Accumulation uses :func:`numpy.add.at` (not ``+=``) so that duplicate
    ``(temp, walker)`` indices from multi-leaf proposals sum instead of being
    silently overwritten by the last write.

    :meth:`Flow.sample_and_log_prob` is used for new points to avoid a second
    forward pass through the flow: the log-probability returned by sampling is
    reused directly as ``-log q(new)``.

    Parameters
    ----------
    flow : eryn.flows.Flow
        Fitted (or freshly initialised) conditional normalizing flow.  Must
        implement :meth:`~eryn.flows.Flow.log_prob`,
        :meth:`~eryn.flows.Flow.sample_and_log_prob`, and the rest of the
        :class:`eryn.flows.Flow` ABC.
    branch_name : str
        Name of the branch this move proposes for.  All other branches in
        ``branches_coords`` are copied unchanged.
    *args, **kwargs
        Passed through to :class:`eryn.moves.MHMove`.

    Attributes
    ----------
    active_condition : int
        Condition id passed to the flow on every ``get_proposal`` call.
        Readable and writable; the setter coerces the value to ``int``.

    Examples
    --------
    >>> move = FlowMove(my_flow, branch_name="x")
    >>> move.active_condition = 2   # switch to condition 2 before sampling
    """

    def __init__(self, flow, branch_name: str, *args, **kwargs):
        self.flow = flow
        self.branch_name = branch_name
        self._active_condition: int = 0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # active_condition property
    # ------------------------------------------------------------------

    @property
    def active_condition(self) -> int:
        """Condition id used when calling the flow."""
        return self._active_condition

    @active_condition.setter
    def active_condition(self, c: int) -> None:
        self._active_condition = int(c)

    # ------------------------------------------------------------------
    # MHMove interface
    # ------------------------------------------------------------------

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate independent-Metropolis proposals from the flow.

        Parameters
        ----------
        branches_coords : dict
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]`` with current
            walker coordinates.
        random : object
            Current random state (unused by this move, passed for API compatibility).
        branches_inds : dict, optional
            Keys are branch names; values are
            ``np.ndarray[ntemps, nwalkers, nleaves_max]`` booleans indicating
            which leaves are active.  Defaults to all-``True`` when ``None``.
        **kwargs
            Accepted and ignored for compatibility.

        Returns
        -------
        q : dict
            Proposed coordinates; same structure as ``branches_coords``.
        factors : np.ndarray, shape (ntemps, nwalkers)
            Hastings log-factors: ``+log q(x_old) - log q(x_new)`` accumulated
            with :func:`numpy.add.at` over all active leaves of ``branch_name``.
        """
        # Lazy import: FlowProposalDistribution is pure NumPy (no torch), but
        # importing it here guarantees that even if eryn.flows.base were ever
        # changed to pull in a backend, this module stays torch-free at the
        # top level.
        from eryn.flows.base import FlowProposalDistribution

        dist = FlowProposalDistribution(self.flow, condition=self.active_condition)

        q = {}
        factors = None

        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords.items()
            }

        for i, (name, coords) in enumerate(branches_coords.items()):
            ntemps, nwalkers = coords.shape[:2]
            q[name] = coords.copy()
            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            if name != self.branch_name:
                continue  # FlowMove only proposes for its own branch

            where = np.where(branches_inds[name])
            old_points = coords[where]
            num = len(where[0])
            if num == 0:
                continue

            # + log q(old): np.add.at so multi-leaf duplicate (temp, walker) indices
            # accumulate instead of last-write-wins on repeated fancy-index +=.
            np.add.at(factors, where[:2], dist.logpdf(old_points))

            # Draw and - log q(new): sample_and_log_prob avoids a second forward
            # pass — the log-prob returned during sampling is reused directly.
            new_points, logq_new = self.flow.sample_and_log_prob(
                num, context=self.active_condition
            )
            np.add.at(factors, where[:2], -logq_new)

            q[name][where] = new_points

        return q, factors
