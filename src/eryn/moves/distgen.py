# -*- coding: utf-8 -*-
"""Independence Metropolis-Hastings move drawing from a fixed distribution."""

import numpy as np

from ..prior import ProbDistContainer
from .mh import MHMove

__all__ = ["DistributionGenerate"]


class DistributionGenerate(MHMove):
    """Independence proposal: redraw every selected leaf from a fixed distribution.

    Each alive leaf of a branch is replaced by a fresh draw from
    ``generate_dist[branch]``, independent of the current point; the log factor is
    ``log q(old) - log q(new)``. With the prior as ``generate_dist`` this is a
    draw-from-the-prior move.

    Args:
        generate_dist (dict): ``{branch_name: ProbDistContainer}``; each must be an
            :class:`eryn.priors.ProbDistContainer` (``logpdf`` and ``rvs``).
        *args: Forwarded to :class:`~eryn.moves.MHMove`.
        **kwargs: Forwarded to :class:`~eryn.moves.MHMove` / :class:`~eryn.moves.Move`.

    Raises:
        ValueError: ``generate_dist`` is not a dict, or a value is not a
            ``ProbDistContainer``.

    Warning:
        The factor accumulation ``factors[t, w] += ...`` uses NumPy fancy indexing, which
        does not accumulate repeated ``(t, w)`` pairs: with several alive leaves per
        walker only one leaf's term is kept.
    """

    def __init__(self, generate_dist, *args, **kwargs):
        if not isinstance(generate_dist, dict):
            raise ValueError(
                "When entering directly into the DistributionGenerate class, generate_dist must be a dictionary. The keys are branch names and the items are ProbDistContainer objects."
            )

        for key in generate_dist:
            if not isinstance(generate_dist[key], ProbDistContainer):
                raise ValueError(
                    "Distributions need to be eryn.prior.ProbDistContainer object."
                )
        self.generate_dist = generate_dist
        super().__init__(*args, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Redraw every selected leaf from ``generate_dist``.

        Args:
            branches_coords (dict): ``{branch_name: coords}``, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (object): Sampler random state (not used; draws come from
                ``generate_dist.rvs``).
            branches_inds (dict, optional): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)``; only these leaves are redrawn. Default
                ``None`` is meant to select every leaf, but that branch iterates the dict
                without ``.items()`` and fails; :meth:`~eryn.moves.MHMove.propose` always
                passes ``inds``.
            **kwargs: Ignored.

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: proposed coordinates with the shapes
            of ``branches_coords`` and log factors ``log q(old) - log q(new)`` of shape
            ``(ntemps, nwalkers)``.
        """

        # set up all initial holders
        q = {}
        factors = {}
        new_inds = {}
        if branches_inds is None:
            branches_inds = {
                name: np.ones(coords.shape[:-1], dtype=bool)
                for name, coords in branches_coords
            }

        # iterate through branches and propose new points where inds == True
        for i, (name, coords, inds) in enumerate(
            zip(
                branches_coords.keys(),
                branches_coords.values(),
                branches_inds.values(),
            )
        ):
            # copy over previous info
            ntemps, nwalkers, _, _ = coords.shape
            q[name] = coords.copy()
            new_inds[name] = inds.copy()

            if i == 0:
                factors = np.zeros((ntemps, nwalkers))

            # add coordinates for new leaves
            current_generate_dist = self.generate_dist[name]
            inds_here = np.where(inds == True)
            num_inds_change = len(inds_here[0])

            old_points = coords[inds_here]

            # old points so + log(qold)
            factors[inds_here[:2]] += +1 * current_generate_dist.logpdf(old_points)

            # Draw
            new_points = current_generate_dist.rvs(size=num_inds_change)

            # new point, so -log(qnew)
            factors[inds_here[:2]] += -1 * current_generate_dist.logpdf(new_points)

            q[name][inds_here] = new_points

        return q, factors
