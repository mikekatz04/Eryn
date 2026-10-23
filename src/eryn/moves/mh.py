# -*- coding: utf-8 -*-
"""General Metropolis-Hastings proposal base class (:class:`MHMove`)."""

from copy import deepcopy

import numpy as np

from ..state import State
from .move import Move

__all__ = ["MHMove"]


class MHMove(Move):
    r"""A general Metropolis-Hastings proposal over the whole ensemble.

    Concrete moves provide ``get_proposal``; :meth:`propose` does the rest: for every Gibbs
    split, propose all walkers at once, score, accept per walker with
    ``factors + logP_new - logP_old > log(U)`` and merge. Stock subclasses:
    :class:`~eryn.moves.GaussianMove`, :class:`~eryn.moves.EigenAxisMove`,
    :class:`~eryn.moves.DistributionGenerate`, :class:`~eryn.moves.NUTSMove`.

    A subclass that has already evaluated the proposed points (for example a multiple-try
    move) can set ``self.mt_ll`` and ``self.mt_lp`` (shape ``(ntemps, nwalkers)``) inside
    ``get_proposal``; :meth:`propose` then uses them instead of calling the model, and
    stores no blobs.

    Args:
        **kwargs: Forwarded to :class:`~eryn.moves.Move` (``temperature_control``,
            ``periodic``, ``gibbs_sampling_setup``, ``skip_supp_names_update``, ...).

    Attributes:
        current_model (:class:`eryn.model.Model`): Model of the current call (set before
            ``get_proposal``, for subclasses).
        current_state (:class:`eryn.state.State`): State of the current call.
    """

    def __init__(self, **kwargs):

        Move.__init__(self, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Propose new coordinates for every walker (implemented by subclasses).

        Args:
            branches_coords (dict): ``{branch_name: coords}`` for the branches of this Gibbs
                split, shape ``(ntemps, nwalkers, nleaves_max, ndim)``.
            random (object): Sampler random state (``model.random``).
            branches_inds (dict, optional): ``{branch_name: inds}``, boolean
                ``(ntemps, nwalkers, nleaves_max)``: the leaves to move (alive leaves, further
                restricted by the Gibbs split). Default ``None``.
            **kwargs: :meth:`propose` also passes ``supps`` and ``branch_supps`` (deep copies
                of the state's supplementals).

        Returns:
            tuple[dict, np.ndarray]: ``(q, factors)``: proposed coordinates with the shapes
            of ``branches_coords`` and log proposal factors (``log q(x | y) - log q(y | x)``)
            of shape ``(ntemps, nwalkers)``.

        Raises:
            NotImplementedError: Always, in this base class.
        """

        raise NotImplementedError("The proposal must be implemented by " "subclasses")

    def setup(self, branches_coords):
        """Hook run at the start of every :meth:`propose` (no-op here).

        Args:
            branches_coords (dict): ``{branch_name: coords}`` of the current state, shape
                ``(ntemps, nwalkers, nleaves_max, ndim)``.
        """

    def propose(self, model, state):
        """Propose every walker once per Gibbs split and accept with Metropolis-Hastings.

        Skips a split with no alive selected leaf. Scores with
        ``model.compute_log_prior_fn(q, inds=...)`` and ``model.compute_log_like_fn`` (with
        the copied supplementals, which the likelihood may adjust in place) unless
        ``mt_ll`` / ``mt_lp`` are set; walkers with no selectable leaf are fixed by
        :meth:`~eryn.moves.Move.fix_logp_gibbs`. Ends with
        :meth:`~eryn.moves.tempering.TemperatureControl.temper_comps` when a
        ``temperature_control`` is set.

        Args:
            model (:class:`eryn.model.Model`): Sampler callables and ``random``.
            state (:class:`eryn.state.State`): Current state; updated in place.

        Returns:
            tuple[:class:`eryn.state.State`, np.ndarray]: ``(state, accepted)``;
            ``accepted`` is the boolean ``(ntemps, nwalkers)`` result of the LAST Gibbs split
            run (all splits are added to :attr:`accepted`).
        """

        self.setup(state.branches_coords)

        # get all branch names for gibbs setup
        all_branch_names = list(state.branches.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # in case there are no leaves yet
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # iterate through gibbs setup
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # setup supplemental information
            if not np.all(
                np.asarray(list(state.branches_supplemental.values())) == None
            ):
                new_branch_supps = deepcopy(state.branches_supplemental)
            else:
                new_branch_supps = None

            if state.supplemental is not None:
                new_supps = deepcopy(state.supplemental)
            else:
                new_supps = None

            # setup information according to gibbs info
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            # if no walkers are actually being proposed
            if not at_least_one_proposal:
                continue

            self.current_model = model
            self.current_state = state

            # Get the move-specific proposal.
            q, factors = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                branches_inds=inds_going_for_proposal,
                supps=new_supps,
                branch_supps=new_branch_supps,
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # order everything properly
            q, _, new_branch_supps = self.ensure_ordering(
                list(state.branches.keys()), q, state.branches_inds, new_branch_supps
            )

            # if not wrapping with mutliple try (normal route)
            if not hasattr(self, "mt_ll") or not hasattr(self, "mt_lp"):
                # Compute prior of the proposed position
                logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

                self.fix_logp_gibbs(
                    branch_names_run, inds_run, logp, state.branches_inds
                )

                # Compute the lnprobs of the proposed position.
                # Can adjust supplementals in place
                logl, new_blobs = model.compute_log_like_fn(
                    q,
                    inds=state.branches_inds,
                    logp=logp,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                )

            else:
                # if already computed in multiple try
                logl = self.mt_ll
                logp = self.mt_lp
                new_blobs = None

            # get log posterior
            logP = self.compute_log_posterior(logl, logp)

            # get previous information
            prev_logl = state.log_like

            prev_logp = state.log_prior

            # takes care of tempering
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # difference
            lnpdiff = factors + logP - prev_logP

            # draw against acceptance fraction
            accepted = lnpdiff > np.log(model.random.rand(ntemps, nwalkers))

            # Update the parameters
            new_state = State(
                q,
                log_like=logl,
                log_prior=logp,
                blobs=new_blobs,
                inds=state.branches_inds,
                supplemental=new_supps,
                branch_supplemental=new_branch_supps,
            )
            state = self.update(state, new_state, accepted)

            # add to move-specific accepted information
            self.accepted += accepted
            self.num_proposals += 1

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted
