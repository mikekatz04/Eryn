# -*- coding: utf-8 -*-

import numpy as np

from .rj import ReversibleJumpMove
from ..prior import ProbDistContainer

__all__ = ["BasicSymmetricModelSwapRJMove"]


class BasicSymmetricModelSwapRJMove(ReversibleJumpMove):
    """
    Args:
        *args (tuple, optional): Additional arguments to pass to parent classes.
        **kwargs (dict, optional): Keyword arguments passed to parent classes.

    """

    def __init__(self, *args, **kwargs):

        super(BasicSymmetricModelSwapRJMove, self).__init__(*args, **kwargs)

    def get_proposal(
        self, all_coords, all_inds, min_k_all, max_k_all, random, **kwargs
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            min_k_all (list): Minimum values of leaf ount for each model. Must have same order as ``all_cords``. 
            max_k_all (list): Maximum values of leaf ount for each model. Must have same order as ``all_cords``. 
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity. 

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
        # prepare the output dictionaries
        q = {}
        new_inds = {}
        all_inds_for_change = {}

        # loop over the models included here
        assert len(all_coords.keys()) == len(max_k_all)

        ntemps, nwalkers, _ = all_inds[list(all_inds.keys())[0]].shape

        num_models = len(max_k_all)

        # switch is between 1 and num_models - 1
        switch = np.random.choice(np.arange(1, num_models), size=(ntemps, nwalkers))

        ndims = []
        total_leaves_check = np.zeros((ntemps, nwalkers, num_models), dtype=bool)
        for i, (name, inds) in enumerate(zip(all_inds.keys(), all_inds.values(),)):

            ndims.append(all_coords[name].shape[-1])
            if inds.shape[-1] > 1:
                raise ValueError(
                    "When using the basic model swap rj proposal, each model in the proposal can only have 1 leaf maximum."
                )
            total_leaves_check[:, :, i] = inds[:, :, 0]

        # confirm they all have the same dimension
        assert np.all(np.asarray(ndims) == ndims[0])

        ndim = ndims[0]

        if np.any(total_leaves_check.sum(axis=-1) > 1):
            raise ValueError(
                "When using the basic model swap rj proposal, only one model can be highlighted."
            )

        old_leaves_info = np.where(total_leaves_check)
        old_leaves_inds_highlight = old_leaves_info[-1]
        new_leaves_inds_highlight = (
            old_leaves_inds_highlight + switch.flatten()
        ) % num_models
        new_leaves_info = old_leaves_info[:-1] + (new_leaves_inds_highlight,)

        # get old coords
        transfer_coords = np.zeros((ntemps, nwalkers, ndim))
        for i, (name, coords, inds) in enumerate(
            zip(all_inds.keys(), all_coords.values(), all_inds.values(),)
        ):
            coords_trans = np.where(old_leaves_inds_highlight == i)
            transfer_coords[
                (old_leaves_info[0][coords_trans], old_leaves_info[1][coords_trans])
            ] = coords[
                (
                    old_leaves_info[0][coords_trans],
                    old_leaves_info[1][coords_trans],
                    np.zeros_like(old_leaves_info[0][coords_trans]),
                )
            ]

        for i, (name, coords, inds) in enumerate(
            zip(all_inds.keys(), all_coords.values(), all_inds.values(),)
        ):
            new_inds[name] = np.zeros_like(inds, dtype=bool)
            q[name] = np.zeros_like(coords)

            change = np.where(new_leaves_inds_highlight == i)
            new_inds[name][
                (
                    new_leaves_info[0][change],
                    new_leaves_info[1][change],
                    np.zeros_like(new_leaves_info[0][change]),
                )
            ] = True

            q[name][
                (
                    new_leaves_info[0][change],
                    new_leaves_info[1][change],
                    np.zeros_like(new_leaves_info[0][change]),
                )
            ] = transfer_coords[
                (new_leaves_info[0][change], new_leaves_info[1][change])
            ]

        # assumes symmetric
        factors = np.zeros((ntemps, nwalkers))

        return q, new_inds, factors
