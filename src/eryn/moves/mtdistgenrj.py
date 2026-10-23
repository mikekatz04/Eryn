"""Multiple-try reversible-jump move drawing births from a fixed distribution."""
import numpy as np

from eryn.moves import DistributionGenerateRJ
from eryn.moves.multipletry import MultipleTryMoveRJ


class MTDistGenMoveRJ(MultipleTryMoveRJ, DistributionGenerateRJ):
    """Reversible-jump multiple-try move with births drawn from a fixed distribution.

    The birth/death choice comes from
    :meth:`~eryn.moves.DistributionGenerateRJ.get_model_change_proposal`; each birth draws
    ``num_try`` candidates from ``generate_dist[branch]`` (independent of the current
    point) and picks one by importance weight; each death is scored as the reverse of such
    a birth. ``rj=True`` is forced. Works on one branch per Gibbs split and needs a
    ``temperature_control`` (the sampler attaches one). Also serves as the reference
    example of the multiple-try RJ inheritance structure.

    Args:
        generate_dist (dict): ``{branch_name: ProbDistContainer}`` with ``logpdf`` and
            ``rvs`` (e.g. the priors).
        *args: Forwarded to :class:`~eryn.moves.DistributionGenerateRJ`.
        **kwargs: Passed to :class:`~eryn.moves.multipletry.MultipleTryMoveRJ`
            (``num_try``) and :class:`~eryn.moves.DistributionGenerateRJ` (``nleaves_max``
            and ``nleaves_min`` are required there).

    Example:
        .. code-block:: python

            from eryn.moves import MTDistGenMoveRJ

            rj_move = MTDistGenMoveRJ(
                priors,  # {"gauss": ProbDistContainer(...)}
                nleaves_max={"gauss": 8},
                nleaves_min={"gauss": 0},
                num_try=25,
            )
    """

    def __init__(self, generate_dist, *args, **kwargs):
        """Initialise both parents with ``rj=True``; see the class docstring."""
        kwargs["rj"] = True
        MultipleTryMoveRJ.__init__(self, **kwargs)
        DistributionGenerateRJ.__init__(self, generate_dist, *args, **kwargs)

        self.generate_dist = generate_dist

    def special_generate_logpdf(self, generated_coords):
        """Log density of ``generate_dist[self.key_in]`` at the given points.

        Args:
            generated_coords (np.ndarray): Points, shape ``(n, ndim)``.

        Returns:
            np.ndarray: Log density, shape ``(n,)``.
        """
        return self.generate_dist[self.key_in].logpdf(generated_coords)

    def special_generate_func(
        self, coords, random, size=1, fill_tuple=None, fill_values=None
    ):
        """Draw ``size`` tries per walker from ``generate_dist[self.key_in]``.

        Args:
            coords (np.ndarray): Current leaf coordinates, shape ``(nwalkers, ndim)`` (only
                the walker count is used).
            random (object): Not used; draws come from ``rvs``.
            size (int, optional): Tries per walker. Default ``1``.
            fill_tuple (tuple, optional): ``(walker_idx, try_idx)`` entries to overwrite with
                ``fill_values`` after drawing (the removed leaves of reverse proposals).
                Default ``None``.
            fill_values (np.ndarray, optional): Values for ``fill_tuple``, shape
                ``(len(fill_tuple[0]), ndim)``. Default ``None``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tries of shape ``(nwalkers, size, ndim)`` and their
            log density, shape ``(nwalkers, size)``.

        Raises:
            ValueError: ``size`` is not an ``int``.
        """
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        generated_coords = self.generate_dist[self.key_in].rvs(size=(nwalkers, size))

        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values

        generated_logpdf = self.special_generate_logpdf(
            generated_coords.reshape(nwalkers * size, -1)
        ).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords, inds_leaves_rj=None):
        """Build coordinate and ``inds`` dicts with each try inserted into its walker's slot.

        Every walker's current leaves are repeated ``num_try`` times along a flattened walker
        axis, the try is written into leaf ``inds_leaves_rj`` (switched on), and the other
        branches are tiled to match.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            inds_leaves_rj (np.ndarray): Leaf slot per walker, shape ``(nwalkers,)``.
                Default ``None`` (required in practice).

        Returns:
            tuple[dict, dict]: ``(coords, inds)`` with leading shape
            ``(1, nwalkers * num_try)``, for ``compute_log_like_fn`` / ``compute_log_prior_fn``.
        """
        coords_in = np.repeat(
            self.current_state.branches[self.key_in].coords.reshape(
                (1, -1,) + self.current_state.branches[self.key_in].coords.shape[-2:]
            ),
            self.num_try,
            axis=1,
        )
        coords_in[
            (
                np.zeros(coords_in.shape[0], dtype=int),
                np.arange(coords_in.shape[1]),
                np.repeat(inds_leaves_rj, self.num_try),
            )
        ] = generated_coords.reshape(-1, coords_in.shape[-1])
        inds_in = np.repeat(
            self.current_state.branches[self.key_in].inds.reshape(
                (1, -1,) + self.current_state.branches[self.key_in].inds.shape[-1:]
            ),
            self.num_try,
            axis=1,
        )
        inds_in[
            (
                np.zeros(coords_in.shape[0], dtype=int),
                np.arange(inds_in.shape[1]),
                np.repeat(inds_leaves_rj, self.num_try),
            )
        ] = True

        coords_in_dict = {}
        inds_in_dict = {}
        for key in self.current_state.branches.keys():
            if key == self.key_in:
                coords_in_dict[key] = coords_in
                inds_in_dict[key] = inds_in

            else:
                coords_in_dict[key] = self.current_state.branches[key].coords.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:]
                )
                # expand to express multiple tries
                coords_in_dict[key] = np.tile(
                    coords_in_dict[key], (1, self.num_try, 1)
                ).reshape(
                    coords_in_dict[key].shape[0],
                    coords_in_dict[key].shape[1] * self.num_try,
                    coords_in_dict[key].shape[2],
                    coords_in_dict[key].shape[3],
                )
                inds_in_dict[key] = self.current_state.branches[key].inds.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:-1]
                )
                # expand to express multiple tries
                inds_in_dict[key] = np.tile(
                    inds_in_dict[key], (1, self.num_try)
                ).reshape(
                    inds_in_dict[key].shape[0],
                    inds_in_dict[key].shape[1] * self.num_try,
                    inds_in_dict[key].shape[2],
                )

        return coords_in_dict, inds_in_dict

    def special_like_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Log-likelihood of every try through ``self.current_model.compute_log_like_fn``.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            inds_leaves_rj (np.ndarray): Leaf slot per walker, shape ``(nwalkers,)``.
                Default ``None`` (required in practice).
            **kwargs: Ignored.

        Returns:
            np.ndarray: Log-likelihood, shape ``(nwalkers, num_try)``.
        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        ll = self.current_model.compute_log_like_fn(coords_in, inds=inds_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Log prior of every try through ``self.current_model.compute_log_prior_fn``.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            inds_leaves_rj (np.ndarray): Leaf slot per walker, shape ``(nwalkers,)``.
                Default ``None`` (required in practice).
            **kwargs: Ignored.

        Returns:
            np.ndarray: Log prior, shape ``(nwalkers, num_try)``.
        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        lp = self.current_model.compute_log_prior_fn(coords_in, inds=inds_in)
        return lp.reshape(-1, self.num_try)

