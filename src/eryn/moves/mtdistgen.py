"""Multiple-try independence move drawing tries from a fixed distribution."""
import numpy as np

from eryn.moves import MHMove
from eryn.moves.multipletry import MultipleTryMove


class MTDistGenMove(MultipleTryMove, MHMove):
    """Multiple-try Metropolis-Hastings move with tries drawn from a fixed distribution.

    The distribution must not depend on the current point, so construct the move with
    ``independent=True`` (the non-independent auxiliary path of
    :meth:`~eryn.moves.MultipleTryMove.get_mt_proposal` is not usable with this class).
    Works on a single branch with one leaf per walker and needs a
    ``temperature_control`` (the sampler attaches one). Also serves as the reference
    example of the multiple-try inheritance structure.

    Args:
        generate_dist (:class:`eryn.priors.ProbDistContainer`): ONE container (not a dict)
            with ``logpdf`` and ``rvs`` over the branch's parameters, e.g. the branch prior.
        **kwargs: Passed to both :class:`~eryn.moves.MultipleTryMove` (``num_try``,
            ``independent``, ``symmetric``) and :class:`~eryn.moves.MHMove` /
            :class:`~eryn.moves.Move`.

    Example:
        >>> from eryn.priors import ProbDistContainer, UniformDistribution
        >>> from eryn.moves import MTDistGenMove
        >>> priors = ProbDistContainer(
        ...     {0: UniformDistribution(-5.0, 5.0), 1: UniformDistribution(-5.0, 5.0)}
        ... )
        >>> move = MTDistGenMove(priors, num_try=25, independent=True)
        >>> move.num_try
        25
    """

    def __init__(self, generate_dist, **kwargs):

        MultipleTryMove.__init__(self, **kwargs)
        MHMove.__init__(self, **kwargs)

        self.generate_dist = generate_dist

    def special_generate_logpdf(self, generated_coords):
        """Log density of ``generate_dist`` at the given points.

        Args:
            generated_coords (np.ndarray): Points, shape ``(n, ndim)``.

        Returns:
            np.ndarray: Log density, shape ``(n,)``.
        """
        return self.generate_dist.logpdf(generated_coords)

    def special_generate_func(
        self, coords, random, size=1, fill_tuple=None, fill_values=None, **kwargs
    ):
        """Draw ``size`` tries per walker from ``generate_dist``.

        Args:
            coords (np.ndarray): Current coordinates, shape ``(nwalkers, ndim)`` (only the
                walker count is used).
            random (object): Not used; draws come from ``generate_dist.rvs``.
            size (int, optional): Tries per walker. Default ``1``.
            fill_tuple (tuple, optional): ``(walker_idx, try_idx)`` entries to overwrite with
                ``fill_values`` after drawing. Default ``None``.
            fill_values (np.ndarray, optional): Values for ``fill_tuple``, shape
                ``(len(fill_tuple[0]), ndim)``. Default ``None``.
            **kwargs: Ignored.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tries of shape ``(nwalkers, size, ndim)`` and their
            log density, shape ``(nwalkers, size)``.

        Raises:
            ValueError: ``size`` is not an ``int``.
        """
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        # generate coordinates
        generated_coords = self.generate_dist.rvs(size=(nwalkers, size))

        # fill special coordinates
        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values

        # get logpdf
        generated_logpdf = self.special_generate_logpdf(
            generated_coords.reshape(nwalkers * size, -1)
        ).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords):
        """Pack the tries into a sampler-shaped coordinate dict for the model callables.

        The tries become ``(1, nwalkers * num_try, 1, ndim)`` for ``self.key_in``; other
        branches are reshaped to ``(1, -1, nleaves_max, ndim)``.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.

        Returns:
            dict: ``{branch_name: coords}`` to pass to ``compute_log_like_fn`` /
            ``compute_log_prior_fn``.
        """
        ndim = self.current_state.branches[self.key_in].shape[-1]

        coords_in_dict = {}
        for key in self.current_state.branches.keys():
            if key == self.key_in:
                coords_in_dict[key] = generated_coords.reshape(-1, 1, ndim)[None, :]

            else:
                coords_in_dict[key] = self.current_state.branches[key].coords.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:]
                )

        return coords_in_dict

    def special_like_func(self, generated_coords, **kwargs):
        """Log-likelihood of every try through ``self.current_model.compute_log_like_fn``.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            **kwargs: Ignored.

        Returns:
            np.ndarray: Log-likelihood, shape ``(nwalkers, num_try)``.
        """
        coords_in = self.set_coords_and_inds(generated_coords)
        ll = self.current_model.compute_log_like_fn(coords_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, **kwargs):
        """Log prior of every try through ``self.current_model.compute_log_prior_fn``.

        Args:
            generated_coords (np.ndarray): Tries, shape ``(nwalkers, num_try, ndim)``.
            **kwargs: Ignored.

        Returns:
            np.ndarray: Log prior, shape ``(nwalkers, num_try)``.
        """
        coords_in = self.set_coords_and_inds(generated_coords)
        lp = self.current_model.compute_log_prior_fn(coords_in)
        return lp.reshape(-1, self.num_try)

