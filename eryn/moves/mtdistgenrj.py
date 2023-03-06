import numpy as np

from eryn.moves.multipletry import MultipleTryMoveRJ
from eryn.moves import DistributionGenerateRJ


class MTDistGenMoveRJ(MultipleTryMoveRJ, DistributionGenerateRJ):
    def __init__(self, dist, *args, **kwargs):
        """Perform a reversible-jump multiple try move based on a distribution.
    
        Distribution must be independent of the current point.
        
        This is effectively an example of the mutliple try class inheritance structure.
        
        """

        # TODO: change RJ min_k max_k to kwargs
        kwargs["rj"] = True
        MultipleTryMoveRJ.__init__(self, **kwargs)
        DistributionGenerateRJ.__init__(self, dist, *args, **kwargs)

        self.dist = dist

    def special_generate_logpdf(self, generated_coords):
        """Get logpdf of generated coordinates.
        
        Args:
            generated_coords (np.ndarray): Current coordinates of walkers. 
            
        Returns:
            np.ndarray: logpdf of generated points.
            """
        return self.dist[self.key_in].logpdf(generated_coords)

    def special_generate_func(
        self, coords, random, size=1, fill_tuple=None, fill_values=None
    ):
        """Generate samples and calculate the logpdf of their proposal function.
        
        Args:
            coords (np.ndarray): Current coordinates of walkers. 
            random (obj): Random generator.
            *args (tuple, optional): additional arguments passed by overwriting the 
                ``get_proposal`` function and passing ``args_generate`` keyword argument.
            size (int, optional): Number of tries to generate. 
            fill_tuple (tuple, optional): Length 2 tuple with the indexing of which values to fill
                when generating. Can be used for auxillary proposals or reverse RJ proposals. First index is the index into walkers and the second index is 
                the index into the number of tries. (default: ``None``)
            fill_values (np.ndarray): values to fill associated with ``fill_tuple``. Should 
                have size ``(len(fill_tuple[0]), ndim)``. (default: ``None``).
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the 
                ``get_proposal`` function and passing ``kwargs_generate`` keyword argument.

        Returns:
            tuple: (generated points, logpdf of generated points).
        
        """
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        generated_coords = self.dist[self.key_in].rvs(size=(nwalkers, size))

        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values

        generated_logpdf = self.special_generate_logpdf(
            generated_coords.reshape(nwalkers * size, -1)
        ).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords, inds_leaves_rj=None):
        """Setup coordinates for prior and Likelihood
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)

        Returns:
            dict: Coordinates for Likelihood and Prior.
        
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
        ] = generated_coords.reshape(-1, 3)
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
                inds_in_dict[key] = self.current_state.branches[key].inds.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-1:]
                )

        return coords_in_dict, inds_in_dict

    def special_like_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Calculate the Likelihood for sampled points.
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``. 
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)

        Returns:
            np.ndarray: Likelihood values with shape ``(generated_coords.shape[0], num_try).``
        
        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        ll = self.current_model.compute_log_like_fn(coords_in, inds=inds_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Calculate the Prior for sampled points.
        
        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``. 
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)
            
        Returns:
            np.ndarray: Prior values with shape ``(generated_coords.shape[0], num_try).``
        
        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        lp = self.current_model.compute_log_prior_fn(coords_in, inds=inds_in)
        return lp.reshape(-1, self.num_try)

