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
        return self.dist["gauss"].logpdf(generated_coords)

    def special_generate_func(self, coords, random, size=1, fill_tuple=None, fill_values=None):
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        generated_coords = self.dist["gauss"].rvs(size=(nwalkers, size))

        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values
            
        generated_logpdf = self.special_generate_logpdf(generated_coords.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords, inds_leaves_rj=None):
        coords_in = np.repeat(self.current_state.branches["gauss"].coords.reshape((1, -1,) + self.current_state.branches["gauss"].coords.shape[-2:]), self.num_try, axis=1)
        coords_in[(np.zeros(coords_in.shape[0], dtype=int), np.arange(coords_in.shape[1]), np.repeat(inds_leaves_rj, self.num_try))] = generated_coords.reshape(-1, 3)
        inds_in = np.repeat(self.current_state.branches["gauss"].inds.reshape((1, -1,) + self.current_state.branches["gauss"].inds.shape[-1:]), self.num_try, axis=1)
        inds_in[(np.zeros(coords_in.shape[0], dtype=int), np.arange(inds_in.shape[1]), np.repeat(inds_leaves_rj, self.num_try))] = True

        return coords_in, inds_in

    def special_like_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        coords_in, inds_in = self.set_coords_and_inds(generated_coords, inds_leaves_rj=inds_leaves_rj)
        ll = self.current_model.compute_log_like_fn({"gauss": coords_in}, inds={"gauss": inds_in})[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        coords_in, inds_in = self.set_coords_and_inds(generated_coords, inds_leaves_rj=inds_leaves_rj)
        lp = self.current_model.compute_log_prior_fn({"gauss": coords_in}, inds={"gauss": inds_in})
        return lp.reshape(-1, self.num_try)


