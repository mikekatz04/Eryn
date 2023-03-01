import numpy as np

from eryn.moves.multipletry import MultipleTryMove
from eryn.moves import MHMove

class MTDistGenMove(MultipleTryMove, MHMove):
    """Perform a multiple try MH move based on a distribution.
    
    Distribution must be independent of the current point.
    
    This is effectively an example of the mutliple try class inheritance structure.
    
    """
    def __init__(self, dist, **kwargs):
        
        MultipleTryMove.__init__(self, **kwargs)
        MHMove.__init__(self, **kwargs)

        self.dist = dist

    def special_generate_logpdf(self, generated_coords):
        return self.dist.logpdf(generated_coords)

    def special_generate_func(self, coords, random, size=1, fill_tuple=None, fill_values=None, **kwargs):
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        generated_coords = self.dist.rvs(size=(nwalkers, size))

        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values
            
        generated_logpdf = self.special_generate_logpdf(generated_coords.reshape(nwalkers * size, -1)).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def special_like_func(self, generated_coords):
        generated_coords_in = {"model_0": generated_coords.reshape(-1, 1, 5)[None, :]}
        ll = self.current_model.compute_log_like_fn(generated_coords_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords):
        generated_coords_in = {"model_0": generated_coords.reshape(-1, 1, 5)[None, :]}
        lp = self.current_model.compute_log_prior_fn(generated_coords_in)
        return lp.reshape(-1, self.num_try)


