import numpy as np
from scipy import stats
import torch
from torch.distributions.normal import Normal


def uniform_dist(min, max):
    """Generate uniform distribution between ``min`` and ``max``

    Args:
        min (double): Minimum in the uniform distribution
        max (double): Maximum in the uniform distribution

    Returns:
        scipy distribution object: Uniform distribution built from
            `scipy.stats.uniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html>_`.

    """
    # adjust ordering if needed
    if min > max:
        temp = min
        min = max
        max = temp

    # setup quantities for scipy
    sig = max - min
    dist = stats.uniform(min, sig)
    return dist


def log_uniform(min, max):
    """Generate log-uniform distribution between ``min`` and ``max``

    Args:
        min (double): Minimum in the log-uniform distribution
        max (double): Maximum in the log-uniform distribution

    Returns:
        scipy distribution object: Log-uniform distribution built from
            `scipy.stats.uniform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html>_`.

    """
    # adjust ordering if needed
    if min > max:
        temp = min
        min = max
        max = temp

    # setup quantities for scipy
    sig = max - min
    dist = stats.loguniform(min, sig)
    return dist


class MappedUniformDistribution:
    """Maps uniform distribution to zero to 1.

    This is a modified uniform distribution that maps
    the input values to a range from zero to 1 by using ``min`` and
    ``max`` values input by user. This ensures the log of the prior value
    from this distribution is zero if the value is between ``min`` and ``max``.
    and ``-np.inf`` if it is outside that range.

    Args:
        min (double): Minimum in the log-uniform distribution
        max (double): Maximum in the log-uniform distribution

    Raises:
        ValueError: If ``min`` is greater than ``max``.


    """

    def __init__(self, min, max):
        self.min, self.max = min, max
        self.diff = self.max - self.min
        if self.min > self.max:
            raise ValueError("min must be less than max.")

        self.dist = uniform_dist(0.0, 1.0)

    def logpdf(self, x):
        """Get the log of the pdf value for this distribution.

        Args:
            x (double np.ndarray):
                Input parameters to get prior values.

        Returns:
            np.ndarray: Associated logpdf values of the input.

        """
        temp = 1.0 - (self.max - x) / self.diff
        return self.dist.logpdf(temp)

    def rvs(self, size=1):
        """Get the log of the pdf value for this distribution.

        Args:
            size (int or tuple of ints, optional): Output size for number of generated
                sources from prior distributions.

        Returns:
            np.ndarray: Generated values.

        """
        # adjust size if int
        if isinstance(size, int):
            size = (size,)

        elif not isinstance(size, tuple):
            raise ValueError("Size must be int or tuple of ints.")

        temp = self.dist.rvs(size=size)

        return self.max + (temp - 1.0) * self.diff


class PriorContainer:
    """Container for holding and generating prior info

    Args:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.

    Attributes:
        priors_in (dict): Dictionary with keys as int or tuple of int
            describing which parameters the prior takes. Values are
            probability distributions with ``logpdf`` and ``rvs`` methods.
        priors (list): list of indexes and their associated distributions arranged
            in a list.
        ndim (int): Full dimensionality.

    Raises:
        ValueError: Missing parameters or incorrect index keys.

    """

    def __init__(self, priors_in):

        # copy to have
        self.priors_in = priors_in.copy()

        # to separate out in list form
        self.priors = []

        # setup lists
        temp_inds = []
        for inds, dist in priors_in.items():

            # multiple index
            if isinstance(inds, tuple):
                inds_in = np.asarray(inds)
                self.priors.append([inds_in, dist])

            # single index
            elif isinstance(inds, int):
                inds_in = np.array([inds])
                self.priors.append([inds_in, dist])

            else:
                raise ValueError(
                    "Keys for prior dictionary must be an integer or tuple."
                )

            temp_inds.append(np.asarray([inds_in]))

        uni_inds = np.unique(np.concatenate(temp_inds, axis=1).flatten())
        if len(uni_inds) != len(np.arange(np.max(uni_inds) + 1)):
            # TODO: make better
            raise ValueError(
                "If providing priors, need to ensure all sampled parameters are included."
            )

        self.ndim = uni_inds.max() + 1

    def logpdf(self, x, groups=None):
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[number of tested sources, ndim]):
                Input parameters to get prior values.

        Returns:
            np.ndarray[number of tested sources]: Prior values.

        """
        # TODO: check if mutliple index prior will work
        prior_vals = np.zeros(x.shape[0])
        # sum the logs (assumes parameters are independent)
        for i, (inds, prior_i) in enumerate(self.priors):
            vals_in = x[:, inds].squeeze()
            if hasattr(prior_i, "logpdf"):
                temp = prior_i.logpdf(vals_in)
            else:
                temp = prior_i.logpmf(vals_in)
                
            prior_vals += temp

        return prior_vals

    def rvs(self, size=1):
        """Generate random values according to prior distribution

        The user will have to be careful if there are prior functions that
        do not have an ``rvs`` method. This means that generated points may lay
        inside the prior of all input priors that have ``rvs`` methods, but
        outside the prior if priors without the ``rvs`` method are included.

        Args:
            size (int or tuple of ints, optional): Output size for number of generated
                sources from prior distributions.

        Returns:
            np.ndarray[``size + (self.ndim,)``]: Generated samples.

        Raises:
            ValueError: If size is not an int or tuple.


        """

        # adjust size if int
        if isinstance(size, int):
            size = (size,)

        elif not isinstance(size, tuple):
            raise ValueError("Size must be int or tuple of ints.")

        # setup the slicing to probably sample points
        out_inds = tuple([slice(None) for _ in range(len(size))])

        # setup output and loop through priors
        out = np.zeros(size + (self.ndim,))
        for i, (inds, prior_i) in enumerate(self.priors):
            # guard against extra prior functions without rvs methods
            if not hasattr(prior_i, "rvs"):
                continue
            # combines outer dimensions with indices of interest
            inds_in = out_inds + (inds,)
            # allows for proper adding of quantities to out array
            adjust_inds = out_inds + (None,)
            out[inds_in] = prior_i.rvs(size=size)[adjust_inds]

        return out


class UniformTorch(torch.distributions.uniform.Uniform):
    """
       For testing Likelihood Ratio.
    """

    def __init__(self, lower, upper):
        super(UniformTorch, self).__init__(lower, upper)

    def log_prob(self, sample):
        return super(UniformTorch, self).log_prob(sample).mean()


if __name__ == "__main__":
    import pickle

    multi = stats.multivariate_normal(
        mean=np.array([0.0, 0.0]), cov=np.array([[1.0, -0.3], [0.3, 1.0]])
    )
    multi_in = multi.rvs(100)

    x = np.concatenate([multi_in, np.array([np.linspace(0, 800, 100)]).T], axis=1)

    with open("amps_dist.pickle", "rb") as f:
        amps_dist = pickle.load(f)

    priors_in = {(0, 1): multi, 2: amps_dist}
    prior = PriorContainer(priors_in)

    pval = prior.logpdf(x)

    out = prior.rvs(1000)
    breakpoint()
