import numpy as np
from scipy import stats
from copy import deepcopy

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass


class UniformDistribution(object):
    """Generate uniform distribution between ``min`` and ``max``

    Args:
        min_val (double): Minimum in the uniform distribution
        max_val (double): Maximum in the uniform distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Raises:
        ValueError: Issue with inputs.

    """

    def __init__(self, min_val, max_val, use_cupy=False):
        if min_val > max_val:
            tmp = min_val
            min_val = max_val
            max_val = tmp
        elif min_val == max_val:
            raise ValueError("Min and max values are the same.")

        self.min_val = min_val
        self.max_val = max_val
        self.diff = max_val - min_val

        self.pdf_val = 1 / self.diff
        self.logpdf_val = np.log(self.pdf_val)

        self.use_cupy = use_cupy
        if use_cupy:
            try:
                cp.abs(1.0)
            except NameError:
                raise ValueError("CuPy not found.")

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")

        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        rand_unif = xp.random.rand(*size)

        out = rand_unif * self.diff + self.min_val

        return out

    def pdf(self, x):
        out = self.pdf_val * ((x >= self.min_val) & (x <= self.max_val))

        return out

    def logpdf(self, x):
        xp = np if not self.use_cupy else cp

        out = xp.zeros_like(x)
        out[(x >= self.min_val) & (x <= self.max_val)] = self.logpdf_val
        out[(x < self.min_val) | (x > self.max_val)] = -np.inf
        return out

    def copy(self):
        return deepcopy(self)


def uniform_dist(min, max, use_cupy=False):
    """Generate uniform distribution between ``min`` and ``max``

    Args:
        min (double): Minimum in the uniform distribution
        max (double): Maximum in the uniform distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Returns:
        :class:`UniformDistribution`: Uniform distribution.


    """
    dist = UniformDistribution(min, max, use_cupy=use_cupy)

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
        min (double): Minimum in the uniform distribution
        max (double): Maximum in the uniform distribution
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Raises:
        ValueError: If ``min`` is greater than ``max``.


    """

    def __init__(self, min, max, use_cupy=False):
        self.min, self.max = min, max
        self.diff = self.max - self.min
        if self.min > self.max:
            raise ValueError("min must be less than max.")

        self.dist = uniform_dist(0.0, 1.0, use_cupy=use_cupy)

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


class ProbDistContainer:
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
        use_cupy (bool, optional): If ``True``, use CuPy. If ``False`` use Numpy.
            (default: ``False``)

    Raises:
        ValueError: Missing parameters or incorrect index keys.

    """

    def __init__(self, priors_in, use_cupy=False):
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
            raise ValueError(
                "Please ensure all sampled parameters are included in priors."
            )

        self.ndim = uni_inds.max() + 1

        self.use_cupy = use_cupy

        for key, item in self.priors_in.items():
            item.use_cupy = use_cupy

    def logpdf(self, x, keys=None):
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[..., ndim]):
                Input parameters to get prior values.
            keys (list, optional): List of keys related to which parameters to gather the logpdf for.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__`` 
                function. Even when using this kwarg, must provide all ``ndim`` parameters as input. The prior will just not 
                be calculated if its associated key is not included. Default is ``None``.

        Returns:
            np.ndarray[...]: Prior values.

        """
        # TODO: check if mutliple index prior will work
        xp = np if not self.use_cupy else cp

        # make sure at least 2D
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True

        elif x.ndim != 2:
            raise ValueError("x needs to 1 or 2 dimensional array.")
        else:
            squeeze = False

        prior_vals = xp.zeros(x.shape[0])

        # sum the logs (assumes parameters are independent)
        for i, (inds, prior_i) in enumerate(self.priors):

            if keys is not None:
                if len(inds) > 1:
                    if tuple(inds) not in keys:
                        continue
                else:
                    if inds[0] not in keys:
                        continue

            vals_in = x[:, inds]
            if hasattr(prior_i, "logpdf"):
                temp = prior_i.logpdf(vals_in)
            else:
                temp = prior_i.logpmf(vals_in)

            prior_vals += temp.squeeze()

        # if only one walker was asked for, return a scalar value not an array
        if squeeze:
            prior_vals = prior_vals[0].item()

        return prior_vals

    def ppf(self, x, groups=None):
        """Get logpdf by summing logpdf of individual distributions

        Args:
            x (double np.ndarray[..., ndim]):
                Input parameters to get prior values.

        Returns:
            np.ndarray[...]: Prior values.

        """
        raise NotImplementedError
        if groups is not None:
            raise NotImplementedError

        xp = np if not self.use_cupy else cp

        # TODO: check if mutliple index prior will work
        is_1d = x.ndim == 1
        x = xp.atleast_2d(x)
        out_vals = xp.zeros_like(x)

        # sum the logs (assumes parameters are independent)
        for i, (inds, prior_i) in enumerate(self.priors):
            if len(inds) > 1:
                raise NotImplementedError

            vals_in = x[:, inds].squeeze()
            temp = prior_i.ppf(vals_in)

            out_vals[:, inds[0]] = temp

        if is_1d:
            return out_vals.squeeze()
        return out_vals

    def rvs(self, size=1, keys=None):
        """Generate random values according to prior distribution

        The user will have to be careful if there are prior functions that
        do not have an ``rvs`` method. This means that generated points may lay
        inside the prior of all input priors that have ``rvs`` methods, but
        outside the prior if priors without the ``rvs`` method are included.

        Args:
            size (int or tuple of ints, optional): Output size for number of generated
                sources from prior distributions.
            keys (list, optional): List of keys related to which parameters to generate.
                They must exactly match the input keys for the ``priors_in`` dictionary for the ``__init__`` 
                function. If used, it will produce and output array of ``tuple(size) + (len(keys),)``. 
                Default is ``None``.

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

        xp = np if not self.use_cupy else cp

        # setup the slicing to properly sample points
        out_inds = tuple([slice(None) for _ in range(len(size))])

        # setup output and loop through priors

        ndim = self.ndim

        out = xp.zeros(size + (ndim,))
        for i, (inds, prior_i) in enumerate(self.priors):
            # only generate desired parameters
            if keys is not None:
                if len(inds) > 1:
                    if tuple(inds) not in keys:
                        continue
                else:
                    if inds[0] not in keys:
                        continue

            # guard against extra prior functions without rvs methods
            if not hasattr(prior_i, "rvs"):
                continue
            # combines outer dimensions with indices of interest
            inds_in = out_inds + (inds,)

            # allows for proper adding of quantities to out array
            if len(inds) == 1:
                adjust_inds = out_inds + (None,)
                out[inds_in] = prior_i.rvs(size=size)[adjust_inds]
            else:
                out[inds_in] = prior_i.rvs(size=size)

        return out
