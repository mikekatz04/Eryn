import numpy as np
from scipy import stats


def uniform_dist(min, max):
    if min > max:
        temp = min
        min = max
        max = temp

    mean = (max + min) / 2.0
    sig = max - min
    dist = stats.uniform(min, sig)
    return dist


def log_uniform(min, max):
    if min > max:
        temp = min
        min = max
        max = temp

    mean = (max + min) / 2.0
    sig = max - min
    dist = stats.loguniform(min, sig)
    return dist


class PriorContainer:
    def __init__(self, priors_in):

        self.priors_in = priors_in.copy()
        self.priors = []

        temp_inds = []
        for inds, dist in priors_in.items():
            if isinstance(inds, tuple):
                inds_in = np.asarray(inds)
                self.priors.append([inds_in, dist])

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
        # TODO: check if mutliple index prior will work
        prior_vals = np.zeros(x.shape[0])
        for i, (inds, prior_i) in enumerate(self.priors):
            vals_in = x[:, inds].squeeze()
            temp = prior_i.logpdf(vals_in)
            prior_vals += temp

        return prior_vals

    def rvs(self, size=1):

        if isinstance(size, int):
            size = (size,)

        elif not isinstance(size, tuple):
            raise ValueError("Size must be int or tuple of ints.")

        out_inds = tuple([slice(None) for _ in range(len(size))])

        out = np.zeros(size + (self.ndim,))
        for i, (inds, prior_i) in enumerate(self.priors):
            inds_in = out_inds + (inds,)
            adjust_inds = out_inds + (None,)
            out[inds_in] = prior_i.rvs(size=size)[adjust_inds]

        return out


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
