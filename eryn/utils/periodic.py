import numpy as np


class PeriodicContainer:
    """Perform operations for periodic parameters

    Args:
        periodic_in (dict): Keys are ``branch_names``. Values are
            dictionaries. These dictionaries have keys as the parameter
            indexes and values their associated period.

    """

    def __init__(self, periodic):
        self.periodic = periodic
        self.inds_periodic = {
            key: np.asarray([i for i in periodic[key].keys()]) for key in periodic
        }
        self.periods = {
            key: np.asarray([i for i in periodic[key].values()]) for key in periodic
        }

    def _check_names(self, names):
        if names is None:
            try:
                names = p1.keys()
            except AttributeError:
                raise ValueError(
                    "If not providing the names kwarg, must provide dictionaries for p1 and p2."
                )

        elif isinstance(names, str):
            names = [names]

        elif not isinstance(names, list):
            raise ValueError("If providing names, must be a str or list of str.")

        return names

    def distance(self, p1, p2, names=None):
        """Move from p1 to p2

        s = p1 / p2 = c[rint]"""

        names = self._check_names(names)

        if isinstance(p1, np.ndarray) or isinstance(p2, np.ndarray):
            if len(names) > 1:
                raise ValueError(
                    "If providing p1 or p2 as np.ndarray, names must be a single string or length-1 list."
                )

            if isinstance(p1, np.ndarray):
                p1 = {names[0]: np.atleast_3d(p1)}

            if isinstance(p2, np.ndarray):
                p2 = {names[0]: np.atleast_3d(p2)}

        out_diff = {}
        for key in names:
            periods = self.periods[key]
            inds_periodic = self.inds_periodic[key]

            diff = p2[key] - p1[key]

            diff_periodic = diff[:, :, inds_periodic]

            inds_fix = np.abs(diff_periodic) > periods[np.newaxis, np.newaxis, :] / 2.0

            new_s = -(
                periods[np.newaxis, np.newaxis, :] - p1[key][:, :, inds_periodic]
            ) * (diff_periodic < 0.0) + (
                periods[np.newaxis, np.newaxis, :] + p1[key][:, :, inds_periodic]
            ) * (
                diff_periodic >= 0.0
            )

            diff_periodic[inds_fix] = (
                p2[key][:, :, inds_periodic][inds_fix] - new_s[inds_fix]
            )
            diff[:, :, inds_periodic] = diff_periodic

            out_diff[key] = diff

        return out_diff

    def wrap(self, p, names=None):

        names = self._check_names(names)

        if isinstance(p, np.ndarray):
            if len(names) > 1:
                raise ValueError(
                    "If providing p as np.ndarray, names must be a single string or length-1 list."
                )

            if isinstance(p, np.ndarray):
                p = {names[0]: np.atleast_3d(p)}

        for key in names:
            pos = p[key]
            periods = self.periods[key]
            inds_periodic = self.inds_periodic[key]
            pos[:, :, inds_periodic] = (
                pos[:, :, inds_periodic] % periods[np.newaxis, np.newaxis, :]
            )

            p[key] = pos

        return p
