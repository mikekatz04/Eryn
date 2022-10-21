from ast import Import
import numpy as np

try:
    import cupy as xp

except (ModuleNotFoundError, ImportError) as e:
    pass


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
            # TODO: fix this
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

    def distance(self, p1, p2, names=None, xp=None):
        """Move from p1 to p2 with periodic distance control

        Args:
            p1 (double np.ndarray or dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
                If array, positions with parameters along the final dimension.
            p2 (double np.ndarray or dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
                If array, positions with parameters along the final dimension.
            names (str or list of str): ``branch_names`` to properly reference
                periods associated with each branch. (default: ``None``)


        """
        if xp is None:
            xp = np

        names = self._check_names(names)

        # setup p1 and p2
        if isinstance(p1, xp.ndarray) or isinstance(p2, xp.ndarray):
            if len(names) > 1:
                raise ValueError(
                    "If providing p1 or p2 as xp.ndarray, names must be a single string or length-1 list."
                )

            if isinstance(p1, xp.ndarray):
                p1 = {names[0]: xp.atleast_3d(p1)}

            if isinstance(p2, xp.ndarray):
                p2 = {names[0]: xp.atleast_3d(p2)}

        out_diff = {}
        for key in names:
            periods = xp.asarray(self.periods[key])
            inds_periodic = xp.asarray(self.inds_periodic[key])

            # get basic distance
            diff = p2[key] - p1[key]

            if len(self.periods[key]) > 0:
                # get specific periodic parameterss
                diff_periodic = diff[:, :, inds_periodic]

                # fix when the distance is over 1/2 period away
                inds_fix = xp.abs(diff_periodic) > periods[xp.newaxis, xp.newaxis, :] / 2.0

                # wrap back to make proper periodic distance
                new_s = -(
                    periods[xp.newaxis, xp.newaxis, :] - p1[key][:, :, inds_periodic]
                ) * (diff_periodic < 0.0) + (
                    periods[xp.newaxis, xp.newaxis, :] + p1[key][:, :, inds_periodic]
                ) * (
                    diff_periodic >= 0.0
                )

                # fill new information
                diff_periodic[inds_fix] = (
                    p2[key][:, :, inds_periodic][inds_fix] - new_s[inds_fix]
                )
                diff[:, :, inds_periodic] = diff_periodic

            out_diff[key] = diff

        return out_diff

    def wrap(self, p, names=None, xp=None):
        """Wrap p with periodic distance control

        Args:
            p (double xp.ndarray or dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
                If array, positions with parameters along the final dimension.
            names (str or list of str): ``branch_names`` to properly reference
                periods associated with each branch. (default: ``None``)

        """
        names = self._check_names(names)

        if xp is None:
            xp = np

        # adjust input
        if isinstance(p, xp.ndarray):
            if len(names) > 1:
                raise ValueError(
                    "If providing p as xp.ndarray, names must be a single string or length-1 list."
                )

            if isinstance(p, xp.ndarray):
                p = {names[0]: xp.atleast_3d(p)}

        # wrap for each branch
        for key in names:
            pos = p[key]

            if len(self.periods[key]) > 0:
                periods = xp.asarray(self.periods[key])
                inds_periodic = xp.asarray(self.inds_periodic[key])
                # wrap
                pos[:, :, inds_periodic] = (
                    pos[:, :, inds_periodic] % periods[xp.newaxis, xp.newaxis, :]
                )

            p[key] = pos

        return p
