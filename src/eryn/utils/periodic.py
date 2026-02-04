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

    def __init__(self, periodic, key_order=None):

        # store all the information
        self.periodic = periodic
        inds_periodic = {}
        periods = {}
        for key in periodic:
            if periodic[key] is None:
                continue
            inds_periodic[key] = []
            periods[key] = []
            for var, period in periodic[key].items():
                if isinstance(var, str):
                    if key_order is None:
                        raise ValueError(f"If providing str values for the variable names, must provide key_order argument.")
                    index = key_order[key].index(var)
                elif isinstance(var, int):
                    index = var
                inds_periodic[key].append(index)
                periods[key].append(period)

        self.inds_periodic = {
            key: np.asarray(tmp) for key, tmp in inds_periodic.items()
        }
        self.periods = {
            key: np.asarray(tmp) for key, tmp in periods.items()
        }

    def distance(self, p1, p2, xp=None):
        """Move from p1 to p2 with periodic distance control

        Args:
            p1 (dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
            p2 (dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        Returns:
            dict: Distances accounting for periodicity.
                    Keys are branch names and values are distance arrays.

        """

        # cupy or numpy
        if xp is None:
            xp = np

        # make sure both have same branches
        assert list(p1.keys()) == list(p2.keys())

        names = list(p1.keys())

        # prepare output
        out_diff = {}
        for key in names:
           
            # get basic distance
            diff = p2[key] - p1[key]

            if  key in self.periods and len(self.periods[key]) > 0:
                 # get period info
                periods = xp.asarray(self.periods[key])
                inds_periodic = xp.asarray(self.inds_periodic[key])

                # get specific periodic parameterss
                diff_periodic = diff[:, :, inds_periodic]

                # fix when the distance is over 1/2 period away
                inds_fix = (
                    xp.abs(diff_periodic) > periods[xp.newaxis, xp.newaxis, :] / 2.0
                )

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

    def wrap(self, p, xp=None):
        """Wrap p with periodic distance control

        Args:
            p (dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        """

        # cupy or numpy
        if xp is None:
            xp = np

        names = list(p.keys())
        # wrap for each branch
        for key in names:
            pos = p[key]

            if  key in self.periods and len(self.periods[key]) > 0:
                # get periodic information
                periods = xp.asarray(self.periods[key])
                inds_periodic = xp.asarray(self.inds_periodic[key])
                # wrap
                pos[:, :, inds_periodic] = (
                    pos[:, :, inds_periodic] % periods[xp.newaxis, xp.newaxis, :]
                )

            # fill new info
            p[key] = pos

        return p
