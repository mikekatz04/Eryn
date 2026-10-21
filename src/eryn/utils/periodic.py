from __future__ import annotations

from typing import Any, TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray

try: 
    import cupy as xp # type: ignore[import]
except (ModuleNotFoundError, ImportError) as e:
    pass

Element = TypeVar("Element", str, int)


class PeriodicContainer:
    """Perform operations for periodic parameters

    Args:
        periodic_in (dict[str, dict[str | int, float]]): Keys are ``branch_names``. 
            Values are dictionaries. These dictionaries have keys as the parameter
            indexes and values their associated period.
        key_order (dict[str, list[str]] | None, optional): Order of the string keys if 
            string keys are used. (default: ``None``)
    """
    
    def __init__(
        self,
        periodic: dict[str, dict[Element, float]],
        key_order: dict[str, list[str]] | None = None,
    ):
        # store all information
        self.periodic = periodic
        self.inds_periodic: dict[str,  NDArray] = {}
        self.periods: dict[str, NDArray] = {}

        for branch, branch_periodic in periodic.items():
            if branch_periodic is None:
                continue

            inds_tmp = []
            periods_tmp = []

            for var, period in branch_periodic.items():
                if isinstance(var, str):
                    if key_order is None or branch not in key_order:
                        raise ValueError("Must provide key_order for string variable names.")
                    index = key_order[branch].index(var)
                else:
                    index = var
                    
                inds_tmp.append(index)
                periods_tmp.append(period)

            self.inds_periodic[branch] = np.asarray(inds_tmp)
            self.periods[branch] = np.asarray(periods_tmp)

    def distance(
        self,
        p1: dict[str, NDArray],
        p2: dict[str, NDArray],
        xp: Any = np,
    ) -> dict[str, NDArray]:
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
        out_diff = {}

        for branch in p1.keys():
            diff = p2[branch] - p1[branch]
            if branch in self.periods and len(self.periods[branch]) > 0:
                periods = xp.asarray(self.periods[branch])
                inds = xp.asarray(self.inds_periodic[branch])

                # View into periodic dimensions
                diff_periodic = diff[..., inds]
                
                # Wrapped diff computation
                wrapped = diff_periodic - xp.sign(diff_periodic) * periods
                needs_wrap = xp.abs(diff_periodic) > (periods / 2.0)
                
                diff[..., inds] = xp.where(needs_wrap, wrapped, diff_periodic)

            out_diff[branch] = diff

        return out_diff

    def wrap(
        self, 
        p: dict[str, NDArray], 
        xp: Any = np
    ) -> dict[str, NDArray]:
        """Wrap p with periodic distance control

        Args:
            p (dict): If dict, keys are ``branch_names``
                and values are positions with parameters along the final dimension.
            xp (object, optional): ``numpy`` or ``cupy``. If ``None``, use ``numpy``.
                (default: ``None``)

        """
        for branch, pos in p.items():
            if branch in self.periods and len(self.periods[branch]) > 0:
                periods = xp.asarray(self.periods[branch])
                inds = xp.asarray(self.inds_periodic[branch])
                pos[..., inds] = pos[..., inds] % periods
            p[branch] = pos
        return p
    