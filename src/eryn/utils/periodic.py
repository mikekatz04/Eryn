"""Periodic-parameter handling: :class:`PeriodicContainer`.

The sampler builds one container from its ``periodic`` argument and hands it to every move;
proposals call :meth:`PeriodicContainer.distance` (shortest signed separation) and
:meth:`PeriodicContainer.wrap` (map back into ``[0, period)``).
"""

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
    """Periodic distance and wrapping for selected parameters of each branch.

    Args:
        periodic: ``{branch_name: {parameter: period}}``. A parameter is a column index
            (``int``) or a name (``str``, resolved through ``key_order``). A branch whose
            value is ``None`` is skipped.
        key_order: ``{branch_name: [names in column order]}``, required for branches that
            use names (e.g. ``{name: prior.key_order for name, prior in priors.items()}``
            with :class:`~eryn.priors.probdist.ProbDistContainer` priors). Default ``None``.
            :class:`~eryn.ensemble.EnsembleSampler` builds its container from a plain
            ``periodic`` dict without ``key_order``, so with names pass a ready
            :class:`PeriodicContainer` instead.

    Attributes:
        periodic (dict): The input mapping.
        inds_periodic (dict[str, np.ndarray]): Per branch, the periodic column indices.
        periods (dict[str, np.ndarray]): Per branch, the matching periods.

    Raises:
        ValueError: If a branch uses names but ``key_order`` lacks that branch, or a name
            is not in ``key_order[branch]``.

    Example:
        >>> import numpy as np
        >>> from eryn.utils import PeriodicContainer
        >>> periodic = PeriodicContainer(
        ...     {"model_0": {"phase": 2 * np.pi}}, key_order={"model_0": ["amp", "phase"]}
        ... )
        >>> p = {"model_0": np.array([[1.0, 2 * np.pi + 0.5]])}
        >>> round(float(periodic.wrap(p)["model_0"][0, 1]), 6)
        0.5
        >>> d = periodic.distance(
        ...     {"model_0": np.array([[0.0, 0.1]])},
        ...     {"model_0": np.array([[0.0, 2 * np.pi - 0.1]])},
        ... )
        >>> round(float(d["model_0"][0, 1]), 6)
        -0.2
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
        """Per-branch separation ``p2 - p1`` with periodic columns taken the short way round.

        For each periodic column, a difference with ``|d| > period / 2`` is shifted by one
        period towards zero (``d - sign(d) * period``). That lands in
        ``[-period / 2, period / 2]`` whenever ``|d| < 1.5 * period``, which holds when both
        inputs are wrapped into ``[0, period)``. Non-periodic columns, and branches without
        periodic columns, get the plain difference. The inputs are not modified.

        Args:
            p1: ``{branch_name: array}``; parameters on the last axis. Branches are taken
                from ``p1``.
            p2: Same layout as ``p1``.
            xp: Array module, ``numpy`` (default) or ``cupy``, matching the arrays.

        Returns:
            dict[str, xp.ndarray]: ``{branch_name: separation}``, each the shape of the
            inputs.
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
        """Wrap the periodic columns of each branch into ``[0, period)``, IN PLACE.

        Uses ``pos[..., inds] % periods``, so the target interval is always
        ``[0, period)`` whatever the parameter's prior range (a phase with a
        ``[-pi, pi]`` prior is mapped into ``[0, 2 pi)``).

        Args:
            p: ``{branch_name: array}``; parameters on the last axis. The arrays are
                modified in place.
            xp: Array module, ``numpy`` (default) or ``cupy``, matching the arrays.

        Returns:
            dict[str, xp.ndarray]: ``p`` itself, with the wrapped arrays.
        """
        for branch, pos in p.items():
            if branch in self.periods and len(self.periods[branch]) > 0:
                periods = xp.asarray(self.periods[branch])
                inds = xp.asarray(self.inds_periodic[branch])
                pos[..., inds] = pos[..., inds] % periods
            p[branch] = pos
        return p
    