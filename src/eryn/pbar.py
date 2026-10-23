# -*- coding: utf-8 -*-
"""Progress-bar helper for the samplers: :func:`get_progress_bar` (``tqdm`` or a no-op)."""

import logging

__all__ = ["get_progress_bar"]

logger = logging.getLogger(__name__)

try:
    import tqdm
except ImportError:
    tqdm = None


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing"""

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, total):
    """Return a progress-bar context manager with an ``update(n)`` method.

    If a bar is requested but ``tqdm`` is not installed, a warning is logged and a bar
    that does nothing is returned.

    Args:
        display (bool or str): ``False`` for no bar, ``True`` for ``tqdm.tqdm``, or a str
            ``s`` for ``tqdm.tqdm_<s>`` (for example ``"notebook"``).
        total (int or None): Total number of updates (``None`` for an open-ended bar).

    Returns:
        object: A ``tqdm`` bar, or an internal no-op bar with the same interface.

    Example:
        >>> from eryn.pbar import get_progress_bar
        >>> with get_progress_bar(False, 10) as pbar:
        ...     pbar.update(1)
    """
    if display:
        if tqdm is None:
            logger.warning(
                "You must install the tqdm library to use progress "
                "indicators with emcee"
            )
            return _NoOpPBar()
        else:
            if display is True:
                return tqdm.tqdm(total=total)
            else:
                return getattr(tqdm, "tqdm_" + display)(total=total)

    return _NoOpPBar()
