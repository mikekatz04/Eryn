"""The branch prior container, :class:`ProbDistContainer`.

The sampler takes one container per branch (``priors={branch_name: ProbDistContainer(...)}``)
and calls :meth:`ProbDistContainer.logpdf` on ``(N, ndim)`` coordinate arrays and
:meth:`ProbDistContainer.rvs` to draw starting points and proposals. The container sums
independent component priors, each owning one column or a tuple of columns, and resolves
conditional components (priors whose ``logpdf`` / ``rvs`` depend on another column).
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Sequence, TypeVar, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import cupy as cp # type: ignore[import]

except (ModuleNotFoundError, ImportError) as e:
    pass

from ..utils.utility import NDArrayLike

Element = TypeVar("Element", str, int)

class ProbDistContainer:
    """Joint prior over the ``ndim`` columns of a coordinate array, built from components.

    Each key of ``priors_in`` names the column(s) one component owns; the container's
    log-prior is the sum of the component log-densities and :meth:`rvs` fills each
    component's columns from its own ``rvs``.

    Column assignment:

    - ``int`` key: that column index. ``(int, ...)`` tuple key: those column indices.
    - ``str`` key: the next free column, in insertion order. Each member of a
      ``(str, ...)`` tuple key also takes the next column, so a tuple's members are
      CONSECUTIVE. ``key_order`` lists the names in column order; call
      :meth:`reset_key_order` to move named columns to other positions (for example to
      scatter a tuple's members to their real basis columns).
    - Integer and string keys cannot be mixed, and the keys must cover every column
      ``0 .. ndim - 1``.

    Component contract (duck-typed; any object with these methods works):

    - ``logpdf(x, **kw)`` (or ``logpmf``), with ``x`` of shape ``(N,)`` for a one-column
      key and ``(N, k)`` for a ``k``-column key, returning ``N`` values.
    - ``rvs(size=..., **kw)`` returning shape ``size`` (one column) or ``size + (k,)``.
    - Optional ``required_variables`` (names of conditional variables; inferred from the
      method signatures for :class:`~eryn.priors.base.Prior` subclasses). Each one is taken
      from the call's keyword arguments, else from the column of that name in the input
      array (or, in :meth:`rvs`, the already drawn output).
    - Keyword arguments are filtered to each method's signature unless it accepts
      ``**kwargs``.

    Plain ``rvs`` / ``logpdf`` are called, so the transform-backed priors of
    :mod:`eryn.priors.analytical` (e.g. :class:`~eryn.priors.analytical.LogUniform`) are
    sampled and scored in their SAMPLING variable.

    Args:
        priors_in: Mapping of key (``int``, ``str``, or a tuple of either) to a component
            distribution. Copied to a ``dict``.
        use_cupy: If ``True``, evaluate with CuPy. The value (also ``False``) is written,
            in place, to every component that has a ``use_cupy`` attribute (recursively
            into nested containers), and those components get ``return_gpu = True`` so
            arrays stay on the device inside the container. Default ``False``.
        return_gpu: When ``use_cupy``, return CuPy arrays from :meth:`logpdf` / :meth:`rvs`;
            if ``False`` they are copied to the host. Default ``False``.

    Attributes:
        priors_in (dict): Copy of the input mapping, in insertion order. Not modified by
            :meth:`reset_key_order`.
        priors (list): ``[column_indices, distribution]`` pairs; ``column_indices`` is an
            int ``np.ndarray``.
        ndim (int): Number of columns.
        key_order (list): Names (string keys) or ``list(range(ndim))`` (integer keys) in
            column order.
        name_to_idx (dict): Key or name -> column index.
        has_strings (bool): Keys are strings.
        has_ints (bool): Keys are integers.
        use_cupy (bool): Backend flag.
        return_gpu (bool): Whether results stay on the device.

    Raises:
        ValueError: If integer and string keys are mixed, a key has an unsupported type,
            or some column in ``0 .. ndim - 1`` has no prior.

    Example:
        >>> from eryn.priors import ProbDistContainer, UniformDistribution
        >>> priors = ProbDistContainer({
        ...     "a": UniformDistribution(0.0, 1.0),
        ...     "b": UniformDistribution(-2.0, 2.0),
        ... })
        >>> priors.ndim, priors.key_order
        (2, ['a', 'b'])
        >>> x = priors.rvs(size=5)
        >>> x.shape, priors.logpdf(x).shape
        ((5, 2), (5,))
        >>> round(priors.logpdf([0.5, 0.0]), 6)
        -1.386294
    """

    def __init__(
        self, 
        priors_in: Mapping[Element | Tuple[Element, ...], Any],
        use_cupy: bool = False,
        return_gpu: bool = False,
    ):
        priors_in = dict(priors_in)
        self.priors_in = priors_in.copy()
        self.priors: list[list[Any]] = []
        self.has_strings: bool = False
        self.has_ints: bool = False
        self.name_to_idx: Dict[int | str, int] = {}
        self.key_order: list[int] | list[str] = []

        self.return_gpu: bool = return_gpu
        # Propagate the array backend RECURSIVELY: a child that is itself a
        # ProbDistContainer (a nested joint prior) forwards ``use_cupy`` only to
        # ITS direct children, so without recursion a grandchild distribution
        # keeps its original backend -- and a numpy grandchild fed cupy values
        # raises "Implicit conversion to a NumPy array is not allowed".
        self._set_use_cupy(use_cupy)

        self._parse_priors()
        self._build_signatures()

    def _set_use_cupy(self, use_cupy: bool) -> None:
        """Set ``use_cupy`` on this container and all nested priors/containers.

        ``use_cupy`` is a flag (``xp`` is derived from it) so this is safe to call after
        construction as well; nested :class:`ProbDistContainer` children recurse so
        arbitrarily deep joints switch backend together. Every child that receives the flag
        also gets ``return_gpu = True`` (the container itself decides what it returns).

        Args:
            use_cupy: New backend flag.
        """
        self.use_cupy = bool(use_cupy)
        for prior in self.priors_in.values():
            if isinstance(prior, ProbDistContainer):
                prior._set_use_cupy(use_cupy)
                prior.return_gpu = True
            elif hasattr(prior, "use_cupy"):
                prior.use_cupy = use_cupy
                prior.return_gpu = True

    def _parse_priors(self) -> None:
        """Assign columns to the input keys and fill ``priors`` / ``name_to_idx`` / ``ndim``.

        Implements the column rules of the class docstring and checks that keys are all
        integers or all strings and that every column has a prior.

        Raises:
            ValueError: On mixed or unsupported key types, or uncovered columns.
        """
        current_ind = 0
        key_order_tmp: list[str] = []
        temp_inds: list[NDArray[np.int_]] = []

        for inds, dist in self.priors_in.items():
            if isinstance(inds, tuple):
                inds_tmp: list[int] = []
                for i in range(len(inds)):
                    if isinstance(inds[i], str):
                        if self.has_ints:
                            raise ValueError("Cannot mix strings and integer keys.")
                        self.has_strings = True
                        inds_tmp.append(current_ind)
                        self.name_to_idx[inds[i]] = current_ind
                        key_order_tmp.append(str(inds[i]))
                        current_ind += 1

                    elif isinstance(inds[i], int):
                        if self.has_strings:
                            raise ValueError("Cannot mix strings and integer keys.")
                        self.has_ints = True
                        inds_tmp.append(int(inds[i]))
                        self.name_to_idx[inds[i]] = int(inds[i])
                        current_ind += 1
                    else:
                        raise ValueError("Index in tuple must be int or str.")

                inds_in = np.asarray(inds_tmp)
                self.priors.append([inds_in, dist])

            elif isinstance(inds, int):
                if self.has_strings:
                    raise ValueError("Cannot mix strings and integer keys.")
                self.has_ints = True
                inds_in = np.array([inds])
                self.name_to_idx[inds] = inds
                self.priors.append([inds_in, dist])
                current_ind += 1

            elif isinstance(inds, str):
                if self.has_ints:
                    raise ValueError("Cannot mix strings and integer keys.")
                self.has_strings = True
                key_order_tmp.append(inds)
                inds_in = np.array([current_ind])
                self.name_to_idx[inds] = current_ind
                self.priors.append([inds_in, dist])
                current_ind += 1
            else:
                raise ValueError("Keys for prior dictionary must be an integer, string, or tuple.")

            if self.has_strings:
                assert key_order_tmp is not None
                self.key_order = key_order_tmp
            if self.has_ints:
                self.key_order = [i for i in range(current_ind)]

            temp_inds.append(inds_in)

        uni_inds = np.unique(np.concatenate(temp_inds).flatten())
        if len(uni_inds) != len(np.arange(np.max(uni_inds) + 1)):
            raise ValueError("Please ensure all sampled parameters are included in priors.")

        self.ndim: int = int(uni_inds.max() + 1)
        
    def _build_signatures(self) -> None:
        """Cache each component's accepted keyword names for :meth:`_filter_kwargs`.

        Per component (in ``priors`` order) and for each of ``logpdf`` / ``logpmf`` / ``rvs``
        it defines, stores the named parameters and whether it takes ``**kwargs``.
        """
        self._prior_signatures: list[dict[str, dict[str, Any]]] = []
        
        for _, dist in self.priors:            
            sig_info: dict[str, dict[str, Any]] = {}
            for method_name in ("logpdf", "logpmf", "rvs"):
                if hasattr(dist, method_name):
                    sig = inspect.signature(getattr(dist, method_name))
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    allowed_keys = {
                        k for k, p in sig.parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD
                    }
                    sig_info[method_name] = {"has_varkw": has_varkw, "allowed_keys": allowed_keys}
            self._prior_signatures.append(sig_info)

    def _filter_kwargs(self, prior_idx: int, method_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only the kwargs that component ``prior_idx``'s ``method_name`` accepts.

        Returns ``kwargs`` unchanged when the method takes ``**kwargs``, and ``{}`` when the
        method is missing or ``kwargs`` is empty.
        """
        if not kwargs:
            return {}

        info = self._prior_signatures[prior_idx].get(method_name)
        if info is None:
            return {}

        if info["has_varkw"]:
            return kwargs

        return {k: v for k, v in kwargs.items() if k in info["allowed_keys"]}

    def _extract_dependencies(
        self, prior_i: Any, x: NDArray[np.float64], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect a component's ``required_variables`` from kwargs or from ``x``.

        Each variable comes from ``kwargs`` if present, else from column
        ``name_to_idx[name]`` of ``x`` (``x[..., idx]``).

        Raises:
            ValueError: If a required variable is in neither place.
        """
        deps: dict[str, Any] = {}
        required = getattr(prior_i, "required_variables", tuple())
        for req in required:
            if req in kwargs:
                deps[req] = kwargs[req]
            elif req in self.name_to_idx:
                idx = self.name_to_idx[req]
                deps[req] = x[..., idx]
            else:
                raise ValueError(f"Conditional variable '{req}' not found.")
        return deps

    @property
    def xp(self) -> Any:
        """module: ``cupy`` when ``use_cupy`` is set, else ``numpy``.

        Note:
            If CuPy failed to import, the module-level name ``cp`` is never bound, so with
            ``use_cupy=True`` this raises ``NameError`` rather than the ``ImportError`` the
            code intends.
        """
        if self.use_cupy:
            if cp is None:
                raise ImportError("CuPy is not installed but use_cupy is True.")
            return cp
        return np

    def logpdf(
        self,
        x: NDArray,
        keys: Sequence[str | int] | None = None,
        **kwargs: Any,
    ) -> NDArrayLike:
        """Sum the component log-densities for each row of ``x``.

        Each component sees its own columns (``(N,)`` for one column, ``(N, k)`` for a
        tuple) plus its conditional variables, and must return ``N`` values; ``logpmf`` is
        used for components without ``logpdf``.

        Args:
            x: Coordinates of shape ``(ndim,)`` (one point) or ``(N, ndim)``, in the
                container's column order.
            keys: Restrict the sum to some components. Entries are matched against each
                component's COLUMN INDICES: an ``int`` for a one-column component, a tuple
                of ints for a multi-column one. For an integer-keyed container these are the
                original keys; for a string-keyed container pass column indices (names never
                match). ``x`` must still hold all ``ndim`` columns. Default ``None`` (all
                components).
            **kwargs: Forwarded to each component's ``logpdf`` (filtered to its signature)
                and used to supply conditional variables.

        Returns:
            float or xp.ndarray: For 2-D ``x``, float64 log-prior of shape ``(N,)``; a numpy
            array unless ``use_cupy`` and ``return_gpu`` are both set. For 1-D ``x``, a
            Python ``float``.

        Raises:
            ValueError: If ``x`` is not 1-D or 2-D, a component returns a number of values
                other than ``N``, or a conditional variable cannot be found.
        """
        x_arr = self.xp.asarray(x)
        squeeze = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            squeeze = True
        elif x_arr.ndim != 2:
            raise ValueError("x needs to be a 1 or 2 dimensional array.")

        nsamples = x_arr.shape[0]
        prior_vals = self.xp.zeros(nsamples, dtype=self.xp.float64)

        for i, (inds, prior_i) in enumerate(self.priors):
            if keys is not None:
                if len(inds) > 1 and tuple(inds) not in keys:
                    continue
                elif len(inds) == 1 and inds[0] not in keys:
                    continue

            vals_in = x_arr[:, inds]
            if len(inds) == 1:
                vals_in = vals_in[:,0]

            deps = self._extract_dependencies(prior_i, x_arr, kwargs)
            all_kwargs = {**kwargs, **deps}

            if hasattr(prior_i, "logpdf"):
                safe_kwargs = self._filter_kwargs(i, "logpdf", all_kwargs)
                temp = prior_i.logpdf(vals_in, **safe_kwargs)
            else:
                safe_kwargs = self._filter_kwargs(i, "logpmf", all_kwargs)
                temp = prior_i.logpmf(vals_in, **safe_kwargs)
            
            if temp.shape == (nsamples,):
                temp_out = temp
            elif temp.size == nsamples:
                temp_out = temp.reshape(nsamples)
            else:
                raise ValueError(
                    f"logpdf for prior at indices {tuple(int(v) for v in inds)} "
                    f"returned {temp.size} value(s) with shape {temp.shape}; expected "
                    f"exactly {nsamples} (one log-density per sample). This usually "
                    f"means a conditional dependency did not broadcast correctly "
                    f"against this prior's input shape (N, {len(inds)})."
                )
            prior_vals += temp_out.squeeze()

        if squeeze:
            prior_vals = prior_vals[0].item()

        if self.use_cupy and not self.return_gpu:
            return prior_vals.get()

        return prior_vals

    def ppf(self, x: NDArray, groups: Any = None, **kwargs: Any) -> NDArrayLike:
        """Percent point function (inverse CDF); not implemented.

        Args:
            x: Ignored.
            groups: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PPF is not yet implemented for the ProbDistContainer.")

    def rvs(
        self,
        size: int | tuple[int, ...] = 1,
        keys: Sequence[str | int] | None = None,
        **kwargs: Any,
    ) -> NDArrayLike:
        """Draw prior samples, one component at a time.

        Components are drawn in ``priors`` order, except that a conditional component is
        deferred until all its ``required_variables`` are known (passed in ``kwargs`` or
        already drawn); the drawn columns are then read from the output array. Columns of
        components that have no ``rvs`` method, or that ``keys`` excludes, are left at
        ``0.0``, so such samples need not lie inside the full prior.

        Args:
            size: Leading output shape; an ``int`` means ``(size,)``. Default ``1``.
            keys: Restrict the draw to some components, matched against COLUMN INDICES as in
                :meth:`logpdf`. The output still has all ``ndim`` columns. Default ``None``.
            **kwargs: Forwarded to each component's ``rvs`` (filtered to its signature);
                their names count as known conditional variables.

        Returns:
            xp.ndarray: float64 samples of shape ``size + (ndim,)``; a numpy array unless
            ``use_cupy`` and ``return_gpu`` are both set.

        Raises:
            ValueError: If ``size`` is not an ``int`` or tuple, or a component returns the
                wrong number of values.
            RuntimeError: If resolving the conditional dependencies needs more than
                ``2 * len(priors) + 1`` component visits (circular or unresolvable
                dependency).
        """
        if isinstance(size, int):
            size = (size,)
        elif not isinstance(size, tuple):
            raise ValueError("Size must be an int or tuple of ints.")

        out_inds = tuple([slice(None) for _ in range(len(size))])
        out = self.xp.zeros(size + (self.ndim,), dtype=self.xp.float64)

        pending = [(i, inds, prior_i) for i, (inds, prior_i) in enumerate(self.priors)]
        evaluated_names = set(kwargs.keys())
        max_attempts = len(pending) * 2
        attempts = 0

        while pending:
            if attempts > max_attempts:
                raise RuntimeError("Circular dependency detected in prior conditionals.")
            attempts += 1

            i, inds, prior_i = pending.pop(0)

            if keys is not None:
                if len(inds) > 1 and tuple(inds) not in keys:
                    continue
                elif len(inds) == 1 and inds[0] not in keys:
                    continue

            if not hasattr(prior_i, "rvs"):
                continue

            required = getattr(prior_i, "required_variables", tuple())
            if not all(req in evaluated_names for req in required):
                pending.append((i, inds, prior_i))
                continue

            inds_in = out_inds + (inds,)
            deps = self._extract_dependencies(prior_i, out, kwargs)
            all_kwargs = {**kwargs, **deps}

            safe_kwargs = self._filter_kwargs(i, "rvs", all_kwargs)

            samples_raw = self.xp.asarray(prior_i.rvs(size=size, **safe_kwargs))
            
            if len(inds) == 1:
                adjust_inds = out_inds + (None,)
                expected_size = int(np.prod(size))
                if samples_raw.shape == size:
                    out[inds_in] = samples_raw[adjust_inds]
                elif samples_raw.size == expected_size:
                    out[inds_in] = samples_raw.reshape(size)[adjust_inds]
                else:
                    raise ValueError(
                        f"rvs for prior at index {int(inds[0])} returned "
                        f"{samples_raw.size} value(s) with shape {samples_raw.shape}; expected "
                        f"exactly {expected_size} matching size {size}. Check that "
                        f"any conditional dependencies broadcast correctly."
                    )
            else:
                expected_shape = size + (len(inds),)
                expected_size = int(np.prod(expected_shape))
                if samples_raw.shape == expected_shape:
                    out[inds_in] = samples_raw
                elif samples_raw.size == expected_size:
                    out[inds_in] = samples_raw.reshape(expected_shape)
                else:
                    raise ValueError(
                        f"rvs for prior at indices {tuple(int(v) for v in inds)} "
                        f"returned {samples_raw.size} value(s) with shape {samples_raw.shape}; "
                        f"expected exactly {expected_size} matching shape "
                        f"{expected_shape}. Check that any conditional dependencies "
                        f"broadcast correctly."
        )

            for idx in inds:
                name = next((k for k, v in self.name_to_idx.items() if v == idx), None)
                if name:
                    evaluated_names.add(name) # type: ignore

        if self.use_cupy and not self.return_gpu:
            return out.get()
        return out

    def reset_key_order(self, new_key_order: list[int] | list[str]) -> None:
        """Move the named columns so that ``new_key_order[j]`` lives in column ``j``.

        Every component's column indices are remapped by name, ``priors`` is re-sorted by
        lowest column, and ``key_order`` / ``name_to_idx`` / the signature cache are
        updated. ``priors_in`` and ``ndim`` are unchanged; a fresh container built from
        ``priors_in`` therefore has the ORIGINAL (insertion) layout and needs the same
        call to match.

        Args:
            new_key_order: A permutation of ``key_order`` with the same element type.

        Raises:
            TypeError: If the element type differs from that of ``key_order``.
            ValueError: If ``new_key_order`` is not a permutation of ``key_order``.

        Example:
            >>> from eryn.priors import ProbDistContainer, UniformDistribution
            >>> priors = ProbDistContainer({
            ...     "b": UniformDistribution(10.0, 20.0),
            ...     "a": UniformDistribution(0.0, 1.0),
            ... })
            >>> priors.key_order
            ['b', 'a']
            >>> priors.reset_key_order(["a", "b"])
            >>> priors.key_order
            ['a', 'b']
            >>> bool((priors.rvs(100)[:, 1] >= 10.0).all())
            True
        """
        if not isinstance(new_key_order[0], type(self.key_order[0])):
            raise TypeError("The new key order must be of the same type as the current key order.")

        if set(new_key_order) != set(self.key_order) or len(new_key_order) != len(self.key_order):
            raise ValueError("The new key order must be a perfect permutation of the original elements.")

        new_idx_map = {key: idx for idx, key in enumerate(new_key_order)}
        old_to_new_idx = {old_idx: new_idx_map[key] for old_idx, key in enumerate(self.key_order)}

        new_priors = []
        for inds_in, dist in self.priors:
            updated_inds = np.asarray([old_to_new_idx[idx] for idx in inds_in])
            new_priors.append([updated_inds, dist])

        new_priors.sort(key=lambda item: np.min(item[0]))

        self.priors = new_priors
        self.key_order = new_key_order

        for key, idx in new_idx_map.items():
            self.name_to_idx[key] = idx
        self._build_signatures()