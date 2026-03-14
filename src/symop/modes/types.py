"""Shared type aliases and small utility helpers for mode handling.

This module centralizes common array type aliases and lightweight
validation helpers used across the ``symop.modes`` subpackage.

It also provides utilities for normalizing Matplotlib Axes inputs
and validating floating-point parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from symop.core.types.arrays import FloatArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.figure import FigureBase as MplFigureBase
else:
    MplAxes = Any  # type: ignore[misc,assignment]
    MplFigureBase = Any  # type: ignore[misc,assignment]

FigureBase: TypeAlias = "MplFigureBase"
Axes: TypeAlias = "MplAxes"


class HasGetFigure(Protocol):
    """Minimal protocol for objects exposing get_figure()."""

    def get_figure(self) -> FigureBase | None:
        """Get `FigureBase`."""
        ...


AxesArray: TypeAlias = NDArray[Any]
AxesLike: TypeAlias = HasGetFigure | Sequence[HasGetFigure] | AxesArray
PlotReturn: TypeAlias = tuple[FigureBase, AxesArray]


def coerce_axes_array(axes: AxesLike) -> tuple[FigureBase, AxesArray]:
    """Normalize Axes-like inputs to a consistent array representation."""
    if isinstance(axes, np.ndarray):
        arr: AxesArray = np.asarray(axes, dtype=object).ravel()
        first = cast(HasGetFigure, arr.flat[0])
        fig = first.get_figure()
        if fig is None:
            raise RuntimeError("Provided Axes array contains an Axes without a Figure.")
        return fig, arr

    if isinstance(axes, Sequence) and not hasattr(axes, "get_figure"):
        first = axes[0]
        fig = first.get_figure()
        if fig is None:
            raise RuntimeError(
                "Provided Axes sequence contains an Axes without a Figure."
            )
        arr = np.asarray(axes, dtype=object).ravel()
        return fig, arr
    # Single Axes-like case (HasGetFigure)
    ax = cast(HasGetFigure, axes)
    fig = ax.get_figure()
    if fig is None:
        raise RuntimeError("Provided Axes is not attached to a Figure.")
    arr = np.asarray([ax], dtype=object)
    return fig, arr


def as_float_array(x: Any) -> FloatArray:
    """Convert input to a NumPy array with ``dtype=float``.

    This is a small convenience used throughout ``symop.modes`` to normalize
    array-like inputs (lists, tuples, NumPy arrays) to a floating-point
    NumPy array.

    Parameters
    ----------
    x:
        Array-like input.

    Returns
    -------
    FloatArray
        NumPy array view/copy with ``dtype=float``.

    """
    return cast(FloatArray, np.asarray(x, dtype=float))


def require_pos_finite(name: str, x: float) -> float:
    """Validate that a float is strictly positive and finite.

    Parameters
    ----------
    name:
        Name of the parameter (used in error messages).
    x:
        Value to validate.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If ``x`` is not finite or not strictly positive.

    """
    v = float(x)
    if not (v > 0.0) or not np.isfinite(v):
        raise ValueError(f"{name} must be positive finite, got {x!r}")
    return v


def require_finite(name: str, x: float) -> float:
    """Validate that a float is finite.

    Parameters
    ----------
    name:
        Name of the parameter (used in error messages).
    x:
        Value to validate.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ValueError
        If ``x`` is not finite.

    """
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v
