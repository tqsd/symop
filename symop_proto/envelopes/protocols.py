from __future__ import annotations
from typing import (
    Callable,
    Optional,
    Protocol,
    Tuple,
    Sequence,
    Union,
    Self,
    cast,
    runtime_checkable,
)
import numpy as np
from numpy.typing import NDArray

try:
    from matplotlib.figure import FigureBase
    from matplotlib.axes import Axes
except Exception:
    from typing import Any as FigureBase, Any as Axes  # type: ignore[misc]

from symop_proto.core.protocols import HasSignature

FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
RCArray = FloatArray | ComplexArray
TimeFunc = Callable[[FloatArray], RCArray]

AxesLike = Union[Axes, Sequence[Axes], np.ndarray]
PlotReturn = Tuple[FigureBase, np.ndarray]


@runtime_checkable
class EnvelopeProto(HasSignature, Protocol):
    """Structural interface for time/frequency envelopes."""

    def time_eval(self, t: FloatArray) -> RCArray: ...
    def freq_eval(self, w: FloatArray) -> RCArray: ...

    def delayed(self, dt: float) -> Self: ...
    def phased(self, dphi: float) -> Self: ...

    def center_and_scale(self) -> Tuple[float, float]: ...
    def overlap(self, other: EnvelopeProto) -> complex: ...
    @property
    def latex(self) -> Optional[str]: ...

    # --- plotting ---
    def plot(
        self,
        *,
        t: Optional[FloatArray] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        n: int = 2000,
        show_real_imag: bool = True,
        show_phase: bool = False,
        show_formula: bool = True,
        title: Optional[str] = None,
        axes: Optional[AxesLike] = None,
        label: Optional[str] = None,
        normalize_envelope: bool = False,
        show_parts: bool = False,
    ) -> PlotReturn: ...


@runtime_checkable
class SupportsOverlapWithGeneric(Protocol):
    def overlap_with_generic(self, other: EnvelopeProto) -> complex: ...


@runtime_checkable
class _HasGetFigure(Protocol):
    def get_figure(self) -> FigureBase | None: ...


@runtime_checkable
class HasSpectralHints(Protocol):
    @property
    def omega0(self) -> float: ...

    @property
    def omega_sigma(self) -> float: ...


def _coerce_axes_array(axes: AxesLike) -> tuple[FigureBase, np.ndarray]:
    """
    Normalize Axes-like inputs to (FigureBase, 1-D object array of Axes).
    Raises if the Axes isn't attached to any Figure.
    """
    if isinstance(axes, _HasGetFigure):
        # single Axes
        fig = axes.get_figure()
        if fig is None:
            raise RuntimeError("Provided Axes is not attached to a Figure.")
        arr = np.asarray([axes], dtype=object)
        return fig, arr

    if isinstance(axes, np.ndarray):
        # numpy array of Axes (any shape)
        arr = np.asarray(axes, dtype=object).ravel()
        first_ax = cast(Axes, arr.flat[0])
        fig = first_ax.get_figure()
        if fig is None:
            raise RuntimeError(
                "Provided Axes array contains an Axes without a Figure."
            )
        return fig, arr

    # sequence of Axes
    seq = cast(Sequence[Axes], axes)
    first_ax = cast(Axes, seq[0])
    fig = first_ax.get_figure()
    if fig is None:
        raise RuntimeError(
            "Provided Axes sequence contains an Axes without a Figure."
        )
    arr = np.asarray(seq, dtype=object).ravel()
    return fig, arr
