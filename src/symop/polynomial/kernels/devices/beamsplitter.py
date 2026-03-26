r"""Polynomial kernels for beamsplitter-style two-mode devices.

This module implements backend kernels for applying one or more
two-mode beamsplitter transformations to polynomial ket and density
states.

Each beamsplitter acts on an ordered pair of modes identified by their
mode signatures. The semantic planning stage is expected to provide the
target pairs, output paths, and unitary parameters in ``action.params``.

Notes
-----
The kernel performs the full representation-specific beamsplitter rewrite,
including construction of output-path modes. No additional label-edit
phase is required for the physical beamsplitter action.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.core.protocols.modes.labels import Path
from symop.core.protocols.ops.operators import ModeOp
from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.channels.models.beamsplitter import (
    beamsplitter_densitypoly,
    beamsplitter_ketpoly,
)
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


@dataclass(frozen=True)
class _BeamSplitterSpec:
    """Parsed specification for one beamsplitter application.

    Attributes
    ----------
    mode0_sig:
        Signature of the first input mode, or ``None`` if the first input
        arm is vacuum and must be synthesized.
    mode1_sig:
        Signature of the second input mode, or ``None`` if the second input
        arm is vacuum and must be synthesized.
    theta:
        Mixing angle.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.
    check_unitary:
        Whether to validate the resulting 2x2 unitary.
    atol:
        Tolerance for optional unitary validation.
    in0, in1:
        Input paths used when a vacuum partner mode must be synthesized.
    out0, out1:
        Output paths used by the beamsplitter rewrite.

    """

    mode0_sig: Any
    mode1_sig: Any
    theta: float
    phi_t: float = 0.0
    phi_r: float = 0.0
    check_unitary: bool = False
    atol: float = 1e-10
    in0: Any | None = None
    in1: Any | None = None
    out0: Any | None = None
    out1: Any | None = None


@dataclass(frozen=True)
class _BeamSplitterKernelParams:
    """Parsed kernel parameters for one or more beamsplitter applications.

    Attributes
    ----------
    pairs:
        Tuple of parsed beamsplitter application specifications.

    """

    pairs: tuple[_BeamSplitterSpec, ...]


def _make_vacuum_partner_mode(
    *,
    existing_mode: ModeOp,
    input_path: Path,
) -> Any:
    """Create a matching partner mode on a missing beamsplitter input path.

    The new mode matches the existing mode in all semantic degrees of
    freedom except for the path label, which is replaced by ``input_path``.
    """
    return existing_mode.with_path(input_path)


def _parse_single_pair(obj: object) -> _BeamSplitterSpec:
    """Parse one beamsplitter pair specification.

    Parameters
    ----------
    obj:
        Mapping-like specification for one beamsplitter action.

    Returns
    -------
    _BeamSplitterSpec
        Parsed beamsplitter specification.

    Raises
    ------
    TypeError
        If the input does not have the expected mapping structure.
    KeyError
        If required keys are missing.

    """
    if not isinstance(obj, Mapping):
        raise TypeError(
            "Beamsplitter kernel expects each pair specification to be a mapping"
        )

    if "mode0" not in obj:
        raise KeyError("Beamsplitter pair is missing required key 'mode0'")
    if "mode1" not in obj:
        raise KeyError("Beamsplitter pair is missing required key 'mode1'")
    if "theta" not in obj:
        raise KeyError("Beamsplitter pair is missing required key 'theta'")

    return _BeamSplitterSpec(
        mode0_sig=obj["mode0"],
        mode1_sig=obj["mode1"],
        theta=float(obj["theta"]),
        phi_t=float(obj.get("phi_t", 0.0)),
        phi_r=float(obj.get("phi_r", 0.0)),
        check_unitary=bool(obj.get("check_unitary", False)),
        atol=float(obj.get("atol", 1e-10)),
        in0=obj.get("in0"),
        in1=obj.get("in1"),
        out0=obj.get("out0"),
        out1=obj.get("out1"),
    )


def _parse_beamsplitter_params(
    action: DeviceActionProtocol,
) -> _BeamSplitterKernelParams:
    """Extract and validate kernel parameters from a device action.

    Parameters
    ----------
    action:
        Semantic device action produced during planning.

    Returns
    -------
    _BeamSplitterKernelParams
        Parsed beamsplitter parameters.

    Raises
    ------
    TypeError
        If ``action.params`` does not have the expected mapping shape.

    """
    params = action.params
    if not isinstance(params, Mapping):
        raise TypeError("Beamsplitter kernel expects action.params to be a mapping")

    raw_pairs = params.get("pairs")
    if raw_pairs is None:
        single = {
            "mode0": params.get("mode0"),
            "mode1": params.get("mode1"),
            "theta": params.get("theta"),
            "phi_t": params.get("phi_t", 0.0),
            "phi_r": params.get("phi_r", 0.0),
            "check_unitary": params.get("check_unitary", False),
            "atol": params.get("atol", 1e-10),
            "in0": params.get("in0"),
            "in1": params.get("in1"),
            "out0": params.get("out0"),
            "out1": params.get("out1"),
        }
        pairs: tuple[_BeamSplitterSpec, ...] = (_parse_single_pair(single),)
        return _BeamSplitterKernelParams(pairs=pairs)

    if not isinstance(raw_pairs, Sequence):
        raise TypeError("Beamsplitter kernel expects params['pairs'] to be a sequence")

    pairs = tuple(_parse_single_pair(item) for item in raw_pairs)
    return _BeamSplitterKernelParams(pairs=pairs)


def _require_output_paths(spec: _BeamSplitterSpec) -> tuple[Path, Path]:
    """Return validated output paths for one beamsplitter specification.

    Parameters
    ----------
    spec:
        Parsed beamsplitter specification.

    Returns
    -------
    tuple[Path, Path]
        Pair ``(out0, out1)`` of validated output paths.

    Raises
    ------
    ValueError
        If either output path is missing.

    """
    if spec.out0 is None or spec.out1 is None:
        raise ValueError("Beamsplitter kernel requires both 'out0' and 'out1' paths.")
    return cast(Path, spec.out0), cast(Path, spec.out1)


def beamsplitter_poly_density(
    *,
    state: DensityPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> DensityPolyState:
    """Apply one or more beamsplitters to a polynomial density state.

    Parameters
    ----------
    state:
        Input polynomial density state.
    action:
        Semantic device action containing beamsplitter parameters.
    ctx:
        Unused for direct beamsplitter application. Included for API
        consistency with other kernels.

    Returns
    -------
    DensityPolyState
        Output density state after all requested beamsplitter operations.

    Notes
    -----
    This kernel performs the full backend-specific beamsplitter rewrite,
    including construction of output-path modes.

    """
    del ctx

    parsed = _parse_beamsplitter_params(action)
    modes_by_sig = state.mode_by_signature
    rho: DensityPolyProtocol = state.rho

    for spec in parsed.pairs:
        mode0 = modes_by_sig.get(spec.mode0_sig) if spec.mode0_sig is not None else None
        mode1 = modes_by_sig.get(spec.mode1_sig) if spec.mode1_sig is not None else None

        if mode0 is None and mode1 is None:
            continue

        if mode0 is None:
            if mode1 is None or spec.in0 is None:
                continue
            mode0 = _make_vacuum_partner_mode(
                existing_mode=mode1,
                input_path=cast(Path, spec.in0),
            )

        if mode1 is None:
            if mode0 is None or spec.in1 is None:
                continue
            mode1 = _make_vacuum_partner_mode(
                existing_mode=mode0,
                input_path=cast(Path, spec.in1),
            )

        out0, out1 = _require_output_paths(spec)

        rho = beamsplitter_densitypoly(
            rho,
            mode0=mode0,
            mode1=mode1,
            out0=out0,
            out1=out1,
            theta=spec.theta,
            phi_t=spec.phi_t,
            phi_r=spec.phi_r,
        )

    return DensityPolyState.from_densitypoly(cast(DensityPoly, rho))


def beamsplitter_poly_ket(
    *,
    state: KetPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> KetPolyState:
    """Apply one or more beamsplitters to a polynomial ket state.

    Parameters
    ----------
    state:
        Input polynomial ket state.
    action:
        Semantic device action containing beamsplitter parameters.
    ctx:
        Unused for direct beamsplitter application. Included for API
        consistency with other kernels.

    Returns
    -------
    KetPolyState
        Output ket state after all requested beamsplitter operations.

    Notes
    -----
    This kernel performs the full backend-specific beamsplitter rewrite,
    including construction of output-path modes.

    """
    del ctx

    parsed = _parse_beamsplitter_params(action)
    modes_by_sig = state.mode_by_signature
    ket = state.ket

    for spec in parsed.pairs:
        mode0 = modes_by_sig.get(spec.mode0_sig) if spec.mode0_sig is not None else None
        mode1 = modes_by_sig.get(spec.mode1_sig) if spec.mode1_sig is not None else None

        if mode0 is None and mode1 is None:
            continue

        if mode0 is None:
            if mode1 is None or spec.in0 is None:
                continue
            mode0 = _make_vacuum_partner_mode(
                existing_mode=mode1,
                input_path=cast(Path, spec.in0),
            )

        if mode1 is None:
            if mode0 is None or spec.in1 is None:
                continue
            mode1 = _make_vacuum_partner_mode(
                existing_mode=mode0,
                input_path=cast(Path, spec.in1),
            )

        out0, out1 = _require_output_paths(spec)

        ket = beamsplitter_ketpoly(
            ket,
            mode0=mode0,
            mode1=mode1,
            out0=out0,
            out1=out1,
            theta=spec.theta,
            phi_t=spec.phi_t,
            phi_r=spec.phi_r,
        )

    return KetPolyState.from_ketpoly(ket)
