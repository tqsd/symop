from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from symop_proto.core.protocols import HasSignature, ModeOpProto, PathProto
from symop_proto.devices.io import DeviceIO
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset
from symop_proto.gaussian.ops.utils import permutation_matrix


def relabel_modes_with_paths(
    modes: Sequence[ModeOpProto],
    *,
    paths: Sequence[object],
) -> Tuple[ModeOpProto, ...]:
    """
    Create output modes by copying each input mode and replacing its path label.
    Assumes ModeOpProto supports .with_path(...).
    """
    if len(paths) != len(modes):
        raise ValueError("paths must have same length as modes")
    out: list[ModeOpProto] = []
    for m, p in zip(modes, paths):
        out.append(m.with_path(p))  # relies on your ModeOp API
    return tuple(out)


def require_arity(
    modes: Sequence[ModeOpProto], *, k: int, name: str
) -> Tuple[ModeOpProto, ...]:
    modes_t = tuple(modes)
    if len(modes_t) != k:
        raise ValueError(f"{name} requires exactly {k} modes, got {len(modes_t)}")
    return modes_t


def build_reroute_io(
    *,
    input_modes: Tuple[ModeOpProto, ...],
    output_modes: Tuple[ModeOpProto, ...],
    env_modes: Tuple[ModeOpProto, ...] = (),
    meta: Optional[dict] = None,
) -> DeviceIO:
    return DeviceIO(
        input_modes=input_modes,
        output_modes=output_modes,
        env_modes=env_modes,
        meta={} if meta is None else dict(meta),
    )


def swap_inputs_into_outputs_passive(
    core: GaussianCore,
    *,
    input_modes: Tuple[ModeOpProto, ...],
    output_modes: Tuple[ModeOpProto, ...],
    trace_inputs: bool = True,
    check_unitary: bool = False,
    atol: float = 1e-12,
) -> GaussianCore:
    """
    Generic routing primitive:
    - ensure output modes exist by extending with vacuum
    - apply passive unitary on subsystem (inputs + outputs) that swaps them
    - optionally trace out the old inputs

    This implements "device changes mode identity".
    """
    k = len(input_modes)
    if len(output_modes) != k:
        raise ValueError("input_modes and output_modes must have same length")

    # add outputs as vacuum if not present
    core2 = core.extend_with_vacuum(output_modes)

    # subsystem ordering: [in0..in{k-1}, out0..out{k-1}]
    joint = input_modes + output_modes
    idx = [core2.basis.require_index_of(m) for m in joint]

    # permutation matrix on the joint annihilator vector:
    # (in, out) -> (out, in)
    p = list(range(k, 2 * k)) + list(range(0, k))
    U = permutation_matrix(p)  # (2k,2k) on annihilators of joint ordering

    core3 = apply_passive_unitary_subset(
        core2, idx=idx, U=U, check_unitary=check_unitary, atol=atol
    )

    if trace_inputs:
        core3 = core3.trace_out(input_modes)
    return core3


def select_modes(
    core: GaussianCore,
    *,
    path: Optional[HasSignature] = None,
    pol: Optional[HasSignature] = None,
    predicate: Optional[Callable[[ModeOpProto], bool]] = None,
) -> Tuple[ModeOpProto, ...]:
    """
    Select modes from the state's basis by label metadata.

    `path` and `pol` are compared via signature (so can be label objects).
    """
    out = []
    for m in core.basis.modes:
        ok = True
        if path is not None:
            ok = ok and (m.label.path.signature == path.signature)
        if pol is not None:
            ok = ok and (m.label.pol.signature == pol.signature)
        if predicate is not None:
            ok = ok and bool(predicate(m))
        if ok:
            out.append(m)
    return tuple(out)


def require_nonempty(
    modes: Sequence[ModeOpProto], *, what: str
) -> Tuple[ModeOpProto, ...]:
    mt = tuple(modes)
    if len(mt) == 0:
        raise ValueError(f"no modes selected for {what}")
    return mt


def relabel_modes_with_path(
    modes: Sequence[ModeOpProto], *, out_path: PathProto
) -> Tuple[ModeOpProto, ...]:
    return tuple(m.with_path(out_path) for m in modes)
