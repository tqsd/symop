from symop_proto.core.operators import LadderOp, ModeOp, OperatorKind
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.rewrites.functions.substitution import (
    rewrite_densitypoly,
    rewrite_ketpoly,
)
from symop_proto.state.polynomial_state import KetPolyState


def rename_path(state, old_path: str, new_path: str):
    """Return state with all ModeOp path labels old_path -> new_path.
    TODO: solve the typing issues (Protocols)
    """

    def _subst(op):
        if not isinstance(op, LadderOp):
            return [(1.0 + 0j, op)]

        mode = op.mode
        label = mode.label
        if isinstance(label, ModeLabel) and label.path.name == old_path:
            new_mode = ModeOp(
                env=mode.env, label=ModeLabel(PathLabel(new_path), label.pol)
            )
            new_op = (
                new_mode.create
                if op.kind is OperatorKind.CREATE
                else new_mode.ann
            )
            return [(1.0 + 0j, new_op)]
        return [(1.0 + 0j, op)]

    return (
        state.__class__(rewrite_ketpoly(state.ket, _subst))
        if isinstance(state, KetPolyState)
        else state.__class__(rewrite_densitypoly(state.rho, _subst))
    )
