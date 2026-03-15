"""Mode relabeling utilities for symbolic CCR term structures.

This module provides helpers for replacing the mode attached to ladder
operators inside symbolic ket and density terms. Relabeling is performed
by matching the signature of each operator's mode against a mapping from
old signatures to replacement modes.

The implementation is structure preserving:

- ladder operators are recreated only when their mode changes,
- monomials are recreated only when at least one contained operator changes,
- terms are recreated only when one of their monomials changes.

This minimizes allocations while keeping the symbolic objects immutable.

The relabeling routines assume that the involved operator, monomial, and
term objects are dataclass instances so that ``dataclasses.replace`` can
be used to construct updated copies.

Notes
-----
The mapping is keyed by ``Signature`` rather than by mode object identity.
This allows structurally equivalent modes to be relabeled consistently
even when distinct mode instances share the same signature.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass, replace

from symop.core.protocols.ops.operators import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops.operators import (
    ModeOp as ModeOpProtocol,
)
from symop.core.terms.density_term import DensityTerm
from symop.core.terms.ket_term import KetTerm
from symop.core.types.signature import Signature


def _maybe_replace_mode(
    op: LadderOpProtocol, mode_map: Mapping[Signature, ModeOpProtocol]
) -> LadderOpProtocol:
    """Replace the mode of a ladder operator if its signature is mapped.

    Parameters
    ----------
    op :
        Ladder operator whose mode may be replaced.
    mode_map :
        Mapping from old mode signatures to replacement mode objects.

    Returns
    -------
    LadderOpProtocol
        The original operator if its mode signature is not present in
        ``mode_map``; otherwise a copied operator with the updated mode.

    Raises
    ------
    TypeError
        If a replacement is required but the operator is not a dataclass
        instance and therefore cannot be copied with ``replace``.

    """
    old_mode = op.mode
    new_mode = mode_map.get(old_mode.signature)
    if new_mode is None:
        return op

    if not is_dataclass(op):
        raise TypeError(
            "Cannot relabel ladder operator: expected a dataclass instance."
        )
    return replace(op, mode=new_mode)


def _map_ops(
    ops: tuple[LadderOpProtocol, ...],
    mode_map: Mapping[Signature, ModeOpProtocol],
) -> tuple[LadderOpProtocol, ...]:
    """Relabel the modes of all ladder operators in a tuple.

    Parameters
    ----------
    ops :
        Tuple of ladder operators to relabel.
    mode_map :
        Mapping from old mode signatures to replacement mode objects.

    Returns
    -------
    tuple of LadderOpProtocol
        Tuple containing relabeled operators. If no operator changes,
        the original tuple is returned.

    """
    changed = False
    out: list[LadderOpProtocol] = []
    for op in ops:
        op2 = _maybe_replace_mode(op, mode_map)
        if op2 is not op:
            changed = True
        out.append(op2)
    return tuple(out) if changed else ops


def ket_relabel_modes(
    terms: tuple[KetTerm, ...],
    *,
    mode_map: Mapping[Signature, ModeOpProtocol],
) -> tuple[KetTerm, ...]:
    """Relabel modes appearing in ket terms.

    Each ladder operator in the creators and annihilators of every ket
    term monomial is inspected. If the operator's mode signature is
    present in ``mode_map``, the operator is copied with the replacement
    mode.

    Parameters
    ----------
    terms :
        Ket terms whose monomials should be relabeled.
    mode_map :
        Mapping from old mode signatures to replacement mode objects.

    Returns
    -------
    tuple of KetTerm
        Tuple of ket terms with updated mode references. Terms whose
        operators are unchanged are returned unchanged.

    Raises
    ------
    TypeError
        If a monomial or term needs to be updated but is not a dataclass
        instance.

    Notes
    -----
    ``mode_map`` is interpreted as

    ``old_mode.signature -> new_mode``.

    """
    if not mode_map:
        return terms

    out_terms: list[KetTerm] = []
    for t in terms:
        m = t.monomial

        old_creators = tuple(m.creators)
        old_annihilators = tuple(m.annihilators)

        creators2 = _map_ops(old_creators, mode_map)
        annihilators2 = _map_ops(old_annihilators, mode_map)

        if creators2 is old_creators and annihilators2 is old_annihilators:
            out_terms.append(t)
            continue

        if not is_dataclass(m):
            raise TypeError(
                "ket_relabel_modes requires monomial to be a dataclass"
                "to replace creators/annihilators."
            )
        m2 = replace(m, creators=creators2, annihilators=annihilators2)

        if not is_dataclass(t):
            raise TypeError(
                "ket_relabel_modes requires ket term to be "
                "a dataclass to replace monomial."
            )
        t2 = replace(t, monomial=m2)
        out_terms.append(t2)

    return tuple(out_terms)


def density_relabel_modes(
    terms: tuple[DensityTerm, ...],
    *,
    mode_map: Mapping[Signature, ModeOpProtocol],
) -> tuple[DensityTerm, ...]:
    """Relabel modes appearing in density terms.

    Each ladder operator in the left and right monomials of every density
    term is inspected. Operators whose mode signature appears in
    ``mode_map`` are copied with the replacement mode.

    Parameters
    ----------
    terms :
        Density terms whose left and right monomials should be relabeled.
    mode_map :
        Mapping from old mode signatures to replacement mode objects.

    Returns
    -------
    tuple of DensityTerm
        Tuple of density terms with updated mode references. Terms whose
        monomials are unchanged are returned unchanged.

    Raises
    ------
    TypeError
        If a density term or one of its monomials needs to be updated but
        is not a dataclass instance.

    Notes
    -----
    This function assumes that each density term exposes dataclass-like
    ``left`` and ``right`` monomials, and that each monomial exposes
    ``creators`` and ``annihilators`` fields compatible with
    ``dataclasses.replace``.

    """
    if not mode_map:
        return terms

    out_terms: list[DensityTerm] = []
    for t in terms:
        left = t.left
        right = t.right

        old_left_creators = tuple(left.creators)
        old_left_annihilators = tuple(left.annihilators)
        old_right_creators = tuple(right.creators)
        old_right_annihilators = tuple(right.annihilators)

        left_creators2 = _map_ops(old_left_creators, mode_map)
        left_ann2 = _map_ops(old_left_annihilators, mode_map)
        right_creators2 = _map_ops(old_right_creators, mode_map)
        right_ann2 = _map_ops(old_right_annihilators, mode_map)

        left_changed = (
            left_creators2 is not old_left_creators
            or left_ann2 is not old_left_annihilators
        )
        right_changed = (
            right_creators2 is not old_right_creators
            or right_ann2 is not old_right_annihilators
        )

        if not left_changed and not right_changed:
            out_terms.append(t)
            continue

        if not is_dataclass(left) or not is_dataclass(right) or not is_dataclass(t):
            raise TypeError(
                "density_relabel_modes requires density term and its monomials to be dataclasses."
            )

        left2 = replace(left, creators=left_creators2, annihilators=left_ann2)
        right2 = replace(right, creators=right_creators2, annihilators=right_ann2)
        t2 = replace(t, left=left2, right=right2)
        out_terms.append(t2)

    return tuple(out_terms)
