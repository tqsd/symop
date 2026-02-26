from __future__ import annotations

from symop_proto.core.protocols import DensityTermProto


def density_scale(
    terms: tuple[DensityTermProto, ...], c: complex
) -> tuple[DensityTermProto, ...]:
    r"""Scale all density terms by a complex coefficient.

    Multiplies each term's coefficient by :math:`c` and returns a new tuple.

    Parameters
    ----------
    terms : Tuple[DensityTermProto, ...]
        Input density polynomial.
    c : complex
        Complex scalar multiplier.

    Returns
    -------
    Tuple[DensityTermProto, ...]
        New density terms with coefficients scaled by :math:`c`.

    """
    from symop_proto.core.terms import DensityTerm

    return tuple(
        DensityTerm(coeff=c * t.coeff, left=t.left, right=t.right) for t in terms
    )
