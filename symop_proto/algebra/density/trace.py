from __future__ import annotations
from typing import Tuple
from symop_proto.core.protocols import DensityTermProto
from .overlap_right_left import overlap_right_left


def density_trace(terms: Tuple[DensityTermProto, ...]) -> complex:
    r"""Compute the trace of a density polynomial.

    Evaluates

    .. math::

        \mathrm{Tr}(\rho) = \sum_i c_i \langle L_i | R_i \rangle,

    where each term corresponds to :math:`c_i |L_i\rangle\langle R_i|`.

    Parameters
    ----------
    terms : Tuple[DensityTermProto, ...]
        Density terms forming the operator.

    Returns
    -------
    complex
        Complex trace value.
    """
    total: complex = 0.0 + 0.0j
    for t in terms:
        total += t.coeff * overlap_right_left(t.right, t.left)
    return total
