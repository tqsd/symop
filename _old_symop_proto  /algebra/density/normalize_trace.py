from __future__ import annotations
from typing import Tuple
from symop_proto.core.protocols import DensityTermProto
from .trace import density_trace
from .scale import density_scale


def density_normalize_trace(
    terms: Tuple[DensityTermProto, ...], *, eps: float = 1e-14
) -> Tuple[DensityTermProto, ...]:
    r"""Scale a density polynomial so that its trace equals 1.

    This function computes :math:`\mathrm{Tr}(\rho)` via
    :func:`density_trace` and returns :math:`\rho' = \rho / \mathrm{Tr}(\rho)`.
    If :math:`|\mathrm{Tr}(\rho)| < \text{eps}`, a :class:`ValueError` is
    raised.

    Mathematically,

    .. math::

        \rho' \;=\; \frac{1}{\mathrm{Tr}(\rho)} \, \rho,
        \qquad
        \mathrm{Tr}(\rho') \;=\; 1.

    Parameters
    ----------
    terms : Tuple[DensityTermProto, ...]
        Input density polynomial :math:`\rho`.
    eps : float, keyword-only
        Minimum absolute threshold for :math:`|\mathrm{Tr}(\rho)|`. If the
        trace magnitude is smaller than ``eps``, normalization is refused.

    Returns
    -------
    Tuple[DensityTermProto, ...]
        The scaled density polynomial with unit trace.

    Raises
    ------
    ValueError
        If :math:`|\mathrm{Tr}(\rho)| < \text{eps}`.

    Notes
    -----
    - Scaling is performed by :func:`density_scale` with factor
      :math:`1/\mathrm{Tr}(\rho)`.
    - Works with complex traces; no assumption of Hermiticity is required.
    """

    tr = density_trace(terms)
    if abs(tr) < eps:
        raise ValueError(f"Cannot normalize trace: |Tr(rho)| < {eps}")
    return density_scale(terms, 1.0 / tr)
