r"""Generic unitary-dilation channels for symbolic density polynomials.

This module provides a reusable density-channel primitive based on a
unitary dilation followed by partial trace. It is intended as the
generic CPTP construction layer on top of passive linear mode rewrites.

A channel is applied as

.. math::

    \Phi(\rho)
    =
    \mathrm{Tr}_{E}\!\left[
        U\, \rho\, U^\dagger
    \right],

where ``U`` acts on an ordered tuple of modes and a chosen subset of
those modes is traced out afterward.

Notes
-----
This implementation is especially convenient when some of the modes in
the dilation are fresh environment modes that are implicitly treated as
vacuum by the polynomial rewrite semantics. This is the same pattern
used by the pure-loss channel implementation.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.polynomial.channels.primitives.linear_mode_unitary import (
    LinearModeMap,
    apply_to_densitypoly,
)


@dataclass(frozen=True)
class UnitaryDilation:
    """Specification of a unitary dilation on an ordered tuple of modes.

    Parameters
    ----------
    modes:
        Ordered modes on which the dilation unitary acts.
    U:
        Square matrix whose ordering matches ``modes``.
    trace_out_modes:
        Modes to trace out after the unitary action.
    check_unitary:
        If True, validate unitarity of ``U``.
    atol:
        Tolerance for optional unitary validation.

    Notes
    -----
    The mode ordering is the same convention as used by ``LinearModeMap``:
    column ``k`` of ``U`` gives the image of input mode ``k`` in the
    ordered output basis ``modes``.

    """

    modes: tuple[ModeOpProtocol, ...]
    U: np.ndarray
    trace_out_modes: tuple[ModeOpProtocol, ...]
    check_unitary: bool = False
    atol: float = 1e-10

    def __post_init__(self) -> None:
        """Validate dilation specification.

        Ensures that

        - all modes are distinct,
        - traced modes are distinct,
        - traced modes are contained in ``modes``.

        Optionally validates that ``U`` is unitary via ``LinearModeMap``.
        """
        mode_sigs = tuple(m.signature for m in self.modes)
        trace_sigs = tuple(m.signature for m in self.trace_out_modes)

        if len(set(mode_sigs)) != len(mode_sigs):
            raise ValueError("UnitaryDilation: modes must be distinct")

        if len(set(trace_sigs)) != len(trace_sigs):
            raise ValueError("UnitaryDilation: trace_out_modes must be distinct")

        mode_sig_set = set(mode_sigs)
        if any(sig not in mode_sig_set for sig in trace_sigs):
            raise ValueError("UnitaryDilation: every traced mode must appear in modes")

        LinearModeMap(
            modes=self.modes,
            U=self.U,
            check_unitary=self.check_unitary,
            atol=self.atol,
        )


def apply_unitary_dilation_densitypoly(
    rho: DensityPolyProtocol,
    *,
    dilation: UnitaryDilation,
    normalize_trace: bool = False,
    eps: float = 1e-14,
) -> DensityPolyProtocol:
    r"""Apply a generic unitary-dilation channel to a density polynomial.

    The channel is implemented by first applying the passive linear mode
    transformation associated with ``dilation.U`` on ``dilation.modes``,
    then tracing out ``dilation.trace_out_modes``:

    .. math::

        \Phi(\rho)
        =
        \mathrm{Tr}_{T}\!\left[
            U\, \rho\, U^\dagger
        \right],

    where :math:`T` is the subsystem spanned by the traced modes.

    Parameters
    ----------
    rho:
        Input symbolic density polynomial.
    dilation:
        Dilation specification.
    normalize_trace:
        If True, trace-normalize the reduced output density.
    eps:
        Threshold used by trace normalization if enabled.

    Returns
    -------
    DensityPoly
        Reduced density polynomial after the unitary action and partial trace.

    Notes
    -----
    This helper is the generic building block for channels such as
    pure loss, thermal-loss variants with suitable ancilla handling,
    and other ancilla-assisted linear bosonic channels.

    In the common "fresh environment mode" pattern, the traced modes
    are newly introduced ancilla modes not otherwise present in the
    retained system.

    """
    lmap = LinearModeMap(
        modes=dilation.modes,
        U=dilation.U,
        check_unitary=dilation.check_unitary,
        atol=dilation.atol,
    )

    rho_after = apply_to_densitypoly(rho, lmap=lmap)
    rho_red = rho_after.partial_trace(dilation.trace_out_modes)

    if normalize_trace:
        rho_red = rho_red.normalize_trace(eps=eps)
    return rho_red


def apply_unitary_dilation_densitypoly_direct(
    rho: DensityPolyProtocol,
    *,
    modes: tuple[ModeOpProtocol, ...],
    U: np.ndarray,
    trace_out_modes: tuple[ModeOpProtocol, ...],
    normalize_trace: bool = False,
    eps: float = 1e-14,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a unitary-dilation channel from explicit dilation data.

    This is a convenience wrapper around
    ``apply_unitary_dilation_densitypoly``. It constructs a
    ``UnitaryDilation`` object from the supplied modes, unitary matrix,
    and traced modes, and then applies the resulting channel.

    The induced channel is

    .. math::

        \Phi(\rho)
        =
        \operatorname{Tr}_{T}\!\left[
            U \rho U^\dagger
        \right],

    where the ordering of the matrix :math:`U` matches the ordering of
    ``modes``.

    Parameters
    ----------
    rho : DensityPolyProtocol
        Input symbolic density polynomial.
    modes : tuple of ModeOpProtocol
        Ordered tuple of modes on which the dilation unitary acts.
    U : ndarray
        Square matrix representing the passive linear transformation in
        the ordered basis given by ``modes``.
    trace_out_modes : tuple of ModeOpProtocol
        Modes to trace out after applying the unitary action.
    normalize_trace : bool, default=False
        If ``True``, normalize the reduced output state to unit trace after
        tracing out the selected modes.
    eps : float, default=1e-14
        Threshold used by the trace-normalization routine when
        ``normalize_trace`` is enabled.
    check_unitary : bool, default=False
        If ``True``, validate that ``U`` is unitary within the tolerance
        specified by ``atol``.
    atol : float, default=1e-10
        Absolute tolerance used when validating unitarity.

    Returns
    -------
    DensityPolyProtocol
        Reduced symbolic density polynomial obtained after the unitary
        action and partial trace.

    Notes
    -----
    This function is useful when the caller does not need to reuse a
    ``UnitaryDilation`` object and wants to specify the dilation data
    directly at the call site.

    See Also
    --------
    apply_unitary_dilation_densitypoly
        Apply a dilation channel from a preconstructed dilation object.
    UnitaryDilation
        Dataclass representing the dilation specification.

    """
    dilation = UnitaryDilation(
        modes=modes,
        U=U,
        trace_out_modes=trace_out_modes,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_unitary_dilation_densitypoly(
        rho,
        dilation=dilation,
        normalize_trace=normalize_trace,
        eps=eps,
    )
