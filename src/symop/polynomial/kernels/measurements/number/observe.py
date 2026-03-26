r"""Observation kernels for polynomial number measurement.

This module implements non-destructive number measurement for polynomial
ket and density states.

Unlike detection (sampling), observation evaluates the full probability
distribution over measurement outcomes without collapsing the state.
The result is returned as an :class:`ObservationResult`.

Both standard number measurement and joint per-port number measurement
are supported. The behavior is selected via
``action.resolution.grouping``.

Notes
-----
For ket states, probabilities are computed from squared norms of projected
states. For density states, probabilities are obtained from traces of the
corresponding projected density operators.

When joint per-port grouping is used, no scalar expectation value is
returned, since the outcome space is multi-dimensional.

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from symop.devices.measurement.action import ObserveAction
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.result import ObservationResult
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.kernels.measurements.number.common import (
    resolve_joint_number_stats_poly_density,
    resolve_joint_number_stats_poly_ket,
    resolve_number_stats_poly_density,
    resolve_number_stats_poly_ket,
)
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


def observe_number_detector_poly_ket(
    *,
    state: KetPolyState,
    action: ObserveAction,
    ctx: ApplyContextProtocol | None = None,
) -> ObservationResult:
    r"""Resolve number-observation statistics for a ket state.

    Parameters
    ----------
    state:
        Polynomial ket-state wrapper to be evaluated.
    action:
        Semantic observe action defining the measurement target and
        resolution.
    ctx:
        Optional apply context. Not used by this kernel.

    Returns
    -------
    ObservationResult
        Observation result containing the probability distribution over
        number outcomes and, when applicable, the expectation value.

    Notes
    -----
    If ``action.resolution.grouping`` is
    :attr:`MeasurementGrouppingEnum.JOINT_PORTS`, joint per-port number
    statistics are resolved and returned. In this case, the expectation
    value is set to ``None``.

    Otherwise, standard number statistics are computed, and the expectation
    value is given by

    .. math::

        \mathbb{E}[N] = \sum_n n\,p(n).

    The state is not modified by this operation.

    """
    if action.resolution.grouping == "joint_ports":
        probabilities = cast(
            Mapping[MeasurementOutcome, float],
            resolve_joint_number_stats_poly_ket(
                state,
                action.target,
            ),
        )
        expectation = None
    else:
        stats = resolve_number_stats_poly_ket(state, action.target)
        probabilities = cast(Mapping[MeasurementOutcome, float], stats.probabilities)
        expectation = stats.expectation

    return ObservationResult(
        action=action,
        probabilities=cast(
            dict[MeasurementOutcome, float],
            dict(probabilities),
        ),
        expectation=expectation,
    )


def observe_number_detector_poly_density(
    *,
    state: DensityPolyState,
    action: ObserveAction,
    ctx: ApplyContextProtocol | None = None,
) -> ObservationResult:
    r"""Resolve number-observation statistics for a density state.

    Parameters
    ----------
    state:
        Polynomial density-state wrapper to be evaluated.
    action:
        Semantic observe action defining the measurement target and
        resolution.
    ctx:
        Optional apply context. Not used by this kernel.

    Returns
    -------
    ObservationResult
        Observation result containing the probability distribution over
        number outcomes and, when applicable, the expectation value.

    Notes
    -----
    If ``action.resolution.grouping`` is
    :attr:`MeasurementGrouppingEnum.JOINT_PORTS`, joint per-port number
    statistics are resolved and returned. In this case, the expectation
    value is set to ``None``.

    Otherwise, standard number statistics are computed, and the expectation
    value is given by

    .. math::

        \mathbb{E}[N] = \sum_n n\,p(n).

    For density states, probabilities are computed from traces of
    projected density operators,

    .. math::

        p(n) = \mathrm{Tr}(\Pi_n \rho \Pi_n).

    The state is not modified by this operation.

    """
    if action.resolution.grouping == "joint_ports":
        probabilities = cast(
            Mapping[MeasurementOutcome, float],
            resolve_joint_number_stats_poly_density(
                state,
                action.target,
            ),
        )
        expectation = None
    else:
        stats = resolve_number_stats_poly_density(state, action.target)
        probabilities = cast(Mapping[MeasurementOutcome, float], stats.probabilities)
        expectation = stats.expectation

    return ObservationResult(
        action=action,
        probabilities=cast(
            dict[MeasurementOutcome, float],
            dict(probabilities),
        ),
        expectation=expectation,
    )
