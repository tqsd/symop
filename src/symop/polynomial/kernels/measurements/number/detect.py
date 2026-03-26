r"""Detection kernel for polynomial density-state number measurement.

This module implements stochastic number-detection evaluation for
polynomial density states.

The kernel resolves the number distribution associated with a semantic
:class:`~symop.devices.measurement.action.DetectAction`, samples an
outcome according to that distribution, projects the density state onto
the sampled outcome sector, and returns the corresponding
:class:`~symop.devices.measurement.result.DetectionResult`.

Both ordinary number detection and joint per-port number detection are
supported. The selected behavior is determined by the measurement
grouping specified in ``action.resolution.grouping``.

Notes
-----
For polynomial density states, the probability of a sampled outcome is
obtained from the trace of the projected density state. If the
measurement is destructive, the measured modes are discarded by tracing
them out after postselection and normalization.

Random sampling uses ``ctx.rng`` when the apply context implements
:class:`~symop.core.protocols.devices.apply_context.SupportsRng`.
Otherwise, a fresh ``random.Random`` instance is created.

Raises
------
ValueError
    If detection is attempted on an empty resolved number distribution.

"""

from __future__ import annotations

import random

from symop.core.protocols.devices.apply_context import SupportsRng
from symop.devices.measurement.action import DetectAction
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.result import DetectionResult
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.kernels.measurements.number.common import (
    discard_measured_modes_number_density,
    project_onto_joint_number_poly_density,
    project_onto_number_poly_density,
    resolve_joint_number_stats_poly_density,
    resolve_number_stats_poly_density,
)
from symop.polynomial.state.density import DensityPolyState


def detect_number_detector_poly_density(
    *,
    state: DensityPolyState,
    action: DetectAction,
    ctx: ApplyContextProtocol | None = None,
) -> DetectionResult:
    r"""Sample a number-detection outcome for a polynomial density state.

    Parameters
    ----------
    state:
        Polynomial density-state wrapper on which number detection is
        performed.
    action:
        Semantic detect action defining the measurement target,
        resolution, and whether the measurement is destructive.
    ctx:
        Optional apply context. If it implements :class:`SupportsRng`,
        its random-number generator is used for sampling.

    Returns
    -------
    DetectionResult
        Detection result containing the sampled outcome, the associated
        record, the probability of the sampled branch, and the postselected
        output state when that branch has nonzero weight.

    Raises
    ------
    ValueError
        If the resolved number distribution is empty and therefore no
        outcome can be sampled.

    Notes
    -----
    If ``action.resolution.grouping`` is ``"joint_ports"``, joint per-port
    number statistics are resolved and sampled. Otherwise, ordinary number
    statistics on the selected target are used.

    The sampled branch is obtained by projecting ``state`` onto the chosen
    outcome sector. Its branch probability is computed as

    .. math::

        p = \mathrm{Tr}(\rho_{\mathrm{proj}}).

    If the probability is strictly positive, the projected state is
    trace-normalized before being returned. If the action is destructive,
    the measured subsystem is then discarded by tracing out the measured
    modes.

    If the sampled branch has non-positive probability, the returned
    result contains ``state=None`` and probability ``0.0``.

    """
    rng = ctx.rng if isinstance(ctx, SupportsRng) else random.Random()

    sampled: MeasurementOutcome

    if action.resolution.grouping == "joint_ports":
        joint_probabilities = resolve_joint_number_stats_poly_density(
            state,
            action.target,
        )
        joint_items = list(joint_probabilities.items())
        if not joint_items:
            raise ValueError("Cannot detect from an empty number distribution.")

        joint_outcomes = [outcome for outcome, _ in joint_items]
        joint_weights = [prob for _, prob in joint_items]
        sampled_joint = rng.choices(joint_outcomes, weights=joint_weights, k=1)[0]

        projected = project_onto_joint_number_poly_density(
            state,
            action.target,
            sampled_joint,
        )
        sampled = sampled_joint
    else:
        stats = resolve_number_stats_poly_density(state, action.target)
        number_probabilities = stats.probabilities
        number_items = list(number_probabilities.items())
        if not number_items:
            raise ValueError("Cannot detect from an empty number distribution.")

        number_outcomes = [outcome for outcome, _ in number_items]
        number_weights = [prob for _, prob in number_items]
        sampled_number = rng.choices(number_outcomes, weights=number_weights, k=1)[0]

        projected = project_onto_number_poly_density(
            state,
            action.target,
            sampled_number,
        )
        sampled = sampled_number

    probability = float(projected.trace().real)

    if probability <= 0.0:
        return DetectionResult(
            action=action,
            record=sampled,
            outcome=sampled,
            probability=0.0,
            state=None,
        )

    post_state = projected.normalize_trace()

    if action.destructive:
        post_state = discard_measured_modes_number_density(
            post_state,
            action.target,
        )

    return DetectionResult(
        action=action,
        record=sampled,
        outcome=sampled,
        probability=probability,
        state=post_state,
    )
