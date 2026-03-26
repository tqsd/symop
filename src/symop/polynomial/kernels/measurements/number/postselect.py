r"""Postselection kernel for polynomial density-state number measurement.

This module implements deterministic (non-sampling) number postselection
for polynomial density states.

Given a desired measurement outcome, the kernel projects the input state
onto the corresponding number sector and returns the resulting branch,
along with its probability.

Both standard number postselection and joint per-port number postselection
are supported. The behavior is selected via
``action.resolution.grouping``.

Notes
-----
For density states, the probability of a postselected outcome is computed
as the trace of the projected density operator,

.. math::

    p = \mathrm{Tr}(\Pi \rho \Pi).

If the probability is strictly positive, the projected state is normalized
to unit trace. If the measurement is destructive, the measured subsystem
is discarded by tracing out the selected modes.

If the requested outcome has zero probability, the returned state is
``None``.

"""

from __future__ import annotations

from symop.devices.measurement.action import PostselectAction
from symop.devices.measurement.outcomes import JointOutcome
from symop.devices.measurement.result import PostselectionResult
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.kernels.measurements.number.common import (
    discard_measured_modes_number_density,
    project_onto_joint_number_poly_density,
    project_onto_number_poly_density,
    require_number_outcome,
)
from symop.polynomial.state.density import DensityPolyState


def postselect_number_detector_poly_density(
    *,
    state: DensityPolyState,
    action: PostselectAction,
    ctx: ApplyContextProtocol | None = None,
) -> PostselectionResult:
    r"""Postselect a photon-number outcome for a polynomial density state.

    Parameters
    ----------
    state:
        Polynomial density-state wrapper on which postselection is
        performed.
    action:
        Semantic postselect action defining the measurement target,
        desired outcome, resolution, and whether the measurement is
        destructive.
    ctx:
        Optional apply context. Not used by this kernel.

    Returns
    -------
    PostselectionResult
        Result containing the requested outcome, its probability, and the
        corresponding postselected state when the probability is nonzero.

    Raises
    ------
    TypeError
        If joint per-port postselection is requested but the provided
        outcome is not a :class:`JointOutcome`.

    Notes
    -----
    If ``action.resolution.grouping`` is
    :attr:`MeasurementGrouppingEnum.JOINT_PORTS`, joint per-port
    postselection is performed. In this case, ``action.outcome`` must be
    a :class:`JointOutcome`.

    Otherwise, standard number postselection is applied, and the outcome
    is validated using :func:`require_number_outcome`.

    The projection is performed structurally by keeping only those terms
    in the density polynomial that match the requested number sector.

    The probability of the postselected branch is computed as

    .. math::

        p = \mathrm{Tr}(\rho_{\mathrm{proj}}).

    If ``p <= 0``, the result contains ``state=None``.

    If ``p > 0``, the projected state is trace-normalized before being
    returned. If ``action.destructive`` is True, the measured modes are
    removed via partial trace.

    """
    if action.resolution.grouping == "joint_ports":
        outcome = action.outcome
        if not isinstance(outcome, JointOutcome):
            raise TypeError("Joint number postselection requires a JointOutcome.")
        projected = project_onto_joint_number_poly_density(
            state,
            action.target,
            outcome,
        )
    else:
        outcome = require_number_outcome(action.outcome)
        projected = project_onto_number_poly_density(
            state,
            action.target,
            outcome,
        )

    probability = float(projected.trace().real)

    if probability <= 0.0:
        return PostselectionResult(
            action=action,
            outcome=outcome,
            probability=0.0,
            state=None,
        )

    post_state = projected.normalize_trace()

    if action.destructive:
        post_state = discard_measured_modes_number_density(
            post_state,
            action.target,
        )

    return PostselectionResult(
        action=action,
        outcome=outcome,
        probability=probability,
        state=post_state,
    )
