r"""Common helpers for polynomial number measurement.

This module implements structural utilities used by polynomial-number
measurement kernels for both ket and density-state wrappers.

The helpers in this module support three main tasks:

- resolving semantic measurement targets into concrete polynomial modes
- enumerating number-measurement support on selected modes or ports
- projecting states onto number or joint-number outcome sectors

The implementation is structural: it infers photon-number content by
counting creation operators associated with the selected mode signatures.
For density-polynomial inputs, projective number statistics are only
resolved from terms whose left and right number counts agree on the
measured subsystem.

Notes
-----
The routines here do not define user-facing measurement devices.
Instead, they provide the low-level logic reused by observe, detect,
and postselect kernels for number measurement.

For ket states, destructive removal of measured modes is not implemented
directly. A destructive number measurement should first be represented as
a density state and then reduced by tracing out the measured subsystem.

Raises
------
NumberMeasurementError
    Raised when number measurement cannot be evaluated structurally, for
    example because a term violates assumptions required by the current
    implementation.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.protocols.base.signature import Signature
from symop.core.protocols.ops.monomial import Monomial as MonomialProtocol
from symop.core.protocols.ops.operators import ModeOp as ModeOpProtocol
from symop.core.terms import DensityTerm, KetTerm
from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome
from symop.devices.measurement.target import MeasurementTarget
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


class NumberMeasurementError(ValueError):
    r"""Raised when number measurement cannot be evaluated structurally.

    This exception indicates that the polynomial state or term structure
    does not satisfy the assumptions required by the current number
    measurement implementation.

    Examples include creator-only requirements being violated in places
    where structural number counting is expected to act on normally
    ordered creation monomials.

    """


@dataclass(frozen=True)
class NumberMeasurementStats:
    r"""Resolved number statistics on a selected target.

    Parameters
    ----------
    probabilities:
        Mapping from number outcomes to their corresponding probabilities.

    Notes
    -----
    The probabilities are expected to represent the distribution of the
    measured number observable restricted to a selected subsystem.

    """

    probabilities: Mapping[NumberOutcome, float]

    @property
    def expectation(self) -> float:
        r"""Return the expected number of quanta.

        Returns
        -------
        float
            Expectation value of the resolved number distribution,
            computed as

            .. math::

                \mathbb{E}[N] = \sum_n n\,p(n).

        """
        return sum(outcome.count * prob for outcome, prob in self.probabilities.items())


def selected_modes(
    state: DensityPolyState | KetPolyState,
    target: MeasurementTarget,
) -> tuple[ModeOpProtocol, ...]:
    r"""Resolve a measurement target into concrete selected modes.

    Parameters
    ----------
    state:
        Polynomial ket or density-state wrapper on which the target is
        resolved.
    target:
        Semantic measurement target specifying paths and/or explicit mode
        signatures.

    Returns
    -------
    tuple[ModeOpProtocol, ...]
        Concrete modes selected by the target.

    Notes
    -----
    This is a thin convenience wrapper around
    :meth:`state.resolve_target_modes`.

    """
    return state.resolve_target_modes(target)


def selected_mode_signatures(
    state: DensityPolyState | KetPolyState,
    target: MeasurementTarget,
) -> frozenset[Signature]:
    r"""Return the signatures of all modes selected by a target.

    Parameters
    ----------
    state:
        Polynomial ket or density-state wrapper on which the target is
        resolved.
    target:
        Semantic measurement target specifying paths and/or explicit mode
        signatures.

    Returns
    -------
    frozenset[Signature]
        Frozen set of signatures for all selected concrete modes.

    Notes
    -----
    Duplicate selections are removed by construction.

    """
    return frozenset(mode.signature for mode in selected_modes(state, target))


def selected_mode_signatures_by_port(
    state: DensityPolyState | KetPolyState,
    target: MeasurementTarget,
) -> dict[str, frozenset[Signature]]:
    r"""Return selected mode signatures grouped by semantic port.

    Parameters
    ----------
    state:
        Polynomial ket or density-state wrapper on which the target is
        resolved.
    target:
        Semantic measurement target containing per-port selections.

    Returns
    -------
    dict[str, frozenset[Signature]]
        Mapping from port name to the set of selected mode signatures for
        that port.

    Notes
    -----
    Within each port selection, modes may be specified by path and/or by
    explicit signature. Duplicate references are collapsed to a single
    signature entry.

    """
    resolved: dict[str, frozenset[Signature]] = {}

    for selection in target.selections:
        selected: dict[Signature, ModeOpProtocol] = {}

        for path in selection.paths:
            for mode in state.modes_on_path(path):
                selected[mode.signature] = mode

        for mode_sig in selection.mode_sigs:
            maybe_mode = state.mode_by_signature.get(mode_sig)
            if maybe_mode is not None:
                selected[mode.signature] = maybe_mode

        resolved[selection.port_name] = frozenset(selected.keys())

    return resolved


def require_number_outcome(outcome: object) -> NumberOutcome:
    r"""Validate that an outcome is a number outcome.

    Parameters
    ----------
    outcome:
        Outcome object to validate.

    Returns
    -------
    NumberOutcome
        The validated number outcome.

    Raises
    ------
    TypeError
        If ``outcome`` is not an instance of :class:`NumberOutcome`.

    """
    if not isinstance(outcome, NumberOutcome):
        raise TypeError("Projective number measurement requires a NumberOutcome.")
    return outcome


def count_selected_creators_by_port(
    monomial: MonomialProtocol,
    measured_sigs_by_port: Mapping[str, frozenset[Signature]],
) -> tuple[tuple[str, int], ...]:
    r"""Count selected creation operators per port in a monomial.

    Parameters
    ----------
    monomial:
        Monomial whose creation operators are to be counted.
    measured_sigs_by_port:
        Mapping from port name to the set of selected mode signatures for
        that port.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Tuple of ``(port_name, count)`` pairs giving the number of
        creation operators in ``monomial`` whose mode signatures belong
        to each port selection.

    Notes
    -----
    Counting is performed only over ``monomial.creators``.

    """
    return tuple(
        (
            port_name,
            sum(1 for op in monomial.creators if op.mode.signature in measured_sigs),
        )
        for port_name, measured_sigs in measured_sigs_by_port.items()
    )


def joint_outcome_from_counts(
    counts_by_port: tuple[tuple[str, int], ...],
) -> JointOutcome:
    r"""Build a joint number outcome from per-port counts.

    Parameters
    ----------
    counts_by_port:
        Tuple of ``(port_name, count)`` pairs.

    Returns
    -------
    JointOutcome
        Joint outcome whose per-port outcomes are number outcomes with the
        supplied counts.

    """
    return JointOutcome(
        outcomes_by_port=tuple(
            (port_name, NumberOutcome(count)) for port_name, count in counts_by_port
        )
    )


def counts_from_joint_outcome(
    outcome: JointOutcome,
) -> tuple[tuple[str, int], ...]:
    r"""Extract per-port number counts from a joint outcome.

    Parameters
    ----------
    outcome:
        Joint outcome to unpack.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Tuple of ``(port_name, count)`` pairs.

    Raises
    ------
    TypeError
        If any per-port outcome is not a :class:`NumberOutcome`.

    """
    counts: list[tuple[str, int]] = []

    for port_name, port_outcome in outcome.outcomes_by_port:
        if not isinstance(port_outcome, NumberOutcome):
            raise TypeError(
                "Joint number measurement requires NumberOutcome at every port."
            )
        counts.append((port_name, port_outcome.count))

    return tuple(counts)


# KET COMMON --------------------------------------------------


def count_selected_quanta_ket_term(
    term: KetTerm,
    measured_sigs: frozenset[Signature],
) -> int:
    r"""Count selected quanta in a ket term.

    Parameters
    ----------
    term:
        Ket term whose monomial is to be inspected.
    measured_sigs:
        Signatures of the selected measured modes.

    Returns
    -------
    int
        Number of creation operators in ``term`` whose mode signatures are
        contained in ``measured_sigs``.

    """
    return count_selected_creators_in_monomial(measured_sigs, term.monomial)


def count_selected_creators_in_monomial(
    measured_sigs: frozenset[Signature],
    monomial: MonomialProtocol,
) -> int:
    r"""Count selected creation operators in a monomial.

    Parameters
    ----------
    measured_sigs:
        Signatures of the selected measured modes.
    monomial:
        Monomial whose creation operators are to be counted.

    Returns
    -------
    int
        Number of creation operators in ``monomial`` whose mode signatures
        belong to ``measured_sigs``.

    """
    return sum(1 for op in monomial.creators if op.mode.signature in measured_sigs)


def iter_number_support_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
) -> tuple[NumberOutcome, ...]:
    r"""Enumerate number outcomes supported by a ket state.

    Parameters
    ----------
    state:
        Ket-state wrapper whose number support should be resolved.
    target:
        Semantic measurement target specifying the measured subsystem.

    Returns
    -------
    tuple[NumberOutcome, ...]
        Sorted tuple of number outcomes that occur across the ket terms of
        the selected subsystem.

    Notes
    -----
    Support is determined structurally by counting selected creation
    operators in each ket term.

    """
    measured_sigs = selected_mode_signatures(state, target)
    counts = {
        count_selected_quanta_ket_term(term, measured_sigs) for term in state.ket.terms
    }
    return tuple(NumberOutcome(count=n) for n in sorted(counts))


def project_onto_number_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
    outcome: NumberOutcome,
    *,
    eps: float = 1e-12,
) -> KetPolyState:
    r"""Project a ket state onto a selected number outcome.

    Parameters
    ----------
    state:
        Ket-state wrapper to project.
    target:
        Semantic measurement target specifying the measured subsystem.
    outcome:
        Number outcome defining the desired number sector.
    eps:
        Numerical threshold forwarded to term combination.

    Returns
    -------
    KetPolyState
        Projected ket state containing only terms whose selected number
        count matches ``outcome.count``.

    Notes
    -----
    Projection is implemented structurally by keeping only those ket terms
    with the requested number of selected creation operators.

    """
    measured_sigs = selected_mode_signatures(state, target)
    n_target = outcome.count

    kept_terms = tuple(
        term
        for term in state.ket.terms
        if count_selected_quanta_ket_term(term, measured_sigs) == n_target
    )

    ket_proj = state.ket.__class__(kept_terms).combine_like_terms(eps=eps)
    return KetPolyState.from_ketpoly(ket_proj)


def resolve_number_stats_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
    *,
    eps: float = 1e-12,
) -> NumberMeasurementStats:
    r"""Resolve projective number statistics for a ket state.

    Parameters
    ----------
    state:
        Ket-state wrapper whose number statistics should be resolved.
    target:
        Semantic measurement target specifying the measured subsystem.
    eps:
        Numerical tolerance used to discard negligible probabilities and
        to guard against small negative numerical artifacts.

    Returns
    -------
    NumberMeasurementStats
        Resolved probability distribution over number outcomes.

    Raises
    ------
    NumberMeasurementError
        If a probability smaller than ``-eps`` is encountered.

    Notes
    -----
    For each supported outcome :math:`n`, the probability is obtained from
    the squared norm of the projected ket state,

    .. math::

        p(n) = \lVert \Pi_n |\psi\rangle \rVert^2.

    """
    probabilities: dict[NumberOutcome, float] = {}

    for outcome in iter_number_support_poly_ket(state, target):
        projected = project_onto_number_poly_ket(
            state,
            target,
            outcome,
            eps=eps,
        )
        prob = projected.norm2

        if abs(prob) <= eps:
            continue
        if prob < -eps:
            raise NumberMeasurementError(
                f"Encountered negative probability for {outcome}: {prob}."
            )
        if prob < 0:
            prob = 0.0

        probabilities[outcome] = float(prob)

    return NumberMeasurementStats(probabilities)


def discard_measured_modes_number_ket(
    state: KetPolyState,
    target: MeasurementTarget,
) -> KetPolyState:
    r"""Discard measured modes after number measurement on a ket state.

    Parameters
    ----------
    state:
        Ket-state wrapper from which measured modes would be discarded.
    target:
        Semantic measurement target specifying the measured subsystem.

    Returns
    -------
    KetPolyState
        This function does not return normally.

    Raises
    ------
    NotImplementedError
        Always raised, because destructive number measurement is not
        implemented directly for ket states.

    Notes
    -----
    Destructive number measurement should first be represented at the
    density-state level and then reduced by tracing out the measured
    subsystem.

    """
    raise NotImplementedError(
        "Destructive number measurement on ket states is not implemented "
        "directly. Convert to density first."
    )


def joint_number_counts_for_ket_term(
    term: KetTerm,
    measured_sigs_by_port: Mapping[str, frozenset[Signature]],
) -> tuple[tuple[str, int], ...]:
    r"""Count selected quanta per port for a ket term.

    Parameters
    ----------
    term:
        Ket term whose monomial is to be inspected.
    measured_sigs_by_port:
        Mapping from port name to selected mode signatures.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Tuple of ``(port_name, count)`` pairs describing the selected
        number counts for the term on each port.

    """
    return count_selected_creators_by_port(
        term.monomial,
        measured_sigs_by_port,
    )


def iter_joint_number_support_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
) -> tuple[JointOutcome, ...]:
    r"""Enumerate joint number outcomes supported by a ket state.

    Parameters
    ----------
    state:
        Ket-state wrapper whose joint number support should be resolved.
    target:
        Semantic measurement target containing per-port selections.

    Returns
    -------
    tuple[JointOutcome, ...]
        Sorted tuple of joint outcomes occurring across the ket terms.

    Notes
    -----
    Support is determined structurally by counting selected creation
    operators port by port for each ket term.

    """
    measured_sigs_by_port = selected_mode_signatures_by_port(state, target)

    counts_seen = {
        joint_number_counts_for_ket_term(term, measured_sigs_by_port)
        for term in state.ket.terms
    }

    return tuple(joint_outcome_from_counts(counts) for counts in sorted(counts_seen))


def project_onto_joint_number_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
    outcome: JointOutcome,
    *,
    eps: float = 1e-12,
) -> KetPolyState:
    r"""Project a ket state onto a selected joint number outcome.

    Parameters
    ----------
    state:
        Ket-state wrapper to project.
    target:
        Semantic measurement target containing per-port selections.
    outcome:
        Joint number outcome defining the desired per-port number sector.
    eps:
        Numerical threshold forwarded to term combination.

    Returns
    -------
    KetPolyState
        Projected ket state containing only terms whose per-port selected
        counts match the requested joint outcome.

    """
    measured_sigs_by_port = selected_mode_signatures_by_port(state, target)
    expected_counts = counts_from_joint_outcome(outcome)

    kept_terms = tuple(
        term
        for term in state.ket.terms
        if joint_number_counts_for_ket_term(term, measured_sigs_by_port)
        == expected_counts
    )

    ket_proj = state.ket.__class__(kept_terms).combine_like_terms(eps=eps)
    return KetPolyState.from_ketpoly(ket_proj)


def resolve_joint_number_stats_poly_ket(
    state: KetPolyState,
    target: MeasurementTarget,
    *,
    eps: float = 1e-12,
) -> Mapping[JointOutcome, float]:
    r"""Resolve joint number statistics for a ket state.

    Parameters
    ----------
    state:
        Ket-state wrapper whose joint number statistics should be resolved.
    target:
        Semantic measurement target containing per-port selections.
    eps:
        Numerical tolerance used to discard negligible probabilities and
        to guard against small negative numerical artifacts.

    Returns
    -------
    Mapping[JointOutcome, float]
        Mapping from joint number outcomes to probabilities.

    Raises
    ------
    NumberMeasurementError
        If a probability smaller than ``-eps`` is encountered.

    Notes
    -----
    For each supported joint outcome, the probability is given by the
    squared norm of the corresponding projected ket state.

    """
    probabilities: dict[JointOutcome, float] = {}

    for outcome in iter_joint_number_support_poly_ket(state, target):
        projected = project_onto_joint_number_poly_ket(
            state,
            target,
            outcome,
            eps=eps,
        )
        prob = projected.norm2

        if abs(prob) <= eps:
            continue
        if prob < -eps:
            raise NumberMeasurementError(
                f"Encountered negative probability for {outcome}: {prob}."
            )
        if prob < 0:
            prob = 0.0

        probabilities[outcome] = float(prob)

    return probabilities


# DENSITY COMMON --------------------------------------------------


def count_selected_quanta_left(
    term: DensityTerm,
    measured_sigs: frozenset[Signature],
) -> int:
    r"""Count selected quanta in the left monomial of a density term.

    Parameters
    ----------
    term:
        Density term whose left monomial is to be inspected.
    measured_sigs:
        Signatures of the selected measured modes.

    Returns
    -------
    int
        Number of selected creation operators in ``term.left``.

    Raises
    ------
    NumberMeasurementError
        If ``term.left`` contains annihilators.

    Notes
    -----
    The current structural number-measurement logic assumes creator-only
    left monomials.

    """
    if term.left.has_annihilators:
        raise NumberMeasurementError(
            "Projective number measurement currently requires "
            "creator-only left monomials."
        )
    return count_selected_creators_in_monomial(measured_sigs, term.left)


def count_selected_quanta_right(
    term: DensityTerm,
    measured_sigs: frozenset[Signature],
) -> int:
    r"""Count selected quanta in the right monomial of a density term.

    Parameters
    ----------
    term:
        Density term whose right monomial is to be inspected.
    measured_sigs:
        Signatures of the selected measured modes.

    Returns
    -------
    int
        Number of selected creation operators in ``term.right``.

    Raises
    ------
    NumberMeasurementError
        If ``term.right`` contains annihilators.

    Notes
    -----
    The current structural number-measurement logic assumes creator-only
    right monomials.

    """
    if term.right.has_annihilators:
        raise NumberMeasurementError(
            "Projective number measurement currently requires "
            "creator-only right monomials."
        )
    return count_selected_creators_in_monomial(measured_sigs, term.right)


def iter_number_support_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
) -> tuple[NumberOutcome, ...]:
    r"""Enumerate number outcomes supported by a density state.

    Parameters
    ----------
    state:
        Density-state wrapper whose number support should be resolved.
    target:
        Semantic measurement target specifying the measured subsystem.

    Returns
    -------
    tuple[NumberOutcome, ...]
        Sorted tuple of number outcomes supported by diagonal number
        sectors of the selected subsystem.

    Notes
    -----
    A count contributes to the support only when the selected number on
    the left and right side of a density term agrees.

    """
    measured_sigs = selected_mode_signatures(state, target)
    counts: set[int] = set()

    for term in state.rho.terms:
        n_left = count_selected_quanta_left(term, measured_sigs)
        n_right = count_selected_quanta_right(term, measured_sigs)
        if n_left == n_right:
            counts.add(n_left)

    return tuple(NumberOutcome(count=n) for n in sorted(counts))


def resolve_number_stats_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
    *,
    eps: float = 1e-12,
) -> NumberMeasurementStats:
    r"""Resolve projective number statistics for a density state.

    Parameters
    ----------
    state:
        Density-state wrapper whose number statistics should be resolved.
    target:
        Semantic measurement target specifying the measured subsystem.
    eps:
        Numerical tolerance used to discard negligible probabilities and
        to guard against small imaginary or negative numerical artifacts.

    Returns
    -------
    NumberMeasurementStats
        Resolved probability distribution over number outcomes.

    Raises
    ------
    NumberMeasurementError
        If a projected trace has a non-negligible imaginary part or a
        real part smaller than ``-eps``.

    Notes
    -----
    For each supported outcome :math:`n`, the probability is obtained as

    .. math::

        p(n) = \mathrm{Tr}(\Pi_n \rho \Pi_n).

    """
    probabilities: dict[NumberOutcome, float] = {}

    for outcome in iter_number_support_poly_density(state, target):
        projected = project_onto_number_poly_density(
            state,
            target,
            outcome,
            eps=eps,
        )
        value = projected.trace()

        if abs(value) <= eps:
            continue
        if abs(value.imag) > eps:
            raise NumberMeasurementError(
                "Encountered non-negligible imaginary contribution in "
                "number probabilities."
            )

        prob = float(value.real)
        if prob < -eps:
            raise NumberMeasurementError(
                f"Encountered negative probability for {outcome}: {prob}."
            )
        if prob < 0:
            prob = 0.0

        probabilities[outcome] = prob

    return NumberMeasurementStats(probabilities)


def project_onto_number_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
    outcome: NumberOutcome,
    *,
    eps: float = 1e-12,
) -> DensityPolyState:
    r"""Project a density state onto a selected number outcome.

    Parameters
    ----------
    state:
        Density-state wrapper to project.
    target:
        Semantic measurement target specifying the measured subsystem.
    outcome:
        Number outcome defining the desired number sector.
    eps:
        Numerical threshold forwarded to term combination.

    Returns
    -------
    DensityPolyState
        Projected density state containing only terms whose selected left
        and right number counts both match ``outcome.count``.

    Notes
    -----
    Projection is implemented structurally by keeping only terms that are
    diagonal in the selected number sector and match the requested count.

    """
    measured_sigs = selected_mode_signatures(state, target)
    n_target = outcome.count

    kept_terms = tuple(
        term
        for term in state.rho.terms
        if count_selected_quanta_left(term, measured_sigs) == n_target
        and count_selected_quanta_right(term, measured_sigs) == n_target
    )

    rho_proj = state.rho.__class__(kept_terms).combine_like_terms(eps=eps)
    return DensityPolyState.from_densitypoly(rho_proj)


def discard_measured_modes_number_density(
    state: DensityPolyState,
    target: MeasurementTarget,
) -> DensityPolyState:
    r"""Discard measured modes after number measurement on a density state.

    Parameters
    ----------
    state:
        Density-state wrapper from which measured modes should be removed.
    target:
        Semantic measurement target specifying the measured subsystem.

    Returns
    -------
    DensityPolyState
        Reduced density state obtained by tracing out the measured modes.

    Notes
    -----
    This models destructive removal of the measured subsystem at the
    density-state level.

    """
    measured_modes = state.resolve_target_modes(target)
    return state.trace_out_modes(measured_modes)


def joint_number_counts_left(
    term: DensityTerm,
    measured_sigs_by_port: Mapping[str, frozenset[Signature]],
) -> tuple[tuple[str, int], ...]:
    r"""Count selected quanta per port in the left monomial of a density term.

    Parameters
    ----------
    term:
        Density term whose left monomial is to be inspected.
    measured_sigs_by_port:
        Mapping from port name to selected mode signatures.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Tuple of ``(port_name, count)`` pairs for ``term.left``.

    Raises
    ------
    NumberMeasurementError
        If ``term.left`` contains annihilators.

    """
    if term.left.has_annihilators:
        raise NumberMeasurementError(
            "Projective joint number measurement currently requires "
            "creator-only left monomials."
        )
    return count_selected_creators_by_port(term.left, measured_sigs_by_port)


def joint_number_counts_right(
    term: DensityTerm,
    measured_sigs_by_port: Mapping[str, frozenset[Signature]],
) -> tuple[tuple[str, int], ...]:
    r"""Count selected quanta per port in the right monomial of a density term.

    Parameters
    ----------
    term:
        Density term whose right monomial is to be inspected.
    measured_sigs_by_port:
        Mapping from port name to selected mode signatures.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Tuple of ``(port_name, count)`` pairs for ``term.right``.

    Raises
    ------
    NumberMeasurementError
        If ``term.right`` contains annihilators.

    """
    if term.right.has_annihilators:
        raise NumberMeasurementError(
            "Projective joint number measurement currently requires "
            "creator-only right monomials."
        )
    return count_selected_creators_by_port(term.right, measured_sigs_by_port)


def iter_joint_number_support_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
) -> tuple[JointOutcome, ...]:
    r"""Enumerate joint number outcomes supported by a density state.

    Parameters
    ----------
    state:
        Density-state wrapper whose joint number support should be resolved.
    target:
        Semantic measurement target containing per-port selections.

    Returns
    -------
    tuple[JointOutcome, ...]
        Sorted tuple of supported joint number outcomes.

    Notes
    -----
    A joint count contributes to the support only when left and right
    per-port counts agree for the density term.

    """
    measured_sigs_by_port = selected_mode_signatures_by_port(state, target)
    counts_seen: set[tuple[tuple[str, int], ...]] = set()

    for term in state.rho.terms:
        left_counts = joint_number_counts_left(term, measured_sigs_by_port)
        right_counts = joint_number_counts_right(term, measured_sigs_by_port)
        if left_counts == right_counts:
            counts_seen.add(left_counts)

    return tuple(joint_outcome_from_counts(counts) for counts in sorted(counts_seen))


def project_onto_joint_number_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
    outcome: JointOutcome,
    *,
    eps: float = 1e-12,
) -> DensityPolyState:
    r"""Project a density state onto a selected joint number outcome.

    Parameters
    ----------
    state:
        Density-state wrapper to project.
    target:
        Semantic measurement target containing per-port selections.
    outcome:
        Joint number outcome defining the desired per-port number sector.
    eps:
        Numerical threshold forwarded to term combination.

    Returns
    -------
    DensityPolyState
        Projected density state containing only terms whose per-port left
        and right counts both match the requested joint outcome.

    """
    measured_sigs_by_port = selected_mode_signatures_by_port(state, target)
    expected_counts = counts_from_joint_outcome(outcome)

    kept_terms = tuple(
        term
        for term in state.rho.terms
        if joint_number_counts_left(term, measured_sigs_by_port) == expected_counts
        and joint_number_counts_right(term, measured_sigs_by_port) == expected_counts
    )

    rho_proj = state.rho.__class__(kept_terms).combine_like_terms(eps=eps)
    return DensityPolyState.from_densitypoly(rho_proj)


def resolve_joint_number_stats_poly_density(
    state: DensityPolyState,
    target: MeasurementTarget,
    *,
    eps: float = 1e-12,
) -> Mapping[JointOutcome, float]:
    r"""Resolve joint number statistics for a density state.

    Parameters
    ----------
    state:
        Density-state wrapper whose joint number statistics should be
        resolved.
    target:
        Semantic measurement target containing per-port selections.
    eps:
        Numerical tolerance used to discard negligible probabilities and
        to guard against small imaginary or negative numerical artifacts.

    Returns
    -------
    Mapping[JointOutcome, float]
        Mapping from joint number outcomes to probabilities.

    Raises
    ------
    NumberMeasurementError
        If a projected trace has a non-negligible imaginary part or a
        real part smaller than ``-eps``.

    Notes
    -----
    For each supported joint outcome, the probability is obtained as the
    trace of the corresponding projected density state.

    """
    probabilities: dict[JointOutcome, float] = {}

    for outcome in iter_joint_number_support_poly_density(state, target):
        projected = project_onto_joint_number_poly_density(
            state,
            target,
            outcome,
            eps=eps,
        )
        value = projected.trace()

        if abs(value) <= eps:
            continue
        if abs(value.imag) > eps:
            raise NumberMeasurementError(
                "Encountered non-negligible imaginary contribution in "
                "joint number probabilities."
            )

        prob = float(value.real)
        if prob < -eps:
            raise NumberMeasurementError(
                f"Encountered negative probability for {outcome}: {prob}."
            )
        if prob < 0:
            prob = 0.0

        probabilities[outcome] = prob

    return probabilities
