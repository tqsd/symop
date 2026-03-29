from __future__ import annotations

from typing import Any

from symop.devices.measurement.result import (
    DetectionResult,
    MeasurementResult,
    ObservationResult,
    PostselectionResult,
)
from symop.viz._dispatch import latex


def _latex_escape_text(s: object) -> str:
    text = str(s)
    text = text.replace("\\", r"\\")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    return text


@latex.register(MeasurementResult)
def _latex_measurement_result(obj: MeasurementResult, /, **kwargs: Any) -> str:
    kind = type(obj).__name__
    action_name = type(obj.action).__name__
    return (
        r"\mathrm{"
        + _latex_escape_text(kind)
        + r"}"
        + r"\left("
        + r"\mathrm{action}="
        + r"\mathrm{"
        + _latex_escape_text(action_name)
        + r"}"
        + r"\right)"
    )


@latex.register(ObservationResult)
def _latex_observation_result(obj: ObservationResult, /, **kwargs: Any) -> str:
    n = len(obj.probabilities)
    expectation = "None" if obj.expectation is None else str(obj.expectation)
    return (
        r"\mathrm{Observation}"
        + r"\left("
        + r"\mathrm{outcomes}="
        + str(n)
        + r",\ "
        + r"\mathbb{E}="
        + _latex_escape_text(expectation)
        + r"\right)"
    )


@latex.register(DetectionResult)
def _latex_detection_result(obj: DetectionResult, /, **kwargs: Any) -> str:
    parts: list[str] = [r"\mathrm{Detection}\left("]
    parts.append(r"\mathrm{outcome}=" + _latex_escape_text(obj.outcome))
    if obj.probability is not None:
        parts.append(r",\ p=" + _latex_escape_text(obj.probability))
    parts.append(r"\right)")

    if obj.state is not None:
        state_latex = latex(obj.state, **kwargs)
        if state_latex:
            parts.append(r"\quad ")
            parts.append(state_latex)

    return "".join(parts)


@latex.register(PostselectionResult)
def _latex_postselection_result(
    obj: PostselectionResult, /, **kwargs: Any
) -> str:
    parts: list[str] = [r"\mathrm{Postselection}\left("]
    parts.append(r"\mathrm{outcome}=" + _latex_escape_text(obj.outcome))
    if obj.probability is not None:
        parts.append(r",\ p=" + _latex_escape_text(obj.probability))
    parts.append(r"\right)")

    if obj.state is not None:
        state_latex = latex(obj.state, **kwargs)
        if state_latex:
            parts.append(r"\quad ")
            parts.append(state_latex)

    return "".join(parts)
