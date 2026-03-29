from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from symop.devices.measurement.result import (
    DetectionResult,
    MeasurementResult,
    ObservationResult,
    PostselectionResult,
)
from symop.viz._dispatch import text

K = TypeVar("K")
V = TypeVar("V")


def _format_mapping(mapping: Mapping[K, V]) -> str:
    if not mapping:
        return "{}"

    lines: list[str] = []
    for key, value in mapping.items():
        lines.append(f"  {key!r}: {value!r}")
    return "{\n" + ",\n".join(lines) + "\n}"


@text.register(MeasurementResult)
def _text_measurement_result(obj: MeasurementResult, /, **kwargs: Any) -> str:
    action_name = type(obj.action).__name__
    return f"{type(obj).__name__}(action={action_name}, metadata={dict(obj.metadata)!r})"


@text.register(ObservationResult)
def _text_observation_result(obj: ObservationResult, /, **kwargs: Any) -> str:
    action_name = type(obj.action).__name__
    parts = [
        "ObservationResult(",
        f"  action={action_name},",
        f"  expectation={obj.expectation!r},",
        "  probabilities=" + _format_mapping(obj.probabilities) + ",",
        f"  metadata={dict(obj.metadata)!r},",
        ")",
    ]
    return "\n".join(parts)


@text.register(DetectionResult)
def _text_detection_result(obj: DetectionResult, /, **kwargs: Any) -> str:
    action_name = type(obj.action).__name__
    state_summary = "<none>" if obj.state is None else repr(obj.state)
    parts = [
        "DetectionResult(",
        f"  action={action_name},",
        f"  record={obj.record!r},",
        f"  outcome={obj.outcome!r},",
        f"  probability={obj.probability!r},",
        f"  state={state_summary},",
        f"  metadata={dict(obj.metadata)!r},",
        ")",
    ]
    return "\n".join(parts)


@text.register(PostselectionResult)
def _text_postselection_result(
    obj: PostselectionResult, /, **kwargs: Any
) -> str:
    action_name = type(obj.action).__name__
    state_summary = "<none>" if obj.state is None else repr(obj.state)
    parts = [
        "PostselectionResult(",
        f"  action={action_name},",
        f"  outcome={obj.outcome!r},",
        f"  probability={obj.probability!r},",
        f"  state={state_summary},",
        f"  metadata={dict(obj.metadata)!r},",
        ")",
    ]
    return "\n".join(parts)
