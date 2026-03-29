from __future__ import annotations

from typing import Any

import numpy as np

from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.measurement.result import ObservationResult
from symop.viz._dispatch import latex as latex_fn
from symop.viz._dispatch import plot
from symop.viz._dispatch import text as text_fn
from symop.viz._optional import require_matplotlib_pyplot


def _outcome_to_int(outcome: object) -> int:
    """Convert a measurement outcome to an integer x-coordinate."""
    if isinstance(outcome, int):
        return outcome

    if isinstance(outcome, NumberOutcome):
        return int(outcome.count)

    count = getattr(outcome, "count", None)
    if count is not None:
        return int(count)

    key = getattr(outcome, "key", None)
    if isinstance(key, int):
        return key

    raise TypeError(f"Cannot convert outcome {outcome!r} to integer.")


def _extract_outcomes_and_probs(
    obs: ObservationResult,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ps = []

    for outcome, prob in obs.probabilities.items():
        xs.append(_outcome_to_int(outcome))
        ps.append(float(prob))

    xs_arr = np.asarray(xs, dtype=int)
    ps_arr = np.asarray(ps, dtype=float)

    order = np.argsort(xs_arr)
    return xs_arr[order], ps_arr[order]


@plot.register(ObservationResult)
def _plot_observation_result(obs: ObservationResult, /, **kwargs: Any) -> Any:
    ...
    """Plot histogram of an ObservationResult.

    Displays the probability distribution over measurement outcomes.
    """
    plt = require_matplotlib_pyplot()

    xs, ps = _extract_outcomes_and_probs(obs)

    title = kwargs.pop("title", None)

    # Build header (same style as envelope plots)
    s = latex_fn(obs)
    if s:
        header = (title + "\n" if title else "") + "$" + s + "$"
    else:
        header = (title + "\n" if title else "") + text_fn(obs)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(xs, ps)

    ax.set_xlabel("Outcome (photon number)")
    ax.set_ylabel("Probability")
    ax.set_xticks(xs)

    if header:
        ax.set_title(header)

    ax.set_ylim(0, max(ps) * 1.2 if len(ps) > 0 else 1)

    if kwargs.pop("show", True):
        plt.show()

    return fig, ax
