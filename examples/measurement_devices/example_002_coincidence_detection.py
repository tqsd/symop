r"""
HOM Coincidence Observation Example
===================================

This example prepares two identical single-photon inputs, interferes them
on a 50/50 beam splitter, and observes the joint output photon-number
distribution across both output ports.

For indistinguishable photons, the Hong-Ou-Mandel effect suppresses the
(1, 1) coincidence event.
"""

from __future__ import annotations

import numpy as np

from symop.core.protocols.states.base import State
from symop.devices.models.beamsplitters.beamsplitter import BeamSplitter
from symop.devices.models.detectors.coincidence_detector import (
    CoincidenceDetector,
)
from symop.devices.models.paths.delay import Delay
from symop.devices.models.sources.number_state_source import NumberStateSource
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.path import Path
from symop.modes.labels.polarization import Polarization
from symop.polynomial.state.ket import KetPolyState
import symop.viz as VI

# %%
# Setup
# =====
# Two identical single-photon sources and a coincidence detector.

src = NumberStateSource(
    envelope=GaussianEnvelope(omega0=100.0, sigma=50.0, tau=0.0),
    polarization=Polarization.H(),
    n=1,
)

coinc = CoincidenceDetector(input_ports=("out0", "out1"))

bs = BeamSplitter(theta=np.pi / 4)


# %%
# Prepare the input state
# =======================
# One photon in each input path.

vac = KetPolyState.vacuum()

state_a = src(vac, ports={"out": Path("src_a")}).with_label("a")
state_b = src(vac, ports={"out": Path("src_b")}).with_label("b")

state_in = state_a.join(state_b).with_label("input")
VI.display(state_in)


# %%
# Interfere on a 50/50 beam splitter
# ==================================

state_out = bs(
    state_in,
    ports={
        "in0": Path("src_a"),
        "in1": Path("src_b"),
        "out0": Path("bs_out0"),
        "out1": Path("bs_out1"),
    },
).with_label("after_bs")

VI.display(state_out)


# %%
# Observe the joint output distribution
# =====================================
# This is the HOM signature:
# the joint (1, 1) event should be suppressed for perfectly
# indistinguishable photons.

observation = coinc.observe(
    state=state_out,
    ports={
        "out0": Path("bs_out0"),
        "out1": Path("bs_out1"),
    },
)

VI.display(observation)
print(observation)

# %%
# Detect one joint output event
# =============================
# Detection samples one concrete joint outcome from the HOM output
# distribution and returns the corresponding post-measurement state.

detection = coinc.detect(
    state=state_out,
    ports={
        "out0": Path("bs_out0"),
        "out1": Path("bs_out1"),
    },
)

VI.display(detection)
print(detection)

# %%
# Post-measurement state after joint detection
# ============================================

VI.display(detection.state)
print(detection.state)

# %%
# Hong-Ou-Mandel coincidence scan
# ===============================
# We delay one input arm, interfere the two photons on the 50/50 beam
# splitter, and record the coincidence probability P(1,1) at the outputs.

import matplotlib.pyplot as plt

from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome


def modify_delay_on_path(state: State, path: Path, delay: float) -> State:
    delay_dev = Delay(dt=delay)
    return delay_dev(state, ports={"in": path, "out": path})


def joint_outcome_probability(
    observation,
    *,
    out0_count: int,
    out1_count: int,
) -> float:
    """Extract P(out0=out0_count, out1=out1_count) from a joint observation."""
    for outcome, prob in observation.probabilities.items():
        if not isinstance(outcome, JointOutcome):
            continue

        counts = {
            port_name: port_outcome.count
            for port_name, port_outcome in outcome.outcomes_by_port
            if isinstance(port_outcome, NumberOutcome)
        }

        if (
            counts.get("out0") == out0_count
            and counts.get("out1") == out1_count
        ):
            return prob

    return 0.0


def hom_coincidence_probability(delay: float) -> float:
    """Return the HOM coincidence probability P(1,1) for a given delay."""
    vac = KetPolyState.vacuum()

    state_a = src(vac, ports={"out": Path("src_a")}).with_label("a")
    state_b = src(vac, ports={"out": Path("src_b")}).with_label("b")

    # Delay one arm before the beamsplitter.
    state_b_delayed = modify_delay_on_path(
        state_b, Path("src_b"), delay
    ).with_label("b_delayed")

    state_in = state_a.join(state_b_delayed).with_label("input")

    state_out = bs(
        state_in,
        ports={
            "in0": Path("src_a"),
            "in1": Path("src_b"),
            "out0": Path("bs_out0"),
            "out1": Path("bs_out1"),
        },
    ).with_label("after_bs")
    VI.display(state_out)

    observation = coinc.observe(
        state=state_out,
        ports={
            "out0": Path("bs_out0"),
            "out1": Path("bs_out1"),
        },
    )

    return joint_outcome_probability(
        observation,
        out0_count=1,
        out1_count=1,
    )


# %%
# Choose a delay window around the coherence scale.
# For Gaussian spectral width sigma, a good first guess is +/- 6 / sigma.

delay_max = 6.0 * src.envelope.sigma
delays = np.linspace(-delay_max, delay_max, 121)
coincidences = np.array([hom_coincidence_probability(dt) for dt in delays])

plt.figure(figsize=(7, 4))
plt.plot(delays, coincidences)
plt.xlabel("Delay")
plt.ylabel("Coincidence probability P(1,1)")
plt.title("Hong-Ou-Mandel coincidence curve")
plt.grid(True)
plt.show()
