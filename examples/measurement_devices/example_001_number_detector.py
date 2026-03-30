r"""
Number Detector Example
=======================

This example demonstrates the use of the
:class:`~symop.devices.models.detectors.number_detector.NumberDetector`.

We show three related measurement concepts:

1. **Observation**
   returns the full outcome distribution for a measurement without
   selecting a particular branch.

2. **Detection**
   samples one concrete measurement outcome and returns the corresponding
   post-measurement state.

3. **Postselection**
   conditions the state on a chosen measurement outcome.

The example first measures a simple product input state and then repeats
the observation/detection process after interference on a 50/50 beam
splitter.

For a number detector, the outcomes correspond to the possible photon
numbers detected in the measured path.
"""

from __future__ import annotations

import numpy as np

from symop.devices.models.beamsplitters.beamsplitter import BeamSplitter
from symop.devices.models.detectors.number_detector import NumberDetector
from symop.devices.models.sources.number_state_source import NumberStateSource
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.path import Path
from symop.modes.labels.polarization import Polarization
from symop.polynomial.state.ket import KetPolyState
import symop.viz as VI

# %%
# Setup
# =====
# Create a two-photon source and a number detector.
#
# The source emits photons into a path with a Gaussian spectral envelope
# and horizontal polarization. Since ``n=2``, each source prepares a
# two-photon excitation in its output mode.

src = NumberStateSource(
    envelope=GaussianEnvelope(omega0=100.0, sigma=50.0, tau=0.0),
    polarization=Polarization.H(),
    n=2,
)

det = NumberDetector()


# %%
# Generate the input state
# ========================
# Start from vacuum and populate two distinct paths with identical
# two-photon states. The joint state therefore contains a total of
# four photons, distributed across two input paths.

vac = KetPolyState.vacuum()

state_a = src(vac, ports={"out": Path("src_out")}).with_label("in")
state_b = src(vac, ports={"out": Path("src_out_aux")}).with_label("in_aux")

state = state_a.join(state_b).with_label("joint")
VI.display(state)

# %%
# Observe the measurement
# =======================
# Observation is a non-destructive query. It does not collapse the state.
# Instead, it returns the full probability distribution over all possible
# detector outcomes for the selected path.
#
# Here the detector measures the path ``src_out``. Since the total state
# contains four photons, the possible number outcomes are determined by
# how many photons can be found in that path.

observation = det.observe(state=state, ports={"in": Path("src_out")})
VI.display(observation)

# %%
# Detect
# ======
# Detection represents one concrete realization of the measurement.
# It returns a sampled outcome together with the corresponding
# post-measurement state.

detection = det.detect(state=state, ports={"in": Path("src_out")})
VI.display(detection)

# %%
# Post-measurement state after detection
# ======================================
# This is the collapsed state associated with the sampled detection event.

VI.display(detection.state)

# %%
# Postselection
# =============
# Postselection conditions the original state on a chosen outcome.
# In this case we reuse the outcome sampled above and explicitly build
# the corresponding conditional branch.

postselection = det.postselect(
    state=state,
    ports={"in": Path("src_out")},
    outcome=detection.outcome,
)
VI.display(postselection)

# %%
# Postselected state
# ==================
# This is the state conditioned on the selected outcome.

VI.display(postselection.state)

# %%
# Interfere the input state on a beam splitter
# ============================================
# Next, we interfere the two input paths on a 50/50 beam splitter.
# Interference changes the amplitudes and therefore changes the
# measurement probabilities at the output ports.

bs = BeamSplitter(theta=np.pi / 4)

state_interfered = bs(
    state,
    ports={
        "in0": Path("src_out"),
        "in1": Path("src_out_aux"),
        "out0": Path("bs_out0"),
        "out1": Path("bs_out1"),
    },
).with_label("interfered")

VI.display(state_interfered)

# %%
# Observe the output distribution after interference
# ==================================================
# We now observe the photon-number distribution in one beam-splitter
# output path. The set of possible outcomes is still determined by the
# allowed photon numbers in that path, but the probabilities are changed
# by interference.

interfered_observation = det.observe(
    state=state_interfered,
    ports={"in": Path("bs_out0")},
)
VI.display(interfered_observation)

# %%
# Quick Plot
VI.plot(interfered_observation)

# %%
# Detect one output event after interference
# ==========================================
# As before, detection returns one concrete sampled outcome and the
# corresponding collapsed branch.

interfered_detection = det.detect(
    state=state_interfered,
    ports={"in": Path("bs_out0")},
)
VI.display(interfered_detection)

# %%
# Post-measurement state after interfered detection
# =================================================
VI.display(interfered_detection.state)
