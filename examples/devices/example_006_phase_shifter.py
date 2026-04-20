r"""
Phase Shifter and Mach-Zehnder Interferometer Example
=====================================================

This example first demonstrates the use of the
:class:`~symop.devices.models.phase_shifters.phase_shifter.PhaseShifter`
model on a single path.

It then shows how the same phase shifter becomes physically relevant
inside a simple Mach-Zehnder interferometer (MZI) constructed from two
beam splitters and one phase shifter.

We also make use of
:class:`~symop.devices.models.sources.number_state_source.NumberStateSource`
to generate the input state.
"""

from __future__ import annotations

import numpy as np

from symop.devices.models import BeamSplitter, NumberStateSource, PhaseShifter
from symop.modes.envelopes import GaussianEnvelope
from symop.modes.labels import Path, Polarization
from symop.polynomial.state.ket import KetPolyState

# Visualization Package
import symop.viz as VI

# %%
# **Setup**
#
# Create one single-photon source, one phase shifter, and two 50/50 beam
# splitters for the Mach-Zehnder interferometer.

src = NumberStateSource(
    envelope=GaussianEnvelope(omega0=100.0, sigma=50.0, tau=0.0),
    polarization=Polarization.H(),
    n=1,
)

ps = PhaseShifter(phi=np.pi / 2)

bs1 = BeamSplitter(theta=np.pi / 4, phi_r=0.0)
bs2 = BeamSplitter(theta=np.pi / 4, phi_r=0.0)

# %%
# **Generate the input state**
#
# Start from vacuum and populate one path with a single photon.

vac = KetPolyState.vacuum()

state_in = src(vac, ports={"out": Path("src_out")}).with_label("input")

# %%
# **2) Standalone phase shifter**
#
# Apply the phase shifter directly to the occupied path.
#
# For an isolated single path, this changes only the phase of the state.
# By itself, this does not change number statistics, but the phase becomes
# important once the state interferes with another path.

state_shifted = ps(
    state_in,
    ports={"path": Path("src_out")},
).with_label("phase_shifted")

VI.display_many(state_in, state_shifted)

# %%
# **Why does the standalone output look almost the same?**
#
# The phase shifter applies the unitary
#
#     U(phi) = exp(i phi n)
#
# to the selected mode. At the operator level, this induces
#
#     a^\dagger -> exp(i phi) a^\dagger
#
# so the excitation on that path acquires a phase factor.
#
# For a single isolated path, this does not change the direct photon-number
# probabilities. The effect becomes observable when the mode later interferes
# with another path.

# %%
# **3) First beam splitter: create the two MZI arms**
#
# Send the single input path through a 50/50 beam splitter.
# The unused second input is treated as vacuum.

state_after_bs1 = bs1(
    state_in,
    ports={
        "in0": Path("src_out"),
        "in1": Path("unused_in"),
        "out0": Path("arm0"),
        "out1": Path("arm1"),
    },
).with_label("after_bs1")

VI.display_many(state_in, state_after_bs1)

# %%
# **4) Apply the phase shifter to one arm**
#
# This creates a relative phase between the two interferometer arms.

state_after_ps = ps(
    state_after_bs1,
    ports={"path": Path("arm0")},
).with_label("after_phase_shifter")

VI.display_many(state_after_bs1, state_after_ps)

# %%
# **5) Second beam splitter: recombine the two arms**
#
# The relative phase now affects the interference at the output.

state_out = bs2(
    state_after_ps,
    ports={
        "in0": Path("arm0"),
        "in1": Path("arm1"),
        "out0": Path("out0"),
        "out1": Path("out1"),
    },
).with_label("mzi_output")

VI.display_many(state_after_ps, state_out)

# %%
# **Why does the phase matter in the MZI?**
#
# After the first beam splitter, the photon amplitude is distributed over
# two paths. The phase shifter changes the relative phase of one arm.
# When both arms are recombined at the second beam splitter, this relative
# phase changes the interference pattern and therefore redistributes the
# output amplitudes between the two output ports.
#
# In this sense, the standalone phase shifter prepares a phase that becomes
# physically visible through interference.

# %%
# **6) Density-state version**
#
# The same sequence can also be applied to density states.

state_in_dense = state_in.to_density().with_label("input_density")

state_after_bs1_dense = bs1(
    state_in_dense,
    ports={
        "in0": Path("src_out"),
        "in1": Path("unused_in"),
        "out0": Path("arm0"),
        "out1": Path("arm1"),
    },
).with_label("after_bs1_density")

state_after_ps_dense = ps(
    state_after_bs1_dense,
    ports={"path": Path("arm0")},
).with_label("after_phase_shifter_density")

state_out_dense = bs2(
    state_after_ps_dense,
    ports={
        "in0": Path("arm0"),
        "in1": Path("arm1"),
        "out0": Path("out0"),
        "out1": Path("out1"),
    },
).with_label("mzi_output_density")

VI.display_many(
    state_in_dense,
    state_after_bs1_dense,
    state_after_ps_dense,
    state_out_dense,
)
