r"""
Number State Source Example
===========================

This example demonstrates the use of the
:class:`~symop.devices.models.sources.number_state_source.NumberStateSource`
with the polynomial state.
"""

from __future__ import annotations
from symop.devices.models.sources import NumberStateSource
from symop.modes.labels import Path, Polarization
from symop.modes.envelopes import GaussianEnvelope
from symop.polynomial.state import KetPolyState

# Visualization Package
import symop.viz as VI


# %%
# Setup: set up a mode labels and then the single photon source
# -------------------------------------------------------------


src_env = GaussianEnvelope(omega0=2.0, sigma=1.0, tau=0.0)
src_pol = Polarization.H()

src_dev = NumberStateSource(envelope=src_env, polarization=src_pol, n=1)

# %%
# 1) Generate the state
# ---------------------
# - Generate the vacuum state for the source to populate
# - For populating the vacuum ``__call__()`` can be used or ``.apply()``

vac = KetPolyState.vacuum()
single_photon_state = src_dev(
    vac,
    ports={"out": Path("src_out")},
)

# %%
# 2) Inspect state
# ----------------
VI.plot(single_photon_state)
VI.display(single_photon_state)
