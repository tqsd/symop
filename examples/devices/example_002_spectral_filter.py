r"""
Spectral Filter Example
=======================

This example demonstrates how spectral filtering is
applied to a state using
:class:`~symop.devices.models.filters.spectral_filter.SpectralFilter`
model, we also make use of
:class:`~symop.devices.models.sources.number_state_source.NumberStateSource`
to generate the state.
"""

from __future__ import annotations
from symop.devices.models import SpectralFilter, NumberStateSource
from symop.modes.labels import Path, Polarization
from symop.modes.envelopes import GaussianEnvelope
from symop.modes.transfer import GaussianLowpass
from symop.polynomial.state import KetPolyState

# Visualization Package
import symop.viz as VI


# %%
# **Setup**
#
# set up a mode labels and then the single photon source
# together with filter defined by a transfer function.


src_env = GaussianEnvelope(omega0=100.0, sigma=50.0, tau=0.0)
src_pol = Polarization.H()

src_dev = NumberStateSource(envelope=src_env, polarization=src_pol, n=1)

tf = GaussianLowpass(w0=100.0, sigma_w=0.01)
filt_dev = SpectralFilter(transfer=tf)

# %%
# **1) Generate the state**
#
# - Generate the vacuum state for the source to populate
# - For populating the vacuum ``__call__()`` can be used or ``.apply()``

vac = KetPolyState.vacuum()
single_photon_state = src_dev(
    vac,
    ports={"out": Path("src_out")},
).with_label("in")


# %%
# **2) Use filtering device to filter**
filtered_state = filt_dev(
    single_photon_state,
    ports={
        "in": Path("src_out"),  # Path leading from the source to filter
        "out": Path("filt_out"),
    },
).with_label("filt")

# %%
# **3) Inspect the state after filtering**

VI.plot(single_photon_state)
VI.plot(filtered_state)

# %%
# **3a) Display the input state**
VI.display(single_photon_state)

# %%
# **3b) Display the filtered state**
VI.display(filtered_state)
