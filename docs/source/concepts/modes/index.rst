Modes
=====

Overview
--------

The *modes* subsystem describes the **structure of optical modes** used
throughout the simulation framework.

A central design principle is the separation between:

- **mode structure**: how a field is distributed in time, frequency,
  polarization, and space
- **quantum state**: photon number, amplitudes, and entanglement

The modes package models only the first.

In the library, an optical mode is described by two complementary layers:

- :doc:`envelopes` — temporal and spectral structure
- **labels** — discrete identifiers such as
  :class:`~symop.modes.labels.path.Path` and
  :class:`~symop.modes.labels.polarization.Polarization`

Together, these define how modes overlap and interfere, while quantum
states operate on top of this structure.


Physical motivation
-------------------

A single photon in one optical channel can be described by

.. math::

    \mathcal{H}_1
    =
    \mathcal{H}_{\mathrm{env}}
    \otimes
    \mathcal{H}_{\mathrm{pol}},

where

- :math:`\mathcal{H}_{\mathrm{env}} = L^2(\mathbb{R}, \mathbb{C})`
  describes temporal or spectral profiles,
- :math:`\mathcal{H}_{\mathrm{pol}} = \mathbb{C}^2`
  describes polarization.

In this library:

- the *envelope subsystem* models
  :math:`\mathcal{H}_{\mathrm{env}}`,
- polarization and path are represented as **mode labels**,
- quantum states (Fock space, amplitudes, entanglement) are handled
  separately.

Multi-photon states are constructed from these single-photon modes,
but the modes package itself defines only the *basis structure*.


Structure of a mode
-------------------

An optical mode in the library is identified by a composite label

.. math::

    m = (p, \pi, \zeta),

where

- :math:`p` is the :class:`~symop.modes.labels.path.Path`
- :math:`\pi` is the :class:`~symop.modes.labels.polarization.Polarization`
- :math:`\zeta` is the :class:`~symop.modes.envelopes.base.BaseEnvelope`

The overlap between two modes factorizes as

.. math::

    \langle m_1, m_2 \rangle
    =
    \langle p_1, p_2 \rangle
    \langle \pi_1, \pi_2 \rangle
    \langle \zeta_1, \zeta_2 \rangle.

This factorization determines distinguishability and interference
behavior.

Minimal example
---------------

.. jupyter-execute::

    from symop.modes.envelopes import GaussianEnvelope
    from symop.modes.labels.mode import ModeLabel
    from symop.modes.labels.path import Path
    from symop.modes.labels.polarization import Polarization

    mode = ModeLabel(
        path=Path("A"),
        polarization=Polarization.H(),
        envelope=GaussianEnvelope(omega0=10.0, sigma=1.0, tau=0.0, phi0=0.0),
    )

    print(mode.signature)

Submodules
----------

.. toctree::
   :maxdepth: 1

   mode_label
   envelopes
   transfer
   polarization
   path

