Packages
========

This document defines what belongs in each top-level package and serves as
the authoritative reference for package boundaries.


``core``
--------

Purpose
^^^^^^^

- Provide the foundational primitives and interfaces used throughout 
  the project.
- Serve as the root of the internal dependency graph.

Scope
^^^^^

- Mathematical primitives of the *CCR* operator framework.
- Structural identity and signature machinery.
- Typing protocols that define cross-package interfaces.
- Immutable, composable datastructures that higher layers build upon.

Primary Objects
^^^^^^^^^^^^^^^

- Mode and ladder operator primitives.
- Normally ordered monomials.
- Elementary term objects used in ket and density representations.
- Signature objects used for identity, ordering, and grouping

Protocols
^^^^^^^^^

- Signature-bearing intirfacse (exact and approximate signatures).
- Envelope-like interface required by mode operators.
- Label-like interfaces required for overlap structure and mode labeling.
- Operator interfaces (mode operators, ladder operators).
- Term interfaces (monomials and elementary ket/density terms).

Invariants and Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Core objects are immutable and hashable when appropriate.
- Monomials represent normally ordered products (creators then annihilators).
- The empty monomial represents the identity operator :math:`\mathbb{I}`.
- ``signature`` identifies exact structural identity.
- ``approx_signature`` provides a relaxed identifier for numerical merging and
  approximate equivalence (e.g. envelope tolerances).

Dependencies
^^^^^^^^^^^^
- The ``core`` package must not import any other internal package.
- External dependencies should remain minimal.


``modes``
---------

Purpose
^^^^^^^

- Provide the analytic and functional realization of abstract CCR modes.
- Implement continuous-variable envelope models and frequency-domain 
  transfer functions.
- Define explicit overlap structure for possibly non-orthogonal modes.
- Serve as the second foundational layer above ``core``.

Scope
^^^^^

- Time-domain envelope abstractions.
- Frequency-domain envelope abstractions consistent with the project
  Fourier convenvion.
- Closed-form analytic envelope models (e.g. Gaussian envelopes).
- Numeric overlap fallback machinery.
- Frequency-domain transfer functions (delay, filtering, dispersion).
- Mode labeling (path, polarization, comopiste labels).
- Plotting and inspection utilities for envelope objects.
- Shared type aliases and small validation helpers specific to model handling.

Primary Objects
^^^^^^^^^^^^^^^

- ``ModeLabel`` full mode label consisting of ``Envelope``, ``PathLabel`` and
  ``PolarizationLabel``.
- Concrete ``Envelope`` implementations (``GaussianEnvelope``, ``FilteredEnvelope``).
- Transfer function objects implementing multiplicative frequency action.

Protocols Implemented
^^^^^^^^^^^^^^^^^^^^^

- ``EnvelopeLike`` interfaces defined in ``core``.
- ``HasSignature`` interfaces defined in ``core``.
- Overlap capable interfaces for non-orghogonal mode handling.

Invariants and Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Envelopes represent complex-valued time-domain fields.
- Frequency-domain transfer functions act multiplicatively:

  .. math::

    \zeta_{\mathrm{out}}(\omega)
    =
    H(\omega)\,\zeta_{\mathrm{in}}(\omega).

- Overlap defines the Gram structure of non-orthogonal modes.
- Envelope and ModeLabel objects are immutable.
- ``signature`` and ``approx_signature`` follow the structural identity rules
  defined in ``core``.
- Closed-form expressions (when available) must remain consistent with the
  global Fourier convention of the project.
- Numeric quadrature may be used when analytic overlap expressions are not
  available.

Layering Constraints
^^^^^^^^^^^^^^^^^^^^

- ``modes`` depends on ``core``.
- ``core`` must not depend on ``modes``.
- Higher-level modeling layers may degend on both ``core`` and ``modes``.
- ``modes`` does not introduce dependencies that violate the acyclic
  dependency graph.

Dependencies
^^^^^^^^^^^^

- Internal deepndency: ``core``.
- External dependendencies: NumPy (required), Matplotlib (optional, for plotting).
- No higher-level modeling packages may be imported.
