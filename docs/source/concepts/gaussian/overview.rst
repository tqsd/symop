Gaussian Logic
==============

Gaussian logic in ``symop_proto`` refers to the representation and
manipulation of quantum optical states that are completely determined
by their first and second moments.

A state is Gaussian if all higher-order moments can be expressed
through the first and second moments via Wick's theorem.
In this framework, we work directly in the ladder-operator basis.

Mathematical Representation
---------------------------

Let :math:`a_i` denote annihilation operators. A Gaussian state is
fully characterized by:

First moments (displacement vector):

.. math::

   \alpha_i = \langle a_i \rangle

Second moments:

.. math::

   N_{ij} = \langle a_i^\dagger a_j \rangle

and, when needed,

.. math::

   M_{ij} = \langle a_i a_j \rangle.

No higher-order correlations are stored explicitly.

Gram Matrix and Commutation Structure
--------------------------------------

Unlike textbook presentations that assume orthonormal modes,
``symop_proto`` allows non-orthogonal mode bases.

The commutation relations are encoded via a Gram matrix:

.. math::

   [a_i, a_j^\dagger] = G_{ij}.

This allows:

- Non-orthogonal modes
- Path-based mode labeling
- Device-aware basis construction

The Gram matrix is part of the state definition and propagates
through transformations.

Internal State Representation
-----------------------------

The core Gaussian state container is:

:class:`symop_proto.gaussian.core.GaussianCore`

It stores:

- The basis (mode ordering + Gram matrix)
- The first moment vector
- The second moment structure

All Gaussian operations are expressed as transformations
on these quantities.

Gaussian Transformations
------------------------

Gaussian transformations fall into two categories:

Passive (unitary mode mixing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear transformations of the ladder operators:

.. math::

   a_i \mapsto \sum_j U_{ij} a_j

with :math:`U` unitary.

These are implemented via passive maps and preserve photon number.

Affine Gaussian Channels
~~~~~~~~~~~~~~~~~~~~~~~~

More general Gaussian processes can be written in affine form:

.. math::

   x \mapsto X x + d

with added noise contribution:

.. math::

   \Gamma \mapsto X \Gamma X^T + Y.

This structure underlies:

- Loss channels
- Thermal noise
- Linear Gaussian devices

In ``symop_proto``, such transformations are embedded into
the full mode space using explicit mode-index mappings.

Devices vs Maps
---------------

A key architectural principle is the separation between:

- Mathematical kernels (maps, affine transforms, Bogoliubov maps)
- Device logic (mode selection, routing, environment handling)

Gaussian devices operate on :class:`GaussianCore` but do not
hard-code physical assumptions into the core state.

This allows:

- Clean composition of devices
- Environment tracing
- Basis-aware transformations
- Future hybrid extensions

Limitations
-----------

The Gaussian layer only captures states that remain Gaussian
under evolution.

Non-Gaussian effects (e.g. photon subtraction, Kerr nonlinearities)
require extensions beyond the second-moment formalism.

These are handled in the hybrid layer.

Summary
-------

Gaussian logic provides:

- A compact representation of optical states
- Basis-aware commutation handling
- Structured affine channel support
- A clean separation between physical devices and mathematical maps

It forms the foundation for efficient simulation of linear
and noisy optical systems.
