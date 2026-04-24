Device Execution and Kernel Dispatch
====================================

This document defines the architectural rules governing device
behavior, kernel dispatch, and state transformation across multiple
representations.

It is normative. If implemented behavior diverges from this
document, the implementation must be revised or this document
must be explicitly updated.

Purpose
-------

The project supports multiple state representations:

- Polynomial (ket and density)
- Gaussian

Devices must operate uniformly across these representations
without embedding representation-specific logic into device
model classes.

This document defines:

- The separation between graph planning and state transformation.
- The dispatch mechanism used to select representation-specific kernels.
- The allowed fallback and conversion behavior.
- Layering and dependency constraints for device-related modules.

Architectural Principles
------------------------
The device system adheres to the following principles:

1. Device models are representation-agnostic.
2. Graph/path effects are separated from state transformations.
3. State transformations are implemented as pure kernels.
4. Dispatch is centralized and deterministic.
5. Adding a new representation must not require modification of
   existing device models.

Separation of Planning and Execution
------------------------------------

Device application consists of two distinct phases.

Planning Phase
^^^^^^^^^^^^^^

Planning concerns the experiment graph and path allocation.

- Ports are bound to concrete paths.
- Missing output paths may be allocated.
- A :class:`DevicePlan` describes the structural graph effect.

Planning is independent of state representation.

Planning responsibilities:

- Binds ports.
- Declare required input paths.
- Declare newly allocated paths.
- Provide optional metadata for layout of diagnostics.

Planning must not inspect or transform state objects.

Execution Phase
^^^^^^^^^^^^^^^

Execution transforms the state using a representation-specific kernel.

A kernel:

- Receives ``(device, state, context)``
- Returns a new state.
- Does not mutate the experiment graph.
- Does not allocate paths.
- Is deterministic and side-effect free with respect to graph structure.

Kernels may allocate internal temporary objects but must not perform graph
rewiring.

Mode label assignment and transformations
-----------------------------------------

A physical optical mode in this project is described by a
:class:`~symop.modes.protocols.labels.ModeLabelProto` which
contains:

- a path label,
- a polarization label,
- an envelope.

The path component identifies the wire/route in the experiment
graph. The envelope and polarization components define the 
continuous-variable realization and overlap structure on that wire.

The experiment maintains a *mode label assignment* per path:

- For each :class:`~symop.core.protocols.labels.PathProto` used in
  the graph, there exists a current
  :class:`~symop.modes.protocols.labels.ModeLabelProto` whose
  ``label.path`` mathches that path.

This assignment is part of the experiment model (graph medatada) and
is representation-agnostic.

Device may request modifications to the mode label assignment as part of
planning.Examples include spectral filtering (envelope changes), dispersion
(envelope changes), and polarization optics (polarization changes).

Rules:

- Mode label transforms are declared in the device plan.
- The executor applies these transforms to the experiment's
  mode label assignment before kernel dispatch.
- Kernels must not mutate the experiment's mode label assignment.
- If a device induces non-unitary physical effects (loss, projection,
  noise), these effects must be implemented in kernels as
  representation specific quantum channels.

Stable Dispatch Key
-------------------

Kernel selection is determined by two stable identifiers:

- The device kind(``device.kind``)
- The state representation key.

Each state provides:

- A representation tag (``state.rep_tag``).
- A state kind (``state.state_kind``).

The dispatch key is constructed as:

.. code-block:: text

   "{representation_family}:{state_kind}"

Examples:

- ``poly:ket``
- ``poly:density``
- ``gaussian:gaussian``

The string values used in dispatch keys are part of the public internal 
contract and must remain stable.

Kernel Registry
---------------

The kernel registry maps:

.. code-block:: text

   (device_kind, dispatch_key) -> kernel_function

The registry must support:

- Registration of kernels.
- Resolution of kernels.
- Clear error reporting when resolution fails.

Duplicate registration for the same key is not permitted.


Fallback Policy
---------------

Dispatch may apply explicit fallback behavior when no kernel matches
the initial dispatch key.

Allowed fallback:

- If the state supports conversion to density, dispatch may convert
  ``poly:ket`` to ``poly:density`` and retry resolution.

Fallback logic must:

- Be centralized in the dispatcher.
- Not be implemented inside device models.
- Not be implemented inside kernels.

Disallowed behavior:

- Silent switching between representation families.
- Implicit conversion inside kernels.
- Device-specific ad hoc dispatch logic.


Output Semantics
----------------

Kernels may return a state whose kind differs from the input kind.

This is expected for:

- Measurement devices.
- Channel-like transformations.

However, output type expectations must be enforced by the calling API,
not by implicit assumptions inside dispatch.


Layering Constraints
--------------------

To preserve the acyclic dependency structure:

- ``symop.devices`` may depend only on foundational protocols
  (``symop.core.protocols`` and related typing abstractions).
- Device model classes must not depend on representation implementations.
- Representation-specific kernel modules may depend on:

  - ``symop.devices`` (for registry types),
  - Representation packages (polynomial, gaussian, hybrid),
  - ``core`` and ``modes`` as required.

- Foundational packages must not depend on ``devices``.


Device execution pipeline
-------------------------

Applying a device consists of three ordered steps:

1. Bind and plan
^^^^^^^^^^^^^^^^

The device is bound to concrete paths (allocating outputs if needed)
and produces a :class:`symop.core.protocols.device_plan.DevicePlan`.

2. Apply mode label transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the plan includes label transforms, the executor updates the
experiment's mode label assignment for affected paths. This updates
the envelope/polarization structure seen by subsequent devices.

3. Dispatch and execute a kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The executor constructs the dispatch key from the input state and
resolves a kernel from the registry. The kernel transform the state
but must not mutate the graph or mode label assignment.

Example: spectral filter on a ket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a filter device with one input port and one output port. It is
parameterized by a transfer function :math:`H(\omega)`.

Mode labels
~~~~~~~~~~~

Let the input path be :math:`p_\mathrm{in}` with current mode label

.. math::

   m_\mathrm{in} = (p_\mathrm{in}, \pi, \zeta_\mathrm{in}),

where :math:`\pi` is a polarization label and :math:`\zeta_\mathrm{in}` is an
envelope.

Planning returns a plan that allocates an output path :math:`p_\mathrm{out}`
and requests an envelope transform:

- allocate :math:`p_\mathrm{out}`
- set

  .. math::

     m_\mathrm{out} = (p_\mathrm{out}, \pi, \zeta_\mathrm{out}), \qquad
     \zeta_\mathrm{out} = \mathrm{FilteredEnvelope}(\zeta_\mathrm{in}, H).

After step (2), the experiment mode label assignment contains :math:`m_\mathrm{out}`.

Quantum map
~~~~~~~~~~~

A general amplitude transfer :math:`H(\omega)` is not unitary unless
:math:`|H(\omega)| = 1`. Therefore, a physically faithful filter is modeled
as a lossy channel implemented via a unitary dilation with an environment mode:

.. math::

   \hat a_\mathrm{out}(\omega)
   =
   H(\omega)\,\hat a_\mathrm{in}(\omega)
   +
   \sqrt{1-|H(\omega)|^2}\,\hat e(\omega),

where :math:`\hat e(\omega)` is an environment vacuum field.

If the input is a ket state, executing this channel requires either:

- returning a density state on the system (trace out the environment), or
- enlarging the state space by keeping the environment mode.

In the default pipeline, the dispatcher may convert a ket input to a density
and then apply the density-kernel for the filter. The resulting state is a
density state on the filtered output mode.

Implementation consequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The envelope transform (construction of :math:`\zeta_\mathrm{out}`) is a
  planning-time mode label update.
- The non-unitary effect (loss/noise) is implemented by the kernel as a
  representation-specific quantum channel.
- If a filter is configured as an ideal, lossless phase-only transfer
  (i.e. :math:`|H(\omega)|=1`), the kernel may be unitary on the system and
  the state may remain a ket.
