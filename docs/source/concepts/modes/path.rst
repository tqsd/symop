Path
====

Overview
--------
A :class:`~symop.modes.labels.path.Path` label identifies the
**spatial or logical channel** associated with an optical mode.

Examples of paths include different waveguides, fibers, interferometer
arms, or abstract routing labels used in a simulation.

In the library, path labels are discrete and orthonormal.


Physical meaning
----------------

For two path labels :math:`p_1` and :math:`p_2`, the overlap is

.. math::

    \langle p_1, p_2 \rangle
    =
    \begin{cases}
    1, & p_1 = p_2, \\
    0, & p_1 \neq p_2.
    \end{cases}

This means that two modes on different paths are treated as fully
distinguishable at the path level.


Examples
--------

Constructing path labels
~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from symop.modes.labels.path import Path

    a = Path("A")
    b = Path("B")

    print(a)
    print(b)
    print(a.signature)
    print(b.signature)


Path overlaps
~~~~~~~~~~~~~

.. jupyter-execute:: 

    from symop.modes.labels.path import Path

    a1 = Path("A")
    a2 = Path("A")
    b = Path("B")

    print("⟨A|A⟩ =", a1.overlap(a2))
    print("⟨A|B⟩ =", a1.overlap(b))

Expected behavior:

- identical names give overlap ``1``
- different names give overlap ``0``


Role in composite mode labels
-----------------------------

Path labels form one component of a composite
:class:`~symop.modes.labels.mode.ModeLabel`.

.. math::

    m = (p, \pi, \zeta),

together with :class:`~symop.modes.labels.polarization.Polarization`
:math:`\pi` and an :doc:`envelope <envelopes>`
(typically a :class:`~symop.modes.envelopes.base.BaseEnvelope`)
:math:`\zeta`.

The total mode overlap factorizes as

.. math::

    \langle m_1, m_2 \rangle
    =
    \langle p_1, p_2 \rangle
    \langle \pi_1, \pi_2 \rangle
    \langle \zeta_1, \zeta_2 \rangle.

This allows routing changes and device actions to modify path labels
independently of polarization or envelope structure.


Design notes
------------

- Path labels are discrete and orthonormal.
- They distinguish spatial or logical channels.
- They contribute multiplicatively to composite mode overlap.
- They provide stable signatures for comparison and caching.
