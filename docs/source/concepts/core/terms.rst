Symbolic terms
==============

Symop represents symbolic operator expressions using a small set of core term
objects. These objects separate three related but distinct ideas:

- general ordered operator words
- normal-ordered monomials
- coefficient-bearing ket- and density-like terms

This page introduces the main classes:

- :class:`symop.core.terms.op_term.OpTerm`
- :class:`symop.core.monomial.Monomial`
- :class:`symop.core.terms.ket_term.KetTerm`
- :class:`symop.core.terms.density_term.DensityTerm`

Setup
-----

.. jupyter-execute::

   import symop.viz as viz

   from symop.core.operators import ModeOp
   from symop.core.monomial import Monomial
   from symop.core.terms.op_term import OpTerm
   from symop.core.terms.ket_term import KetTerm
   from symop.core.terms.density_term import DensityTerm

   from symop.modes.labels import ModeLabel
   from symop.modes.labels.path import Path
   from symop.modes.labels.polarization import Polarization
   from symop.modes.envelopes import GaussianEnvelope

   env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0)
   mode_a = ModeOp(
       label=ModeLabel(
           path=Path("A"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="a",
   )
   mode_b = ModeOp(
       label=ModeLabel(
           path=Path("B"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="b",
   )

Operator words with ``OpTerm``
------------------------------

:class:`symop.core.terms.op_term.OpTerm` stores a general ordered product of
ladder operators together with a complex coefficient.

.. jupyter-execute::

   op_term = OpTerm(
       ops=(mode_a.cre, mode_b.ann, mode_a.ann),
       coeff=2.0 + 1.0j,
   )
   viz.display(op_term)

Unlike a normal-ordered object, an ``OpTerm`` preserves the explicit operator
order. This makes it useful for intermediate symbolic manipulations and
adjoint operations.

.. jupyter-execute::

   op_term.adjoint()

Normal-ordered monomials
------------------------

:class:`symop.core.monomial.Monomial` stores creators and annihilators
separately.

.. jupyter-execute::

   mon = Monomial(
       creators=(mode_a.cre, mode_b.cre),
       annihilators=(mode_b.ann,),
   )
   viz.display(mon)

This representation is structurally richer than a raw operator word. It makes
queries such as ``is_creator_only``, ``is_identity``, and ``mode_ops`` easy and
stable.

.. jupyter-execute::

   mon.is_creator_only, mon.is_identity, mon.mode_ops

Ket-like terms
--------------

:class:`symop.core.terms.ket_term.KetTerm` combines a scalar coefficient with a
single monomial.

.. jupyter-execute::

   ket_term = KetTerm(2.0, mon)
   viz.display(ket_term)

.. jupyter-execute::

   ket_term.creation_count, ket_term.annihilation_count, ket_term.total_degree

Density-like terms
------------------

:class:`symop.core.terms.density_term.DensityTerm` stores a coefficient and two
monomials, one acting on the left and one on the right.

.. jupyter-execute::

   left = Monomial(creators=(mode_a.cre,), annihilators=())
   right = Monomial(creators=(), annihilators=(mode_b.ann,))
   density_term = DensityTerm(1.0, left, right)
   viz.display(density_term)

This is the natural structural form for density-operator bookkeeping and
related symbolic manipulations.

.. jupyter-execute::

   density_term.is_creator_only_left, density_term.is_annihilator_only_right

Choosing between the term types
-------------------------------

Use :class:`symop.core.terms.op_term.OpTerm` when the explicit operator order
matters.

Use :class:`symop.core.monomial.Monomial` when you want a normal-ordered
creator/annihilator decomposition.

Use :class:`symop.core.terms.ket_term.KetTerm` when you need a scalar
coefficient attached to one monomial.

Use :class:`symop.core.terms.density_term.DensityTerm` when you need separate
left and right monomials.

API reference
-------------

- :class:`symop.core.terms.op_term.OpTerm`
- :class:`symop.core.monomial.Monomial`
- :class:`symop.core.terms.ket_term.KetTerm`
- :class:`symop.core.terms.density_term.DensityTerm`
- :class:`symop.core.operators.ModeOp`
- :class:`symop.core.operators.LadderOp`
