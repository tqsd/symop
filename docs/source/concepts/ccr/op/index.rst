Operator algebra
================

The operator layer represents finite linear combinations of operator words,

.. math::

   \mathcal{O} = \sum_i c_i W_i,

where each :math:`W_i` is an ordered ladder-operator word.

The central object is :class:`symop.ccr.op.poly.OpPoly`.

Main ideas
----------

Unlike the ket and density layers, operator multiplication here is based on
word concatenation. No CCR rewriting is applied automatically during operator
polynomial multiplication. If identical words arise, they can be merged by
normalization.

Basic constructors
------------------

.. jupyter-execute::

   import symop.viz as viz

   from symop.core.operators import ModeOp
   from symop.ccr.algebra.op.poly import OpPoly
   from symop.modes.envelopes import GaussianEnvelope
   from symop.modes.labels import ModeLabel, Path, Polarization

   env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0)

   mode = ModeOp(
       label=ModeLabel(
           envelope=env,
           path=Path("A"),
           polarization=Polarization.H(),
           ),
       user_label="a",
   )
   a = OpPoly.a(mode)
   adag = OpPoly.adag(mode)
   n = OpPoly.n(mode)

   viz.display(a)
   viz.display(adag)
   viz.display(n)

Quadratures
-----------

.. jupyter-execute::

   q = OpPoly.q(mode)
   p = OpPoly.p(mode)

   viz.display(q)
   viz.display(p)

Actions via ``@``
-----------------

Operator polynomials can act symbolically on ket and density polynomials.

.. jupyter-execute::

   from symop.ccr.algebra.ket.poly import KetPoly

   psi = KetPoly.from_ops(coeff=1.0)
   result = a @ psi
   viz.display(result)

API links
---------

- :class:`symop.ccr.algebra.op.poly.OpPoly`
- :func:`symop.ccr.algebra.op.from_words.from_words`
- :func:`symop.ccr.algebra.op.multiply.multiply`
- :func:`symop.ccr.algebra.op.combine.combine_like_terms`
