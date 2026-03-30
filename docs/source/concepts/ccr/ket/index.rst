Ket algebra
===========

The ket layer represents symbolic state-like expressions as finite sums of
normally ordered monomials,

.. math::

   |\psi\rangle \sim \sum_i c_i M_i,

where each :math:`M_i` stores creators on the left and annihilators on the
right.

The central object is :class:`symop.ccr.ket.poly.KetPoly`.

Main ideas
----------

There are two main ways to build symbolic ket expressions.

- :func:`symop.ccr.algebra.ket.from_ops.ket_from_ops` builds a ket term from
  creators and annihilators that are already assumed to be in normal order.
- :func:`symop.ccr.algebra.ket.from_word.ket_from_word` takes an arbitrary
  operator word and expands it into a normally ordered sum using CCR
  contractions.

This distinction is important: ``from_ops`` does not commute operators, while
``from_word`` does.

Constructing from a normal-ordered monomial
-------------------------------------------

.. jupyter-execute::

   import symop.viz as viz

   from symop.core.operators import ModeOp
   from symop.modes.labels import ModeLabel
   from symop.modes.labels.path import Path
   from symop.modes.labels.polarization import Polarization
   from symop.modes.envelopes import GaussianEnvelope

   from symop.ccr.algebra.ket.poly import KetPoly

   env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0)
   mode = ModeOp(
       label=ModeLabel(
           path=Path("A"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="a",
   )

   psi = KetPoly.from_ops(creators=(mode.cre,))
   viz.display(psi)

Normal ordering from a word
---------------------------

.. jupyter-execute::

   import symop.viz as viz
   from symop.ccr.algebra.ket.poly import KetPoly

   expr = KetPoly.from_word(ops=(mode.ann, mode.cre))
   viz.display(expr)

This expansion contains both the contracted scalar part and the normally
ordered operator part.

Main operations
---------------

- :meth:`symop.ccr.ket.poly.KetPoly.combine_like_terms`
- :meth:`symop.ccr.ket.poly.KetPoly.multiply`
- :meth:`symop.ccr.ket.poly.KetPoly.apply_word`
- :meth:`symop.ccr.ket.poly.KetPoly.inner`
- :meth:`symop.ccr.ket.poly.KetPoly.normalize`

Example: symbolic inner product
-------------------------------

.. jupyter-execute::

   psi = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
   phi = KetPoly.from_ops(creators=(mode.cre,), coeff=3.0)

   psi.inner(phi)

API links
---------

- :class:`symop.ccr.algebra.ket.poly.KetPoly`
- :func:`symop.ccr.algebra.ket.from_ops.ket_from_ops`
- :func:`symop.ccr.algebra.ket.from_word.ket_from_word`
- :func:`symop.ccr.algebra.ket.multiply.ket_multiply`
- :func:`symop.ccr.algebra.ket.inner.ket_inner`
