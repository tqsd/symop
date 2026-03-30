CCR algebra
===========

The CCR layer implements symbolic bosonic operator algebra on top of the core
term objects. Instead of forming matrix representations, it rewrites operator
words into normally ordered expansions using commutation relations.

At this layer, Symop works with three closely related symbolic objects:

- operator polynomials, represented by :class:`symop.ccr.algebra.op.poly.OpPoly`
- ket polynomials, represented by :class:`symop.ccr.algebra.ket.poly.KetPoly`
- density polynomials, represented by :class:`symop.ccr.algebra.density.poly.DensityPoly`

The main ideas are:

- operator words are expanded symbolically using CCR normal ordering
- kets are finite sums of normally ordered monomials
- density operators are finite sums of outer products :math:`|L\rangle\langle R|`
- actions, products, traces, and inner products are all computed symbolically

.. toctree::
   :maxdepth: 1

   ket/index
   density/index
   op/index
   normal_ordering_and_trace
