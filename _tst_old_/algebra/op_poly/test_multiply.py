from __future__ import annotations
import unittest
from dataclasses import dataclass
from typing import Tuple

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp, LadderOp
from symop_proto.algebra.op_poly.multiply import op_multiply


def make_mode(
    path: str = "A",
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


@dataclass(frozen=True)
class _Term:
    ops: Tuple[LadderOp, ...]
    coeff: complex

    @property
    def signature(self) -> tuple:
        return (
            "stub",
            tuple(op.mode.label.signature for op in self.ops),
            len(self.ops),
        )

    def approx_signature(self, **env_kw) -> tuple:
        return self.signature


def _factory(ops: Tuple[LadderOp, ...], coeff: complex) -> _Term:
    return _Term(ops=ops, coeff=coeff)


class TestOpMultiply(ExtendedTestCase):
    def test_cartesian_product_and_coeff_product(self):
        mA = make_mode("A")
        mB = make_mode("B")
        a = (
            _Term(ops=(mA.create,), coeff=2.0),
            _Term(ops=(mA.ann,), coeff=-3.0),
        )
        b = (_Term(ops=(mB.create,), coeff=1.5),)
        out = op_multiply(a, b, term_factory=_factory)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].ops, (mA.create, mB.create))
        self.assertAlmostEqual(out[0].coeff, 3.0)
        self.assertEqual(out[1].ops, (mA.ann, mB.create))
        self.assertAlmostEqual(out[1].coeff, -4.5)

    def test_empty_right_behaves_as_identity(self):
        mA = make_mode("A")
        a = (_Term(ops=(mA.create,), coeff=2.0),)
        b: Tuple[_Term, ...] = ()
        out = op_multiply(a, b, term_factory=_factory)
        self.assertEqual(out, ())

    def test_empty_left_behaves_as_identity(self):
        mA = make_mode("A")
        a: Tuple[_Term, ...] = ()
        b = (_Term(ops=(mA.create,), coeff=2.0),)
        out = op_multiply(a, b, term_factory=_factory)
        self.assertEqual(out, ())

    def test_order_preserved_in_concatenation(self):
        mA = make_mode("A")
        mB = make_mode("B")
        a = (_Term(ops=(mA.ann,), coeff=1.0),)
        b = (_Term(ops=(mB.create, mB.ann), coeff=2.0),)
        out = op_multiply(a, b, term_factory=_factory)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ops, (mA.ann, mB.create, mB.ann))
        self.assertAlmostEqual(out[0].coeff, 2.0)

    def test_uses_injected_factory(self):
        mA = make_mode("A")

        @dataclass(frozen=True)
        class _TermX(_Term):
            marker: int = 1

        def _factory_x(ops: Tuple[LadderOp, ...], coeff: complex) -> _TermX:
            return _TermX(ops=ops, coeff=coeff)

        a = (_Term(ops=(mA.create,), coeff=1.0),)
        b = (_Term(ops=(mA.ann,), coeff=2.0),)
        out = op_multiply(a, b, term_factory=_factory_x)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], _TermX)
        self.assertEqual(out[0].ops, (mA.create, mA.ann))
        self.assertAlmostEqual(out[0].coeff, 2.0)


if __name__ == "__main__":
    unittest.main()
