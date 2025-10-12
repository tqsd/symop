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
from symop_proto.algebra.op_poly.from_words import op_from_words


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


class TestOpFromWords(ExtendedTestCase):
    def test_defaults_coefficients_to_ones(self):
        mA = make_mode("A")
        mB = make_mode("B")
        words = [(mA.create,), (mB.ann, mB.create)]
        out = op_from_words(words, term_factory=_factory)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].ops, (mA.create,))
        self.assertAlmostEqual(out[0].coeff, 1.0)
        self.assertEqual(out[1].ops, (mB.ann, mB.create))
        self.assertAlmostEqual(out[1].coeff, 1.0)

    def test_uses_provided_coefficients(self):
        mA = make_mode("A")
        mB = make_mode("B")
        words = [(mA.create,), (mB.create,)]
        coeffs = [2.5, -0.75]
        out = op_from_words(words, coeffs=coeffs, term_factory=_factory)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].ops, (mA.create,))
        self.assertAlmostEqual(out[0].coeff, 2.5)
        self.assertEqual(out[1].ops, (mB.create,))
        self.assertAlmostEqual(out[1].coeff, -0.75)

    def test_zip_semantics_min_length(self):
        mA = make_mode("A")
        mB = make_mode("B")
        mC = make_mode("C")
        words = [(mA.create,), (mB.create,), (mC.create,)]
        coeffs = [3.0, 4.0]
        out = op_from_words(words, coeffs=coeffs, term_factory=_factory)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].ops, (mA.create,))
        self.assertAlmostEqual(out[0].coeff, 3.0)
        self.assertEqual(out[1].ops, (mB.create,))
        self.assertAlmostEqual(out[1].coeff, 4.0)

    def test_accepts_generators(self):
        mA = make_mode("A")
        mB = make_mode("B")

        def words_gen():
            yield (mA.create,)
            yield (mB.ann,)

        out = op_from_words(
            words_gen(), coeffs=[1.2, -2.3], term_factory=_factory
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].ops, (mA.create,))
        self.assertAlmostEqual(out[0].coeff, 1.2)
        self.assertEqual(out[1].ops, (mB.ann,))
        self.assertAlmostEqual(out[1].coeff, -2.3)

    def test_uses_injected_factory_type(self):
        mA = make_mode("A")

        @dataclass(frozen=True)
        class _TermX(_Term):
            marker: int = 7

        def _factory_x(ops: Tuple[LadderOp, ...], coeff: complex) -> _TermX:
            return _TermX(ops=ops, coeff=coeff)

        out = op_from_words(
            [(mA.create,)], coeffs=[2.0], term_factory=_factory_x
        )
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], _TermX)
        self.assertEqual(out[0].ops, (mA.create,))
        self.assertAlmostEqual(out[0].coeff, 2.0)


if __name__ == "__main__":
    unittest.main()
