from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.from_ops import ket_from_ops
from symop_proto.core.monomial import Monomial
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
import symop_proto.algebra.ket.from_ops as fops
import symop_proto.core.monomial as mon_mod

print("from_ops path:", fops.__file__)
print("Monomial path:", mon_mod.__file__)


def make_mode(
    path: str = "A",
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    tau: float = 0.0,
    phi: float = 0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestKetFromOps(ExtendedTestCase):
    def test_default_vacuum(self):
        terms = ket_from_ops()
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0j)
        self.assertEqual(t.monomial.creators, ())
        self.assertEqual(t.monomial.annihilators, ())

    def test_creators_only(self):
        a = make_mode("A")
        b = make_mode("B")
        coeff = 3.0 - 2.0j
        terms = ket_from_ops(creators=(a.create, b.create), coeff=coeff)
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertComplexAlmostEqual(t.coeff, coeff)
        expected_sig = Monomial((a.create, b.create), ()).signature
        self.assertEqual(t.monomial.signature, expected_sig)

    def test_with_annihilators(self):
        a = make_mode("A")
        terms = ket_from_ops(
            creators=(a.create,), annihilators=(a.ann,), coeff=1.0
        )
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.monomial.creators, (a.create,))
        self.assertEqual(t.monomial.annihilators, (a.ann,))

    def test_tiny_coeff_is_dropped_by_eps(self):
        a = make_mode("A")
        tiny = 1e-13
        terms = ket_from_ops(creators=(a.create,), coeff=tiny)
        self.assertEqual(terms, ())

    def test_iterable_inputs_allow_generators(self):
        a = make_mode("A")
        b = make_mode("B")
        creators_gen = (op for op in (a.create, b.create))
        annihilators_gen = (op for op in ())  # empty generator
        terms = ket_from_ops(
            creators=creators_gen, annihilators=annihilators_gen, coeff=2.0
        )
        self.assertEqual(len(terms), 1)
        t = terms[0]
        expected_sig = Monomial((a.create, b.create), ()).signature
        self.assertEqual(t.monomial.signature, expected_sig)
        self.assertComplexAlmostEqual(t.coeff, 2.0 + 0j)
