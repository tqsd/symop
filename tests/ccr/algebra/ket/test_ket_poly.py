import unittest

from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op.poly import OpPoly
from symop.core.operators import ModeOp
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def _make_mode(tag: str) -> ModeOp:
    env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0, phi0=0.0)
    path = PathLabel(tag)
    pol = PolarizationLabel.H()
    label = ModeLabel(path=path, pol=pol, envelope=env)
    return ModeOp(label=label)


def _ket_to_sig_map(k: KetPoly, *, eps: float = 0.0) -> dict[tuple, complex]:
    """
    Convert a KetPoly into a dict keyed by monomial signature.

    We sum coefficients for equal monomials and optionally drop near-zero
    coefficients.
    """
    out: dict[tuple, complex] = {}
    for t in k.terms:
        key = t.monomial.signature
        out[key] = out.get(key, 0.0 + 0.0j) + t.coeff
    if eps > 0.0:
        out = {k: v for k, v in out.items() if abs(v) > eps}
    return out


class TestKetPolyConstruction(unittest.TestCase):
    def test_from_ops_identity(self) -> None:
        ket = KetPoly.from_ops()
        self.assertTrue(ket.is_identity)
        self.assertEqual(len(ket.terms), 1)

    def test_from_ops_creator_only(self) -> None:
        m = _make_mode("a")
        ket = KetPoly.from_ops(creators=[m.create])
        self.assertTrue(ket.is_creator_only)
        self.assertFalse(ket.is_identity)

    def test_unique_modes(self) -> None:
        m1 = _make_mode("a")
        m2 = _make_mode("b")
        ket = KetPoly.from_ops(creators=[m1.create, m2.create]).combine_like_terms()
        modes = ket.unique_modes
        self.assertEqual(len(modes), 2)
        self.assertEqual({mm.signature for mm in modes}, {m1.signature, m2.signature})


class TestKetPolyAlgebra(unittest.TestCase):
    def test_add_sub_neg(self) -> None:
        m = _make_mode("a")
        k1 = KetPoly.from_ops(creators=[m.create], coeff=2.0)
        k2 = KetPoly.from_ops(creators=[m.create], coeff=5.0)

        s_add = (k1 + k2).combine_like_terms()
        s_sub = (k2 - k1).combine_like_terms()
        s_neg = (-k1).combine_like_terms()

        add_map = _ket_to_sig_map(s_add)
        sub_map = _ket_to_sig_map(s_sub)
        neg_map = _ket_to_sig_map(s_neg)

        self.assertEqual(len(add_map), 1)
        self.assertEqual(len(sub_map), 1)
        self.assertEqual(len(neg_map), 1)

        only_key = next(iter(add_map.keys()))
        self.assertAlmostEqual(add_map[only_key], 7.0 + 0.0j)
        self.assertAlmostEqual(sub_map[only_key], 3.0 + 0.0j)
        self.assertAlmostEqual(neg_map[only_key], -2.0 + 0.0j)

    def test_scalar_mul_div(self) -> None:
        m = _make_mode("a")
        k = KetPoly.from_ops(creators=[m.create], coeff=2.0)

        k2 = (3.0 * k).combine_like_terms()
        k3 = (k / 2.0).combine_like_terms()

        map2 = _ket_to_sig_map(k2)
        map3 = _ket_to_sig_map(k3)

        key2 = next(iter(map2.keys()))
        key3 = next(iter(map3.keys()))

        self.assertAlmostEqual(map2[key2], 6.0 + 0.0j)
        self.assertAlmostEqual(map3[key3], 1.0 + 0.0j)

        with self.assertRaises(ZeroDivisionError):
            _ = k / 0.0

        with self.assertRaises(TypeError):
            _ = k / "x"  # type: ignore[arg-type]

    def test_bool_and_eq_structural(self) -> None:
        k0 = KetPoly(())
        self.assertFalse(bool(k0))

        m = _make_mode("a")
        k1 = KetPoly.from_ops(creators=[m.create])
        k2 = KetPoly.from_ops(creators=[m.create])
        self.assertTrue(bool(k1))
        self.assertTrue(k1 == k2)

    def test_norm_and_normalize(self) -> None:
        m = _make_mode("a")
        k = KetPoly.from_ops(creators=[m.create])

        n2 = k.norm2()
        self.assertGreaterEqual(n2, 0.0)

        # normalize() should produce a unit-norm ket according to your inner().
        kn = k.normalize()
        self.assertTrue(kn.is_normalized(eps=1e-10))

        # Near-zero norm should raise.
        with self.assertRaises(ValueError):
            _ = KetPoly(()).normalize(eps=1e-14)


class TestKetPolyApply(unittest.TestCase):
    def test_apply_word_matches_from_word_times_ket(self) -> None:
        m = _make_mode("a")
        ket = KetPoly.from_ops(creators=[m.create])

        word = [m.ann, m.create]
        lhs = ket.apply_word(word).combine_like_terms()
        rhs = (KetPoly.from_word(ops=word) * ket).combine_like_terms()

        self.assertEqual(
            _ket_to_sig_map(lhs, eps=1e-12), _ket_to_sig_map(rhs, eps=1e-12)
        )

    def test_apply_words_linearity(self) -> None:
        m = _make_mode("a")
        ket = KetPoly.from_ops(creators=[m.create])

        w1 = [m.create]
        w2 = [m.ann, m.create]
        c1 = 2.0 + 0.0j
        c2 = -0.5 + 0.25j

        lhs = ket.apply_words([(c1, w1), (c2, w2)]).combine_like_terms()
        rhs = (
            (KetPoly.from_word(ops=w1) * ket) * c1
            + (KetPoly.from_word(ops=w2) * ket) * c2
        ).combine_like_terms()

        self.assertEqual(
            _ket_to_sig_map(lhs, eps=1e-12), _ket_to_sig_map(rhs, eps=1e-12)
        )

    def test_rmatmul_fallback(self) -> None:
        m = _make_mode("a")
        ket = KetPoly.from_ops(creators=[m.create])

        op = OpPoly.from_words([[m.create], [m.ann, m.create]], coeffs=[2.0, -1.0])
        lhs = (op @ ket).combine_like_terms()

        rhs = ket.apply_words((t.coeff, t.ops) for t in op.terms).combine_like_terms()
        self.assertEqual(
            _ket_to_sig_map(lhs, eps=1e-12), _ket_to_sig_map(rhs, eps=1e-12)
        )

    def test_action_associativity(self) -> None:
        m = _make_mode("a")
        ket = KetPoly.from_ops(creators=[m.create])

        op1 = OpPoly.from_words([[m.create], [m.ann]], coeffs=[1.0, 2.0])
        op2 = OpPoly.from_words([[m.ann, m.create], [m.create]], coeffs=[-0.5, 3.0])

        lhs = ((op1 @ op2) @ ket).combine_like_terms()
        rhs = (op1 @ (op2 @ ket)).combine_like_terms()

        self.assertEqual(
            _ket_to_sig_map(lhs, eps=1e-12), _ket_to_sig_map(rhs, eps=1e-12)
        )
