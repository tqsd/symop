from __future__ import annotations

import unittest

from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.operators import OperatorKind
from symop.polynomial.state.ket_state import KetPolyState

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def _make_mode():
    """
    Construct a concrete ModeOpProto instance using the actual library.

    This function intentionally performs a few import/constructor fallbacks,
    because different refactors tend to move the concrete Mode class around.

    If this fails in your tree, update ONLY this function to match your
    concrete mode implementation.
    """
    env = GaussianEnvelope(omega0=2.0, sigma=0.8, tau=0.0, phi0=0.0)
    label = ModeLabel(
        path=PathLabel("p0"), pol=PolarizationLabel.H(), envelope=env
    )

    # Try common locations for a concrete Mode implementation.
    candidates = []
    try:
        from symop.modes.mode import Mode  # type: ignore[attr-defined]

        candidates.append(("symop.modes.mode.Mode", Mode))
    except Exception:
        pass

    try:
        # type: ignore[attr-defined]
        from symop.modes.operators.mode import Mode

        candidates.append(("symop.modes.operators.mode.Mode", Mode))
    except Exception:
        pass

    try:
        from symop.core.operators import ModeOp  # type: ignore[attr-defined]

        candidates.append(("symop.core.operators.ModeOp", ModeOp))
    except Exception:
        pass

    try:
        # type: ignore[attr-defined]
        from symop.core.operators import ModeOperator

        candidates.append(("symop.core.operators.ModeOperator", ModeOperator))
    except Exception:
        pass

    last_err: Exception | None = None
    for _name, cls in candidates:
        # Try a few constructor styles:
        # - Mode(label)
        # - Mode(mode_label=label)
        # - Mode(label=label)
        # - Mode.from_label(label)
        try:
            try:
                return cls(label)
            except TypeError:
                pass

            try:
                return cls(mode_label=label)
            except TypeError:
                pass

            try:
                return cls(label=label)
            except TypeError:
                pass

            if hasattr(cls, "from_label"):
                return cls.from_label(label)

        except Exception as e:  # pragma: no cover
            last_err = e

    if last_err is None:
        raise RuntimeError(
            "Could not locate a concrete Mode implementation. "
            "Update _make_mode() to import your Mode/ModeOp class."
        )

    raise RuntimeError(
        "Failed to construct a Mode instance from the discovered candidates. "
        "Update _make_mode() to match your concrete Mode constructor."
    ) from last_err


class TestKetPolyStateConstruction(unittest.TestCase):
    def test_vacuum_is_creator_only_and_identity(self):
        s = KetPolyState.vacuum()
        self.assertTrue(s.ket.is_creator_only)
        self.assertTrue(s.ket.is_identity)
        self.assertEqual(s.norm2, 1.0)

    def test_from_creators_rejects_annihilators(self):
        m = _make_mode()
        with self.assertRaises(ValueError):
            KetPolyState.from_creators([m.ann])

    def test_from_creators_accepts_creators(self):
        m = _make_mode()
        s = KetPolyState.from_creators([m.create])
        self.assertTrue(s.ket.is_creator_only)
        self.assertGreaterEqual(s.ket.creation_count, 1)
        self.assertEqual(s.ket.annihilation_count, 0)

    def test_post_init_rejects_non_creator_only_ket(self):
        m = _make_mode()
        bad = KetPoly.from_ops(creators=(), annihilators=(m.ann,), coeff=1.0)
        with self.assertRaises(ValueError):
            KetPolyState.from_ketpoly(bad)

    def test_from_ketpoly_combines_like_terms(self):
        m = _make_mode()

        # Build a ket with duplicate identical terms via addition,
        # then rely on from_ketpoly calling combine_like_terms().
        k1 = KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=1.0)
        k2 = KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=2.0)
        k = k1 + k2

        # If combine_like_terms works, the combined ket should have 1 term.
        s = KetPolyState.from_ketpoly(k)
        self.assertEqual(len(s.ket.terms), 1)

        # Norm should be positive and finite.
        self.assertTrue(s.norm2 > 0.0)

    def test_rep_tag_is_poly_rep(self):
        s = KetPolyState.vacuum()
        self.assertIsNotNone(s.rep_tag)


class TestKetPolyStateMutationHelpers(unittest.TestCase):
    def test_with_label_roundtrip(self):
        s0 = KetPolyState.vacuum()
        s1 = s0.with_label("hello")
        s2 = s1.with_label(None)

        self.assertIsNone(s0.label)
        self.assertEqual(s1.label, "hello")
        self.assertIsNone(s2.label)

        # Ensure immutability behavior: new instances.
        self.assertIsNot(s0, s1)
        self.assertIsNot(s1, s2)

    def test_with_index_overrides(self):
        s0 = KetPolyState.vacuum()
        s1 = s0.with_index(123)
        s2 = s1.with_index(None)

        self.assertNotEqual(s0.index, 123)
        self.assertEqual(s1.index, 123)
        self.assertIsNone(s2.index)


class TestKetPolyStateNormalization(unittest.TestCase):
    def test_normalized_makes_unit_norm(self):
        m = _make_mode()
        # Unnormalized ket: coeff=3
        s0 = KetPolyState.from_creators([m.create], coeff=3.0)
        self.assertFalse(s0.is_normalized(eps=1e-12))

        s1 = s0.normalized(eps=1e-14)
        self.assertTrue(s1.is_normalized(eps=1e-12))
        self.assertAlmostEqual(s1.norm2, 1.0, places=12)

    def test_normalized_does_not_mutate_original(self):
        m = _make_mode()
        s0 = KetPolyState.from_creators([m.create], coeff=2.0)
        n0 = s0.norm2

        s1 = s0.normalized(eps=1e-14)
        self.assertNotEqual(s0.norm2, 1.0)
        self.assertAlmostEqual(s0.norm2, n0, places=12)
        self.assertAlmostEqual(s1.norm2, 1.0, places=12)

    def test_normalize_raises_on_near_zero_norm(self):
        m = _make_mode()
        # Build a near-zero ket by scaling vacuum.
        k = KetPoly.identity().scaled(1e-30)
        # KetPolyState.from_ketpoly is allowed (creator-only), but normalize should fail.
        s = KetPolyState.from_ketpoly(k)

        with self.assertRaises(ValueError):
            _ = s.normalized(eps=1e-14)


class TestKetPolyStateToDensity(unittest.TestCase):
    def test_to_density_returns_density_state(self):
        m = _make_mode()
        s = KetPolyState.from_creators([m.create], coeff=1.0)
        rho = s.to_density()

        self.assertTrue(hasattr(rho, "rho"))


class TestKetPolyStateOperatorKindConsistency(unittest.TestCase):
    def test_created_ops_are_tagged_create(self):
        m = _make_mode()
        op = m.create
        self.assertEqual(getattr(op, "kind", None), OperatorKind.CREATE)

    def test_ann_ops_are_tagged_annihilate(self):
        m = _make_mode()
        op = m.ann
        self.assertEqual(getattr(op, "kind", None), OperatorKind.ANN)


class TestKetPolyStateAlgebraicContent(unittest.TestCase):
    def test_from_ketpoly_combines_like_terms_sums_coeff(self):
        m = _make_mode()

        k1 = KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=1.0)
        k2 = KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=2.0)
        k = k1 + k2

        s = KetPolyState.from_ketpoly(k)

        self.assertEqual(len(s.ket.terms), 1)
        term = s.ket.terms[0]
        self.assertAlmostEqual(term.coeff.real, 3.0, places=12)
        self.assertAlmostEqual(term.coeff.imag, 0.0, places=12)

    def test_from_creators_preserves_coeff_on_single_word(self):
        m = _make_mode()

        alpha = 1.25 - 0.5j
        s = KetPolyState.from_creators([m.create], coeff=alpha)

        self.assertEqual(len(s.ket.terms), 1)
        term = s.ket.terms[0]
        self.assertAlmostEqual(term.coeff.real, alpha.real, places=12)
        self.assertAlmostEqual(term.coeff.imag, alpha.imag, places=12)

    def test_normalized_scales_coeff_correctly_single_term(self):
        m = _make_mode()

        alpha = 3.0 + 4.0j
        s0 = KetPolyState.from_creators([m.create], coeff=alpha)
        s1 = s0.normalized(eps=1e-14)

        self.assertEqual(len(s1.ket.terms), 1)
        t1 = s1.ket.terms[0]

        # For a single ket term c*M where <M|M> = 1, normalization scales by 1/|c|.
        mag = abs(alpha)
        self.assertAlmostEqual(t1.coeff.real, (alpha / mag).real, places=12)
        self.assertAlmostEqual(t1.coeff.imag, (alpha / mag).imag, places=12)
        self.assertAlmostEqual(s1.norm2, 1.0, places=12)


class TestKetPolyStateToDensityContent(unittest.TestCase):
    def test_to_density_vacuum_is_single_identity_density_term(self):
        s = KetPolyState.vacuum()
        rho_state = s.to_density()

        rho = rho_state.rho
        self.assertEqual(len(rho.terms), 1)

        dt = rho.terms[0]
        self.assertAlmostEqual(dt.coeff.real, 1.0, places=12)
        self.assertAlmostEqual(dt.coeff.imag, 0.0, places=12)
        self.assertTrue(dt.left.is_identity)
        self.assertTrue(dt.right.is_identity)

    def test_to_density_pure_single_term_outer_product(self):
        m = _make_mode()

        alpha = 2.0 - 1.0j
        s = KetPolyState.from_creators([m.create], coeff=alpha)
        rho_state = s.to_density()

        # Ket has exactly one term here.
        self.assertEqual(len(s.ket.terms), 1)
        kt = s.ket.terms[0]

        rho = rho_state.rho
        self.assertEqual(len(rho.terms), 1)

        dt = rho.terms[0]

        # For a pure state |psi> = alpha * M, density is |psi><psi|:
        # coeff should be alpha * conj(alpha), and left/right monomials should match M.
        expected = alpha * alpha.conjugate()

        self.assertAlmostEqual(dt.coeff.real, expected.real, places=12)
        self.assertAlmostEqual(dt.coeff.imag, expected.imag, places=12)

        self.assertEqual(dt.left.signature, kt.monomial.signature)
        self.assertEqual(dt.right.signature, kt.monomial.signature)
        self.assertTrue(dt.is_diagonal_in_monomials)

    def test_to_density_pure_two_term_has_cross_terms(self):
        m = _make_mode()

        k1 = KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=1.0)
        k2 = KetPoly.from_ops(creators=(), annihilators=(), coeff=1.0)
        s = KetPolyState.from_ketpoly(k1 + k2)

        rho_state = s.to_density()
        rho = rho_state.rho

        # If |psi> has 2 ket terms, a pure density generally has 4 outer-product terms
        # before any combining/canonicalization.
        # We avoid assuming exact monomial ordering; just check term count >= 4 or == 4.
        self.assertGreaterEqual(len(rho.terms), 4)

        # Also check that at least one term is diagonal (M_i, M_i).
        self.assertTrue(any(dt.is_diagonal_in_monomials for dt in rho.terms))
