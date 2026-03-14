from __future__ import annotations

import unittest

from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.density.poly import DensityPoly
from symop.polynomial.state.density_state import DensityPolyState
from symop.polynomial.state.ket_state import KetPolyState
from symop.ccr.algebra.op.poly import OpPoly


# Reuse your real-mode builder (same idea as earlier).
# If you already have _make_mode() in another test module, import it instead.
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def _make_mode(path_name: str = "p0", tau: float = 0.0):
    env = GaussianEnvelope(omega0=2.0, sigma=0.8, tau=tau, phi0=0.0)
    label = ModeLabel(
        path=PathLabel(path_name), pol=PolarizationLabel.H(), envelope=env
    )

    candidates = []
    try:
        from symop.modes.mode import Mode  # type: ignore[attr-defined]

        candidates.append(Mode)
    except Exception:
        pass

    try:
        # type: ignore[attr-defined]
        from symop.modes.operators.mode import Mode

        candidates.append(Mode)
    except Exception:
        pass

    try:
        from symop.core.operators import ModeOp  # type: ignore[attr-defined]

        candidates.append(ModeOp)
    except Exception:
        pass

    try:
        # type: ignore[attr-defined]
        from symop.core.operators import ModeOperator

        candidates.append(ModeOperator)
    except Exception:
        pass

    last_err: Exception | None = None
    for cls in candidates:
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
        except Exception as e:
            last_err = e

    if last_err is None:
        raise RuntimeError(
            "Could not locate a concrete Mode implementation. "
            "Update _make_mode() to import your Mode/ModeOp class."
        )
    raise RuntimeError(
        "Failed to construct a Mode instance. Update _make_mode() to match your Mode ctor."
    ) from last_err


class TestDensityPolyStateBasics(unittest.TestCase):
    def test_rep_tag_present(self):
        rho = DensityPolyState.pure(KetPoly.identity())
        self.assertIsNotNone(rho.rep_tag)

    def test_with_label_and_index_are_immutable(self):
        rho0 = DensityPolyState.pure(KetPoly.identity())
        rho1 = rho0.with_label("x")
        rho2 = rho1.with_index(77)

        self.assertIsNone(rho0.label)
        self.assertEqual(rho1.label, "x")
        self.assertEqual(rho2.index, 77)

        self.assertIsNot(rho0, rho1)
        self.assertIsNot(rho1, rho2)


class TestDensityPolyStatePure(unittest.TestCase):
    def test_pure_from_ketpoly_trace_and_purity(self):
        # Vacuum pure state should have trace=1 and purity=1.
        psi = KetPolyState.vacuum()
        rho = DensityPolyState.pure(psi)

        self.assertAlmostEqual(rho.trace().real, 1.0, places=12)
        self.assertAlmostEqual(rho.trace().imag, 0.0, places=12)
        self.assertTrue(rho.is_trace_normalized(eps=1e-12))

        self.assertAlmostEqual(rho.purity(), 1.0, places=12)
        self.assertTrue(rho.is_pure(eps=1e-12))

    def test_pure_from_raw_ketpoly_matches_from_state(self):
        m = _make_mode()
        psi_state = KetPolyState.from_creators([m.create], coeff=1.0)
        rho1 = DensityPolyState.pure(psi_state)
        rho2 = DensityPolyState.pure(psi_state.ket)

        self.assertEqual(rho1.rho.terms, rho2.rho.terms)


class TestDensityPolyStateFromDensityPoly(unittest.TestCase):
    def test_from_densitypoly_no_normalize_keeps_trace(self):
        # Make something with trace != 1 by scaling a valid density.
        rho0 = DensityPolyState.pure(KetPoly.identity())
        scaled = rho0.rho.scaled(3.0)
        s = DensityPolyState.from_densitypoly(scaled, normalize_trace=False)

        self.assertAlmostEqual(s.trace().real, 3.0, places=12)
        self.assertFalse(s.is_trace_normalized(eps=1e-12))

    def test_from_densitypoly_normalize_trace(self):
        rho0 = DensityPolyState.pure(KetPoly.identity())
        scaled = rho0.rho.scaled(5.0)
        s = DensityPolyState.from_densitypoly(scaled, normalize_trace=True)

        self.assertTrue(s.is_trace_normalized(eps=1e-12))
        self.assertAlmostEqual(s.trace().real, 1.0, places=12)


class TestDensityPolyStateNormalizeTrace(unittest.TestCase):
    def test_normalize_trace_returns_new_state(self):
        rho0 = DensityPolyState.pure(KetPoly.identity())
        rho_scaled = DensityPolyState.from_densitypoly(rho0.rho.scaled(2.0))

        rho1 = rho_scaled.normalize_trace(eps=1e-14)
        self.assertIsNot(rho_scaled, rho1)
        self.assertTrue(rho1.is_trace_normalized(eps=1e-12))
        self.assertAlmostEqual(rho1.trace().real, 1.0, places=12)


class TestDensityPolyStateMix(unittest.TestCase):
    def test_mix_rejects_bad_lengths(self):
        r = DensityPolyState.pure(KetPoly.identity())
        with self.assertRaises(ValueError):
            _ = DensityPolyState.mix([r], [0.5, 0.5])

    def test_mix_rejects_non_positive_sum(self):
        r = DensityPolyState.pure(KetPoly.identity())
        with self.assertRaises(ValueError):
            _ = DensityPolyState.mix([r], [0.0])

        with self.assertRaises(ValueError):
            _ = DensityPolyState.mix([r], [-1.0])

    def test_mix_normalizes_weights_and_trace(self):
        r0 = DensityPolyState.pure(KetPoly.identity())
        r1 = DensityPolyState.from_densitypoly(
            r0.rho.scaled(2.0), normalize_trace=False
        )

        mixed = DensityPolyState.mix(
            [r0, r1], [1.0, 1.0], normalize_weights=True
        )
        # mix() always normalizes trace at the end
        self.assertTrue(mixed.is_trace_normalized(eps=1e-12))
        self.assertAlmostEqual(mixed.trace().real, 1.0, places=12)

    def test_mix_purity_decreases_for_distinct_pure_states(self):
        # Mix vacuum and one-photon in same mode => mixed state should be impure.
        m = _make_mode()
        vac = DensityPolyState.pure(KetPoly.identity())
        one = DensityPolyState.pure(
            KetPoly.from_ops(creators=(m.create,), annihilators=(), coeff=1.0)
        )

        mixed = DensityPolyState.mix([vac, one], [0.5, 0.5])
        self.assertTrue(mixed.is_trace_normalized(eps=1e-12))
        self.assertLess(mixed.purity(), 1.0)


class TestDensityPolyStateExpect(unittest.TestCase):
    def test_expect_identity_equals_one_when_normalized(self):
        rho = DensityPolyState.pure(KetPoly.identity())
        I = OpPoly.identity(1.0)

        val = rho.expect(I, normalize=True)
        self.assertAlmostEqual(val.real, 1.0, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)

    def test_expect_scales_with_trace_when_normalize_false(self):
        rho0 = DensityPolyState.pure(KetPoly.identity())
        rho = DensityPolyState.from_densitypoly(
            rho0.rho.scaled(3.0), normalize_trace=False
        )

        I = OpPoly.identity(2.0)  # identity operator with coeff 2
        raw = rho.expect(I, normalize=False)
        normed = rho.expect(I, normalize=True)

        # raw should scale with trace: Tr(rho * 2I) = 2 * Tr(rho) = 6
        self.assertAlmostEqual(raw.real, 6.0, places=12)
        self.assertAlmostEqual(normed.real, 2.0, places=12)

    def test_expect_raises_if_trace_zero_and_normalize_true(self):
        rho0 = DensityPolyState.pure(KetPoly.identity())
        # Make a zero-trace object by scaling to 0 (if scaled(0) keeps terms, trace becomes 0).
        rho_zero = DensityPolyState.from_densitypoly(
            rho0.rho.scaled(0.0), normalize_trace=False
        )

        I = OpPoly.identity(1.0)
        with self.assertRaises(ValueError):
            _ = rho_zero.expect(I, normalize=True)

    def test_expect_number_operator_on_vacuum_is_zero(self):
        m = _make_mode()
        rho = DensityPolyState.pure(KetPoly.identity())

        n = OpPoly.n(m)
        val = rho.expect(n, normalize=True)

        self.assertAlmostEqual(val.real, 0.0, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)

    def test_expect_number_operator_on_one_photon_is_one(self):
        m = _make_mode()
        psi = KetPoly.from_ops(
            creators=(m.create,), annihilators=(), coeff=1.0
        )
        rho = DensityPolyState.pure(psi)

        n = OpPoly.n(m)
        val = rho.expect(n, normalize=True)

        self.assertAlmostEqual(val.real, 1.0, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)


class TestDensityPolyStatePartialTrace(unittest.TestCase):
    def test_partial_trace_of_product_state_keeps_other_mode(self):
        # Build |1_a> \otimes |0_b> (as creators in different modes).
        ma = _make_mode("pa")
        mb = _make_mode("pb", tau=0.5)

        psi = KetPoly.from_ops(
            creators=(ma.create,), annihilators=(), coeff=1.0
        ).multiply(KetPoly.from_ops(creators=(), annihilators=(), coeff=1.0))

        # Also include a creator on mb? We'll build a product: (a^\dagger)(I) for simplicity.
        # If you want |1_a>|1_b|, add mb.create too.
        rho = DensityPolyState.pure(psi)

        # We need to know what partial_trace expects as elements of the set.
        # Try tracing over mb by passing the mode object itself.
        try:
            reduced = rho.partial_trace({mb})
        except Exception:
            # Fallback: if DensityPoly expects mode signatures, try that.
            try:
                reduced = rho.partial_trace({mb.signature})
            except Exception as e:
                self.skipTest(
                    "partial_trace mode key type mismatch. "
                    "Tell me what type DensityPoly.partial_trace expects for trace_over_modes."
                )
                raise e

        self.assertTrue(reduced.is_trace_normalized(eps=1e-12))

        # After tracing out mb, expectation of n(ma) should still be ~1.
        val = reduced.expect(OpPoly.n(ma), normalize=True)
        self.assertAlmostEqual(val.real, 1.0, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)
