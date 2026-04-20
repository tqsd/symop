from __future__ import annotations

import unittest

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.polynomial.channels.primitives.unitary_dilation import (
    UnitaryDilation,
    apply_unitary_dilation_densitypoly,
    apply_unitary_dilation_densitypoly_direct,
)

from tests.polynomial.state._builders import make_test_mode


class TestUnitaryDilation(unittest.TestCase):
    def test_rejects_duplicate_modes(self):
        mode = make_test_mode(name="a", path="p0")
        U = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            UnitaryDilation(
                modes=(mode, mode),
                U=U,
                trace_out_modes=(),
            )

    def test_rejects_duplicate_trace_out_modes(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        U = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            UnitaryDilation(
                modes=(mode_a, mode_b),
                U=U,
                trace_out_modes=(mode_a, mode_a),
            )

    def test_rejects_trace_out_mode_not_in_modes(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        U = np.eye(1, dtype=np.complex128)

        with self.assertRaises(ValueError):
            UnitaryDilation(
                modes=(mode_a,),
                U=U,
                trace_out_modes=(mode_b,),
            )

    def test_reuses_linear_mode_map_unitary_validation(self):
        mode = make_test_mode(name="a", path="p0")
        U = np.array([[2.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            UnitaryDilation(
                modes=(mode,),
                U=U,
                trace_out_modes=(),
                check_unitary=True,
            )


class TestApplyUnitaryDilationDensityPoly(unittest.TestCase):
    def test_identity_dilation_without_trace_out_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)
        dilation = UnitaryDilation(
            modes=(mode,),
            U=np.array([[1.0]], dtype=np.complex128),
            trace_out_modes=(),
        )

        result = apply_unitary_dilation_densitypoly(rho, dilation=dilation)

        self.assertEqual(result, rho.combine_like_terms())

    def test_identity_dilation_tracing_out_one_mode_removes_that_mode(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(
            creators=(mode_a.cre, mode_b.cre),
            annihilators=(),
            coeff=1.0,
        )
        rho = DensityPoly.pure(ket)
        dilation = UnitaryDilation(
            modes=(mode_a, mode_b),
            U=np.eye(2, dtype=np.complex128),
            trace_out_modes=(mode_b,),
        )

        result = apply_unitary_dilation_densitypoly(rho, dilation=dilation)

        remaining_sigs = {mode.signature for mode in result.unique_modes}
        self.assertIn(mode_a.signature, remaining_sigs)
        self.assertNotIn(mode_b.signature, remaining_sigs)

    def test_direct_wrapper_matches_object_wrapper(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(
            creators=(mode_a.cre, mode_b.cre),
            annihilators=(),
            coeff=1.0,
        )
        rho = DensityPoly.pure(ket)
        U = np.eye(2, dtype=np.complex128)
        dilation = UnitaryDilation(
            modes=(mode_a, mode_b),
            U=U,
            trace_out_modes=(mode_b,),
        )

        result_a = apply_unitary_dilation_densitypoly(rho, dilation=dilation)
        result_b = apply_unitary_dilation_densitypoly_direct(
            rho,
            modes=(mode_a, mode_b),
            U=U,
            trace_out_modes=(mode_b,),
        )

        self.assertEqual(result_a, result_b)

    def test_normalize_trace_option_preserves_trace_one_for_identity_case(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)
        dilation = UnitaryDilation(
            modes=(mode,),
            U=np.array([[1.0]], dtype=np.complex128),
            trace_out_modes=(),
        )

        result = apply_unitary_dilation_densitypoly(
            rho,
            dilation=dilation,
            normalize_trace=True,
        )

        self.assertEqual(result.trace(), 1.0 + 0.0j)
