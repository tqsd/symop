from __future__ import annotations

import unittest


class FakeMode:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def signature(self):
        return ("mode", self._name)


class FakeLadderOp:
    def __init__(self, op_id: str, mode: FakeMode) -> None:
        self._op_id = op_id
        self.mode = mode

    @property
    def signature(self):
        return ("lop", self._op_id, self.mode.signature)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return (
            "lop_approx",
            self._op_id,
            decimals,
            ignore_global_phase,
            self.mode.signature,
        )

    def dagger(self):
        raise NotImplementedError


class FakeMonomial:
    """Minimal stand-in for MonomialProto as used by KetTerm and DensityTerm.

    Supports:
      - .signature / .approx_signature(...)
      - .adjoint()
      - .is_creator_only / .is_annihilator_only / .is_identity
      - .creators / .annihilators
      - .mode_ops
    """

    def __init__(
        self,
        *,
        name: str,
        creators: tuple[FakeLadderOp, ...] = (),
        annihilators: tuple[FakeLadderOp, ...] = (),
        mode_ops: tuple[FakeMode, ...] | None = None,
        is_creator_only: bool | None = None,
        is_annihilator_only: bool | None = None,
        is_identity: bool | None = None,
        adjoint_of: FakeMonomial | None = None,
    ) -> None:
        self.name = name
        self.creators = creators
        self.annihilators = annihilators
        self._mode_ops = (
            mode_ops
            if mode_ops is not None
            else tuple({op.mode for op in creators + annihilators})
        )
        self._is_identity = (
            is_identity
            if is_identity is not None
            else (len(creators) == 0 and len(annihilators) == 0)
        )
        self._is_creator_only = (
            is_creator_only
            if is_creator_only is not None
            else (len(creators) > 0 and len(annihilators) == 0)
        )
        self._is_annihilator_only = (
            is_annihilator_only
            if is_annihilator_only is not None
            else (len(annihilators) > 0 and len(creators) == 0)
        )
        self._adjoint_of = adjoint_of
        self.last_approx_args = None

    @property
    def mode_ops(self):
        return self._mode_ops

    @property
    def is_creator_only(self):
        return self._is_creator_only

    @property
    def is_annihilator_only(self):
        return self._is_annihilator_only

    @property
    def is_identity(self):
        return self._is_identity

    @property
    def signature(self):
        return ("mon", self.name)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        self.last_approx_args = (decimals, ignore_global_phase)
        return ("mon_approx", self.name, decimals, ignore_global_phase)

    def adjoint(self):
        if self._adjoint_of is None:
            raise RuntimeError("FakeMonomial.adjoint not configured")
        return self._adjoint_of


from symop.core.terms import DensityTerm, KetTerm  # noqa: E402


class TestKetTerm(unittest.TestCase):
    def test_identity(self):
        kt = KetTerm.identity()
        self.assertIsInstance(kt, KetTerm)
        self.assertEqual(kt.coeff, 1.0)

        # The implementation constructs Monomial() from the real module; we only
        # check the public guarantees of KetTerm around it.
        self.assertEqual(kt.signature[0], "KT")

    def test_scaled(self):
        m = FakeMonomial(name="M")
        kt = KetTerm(coeff=2.0 + 1.0j, monomial=m)

        out = kt.scaled(-0.5 + 0.0j)
        self.assertIsNot(out, kt)
        self.assertEqual(out.coeff, (2.0 + 1.0j) * (-0.5 + 0.0j))
        self.assertIs(out.monomial, m)

    def test_adjoint_conjugates_coeff_and_adjoins_monomial(self):
        m = FakeMonomial(name="M")
        m_dag = FakeMonomial(name="M_dag")
        m._adjoint_of = m_dag

        kt = KetTerm(coeff=2.0 - 3.0j, monomial=m)
        out = kt.adjoint()

        self.assertIsInstance(out, KetTerm)
        self.assertEqual(out.coeff, (2.0 - 3.0j).conjugate())
        self.assertIs(out.monomial, m_dag)

    def test_signature_uses_monomial_signature_only(self):
        m = FakeMonomial(name="M")
        kt1 = KetTerm(coeff=1.0 + 0.0j, monomial=m)
        kt2 = KetTerm(coeff=99.0 + 3.0j, monomial=m)
        self.assertEqual(kt1.signature, kt2.signature)
        self.assertEqual(kt1.signature, ("KT", m.signature))

    def test_approx_signature_forwards_parameters(self):
        m = FakeMonomial(name="M")
        kt = KetTerm(coeff=1.0, monomial=m)

        out = kt.approx_signature(decimals=7, ignore_global_phase=True)
        self.assertEqual(out[0], "KT_approx")
        self.assertEqual(out[1], ("mon_approx", "M", 7, True))
        self.assertEqual(m.last_approx_args, (7, True))

    def test_flags_and_counts_delegate_to_monomial(self):
        mode_a = FakeMode("A")
        mode_b = FakeMode("B")

        a_dag = FakeLadderOp("adag_A", mode_a)
        b_dag = FakeLadderOp("adag_B", mode_b)
        a = FakeLadderOp("a_A", mode_a)

        m = FakeMonomial(
            name="M",
            creators=(a_dag, b_dag),
            annihilators=(a,),
            mode_ops=(mode_a, mode_b),
        )
        kt = KetTerm(coeff=1.0, monomial=m)

        self.assertFalse(kt.is_creator_only)
        self.assertFalse(kt.is_annihilator_only)
        self.assertFalse(kt.is_identity)

        self.assertEqual(kt.creation_count, 2)
        self.assertEqual(kt.annihilation_count, 1)
        self.assertEqual(kt.total_degree, 3)

        self.assertEqual(kt.mode_ops, (mode_a, mode_b))


class TestDensityTerm(unittest.TestCase):
    def test_identity(self):
        dt = DensityTerm.identity()
        self.assertIsInstance(dt, DensityTerm)
        self.assertEqual(dt.coeff, 1.0)
        self.assertEqual(dt.signature[0], "DT")

    def test_scaled(self):
        left = FakeMonomial(name="L")
        right = FakeMonomial(name="R")
        dt = DensityTerm(coeff=2.0 + 0.5j, left=left, right=right)

        out = dt.scaled(3.0 - 1.0j)
        self.assertIsNot(out, dt)
        self.assertEqual(out.coeff, (2.0 + 0.5j) * (3.0 - 1.0j))
        self.assertIs(out.left, left)
        self.assertIs(out.right, right)

    def test_adjoint_conjugates_and_swaps(self):
        left = FakeMonomial(name="L")
        right = FakeMonomial(name="R")
        dt = DensityTerm(coeff=1.0 - 2.0j, left=left, right=right)

        out = dt.adjoint()
        self.assertEqual(out.coeff, (1.0 - 2.0j).conjugate())
        self.assertIs(out.left, right)
        self.assertIs(out.right, left)

    def test_signature_uses_left_and_right_signatures_only(self):
        left = FakeMonomial(name="L")
        right = FakeMonomial(name="R")
        dt1 = DensityTerm(coeff=1.0 + 0.0j, left=left, right=right)
        dt2 = DensityTerm(coeff=9.0 + 9.0j, left=left, right=right)

        self.assertEqual(dt1.signature, dt2.signature)
        self.assertEqual(
            dt1.signature, ("DT", "L", left.signature, "R", right.signature)
        )

    def test_approx_signature_forwards_parameters(self):
        left = FakeMonomial(name="L")
        right = FakeMonomial(name="R")
        dt = DensityTerm(coeff=1.0, left=left, right=right)

        out = dt.approx_signature(decimals=6, ignore_global_phase=True)
        self.assertEqual(out[0], "DT_approx")
        self.assertEqual(out[1], "L")
        self.assertEqual(out[3], "R")
        self.assertEqual(left.last_approx_args, (6, True))
        self.assertEqual(right.last_approx_args, (6, True))

    def test_creator_only_logic(self):
        # creator-only OR identity counts as creator-only on each side
        left_creator = FakeMonomial(name="Lc", is_creator_only=True, is_identity=False)
        left_identity = FakeMonomial(name="Li", is_creator_only=False, is_identity=True)
        left_mixed = FakeMonomial(name="Lm", is_creator_only=False, is_identity=False)

        right_creator = FakeMonomial(name="Rc", is_creator_only=True, is_identity=False)
        right_identity = FakeMonomial(
            name="Ri", is_creator_only=False, is_identity=True
        )
        right_mixed = FakeMonomial(name="Rm", is_creator_only=False, is_identity=False)

        self.assertTrue(DensityTerm(1.0, left_creator, right_creator).is_creator_only)
        self.assertTrue(DensityTerm(1.0, left_creator, right_identity).is_creator_only)
        self.assertTrue(DensityTerm(1.0, left_identity, right_creator).is_creator_only)
        self.assertTrue(DensityTerm(1.0, left_identity, right_identity).is_creator_only)

        self.assertFalse(DensityTerm(1.0, left_mixed, right_creator).is_creator_only)
        self.assertFalse(DensityTerm(1.0, left_creator, right_mixed).is_creator_only)

    def test_annihilator_only_logic(self):
        left_ann = FakeMonomial(name="La", is_annihilator_only=True, is_identity=False)
        left_identity = FakeMonomial(
            name="Li", is_annihilator_only=False, is_identity=True
        )
        left_mixed = FakeMonomial(
            name="Lm", is_annihilator_only=False, is_identity=False
        )

        right_ann = FakeMonomial(name="Ra", is_annihilator_only=True, is_identity=False)
        right_identity = FakeMonomial(
            name="Ri", is_annihilator_only=False, is_identity=True
        )
        right_mixed = FakeMonomial(
            name="Rm", is_annihilator_only=False, is_identity=False
        )

        self.assertTrue(DensityTerm(1.0, left_ann, right_ann).is_annihilator_only)
        self.assertTrue(DensityTerm(1.0, left_ann, right_identity).is_annihilator_only)
        self.assertTrue(DensityTerm(1.0, left_identity, right_ann).is_annihilator_only)
        self.assertTrue(
            DensityTerm(1.0, left_identity, right_identity).is_annihilator_only
        )

        self.assertFalse(DensityTerm(1.0, left_mixed, right_ann).is_annihilator_only)
        self.assertFalse(DensityTerm(1.0, left_ann, right_mixed).is_annihilator_only)

    def test_identity_and_diagonal_checks(self):
        left = FakeMonomial(name="X", is_identity=True)
        right = FakeMonomial(name="Y", is_identity=False)
        dt = DensityTerm(1.0, left, right)

        self.assertTrue(dt.is_identity_left)
        self.assertFalse(dt.is_identity_right)

        same_left = FakeMonomial(name="Z")
        same_right = FakeMonomial(name="Z")
        dt_diag = DensityTerm(1.0, same_left, same_right)
        self.assertTrue(dt_diag.is_diagonal_in_monomials)

        dt_not_diag = DensityTerm(1.0, FakeMonomial(name="A"), FakeMonomial(name="B"))
        self.assertFalse(dt_not_diag.is_diagonal_in_monomials)

    def test_counts_and_mode_ops_delegate(self):
        mode_a = FakeMode("A")
        mode_b = FakeMode("B")

        a_dag = FakeLadderOp("adag_A", mode_a)
        b_dag = FakeLadderOp("adag_B", mode_b)
        a = FakeLadderOp("a_A", mode_a)

        left = FakeMonomial(
            name="L", creators=(a_dag,), annihilators=(a,), mode_ops=(mode_a,)
        )
        right = FakeMonomial(
            name="R", creators=(b_dag,), annihilators=(), mode_ops=(mode_b,)
        )

        dt = DensityTerm(1.0, left, right)

        self.assertEqual(dt.creation_count_left, 1)
        self.assertEqual(dt.annihilation_count_left, 1)
        self.assertEqual(dt.creation_count_right, 1)
        self.assertEqual(dt.annihilation_count_right, 0)

        self.assertEqual(dt.mode_ops_left, (mode_a,))
        self.assertEqual(dt.mode_ops_right, (mode_b,))


if __name__ == "__main__":
    unittest.main()
