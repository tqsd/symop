import unittest
from dataclasses import dataclass

from symop.ccr.algebra.op import OpTerm


@dataclass(frozen=True)
class DummyLadderOp:
    name: str
    is_creation: bool = False

    @property
    def signature(self):
        return ("lop", self.name, self.is_creation)

    def approx_signature(self, *, decimals=12, ignore_global_phase=False):
        return (
            "lop_approx",
            self.name,
            self.is_creation,
            decimals,
            ignore_global_phase,
        )

    def dagger(self):
        return DummyLadderOp(self.name, not self.is_creation)


class TestOpTerm(unittest.TestCase):
    def test_identity(self):
        t = OpTerm.identity(2.5)

        self.assertEqual(t.ops, ())
        self.assertEqual(t.coeff, 2.5)

    def test_scaled(self):
        op = DummyLadderOp("a")
        t = OpTerm((op,), 3.0)

        out = t.scaled(2.0)

        self.assertEqual(out.ops, (op,))
        self.assertEqual(out.coeff, 6.0)

    def test_adjoint_reverses_and_daggers(self):
        a = DummyLadderOp("a", is_creation=False)
        b = DummyLadderOp("b", is_creation=True)

        t = OpTerm((a, b), 1.0 + 2.0j)
        adj = t.adjoint()

        self.assertEqual(adj.coeff, (1.0 + 2.0j).conjugate())
        self.assertEqual(adj.ops, (b.dagger(), a.dagger()))

    def test_signature(self):
        a = DummyLadderOp("a", is_creation=False)
        b = DummyLadderOp("b", is_creation=True)

        t = OpTerm((a, b), 1.0)

        self.assertEqual(
            t.signature,
            ("op_term", (a.signature, b.signature)),
        )

    def test_approx_signature_forwards_params(self):
        a = DummyLadderOp("a", is_creation=False)
        b = DummyLadderOp("b", is_creation=True)

        t = OpTerm((a, b), 1.0)

        sig = t.approx_signature(decimals=7, ignore_global_phase=True)

        self.assertEqual(
            sig,
            (
                "op_term_approx",
                (
                    a.approx_signature(decimals=7, ignore_global_phase=True),
                    b.approx_signature(decimals=7, ignore_global_phase=True),
                ),
            ),
        )
