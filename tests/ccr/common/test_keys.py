from __future__ import annotations

import unittest

from symop.ccr.common.keys import sig_lop, sig_obj, sig_word


class _DummySig:
    def __init__(self, signature):
        self.signature = signature
        self.last_call = None

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        self.last_call = {
            "decimals": decimals,
            "ignore_global_phase": ignore_global_phase,
        }
        return ("approx", self.signature, decimals, ignore_global_phase)


class TestKeys(unittest.TestCase):
    def test_sig_obj_exact(self):
        x = _DummySig(("x", 1))
        got = sig_obj(x)
        self.assertEqual(got, ("x", 1))
        self.assertIsNone(x.last_call)

    def test_sig_obj_approx_forwards_strict_kwargs(self):
        x = _DummySig(("x", 2))
        got = sig_obj(x, approx=True, decimals=7, ignore_global_phase=True)
        self.assertEqual(got, ("approx", ("x", 2), 7, True))
        self.assertEqual(x.last_call, {"decimals": 7, "ignore_global_phase": True})

    def test_sig_lop_aliases_sig_obj(self):
        op = _DummySig(("lop", "a"))
        got = sig_lop(op, approx=True, decimals=3, ignore_global_phase=False)
        self.assertEqual(got, ("approx", ("lop", "a"), 3, False))
        self.assertEqual(op.last_call, {"decimals": 3, "ignore_global_phase": False})

    def test_sig_word_exact_materializes_and_preserves_order(self):
        a = _DummySig(("a", 1))
        b = _DummySig(("b", 2))
        got = sig_word([a, b])
        self.assertEqual(got, (("a", 1), ("b", 2)))
        self.assertIsNone(a.last_call)
        self.assertIsNone(b.last_call)

    def test_sig_word_approx_forwards_to_each_operator(self):
        a = _DummySig(("a", 1))
        b = _DummySig(("b", 2))
        got = sig_word(
            (x for x in [a, b]),
            approx=True,
            decimals=9,
            ignore_global_phase=True,
        )
        self.assertEqual(
            got,
            (
                ("approx", ("a", 1), 9, True),
                ("approx", ("b", 2), 9, True),
            ),
        )
        self.assertEqual(a.last_call, {"decimals": 9, "ignore_global_phase": True})
        self.assertEqual(b.last_call, {"decimals": 9, "ignore_global_phase": True})


if __name__ == "__main__":
    unittest.main()
