from __future__ import annotations

import unittest

from symop.ccr.common.signatures import sig_density, sig_ket, sig_mono


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


class TestSignatures(unittest.TestCase):
    def test_sig_mono_exact(self):
        obj = _DummySig(("mono", 1))
        got = sig_mono(obj)
        self.assertEqual(got, ("mono", 1))
        self.assertIsNone(obj.last_call)

    def test_sig_mono_approx_forwards_strict_kwargs(self):
        obj = _DummySig(("mono", 2))
        got = sig_mono(obj, approx=True, decimals=7, ignore_global_phase=True)
        self.assertEqual(got, ("approx", ("mono", 2), 7, True))
        self.assertEqual(obj.last_call, {"decimals": 7, "ignore_global_phase": True})

    def test_sig_ket_exact(self):
        obj = _DummySig(("ket", 3))
        got = sig_ket(obj)
        self.assertEqual(got, ("ket", 3))
        self.assertIsNone(obj.last_call)

    def test_sig_ket_approx_defaults(self):
        obj = _DummySig(("ket", 4))
        got = sig_ket(obj, approx=True)
        self.assertEqual(got, ("approx", ("ket", 4), 12, False))
        self.assertEqual(obj.last_call, {"decimals": 12, "ignore_global_phase": False})

    def test_sig_density_exact(self):
        obj = _DummySig(("rho", 5))
        got = sig_density(obj)
        self.assertEqual(got, ("rho", 5))
        self.assertIsNone(obj.last_call)

    def test_sig_density_approx_custom(self):
        obj = _DummySig(("rho", 6))
        got = sig_density(obj, approx=True, decimals=3, ignore_global_phase=False)
        self.assertEqual(got, ("approx", ("rho", 6), 3, False))
        self.assertEqual(obj.last_call, {"decimals": 3, "ignore_global_phase": False})


if __name__ == "__main__":
    unittest.main()
