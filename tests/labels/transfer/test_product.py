import unittest

import numpy as np

from symop.modes.transfer.product import Product


class DummyTF:
    def __init__(self, name, fn, *, sig=None, approx_sig=None):
        self._name = name
        self._fn = fn
        self._sig = sig if sig is not None else (name,)
        self._approx_sig = approx_sig
        self.calls = []

    @property
    def signature(self):
        return self._sig

    def approx_signature(self, *, decimals=12, ignore_global_phase=False):
        self.calls.append(("approx_signature", decimals, ignore_global_phase))
        if self._approx_sig is not None:
            return self._approx_sig(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            )
        return ("approx", self._name, decimals, ignore_global_phase)

    def __call__(self, w):
        w_arr = np.asarray(w)
        self.calls.append(("__call__", w_arr))
        return self._fn(w_arr)


class TestProduct(unittest.TestCase):
    def test_signature_uses_part_signatures_in_order(self):
        a = DummyTF("a", lambda w: np.ones_like(w, dtype=complex), sig=("a_sig",))
        b = DummyTF("b", lambda w: np.ones_like(w, dtype=complex), sig=("b_sig",))

        p = Product(a=a, b=b)
        self.assertEqual(p.signature, ("product_tf", ("a_sig",), ("b_sig",)))

    def test_approx_signature_forwards_kwargs(self):
        def mk(name):
            return DummyTF(
                name,
                lambda w: np.ones_like(w, dtype=complex),
                approx_sig=lambda decimals, ignore_global_phase: (
                    "approx_sig",
                    name,
                    decimals,
                    ignore_global_phase,
                ),
            )

        a = mk("a")
        b = mk("b")
        p = Product(a=a, b=b)

        got = p.approx_signature(decimals=7, ignore_global_phase=True)
        self.assertEqual(
            got,
            (
                "product_tf_approx",
                ("approx_sig", "a", 7, True),
                ("approx_sig", "b", 7, True),
            ),
        )

        self.assertIn(("approx_signature", 7, True), a.calls)
        self.assertIn(("approx_signature", 7, True), b.calls)

    def test_call_multiplies_pointwise(self):
        a = DummyTF("a", lambda w: (2.0 + 0.0j) * np.ones_like(w, dtype=complex))
        b = DummyTF("b", lambda w: 1.0j * w.astype(float))

        p = Product(a=a, b=b)

        w = np.array([0.0, 1.0, 2.5], dtype=float)
        out = p(w)

        expected = (2.0 + 0.0j) * (1.0j * w)
        np.testing.assert_allclose(out, expected)
        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_accepts_python_sequence(self):
        a = DummyTF("a", lambda w: (3.0 + 0.0j) * np.ones_like(w, dtype=complex))
        b = DummyTF("b", lambda w: (1.0 + 2.0j) * np.ones_like(w, dtype=complex))
        p = Product(a=a, b=b)

        out = p([0.0, 1.0, 2.0])
        expected = (3.0 + 0.0j) * (1.0 + 2.0j) * np.ones(3, dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_does_not_modify_input(self):
        a = DummyTF("a", lambda w: np.ones_like(w, dtype=complex))
        b = DummyTF("b", lambda w: np.ones_like(w, dtype=complex))
        p = Product(a=a, b=b)

        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = p(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
