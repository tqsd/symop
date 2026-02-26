import unittest

import numpy as np

from symop.modes.transfer.cascade import Cascade


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
        self.calls.append(("__call__", np.asarray(w)))
        return self._fn(np.asarray(w))


class TestCascade(unittest.TestCase):
    def test_signature_includes_all_parts_in_order(self):
        p1 = DummyTF("p1", lambda w: np.ones_like(w, dtype=complex), sig=("p1_sig",))
        p2 = DummyTF("p2", lambda w: np.ones_like(w, dtype=complex), sig=("p2_sig",))

        c = Cascade(parts=(p1, p2))
        self.assertEqual(c.signature, ("cascade", (("p1_sig",), ("p2_sig",))))

    def test_approx_signature_forwards_kwargs_and_preserves_order(self):
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

        p1 = mk("p1")
        p2 = mk("p2")

        c = Cascade(parts=(p1, p2))
        got = c.approx_signature(decimals=7, ignore_global_phase=True)

        self.assertEqual(
            got,
            (
                "cascade_approx",
                (
                    ("approx_sig", "p1", 7, True),
                    ("approx_sig", "p2", 7, True),
                ),
            ),
        )

        self.assertIn(("approx_signature", 7, True), p1.calls)
        self.assertIn(("approx_signature", 7, True), p2.calls)

    def test_call_multiplies_pointwise(self):
        # H(w) = (2 + 0j) * (1j * w)
        p1 = DummyTF("p1", lambda w: (2.0 + 0.0j) * np.ones_like(w, dtype=complex))
        p2 = DummyTF("p2", lambda w: 1.0j * w.astype(float))

        c = Cascade(parts=(p1, p2))

        w = np.array([0.0, 1.0, 2.5], dtype=float)
        out = c(w)

        expected = (2.0 + 0.0j) * (1.0j * w)
        np.testing.assert_allclose(out, expected)

    def test_call_accepts_python_sequences(self):
        p1 = DummyTF("p1", lambda w: (3.0 + 0.0j) * np.ones_like(w, dtype=complex))
        p2 = DummyTF("p2", lambda w: (1.0 + 2.0j) * np.ones_like(w, dtype=complex))
        c = Cascade(parts=(p1, p2))

        out = c([0.0, 1.0, 2.0])
        expected = (3.0 + 0.0j) * (1.0 + 2.0j) * np.ones(3, dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.iscomplexobj(out))

    def test_empty_parts_returns_ones_like(self):
        c = Cascade(parts=())

        w = np.array([10.0, 20.0], dtype=float)
        out = c(w)

        np.testing.assert_allclose(out, np.ones_like(w, dtype=complex))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_does_not_modify_input(self):
        p1 = DummyTF("p1", lambda w: np.ones_like(w, dtype=complex) * (2.0 + 0.0j))
        c = Cascade(parts=(p1,))

        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()
        _ = c(w)

        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
