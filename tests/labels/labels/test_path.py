import unittest
from dataclasses import FrozenInstanceError

from symop.modes.labels.path import PathLabel


class TestPathLabel(unittest.TestCase):
    def test_overlap_same_name_is_one(self):
        a = PathLabel(name="a")
        b = PathLabel(name="a")
        self.assertEqual(a.overlap(b), 1.0 + 0.0j)
        self.assertEqual(b.overlap(a), 1.0 + 0.0j)

    def test_overlap_different_name_is_zero(self):
        a = PathLabel(name="a")
        b = PathLabel(name="b")
        self.assertEqual(a.overlap(b), 0.0 + 0.0j)
        self.assertEqual(b.overlap(a), 0.0 + 0.0j)

    def test_overlap_is_orthonormal_like(self):
        labels = [
            PathLabel(name="p0"),
            PathLabel(name="p1"),
            PathLabel(name="p2"),
        ]
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                expected = 1.0 + 0.0j if i == j else 0.0 + 0.0j
                self.assertEqual(li.overlap(lj), expected)

    def test_signature(self):
        a = PathLabel(name="arm_1")
        self.assertEqual(a.signature, ("path", "arm_1"))

    def test_approx_signature_equals_signature(self):
        a = PathLabel(name="x")
        self.assertEqual(
            a.approx_signature(decimals=3, ignore_global_phase=True),
            a.signature,
        )
        self.assertEqual(
            a.approx_signature(decimals=12, ignore_global_phase=False),
            a.signature,
        )

    def test_frozen_dataclass(self):
        a = PathLabel(name="x")
        with self.assertRaises(FrozenInstanceError):
            a.name = "y"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
