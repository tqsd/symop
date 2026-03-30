import unittest

from symop.modes.labels.path import Path


class TestPath(unittest.TestCase):
    def test_name_is_stored(self) -> None:
        path = Path("A")
        self.assertEqual(path.name, "A")

    def test_overlap_with_same_name_is_one(self) -> None:
        left = Path("A")
        right = Path("A")

        self.assertEqual(left.overlap(right), 1.0 + 0.0j)

    def test_overlap_with_different_name_is_zero(self) -> None:
        left = Path("A")
        right = Path("B")

        self.assertEqual(left.overlap(right), 0.0 + 0.0j)

    def test_signature_is_stable(self) -> None:
        path = Path("A")
        self.assertEqual(path.signature, ("path", "A"))

    def test_approx_signature_matches_signature(self) -> None:
        path = Path("A")

        self.assertEqual(
            path.approx_signature(),
            path.signature,
        )
        self.assertEqual(
            path.approx_signature(decimals=4),
            path.signature,
        )
        self.assertEqual(
            path.approx_signature(ignore_global_phase=True),
            path.signature,
        )

    def test_equal_paths_compare_equal(self) -> None:
        self.assertEqual(Path("A"), Path("A"))

    def test_different_paths_compare_unequal(self) -> None:
        self.assertNotEqual(Path("A"), Path("B"))

    def test_equal_paths_have_equal_hash(self) -> None:
        self.assertEqual(hash(Path("A")), hash(Path("A")))

    def test_different_paths_can_be_used_in_a_set(self) -> None:
        paths = {Path("A"), Path("B"), Path("A")}
        self.assertEqual(paths, {Path("A"), Path("B")})
