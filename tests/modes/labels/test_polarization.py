import math
import unittest
import numpy as np

from symop.modes.labels.polarization import Polarization


class TestPolarization(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 12,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def assertJonesAlmostEqual(
        self,
        left: tuple[complex, complex],
        right: tuple[complex, complex],
        places: int = 12,
    ) -> None:
        self.assertComplexAlmostEqual(left[0], right[0], places=places)
        self.assertComplexAlmostEqual(left[1], right[1], places=places)

    def test_zero_vector_raises(self) -> None:
        with self.assertRaises(ValueError):
            Polarization((0.0 + 0.0j, 0.0 + 0.0j))

    def test_constructor_normalizes_vector(self) -> None:
        pol = Polarization((2.0 + 0.0j, 0.0 + 0.0j))
        self.assertJonesAlmostEqual(pol.jones, (1.0 + 0.0j, 0.0 + 0.0j))

    def test_global_phase_is_removed(self) -> None:
        pol = Polarization((1.0j, 0.0 + 0.0j))
        self.assertJonesAlmostEqual(pol.jones, (1.0 + 0.0j, 0.0 + 0.0j))

    def test_equal_up_to_global_phase_compare_equal(self) -> None:
        left = Polarization((1.0 + 0.0j, 1.0j))
        right = Polarization((1.0j, -1.0 + 0.0j))
        self.assertEqual(left, right)

    def test_self_overlap_is_one(self) -> None:
        pol = Polarization.D()
        self.assertComplexAlmostEqual(pol.overlap(pol), 1.0 + 0.0j)

    def test_overlap_is_conjugate_symmetric(self) -> None:
        left = Polarization.R()
        right = Polarization.D()

        overlap_lr = left.overlap(right)
        overlap_rl = right.overlap(left)

        self.assertComplexAlmostEqual(overlap_lr, overlap_rl.conjugate())

    def test_h_and_v_are_orthogonal(self) -> None:
        self.assertComplexAlmostEqual(
            Polarization.H().overlap(Polarization.V()),
            0.0 + 0.0j,
        )

    def test_d_and_a_are_orthogonal(self) -> None:
        self.assertComplexAlmostEqual(
            Polarization.D().overlap(Polarization.A()),
            0.0 + 0.0j,
        )

    def test_r_and_l_are_orthogonal(self) -> None:
        self.assertComplexAlmostEqual(
            Polarization.R().overlap(Polarization.L()),
            0.0 + 0.0j,
        )

    def test_h_constructor(self) -> None:
        self.assertJonesAlmostEqual(
            Polarization.H().jones,
            (1.0 + 0.0j, 0.0 + 0.0j),
        )

    def test_v_constructor(self) -> None:
        self.assertJonesAlmostEqual(
            Polarization.V().jones,
            (0.0 + 0.0j, 1.0 + 0.0j),
        )

    def test_d_constructor(self) -> None:
        s = 2**-0.5
        self.assertJonesAlmostEqual(
            Polarization.D().jones,
            (s + 0.0j, s + 0.0j),
        )

    def test_a_constructor(self) -> None:
        s = 2**-0.5
        self.assertJonesAlmostEqual(
            Polarization.A().jones,
            (s + 0.0j, -s + 0.0j),
        )

    def test_r_constructor(self) -> None:
        s = 2**-0.5
        self.assertJonesAlmostEqual(
            Polarization.R().jones,
            (s + 0.0j, -1j * s),
        )

    def test_l_constructor(self) -> None:
        s = 2**-0.5
        self.assertJonesAlmostEqual(
            Polarization.L().jones,
            (s + 0.0j, 1j * s),
        )

    def test_linear_zero_is_horizontal(self) -> None:
        self.assertEqual(
            Polarization.linear(0.0),
            Polarization.H(),
        )

    def test_linear_pi_over_two_is_vertical(self) -> None:
        self.assertEqual(
            Polarization.linear(math.pi / 2.0),
            Polarization.V(),
        )

    def test_rotation_preserves_normalization(self) -> None:
        pol = Polarization.R().rotated(math.pi / 7.0)
        self.assertComplexAlmostEqual(pol.overlap(pol), 1.0 + 0.0j)

    def test_rotating_horizontal_by_pi_over_two_gives_vertical(self) -> None:
        rotated = Polarization.H().rotated(math.pi / 2.0)
        self.assertEqual(rotated, Polarization.V())

    def test_linear_theta_matches_rotated_horizontal(self) -> None:
        theta = math.pi / 5.0
        self.assertEqual(
            Polarization.linear(theta),
            Polarization.H().rotated(theta),
        )

    def test_signature_is_stable(self) -> None:
        pol = Polarization.H()
        self.assertEqual(
            pol.signature,
            ("pol", 1.0, 0.0, 0.0, 0.0),
        )

    def test_approx_signature_rounds_components(self) -> None:
        pol = Polarization(
            (1.0 + 0.0j, 1e-13 + 1e-13j),
        )
        sig = pol.approx_signature(decimals=12)

        self.assertEqual(sig[0], "pol_approx")
        self.assertEqual(sig[1:], (1.0, 0.0, 0.0, 0.0))

    def test_transformed_rejects_non_2x2_matrix(self) -> None:
        pol = Polarization.H()
        with self.assertRaises(ValueError):
            pol.transformed(np.eye(3, dtype=complex))

    def test_transformed_rejects_non_unitary_matrix(self) -> None:
        pol = Polarization.H()
        bad = np.array(
            [
                [1.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j],
            ],
            dtype=complex,
        )
        with self.assertRaises(ValueError):
            pol.transformed(bad)

    def test_identity_unitary_leaves_state_unchanged(self) -> None:
        pol = Polarization.D()
        out = pol.transformed(np.eye(2, dtype=complex))
        self.assertEqual(out, pol)

    def test_hadamard_maps_h_to_d(self) -> None:
        s = 2**-0.5
        hadamard = np.array(
            [
                [s + 0.0j, s + 0.0j],
                [s + 0.0j, -s + 0.0j],
            ],
            dtype=complex,
        )
        out = Polarization.H().transformed(hadamard)
        self.assertEqual(out, Polarization.D())

    def test_phase_only_difference_is_canonicalized(self) -> None:
        phase = np.exp(1j * math.pi / 3.0)
        unitary = np.array(
            [
                [phase, 0.0 + 0.0j],
                [0.0 + 0.0j, phase],
            ],
            dtype=complex,
        )
        out = Polarization.R().transformed(unitary)
        expected = Polarization.R()
        self.assertJonesAlmostEqual(out.jones, expected.jones)

    def test_transformed_preserves_norm(self) -> None:
        s = 2**-0.5
        unitary = np.array(
            [
                [s + 0.0j, -s + 0.0j],
                [s + 0.0j, s + 0.0j],
            ],
            dtype=complex,
        )
        out = Polarization.R().transformed(unitary)
        self.assertComplexAlmostEqual(out.overlap(out), 1.0 + 0.0j)

    def test_rotated_matches_transformed_with_rotation_matrix(self) -> None:
        theta = math.pi / 5.0
        c = math.cos(theta)
        s = math.sin(theta)
        rotation = np.array(
            [
                [c + 0.0j, -s + 0.0j],
                [s + 0.0j, c + 0.0j],
            ],
            dtype=complex,
        )

        left = Polarization.H().rotated(theta)
        right = Polarization.H().transformed(rotation)
        self.assertEqual(left, right)
