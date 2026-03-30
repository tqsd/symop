import unittest

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import Path
from symop.modes.labels.polarization import Polarization


class StubEnvelope:
    def __init__(self, signature, overlap_value: complex) -> None:
        self.signature = signature
        self.overlap_value = overlap_value
        self.calls = 0

    def overlap(self, other: "StubEnvelope") -> complex:
        self.calls += 1
        return self.overlap_value

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return ("stub_env_approx", self.signature, decimals, ignore_global_phase)


class StubPath:
    def __init__(self, signature, overlap_value: complex) -> None:
        self.signature = signature
        self.overlap_value = overlap_value
        self.calls = 0

    def overlap(self, other: "StubPath") -> complex:
        self.calls += 1
        return self.overlap_value

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return ("stub_path_approx", self.signature, decimals, ignore_global_phase)


class StubPolarization:
    def __init__(self, signature, overlap_value: complex) -> None:
        self.signature = signature
        self.overlap_value = overlap_value
        self.calls = 0

    def overlap(self, other: "StubPolarization") -> complex:
        self.calls += 1
        return self.overlap_value

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return ("stub_pol_approx", self.signature, decimals, ignore_global_phase)


class TestModeLabel(unittest.TestCase):
    def test_overlap_multiplies_component_overlaps(self) -> None:
        left = ModeLabel(
            path=StubPath(("path", "A"), 1.0 + 0.0j),
            polarization=StubPolarization(("pol", "H"), 0.5 + 0.0j),
            envelope=StubEnvelope(("env", "E1"), 0.25 + 0.0j),
        )
        right = ModeLabel(
            path=StubPath(("path", "A"), 1.0 + 0.0j),
            polarization=StubPolarization(("pol", "H"), 0.5 + 0.0j),
            envelope=StubEnvelope(("env", "E1"), 0.25 + 0.0j),
        )

        overlap = left.overlap(right)

        self.assertEqual(overlap, 0.125 + 0.0j)

    def test_overlap_short_circuits_when_path_overlap_is_zero(self) -> None:
        left_path = StubPath(("path", "A"), 0.0 + 0.0j)
        right_path = StubPath(("path", "B"), 0.0 + 0.0j)

        left_pol = StubPolarization(("pol", "H"), 1.0 + 0.0j)
        right_pol = StubPolarization(("pol", "H"), 1.0 + 0.0j)

        left_env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)
        right_env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)

        left = ModeLabel(
            path=left_path,
            polarization=left_pol,
            envelope=left_env,
        )
        right = ModeLabel(
            path=right_path,
            polarization=right_pol,
            envelope=right_env,
        )

        overlap = left.overlap(right)

        self.assertEqual(overlap, 0.0 + 0.0j)
        self.assertEqual(left_path.calls, 1)
        self.assertEqual(left_pol.calls, 0)
        self.assertEqual(left_env.calls, 0)

    def test_overlap_short_circuits_when_polarization_overlap_is_zero(
        self,
    ) -> None:
        left_path = StubPath(("path", "A"), 1.0 + 0.0j)
        right_path = StubPath(("path", "A"), 1.0 + 0.0j)

        left_pol = StubPolarization(("pol", "H"), 0.0 + 0.0j)
        right_pol = StubPolarization(("pol", "V"), 0.0 + 0.0j)

        left_env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)
        right_env = StubEnvelope(("env", "E2"), 1.0 + 0.0j)

        left = ModeLabel(
            path=left_path,
            polarization=left_pol,
            envelope=left_env,
        )
        right = ModeLabel(
            path=right_path,
            polarization=right_pol,
            envelope=right_env,
        )

        overlap = left.overlap(right)

        self.assertEqual(overlap, 0.0 + 0.0j)
        self.assertEqual(left_path.calls, 1)
        self.assertEqual(left_pol.calls, 1)
        self.assertEqual(left_env.calls, 0)

    def test_overlap_evaluates_envelope_when_other_factors_are_nonzero(
        self,
    ) -> None:
        left_path = StubPath(("path", "A"), 1.0 + 0.0j)
        right_path = StubPath(("path", "A"), 1.0 + 0.0j)

        left_pol = StubPolarization(("pol", "H"), 0.5 + 0.0j)
        right_pol = StubPolarization(("pol", "D"), 0.5 + 0.0j)

        left_env = StubEnvelope(("env", "E1"), 0.25 + 0.0j)
        right_env = StubEnvelope(("env", "E2"), 0.25 + 0.0j)

        left = ModeLabel(
            path=left_path,
            polarization=left_pol,
            envelope=left_env,
        )
        right = ModeLabel(
            path=right_path,
            polarization=right_pol,
            envelope=right_env,
        )

        overlap = left.overlap(right)

        self.assertEqual(overlap, 0.125 + 0.0j)
        self.assertEqual(left_path.calls, 1)
        self.assertEqual(left_pol.calls, 1)
        self.assertEqual(left_env.calls, 1)

    def test_with_path_returns_updated_copy(self) -> None:
        label = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=StubEnvelope(("env", "E1"), 1.0 + 0.0j),
        )

        updated = label.with_path(Path("B"))

        self.assertEqual(updated.path, Path("B"))
        self.assertEqual(updated.polarization, label.polarization)
        self.assertEqual(updated.envelope, label.envelope)
        self.assertEqual(label.path, Path("A"))
        self.assertIsNot(updated, label)

    def test_with_polarization_returns_updated_copy(self) -> None:
        label = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=StubEnvelope(("env", "E1"), 1.0 + 0.0j),
        )

        updated = label.with_polarization(Polarization.V())

        self.assertEqual(updated.path, label.path)
        self.assertEqual(updated.polarization, Polarization.V())
        self.assertEqual(updated.envelope, label.envelope)
        self.assertEqual(label.polarization, Polarization.H())
        self.assertIsNot(updated, label)

    def test_with_envelope_returns_updated_copy(self) -> None:
        env1 = StubEnvelope(("env", "E1"), 1.0 + 0.0j)
        env2 = StubEnvelope(("env", "E2"), 0.5 + 0.0j)

        label = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env1,
        )

        updated = label.with_envelope(env2)

        self.assertEqual(updated.path, label.path)
        self.assertEqual(updated.polarization, label.polarization)
        self.assertEqual(updated.envelope, env2)
        self.assertEqual(label.envelope, env1)
        self.assertIsNot(updated, label)

    def test_signature_contains_component_signatures(self) -> None:
        path = Path("A")
        pol = Polarization.H()
        env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)

        label = ModeLabel(
            path=path,
            polarization=pol,
            envelope=env,
        )

        self.assertEqual(
            label.signature,
            (
                "mode_label",
                path.signature,
                pol.signature,
                env.signature,
            ),
        )

    def test_approx_signature_forwards_parameters_to_components(self) -> None:
        path = StubPath(("path", "A"), 1.0 + 0.0j)
        pol = StubPolarization(("pol", "H"), 1.0 + 0.0j)
        env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)

        label = ModeLabel(
            path=path,
            polarization=pol,
            envelope=env,
        )

        signature = label.approx_signature(
            decimals=7,
            ignore_global_phase=True,
        )

        self.assertEqual(
            signature,
            (
                "mode_label_approx",
                ("stub_path_approx", ("path", "A"), 7, True),
                ("stub_pol_approx", ("pol", "H"), 7, True),
                ("stub_env_approx", ("env", "E1"), 7, True),
            ),
        )

    def test_equal_mode_labels_compare_equal(self) -> None:
        env = StubEnvelope(("env", "E1"), 1.0 + 0.0j)

        left = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env,
        )
        right = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env,
        )

        self.assertEqual(left, right)

    def test_distinct_mode_labels_compare_unequal(self) -> None:
        env1 = StubEnvelope(("env", "E1"), 1.0 + 0.0j)
        env2 = StubEnvelope(("env", "E2"), 1.0 + 0.0j)

        left = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env1,
        )
        right = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env2,
        )

        self.assertNotEqual(left, right)

    def test_overlap_with_real_components_uses_real_implementations(self) -> None:
        env_left = StubEnvelope(("env", "E1"), 0.5 + 0.0j)
        env_right = StubEnvelope(("env", "E1"), 0.5 + 0.0j)

        left = ModeLabel(
            path=Path("A"),
            polarization=Polarization.H(),
            envelope=env_left,
        )
        right = ModeLabel(
            path=Path("A"),
            polarization=Polarization.D(),
            envelope=env_right,
        )

        overlap = left.overlap(right)

        expected = (1.0 + 0.0j) * Polarization.H().overlap(Polarization.D()) * (
            0.5 + 0.0j
        )
        self.assertEqual(overlap, expected)
