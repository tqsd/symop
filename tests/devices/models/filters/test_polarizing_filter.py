from __future__ import annotations

import unittest
from dataclasses import dataclass

from symop.devices.models.filters.polarizing_filter import PolarizingFilter
from symop.devices.types.device_kind import DeviceKind
from tests.core.support.fakes import FakeComponentLabel, make_mode
from tests.devices.support.fakes import FakeEditableState, FakePath, FakeState


@dataclass(frozen=True)
class FakePolarization:
    name: str
    overlap_map: dict[str, complex] | None = None

    @property
    def signature(self) -> tuple[str, str]:
        return ("fake_polarization", self.name)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> tuple[str, str, int, bool]:
        return ("fake_polarization_approx", self.name, decimals, ignore_global_phase)

    def overlap(self, other: object) -> complex:
        if not isinstance(other, FakePolarization):
            return 0.0 + 0.0j
        if self.overlap_map is not None and other.name in self.overlap_map:
            return self.overlap_map[other.name]
        return 1.0 + 0.0j if self.name == other.name else 0.0 + 0.0j


class TestPolarizingFilter(unittest.TestCase):
    def test_kind(self) -> None:
        device = PolarizingFilter(passed_polarization=FakePolarization("H"))
        self.assertEqual(device.kind, DeviceKind.POLARIZING_FILTER)

    def test_port_specs(self) -> None:
        device = PolarizingFilter(passed_polarization=FakePolarization("H"))

        self.assertEqual(len(device.port_specs), 2)
        self.assertEqual(device.port_specs[0].name, "in")
        self.assertEqual(device.port_specs[1].name, "out")

    def test_plan_requires_label_editable_state(self) -> None:
        device = PolarizingFilter(passed_polarization=FakePolarization("H"))

        with self.assertRaises(TypeError):
            device.plan(
                state=FakeState(),
                ports={"in": FakePath("in0"), "out": FakePath("out0")},
                selection=None,
                ctx=None,
            )

    def test_plan_computes_eta_and_label_edit(self) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        pol_in = FakePolarization("D")
        pol_out = FakePolarization("H", overlap_map={"D": 0.6 + 0.0j})

        mode = make_mode(path="in0", polarization="unused", envelope="env")
        mode = mode.with_label(mode.label.with_polarization(pol_in))

        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )

        device = PolarizingFilter(passed_polarization=pol_out)

        action = device.plan(
            state=state,
            ports={"in": in_path, "out": out_path},
            selection=None,
            ctx=None,
        )

        self.assertEqual(action.kind, DeviceKind.POLARIZING_FILTER)
        self.assertEqual(len(action.edits), 1)

        eta_by_mode = action.params["eta_by_mode"]
        self.assertAlmostEqual(eta_by_mode[mode.signature], 0.36)

        edit = action.edits[0]
        self.assertEqual(edit.mode_sig, mode.signature)
        self.assertEqual(edit.label.path, out_path)
        self.assertEqual(edit.label.polarization, pol_out)

    def test_apply_forces_density_out_kind(self) -> None:
        class RecordingRuntime:
            def __init__(self) -> None:
                self.calls = []

            def apply(
                self,
                *,
                device: object,
                state: object,
                ports: object,
                selection: object,
                ctx: object,
                out_kind: object,
            ) -> object:
                self.calls.append({"out_kind": out_kind})
                return "ok"

        rt = RecordingRuntime()
        device = PolarizingFilter(passed_polarization=FakePolarization("H"))

        result = device.apply(
            FakeState(),
            ports={"in": FakePath("in0"), "out": FakePath("out0")},
            runtime=rt,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(rt.calls[0]["out_kind"], "density")
