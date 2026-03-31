from __future__ import annotations

import unittest

from symop.devices.models.phase_shifters.phase_shifter import PhaseShifter
from symop.devices.types.device_kind import DeviceKind
from tests.devices.support.fakes import FakePath, FakeState


class TestPhaseShifter(unittest.TestCase):
    def test_kind(self) -> None:
        device = PhaseShifter(phi=0.5)
        self.assertEqual(device.kind, DeviceKind.PHASE_SHIFTER)

    def test_port_specs(self) -> None:
        device = PhaseShifter(phi=0.5)

        self.assertEqual(len(device.port_specs), 1)
        self.assertEqual(device.port_specs[0].name, "path")
        self.assertEqual(device.port_specs[0].direction, "inout")
        self.assertTrue(device.port_specs[0].required)

    def test_plan_returns_expected_action(self) -> None:
        device = PhaseShifter(phi=1.25)
        path = FakePath("p0")

        action = device.plan(
            state=FakeState(),
            ports={"path": path},
            selection=None,
            ctx=None,
        )

        self.assertEqual(action.kind, DeviceKind.PHASE_SHIFTER)
        self.assertEqual(action.ports["path"], path)
        self.assertEqual(action.params["path"], path)
        self.assertEqual(action.params["phi"], 1.25)
        self.assertEqual(tuple(action.edits), ())
        self.assertTrue(action.requires_kernel)
        self.assertIsNone(action.selection)
