from __future__ import annotations

import unittest

from symop.devices.measurement.resolution import MeasurementResolution


class TestMeasurementResolution(unittest.TestCase):
    def test_default_resolution_is_valid(self) -> None:
        resolution = MeasurementResolution()

        self.assertEqual(resolution.axes, ("path",))
        self.assertEqual(resolution.grouping, "total")
        self.assertEqual(resolution.readout, "number")

    def test_empty_axes_raises(self) -> None:
        with self.assertRaises(ValueError):
            MeasurementResolution(axes=())

    def test_duplicate_axes_raise(self) -> None:
        with self.assertRaises(ValueError):
            MeasurementResolution(axes=("path", "path"))
