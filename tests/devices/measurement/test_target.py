from __future__ import annotations

import unittest

from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget
from symop.core.types.signature import Signature

from tests.devices.support.fakes import FakePath


class TestMeasurementTarget(unittest.TestCase):
    def test_selection_deduplicates_paths_and_mode_sigs(self) -> None:
        path = FakePath("p0")
        sig = Signature(("mode", "a"))

        selection = MeasurementSelection(
            port_name="in",
            paths=(path, path),
            mode_sigs=(sig, sig),
        )

        self.assertEqual(selection.paths, (path,))
        self.assertEqual(selection.mode_sigs, (sig,))
        self.assertFalse(selection.is_empty)
        self.assertTrue(selection.has_paths)
        self.assertTrue(selection.has_mode_sigs)

    def test_target_rejects_duplicate_port_names(self) -> None:
        path0 = FakePath("p0")
        path1 = FakePath("p1")

        with self.assertRaises(ValueError):
            MeasurementTarget(
                selections=(
                    MeasurementSelection(port_name="in", paths=(path0,)),
                    MeasurementSelection(port_name="in", paths=(path1,)),
                )
            )

    def test_target_flattens_unique_paths_and_mode_sigs(self) -> None:
        path0 = FakePath("p0")
        path1 = FakePath("p1")
        sig0 = Signature(("mode", "a"))
        sig1 = Signature(("mode", "b"))

        target = MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="in0",
                    paths=(path0, path1),
                    mode_sigs=(sig0,),
                ),
                MeasurementSelection(
                    port_name="in1",
                    paths=(path1,),
                    mode_sigs=(sig1, sig0),
                ),
            )
        )

        self.assertEqual(target.paths, (path0, path1))
        self.assertEqual(target.mode_sigs, (sig0, sig1))
        self.assertTrue(target.has_paths)
        self.assertTrue(target.has_mode_sigs)
        self.assertFalse(target.is_empty)

    def test_empty_target_reports_empty(self) -> None:
        target = MeasurementTarget(
            selections=(
                MeasurementSelection(port_name="in"),
                MeasurementSelection(port_name="aux"),
            )
        )

        self.assertTrue(target.is_empty)
        self.assertFalse(target.has_paths)
        self.assertFalse(target.has_mode_sigs)
