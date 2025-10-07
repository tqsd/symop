from symop_proto.labels.path_label import PathLabel
from tests.utils.case import ExtendedTestCase


class TestPathLabel(ExtendedTestCase):
    def test_initiation(self):
        pl = PathLabel("A")
        self.assertEqual(pl.name, "A")

    def test_overlap(self):
        a = PathLabel("A")
        b = PathLabel("B")
        self.assertComplexAlmostEqual(a.overlap(a), 1.0 + 0j)
        self.assertComplexAlmostEqual(a.overlap(b), 0.0 + 0j)
        self.assertComplexAlmostEqual(b.overlap(a), 0.0 + 0j)
        self.assertComplexAlmostEqual(b.overlap(b), 1.0 + 0j)

    def test_signatures_roundtrip(self):
        a = PathLabel("A")
        sig = a.signature()
        approx = a.approx_signature()
        self.assertIsInstance(sig, tuple)
        self.assertIsInstance(approx, tuple)
