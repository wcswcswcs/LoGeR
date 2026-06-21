from __future__ import annotations

import unittest
from collections import Counter

from stream4d_native.v55_material_atoms import _dominant


class V55MaterialAtomsTest(unittest.TestCase):
    def test_dominant_reports_weighted_purity(self) -> None:
        label, purity, total = _dominant(Counter({"a": 9, "b": 1}))
        self.assertEqual(label, "a")
        self.assertEqual(total, 10)
        self.assertAlmostEqual(purity or 0.0, 0.9)

    def test_dominant_handles_empty_counter(self) -> None:
        label, purity, total = _dominant(Counter())
        self.assertIsNone(label)
        self.assertIsNone(purity)
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
