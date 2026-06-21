from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import build_component_lattice


class TestV49ComponentLattice(unittest.TestCase):
    def test_lattice_reports_required_scales_without_gt_prediction(self) -> None:
        payload = build_component_lattice(scales=[24, 32])
        scales = {row["scale"] for row in payload["scale_rows"]}
        self.assertEqual(scales, {24, 32})
        self.assertFalse(payload["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
