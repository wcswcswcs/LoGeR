from __future__ import annotations

import unittest

from tools.run_v45_stage2_geometry_diagnostic import ROOT


class V45Stage2BlockedContractTest(unittest.TestCase):
    def test_root_points_to_stream3d_checkout(self) -> None:
        self.assertEqual(ROOT.name, "Stream3D")


if __name__ == "__main__":
    unittest.main()

