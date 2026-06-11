from __future__ import annotations

import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("open3d") is None, "open3d is not installed in this environment")
class ExportScanNetOpen3DTests(unittest.TestCase):
    def test_open3d_optional_dependency_is_available(self) -> None:
        import open3d as o3d

        self.assertIsNotNone(o3d)


if __name__ == "__main__":
    unittest.main()
