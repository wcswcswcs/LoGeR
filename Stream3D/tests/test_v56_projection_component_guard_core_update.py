from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v56_core_update_projection_component_guard import (
    build_v56_projection_component_guard_core_update,
    write_v56_projection_component_guard_core_update,
)


class V56ProjectionComponentGuardCoreUpdateTest(unittest.TestCase):
    def test_projection_component_guard_payload_has_no_gt_prediction(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="v56_projection_component_guard_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        payload = build_v56_projection_component_guard_core_update(
            boundary_component_min_support=10**9,
            boundary_component_min_ratio=1.0,
            uv_component_min_support=10**9,
            uv_component_min_ratio=1.0,
        )
        write_v56_projection_component_guard_core_update(tmp, payload)
        summary = payload["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
        self.assertEqual(summary["confirmed_added_component_count"], 0)
        self.assertFalse(summary["gate"]["pass"])
        self.assertTrue((tmp / "core_update_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
