from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v56_core_update_carrier_overlap import (
    build_v56_carrier_overlap_core_update,
    write_v56_carrier_overlap_core_update,
)


class V56CarrierOverlapCoreUpdateTest(unittest.TestCase):
    def test_carrier_overlap_payload_has_no_gt_prediction(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="v56_carrier_overlap_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        payload = build_v56_carrier_overlap_core_update(
            component_min_shared_carrier_count=10**9,
            component_min_carrier_overlap_ratio=1.0,
            objectlet_min_component_count=10**9,
            objectlet_min_total_shared_carriers=10**9,
        )
        write_v56_carrier_overlap_core_update(tmp, payload)
        summary = payload["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
        self.assertEqual(summary["confirmed_added_component_count"], 0)
        self.assertFalse(summary["gate"]["pass"])
        self.assertTrue((tmp / "core_update_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
