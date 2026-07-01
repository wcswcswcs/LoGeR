from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v56_promotion_repeated_masks import (
    build_v56_repeated_mask_promotion,
    write_v56_repeated_mask_promotion,
)


class V56RepeatedMaskPromotionTest(unittest.TestCase):
    def test_repeated_mask_promotion_payload_has_no_gt_prediction(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="v56_repeated_mask_promotion_test_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        payload = build_v56_repeated_mask_promotion(min_independent_chunks=99, min_co_support_masks=99)
        write_v56_repeated_mask_promotion(tmp, payload)
        summary = payload["summary"]
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertTrue(summary["uses_gt_for_diagnostic_labels"])
        self.assertEqual(summary["promoted_component_count"], 0)
        self.assertFalse(summary["gate"]["pass"])
        self.assertTrue((tmp / "promotion_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
