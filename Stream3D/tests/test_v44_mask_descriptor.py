from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import auc_score, cluster_metrics


class V44MaskDescriptorTest(unittest.TestCase):
    def test_auc_handles_ranked_descriptor_signal(self) -> None:
        auc = auc_score([True, True, False, False], [0.9, 0.8, 0.2, 0.1])
        self.assertEqual(auc, 1.0)

    def test_cluster_metrics_reports_identity_quality(self) -> None:
        metrics = cluster_metrics({1: 10, 2: 10, 3: 20}, {1: 1, 2: 1, 3: 2})
        self.assertAlmostEqual(metrics["purity"], 1.0)
        self.assertAlmostEqual(metrics["completeness"], 1.0)


if __name__ == "__main__":
    unittest.main()
