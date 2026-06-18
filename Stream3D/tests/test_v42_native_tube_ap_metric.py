from __future__ import annotations

import unittest

from tools.run_v42_native_tube_ap_metric import (
    _average_precision,
    _prediction_attribution_rows,
    _score_prediction,
    _score_predictions_at_threshold,
    _tube_iou,
)


class V42NativeTubeAPMetricTests(unittest.TestCase):
    def test_tube_iou_uses_set_union(self) -> None:
        self.assertEqual(_tube_iou({1, 2}, {1, 2}), 1.0)
        self.assertAlmostEqual(_tube_iou({1, 2}, {2, 3}), 1.0 / 3.0)
        self.assertEqual(_tube_iou({1}, {2}), 0.0)

    def test_score_prediction_supports_prediction_side_tube_count_modes(self) -> None:
        base = _score_prediction(0.5, 9, "confidence")
        weighted = _score_prediction(0.5, 9, "confidence_log_tube_count")
        self.assertEqual(base, 0.5)
        self.assertGreater(weighted, base)

    def test_average_precision_integrates_precision_recall_envelope(self) -> None:
        self.assertEqual(_average_precision([1, 1], [0, 0], 2), 1.0)
        self.assertAlmostEqual(_average_precision([0, 1], [1, 0], 1), 0.5)
        self.assertIsNone(_average_precision([1], [0], 0))

    def test_score_predictions_matches_one_gt_once(self) -> None:
        predictions = [
            {"scene": "s0", "object_id": 0, "score": 0.9, "tube_ids": {1, 2, 3}},
            {"scene": "s0", "object_id": 1, "score": 0.8, "tube_ids": {1, 2, 3}},
            {"scene": "s1", "object_id": 0, "score": 0.7, "tube_ids": {4, 5}},
        ]
        gt_sets = {
            "s0": [{"gt_id": 10, "tube_ids": {1, 2, 3}}],
            "s1": [{"gt_id": 20, "tube_ids": {4, 6}}],
        }
        result_50 = _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.50)
        self.assertEqual(result_50["true_positive_count"], 1)
        self.assertEqual(result_50["false_positive_count"], 2)
        self.assertAlmostEqual(result_50["AP"], 0.5)

        result_25 = _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.25)
        self.assertEqual(result_25["true_positive_count"], 2)
        self.assertEqual(result_25["false_positive_count"], 1)
        self.assertAlmostEqual(result_25["AP"], 5.0 / 6.0)

    def test_prediction_attribution_separates_unlabeled_and_overmix(self) -> None:
        predictions = [
            {"scene": "s0", "variant": "Q5", "source": "dino", "object_id": 0, "score": 0.9, "tube_ids": {1, 2, 3, 4}},
            {"scene": "s0", "variant": "Q5", "source": "dino", "object_id": 1, "score": 0.8, "tube_ids": {5, 6, 7, 8}},
        ]
        gt_labels = {"s0": {1: 10, 2: 11, 3: 12, 4: 12, 5: 0, 6: 0, 7: 10, 8: 0}}
        gt_sets = {
            "s0": [
                {"gt_id": 10, "tube_ids": {1, 9}},
                {"gt_id": 11, "tube_ids": {2, 10}},
                {"gt_id": 12, "tube_ids": {3, 4, 11, 12}},
            ]
        }
        match = _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.50)
        rows = _prediction_attribution_rows(predictions, gt_labels, gt_sets, match["match_rows"])
        by_id = {int(row["object_id"]): row for row in rows}
        self.assertEqual(by_id[0]["fp_category"], "multi_gt_overmix")
        self.assertEqual(by_id[1]["fp_category"], "unlabeled_overmix")


if __name__ == "__main__":
    unittest.main()
