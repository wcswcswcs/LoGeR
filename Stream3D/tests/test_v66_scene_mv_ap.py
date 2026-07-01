from __future__ import annotations

import unittest

import numpy as np

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _summarize_iou


class V66SceneMultiViewAPTests(unittest.TestCase):
    def _summary(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        acc = SparseSceneIoU()
        acc.add(pred.astype(np.int64), gt.astype(np.int64))
        summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="constant",
            input_scores=None,
        )
        return summary

    def test_all_background_prediction_with_gt_is_zero_not_crash(self) -> None:
        pred = np.zeros((8, 8), dtype=np.int64)
        gt = np.zeros((8, 8), dtype=np.int64)
        gt[:4, :4] = 1
        summary = self._summary(pred, gt)
        self.assertEqual(summary["ap"], 0.0)
        self.assertEqual(summary["ap50"], 0.0)
        self.assertEqual(summary["evaluated_pred_count"], 0)
        self.assertEqual(summary["evaluated_gt_count"], 1)
        self.assertEqual(summary["gt_best_iou_mean"], 0.0)

    def test_perfect_prediction_is_one(self) -> None:
        gt = np.zeros((8, 8), dtype=np.int64)
        gt[:4, :4] = 1
        gt[4:, 4:] = 2
        summary = self._summary(gt.copy(), gt)
        self.assertEqual(summary["ap"], 1.0)
        self.assertEqual(summary["ap50"], 1.0)
        self.assertEqual(summary["score_free_match_at_050"]["tp"], 2)


if __name__ == "__main__":
    unittest.main()

