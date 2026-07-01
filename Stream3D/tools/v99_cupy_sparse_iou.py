#!/usr/bin/env python3
"""CuPy-backed SparseSceneIoU accumulator for v99 evaluator experiments.

This keeps the public build() contract compatible with
tools.run_v65_scene_multiview_ap.SparseSceneIoU: the AP summarizer still sees
plain NumPy arrays and Python dictionaries. Only the per-frame unique/count
work in add() is moved to CUDA.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class CuPySparseSceneIoU:
    def __init__(self, device_id: int = 0) -> None:
        import cupy as cp

        self.cp = cp
        self.device_id = int(device_id)
        self.pred_area: defaultdict[int, int] = defaultdict(int)
        self.gt_area: defaultdict[int, int] = defaultdict(int)
        self.intersection: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.frame_count = 0
        self.pixel_count = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            pred_gpu = cp.asarray(pred, dtype=cp.int64)
            gt_gpu = cp.asarray(gt, dtype=cp.int64)
            self.frame_count += 1
            self.pixel_count += int(pred_gpu.size)
            pred_pos = pred_gpu > 0
            gt_pos = gt_gpu > 0
            if bool(cp.any(pred_pos).get()):
                ids, counts = cp.unique(pred_gpu[pred_pos], return_counts=True)
                for value, count in zip(cp.asnumpy(ids), cp.asnumpy(counts)):
                    self.pred_area[int(value)] += int(count)
            if bool(cp.any(gt_pos).get()):
                ids, counts = cp.unique(gt_gpu[gt_pos], return_counts=True)
                for value, count in zip(cp.asnumpy(ids), cp.asnumpy(counts)):
                    self.gt_area[int(value)] += int(count)
            both = pred_pos & gt_pos
            if bool(cp.any(both).get()):
                pred_vals = pred_gpu[both]
                gt_vals = gt_gpu[both]
                base = int(cp.max(gt_vals).get()) + 1
                encoded = pred_vals * base + gt_vals
                ids, counts = cp.unique(encoded, return_counts=True)
                for value, count in zip(cp.asnumpy(ids), cp.asnumpy(counts)):
                    self.intersection[(int(value // base), int(value % base))] += int(count)

    def build(self, *, min_pred_pixels: int, min_gt_pixels: int) -> dict[str, Any]:
        pred_ids_all = sorted(self.pred_area)
        gt_ids_all = sorted(self.gt_area)
        pred_ids = [pid for pid in pred_ids_all if self.pred_area[pid] >= int(min_pred_pixels)]
        gt_ids = [gid for gid in gt_ids_all if self.gt_area[gid] >= int(min_gt_pixels)]
        pred_index = {pid: idx for idx, pid in enumerate(pred_ids)}
        gt_index = {gid: idx for idx, gid in enumerate(gt_ids)}
        iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
        inter = np.zeros_like(iou)
        for (pid, gid), count in self.intersection.items():
            if pid not in pred_index or gid not in gt_index:
                continue
            pidx = pred_index[pid]
            gidx = gt_index[gid]
            union = int(self.pred_area[pid]) + int(self.gt_area[gid]) - int(count)
            if union > 0:
                inter[pidx, gidx] = float(count)
                iou[pidx, gidx] = float(count) / float(union)
        return {
            "pred_ids": pred_ids,
            "gt_ids": gt_ids,
            "iou": iou,
            "intersection": inter,
            "pred_area": np.asarray([self.pred_area[pid] for pid in pred_ids], dtype=np.int64),
            "gt_area": np.asarray([self.gt_area[gid] for gid in gt_ids], dtype=np.int64),
            "pred_ids_all_count": len(pred_ids_all),
            "gt_ids_all_count": len(gt_ids_all),
        }
