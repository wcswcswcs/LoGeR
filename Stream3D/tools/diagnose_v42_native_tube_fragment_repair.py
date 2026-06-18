from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v42_native_tube_ap_metric import _parse_json_int_list, _score_predictions_at_threshold


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        current = float(value)
    except (TypeError, ValueError):
        return float(default)
    return current if np.isfinite(current) else float(default)


def _load_fields(path: Path, *, variant: str, source: str) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for row in _read_csv(path):
        if str(row.get("variant", "")) != str(variant):
            continue
        if str(row.get("source", "")) != str(source):
            continue
        fields.append(
            {
                "scene": str(row["scene"]),
                "object_id": int(row["object_id"]),
                "tokens": set(_parse_json_int_list(row.get("semantic_masklet_ids", "[]"))),
                "tubes": set(_parse_json_int_list(row.get("attached_tube_ids", "[]"))),
                "confidence": _safe_float(row.get("confidence", 0.0)),
            }
        )
    return fields


def _load_gt_sets(path: Path) -> dict[str, list[dict[str, Any]]]:
    gt_sets: dict[str, list[dict[str, Any]]] = {}
    for row in _read_csv(path):
        gt_sets.setdefault(str(row["scene"]), []).append(
            {
                "gt_id": int(row["gt_id"]),
                "tube_ids": set(_parse_json_int_list(row.get("tube_ids", "[]"))),
            }
        )
    return gt_sets


def _score(field: dict[str, Any], mode: str) -> float:
    confidence = float(field["confidence"])
    tube_count = len(field["tubes"])
    if mode == "confidence":
        return confidence
    if mode == "confidence_log_tube_count":
        return confidence * math.log1p(tube_count)
    if mode == "confidence_sqrt_tube_count":
        return confidence * math.sqrt(tube_count)
    if mode == "confidence_tube_count":
        return confidence * tube_count
    raise ValueError(f"unsupported score mode: {mode}")


def _evaluate(fields: list[dict[str, Any]], gt_sets: dict[str, list[dict[str, Any]]], *, score_mode: str) -> dict[str, Any]:
    predictions = [
        {
            "scene": field["scene"],
            "object_id": int(field["object_id"]),
            "score": _score(field, score_mode),
            "tube_ids": set(field["tubes"]),
        }
        for field in fields
        if field["tubes"]
    ]
    ap25 = _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.25)
    ap50 = _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.50)
    threshold_rows = [
        _score_predictions_at_threshold(predictions, gt_sets, iou_threshold=0.50 + 0.05 * idx)
        for idx in range(10)
    ]
    ap_values = [row["AP"] for row in threshold_rows if row.get("AP") is not None]
    return {
        "prediction_count": int(len(predictions)),
        "native_tube_AP": float(np.mean(np.asarray(ap_values, dtype=np.float64))) if ap_values else None,
        "native_tube_AP50": ap50.get("AP"),
        "native_tube_AP25": ap25.get("AP"),
        "true_positive_count_at_50": ap50.get("true_positive_count"),
        "false_positive_count_at_50": ap50.get("false_positive_count"),
        "score_mode": score_mode,
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.merge_count = 0

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        self.parent[root_right] = root_left
        self.merge_count += 1


def _merge_fields(
    fields: list[dict[str, Any]],
    alignment_rows: list[dict[str, str]],
    *,
    variant: str,
    source: str,
    semantic_threshold: float,
    object_threshold: float,
    residual_max: float,
    visible_outside_conflict_max: float,
    max_small_tube_count: int,
    ignore_role_conflict: bool,
    ignore_same_frame_cannot_link: bool,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    total_merges = 0
    for scene in sorted({str(field["scene"]) for field in fields}):
        scene_fields = [deepcopy(field) for field in fields if str(field["scene"]) == scene]
        token_to_index = {
            int(token_id): int(index)
            for index, field in enumerate(scene_fields)
            for token_id in field["tokens"]
        }
        uf = _UnionFind(len(scene_fields))
        for row in alignment_rows:
            if str(row.get("scene", "")) != scene:
                continue
            if str(row.get("variant", "")) != str(variant):
                continue
            if str(row.get("source", "")) != str(source):
                continue
            left = token_to_index.get(int(row["token_i"]))
            right = token_to_index.get(int(row["token_j"]))
            if left is None or right is None or left == right:
                continue
            if not ignore_same_frame_cannot_link and _as_bool(row.get("same_frame_cannot_link", "")):
                continue
            if not ignore_role_conflict and _as_bool(row.get("role_conflict", "")):
                continue
            if _safe_float(row.get("semantic_affinity")) < float(semantic_threshold):
                continue
            if _safe_float(row.get("object_affinity")) < float(object_threshold):
                continue
            if _safe_float(row.get("residual_proxy")) > float(residual_max):
                continue
            if _safe_float(row.get("visible_outside_conflict_ratio")) > float(visible_outside_conflict_max):
                continue
            if not (
                len(scene_fields[left]["tubes"]) <= int(max_small_tube_count)
                or len(scene_fields[right]["tubes"]) <= int(max_small_tube_count)
            ):
                continue
            uf.union(left, right)
        total_merges += uf.merge_count
        groups: dict[int, list[dict[str, Any]]] = {}
        for index, field in enumerate(scene_fields):
            groups.setdefault(uf.find(index), []).append(field)
        for group in groups.values():
            token_count = sum(max(1, len(field["tokens"])) for field in group)
            confidence = sum(float(field["confidence"]) * max(1, len(field["tokens"])) for field in group) / max(
                token_count,
                1,
            )
            output.append(
                {
                    "scene": scene,
                    "object_id": int(len(output)),
                    "tokens": set().union(*(field["tokens"] for field in group)),
                    "tubes": set().union(*(field["tubes"] for field in group)),
                    "confidence": confidence,
                }
            )
    return output, int(total_merges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v42 native tube fragment score/merge repair options.")
    parser.add_argument(
        "--memory-object-rows",
        default="outputs/audit/v42_streaming_memory_unioncap320_allframe_r1/memory_object_field_rows.csv",
    )
    parser.add_argument(
        "--alignment-rows",
        default="outputs/audit/v42_part_gated_alignment_dino_q5_stride1_r3/alignment_rows.csv",
    )
    parser.add_argument(
        "--gt-rows",
        default="outputs/audit/v42_native_tube_ap_metric_allframe_r1/native_tube_ap_gt_rows.csv",
    )
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--source", default="dinov2_maskcut")
    parser.add_argument("--output-root", default="outputs/audit/v42_native_tube_fragment_repair_probe_r1")
    args = parser.parse_args()

    fields = _load_fields(ROOT / str(args.memory_object_rows), variant=str(args.variant), source=str(args.source))
    alignment_rows = _read_csv(ROOT / str(args.alignment_rows))
    gt_sets = _load_gt_sets(ROOT / str(args.gt_rows))
    output_root = ROOT / str(args.output_root)

    rows: list[dict[str, Any]] = []
    for mode in [
        "confidence",
        "confidence_log_tube_count",
        "confidence_sqrt_tube_count",
        "confidence_tube_count",
    ]:
        metric = _evaluate(fields, gt_sets, score_mode=mode)
        rows.append({"probe": "score_only", "merge_count": 0, **metric})

    for min_count in [2, 4, 8, 16, 32, 64]:
        filtered = [field for field in fields if len(field["tubes"]) >= min_count]
        metric = _evaluate(filtered, gt_sets, score_mode="confidence_log_tube_count")
        rows.append({"probe": "min_tube_count_filter", "min_tube_count": min_count, "merge_count": 0, **metric})

    for semantic_threshold in [0.45, 0.55, 0.65, 0.75, 0.85]:
        for object_threshold in [0.05, 0.10, 0.20, 0.30, 0.40]:
            for residual_max in [0.10, 0.20, 0.35]:
                for visible_outside_conflict_max in [0.35, 0.70, 1.00]:
                    for max_small_tube_count in [2, 4, 8, 16]:
                        for ignore_role_conflict in [False, True]:
                            merged, merge_count = _merge_fields(
                                fields,
                                alignment_rows,
                                variant=str(args.variant),
                                source=str(args.source),
                                semantic_threshold=semantic_threshold,
                                object_threshold=object_threshold,
                                residual_max=residual_max,
                                visible_outside_conflict_max=visible_outside_conflict_max,
                                max_small_tube_count=max_small_tube_count,
                                ignore_role_conflict=ignore_role_conflict,
                                ignore_same_frame_cannot_link=False,
                            )
                            if merge_count <= 0:
                                continue
                            metric = _evaluate(merged, gt_sets, score_mode="confidence_log_tube_count")
                            rows.append(
                                {
                                    "probe": "alignment_fragment_merge",
                                    "merge_count": merge_count,
                                    "semantic_threshold": semantic_threshold,
                                    "object_threshold": object_threshold,
                                    "residual_max": residual_max,
                                    "visible_outside_conflict_max": visible_outside_conflict_max,
                                    "max_small_tube_count": max_small_tube_count,
                                    "ignore_role_conflict": bool(ignore_role_conflict),
                                    **metric,
                                }
                            )

    best_by_ap50 = sorted(
        rows,
        key=lambda row: (
            -_safe_float(row.get("native_tube_AP50"), default=-1.0),
            -_safe_float(row.get("native_tube_AP25"), default=-1.0),
        ),
    )
    summary = {
        "phase": "v42_native_tube_fragment_repair_probe",
        "status": "OK_FRAGMENT_REPAIR_PROBE_COMPUTED",
        "memory_object_rows": str(ROOT / str(args.memory_object_rows)),
        "alignment_rows": str(ROOT / str(args.alignment_rows)),
        "gt_rows": str(ROOT / str(args.gt_rows)),
        "variant": str(args.variant),
        "source": str(args.source),
        "input_object_count": int(len(fields)),
        "sweep_row_count": int(len(rows)),
        "best_by_native_tube_AP50": best_by_ap50[:10],
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "conclusion": [
            "Tube-count-aware score improves ranking but not TP count.",
            "Alignment-based fragment merge does not improve AP50 in this sweep.",
            "Remaining repair must improve object formation beyond simple score or affinity merge.",
        ],
    }
    _write_csv(output_root / "fragment_repair_sweep_rows.csv", rows)
    _write_json(output_root / "fragment_repair_summary.json", summary)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "status": summary["status"],
                "input_object_count": summary["input_object_count"],
                "sweep_row_count": summary["sweep_row_count"],
                "best_native_tube_AP50": best_by_ap50[0].get("native_tube_AP50") if best_by_ap50 else None,
                "best_probe": best_by_ap50[0].get("probe") if best_by_ap50 else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
