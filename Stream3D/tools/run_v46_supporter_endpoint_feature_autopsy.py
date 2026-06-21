from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d_native.frozen_feature_adapter import (
    FrozenFeatureAdapter,
    locate_default_dinov2_checkpoint,
    locate_default_radio_checkpoint,
)
from tools.run_v46_raw_carrier_incidence_repair import ROOT, _json_safe, _load_mask_label


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_rgb(scene: str, frame_id: int) -> np.ndarray:
    path = ROOT / "data/scannet/processed" / str(scene) / "color" / f"{int(frame_id)}.jpg"
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"missing color frame: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class _DescriptorCache:
    def __init__(self, adapter: FrozenFeatureAdapter, *, min_pixels: int) -> None:
        self.adapter = adapter
        self.min_pixels = int(min_pixels)
        self.feature_maps: dict[tuple[str, int], Any] = {}
        self.descriptors: dict[tuple[str, int, int], np.ndarray | None] = {}
        self.diag: dict[tuple[str, int, int], dict[str, Any]] = {}

    def descriptor(self, scene: str, frame_id: int, mask_id: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        key = (str(scene), int(frame_id), int(mask_id))
        if key in self.descriptors:
            return self.descriptors[key], self.diag[key]
        diag = {
            "scene": str(scene),
            "frame_id": int(frame_id),
            "mask_id": int(mask_id),
            "descriptor_available": False,
            "descriptor_pixel_count": 0,
            "missing_reason": None,
        }
        label = _load_mask_label(str(scene), int(frame_id))
        if label is None:
            diag["missing_reason"] = "missing_mask_label"
            self.descriptors[key] = None
            self.diag[key] = diag
            return None, diag
        mask = label == int(mask_id)
        pixel_count = int(mask.sum())
        diag["descriptor_pixel_count"] = pixel_count
        if pixel_count < self.min_pixels:
            diag["missing_reason"] = "too_few_mask_pixels"
            self.descriptors[key] = None
            self.diag[key] = diag
            return None, diag
        frame_key = (str(scene), int(frame_id))
        feature_map = self.feature_maps.get(frame_key)
        if feature_map is None:
            rgb = _load_rgb(str(scene), int(frame_id))
            feature_map = self.adapter.extract_dense_features(rgb)
            self.feature_maps[frame_key] = feature_map
        vec = np.asarray(self.adapter.pool_mask_feature(feature_map, mask), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if vec.size == 0 or norm <= 1e-8:
            diag["missing_reason"] = "empty_descriptor"
            self.descriptors[key] = None
            self.diag[key] = diag
            return None, diag
        vec = vec / norm
        diag["descriptor_available"] = True
        diag["descriptor_dim"] = int(vec.shape[0])
        self.descriptors[key] = vec
        self.diag[key] = diag
        return vec, diag


def _cosine01(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return None
    cosine = float(np.dot(left, right) / denom)
    return float(max(0.0, min(1.0, 0.5 * (cosine + 1.0))))


def _edge_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("edge_group"),
        _parse_int(row.get("edge_rank")),
        row.get("scene"),
        _parse_int(row.get("left_node_id")),
        _parse_int(row.get("right_node_id")),
    )


def _select_rows(rows: list[dict[str, Any]], max_observers_per_edge: int, min_q_score: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _parse_int(row.get("best_supporter_mask_id")) <= 0:
            continue
        if _parse_float(row.get("best_supporter_q_score")) < float(min_q_score):
            continue
        grouped[_edge_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: _parse_float(row.get("best_supporter_q_score")), reverse=True)
        selected.extend(group_rows[: int(max_observers_per_edge)])
    return selected


def _mean(values: list[float | None]) -> float | None:
    nums = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not nums else float(np.mean(nums))


def _quantile(values: list[float | None], q: float) -> float | None:
    nums = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not nums:
        return None
    return float(np.quantile(nums, float(q)))


def _summarize(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("edge_group") == group]
    if not selected:
        return {"edge_group": group, "row_count": 0}
    return {
        "edge_group": group,
        "row_count": len(selected),
        "support_endpoint_min_cos01_mean": _mean([row.get("support_endpoint_min_cos01") for row in selected]),
        "support_endpoint_min_cos01_p10": _quantile([row.get("support_endpoint_min_cos01") for row in selected], 0.10),
        "support_endpoint_min_cos01_p50": _quantile([row.get("support_endpoint_min_cos01") for row in selected], 0.50),
        "support_endpoint_min_cos01_p90": _quantile([row.get("support_endpoint_min_cos01") for row in selected], 0.90),
        "support_endpoint_mean_cos01_mean": _mean([row.get("support_endpoint_mean_cos01") for row in selected]),
        "support_endpoint_absdiff_cos01_mean": _mean([row.get("support_endpoint_absdiff_cos01") for row in selected]),
        "support_left_cos01_mean": _mean([row.get("support_left_cos01") for row in selected]),
        "support_right_cos01_mean": _mean([row.get("support_right_cos01") for row in selected]),
        "endpoint_pair_cos01_mean": _mean([row.get("endpoint_pair_cos01") for row in selected]),
        "best_supporter_q_score_mean": _mean([row.get("best_supporter_q_score") for row in selected]),
        "diagnostic_supporter_matches_both_rate": _mean(
            [1.0 if _parse_bool(row.get("best_supporter_gt_matches_both")) else 0.0 for row in selected]
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v46 supporter-to-endpoint frozen feature consistency autopsy.")
    parser.add_argument("--observer-rows-csv", required=True)
    parser.add_argument("--max-observers-per-edge", type=int, default=5)
    parser.add_argument("--min-q-score", type=float, default=0.0)
    parser.add_argument("--feature-backend", choices=["dinov2_timm", "radio_radseg", "rgb_stats"], default="radio_radseg")
    parser.add_argument("--feature-device", default="cuda:0")
    parser.add_argument("--feature-short-side", type=int, default=518)
    parser.add_argument("--feature-checkpoint", default="")
    parser.add_argument("--radio-lang-model", default="siglip2")
    parser.add_argument("--radio-lang-align", action="store_true")
    parser.add_argument("--radio-slide-crop", type=int, default=0)
    parser.add_argument("--radio-slide-stride", type=int, default=224)
    parser.add_argument("--descriptor-min-pixels", type=int, default=64)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    checkpoint = str(args.feature_checkpoint).strip() or None
    if checkpoint is None and str(args.feature_backend) == "radio_radseg":
        checkpoint = locate_default_radio_checkpoint()
    if checkpoint is None and str(args.feature_backend) == "dinov2_timm":
        checkpoint = locate_default_dinov2_checkpoint()
    adapter = FrozenFeatureAdapter(
        backend=str(args.feature_backend),
        device=str(args.feature_device),
        checkpoint=checkpoint,
        short_side=int(args.feature_short_side),
        radio_lang_model=str(args.radio_lang_model),
        radio_lang_align=bool(args.radio_lang_align),
        radio_slide_crop=int(args.radio_slide_crop),
        radio_slide_stride=int(args.radio_slide_stride),
    )
    cache = _DescriptorCache(adapter, min_pixels=int(args.descriptor_min_pixels))
    input_rows = _read_csv(Path(args.observer_rows_csv))
    selected_rows = _select_rows(input_rows, int(args.max_observers_per_edge), float(args.min_q_score))

    out_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        scene = str(row["scene"])
        left_frame_id = _parse_int(row.get("left_frame_id"))
        right_frame_id = _parse_int(row.get("right_frame_id"))
        observer_frame_id = _parse_int(row.get("observer_frame_id"))
        left_mask_id = _parse_int(row.get("left_mask_id"))
        right_mask_id = _parse_int(row.get("right_mask_id"))
        supporter_mask_id = _parse_int(row.get("best_supporter_mask_id"))
        left_desc, left_diag = cache.descriptor(scene, left_frame_id, left_mask_id)
        right_desc, right_diag = cache.descriptor(scene, right_frame_id, right_mask_id)
        support_desc, support_diag = cache.descriptor(scene, observer_frame_id, supporter_mask_id)
        support_left = _cosine01(support_desc, left_desc)
        support_right = _cosine01(support_desc, right_desc)
        endpoint_pair = _cosine01(left_desc, right_desc)
        pair_values = [value for value in [support_left, support_right] if value is not None]
        out = {
            **row,
            "feature_backend": str(args.feature_backend),
            "feature_checkpoint": checkpoint,
            "radio_lang_align": bool(args.radio_lang_align),
            "left_descriptor_available": left_desc is not None,
            "right_descriptor_available": right_desc is not None,
            "supporter_descriptor_available": support_desc is not None,
            "left_descriptor_pixels": left_diag.get("descriptor_pixel_count"),
            "right_descriptor_pixels": right_diag.get("descriptor_pixel_count"),
            "supporter_descriptor_pixels": support_diag.get("descriptor_pixel_count"),
            "support_left_cos01": support_left,
            "support_right_cos01": support_right,
            "support_endpoint_min_cos01": min(pair_values) if len(pair_values) == 2 else None,
            "support_endpoint_max_cos01": max(pair_values) if len(pair_values) == 2 else None,
            "support_endpoint_mean_cos01": _mean(pair_values),
            "support_endpoint_absdiff_cos01": None if len(pair_values) != 2 else float(abs(pair_values[0] - pair_values[1])),
            "endpoint_pair_cos01": endpoint_pair,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "diagnostic_only": True,
        }
        out_rows.append(out)

    summary_rows = [
        _summarize(out_rows, "top_false_positive"),
        _summarize(out_rows, "top_false_negative"),
    ]
    payload = {
        "phase": "v46_supporter_endpoint_feature_autopsy",
        "created_at": _utc_now(),
        "observer_rows_csv": str(args.observer_rows_csv),
        "max_observers_per_edge": int(args.max_observers_per_edge),
        "min_q_score": float(args.min_q_score),
        "feature_backend": str(args.feature_backend),
        "feature_device": str(args.feature_device),
        "feature_short_side": int(args.feature_short_side),
        "feature_checkpoint": checkpoint,
        "radio_lang_model": str(args.radio_lang_model),
        "radio_lang_align": bool(args.radio_lang_align),
        "radio_slide_crop": int(args.radio_slide_crop),
        "radio_slide_stride": int(args.radio_slide_stride),
        "descriptor_min_pixels": int(args.descriptor_min_pixels),
        "selected_observer_row_count": len(selected_rows),
        "feature_map_cache_frame_count": len(cache.feature_maps),
        "descriptor_cache_count": len(cache.descriptors),
        "summary_rows": summary_rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "supporter_endpoint_feature_autopsy.json", payload)
    _write_csv(out / "supporter_endpoint_feature_rows.csv", out_rows)
    _write_csv(out / "supporter_endpoint_feature_summary_rows.csv", summary_rows)
    print(json.dumps({"summary": str(out / "supporter_endpoint_feature_autopsy.json"), "summary_rows": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
