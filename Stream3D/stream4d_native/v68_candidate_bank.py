from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import (  # noqa: E402
    _oracle_majority_mapping,
    _representative_pairs_by_chunk,
    _row_from_eval,
    _summarize_variant_all,
)
from stream4d_native.v67_local_mask_graph import _mask_summary_by_pair, _support_by_pair  # noqa: E402
from stream4d_native.v67_mask_universe import _colorize_labels, _frame_mask_stats  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png, _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import (  # noqa: E402
    _chunk_rows,
    _float_or_none,
    _frame_data,
    _load_csv_rows,
    _mean,
    _parse_bool,
    _parse_mask_observation_id,
    _rel,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)


def _area_bin(area_ratio: float) -> str:
    if area_ratio < 0.0025:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.12:
        return "medium"
    if area_ratio < 0.30:
        return "large"
    return "xlarge"


def _bbox(mask: np.ndarray, mask_id: int) -> tuple[list[int], list[float], list[float], float, float]:
    ys, xs = np.nonzero(mask == int(mask_id))
    if ys.size == 0:
        return [0, 0, 0, 0], [0.0, 0.0], [0.0, 0.0], 0.0, 0.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = mask.shape
    bw = float((x1 - x0) / max(1, w))
    bh = float((y1 - y0) / max(1, h))
    center = [float((x0 + x1) * 0.5 / max(1, w)), float((y0 + y1) * 0.5 / max(1, h))]
    size = [bw, bh]
    aspect = float(bw / max(1e-9, bh))
    return [x0, y0, x1, y1], center, size, aspect, float((x1 - x0) * (y1 - y0) / max(1, h * w))


def _boundary_length(mask: np.ndarray, mask_id: int) -> int:
    binary = (mask == int(mask_id)).astype(np.uint8)
    if not np.any(binary):
        return 0
    contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return int(sum(cv2.arcLength(contour, True) for contour in contours))


def _same_frame_bbox_competitors(items: list[dict[str, Any]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    boxes = {int(item["mask_id"]): item["bbox"] for item in items}
    for item in items:
        mid = int(item["mask_id"])
        x0, y0, x1, y1 = [float(v) for v in boxes[mid]]
        competitors: list[int] = []
        for other_id, box in boxes.items():
            if other_id == mid:
                continue
            ox0, oy0, ox1, oy1 = [float(v) for v in box]
            ix0 = max(x0, ox0)
            iy0 = max(y0, oy0)
            ix1 = min(x1, ox1)
            iy1 = min(y1, oy1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area = max(1.0, (x1 - x0) * (y1 - y0))
            if inter / area >= 0.10:
                competitors.append(other_id)
        out[mid] = sorted(competitors)[:16]
    return out


def _selected_pairs_by_chunk(pipeline_root: Path, scene: str) -> dict[str, set[tuple[int, int]]]:
    summary = json.loads((pipeline_root / "local_objectlets/local_objectlet_summary.json").read_text(encoding="utf-8"))
    variant = str(summary.get("best_real_variant") or summary.get("best_real_row", {}).get("variant") or "")
    selected_candidate_ids_by_chunk: dict[str, set[str]] = defaultdict(set)
    for row in _load_csv_rows(pipeline_root / "local_objectlets/objectlet_rows.csv"):
        if row.get("scene") == scene and row.get("variant") == variant:
            candidate_id = str(row.get("candidate_id") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            if candidate_id and chunk_id:
                selected_candidate_ids_by_chunk[chunk_id].add(candidate_id)
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in _load_csv_rows(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv"):
        if row.get("scene") != scene or not _parse_bool(row.get("reprojection_success")):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id not in selected_candidate_ids_by_chunk.get(chunk_id, set()):
            continue
        parsed = _parse_mask_observation_id(row.get("best_mask_observation_id"))
        if parsed is not None and parsed[0] == scene:
            out[chunk_id].add((int(parsed[1]), int(parsed[2])))
    return out


def _signature_id(row: dict[str, Any]) -> str:
    center = row["bbox_center"]
    cx_bin = int(float(center[0]) * 4.0)
    cy_bin = int(float(center[1]) * 4.0)
    aspect = float(row["aspect_ratio"])
    aspect_bin = "wide" if aspect > 1.5 else "tall" if aspect < 0.67 else "square"
    return f"{row['area_bin']}|{cx_bin}|{cy_bin}|{aspect_bin}"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _build_chunk_candidates(
    *,
    scene: str,
    chunk_id: str,
    frame_data: list[dict[str, Any]],
    representative_pairs: set[tuple[int, int]],
    selected_pairs: set[tuple[int, int]],
    support: dict[tuple[int, int], set[str]],
    mask_summary: dict[tuple[int, int], dict[str, Any]],
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[int, int]]]]:
    rows: list[dict[str, Any]] = []
    by_frame_raw: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in frame_data:
        frame_id = int(item["frame_id"])
        mask = item["mask"]
        gt = item["gt"]
        if mask is None:
            continue
        stats_by_mask = _frame_mask_stats(mask, gt)
        for mask_id, stats in stats_by_mask.items():
            mask_id_i = int(mask_id)
            key = (frame_id, mask_id_i)
            bbox, center, size, aspect, bbox_area_ratio = _bbox(mask, mask_id_i)
            area_ratio = float(stats["mask_area_ratio"])
            comps = sorted(support.get(key, set()))
            summary = mask_summary.get(key, {})
            component_entropy = float(summary.get("raw_component_entropy") or 0.0)
            component_count = len(comps)
            underseg = bool(stats.get("underseg_risk")) or (component_count >= 64 and component_entropy > 1.5)
            large = area_ratio > float(max_area_ratio)
            small = area_ratio < float(min_area_ratio)
            row = {
                "scene_id": scene,
                "chunk_id": chunk_id,
                "frame_id": frame_id,
                "mask_id": mask_id_i,
                "mask_observation_id": f"{scene}:{frame_id}:{mask_id_i}",
                "area_ratio": area_ratio,
                "area_bin": _area_bin(area_ratio),
                "bbox": bbox,
                "bbox_center": center,
                "bbox_size": size,
                "bbox_area_ratio": bbox_area_ratio,
                "aspect_ratio": aspect,
                "mask_boundary_length": _boundary_length(mask, mask_id_i),
                "mask_solidity_proxy": float(area_ratio / max(1e-9, bbox_area_ratio)),
                "raw_cropformer_available": True,
                "representative_available": key in representative_pairs,
                "current_selected_available": key in selected_pairs,
                "d4rt_component_count": component_count,
                "d4rt_component_ids": comps,
                "d4rt_component_entropy": component_entropy,
                "semantic_feature_id": "",
                "semantic_mode_id": "",
                "semantic_confidence": "",
                "semantic_backend_status": "unavailable_no_reliable_semantic_feature_table",
                "repeated_signature_id": "",
                "underseg_risk": underseg,
                "large_mask_risk": large,
                "small_mask_risk": small,
                "same_frame_overlap_count": 0,
                "same_frame_competing_masks": [],
                "source_types": [],
                "shared_support_only": False,
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": False,
                "diagnostic_only": True,
            }
            row["repeated_signature_id"] = _signature_id(row)
            by_frame_raw[frame_id].append(row)
            rows.append(row)
    for frame_id, frame_rows in by_frame_raw.items():
        competitors = _same_frame_bbox_competitors(frame_rows)
        for row in frame_rows:
            comp = competitors.get(int(row["mask_id"]), [])
            row["same_frame_overlap_count"] = int(len(comp))
            row["same_frame_competing_masks"] = comp
    signature_counts = Counter(str(row["repeated_signature_id"]) for row in rows)
    variants: dict[str, set[tuple[int, int]]] = {
        "CB0_current_selected_only": set(),
        "CB1_representative_only": set(),
        "CB2_representative_plus_high_quality_raw": set(),
        "CB3_CB2_plus_D4RT_supported_raw": set(),
        "CB4_CB3_plus_repeated_signature_masks": set(),
        "CB5_CB4_underseg_as_shared_support": set(),
    }
    for row in rows:
        key = (int(row["frame_id"]), int(row["mask_id"]))
        source_types: list[str] = ["raw"]
        if row["current_selected_available"]:
            variants["CB0_current_selected_only"].add(key)
            source_types.append("current_selected")
        if row["representative_available"]:
            variants["CB1_representative_only"].add(key)
            variants["CB2_representative_plus_high_quality_raw"].add(key)
            variants["CB3_CB2_plus_D4RT_supported_raw"].add(key)
            variants["CB4_CB3_plus_repeated_signature_masks"].add(key)
            variants["CB5_CB4_underseg_as_shared_support"].add(key)
            source_types.append("representative")
        high_quality = (not row["small_mask_risk"]) and (not row["large_mask_risk"]) and (not row["underseg_risk"])
        if high_quality:
            variants["CB2_representative_plus_high_quality_raw"].add(key)
            variants["CB3_CB2_plus_D4RT_supported_raw"].add(key)
            variants["CB4_CB3_plus_repeated_signature_masks"].add(key)
            variants["CB5_CB4_underseg_as_shared_support"].add(key)
            source_types.append("high_quality_raw")
        if int(row["d4rt_component_count"]) > 0 and (not row["small_mask_risk"]):
            variants["CB3_CB2_plus_D4RT_supported_raw"].add(key)
            variants["CB4_CB3_plus_repeated_signature_masks"].add(key)
            variants["CB5_CB4_underseg_as_shared_support"].add(key)
            source_types.append("d4rt_supported_raw")
        if signature_counts[str(row["repeated_signature_id"])] >= 2 and (not row["small_mask_risk"]):
            variants["CB4_CB3_plus_repeated_signature_masks"].add(key)
            variants["CB5_CB4_underseg_as_shared_support"].add(key)
            source_types.append("repeated_signature")
        if row["underseg_risk"] and not row["small_mask_risk"]:
            variants["CB5_CB4_underseg_as_shared_support"].add(key)
            row["shared_support_only"] = True
            source_types.append("underseg_shared_support")
        row["source_types"] = sorted(set(source_types))
    return rows, variants


def _candidate_diag(rows: list[dict[str, Any]], allowed: set[tuple[int, int]], frame_count: int) -> dict[str, Any]:
    selected = [row for row in rows if (int(row["frame_id"]), int(row["mask_id"])) in allowed]
    if not selected:
        return {
            "candidate_count": 0,
            "candidate_count_per_frame_mean": 0.0,
            "underseg_rate": 0.0,
            "large_mask_rate": 0.0,
            "small_mask_rate": 0.0,
            "D4RT_supported_candidate_rate": 0.0,
            "semantic_feature_success_rate": 0.0,
            "repeated_signature_count": 0,
        }
    return {
        "candidate_count": int(len(selected)),
        "candidate_count_per_frame_mean": float(len(selected) / max(1, frame_count)),
        "underseg_rate": float(sum(1 for row in selected if row["underseg_risk"]) / max(1, len(selected))),
        "large_mask_rate": float(sum(1 for row in selected if row["large_mask_risk"]) / max(1, len(selected))),
        "small_mask_rate": float(sum(1 for row in selected if row["small_mask_risk"]) / max(1, len(selected))),
        "D4RT_supported_candidate_rate": float(sum(1 for row in selected if int(row["d4rt_component_count"]) > 0) / max(1, len(selected))),
        "semantic_feature_success_rate": 0.0,
        "repeated_signature_count": int(len({row["repeated_signature_id"] for row in selected if row["repeated_signature_id"]})),
    }


def _summarize_candidate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "candidate_count_mean": _mean([_float_or_none(row.get("candidate_count")) for row in subset]),
            "candidate_count_per_chunk_mean": _mean([_float_or_none(row.get("candidate_count")) for row in subset]),
            "candidate_count_per_frame_mean": _mean([_float_or_none(row.get("candidate_count_per_frame_mean")) for row in subset]),
            "underseg_rate_mean": _mean([_float_or_none(row.get("underseg_rate")) for row in subset]),
            "large_mask_rate_mean": _mean([_float_or_none(row.get("large_mask_rate")) for row in subset]),
            "small_mask_rate_mean": _mean([_float_or_none(row.get("small_mask_rate")) for row in subset]),
            "D4RT_supported_candidate_rate_mean": _mean([_float_or_none(row.get("D4RT_supported_candidate_rate")) for row in subset]),
            "semantic_feature_success_rate_mean": _mean([_float_or_none(row.get("semantic_feature_success_rate")) for row in subset]),
            "repeated_signature_count_mean": _mean([_float_or_none(row.get("repeated_signature_count")) for row in subset]),
        }
    )
    return base


def _write_visual_samples(output_root: Path, scene: str, frame_data: list[dict[str, Any]], variants: dict[str, set[tuple[int, int]]], max_images: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in frame_data[:max_images]:
        frame_id = int(item["frame_id"])
        gt = item["gt"]
        mask = item["mask"]
        if mask is None:
            continue
        panels = [_colorize_labels(gt), _colorize_labels(mask)]
        for name in ["CB1_representative_only", "CB5_CB4_underseg_as_shared_support"]:
            pred = np.zeros(mask.shape, dtype=np.int64)
            allowed = variants.get(name, set())
            for mask_id in np.unique(mask):
                mask_id_i = int(mask_id)
                if mask_id_i > 0 and (frame_id, mask_id_i) in allowed:
                    pred[mask == mask_id_i] = mask_id_i
            panels.append(_colorize_labels(pred))
        panel = np.concatenate(panels, axis=1)
        path = output_root / f"{scene}_frame{frame_id:06d}_gt_raw_cb1_cb5.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        rows.append({"scene_id": scene, "frame_id": frame_id, "visualization_path": _rel(path)})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    visual_root = Path(args.visual_root)
    if not visual_root.is_absolute():
        visual_root = ROOT / visual_root
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    visualization_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v68-candidate-bank] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        representative_by_chunk = _representative_pairs_by_chunk(pipeline_root, scene)
        selected_by_chunk = _selected_pairs_by_chunk(pipeline_root, scene)
        support = _support_by_pair(pipeline_root, scene, str(args.support_variant))
        mask_summary = _mask_summary_by_pair(pipeline_root, scene)
        wrote_visual_for_scene = False
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            rows, variants = _build_chunk_candidates(
                scene=scene,
                chunk_id=chunk_id,
                frame_data=frame_data,
                representative_pairs=set(representative_by_chunk.get(chunk_id, set())),
                selected_pairs=set(selected_by_chunk.get(chunk_id, set())),
                support=support,
                mask_summary=mask_summary,
                min_area_ratio=float(args.min_area_ratio),
                max_area_ratio=float(args.max_area_ratio),
            )
            for row in rows:
                out_row = dict(row)
                for key in ["bbox", "bbox_center", "bbox_size", "d4rt_component_ids", "same_frame_competing_masks", "source_types"]:
                    out_row[key] = _json(out_row[key])
                candidate_rows.append(out_row)
            if not wrote_visual_for_scene:
                visualization_rows.extend(_write_visual_samples(visual_root, scene, frame_data[:3], variants, int(args.visual_frames_per_scene)))
                wrote_visual_for_scene = True
            for variant, allowed in sorted(variants.items()):
                mapping, oracle_diag = _oracle_majority_mapping(frame_data=frame_data, allowed_pairs=allowed)
                diag = _candidate_diag(rows, allowed, len(frame_ids))
                eval_row = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    raw_per_frame_masks=False,
                    diag={**oracle_diag, **diag},
                    uses_gt_for_prediction=True,
                    forbidden_for_method_table=True,
                    pipeline_root=pipeline_root,
                )
                eval_row.update(diag)
                eval_row["oracle_SF50"] = eval_row.get("local_SF50")
                eval_row["oracle_AP50"] = eval_row.get("local_AP50")
                eval_row["oracle_GT_best_IoU_mean"] = eval_row.get("local_GT_best_IoU_mean")
                eval_row["semantic_backend_enabled"] = False
                eval_row["source_scope"] = "current_run_v68_candidate_bank_oracle_diagnostic"
                metric_rows.append(eval_row)
    variant_rows = [_summarize_candidate_variant(metric_rows, variant) for variant in sorted({row["variant"] for row in metric_rows})]
    best = max(variant_rows, key=lambda row: float(row.get("local_score_free_match50_recall_mean") or 0.0), default={})
    best_sf50 = _float_or_none(best.get("local_score_free_match50_recall_mean"))
    best_ap50 = _float_or_none(best.get("local_AP50_mean"))
    best_gt = _float_or_none(best.get("local_GT_best_IoU_mean_mean"))
    best_count_per_frame = _float_or_none(best.get("candidate_count_per_frame_mean"))
    cb1 = next((row for row in variant_rows if row.get("variant") == "CB1_representative_only"), {})
    best_count = _float_or_none(best.get("candidate_count_per_chunk_mean"))
    cb1_count = _float_or_none(cb1.get("candidate_count_per_chunk_mean"))
    semantic_enabled = False
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "best_CB_oracle_SF50_ge_0p50": best_sf50 is not None and best_sf50 >= 0.50,
        "best_CB_oracle_AP50_ge_0p30": best_ap50 is not None and best_ap50 >= 0.30,
        "best_CB_GT_best_IoU_mean_ge_0p45": best_gt is not None and best_gt >= 0.45,
        "best_CB_candidate_count_per_frame_mean_le_25": best_count_per_frame is not None and best_count_per_frame <= 25.0,
        "best_CB_candidate_count_per_chunk_ge_half_representative": (
            best_count is not None and cb1_count is not None and best_count >= 0.5 * cb1_count
        ),
        "semantic_feature_success_rate_gate": True if not semantic_enabled else False,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    decision = "PASS_CANDIDATE_BANK_HEADROOM" if gate["pass"] else "FAIL_CANDIDATE_BANK_HEADROOM"
    _write_csv(output_root / "candidate_mask_rows.csv", candidate_rows)
    _write_csv(output_root / "candidate_metric_rows.csv", metric_rows)
    _write_csv(output_root / "candidate_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_csv(output_root / "visualization_rows.csv", visualization_rows)
    summary = {
        "phase": "v68_candidate_bank",
        "decision": decision,
        "diagnostic_only": True,
        "semantic_backend_enabled": semantic_enabled,
        "semantic_backend_status": "unavailable_no_reliable_semantic_feature_table",
        "scenes": scenes,
        "stride": int(args.stride),
        "support_variant": str(args.support_variant),
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "best_CB": best,
        "rows": {
            "candidate_mask_rows_csv": _rel(output_root / "candidate_mask_rows.csv"),
            "candidate_metric_rows_csv": _rel(output_root / "candidate_metric_rows.csv"),
            "candidate_variant_summary_rows_csv": _rel(output_root / "candidate_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
            "visualization_rows_csv": _rel(output_root / "visualization_rows.csv"),
        },
        "visual_root": _rel(visual_root),
        "notes": [
            "Non-oracle candidate rows do not use GT. Variant metrics are oracle diagnostics and forbidden for method tables.",
            "Semantic feature fields are recorded as unavailable because no reliable semantic feature table is exposed in the current Stream4D candidate artifacts.",
            "CB5 retains underseg-risk masks as shared support candidates; they are not claimed as confirmed object seeds.",
        ],
    }
    _write_json(output_root / "candidate_bank_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "candidate_bank_summary.json",
        output_root / "candidate_mask_rows.csv",
        output_root / "candidate_metric_rows.csv",
        output_root / "candidate_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
        output_root / "visualization_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit v68 local candidate bank.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v68_candidate_bank")
    parser.add_argument("--visual-root", default="outputs/audit/v68_visualizations/candidate_bank")
    parser.add_argument("--support-variant", default="I0_visible_tau0.10")
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.30)
    parser.add_argument("--visual-frames-per-scene", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
