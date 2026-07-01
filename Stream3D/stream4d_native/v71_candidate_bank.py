from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402


CANDIDATE_FIELDS = [
    "scene_id",
    "chunk_id",
    "frame_id",
    "mask_id",
    "mask_observation_id",
    "candidate_source_flags",
    "area_pixels",
    "area_ratio",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "mask_boundary_length",
    "mask_solidity_proxy",
    "representative_available",
    "raw_cropformer_available",
    "D4RT_carrier_count",
    "D4RT_visible_carrier_count",
    "D4RT_carrier_confidence_mean",
    "D4RT_carrier_reliability_mean",
    "D4RT_carrier_trajectory_entropy",
    "D4RT_component_count",
    "semantic_backend",
    "semantic_feature_available",
    "semantic_feature_norm",
    "semantic_entropy",
    "semantic_intra_mask_variance",
    "semantic_prototype_id",
    "semantic_prototype_margin",
    "RADIO_feature_available",
    "DINO_feature_available",
    "same_frame_overlap_count",
    "same_frame_competing_mask_count",
    "small_mask_risk",
    "large_mask_risk",
    "broad_background_risk",
    "underseg_proxy_score",
    "repeated_signature_id",
    "uses_gt_for_prediction",
    "diagnostic_only",
    "forbidden_for_method_table",
    "source_v68_row_hash_key",
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_load(value: Any, default: Any) -> Any:
    if value is None or str(value).strip() == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _percentile(values: list[float | None], q: float) -> float | None:
    valid = sorted(float(value) for value in values if value is not None)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    idx = int(round((len(valid) - 1) * q))
    idx = max(0, min(len(valid) - 1, idx))
    return valid[idx]


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _frame_area_from_bbox(row: dict[str, str], bbox: list[Any], bbox_size: list[Any]) -> float | None:
    if len(bbox) != 4 or len(bbox_size) != 2:
        return None
    width_norm = _float(bbox_size[0])
    height_norm = _float(bbox_size[1])
    if not width_norm or not height_norm:
        return None
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    if width_px <= 0.0 or height_px <= 0.0:
        return None
    image_w = width_px / max(1e-9, width_norm)
    image_h = height_px / max(1e-9, height_norm)
    return float(image_w * image_h)


def _candidate_source_flags(row: dict[str, str], source_types: list[str]) -> dict[str, bool]:
    d4rt_supported = int(float(row.get("d4rt_component_count") or 0)) > 0
    repeated = bool(row.get("repeated_signature_id")) and "repeated_signature" in source_types
    high_quality = "high_quality_raw" in source_types
    return {
        "C0_current_selected": _bool(row.get("current_selected_available")),
        "C1_representative_masks": _bool(row.get("representative_available")),
        "C2_high_quality_raw_CropFormer": high_quality,
        "C3_D4RT_supported_masks": d4rt_supported,
        "C4_semantic_compact_masks": False,
        "C5_temporal_signature_masks": repeated,
        "C6_union_candidate_bank": _bool(row.get("representative_available")) or high_quality or d4rt_supported or repeated,
    }


def _load_semantic_index(path: Path | None) -> dict[tuple[str, int, int], dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size <= 1:
        return {}
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (str(row.get("scene_id") or ""), int(float(row.get("frame_id") or 0)), int(float(row.get("mask_id") or 0)))
            except Exception:
                continue
            out[key] = row
    return out


def _convert_candidate_row(row: dict[str, str], semantic_index: dict[tuple[str, int, int], dict[str, str]]) -> dict[str, Any]:
    bbox = _json_load(row.get("bbox"), [0, 0, 0, 0])
    bbox_size = _json_load(row.get("bbox_size"), [0.0, 0.0])
    source_types = _json_load(row.get("source_types"), [])
    if not isinstance(source_types, list):
        source_types = []
    flags = _candidate_source_flags(row, [str(item) for item in source_types])
    area_ratio = _float(row.get("area_ratio"))
    frame_area = _frame_area_from_bbox(row, bbox, bbox_size)
    area_pixels = int(round(float(area_ratio) * frame_area)) if area_ratio is not None and frame_area is not None else ""
    semantic_backend_status = str(row.get("semantic_backend_status") or "")
    semantic_row = semantic_index.get((str(row.get("scene_id") or ""), int(float(row.get("frame_id") or 0)), int(float(row.get("mask_id") or 0))), {})
    semantic_available = _bool(semantic_row.get("feature_available")) or (
        bool(row.get("semantic_feature_id")) and "unavailable" not in semantic_backend_status
    )
    semantic_backend = str(semantic_row.get("semantic_backend") or semantic_backend_status or "unavailable")
    bbox_area_ratio = _float(row.get("bbox_area_ratio"))
    solidity = _float(row.get("mask_solidity_proxy"))
    large_mask = _bool(row.get("large_mask_risk"))
    underseg = _bool(row.get("underseg_risk"))
    broad = bool(large_mask or underseg or ((bbox_area_ratio or 0.0) >= 0.70 and (solidity or 1.0) <= 0.15))
    component_entropy = _float(row.get("d4rt_component_entropy")) or 0.0
    component_count = int(float(row.get("d4rt_component_count") or 0))
    underseg_score = 0.0
    if underseg:
        underseg_score = 1.0
    elif component_count > 0:
        underseg_score = min(1.0, component_entropy / 4.0)
    competing = _json_load(row.get("same_frame_competing_masks"), [])
    if not isinstance(competing, list):
        competing = []
    return {
        "scene_id": row.get("scene_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "frame_id": row.get("frame_id", ""),
        "mask_id": row.get("mask_id", ""),
        "mask_observation_id": row.get("mask_observation_id", ""),
        "candidate_source_flags": _json_dump(flags),
        "area_pixels": area_pixels,
        "area_ratio": row.get("area_ratio", ""),
        "bbox_x0": bbox[0] if len(bbox) == 4 else "",
        "bbox_y0": bbox[1] if len(bbox) == 4 else "",
        "bbox_x1": bbox[2] if len(bbox) == 4 else "",
        "bbox_y1": bbox[3] if len(bbox) == 4 else "",
        "bbox_area_ratio": row.get("bbox_area_ratio", ""),
        "bbox_aspect_ratio": row.get("aspect_ratio", ""),
        "mask_boundary_length": row.get("mask_boundary_length", ""),
        "mask_solidity_proxy": row.get("mask_solidity_proxy", ""),
        "representative_available": _bool(row.get("representative_available")),
        "raw_cropformer_available": _bool(row.get("raw_cropformer_available")),
        "D4RT_carrier_count": "",
        "D4RT_visible_carrier_count": "",
        "D4RT_carrier_confidence_mean": "",
        "D4RT_carrier_reliability_mean": "",
        "D4RT_carrier_trajectory_entropy": "",
        "D4RT_component_count": component_count,
        "semantic_backend": semantic_backend,
        "semantic_feature_available": semantic_available,
        "semantic_feature_norm": semantic_row.get("feature_norm", ""),
        "semantic_entropy": semantic_row.get("semantic_entropy", ""),
        "semantic_intra_mask_variance": semantic_row.get("semantic_intra_variance", ""),
        "semantic_prototype_id": semantic_row.get("semantic_prototype_id") or row.get("semantic_mode_id", ""),
        "semantic_prototype_margin": semantic_row.get("semantic_prototype_margin", ""),
        "RADIO_feature_available": False,
        "DINO_feature_available": semantic_available and "dino" in semantic_backend.lower(),
        "same_frame_overlap_count": row.get("same_frame_overlap_count", ""),
        "same_frame_competing_mask_count": len(competing),
        "small_mask_risk": _bool(row.get("small_mask_risk")),
        "large_mask_risk": large_mask,
        "broad_background_risk": broad,
        "underseg_proxy_score": underseg_score,
        "repeated_signature_id": row.get("repeated_signature_id", ""),
        "uses_gt_for_prediction": False,
        "diagnostic_only": False,
        "forbidden_for_method_table": False,
        "source_v68_row_hash_key": f"{row.get('scene_id','')}|{row.get('chunk_id','')}|{row.get('frame_id','')}|{row.get('mask_id','')}",
    }


def _write_converted_candidate_rows(
    input_csv: Path,
    output_csv: Path,
    semantic_index: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[dict[str, Any], Counter[str]]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    per_scene: Counter[str] = Counter()
    per_chunk: Counter[str] = Counter()
    per_frame: Counter[str] = Counter()
    d4rt_supported = 0
    semantic_available = 0
    representative = 0
    raw = 0
    small = 0
    large = 0
    broad = 0
    underseg = 0
    carrier_count_values: list[float | None] = []
    semantic_entropy_values: list[float | None] = []
    dino_available = 0
    radio_available = 0
    with input_csv.open(newline="", encoding="utf-8") as source, output_csv.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in reader:
            out = _convert_candidate_row(row, semantic_index)
            writer.writerow(out)
            counters["candidate_count_total"] += 1
            per_scene[str(out["scene_id"])] += 1
            per_chunk[str(out["chunk_id"])] += 1
            per_frame[f"{out['scene_id']}|{out['frame_id']}"] += 1
            representative += int(bool(out["representative_available"]))
            raw += int(bool(out["raw_cropformer_available"]))
            d4rt_supported += int(int(out["D4RT_component_count"] or 0) > 0)
            semantic_available += int(bool(out["semantic_feature_available"]))
            dino_available += int(bool(out["DINO_feature_available"]))
            radio_available += int(bool(out["RADIO_feature_available"]))
            small += int(bool(out["small_mask_risk"]))
            large += int(bool(out["large_mask_risk"]))
            broad += int(bool(out["broad_background_risk"]))
            underseg += int(float(out["underseg_proxy_score"] or 0.0) > 0.0)
            carrier_count_values.append(_float(out.get("D4RT_carrier_count")))
            semantic_entropy_values.append(_float(out.get("semantic_entropy")))
    total = counters["candidate_count_total"]
    stats: dict[str, Any] = {
        "candidate_count_total": int(total),
        "candidate_count_per_scene": dict(sorted(per_scene.items())),
        "candidate_count_per_chunk_mean": _mean([float(value) for value in per_chunk.values()]),
        "candidate_count_per_frame_mean": _mean([float(value) for value in per_frame.values()]),
        "candidate_count_per_frame_p90": _percentile([float(value) for value in per_frame.values()], 0.90),
        "representative_retention_rate": float(representative / max(1, total)),
        "raw_retention_rate": float(raw / max(1, total)),
        "D4RT_supported_candidate_rate": float(d4rt_supported / max(1, total)),
        "semantic_feature_success_rate": float(semantic_available / max(1, total)),
        "DINO_feature_success_rate": float(dino_available / max(1, total)),
        "RADIO_feature_success_rate": float(radio_available / max(1, total)),
        "semantic_entropy_mean": _mean(semantic_entropy_values),
        "semantic_entropy_p90": _percentile(semantic_entropy_values, 0.90),
        "D4RT_carrier_count_mean": _mean(carrier_count_values),
        "D4RT_reliability_mean": None,
        "small_mask_rate": float(small / max(1, total)),
        "large_mask_rate": float(large / max(1, total)),
        "broad_background_risk_rate": float(broad / max(1, total)),
        "underseg_proxy_rate": float(underseg / max(1, total)),
    }
    return stats, per_chunk


def _variant_alias(variant: str) -> str:
    aliases = {
        "CB0_current_selected_only": "C0_current_selected",
        "CB1_representative_only": "C1_representative_masks",
        "CB2_representative_plus_high_quality_raw": "C2_high_quality_raw_CropFormer",
        "CB3_CB2_plus_D4RT_supported_raw": "C3_D4RT_supported_masks",
        "CB4_CB3_plus_repeated_signature_masks": "C5_temporal_signature_masks",
        "CB5_CB4_underseg_as_shared_support": "C6_union_candidate_bank",
    }
    return aliases.get(variant, variant)


def _adapt_metric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["v68_variant"] = row.get("variant", "")
        new_row["variant"] = _variant_alias(row.get("variant", ""))
        new_row["uses_gt_for_prediction"] = True
        new_row["diagnostic_only"] = True
        new_row["forbidden_for_method_table"] = True
        out.append(new_row)
    return out


def _adapt_variant_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["v68_variant"] = row.get("variant", "")
        new_row["variant"] = _variant_alias(row.get("variant", ""))
        new_row["uses_gt_for_prediction"] = True
        new_row["diagnostic_only"] = True
        new_row["forbidden_for_method_table"] = True
        out.append(new_row)
    return out


def _copy_visualizations(source_root: Path, target_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not source_root.exists():
        return rows
    target_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_root.glob("*.png")):
        target = target_root / source.name
        shutil.copy2(source, target)
        rows.append({"source_visualization_path": _rel(source), "visualization_path": _rel(target), "sha256": _sha256(target)})
    return rows


def _best_row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    input_rows = _rooted(args.v68_candidate_rows)
    input_metric_rows = _rooted(args.v68_candidate_metric_rows)
    input_variant_rows = _rooted(args.v68_candidate_variant_summary_rows)
    input_summary = _rooted(args.v68_candidate_summary)
    semantic_feature_rows = _rooted(args.semantic_feature_rows) if args.semantic_feature_rows else None
    missing = []
    for name, path in [
        ("v68_candidate_rows", input_rows),
        ("v68_candidate_metric_rows", input_metric_rows),
        ("v68_candidate_variant_summary_rows", input_variant_rows),
        ("v68_candidate_summary", input_summary),
    ]:
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v71_candidate_bank",
            "decision": "FAIL_MISSING_INPUTS",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_inputs": missing,
        }
        _write_json(output_root / "candidate_bank_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    v68_summary = _read_json(input_summary)
    semantic_index = _load_semantic_index(semantic_feature_rows)
    candidate_stats, _per_chunk = _write_converted_candidate_rows(input_rows, output_root / "candidate_mask_rows.csv", semantic_index)
    metric_rows = _adapt_metric_rows(_load_csv_rows(input_metric_rows))
    variant_rows = _adapt_variant_rows(_load_csv_rows(input_variant_rows))
    for row in variant_rows:
        if str(row.get("variant") or "").startswith("C"):
            row["semantic_feature_success_rate_mean"] = candidate_stats["semantic_feature_success_rate"]
    visualization_rows = _copy_visualizations(_rooted(args.v68_visual_root), visual_root)
    _write_csv(output_root / "candidate_metric_rows.csv", metric_rows)
    _write_csv(output_root / "candidate_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "visualization_rows.csv", visualization_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    c6 = _best_row(variant_rows, "C6_union_candidate_bank")
    c0 = _best_row(variant_rows, "C0_current_selected")
    semantic_feature_success_rate = _float(candidate_stats.get("semantic_feature_success_rate"))
    c6_sf50 = _first_float(c6, "local_SF50_mean", "local_score_free_match50_recall_mean")
    c6_ap50 = _first_float(c6, "local_AP50_mean")
    c6_gt = _first_float(c6, "local_GT_best_IoU_mean_mean")
    candidate_count_per_frame_mean = _float(c6.get("candidate_count_per_frame_mean"))
    metric_frame_means = [
        _float(row.get("candidate_count_per_frame_mean"))
        for row in metric_rows
        if row.get("variant") == "C6_union_candidate_bank"
    ]
    candidate_count_per_frame_p90 = _percentile(metric_frame_means, 0.90)
    d4rt_supported_rate = _float(c6.get("D4RT_supported_candidate_rate_mean"))
    c0_sf50 = _first_float(c0, "local_SF50_mean", "local_score_free_match50_recall_mean")
    non_oracle_no_gt = True
    gate = {
        "all_inputs_present": True,
        "C6_union_candidate_bank_oracle_SF50_ge_0p50": c6_sf50 is not None and c6_sf50 >= 0.50,
        "C6_union_candidate_bank_AP50_ge_0p30": c6_ap50 is not None and c6_ap50 >= 0.30,
        "C6_union_candidate_bank_GT_best_IoU_mean_ge_0p45": c6_gt is not None and c6_gt >= 0.45,
        "candidate_count_per_frame_mean_le_35": candidate_count_per_frame_mean is not None and candidate_count_per_frame_mean <= 35.0,
        "candidate_count_per_frame_p90_le_60": candidate_count_per_frame_p90 is not None and candidate_count_per_frame_p90 <= 60.0,
        "D4RT_supported_candidate_rate_ge_0p40": d4rt_supported_rate is not None and d4rt_supported_rate >= 0.40,
        "semantic_feature_success_rate_ge_0p95": semantic_feature_success_rate is not None and semantic_feature_success_rate >= 0.95,
        "non_oracle_rows_use_gt_for_prediction_false": non_oracle_no_gt,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    candidate_headroom_pass = all(
        bool(gate[key])
        for key in [
            "C6_union_candidate_bank_oracle_SF50_ge_0p50",
            "C6_union_candidate_bank_AP50_ge_0p30",
            "C6_union_candidate_bank_GT_best_IoU_mean_ge_0p45",
            "candidate_count_per_frame_mean_le_35",
            "candidate_count_per_frame_p90_le_60",
            "D4RT_supported_candidate_rate_ge_0p40",
            "non_oracle_rows_use_gt_for_prediction_false",
        ]
    )
    if gate["pass"]:
        decision = "PASS_V71_CANDIDATE_BANK"
    elif candidate_headroom_pass and not gate["semantic_feature_success_rate_ge_0p95"]:
        decision = "NO_GO_PHASE1_SEMANTIC_FEATURE_BLOCKER"
    else:
        decision = "NO_GO_PHASE1_CANDIDATE_BANK"
    candidate_source_breakdown = Counter()
    with (output_root / "candidate_mask_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            flags = _json_load(row.get("candidate_source_flags"), {})
            if isinstance(flags, dict):
                for key, value in flags.items():
                    if value:
                        candidate_source_breakdown[key] += 1
    key_metrics = {
        **candidate_stats,
        "candidate_source_breakdown": dict(sorted(candidate_source_breakdown.items())),
        "C0_current_selected_oracle_SF50": c0_sf50,
        "C6_union_candidate_bank_oracle_SF50": c6_sf50,
        "C6_union_candidate_bank_AP50": c6_ap50,
        "C6_union_candidate_bank_GT_best_IoU_mean": c6_gt,
        "candidate_count_per_frame_mean": candidate_count_per_frame_mean,
        "candidate_count_per_frame_p90": candidate_count_per_frame_p90,
        "D4RT_supported_candidate_rate": d4rt_supported_rate,
        "semantic_feature_success_rate": semantic_feature_success_rate,
        "semantic_entropy_mean": candidate_stats.get("semantic_entropy_mean"),
        "semantic_entropy_p90": candidate_stats.get("semantic_entropy_p90"),
        "D4RT_carrier_count_mean": None,
        "D4RT_reliability_mean": None,
    }
    summary = {
        "phase": "v71_candidate_bank",
        "decision": decision,
        "gate": gate,
        "candidate_headroom_pass": candidate_headroom_pass,
        "semantic_backend_status": "dinov2_timm_joined" if semantic_index else v68_summary.get("semantic_backend_status", "unavailable_no_reliable_semantic_feature_table"),
        "source_artifacts": {
            "v68_candidate_summary": _rel(input_summary),
            "v68_candidate_rows": _rel(input_rows),
            "v68_candidate_metric_rows": _rel(input_metric_rows),
            "v68_candidate_variant_summary_rows": _rel(input_variant_rows),
            "semantic_feature_rows": _rel(semantic_feature_rows) if semantic_feature_rows and semantic_feature_rows.exists() else "",
        },
        "key_metrics": key_metrics,
        "rows": {
            "candidate_bank_summary_json": _rel(output_root / "candidate_bank_summary.json"),
            "candidate_mask_rows_csv": _rel(output_root / "candidate_mask_rows.csv"),
            "candidate_metric_rows_csv": _rel(output_root / "candidate_metric_rows.csv"),
            "candidate_variant_summary_rows_csv": _rel(output_root / "candidate_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
            "visualization_rows_csv": _rel(output_root / "visualization_rows.csv"),
        },
        "visual_root": _rel(visual_root),
        "notes": [
            "Phase 1 is a schema-normalized v71 candidate-universe audit derived from v68 candidate artifacts.",
            "Non-oracle candidate rows have uses_gt_for_prediction=false; oracle metric rows remain diagnostic_only=true and forbidden_for_method_table=true.",
            "D4RT carrier fields are left empty because v68 exposes component counts, not v70 carrier witness counts at candidate-row granularity.",
            "Semantic feature fields are joined from v71_semantic_features when provided; otherwise they remain unavailable.",
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
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v71 Phase 1 candidate bank audit.")
    parser.add_argument("--output-root", default="outputs/audit/v71_candidate_bank")
    parser.add_argument("--visual-root", default="outputs/audit/v71_visualizations/candidate_bank")
    parser.add_argument("--v68-candidate-summary", default="outputs/audit/v68_candidate_bank/candidate_bank_summary.json")
    parser.add_argument("--v68-candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--v68-candidate-metric-rows", default="outputs/audit/v68_candidate_bank/candidate_metric_rows.csv")
    parser.add_argument("--v68-candidate-variant-summary-rows", default="outputs/audit/v68_candidate_bank/candidate_variant_summary_rows.csv")
    parser.add_argument("--v68-visual-root", default="outputs/audit/v68_visualizations/candidate_bank")
    parser.add_argument("--semantic-feature-rows", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
