from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value in {"", "None", None}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in {"", "None", None}:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _bool_str(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _group(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "")), []).append(row)
    return out


def _token_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    gt_to_best_iou: dict[int, float] = {}
    purity_vals: list[float] = []
    areas: list[int] = []
    for row in rows:
        gt = _int(row, "diagnostic_gt_instance", default=0)
        iou = _float(row, "diagnostic_gt_iou", default=None)
        purity = _float(row, "diagnostic_gt_purity", default=None)
        if gt > 0 and iou is not None:
            gt_to_best_iou[gt] = max(gt_to_best_iou.get(gt, 0.0), float(iou))
        if purity is not None:
            purity_vals.append(float(purity))
        areas.append(_int(row, "area", default=0))
    missing_010 = [gt for gt, iou in sorted(gt_to_best_iou.items()) if float(iou) < 0.10]
    missing_025 = [gt for gt, iou in sorted(gt_to_best_iou.items()) if float(iou) < 0.25]
    mixed = [p for p in purity_vals if float(p) < 0.80]
    return {
        "token_rows": int(len(rows)),
        "token_area_mean": _mean([float(v) for v in areas if v > 0]),
        "token_area_p90": _quantile([float(v) for v in areas if v > 0], 0.90),
        "gt_instance_count_from_tokens": int(len(gt_to_best_iou)),
        "covered_gt_at_010_count": int(sum(1 for v in gt_to_best_iou.values() if float(v) >= 0.10)),
        "covered_gt_at_025_count": int(sum(1 for v in gt_to_best_iou.values() if float(v) >= 0.25)),
        "missing_gt_at_010_count": int(len(missing_010)),
        "missing_gt_at_010_ids": ",".join(str(v) for v in missing_010[:20]),
        "missing_gt_at_025_count": int(len(missing_025)),
        "mixed_token_count_purity_lt_080": int(len(mixed)),
        "diagnostic_purity_mean_from_tokens": _mean(purity_vals),
        "diagnostic_purity_p25_from_tokens": _quantile(purity_vals, 0.25),
    }


def _edge_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled: list[dict[str, str]] = [row for row in rows if _bool_str(str(row.get("diagnostic_same_gt", ""))) is not None]
    pos_scores = [float(row["semantic_affinity"]) for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is True]
    neg_scores = [float(row["semantic_affinity"]) for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is False]
    pos_obj = [float(row["object_affinity"]) for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is True]
    neg_obj = [float(row["object_affinity"]) for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is False]
    false_pos = [
        row
        for row in labeled
        if _bool_str(str(row.get("diagnostic_same_gt", ""))) is False and float(row.get("object_affinity", "0")) >= 0.50
    ]
    false_neg = [
        row
        for row in labeled
        if _bool_str(str(row.get("diagnostic_same_gt", ""))) is True and float(row.get("object_affinity", "0")) < 0.50
    ]
    same_frame_cannot_neg = [
        row
        for row in labeled
        if row.get("same_frame_cannot_link") == "True" and _bool_str(str(row.get("diagnostic_same_gt", ""))) is False
    ]
    same_frame_cannot_false_pos = [
        row for row in same_frame_cannot_neg if float(row.get("object_affinity", "0")) >= 0.50
    ]
    hard_neg_high_sem = [row for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is False and float(row["semantic_affinity"]) >= 0.75]
    hard_pos_low_sem = [row for row in labeled if _bool_str(str(row.get("diagnostic_same_gt", ""))) is True and float(row["semantic_affinity"]) < 0.50]
    return {
        "edge_rows": int(len(rows)),
        "gt_labeled_edge_rows": int(len(labeled)),
        "positive_edge_count": int(len(pos_scores)),
        "negative_edge_count": int(len(neg_scores)),
        "semantic_pos_mean": _mean(pos_scores),
        "semantic_neg_mean": _mean(neg_scores),
        "semantic_pos_p10": _quantile(pos_scores, 0.10),
        "semantic_neg_p90": _quantile(neg_scores, 0.90),
        "object_pos_mean": _mean(pos_obj),
        "object_neg_mean": _mean(neg_obj),
        "object_false_positive_count_at_050": int(len(false_pos)),
        "object_false_negative_count_at_050": int(len(false_neg)),
        "same_frame_cannot_link_negative_count": int(len(same_frame_cannot_neg)),
        "same_frame_cannot_link_false_positive_count_at_050": int(len(same_frame_cannot_false_pos)),
        "hard_negative_semantic_ge_075_count": int(len(hard_neg_high_sem)),
        "hard_positive_semantic_lt_050_count": int(len(hard_pos_low_sem)),
        "high_semantic_negative_rate": float(len(hard_neg_high_sem) / max(len(neg_scores), 1)),
        "low_semantic_positive_rate": float(len(hard_pos_low_sem) / max(len(pos_scores), 1)),
    }


def _material_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source", ""))
        variant = str(row.get("variant", ""))
        if variant not in {"P2_semantic_only", "P4_semantic_material", "P5_semantic_material_boundary"}:
            continue
        slot = out.setdefault(source, {})
        prefix = variant.replace("P2_", "p2_").replace("P4_", "p4_").replace("P5_", "p5_")
        slot[f"{prefix}_auc"] = _float(row, "object_part_compatibility_AUC", default=None)
        slot[f"{prefix}_false_merge_rate"] = _float(row, "false_merge_rate", default=None)
        slot[f"{prefix}_false_merge_reduction_vs_semantic_graph"] = _float(
            row, "false_merge_reduction_vs_semantic_graph", default=None
        )
        slot[f"{prefix}_phase2_gate_pass"] = row.get("phase2_gate_pass", "")
        slot["measurement_count"] = _int(row, "measurement_count", default=0)
    return out


def _phase1_blocker(auc: float | None, coverage: float | None) -> str:
    auc_fail = auc is None or float(auc) < 0.75
    coverage_fail = coverage is None or float(coverage) < 0.70
    if auc_fail and coverage_fail:
        return "both_auc_and_coverage"
    if auc_fail:
        return "semantic_auc_only"
    if coverage_fail:
        return "coverage_only"
    return "none"


def _interpret(row: dict[str, Any], *, max_tokens: int) -> str:
    auc = row.get("semantic_affinity_AUC")
    coverage = row.get("coverage@0.10")
    token_cap_hit = bool(row.get("token_cap_hit"))
    material_split_count = int(row.get("material_split_split_mask_count") or 0)
    if auc is not None and float(auc) >= 0.73 and coverage is not None and float(coverage) < 0.70:
        if token_cap_hit:
            return "near_auc_but_coverage_gap_with_token_cap"
        return "near_auc_but_coverage_gap"
    if coverage is not None and float(coverage) >= 0.70 and auc is not None and float(auc) < 0.70:
        return "coverage_ok_but_semantic_affinity_low"
    if coverage is not None and float(coverage) >= 0.70 and auc is not None and float(auc) < 0.75:
        return "coverage_ok_but_auc_below_gate"
    if material_split_count > 0 and auc is not None and float(auc) < 0.75:
        return "material_split_applied_but_did_not_fix_auc"
    if token_cap_hit and int(row.get("part_token_count") or 0) >= int(max_tokens):
        return "token_cap_hit"
    return "multi_factor_failure"


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene_source = {(str(row["scene"]), str(row["source"])): row for row in rows}
    out: list[dict[str, Any]] = []
    for (scene, source), row in by_scene_source.items():
        if not source.endswith("_material_split"):
            continue
        base_source = source.replace("_material_split", "")
        base = by_scene_source.get((scene, base_source))
        if base is None and source == "hybrid_union_feature_merge_material_split":
            base = by_scene_source.get((scene, "hybrid_union_feature_merge"))
        if base is None:
            continue
        out.append(
            {
                "scene": scene,
                "base_source": base["source"],
                "repair_source": source,
                "semantic_affinity_AUC_delta": _none_delta(row.get("semantic_affinity_AUC"), base.get("semantic_affinity_AUC")),
                "coverage@0.10_delta": _none_delta(row.get("coverage@0.10"), base.get("coverage@0.10")),
                "mixed_part_rate_delta": _none_delta(row.get("mixed_part_rate"), base.get("mixed_part_rate")),
                "part_token_count_delta": int(row.get("part_token_count") or 0) - int(base.get("part_token_count") or 0),
                "material_split_split_mask_count": row.get("material_split_split_mask_count"),
                "material_split_created_fragment_count": row.get("material_split_created_fragment_count"),
                "gate_after_repair": row.get("gate_pass_phase1"),
            }
        )
    return out


def _none_delta(left: Any, right: Any) -> float | None:
    if left in {"", None} or right in {"", None}:
        return None
    return float(left) - float(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    root_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = audit_root / scene
        source_rows = _read_csv(scene_dir / "source_audit_rows.csv")
        tokens_by_source = _group(_read_csv(scene_dir / "part_token_rows.csv"), "source")
        edges_by_source = _group(_read_csv(scene_dir / "part_edge_rows.csv"), "source")
        material_by_source = _material_stats(_read_csv(scene_dir / "material_graph_rows.csv"))
        for source_row in source_rows:
            source = str(source_row.get("source", ""))
            auc = _float(source_row, "semantic_affinity_AUC", default=None)
            coverage = _float(source_row, "coverage@0.10", default=None)
            token_count = _int(source_row, "part_token_count", default=0)
            row: dict[str, Any] = {
                "scene": scene,
                "source": source,
                "gate_pass_phase1": source_row.get("gate_pass_phase1", ""),
                "phase1_blocker": _phase1_blocker(auc, coverage),
                "semantic_affinity_AUC": auc,
                "semantic_auc_gap_to_075": None if auc is None else max(0.0, 0.75 - float(auc)),
                "coverage@0.10": coverage,
                "coverage_gap_to_070": None if coverage is None else max(0.0, 0.70 - float(coverage)),
                "coverage@0.25": _float(source_row, "coverage@0.25", default=None),
                "object_part_compatibility_AUC": _float(source_row, "object_part_compatibility_AUC", default=None),
                "same_frame_same_class_false_merge_rate": _float(
                    source_row, "same_frame_same_class_false_merge_rate", default=None
                ),
                "part_token_count": token_count,
                "token_cap_hit": bool(token_count >= int(args.max_tokens)),
                "mixed_part_rate": _float(source_row, "mixed_part_rate", default=None),
                "part_purity_diagnostic_mean": _float(source_row, "part_purity_diagnostic_mean", default=None),
                "repair_strategy": source_row.get("repair_strategy", ""),
                "material_split_input_mask_count": _int(source_row, "material_split_input_mask_count", default=0),
                "material_split_output_mask_count": _int(source_row, "material_split_output_mask_count", default=0),
                "material_split_split_mask_count": _int(source_row, "material_split_split_mask_count", default=0),
                "material_split_created_fragment_count": _int(
                    source_row, "material_split_created_fragment_count", default=0
                ),
                "material_split_total_visible_tube_anchors_inside_masks": _int(
                    source_row, "material_split_total_visible_tube_anchors_inside_masks", default=0
                ),
            }
            row.update(_token_stats(tokens_by_source.get(source, [])))
            row.update(_edge_stats(edges_by_source.get(source, [])))
            row.update(material_by_source.get(source, {}))
            row["root_cause_interpretation"] = _interpret(row, max_tokens=int(args.max_tokens))
            root_rows.append(row)

    delta_rows = _delta_rows(root_rows)
    best_by_scene: dict[str, Any] = {}
    for scene in scenes:
        scene_rows = [row for row in root_rows if row["scene"] == scene]
        best = max(
            scene_rows,
            key=lambda row: (
                bool(row.get("gate_pass_phase1") == "True"),
                float(row.get("semantic_affinity_AUC") or -1.0),
                float(row.get("coverage@0.10") or -1.0),
            ),
            default={},
        )
        best_by_scene[scene] = {
            key: best.get(key)
            for key in [
                "source",
                "gate_pass_phase1",
                "phase1_blocker",
                "semantic_affinity_AUC",
                "coverage@0.10",
                "mixed_part_rate",
                "part_token_count",
                "root_cause_interpretation",
            ]
        }
    summary = {
        "audit_root": str(audit_root),
        "scenes": scenes,
        "max_tokens": int(args.max_tokens),
        "best_by_scene": best_by_scene,
        "material_split_deltas": delta_rows,
        "root_cause_rows_csv": str(Path(args.output_root) / "hard_scene_root_cause_rows.csv"),
        "material_split_delta_rows_csv": str(Path(args.output_root) / "material_split_delta_rows.csv"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = Path(args.output_root)
    _write_csv(out / "hard_scene_root_cause_rows.csv", root_rows)
    _write_csv(out / "material_split_delta_rows.csv", delta_rows)
    _write_json(out / "hard_scene_root_cause_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
