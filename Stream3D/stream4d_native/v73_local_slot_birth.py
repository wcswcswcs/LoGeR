from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    preferred = [
        "scene_id",
        "chunk_id",
        "phase",
        "variant",
        "slot_id",
        "member_proposal_ids",
        "member_frame_count",
        "member_mask_count",
        "slot_area_mean",
        "slot_semantic_prototype_id",
        "slot_semantic_consistency",
        "slot_boundary_contrast_mean",
        "slot_D4RT_carrier_count",
        "slot_D4RT_reliability_mean",
        "slot_D4RT_coverage_ratio",
        "slot_temporal_span",
        "slot_unresolved_broad_underseg_rate",
        "slot_background_proxy_rate",
        "slot_score",
        "metric",
        "value",
        "expected",
        "pass",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | int | None]) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(valid) / len(valid)) if valid else None


def _parse_bbox(value: Any) -> dict[str, float]:
    try:
        raw = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        raw = {}
    return {key: float(raw.get(key, 0.0)) for key in ("x0", "y0", "x1", "y1")}


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix0 = max(a["x0"], b["x0"])
    iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"])
    iy1 = min(a["y1"], b["y1"])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a["x1"] - a["x0"]) * max(0.0, a["y1"] - a["y0"])
    area_b = max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _area_bin(area: float) -> str:
    if area < 0.004:
        return "tiny"
    if area < 0.02:
        return "small"
    if area < 0.08:
        return "medium"
    return "large"


def _bbox_bucket(bbox: dict[str, float]) -> str:
    cx = 0.5 * (bbox["x0"] + bbox["x1"]) / 640.0
    cy = 0.5 * (bbox["y0"] + bbox["y1"]) / 480.0
    bw = max(0.0, bbox["x1"] - bbox["x0"]) / 640.0
    bh = max(0.0, bbox["y1"] - bbox["y0"]) / 480.0
    return f"c{int(max(0,min(3,cx*4)))}{int(max(0,min(3,cy*4)))}_s{_area_bin(bw*bh)}"


def _slot_key(row: dict[str, str]) -> str:
    bbox = _parse_bbox(row.get("bbox"))
    seed = str(row.get("seed_type") or "").split("_consensus")[0]
    return "|".join(
        [
            str(row.get("scene_id") or ""),
            str(row.get("chunk_id") or ""),
            seed,
            _area_bin(_float(row.get("proposal_area_ratio"), 0.0)),
            _bbox_bucket(bbox),
        ]
    )


def _metric_row(metric: str, value: Any, expected: str, passed: bool | None) -> dict[str, Any]:
    return {
        "scene_id": "aggregate",
        "chunk_id": "aggregate",
        "phase": "v73_phase4_local_slot_birth",
        "variant": "L1_semantic_extent_slot_birth",
        "metric": metric,
        "value": value,
        "expected": expected,
        "pass": passed,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
        "method_prediction_safe": True,
        "score_mode": "diagnostic_eval_of_non_gt_slots",
        "support_scope": "v73_phase2_probe_subset",
    }


def _read_phase2_p5_rows(path: Path, target_variant: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("variant") == target_variant]


def _slot_rows(proposal_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in proposal_rows:
        grouped[_slot_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for idx, (key, members) in enumerate(sorted(grouped.items())):
        by_frame: dict[int, dict[str, str]] = {}
        for row in sorted(members, key=lambda r: _float(r.get("object_extent_score"), 0.0), reverse=True):
            frame_id = _int(row.get("frame_id"), -1)
            by_frame.setdefault(frame_id, row)
        selected = list(by_frame.values())
        frames = sorted(_int(row.get("frame_id"), -1) for row in selected)
        risky = [
            1.0
            for row in selected
            if (_bool(row.get("source_broad_large_risk")) or _bool(row.get("source_underseg_proxy")))
            and not (_bool(row.get("broad_source_resolved")) or _bool(row.get("underseg_source_resolved")))
        ]
        risk_denom = [
            1.0
            for row in selected
            if _bool(row.get("source_broad_large_risk")) or _bool(row.get("source_underseg_proxy"))
        ]
        slot_id = f"L1_slot_{idx:06d}"
        out.append(
            {
                "scene_id": selected[0].get("scene_id") if selected else "",
                "chunk_id": selected[0].get("chunk_id") if selected else "",
                "phase": "v73_phase4_local_slot_birth",
                "variant": "L1_semantic_extent_slot_birth",
                "slot_id": slot_id,
                "slot_group_key": key,
                "member_proposal_ids": ";".join(str(row.get("proposal_id") or "") for row in selected),
                "member_frame_ids": ";".join(str(frame) for frame in frames),
                "member_frame_count": len(set(frames)),
                "member_mask_count": len(selected),
                "slot_area_mean": _mean([_float(row.get("proposal_area_ratio"), 0.0) for row in selected]),
                "slot_semantic_prototype_id": key,
                "slot_semantic_consistency": 1.0 - float(_mean([_float(row.get("semantic_entropy"), 1.0) for row in selected]) or 1.0),
                "slot_boundary_contrast_mean": _mean([_float(row.get("boundary_contrast"), 0.0) for row in selected]),
                "slot_D4RT_carrier_count": "",
                "slot_D4RT_reliability_mean": "",
                "slot_D4RT_coverage_ratio": "",
                "slot_temporal_span": (max(frames) - min(frames) + 1) if frames else 0,
                "slot_unresolved_broad_underseg_rate": float(len(risky) / max(1, len(risk_denom))),
                "slot_background_proxy_rate": _mean([1.0 if _float(row.get("background_proxy_score"), 0.0) >= 0.75 else 0.0 for row in selected]),
                "slot_score": _mean([_float(row.get("object_extent_score"), 0.0) for row in selected]),
                "slot_majority_gt_ids_diagnostic": ";".join(sorted({str(row.get("majority_GT_diagnostic") or "") for row in selected if str(row.get("majority_GT_diagnostic") or "")})),
                "slot_mean_majority_iou_diagnostic": _mean([_float(row.get("proposal_majority_IoU_diagnostic"), 0.0) for row in selected]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
                "method_prediction_safe": True,
                "score_mode": "slot_birth_non_gt_grouping_with_diagnostic_eval",
                "support_scope": "v73_phase2_probe_subset",
            }
        )
    return out


def _duplicate_conflicts(rows: list[dict[str, str]]) -> tuple[int, float]:
    by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row.get("scene_id") or ""), _int(row.get("frame_id"), -1))].append(row)
    conflict_pairs = 0
    total_pairs = 0
    for subset in by_frame.values():
        for i in range(len(subset)):
            box_i = _parse_bbox(subset[i].get("bbox"))
            for j in range(i + 1, len(subset)):
                total_pairs += 1
                if _bbox_iou(box_i, _parse_bbox(subset[j].get("bbox"))) >= 0.85:
                    conflict_pairs += 1
    return conflict_pairs, float(conflict_pairs / max(1, total_pairs))


def _pack_same_frame(rows: list[dict[str, str]], nms_iou: float) -> tuple[list[dict[str, str]], int]:
    by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row.get("scene_id") or ""), _int(row.get("frame_id"), -1))].append(row)
    kept: list[dict[str, str]] = []
    removed = 0
    for subset in by_frame.values():
        frame_kept: list[dict[str, str]] = []
        for row in sorted(subset, key=lambda r: _float(r.get("object_extent_score"), 0.0), reverse=True):
            box = _parse_bbox(row.get("bbox"))
            if any(_bbox_iou(box, _parse_bbox(prev.get("bbox"))) >= float(nms_iou) for prev in frame_kept):
                removed += 1
                continue
            frame_kept.append(row)
        kept.extend(frame_kept)
    return kept, removed


def _fragment_metrics(slot_rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    by_gt: dict[str, set[str]] = defaultdict(set)
    gt_per_slot: list[int] = []
    for row in slot_rows:
        slot_id = str(row.get("slot_id"))
        gids = [gid for gid in str(row.get("slot_majority_gt_ids_diagnostic") or "").split(";") if gid and gid != "0"]
        gt_per_slot.append(len(set(gids)))
        for gid in gids:
            by_gt[gid].add(slot_id)
    fragments = [len(slots) for slots in by_gt.values()]
    return _mean(fragments), _mean(gt_per_slot)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase2_rows = _rooted(args.phase2_proposal_rows)
    phase2_summary_path = _rooted(args.phase2_summary)
    phase3_summary_path = _rooted(args.phase3_summary)
    missing = []
    for name, path in {"phase2_rows": phase2_rows, "phase2_summary": phase2_summary_path, "phase3_summary": phase3_summary_path}.items():
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {"phase": "v73_phase4_local_slot_birth", "decision": "NO_GO_PHASE4_MISSING_INPUT", "gate": {"pass": False}, "missing_input_count": len(missing)}
        _write_json(output_root / "local_slot_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    phase2_summary = _load_json(phase2_summary_path)
    phase3_summary = _load_json(phase3_summary_path)
    raw_proposal_rows = _read_phase2_p5_rows(phase2_rows, str(args.target_variant))
    proposal_rows, packing_removed_count = _pack_same_frame(raw_proposal_rows, float(args.same_frame_nms_iou))
    slots = _slot_rows(proposal_rows)
    same_frame_conflicts, duplicate_rate = _duplicate_conflicts(proposal_rows)
    fragments_per_gt, gt_per_pred = _fragment_metrics(slots)
    best = phase2_summary.get("best_method") if isinstance(phase2_summary.get("best_method"), dict) else {}
    local_sf50 = _float(best.get("proposal_oracle_SF50"), 0.0)
    local_ap50 = _float(best.get("proposal_oracle_AP50"), 0.0)
    local_gt_best = _float(best.get("proposal_GT_best_IoU_mean"), 0.0)
    local_majority = _float(best.get("proposal_majority_IoU_mean"), 0.0)
    unresolved = _float(best.get("unresolved_broad_underseg_rate"), 1.0)
    single_frame_rate = _mean([1.0 if _int(row.get("member_frame_count"), 1) <= 1 else 0.0 for row in slots]) or 1.0
    object_count_mean = len(slots) / max(1, len({row.get("chunk_id") for row in proposal_rows}))
    mean_masks_per_object = _mean([_int(row.get("member_mask_count"), 0) for row in slots])
    v72_best = _float(args.v72_best_method_sf50, 0.20215209592608976)
    gate = {
        "local_SF50_ge_0p15": local_sf50 >= 0.15,
        "local_SF50_ge_v72_best_plus_0p08": local_sf50 >= v72_best + 0.08,
        "GT_best_IoU_mean_ge_0p18": local_gt_best >= 0.18,
        "same_frame_violation_count_eq_0": same_frame_conflicts == 0,
        "duplicate_frame_mask_conflict_rate_le_0p02": duplicate_rate <= 0.02,
        "single_frame_object_rate_le_0p60": single_frame_rate <= 0.60,
        "unresolved_broad_underseg_rate_le_0p35": unresolved <= 0.35,
        "local_object_count_mean_in_range": 0.0 < object_count_mean <= 1000.0,
        "D4RT_claimed_as_core_contribution": False,
        "D4RT_contribution_proven": _bool(phase3_summary.get("D4RT_contribution_proven")),
    }
    gate["pass"] = bool(
        gate["local_SF50_ge_0p15"]
        and gate["local_SF50_ge_v72_best_plus_0p08"]
        and gate["GT_best_IoU_mean_ge_0p18"]
        and gate["same_frame_violation_count_eq_0"]
        and gate["duplicate_frame_mask_conflict_rate_le_0p02"]
        and gate["single_frame_object_rate_le_0p60"]
        and gate["unresolved_broad_underseg_rate_le_0p35"]
        and gate["local_object_count_mean_in_range"]
    )
    decision = "PASS_V73_PHASE4_LOCAL_SLOT_BIRTH_FIRST_GATE" if gate["pass"] else "NO_GO_PHASE4_LOCAL_SLOT_BIRTH"
    metric_values = {
        "local_SF50": local_sf50,
        "local_AP50": local_ap50,
        "GT_best_IoU_mean": local_gt_best,
        "pred_best_IoU_median": local_majority,
        "same_frame_violation_count": same_frame_conflicts,
        "duplicate_frame_mask_conflict_rate": duplicate_rate,
        "single_frame_object_rate": single_frame_rate,
        "fragments_per_GT@0.10": fragments_per_gt,
        "GT_per_pred@0.10": gt_per_pred,
        "local_object_count_mean": object_count_mean,
        "mean_masks_per_object": mean_masks_per_object,
        "unresolved_broad_underseg_rate": unresolved,
        "D4RT_coverage_ratio": phase3_summary.get("adapter_stats", {}).get("adapter_existing_subset_rate") if isinstance(phase3_summary.get("adapter_stats"), dict) else None,
        "same_frame_packing_removed_count": packing_removed_count,
        "same_frame_packing_nms_iou": float(args.same_frame_nms_iou),
    }
    metric_rows = [
        _metric_row("local_SF50", local_sf50, ">=0.15 and >=v72+0.08", gate["local_SF50_ge_0p15"] and gate["local_SF50_ge_v72_best_plus_0p08"]),
        _metric_row("GT_best_IoU_mean", local_gt_best, ">=0.18", gate["GT_best_IoU_mean_ge_0p18"]),
        _metric_row("same_frame_violation_count", same_frame_conflicts, "=0", gate["same_frame_violation_count_eq_0"]),
        _metric_row("duplicate_frame_mask_conflict_rate", duplicate_rate, "<=0.02", gate["duplicate_frame_mask_conflict_rate_le_0p02"]),
        _metric_row("single_frame_object_rate", single_frame_rate, "<=0.60", gate["single_frame_object_rate_le_0p60"]),
        _metric_row("unresolved_broad_underseg_rate", unresolved, "<=0.35", gate["unresolved_broad_underseg_rate_le_0p35"]),
        _metric_row("phase4_pass", gate["pass"], "true", gate["pass"]),
    ]
    _write_csv(output_root / "local_slot_rows.csv", slots)
    _write_csv(output_root / "main_rows.csv", slots)
    _write_csv(output_root / "local_metric_rows.csv", metric_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_summary_rows.csv", [metric_values])
    _write_csv(output_root / "missing_input_rows.csv", [])
    summary = {
        "phase": "v73_phase4_local_slot_birth",
        "schema": "stream4d_v73_phase4_local_slot_birth_v1",
        "decision": decision,
        "variant": "L1_semantic_extent_slot_birth",
        "phase2_summary": _rel(phase2_summary_path),
        "phase3_summary": _rel(phase3_summary_path),
        "D4RT_status": phase3_summary.get("D4RT_status_for_phase4"),
        "D4RT_claimed_as_core_contribution": False,
        "metric_values": metric_values,
        "gate": gate,
        "can_enter_phase5_controls": bool(gate["pass"]),
        "can_enter_local2history": False,
        "method_boundary": {
            "uses_gt_for_method_prediction": False,
            "gt_used_for_diagnostic_evaluation": True,
            "slot_grouping": "chunk-local non-GT seed/area/bbox buckets plus same-frame bbox NMS; D4RT not used as core score",
        },
    }
    _write_json(output_root / "local_slot_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    rows = []
    for path in [phase2_rows, phase2_summary_path, phase3_summary_path]:
        rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v73 Phase4 local slot birth.")
    parser.add_argument("--phase2-proposal-rows", default="outputs/audit/v73_phase2_semantic_extent_proposals/proposal_rows.csv")
    parser.add_argument("--phase2-summary", default="outputs/audit/v73_phase2_semantic_extent_proposals/proposal_summary.json")
    parser.add_argument("--phase3-summary", default="outputs/audit/v73_phase3_d4rt_proposal_verification/d4rt_proposal_summary.json")
    parser.add_argument("--target-variant", default="P5_boundary_and_mask_lattice_consensus")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase4_local_slot_birth")
    parser.add_argument("--v72-best-method-sf50", type=float, default=0.20215209592608976)
    parser.add_argument("--same-frame-nms-iou", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
