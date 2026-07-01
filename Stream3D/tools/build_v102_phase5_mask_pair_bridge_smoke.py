from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
PHASE1_DIR = AUDIT_ROOT / "v102_phase1_fragment_casebook"
PHASE4_DIR = AUDIT_ROOT / "v102_phase4_persistent_primitive_diagnostic"
OUT_DIR = AUDIT_ROOT / "v102_phase5_mask_pair_bridge_smoke"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
FIXED_AUDIT_THRESHOLD = 0.05
BROAD_MASK_AREA_RATIO = 0.20


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | str:
    labels = labels.astype(bool)
    pos = int(np.sum(labels))
    neg = int(np.sum(~labels))
    if pos == 0 or neg == 0:
        return ""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average tied ranks.
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for group_id in range(len(unique_scores)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _mask_sets(participation: pd.DataFrame) -> dict[tuple[int, int], set[int]]:
    grouped: dict[tuple[int, int], set[int]] = {}
    for row in participation[["frame_id", "mask_id", "primitive_index"]].itertuples(index=False):
        key = (int(row.frame_id), int(row.mask_id))
        grouped.setdefault(key, set()).add(int(row.primitive_index))
    return grouped


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1_DIR / "summary.json")
    phase4 = _read_json(PHASE4_DIR / "summary.json")
    if not bool(phase4.get("phase4_pass_for_phase5_bridge")):
        raise RuntimeError("Phase4 did not produce primitive-mask participation and purity rows.")

    participation = pd.read_parquet(PHASE4_DIR / "primitive_mask_participation_rows.parquet")
    purity = pd.read_csv(PHASE4_DIR / "primitive_mask_purity_rows.csv")
    if len(participation) == 0 or len(purity) == 0:
        raise RuntimeError("Phase4 participation/purity artifacts are empty.")

    purity = purity.copy()
    purity["frame_id"] = purity["frame_id"].astype(int)
    purity["mask_id"] = purity["mask_id"].astype(int)
    purity["diagnostic_gt_instance"] = pd.to_numeric(purity["diagnostic_gt_instance"], errors="coerce")
    purity["diagnostic_gt_purity"] = pd.to_numeric(purity["diagnostic_gt_purity"], errors="coerce")
    purity["mask_area"] = pd.to_numeric(purity["mask_area"], errors="coerce")
    purity["mask_area_ratio"] = purity["mask_area"] / float(968 * 1296)
    purity_index = {(int(r.frame_id), int(r.mask_id)): r for r in purity.itertuples(index=False)}

    mask_to_primitives = _mask_sets(participation)
    frame_ids = sorted({frame for frame, _ in mask_to_primitives})
    if len(frame_ids) < 2:
        raise RuntimeError("Need at least two frames for cross-frame bridge smoke.")
    left_frame, right_frame = frame_ids[:2]
    left_masks = sorted(mask for frame, mask in mask_to_primitives if frame == left_frame)
    right_masks = sorted(mask for frame, mask in mask_to_primitives if frame == right_frame)

    rows: list[dict[str, Any]] = []
    for mask_a in left_masks:
        key_a = (left_frame, mask_a)
        set_a = mask_to_primitives[key_a]
        meta_a = purity_index.get(key_a)
        for mask_b in right_masks:
            key_b = (right_frame, mask_b)
            set_b = mask_to_primitives[key_b]
            meta_b = purity_index.get(key_b)
            shared = set_a & set_b
            union_count = len(set_a) + len(set_b) - len(shared)
            min_support = min(len(set_a), len(set_b))
            gt_a = float(meta_a.diagnostic_gt_instance) if meta_a is not None else np.nan
            gt_b = float(meta_b.diagnostic_gt_instance) if meta_b is not None else np.nan
            purity_a = float(meta_a.diagnostic_gt_purity) if meta_a is not None else np.nan
            purity_b = float(meta_b.diagnostic_gt_purity) if meta_b is not None else np.nan
            area_ratio_a = float(meta_a.mask_area_ratio) if meta_a is not None else np.nan
            area_ratio_b = float(meta_b.mask_area_ratio) if meta_b is not None else np.nan
            label_available = np.isfinite(gt_a) and np.isfinite(gt_b) and gt_a > 0 and gt_b > 0
            same_gt = bool(label_available and gt_a == gt_b)
            different_gt = bool(label_available and gt_a != gt_b)
            broad_risk = bool(
                (np.isfinite(area_ratio_a) and area_ratio_a > BROAD_MASK_AREA_RATIO)
                or (np.isfinite(area_ratio_b) and area_ratio_b > BROAD_MASK_AREA_RATIO)
            )
            ratio_min = float(len(shared) / max(min_support, 1))
            ratio_union = float(len(shared) / max(union_count, 1))
            score = ratio_min
            final_decision = bool(
                len(shared) >= 5
                and ratio_min >= FIXED_AUDIT_THRESHOLD
                and not broad_risk
            )
            rows.append(
                {
                    "schema_version": "stream4d_v102_phase5_mask_pair_bridge_row_v1",
                    "phase_id": "v102_phase5_mask_pair_bridge_smoke",
                    "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
                    "candidate_source": "smoke2_adjacent_frame_cropformer_masks",
                    "mask_a_observation_id": f"scene0050_00:{left_frame}:{mask_a}",
                    "mask_b_observation_id": f"scene0050_00:{right_frame}:{mask_b}",
                    "frame_a": left_frame,
                    "frame_b": right_frame,
                    "mask_a_id": mask_a,
                    "mask_b_id": mask_b,
                    "mask_a_primitive_count": len(set_a),
                    "mask_b_primitive_count": len(set_b),
                    "gs_shared_gaussian_count": len(shared),
                    "gs_bridge_ratio_min_support": ratio_min,
                    "gs_bridge_ratio_union": ratio_union,
                    "gs_opacity_mass_shared": "",
                    "same_frame_competing_cannot_link": False,
                    "broad_contamination_score": max(
                        area_ratio_a if np.isfinite(area_ratio_a) else 0.0,
                        area_ratio_b if np.isfinite(area_ratio_b) else 0.0,
                    ),
                    "broad_contamination_risk": broad_risk,
                    "semantic_residual_available": False,
                    "same_semantic_label_available": False,
                    "diagnostic_gt_a": gt_a if np.isfinite(gt_a) else "",
                    "diagnostic_gt_b": gt_b if np.isfinite(gt_b) else "",
                    "diagnostic_same_gt": same_gt,
                    "diagnostic_different_gt": different_gt,
                    "diagnostic_purity_min": np.nanmin([purity_a, purity_b]),
                    "final_bridge_score": score,
                    "final_bridge_decision": final_decision,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    bridge_df = pd.DataFrame(rows)
    bridge_path = OUT_DIR / "mask_pair_primitive_bridge_rows.parquet"
    bridge_df.to_parquet(bridge_path, index=False)

    label_mask = bridge_df["diagnostic_same_gt"] | bridge_df["diagnostic_different_gt"]
    labeled = bridge_df[label_mask].copy()
    auc = _auc(labeled["final_bridge_score"].to_numpy(dtype=np.float64), labeled["diagnostic_same_gt"].to_numpy(dtype=bool)) if len(labeled) else ""
    positive_count = int(np.sum(labeled["diagnostic_same_gt"])) if len(labeled) else 0
    negative_count = int(np.sum(labeled["diagnostic_different_gt"])) if len(labeled) else 0

    curve_rows = []
    passing_thresholds = []
    for threshold in THRESHOLDS:
        accepted = (
            (bridge_df["gs_shared_gaussian_count"] >= 5)
            & (bridge_df["gs_bridge_ratio_min_support"] >= threshold)
            & (~bridge_df["broad_contamination_risk"])
        )
        accepted_labeled = accepted & label_mask
        tp = int(np.sum(accepted & bridge_df["diagnostic_same_gt"]))
        fp = int(np.sum(accepted & bridge_df["diagnostic_different_gt"]))
        accepted_count = int(np.sum(accepted))
        recall = float(tp / positive_count) if positive_count else ""
        false_bridge_among_accepted = float(fp / max(int(np.sum(accepted_labeled)), 1)) if int(np.sum(accepted_labeled)) else ""
        hard_negative_false_accept_rate = float(fp / max(negative_count, 1)) if negative_count else ""
        purity_min_accepted = (
            float(np.nanmin(bridge_df.loc[accepted, "diagnostic_purity_min"].to_numpy(dtype=np.float64)))
            if accepted_count
            else ""
        )
        formal_gate_pass = bool(
            recall != ""
            and false_bridge_among_accepted != ""
            and recall >= 0.35
            and false_bridge_among_accepted <= 0.20
            and auc != ""
            and auc >= 0.65
            and hard_negative_false_accept_rate != ""
            and hard_negative_false_accept_rate <= 0.20
        )
        if formal_gate_pass:
            passing_thresholds.append(threshold)
        curve_rows.append(
            {
                "schema_version": "stream4d_v102_phase5_bridge_curve_row_v1",
                "phase_id": "v102_phase5_mask_pair_bridge_smoke",
                "threshold": threshold,
                "accepted_count": accepted_count,
                "diagnostic_positive_pair_count": positive_count,
                "diagnostic_negative_pair_count": negative_count,
                "true_positive_count": tp,
                "false_positive_different_gt_count": fp,
                "same_object_bridge_recall": recall,
                "different_gt_false_bridge_among_accepted_proxy": false_bridge_among_accepted,
                "hard_negative_false_accept_rate_proxy": hard_negative_false_accept_rate,
                "bridge_auc": auc,
                "purity_min_accepted": purity_min_accepted,
                "formal_same_semantic_false_bridge_available": False,
                "formal_gate_pass": False,
                "formal_gate_blocker": "same-semantic different-GT hard-negative labels are unavailable in this smoke.",
            }
        )

    fixed_row = next(row for row in curve_rows if row["threshold"] == FIXED_AUDIT_THRESHOLD)
    bridge_curve_csv = OUT_DIR / "bridge_curve_rows.csv"
    _write_csv(bridge_curve_csv, curve_rows)

    repair_specs = [
        ("fixed_strict", 5, 0.05, BROAD_MASK_AREA_RATIO),
        ("relax_threshold_0p001", 5, 0.001, BROAD_MASK_AREA_RATIO),
        ("relax_shared_count_1", 1, 0.001, BROAD_MASK_AREA_RATIO),
        ("relax_broad_veto_0p35", 5, 0.001, 0.35),
        ("no_broad_veto", 5, 0.001, None),
        ("no_broad_min_shared_1", 1, 0.001, None),
    ]
    repair_rows = []
    fixed_recall = fixed_row["same_object_bridge_recall"] if fixed_row["same_object_bridge_recall"] != "" else 0.0
    fixed_false = (
        fixed_row["different_gt_false_bridge_among_accepted_proxy"]
        if fixed_row["different_gt_false_bridge_among_accepted_proxy"] != ""
        else 1.0
    )
    for variant_id, min_shared, threshold, broad_limit in repair_specs:
        broad_ok = (
            np.ones(len(bridge_df), dtype=bool)
            if broad_limit is None
            else bridge_df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(broad_limit)
        )
        accepted = (
            (bridge_df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= min_shared)
            & (bridge_df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= threshold)
            & broad_ok
        )
        tp = int(np.sum(accepted & bridge_df["diagnostic_same_gt"].to_numpy(dtype=bool)))
        fp = int(np.sum(accepted & bridge_df["diagnostic_different_gt"].to_numpy(dtype=bool)))
        accepted_labeled = accepted & label_mask.to_numpy(dtype=bool)
        recall = float(tp / positive_count) if positive_count else ""
        false_bridge = float(fp / max(int(np.sum(accepted_labeled)), 1)) if int(np.sum(accepted_labeled)) else ""
        hard_negative_false_accept = float(fp / max(negative_count, 1)) if negative_count else ""
        recall_delta = recall - fixed_recall if recall != "" else ""
        false_delta = false_bridge - fixed_false if false_bridge != "" else ""
        repair_rows.append(
            {
                "schema_version": "stream4d_v102_phase5_repair_variant_row_v1",
                "phase_id": "v102_phase5_mask_pair_bridge_smoke",
                "variant_id": variant_id,
                "repair_family": "recall_low_and_broad_veto_sensitivity",
                "min_shared_gaussian_count": min_shared,
                "threshold": threshold,
                "broad_veto_area_ratio": "" if broad_limit is None else broad_limit,
                "accepted_count": int(np.sum(accepted)),
                "true_positive_count": tp,
                "false_positive_different_gt_count": fp,
                "same_object_bridge_recall": recall,
                "different_gt_false_bridge_among_accepted_proxy": false_bridge,
                "hard_negative_false_accept_rate_proxy": hard_negative_false_accept,
                "recall_delta_vs_fixed": recall_delta,
                "false_bridge_delta_vs_fixed": false_delta,
                "same_semantic_diff_gt_false_bridge_available": False,
                "promotable": False,
                "stop_reason": "formal same-semantic false-bridge labels unavailable; Phase1 repair candidate count is 0.",
            }
        )
    repair_variant_csv = OUT_DIR / "repair_variant_rows.csv"
    _write_csv(repair_variant_csv, repair_rows)

    hard_negative_df = bridge_df[bridge_df["diagnostic_different_gt"]].copy()
    hard_negative_path = OUT_DIR / "hard_negative_rows.csv"
    hard_negative_df.to_csv(hard_negative_path, index=False)
    pseudo_positive_df = bridge_df[bridge_df["diagnostic_same_gt"]].copy()
    pseudo_positive_path = OUT_DIR / "pseudo_positive_rows.csv"
    pseudo_positive_df.to_csv(pseudo_positive_path, index=False)

    formal_phase5_pass = False
    provider_summary_rows = [
        {
            "schema_version": "stream4d_v102_phase5_provider_bridge_summary_row_v1",
            "phase_id": "v102_phase5_mask_pair_bridge_smoke",
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official",
            "candidate_pair_count": int(len(bridge_df)),
            "diagnostic_positive_pair_count": positive_count,
            "diagnostic_negative_pair_count": negative_count,
            "bridge_auc": auc,
            "fixed_threshold": FIXED_AUDIT_THRESHOLD,
            "fixed_same_object_bridge_recall": fixed_row["same_object_bridge_recall"],
            "fixed_different_gt_false_bridge_among_accepted_proxy": fixed_row[
                "different_gt_false_bridge_among_accepted_proxy"
            ],
            "fixed_hard_negative_false_accept_rate_proxy": fixed_row["hard_negative_false_accept_rate_proxy"],
            "same_semantic_diff_gt_false_bridge_available": False,
            "surfel_or_gaussian_purity_available": True,
            "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
            "broad_contamination_rate": phase1.get("broad_contamination_rate"),
            "phase5_formal_pass": formal_phase5_pass,
            "blocker": "Bridge smoke is only two frames and lacks same-semantic hard-negative labels; Phase1 also has zero AP repair candidates.",
        }
    ]
    provider_summary_csv = OUT_DIR / "provider_bridge_summary_rows.csv"
    _write_csv(provider_summary_csv, provider_summary_rows)

    gate_rows = [
        {
            "gate_id": "same_object_bridge_recall",
            "pass": fixed_row["same_object_bridge_recall"] != "" and fixed_row["same_object_bridge_recall"] >= 0.35,
            "expected": ">=0.35",
            "observed": fixed_row["same_object_bridge_recall"],
            "severity": "diagnostic",
        },
        {
            "gate_id": "same_semantic_different_gt_false_bridge",
            "pass": False,
            "expected": "<=0.20 with same-semantic hard negatives",
            "observed": "unavailable_in_smoke2",
            "severity": "required_for_formal_phase5",
        },
        {
            "gate_id": "different_gt_false_bridge_proxy",
            "pass": fixed_row["different_gt_false_bridge_among_accepted_proxy"] != ""
            and fixed_row["different_gt_false_bridge_among_accepted_proxy"] <= 0.20,
            "expected": "<=0.20 proxy only",
            "observed": fixed_row["different_gt_false_bridge_among_accepted_proxy"],
            "severity": "diagnostic_proxy",
        },
        {
            "gate_id": "bridge_auc",
            "pass": auc != "" and auc >= 0.65,
            "expected": ">=0.65",
            "observed": auc,
            "severity": "diagnostic",
        },
        {
            "gate_id": "surfel_or_gaussian_purity_available",
            "pass": True,
            "expected": True,
            "observed": True,
            "severity": "required",
        },
        {
            "gate_id": "hard_negative_false_accept_rate",
            "pass": fixed_row["hard_negative_false_accept_rate_proxy"] != ""
            and fixed_row["hard_negative_false_accept_rate_proxy"] <= 0.20,
            "expected": "<=0.20 proxy",
            "observed": fixed_row["hard_negative_false_accept_rate_proxy"],
            "severity": "diagnostic_proxy",
        },
        {
            "gate_id": "phase1_repair_candidate_pair_count",
            "pass": int(phase1.get("repair_candidate_pair_count", 0)) >= 30,
            "expected": ">=30 before Phase6 AP repair",
            "observed": phase1.get("repair_candidate_pair_count"),
            "severity": "blocks_phase6_ap_repair",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
            "severity": "required",
        },
    ]
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(gate_csv, gate_rows)

    decision = (
        "DIAGNOSTIC_PROVIDER_ADVANCE_ONLY__FORMAL_PHASE5_AND_PHASE6_BLOCKED"
        if bool(provider_summary_rows[0]["surfel_or_gaussian_purity_available"])
        else "NO_GO_PROVIDER_BRIDGE_NOT_CLEAN"
    )
    summary = {
        "schema_version": "stream4d_v102_phase5_mask_pair_bridge_smoke_summary_v1",
        "phase_id": "v102_phase5_mask_pair_bridge_smoke",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase5_completed": True,
        "phase5_smoke_bridge_constructed": True,
        "phase5_formal_pass_for_phase6": formal_phase5_pass,
        "candidate_pair_count": int(len(bridge_df)),
        "diagnostic_positive_pair_count": positive_count,
        "diagnostic_negative_pair_count": negative_count,
        "bridge_auc": auc,
        "fixed_threshold": FIXED_AUDIT_THRESHOLD,
        "fixed_same_object_bridge_recall": fixed_row["same_object_bridge_recall"],
        "fixed_different_gt_false_bridge_among_accepted_proxy": fixed_row[
            "different_gt_false_bridge_among_accepted_proxy"
        ],
        "fixed_hard_negative_false_accept_rate_proxy": fixed_row["hard_negative_false_accept_rate_proxy"],
        "same_semantic_diff_gt_false_bridge_available": False,
        "surfel_or_gaussian_purity_available": True,
        "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
        "phase1_broad_contamination_rate": phase1.get("broad_contamination_rate"),
        "phase6_ap_repair_allowed": False,
        "phase6_blocker": "Phase5 formal gate is incomplete/blocked and Phase1 repair_candidate_pair_count is 0 (<30).",
        "truthfulness_note": "This is a two-frame Smoke-2 bridge diagnostic. GT labels are used only for diagnostic recall/false-bridge reporting, not for prediction or threshold promotion.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "mask_pair_primitive_bridge_rows": _rel(bridge_path),
            "bridge_curve_rows": _rel(bridge_curve_csv),
            "hard_negative_rows": _rel(hard_negative_path),
            "pseudo_positive_rows": _rel(pseudo_positive_path),
            "provider_bridge_summary_rows": _rel(provider_summary_csv),
            "repair_variant_rows": _rel(repair_variant_csv),
            "variant_gate_rows": _rel(gate_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
