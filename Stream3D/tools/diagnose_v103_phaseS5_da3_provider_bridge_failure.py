from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "Stream3D" / "outputs" / "audit"
DEFAULT_PHASE9B_ROOT = AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_all_r1"
DEFAULT_OUTPUT_ROOT = AUDIT_ROOT / "v103_supp_phaseS5_da3_provider_bridge_failure_r1"

SCENES = ("scene0011_00", "scene0050_00")
GATE_RECALL_MIN = 0.35
GATE_DIFF_FALSE_MAX = 0.20
GATE_SAME_SEM_FALSE_MAX = 0.20
GATE_HARD_NEG_FALSE_MAX = 0.20
GATE_AUC_MIN = 0.65


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(ROOT))
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _quantile_rows(scene_id: str, label: str, df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if len(values) == 0:
            continue
        q = values.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        rows.append(
            {
                "schema_version": "stream4d_v103_phaseS5_bridge_feature_quantile_row_v1",
                "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
                "scene_id": scene_id,
                "diagnostic_group": label,
                "row_count": int(len(values)),
                "feature": feature,
                "q00": float(q.loc[0.0]),
                "q10": float(q.loc[0.1]),
                "q25": float(q.loc[0.25]),
                "q50": float(q.loc[0.5]),
                "q75": float(q.loc[0.75]),
                "q90": float(q.loc[0.9]),
                "q100": float(q.loc[1.0]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return rows


def _frontier_rows(scene_id: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    diff_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    same_sem_diff_gt = df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)
    labeled = same_gt | diff_gt
    pos_total = int(np.sum(same_gt))
    neg_total = int(np.sum(diff_gt))
    sem = np.nan_to_num(df["semantic_residual_cosine"].to_numpy(dtype=np.float64), nan=-1.0)
    sem_available = df["semantic_residual_available"].to_numpy(dtype=bool)
    final_score = df["final_bridge_score"].to_numpy(dtype=np.float64)
    union_score = df["gs_bridge_ratio_union"].to_numpy(dtype=np.float64)
    shared = df["gs_shared_gaussian_count"].to_numpy(dtype=np.float64)
    broad = df["broad_contamination_score"].to_numpy(dtype=np.float64)
    gap = df["frame_gap_index"].to_numpy(dtype=np.int64)
    common = (gap <= 4) & (shared >= 1) & (broad <= 0.20)
    base_specs = {
        "sem_available": common & sem_available,
        "sem_ge04": common & sem_available & (sem >= 0.40),
        "sem_ge05": common & sem_available & (sem >= 0.50),
        "missing_allowed_sem_ge04": common & ((sem_available & (sem >= 0.40)) | (~sem_available)),
    }
    score_specs = {
        "final_bridge_score": final_score,
        "union_bridge_score": union_score,
        "semantic_residual_cosine": sem,
        "final_sem_product": final_score * (sem + 1.1),
        "final_sem_min": np.minimum(final_score, np.clip((sem - 0.4) / 0.6, 0, 1)),
        "logshared_sem": np.log1p(shared) + sem,
        "final_minus_broad": final_score - broad,
        "final_sem_minus_broad": final_score * (sem + 1.1) - broad,
    }
    rows: list[dict[str, Any]] = []
    for base_id, base_mask in base_specs.items():
        base_count = int(np.sum(base_mask))
        if base_count == 0:
            continue
        for score_id, score in score_specs.items():
            idx = np.flatnonzero(base_mask)
            order = idx[np.argsort(score[idx])[::-1]]
            tp = np.cumsum(same_gt[order])
            fp = np.cumsum(diff_gt[order])
            fp_same_sem = np.cumsum(same_sem_diff_gt[order])
            lab = np.cumsum(labeled[order])
            valid = lab > 0
            recall = tp / max(pos_total, 1)
            diff_false = np.divide(fp, lab, out=np.zeros_like(fp, dtype=np.float64), where=valid)
            same_sem_false = np.divide(fp_same_sem, lab, out=np.zeros_like(fp, dtype=np.float64), where=valid)
            hard_neg_false = fp / max(neg_total, 1)
            formal = (
                (recall >= GATE_RECALL_MIN)
                & (diff_false <= GATE_DIFF_FALSE_MAX)
                & (same_sem_false <= GATE_SAME_SEM_FALSE_MAX)
                & (hard_neg_false <= GATE_HARD_NEG_FALSE_MAX)
            )
            clean = diff_false <= GATE_DIFF_FALSE_MAX
            if np.any(formal):
                best_index = int(np.flatnonzero(formal)[0])
                status = "formal_frontier_pass"
            elif np.any(clean):
                clean_idx = np.flatnonzero(clean)
                best_index = int(clean_idx[np.argmax(recall[clean_idx])])
                status = "best_under_diff_false_gate"
            else:
                trade = recall - np.maximum(0.0, diff_false - GATE_DIFF_FALSE_MAX) * 2.0
                best_index = int(np.argmax(trade))
                status = "best_tradeoff"
            rows.append(
                {
                    "schema_version": "stream4d_v103_phaseS5_bridge_frontier_row_v1",
                    "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
                    "scene_id": scene_id,
                    "base_filter_id": base_id,
                    "score_id": score_id,
                    "status": status,
                    "base_candidate_count": base_count,
                    "accepted_prefix_count": int(best_index + 1),
                    "diagnostic_positive_pair_count": pos_total,
                    "diagnostic_negative_pair_count": neg_total,
                    "true_positive_same_gt_count": int(tp[best_index]),
                    "false_positive_different_gt_count": int(fp[best_index]),
                    "false_positive_same_semantic_different_gt_count": int(fp_same_sem[best_index]),
                    "same_object_bridge_recall": float(recall[best_index]),
                    "different_gt_false_bridge_among_accepted": float(diff_false[best_index]),
                    "same_semantic_different_gt_false_bridge_among_accepted": float(same_sem_false[best_index]),
                    "hard_negative_false_accept_rate": float(hard_neg_false[best_index]),
                    "score_cutoff": float(score[order[best_index]]),
                    "frontier_formal_gate_pass": bool(np.any(formal)),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )
    return rows


def _label_pair_rows(scene_id: str, df: pd.DataFrame, *, max_rows: int = 20) -> list[dict[str, Any]]:
    false_df = df[df["diagnostic_different_gt"]].copy()
    if len(false_df) == 0:
        return []
    labels = (
        false_df["diagnostic_semantic_label_a"].astype(str)
        + " -> "
        + false_df["diagnostic_semantic_label_b"].astype(str)
    )
    counts = Counter(labels)
    rows: list[dict[str, Any]] = []
    for rank, (label_pair, count) in enumerate(counts.most_common(max_rows), start=1):
        rows.append(
            {
                "schema_version": "stream4d_v103_phaseS5_false_bridge_label_pair_row_v1",
                "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
                "scene_id": scene_id,
                "rank": rank,
                "diagnostic_label_pair": label_pair,
                "false_positive_different_gt_count": int(count),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return rows


def _scene_rows(phase9b_root: Path, scene_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bridge_path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    if not bridge_path.exists():
        failure = {
            "schema_version": "stream4d_v103_phaseS5_failure_row_v1",
            "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
            "scene_id": scene_id,
            "failure_id": "bridge_rows_missing",
            "path": bridge_path,
        }
        return [], [], [], [failure], {"scene_id": scene_id, "decision": "MISSING_INPUT"}
    df = pd.read_parquet(bridge_path)
    frontier = _frontier_rows(scene_id, df)
    best_clean = max(
        frontier,
        key=lambda row: (
            row["different_gt_false_bridge_among_accepted"] <= GATE_DIFF_FALSE_MAX,
            row["same_object_bridge_recall"],
        ),
    )
    base = (
        df["semantic_residual_available"].to_numpy(dtype=bool)
        & (df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= 0.40)
        & (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
        & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= 1)
        & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= 0.001)
        & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
    )
    sub = df[base].copy()
    features = [
        "semantic_residual_cosine",
        "final_bridge_score",
        "gs_shared_gaussian_count",
        "gs_bridge_ratio_min_support",
        "gs_bridge_ratio_union",
        "broad_contamination_score",
        "mask_a_primitive_count",
        "mask_b_primitive_count",
        "frame_gap_index",
    ]
    quantiles: list[dict[str, Any]] = []
    quantiles += _quantile_rows(scene_id, "TP_same_gt_tau04_block", sub[sub["diagnostic_same_gt"]], features)
    quantiles += _quantile_rows(scene_id, "FP_different_gt_tau04_block", sub[sub["diagnostic_different_gt"]], features)
    quantiles += _quantile_rows(
        scene_id,
        "FP_same_semantic_different_gt_tau04_block",
        sub[sub["diagnostic_same_semantic_different_gt"]],
        features,
    )
    label_pairs = _label_pair_rows(scene_id, sub)
    frontier_pass = any(row["frontier_formal_gate_pass"] for row in frontier)
    scene_summary = {
        "scene_id": scene_id,
        "bridge_rows": int(len(df)),
        "tau04_block_accepted_count": int(len(sub)),
        "tau04_block_true_positive_same_gt_count": int(sub["diagnostic_same_gt"].sum()),
        "tau04_block_false_positive_different_gt_count": int(sub["diagnostic_different_gt"].sum()),
        "best_clean_frontier_base_filter_id": best_clean["base_filter_id"],
        "best_clean_frontier_score_id": best_clean["score_id"],
        "best_clean_frontier_same_object_bridge_recall": best_clean["same_object_bridge_recall"],
        "best_clean_frontier_different_gt_false_bridge_among_accepted": best_clean[
            "different_gt_false_bridge_among_accepted"
        ],
        "frontier_formal_gate_pass": frontier_pass,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    failures: list[dict[str, Any]] = []
    if not frontier_pass:
        failures.append(
            {
                "schema_version": "stream4d_v103_phaseS5_failure_row_v1",
                "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
                "scene_id": scene_id,
                "failure_id": "no_single_score_frontier_reaches_provider_gate",
                "observed_best_clean_recall": best_clean["same_object_bridge_recall"],
                "expected_recall_min": GATE_RECALL_MIN,
                "observed_best_clean_diff_false": best_clean["different_gt_false_bridge_among_accepted"],
                "expected_diff_false_max": GATE_DIFF_FALSE_MAX,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    return frontier, quantiles, label_pairs, failures, scene_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase9b-root", type=Path, default=DEFAULT_PHASE9B_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    args = parser.parse_args()

    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    frontier_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    label_pair_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for scene_id in scenes:
        f_rows, q_rows, l_rows, failures, scene_summary = _scene_rows(args.phase9b_root, scene_id)
        frontier_rows.extend(f_rows)
        quantile_rows.extend(q_rows)
        label_pair_rows.extend(l_rows)
        failure_rows.extend(failures)
        scene_summaries.append(scene_summary)

    gate_rows = [
        {
            "gate_id": "all_requested_scene_frontiers_formal_pass",
            "pass": all(row.get("frontier_formal_gate_pass") for row in scene_summaries),
            "expected": True,
            "observed": {row["scene_id"]: row.get("frontier_formal_gate_pass") for row in scene_summaries},
            "scope": "phaseS5_da3_provider_bridge_failure",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
            "scope": "all",
        },
    ]
    decision = (
        "PASS_PHASES5_DA3_PROVIDER_BRIDGE_FRONTIER_DIAGNOSTIC"
        if all(row["pass"] for row in gate_rows)
        else "NO_GO_PHASES5_DA3_PROVIDER_BRIDGE_FRONTIER_DIAGNOSTIC"
    )
    summary = {
        "schema_version": "stream4d_v103_phaseS5_bridge_failure_summary_v1",
        "phase_id": "v103_supp_phaseS5_da3_provider_bridge_failure",
        "decision": decision,
        "phase9b_root": args.phase9b_root,
        "output_root": out,
        "scene_count": len(scenes),
        "failure_count": len(failure_rows),
        "scene_summaries": scene_summaries,
        "gate_recall_min": GATE_RECALL_MIN,
        "gate_diff_false_max": GATE_DIFF_FALSE_MAX,
        "gate_same_sem_false_max": GATE_SAME_SEM_FALSE_MAX,
        "gate_hard_neg_false_max": GATE_HARD_NEG_FALSE_MAX,
        "gate_auc_min": GATE_AUC_MIN,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    _write_csv(out / "frontier_rows.csv", frontier_rows)
    _write_csv(out / "feature_quantile_rows.csv", quantile_rows)
    _write_csv(out / "false_bridge_label_pair_rows.csv", label_pair_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
