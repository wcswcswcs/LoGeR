from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "Stream3D" / "outputs" / "audit"
DEFAULT_OUTPUT_ROOT = AUDIT_ROOT / "v103_supp_phaseS5_da3_preregistered_barrier_r1"

GATE_RECALL_MIN = 0.35
GATE_DIFF_FALSE_MAX = 0.20
GATE_SAME_SEM_FALSE_MAX = 0.20
GATE_HARD_NEG_FALSE_MAX = 0.20
GATE_AUC_MIN = 0.65


RUN_ROOTS = [
    {
        "run_id": "full32_start029_all",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_all_r1",
        "validation_role": "full_window",
    },
    {
        "run_id": "subchunk16_start029_all",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_subchunk16_start029_r1",
        "validation_role": "subchunk16",
    },
    {
        "run_id": "subchunk16_start045_all",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_subchunk16_start045_r1",
        "validation_role": "subchunk16",
    },
    {
        "run_id": "subchunk8_scene0011_start029",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_subchunk8_scene0011_start029_r1",
        "validation_role": "targeted_subchunk8",
    },
    {
        "run_id": "subchunk8_scene0011_start037",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_subchunk8_scene0011_start037_r1",
        "validation_role": "targeted_subchunk8",
    },
    {
        "run_id": "subchunk8_scene0050_start045",
        "phase9b_root": AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_subchunk8_scene0050_start045_r1",
        "validation_role": "targeted_subchunk8",
    },
]


BARRIER_VARIANTS = [
    {
        "variant_id": "B0_phase9b_sem05_missing_allow",
        "semantic_min": 0.50,
        "missing_policy": "allow",
        "bridge_min": 0.001,
        "score_id": "final_bridge_score",
        "score_min": None,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B1_sem05_missing_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_bridge_score",
        "score_min": None,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B2_sem05_final030_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_bridge_score",
        "score_min": 0.30,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B3_sem05_final_minus_broad030_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_minus_broad",
        "score_min": 0.30,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B4_sem05_final_sem_minus_broad070_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_sem_minus_broad",
        "score_min": 0.70,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B5_sem05_union005_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "union_bridge_score",
        "score_min": 0.05,
        "topk_per_mask": 0,
    },
    {
        "variant_id": "B6_sem05_top3_final_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_bridge_score",
        "score_min": None,
        "topk_per_mask": 3,
    },
    {
        "variant_id": "B7_sem05_top5_final_sem_minus_broad_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_sem_minus_broad",
        "score_min": None,
        "topk_per_mask": 5,
    },
    {
        "variant_id": "B8_sem04_top3_final_sem_minus_broad_allow",
        "semantic_min": 0.40,
        "missing_policy": "allow",
        "bridge_min": 0.001,
        "score_id": "final_sem_minus_broad",
        "score_min": None,
        "topk_per_mask": 3,
    },
    {
        "variant_id": "B9_sem05_final030_top5_block",
        "semantic_min": 0.50,
        "missing_policy": "block",
        "bridge_min": 0.001,
        "score_id": "final_bridge_score",
        "score_min": 0.30,
        "topk_per_mask": 5,
    },
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(ROOT))
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | str:
    labels = labels.astype(bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    pos = int(np.sum(labels))
    neg = int(np.sum(~labels))
    if pos == 0 or neg == 0:
        return ""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for group_id in range(len(unique_scores)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    final_score = df["final_bridge_score"].to_numpy(dtype=np.float64)
    semantic = np.nan_to_num(df["semantic_residual_cosine"].to_numpy(dtype=np.float64), nan=-1.0)
    broad = df["broad_contamination_score"].to_numpy(dtype=np.float64)
    shared = df["gs_shared_gaussian_count"].to_numpy(dtype=np.float64)
    union = df["gs_bridge_ratio_union"].to_numpy(dtype=np.float64)
    return {
        "final_bridge_score": final_score,
        "union_bridge_score": union,
        "semantic_residual_cosine": semantic,
        "final_sem_product": final_score * (semantic + 1.1),
        "final_sem_minus_broad": final_score * (semantic + 1.1) - broad,
        "final_minus_broad": final_score - broad,
        "logshared_sem": np.log1p(shared) + semantic,
    }


def _topk_accept(df: pd.DataFrame, base: np.ndarray, score: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return base
    idx = np.flatnonzero(base)
    out = np.zeros(len(df), dtype=bool)
    if len(idx) == 0:
        return out
    work = pd.DataFrame(
        {
            "mask_a": df["mask_a_observation_id"].astype(str).to_numpy()[idx],
            "mask_b": df["mask_b_observation_id"].astype(str).to_numpy()[idx],
            "score": score[idx],
            "row_index": idx,
        }
    )
    for col in ["mask_a", "mask_b"]:
        ranked = work.sort_values([col, "score"], ascending=[True, False], kind="mergesort")
        ranks = ranked.groupby(col, sort=False).cumcount().to_numpy() + 1
        out[ranked.loc[ranks <= topk, "row_index"].to_numpy(dtype=np.int64)] = True
    return out & base


def _variant_accept(df: pd.DataFrame, variant: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    score = _scores(df)[str(variant["score_id"])]
    semantic = df["semantic_residual_cosine"].to_numpy(dtype=np.float64)
    semantic_available = df["semantic_residual_available"].to_numpy(dtype=bool)
    if variant["missing_policy"] == "allow":
        semantic_ok = (~semantic_available) | (semantic >= float(variant["semantic_min"]))
    else:
        semantic_ok = semantic_available & (semantic >= float(variant["semantic_min"]))
    base = (
        (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
        & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= 1)
        & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= float(variant["bridge_min"]))
        & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
        & semantic_ok
    )
    if variant["score_min"] is not None:
        base &= score >= float(variant["score_min"])
    accepted = _topk_accept(df, base, score, int(variant["topk_per_mask"]))
    return accepted, score


def _eval_variant(run_id: str, validation_role: str, phase9b_root: Path, scene_id: str, variant: dict[str, Any]) -> dict[str, Any]:
    path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    df = pd.read_parquet(path)
    accepted, score = _variant_accept(df, variant)
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    diff_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    same_sem_diff = df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)
    labeled = same_gt | diff_gt
    accepted_labeled = accepted & labeled
    tp = int(np.sum(accepted & same_gt))
    fp = int(np.sum(accepted & diff_gt))
    fp_same_sem = int(np.sum(accepted & same_sem_diff))
    accepted_count = int(np.sum(accepted))
    accepted_labeled_count = int(np.sum(accepted_labeled))
    positive_total = int(np.sum(same_gt))
    negative_total = int(np.sum(diff_gt))
    recall = float(tp / max(positive_total, 1))
    diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    hard_neg_false = float(fp / max(negative_total, 1))
    label_mask = labeled
    score_auc = _auc(score[label_mask], same_gt[label_mask])
    pass_gate = bool(
        accepted_labeled_count > 0
        and score_auc != ""
        and recall >= GATE_RECALL_MIN
        and diff_false <= GATE_DIFF_FALSE_MAX
        and same_sem_false <= GATE_SAME_SEM_FALSE_MAX
        and hard_neg_false <= GATE_HARD_NEG_FALSE_MAX
        and float(score_auc) >= GATE_AUC_MIN
    )
    return {
        "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_variant_row_v1",
        "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
        "run_id": run_id,
        "validation_role": validation_role,
        "phase9b_root": phase9b_root,
        "scene_id": scene_id,
        "variant_id": variant["variant_id"],
        "semantic_min": variant["semantic_min"],
        "missing_policy": variant["missing_policy"],
        "bridge_min": variant["bridge_min"],
        "score_id": variant["score_id"],
        "score_min": "" if variant["score_min"] is None else variant["score_min"],
        "topk_per_mask": variant["topk_per_mask"],
        "accepted_count": accepted_count,
        "accepted_labeled_count": accepted_labeled_count,
        "true_positive_same_gt_count": tp,
        "false_positive_different_gt_count": fp,
        "false_positive_same_semantic_different_gt_count": fp_same_sem,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_object_bridge_recall": recall,
        "different_gt_false_bridge_among_accepted": diff_false,
        "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
        "hard_negative_false_accept_rate": hard_neg_false,
        "score_auc": score_auc,
        "pre_registered_barrier_gate_pass": pass_gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _scene_ids(phase9b_root: Path) -> list[str]:
    path = phase9b_root / "provider_scene_summary_rows.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return sorted(str(x) for x in df["scene_id"].dropna().unique().tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    variant_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for run in RUN_ROOTS:
        phase9b_root = Path(run["phase9b_root"])
        scenes = _scene_ids(phase9b_root)
        if not scenes:
            missing_rows.append(
                {
                    "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_missing_row_v1",
                    "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
                    "run_id": run["run_id"],
                    "phase9b_root": phase9b_root,
                    "missing_id": "provider_scene_summary_rows_missing_or_empty",
                }
            )
            continue
        for scene_id in scenes:
            bridge_path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
            if not bridge_path.exists():
                missing_rows.append(
                    {
                        "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_missing_row_v1",
                        "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
                        "run_id": run["run_id"],
                        "phase9b_root": phase9b_root,
                        "scene_id": scene_id,
                        "missing_id": "bridge_rows_missing",
                        "path": bridge_path,
                    }
                )
                continue
            for variant in BARRIER_VARIANTS:
                variant_rows.append(
                    _eval_variant(
                        str(run["run_id"]),
                        str(run["validation_role"]),
                        phase9b_root,
                        scene_id,
                        variant,
                    )
                )

    df = pd.DataFrame(variant_rows)
    summary_rows: list[dict[str, Any]] = []
    stable_all_root_variant_ids: list[str] = []
    stable_subchunk16_variant_ids: list[str] = []
    if len(df):
        grouped = df.groupby(["variant_id", "validation_role"], dropna=False)
        for (variant_id, role), group in grouped:
            summary_rows.append(
                {
                    "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_summary_row_v1",
                    "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
                    "variant_id": variant_id,
                    "validation_role": role,
                    "row_count": int(len(group)),
                    "pass_count": int(group["pre_registered_barrier_gate_pass"].astype(bool).sum()),
                    "all_rows_pass": bool(group["pre_registered_barrier_gate_pass"].astype(bool).all()),
                    "min_recall": float(group["same_object_bridge_recall"].min()),
                    "max_diff_false": float(group["different_gt_false_bridge_among_accepted"].max()),
                    "min_auc": float(pd.to_numeric(group["score_auc"], errors="coerce").min()),
                }
            )
        all_roots = df[df["run_id"].isin(["full32_start029_all", "subchunk16_start029_all", "subchunk16_start045_all"])]
        for variant_id, group in all_roots.groupby("variant_id"):
            if bool(group["pre_registered_barrier_gate_pass"].astype(bool).all()):
                stable_all_root_variant_ids.append(str(variant_id))
        sub16 = df[df["validation_role"] == "subchunk16"]
        for variant_id, group in sub16.groupby("variant_id"):
            if bool(group["pre_registered_barrier_gate_pass"].astype(bool).all()):
                stable_subchunk16_variant_ids.append(str(variant_id))
    if not stable_all_root_variant_ids:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_failure_row_v1",
                "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
                "failure_id": "no_pre_registered_variant_passes_full32_and_subchunk16_roots",
                "expected": "same fixed GT-free variant passes all requested scenes in full32 and both subchunk16 roots",
                "observed_stable_variant_ids": stable_all_root_variant_ids,
            }
        )

    gate_rows = [
        {
            "gate_id": "stable_variant_full32_and_subchunk16_roots",
            "pass": bool(stable_all_root_variant_ids),
            "expected": "nonempty",
            "observed": stable_all_root_variant_ids,
        },
        {
            "gate_id": "stable_variant_subchunk16_roots",
            "pass": bool(stable_subchunk16_variant_ids),
            "expected": "nonempty",
            "observed": stable_subchunk16_variant_ids,
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
        },
    ]
    decision = (
        "PASS_PHASES5_DA3_PREREGISTERED_BARRIER_STABLE_PROVIDER_DIAGNOSTIC"
        if bool(stable_all_root_variant_ids)
        else "NO_GO_PHASES5_DA3_PREREGISTERED_BARRIER_STABLE_PROVIDER_DIAGNOSTIC"
    )
    summary = {
        "schema_version": "stream4d_v103_phaseS5_preregistered_barrier_summary_v1",
        "phase_id": "v103_supp_phaseS5_da3_preregistered_barrier",
        "decision": decision,
        "output_root": out,
        "run_root_count": len(RUN_ROOTS),
        "variant_count": len(BARRIER_VARIANTS),
        "variant_row_count": len(variant_rows),
        "summary_row_count": len(summary_rows),
        "missing_count": len(missing_rows),
        "failure_count": len(failure_rows),
        "stable_all_root_variant_ids": stable_all_root_variant_ids,
        "stable_subchunk16_variant_ids": stable_subchunk16_variant_ids,
        "truthfulness_note": (
            "Variants are fixed before this script runs and use only GT-free bridge/semantic/broad/rank "
            "features for acceptance. Diagnostic GT labels are used only to score recall and false bridge."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_csv(out / "pre_registered_variant_rows.csv", variant_rows)
    _write_csv(out / "pre_registered_variant_summary_rows.csv", summary_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "missing_rows.csv", missing_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
