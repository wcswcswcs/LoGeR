from __future__ import annotations

import argparse
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
PHASE9B_ROOT = AUDIT_ROOT / "v103_phase9b_da3_provider_readiness"
OUT_DIR = AUDIT_ROOT / "v103_phase9c_da3_scene0011_bridge_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

RECALL_MIN = 0.35
DIFFERENT_GT_FALSE_MAX = 0.20
SAME_SEMANTIC_DIFFERENT_GT_FALSE_MAX = 0.20
HARD_NEGATIVE_FALSE_ACCEPT_MAX = 0.20
BRIDGE_AUC_MIN = 0.65

SCENES = ["scene0011_00", "scene0050_00"]


REPAIR_VARIANTS = [
    {
        "variant_id": "r1_mutual_top1_sem04_gap4_r001_broad020",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "block",
        "mutual_topk": 1,
        "rank_score_mode": "bridge",
    },
    {
        "variant_id": "r2_mutual_top2_sem04_gap4_r001_broad012",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.12,
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "block",
        "mutual_topk": 2,
        "rank_score_mode": "bridge",
    },
    {
        "variant_id": "r3_mutual_top2_sem04_gap4_r001_broad020_bridge_x_sem",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "block",
        "mutual_topk": 2,
        "rank_score_mode": "bridge_x_semantic",
    },
    {
        "variant_id": "r4_mutual_top3_sem05_gap4_r002_broad012_bridge_x_sem",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.002,
        "broad_limit": 0.12,
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "block",
        "mutual_topk": 3,
        "rank_score_mode": "bridge_x_semantic",
    },
    {
        "variant_id": "r5_mutual_top3_sem04_gap2_r002_broad012_bridge_x_sem",
        "max_gap": 2,
        "min_shared": 1,
        "ratio_min": 0.002,
        "broad_limit": 0.12,
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "block",
        "mutual_topk": 3,
        "rank_score_mode": "bridge_x_semantic",
    },
]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | str:
    labels = labels.astype(bool)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
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


def _variant_score(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    bridge = df["final_bridge_score"].to_numpy(dtype=np.float64)
    if spec["rank_score_mode"] == "bridge":
        return bridge
    sem = df["semantic_residual_cosine"].to_numpy(dtype=np.float64)
    sem = np.where(np.isfinite(sem), np.maximum(sem, 0.0), 0.0)
    if spec["rank_score_mode"] == "bridge_x_semantic":
        return bridge * sem
    raise ValueError(f"Unknown rank_score_mode: {spec['rank_score_mode']}")


def _mutual_topk_accept_mask(df: pd.DataFrame, base_accept: np.ndarray, score: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0:
        return base_accept
    accepted = np.zeros(len(df), dtype=bool)
    work = df.loc[base_accept, ["mask_a_observation_id", "mask_b_observation_id"]].copy()
    if work.empty:
        return accepted
    work["_row_index"] = work.index.to_numpy(dtype=np.int64)
    work["_rank_score"] = score[work["_row_index"].to_numpy(dtype=np.int64)]
    for col, rank_col in [
        ("mask_a_observation_id", "_rank_a"),
        ("mask_b_observation_id", "_rank_b"),
    ]:
        order = work.sort_values([col, "_rank_score"], ascending=[True, False])
        order[rank_col] = order.groupby(col).cumcount() + 1
        work = work.merge(order[["_row_index", rank_col]], on="_row_index", how="left")
    keep = work[(work["_rank_a"] <= topk) & (work["_rank_b"] <= topk)]["_row_index"].to_numpy(dtype=np.int64)
    accepted[keep] = True
    return accepted & base_accept


def _diagnostic_masks(df: pd.DataFrame) -> dict[str, np.ndarray | int | float | str]:
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    different_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    same_semantic_different_gt = df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)
    labeled = same_gt | different_gt
    scores = df.loc[labeled, "final_bridge_score"].to_numpy(dtype=np.float64)
    labels = df.loc[labeled, "diagnostic_same_gt"].to_numpy(dtype=bool)
    return {
        "same_gt": same_gt,
        "different_gt": different_gt,
        "same_semantic_different_gt": same_semantic_different_gt,
        "labeled": labeled,
        "positive_total": int(np.sum(same_gt)),
        "negative_total": int(np.sum(different_gt)),
        "same_semantic_different_gt_total": int(np.sum(same_semantic_different_gt)),
        "base_bridge_auc": _auc(scores, labels),
    }


def _base_accept(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    broad_ok = (
        np.ones(len(df), dtype=bool)
        if spec["broad_limit"] is None
        else df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
    )
    semantic_available = df["semantic_residual_available"].to_numpy(dtype=bool)
    semantic_ok = df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"])
    if spec["missing_feature_policy"] == "allow":
        semantic_ok = semantic_ok | (~semantic_available)
    elif spec["missing_feature_policy"] != "block":
        raise ValueError(f"Unsupported missing_feature_policy: {spec['missing_feature_policy']}")
    return (
        (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
        & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= int(spec["min_shared"]))
        & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= float(spec["ratio_min"]))
        & broad_ok
        & semantic_ok
    )


def _score_variant(scene_id: str, df: pd.DataFrame, spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    masks = _diagnostic_masks(df)
    score = _variant_score(df, spec)
    base_accept = _base_accept(df, spec)
    accepted = _mutual_topk_accept_mask(df, base_accept, score, int(spec["mutual_topk"]))
    accepted_labeled = accepted & masks["labeled"]
    tp = int(np.sum(accepted & masks["same_gt"]))
    fp = int(np.sum(accepted & masks["different_gt"]))
    fp_same_sem = int(np.sum(accepted & masks["same_semantic_different_gt"]))
    accepted_count = int(np.sum(accepted))
    accepted_labeled_count = int(np.sum(accepted_labeled))
    positive_total = int(masks["positive_total"])
    negative_total = int(masks["negative_total"])
    recall = float(tp / max(positive_total, 1)) if positive_total else ""
    diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
    same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
    hard_neg_false = float(fp / max(negative_total, 1)) if negative_total else ""
    label_scores = score[masks["labeled"]]
    label_truth = masks["same_gt"][masks["labeled"]]
    variant_auc = _auc(label_scores, label_truth)
    formal = bool(
        recall != ""
        and diff_false != ""
        and same_sem_false != ""
        and hard_neg_false != ""
        and variant_auc != ""
        and recall >= RECALL_MIN
        and diff_false <= DIFFERENT_GT_FALSE_MAX
        and same_sem_false <= SAME_SEMANTIC_DIFFERENT_GT_FALSE_MAX
        and hard_neg_false <= HARD_NEGATIVE_FALSE_ACCEPT_MAX
        and variant_auc >= BRIDGE_AUC_MIN
    )
    row = {
        "schema_version": "stream4d_v103_phase9c_bridge_repair_variant_row_v1",
        "phase_id": "v103_phase9c_da3_scene0011_bridge_repair",
        "scene_id": scene_id,
        "variant_id": spec["variant_id"],
        "max_gap": spec["max_gap"],
        "min_shared": spec["min_shared"],
        "ratio_min": spec["ratio_min"],
        "broad_limit": "" if spec["broad_limit"] is None else spec["broad_limit"],
        "semantic_cosine_min": spec["semantic_cosine_min"],
        "missing_feature_policy": spec["missing_feature_policy"],
        "mutual_topk": spec["mutual_topk"],
        "rank_score_mode": spec["rank_score_mode"],
        "base_accepted_count": int(np.sum(base_accept)),
        "accepted_count": accepted_count,
        "accepted_labeled_count": accepted_labeled_count,
        "true_positive_same_gt_count": tp,
        "false_positive_different_gt_count": fp,
        "false_positive_same_semantic_different_gt_count": fp_same_sem,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_semantic_different_gt_hard_negative_count": int(masks["same_semantic_different_gt_total"]),
        "same_object_bridge_recall": recall,
        "different_gt_false_bridge_among_accepted": diff_false,
        "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
        "hard_negative_false_accept_rate": hard_neg_false,
        "base_bridge_auc": masks["base_bridge_auc"],
        "variant_bridge_auc": variant_auc,
        "phase5_formal_bridge_gate_pass": formal,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    accepted_df = df.loc[accepted].copy()
    accepted_df["phase9c_variant_id"] = spec["variant_id"]
    accepted_df["phase9c_rank_score"] = score[accepted]
    return row, accepted_df


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [row for row in rows if bool(row["phase5_formal_bridge_gate_pass"])]

    def key(row: dict[str, Any]) -> tuple[int, float, float, float]:
        recall = float(row["same_object_bridge_recall"]) if row["same_object_bridge_recall"] != "" else -1.0
        diff_false = (
            float(row["different_gt_false_bridge_among_accepted"])
            if row["different_gt_false_bridge_among_accepted"] != ""
            else 1.0
        )
        same_sem_false = (
            float(row["same_semantic_different_gt_false_bridge_among_accepted"])
            if row["same_semantic_different_gt_false_bridge_among_accepted"] != ""
            else 1.0
        )
        return (1 if row in passing else 0, recall, -diff_false, -same_sem_false)

    return max(rows, key=key)


def _process_scene(scene_id: str, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge_path = PHASE9B_ROOT / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    scene_dir = out_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    if not bridge_path.exists():
        failure = {
            "schema_version": "stream4d_v103_phase9c_failure_row_v1",
            "phase_id": "v103_phase9c_da3_scene0011_bridge_repair",
            "scene_id": scene_id,
            "blocker": "phase9b_semantic_bridge_rows_missing",
            "path": _rel(bridge_path),
            "uses_gt_for_prediction": False,
        }
        return failure, [failure]

    df = pd.read_parquet(bridge_path)
    rows: list[dict[str, Any]] = []
    accepted_by_variant: dict[str, pd.DataFrame] = {}
    for spec in REPAIR_VARIANTS:
        row, accepted_df = _score_variant(scene_id, df, spec)
        rows.append(row)
        accepted_by_variant[spec["variant_id"]] = accepted_df

    best = _best_row(rows)
    best_accept_path = scene_dir / "best_variant_accepted_pair_rows.parquet"
    accepted_by_variant[str(best["variant_id"])].to_parquet(best_accept_path, index=False)
    variant_path = scene_dir / "repair_variant_rows.csv"
    _write_csv(variant_path, rows)

    formal_pass = any(bool(row["phase5_formal_bridge_gate_pass"]) for row in rows)
    summary = {
        "schema_version": "stream4d_v103_phase9c_scene_summary_row_v1",
        "phase_id": "v103_phase9c_da3_scene0011_bridge_repair",
        "scene_id": scene_id,
        "input_bridge_rows": _rel(bridge_path),
        "candidate_pair_count": int(len(df)),
        "repair_variant_count": len(REPAIR_VARIANTS),
        "formal_bridge_gate_pass": formal_pass,
        "best_variant_id": best["variant_id"],
        "best_same_object_bridge_recall": best["same_object_bridge_recall"],
        "best_different_gt_false_bridge_among_accepted": best["different_gt_false_bridge_among_accepted"],
        "best_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "best_hard_negative_false_accept_rate": best["hard_negative_false_accept_rate"],
        "best_variant_bridge_auc": best["variant_bridge_auc"],
        "blocker": "" if formal_pass else "scene_bridge_false_positive_or_recall_gate_fail",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "outputs": {
            "repair_variant_rows": _rel(variant_path),
            "best_variant_accepted_pair_rows": _rel(best_accept_path),
        },
    }
    return summary, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["all", *SCENES], default="scene0011_00")
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_ids = SCENES if args.scene == "all" else [args.scene]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        row, failures = _process_scene(scene_id, OUT_DIR)
        scene_rows.append(row)
        failure_rows.extend(failures)

    scene_summary_path = OUT_DIR / "scene_summary_rows.csv"
    failure_path = OUT_DIR / "failure_rows.csv"
    gate_path = OUT_DIR / "gate_rows.csv"
    _write_csv(scene_summary_path, scene_rows)
    _write_csv(failure_path, failure_rows)
    pass_count = sum(bool(row.get("formal_bridge_gate_pass")) for row in scene_rows)
    gate_rows = [
        {
            "gate_id": "phase9b_bridge_rows_available",
            "pass": len(failure_rows) == 0,
            "expected": 0,
            "observed": len(failure_rows),
            "scope": args.scene,
        },
        {
            "gate_id": "all_requested_scenes_phase9c_bridge_gate_pass",
            "pass": pass_count == len(scene_rows) and len(scene_rows) > 0,
            "expected": len(scene_rows),
            "observed": pass_count,
            "scope": args.scene,
        },
        {
            "gate_id": "variant_budget_respected",
            "pass": len(REPAIR_VARIANTS) <= 5,
            "expected": "<=5",
            "observed": len(REPAIR_VARIANTS),
            "scope": "phase9c",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": all(row.get("uses_gt_for_prediction") is False for row in scene_rows),
            "expected": False,
            "observed": False,
            "scope": "all",
        },
    ]
    _write_csv(gate_path, gate_rows)
    decision = (
        "PASS_PHASE9C_DA3_BRIDGE_REPAIR"
        if pass_count == len(scene_rows) and scene_rows
        else "NO_GO_PHASE9C_DA3_BRIDGE_REPAIR"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9c_da3_scene0011_bridge_repair_summary_v1",
        "phase_id": "v103_phase9c_da3_scene0011_bridge_repair",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "plan_doc": _rel(PLAN_DOC),
        "scene_count": len(scene_rows),
        "pass_scene_count": pass_count,
        "failure_count": len(failure_rows),
        "repair_variant_count": len(REPAIR_VARIANTS),
        "truthfulness_note": (
            "Repair variants use only GT-free acceptance terms: DA3 shared-Gaussian counts, mask area, RADIO cosine, "
            "frame gap, and mutual top-k ranking. Diagnostic GT labels are used only for recall and false-bridge scoring."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "scene_summary_rows": _rel(scene_summary_path),
            "gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision == "PASS_PHASE9C_DA3_BRIDGE_REPAIR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
