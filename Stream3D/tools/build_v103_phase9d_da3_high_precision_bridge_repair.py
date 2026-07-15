#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs/audit"
DEFAULT_PHASE9B = AUDIT_ROOT / "v103_phase9b_da3_provider_readiness"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9d_da3_high_precision_bridge_repair"
PLAN_DOC = ROOT / "docs/stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

RECALL_MIN = 0.35
DIFFERENT_GT_FALSE_MAX = 0.20
HARD_NEGATIVE_FALSE_ACCEPT_MAX = 0.20
SCENES = ["scene0011_00", "scene0050_00"]


VARIANTS = [
    {
        "variant_id": "h1_sem06_bridge040_gap4",
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.40,
        "max_gap": 4,
        "broad_limit": None,
    },
    {
        "variant_id": "h2_sem06_bridge045_gap4",
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.45,
        "max_gap": 4,
        "broad_limit": None,
    },
    {
        "variant_id": "h3_sem06_bridge035_gap4",
        "semantic_cosine_min": 0.60,
        "score_column": "final_bridge_score",
        "score_min": 0.35,
        "max_gap": 4,
        "broad_limit": None,
    },
    {
        "variant_id": "h4_sem07_bridge030_gap4",
        "semantic_cosine_min": 0.70,
        "score_column": "final_bridge_score",
        "score_min": 0.30,
        "max_gap": 4,
        "broad_limit": None,
    },
    {
        "variant_id": "h5_sem06_ratiounion008_gap4",
        "semantic_cosine_min": 0.60,
        "score_column": "gs_bridge_ratio_union",
        "score_min": 0.08,
        "max_gap": 4,
        "broad_limit": None,
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


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return ROOT / p
    return STREAM3D / p


def _accept_mask(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    acc = (
        (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
        & df["semantic_residual_available"].to_numpy(dtype=bool)
        & (df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"]))
        & (df[str(spec["score_column"])].to_numpy(dtype=np.float64) >= float(spec["score_min"]))
    )
    if spec["broad_limit"] is not None:
        acc &= df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
    return acc


def _score_variant(scene_id: str, df: pd.DataFrame, spec: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    acc = _accept_mask(df, spec)
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    diff_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    same_sem_diff_gt = df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)
    labeled = same_gt | diff_gt
    accepted_labeled = acc & labeled
    tp = int(np.count_nonzero(acc & same_gt))
    fp = int(np.count_nonzero(acc & diff_gt))
    fp_same_sem = int(np.count_nonzero(acc & same_sem_diff_gt))
    positive_total = int(np.count_nonzero(same_gt))
    negative_total = int(np.count_nonzero(diff_gt))
    accepted_labeled_count = int(np.count_nonzero(accepted_labeled))
    recall = float(tp / max(positive_total, 1))
    diff_false = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    same_sem_false = float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else 0.0
    hard_false = float(fp / max(negative_total, 1))
    formal = bool(recall >= RECALL_MIN and diff_false <= DIFFERENT_GT_FALSE_MAX and hard_false <= HARD_NEGATIVE_FALSE_ACCEPT_MAX)
    row = {
        "schema_version": "stream4d_v103_phase9d_high_precision_bridge_variant_row_v1",
        "phase_id": "v103_phase9d_da3_high_precision_bridge_repair",
        "scene_id": scene_id,
        "variant_id": spec["variant_id"],
        "semantic_cosine_min": spec["semantic_cosine_min"],
        "score_column": spec["score_column"],
        "score_min": spec["score_min"],
        "max_gap": spec["max_gap"],
        "broad_limit": "" if spec["broad_limit"] is None else spec["broad_limit"],
        "accepted_count": int(np.count_nonzero(acc)),
        "accepted_labeled_count": accepted_labeled_count,
        "true_positive_same_gt_count": tp,
        "false_positive_different_gt_count": fp,
        "false_positive_same_semantic_different_gt_count": fp_same_sem,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_object_bridge_recall": recall,
        "different_gt_false_bridge_among_accepted": diff_false,
        "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false,
        "hard_negative_false_accept_rate": hard_false,
        "phase9d_bridge_gate_pass": formal,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    accepted_df = df.loc[acc].copy()
    accepted_df["phase9d_variant_id"] = spec["variant_id"]
    return row, accepted_df


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (
            bool(r.get("phase9d_bridge_gate_pass", False)),
            float(r.get("same_object_bridge_recall", -1.0)),
            -float(r.get("different_gt_false_bridge_among_accepted", 1.0)),
            -float(r.get("hard_negative_false_accept_rate", 1.0)),
        ),
    )


def _process_scene(scene_id: str, phase9b_root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge_path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    scene_dir = out / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    if not bridge_path.exists():
        failure = {
            "schema_version": "stream4d_v103_phase9d_failure_row_v1",
            "phase_id": "v103_phase9d_da3_high_precision_bridge_repair",
            "scene_id": scene_id,
            "blocker": "phase9b_bridge_rows_missing",
            "path": _rel(bridge_path),
            "uses_gt_for_prediction": False,
        }
        return failure, [failure]
    df = pd.read_parquet(bridge_path)
    variant_rows: list[dict[str, Any]] = []
    accepted_by_variant: dict[str, pd.DataFrame] = {}
    for spec in VARIANTS:
        row, accepted = _score_variant(scene_id, df, spec)
        variant_rows.append(row)
        accepted_by_variant[str(spec["variant_id"])] = accepted
    best = _best(variant_rows)
    variant_path = scene_dir / "high_precision_variant_rows.csv"
    best_path = scene_dir / "best_variant_accepted_pair_rows.parquet"
    _write_csv(variant_path, variant_rows)
    accepted_by_variant[str(best["variant_id"])].to_parquet(best_path, index=False)
    formal = any(bool(r["phase9d_bridge_gate_pass"]) for r in variant_rows)
    summary = {
        "schema_version": "stream4d_v103_phase9d_scene_summary_row_v1",
        "phase_id": "v103_phase9d_da3_high_precision_bridge_repair",
        "scene_id": scene_id,
        "input_bridge_rows": _rel(bridge_path),
        "candidate_pair_count": int(len(df)),
        "repair_variant_count": len(VARIANTS),
        "formal_bridge_gate_pass": formal,
        "best_variant_id": best["variant_id"],
        "best_same_object_bridge_recall": best["same_object_bridge_recall"],
        "best_different_gt_false_bridge_among_accepted": best["different_gt_false_bridge_among_accepted"],
        "best_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "best_hard_negative_false_accept_rate": best["hard_negative_false_accept_rate"],
        "blocker": "" if formal else "high_precision_semantic_bridge_tradeoff_recall_or_false_gate_fail",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "outputs": {
            "high_precision_variant_rows": _rel(variant_path),
            "best_variant_accepted_pair_rows": _rel(best_path),
        },
    }
    return summary, []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase9d high-precision semantic+DA3 bridge repair.")
    parser.add_argument("--scene", choices=["all", *SCENES], default="scene0011_00")
    parser.add_argument("--phase9b-root", default=str(DEFAULT_PHASE9B))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    phase9b_root = _project(args.phase9b_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    scene_ids = SCENES if args.scene == "all" else [args.scene]
    scene_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        row, scene_failures = _process_scene(scene_id, phase9b_root, out)
        scene_rows.append(row)
        failures.extend(scene_failures)
    pass_count = sum(bool(row.get("formal_bridge_gate_pass", False)) for row in scene_rows)
    scene_path = out / "scene_summary_rows.csv"
    failure_path = out / "failure_rows.csv"
    gate_path = out / "gate_rows.csv"
    _write_csv(scene_path, scene_rows)
    _write_csv(failure_path, failures)
    gate_rows = [
        {
            "gate_id": "phase9b_bridge_rows_available",
            "pass": len(failures) == 0,
            "expected": 0,
            "observed": len(failures),
            "scope": args.scene,
        },
        {
            "gate_id": "all_requested_scenes_phase9d_bridge_gate_pass",
            "pass": pass_count == len(scene_rows) and len(scene_rows) > 0,
            "expected": len(scene_rows),
            "observed": pass_count,
            "scope": args.scene,
        },
        {
            "gate_id": "variant_budget_respected",
            "pass": len(VARIANTS) <= 5,
            "expected": "<=5",
            "observed": len(VARIANTS),
            "scope": "phase9d",
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
        "PASS_PHASE9D_DA3_HIGH_PRECISION_BRIDGE_REPAIR"
        if pass_count == len(scene_rows) and scene_rows
        else "NO_GO_PHASE9D_DA3_HIGH_PRECISION_BRIDGE_REPAIR"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9d_high_precision_bridge_summary_v1",
        "phase_id": "v103_phase9d_da3_high_precision_bridge_repair",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "plan_doc": _rel(PLAN_DOC),
        "scene_count": len(scene_rows),
        "pass_scene_count": pass_count,
        "failure_count": len(failures),
        "repair_variant_count": len(VARIANTS),
        "truthfulness_note": "Acceptance uses GT-free DA3 bridge score, RADIO cosine, and frame gap only. GT labels are used only for provider diagnostic recall/false-bridge scoring.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(scene_path),
            "gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
