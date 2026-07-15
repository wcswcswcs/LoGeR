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
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9c_da3_bridge_upper_bound_r1"
PLAN_DOC = ROOT / "docs/stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

RECALL_MIN = 0.35
FALSE_MAX = 0.20
SCENES = ["scene0011_00", "scene0050_00"]


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
            writer.writerow({k: _jsonable(row.get(k, "")) for k in fields})


def _thresholds(values: np.ndarray, limit: int = 1000) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    vals = np.unique(values)
    if vals.size > limit:
        vals = np.unique(np.quantile(vals, np.linspace(0, 1, limit + 1)))
    return vals


def _metrics(acc: np.ndarray, same_gt: np.ndarray, diff_gt: np.ndarray) -> dict[str, Any]:
    labeled = same_gt | diff_gt
    lab_acc = acc & labeled
    tp = int(np.count_nonzero(acc & same_gt))
    fp = int(np.count_nonzero(acc & diff_gt))
    positive_total = int(np.count_nonzero(same_gt))
    negative_total = int(np.count_nonzero(diff_gt))
    accepted_labeled = int(np.count_nonzero(lab_acc))
    recall = float(tp / max(1, positive_total))
    false_among = float(fp / max(1, accepted_labeled)) if accepted_labeled else 0.0
    hard_false = float(fp / max(1, negative_total))
    return {
        "accepted_count": int(np.count_nonzero(acc)),
        "accepted_labeled_count": accepted_labeled,
        "true_positive_same_gt_count": tp,
        "false_positive_different_gt_count": fp,
        "same_object_bridge_recall": recall,
        "different_gt_false_bridge_among_accepted": false_among,
        "hard_negative_false_accept_rate": hard_false,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
    }


def _best_one_dim(
    *,
    scene_id: str,
    base_name: str,
    score_name: str,
    score: np.ndarray,
    base: np.ndarray,
    same_gt: np.ndarray,
    diff_gt: np.ndarray,
) -> dict[str, Any]:
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for threshold in _thresholds(score[base]):
        acc = base & (score >= float(threshold))
        m = _metrics(acc, same_gt, diff_gt)
        key = (
            m["different_gt_false_bridge_among_accepted"] <= FALSE_MAX,
            m["same_object_bridge_recall"],
            -m["different_gt_false_bridge_among_accepted"],
            -m["hard_negative_false_accept_rate"],
            m["accepted_labeled_count"],
        )
        row = {
            "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_row_v1",
            "phase_id": "v103_phase9c_da3_bridge_upper_bound",
            "scene_id": scene_id,
            "base_filter_id": base_name,
            "score_family": score_name,
            "semantic_threshold": "",
            "score_threshold": float(threshold),
            **m,
            "passes_formal_recall_false_gate": bool(
                m["same_object_bridge_recall"] >= RECALL_MIN
                and m["different_gt_false_bridge_among_accepted"] <= FALSE_MAX
            ),
            "diagnostic_only_uses_gt_for_threshold_search": True,
            "uses_gt_for_prediction": False,
        }
        if best is None or key > best[0]:
            best = (key, row)
    if best is None:
        return {
            "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_row_v1",
            "phase_id": "v103_phase9c_da3_bridge_upper_bound",
            "scene_id": scene_id,
            "base_filter_id": base_name,
            "score_family": score_name,
            "blocker": "no_finite_score_rows",
            "diagnostic_only_uses_gt_for_threshold_search": True,
            "uses_gt_for_prediction": False,
        }
    return best[1]


def _best_two_dim(
    *,
    scene_id: str,
    base_name: str,
    score_name: str,
    score: np.ndarray,
    semantic: np.ndarray,
    base: np.ndarray,
    same_gt: np.ndarray,
    diff_gt: np.ndarray,
) -> dict[str, Any]:
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for sem_t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sem_ok = np.nan_to_num(semantic, nan=-1.0) >= sem_t
        local = base & sem_ok
        for threshold in _thresholds(score[local], limit=250):
            acc = local & (score >= float(threshold))
            m = _metrics(acc, same_gt, diff_gt)
            key = (
                m["different_gt_false_bridge_among_accepted"] <= FALSE_MAX,
                m["same_object_bridge_recall"],
                -m["different_gt_false_bridge_among_accepted"],
                -m["hard_negative_false_accept_rate"],
                m["accepted_labeled_count"],
            )
            row = {
                "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_row_v1",
                "phase_id": "v103_phase9c_da3_bridge_upper_bound",
                "scene_id": scene_id,
                "base_filter_id": base_name,
                "score_family": f"{score_name}_plus_semantic_threshold",
                "semantic_threshold": float(sem_t),
                "score_threshold": float(threshold),
                **m,
                "passes_formal_recall_false_gate": bool(
                    m["same_object_bridge_recall"] >= RECALL_MIN
                    and m["different_gt_false_bridge_among_accepted"] <= FALSE_MAX
                ),
                "diagnostic_only_uses_gt_for_threshold_search": True,
                "uses_gt_for_prediction": False,
            }
            if best is None or key > best[0]:
                best = (key, row)
    if best is None:
        return {
            "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_row_v1",
            "phase_id": "v103_phase9c_da3_bridge_upper_bound",
            "scene_id": scene_id,
            "base_filter_id": base_name,
            "score_family": f"{score_name}_plus_semantic_threshold",
            "blocker": "no_finite_score_rows",
            "diagnostic_only_uses_gt_for_threshold_search": True,
            "uses_gt_for_prediction": False,
        }
    return best[1]


def _process_scene(scene_id: str, phase9b_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = phase9b_root / scene_id / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    if not path.exists():
        return [], {
            "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_scene_summary_v1",
            "phase_id": "v103_phase9c_da3_bridge_upper_bound",
            "scene_id": scene_id,
            "blocker": "input_missing",
            "input_path": _rel(path),
        }
    df = pd.read_parquet(path)
    same_gt = df["diagnostic_same_gt"].to_numpy(dtype=bool)
    diff_gt = df["diagnostic_different_gt"].to_numpy(dtype=bool)
    semantic = df["semantic_residual_cosine"].to_numpy(dtype=np.float64)
    bridge = df["final_bridge_score"].to_numpy(dtype=np.float64)
    ratio_union = df["gs_bridge_ratio_union"].to_numpy(dtype=np.float64)
    ratio_min = df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64)
    shared = df["gs_shared_gaussian_count"].to_numpy(dtype=np.float64)
    sem_pos = np.maximum(np.nan_to_num(semantic, nan=0.0), 0.0)
    features = {
        "final_bridge_score": bridge,
        "shared_count": shared,
        "ratio_min": ratio_min,
        "ratio_union": ratio_union,
        "semantic": semantic,
        "bridge_x_semantic": bridge * sem_pos,
        "ratio_min_x_semantic": ratio_min * sem_pos,
    }
    base_filters = {
        "all_rows": np.ones(len(df), dtype=bool),
        "gap_le4_semantic_available_broad_le020": (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
            & df["semantic_residual_available"].to_numpy(dtype=bool)
            & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
        ),
        "gap_le2_semantic_available_broad_le020": (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 2)
            & df["semantic_residual_available"].to_numpy(dtype=bool)
            & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
        ),
        "gap_le4_semantic_available_broad_le012": (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
            & df["semantic_residual_available"].to_numpy(dtype=bool)
            & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.12)
        ),
    }
    rows: list[dict[str, Any]] = []
    for base_name, base in base_filters.items():
        for score_name, score in features.items():
            rows.append(
                _best_one_dim(
                    scene_id=scene_id,
                    base_name=base_name,
                    score_name=score_name,
                    score=score,
                    base=base,
                    same_gt=same_gt,
                    diff_gt=diff_gt,
                )
            )
        for score_name in ["final_bridge_score", "ratio_union", "bridge_x_semantic", "ratio_min_x_semantic"]:
            rows.append(
                _best_two_dim(
                    scene_id=scene_id,
                    base_name=base_name,
                    score_name=score_name,
                    score=features[score_name],
                    semantic=semantic,
                    base=base,
                    same_gt=same_gt,
                    diff_gt=diff_gt,
                )
            )
    formal_rows = [r for r in rows if bool(r.get("passes_formal_recall_false_gate", False))]
    best = max(
        rows,
        key=lambda r: (
            bool(r.get("passes_formal_recall_false_gate", False)),
            float(r.get("same_object_bridge_recall", -1.0) or -1.0),
            -float(r.get("different_gt_false_bridge_among_accepted", 1.0) or 1.0),
        ),
    )
    summary = {
        "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_scene_summary_v1",
        "phase_id": "v103_phase9c_da3_bridge_upper_bound",
        "scene_id": scene_id,
        "input_rows": int(len(df)),
        "diagnostic_positive_pair_count": int(np.count_nonzero(same_gt)),
        "diagnostic_negative_pair_count": int(np.count_nonzero(diff_gt)),
        "upper_bound_family_count": int(len(rows)),
        "formal_upper_bound_exists": bool(len(formal_rows) > 0),
        "best_score_family": best.get("score_family", ""),
        "best_base_filter_id": best.get("base_filter_id", ""),
        "best_same_object_bridge_recall": best.get("same_object_bridge_recall", ""),
        "best_different_gt_false_bridge_among_accepted": best.get("different_gt_false_bridge_among_accepted", ""),
        "best_score_threshold": best.get("score_threshold", ""),
        "best_semantic_threshold": best.get("semantic_threshold", ""),
        "blocker": "" if formal_rows else "no_threshold_family_reaches_recall_0p35_with_false_bridge_le_0p20",
        "diagnostic_only_uses_gt_for_threshold_search": True,
        "uses_gt_for_prediction": False,
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GT-only upper-bound diagnostic for v103 Phase9c DA3 bridge separability.")
    parser.add_argument("--scene", choices=["all", *SCENES], default="scene0011_00")
    parser.add_argument("--phase9b-root", default=str(DEFAULT_PHASE9B))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = STREAM3D / args.output_root if not str(args.output_root).startswith("Stream3D/") else ROOT / args.output_root
    phase9b_root = Path(args.phase9b_root)
    if not phase9b_root.is_absolute():
        phase9b_root = STREAM3D / args.phase9b_root if not str(args.phase9b_root).startswith("Stream3D/") else ROOT / args.phase9b_root
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    scene_ids = SCENES if args.scene == "all" else [args.scene]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_rows, scene_summary = _process_scene(scene_id, phase9b_root)
        rows.extend(scene_rows)
        summaries.append(scene_summary)
    _write_csv(out / "upper_bound_rows.csv", rows)
    _write_csv(out / "scene_summary_rows.csv", summaries)
    pass_count = sum(bool(s.get("formal_upper_bound_exists", False)) for s in summaries)
    summary = {
        "schema_version": "stream4d_v103_phase9c_bridge_upper_bound_summary_v1",
        "phase_id": "v103_phase9c_da3_bridge_upper_bound",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "scene_count": len(summaries),
        "formal_upper_bound_scene_count": pass_count,
        "decision": "PASS_DIAGNOSTIC_UPPER_BOUND_EXISTS" if pass_count == len(summaries) else "NO_GO_DIAGNOSTIC_UPPER_BOUND_NOT_FOUND",
        "plan_doc": _rel(PLAN_DOC),
        "truthfulness_note": "This diagnostic searches thresholds with GT labels only to estimate separability upper bounds. It does not select method thresholds or produce predictions.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_threshold_search": True,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "upper_bound_rows": _rel(out / "upper_bound_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"] == "PASS_DIAGNOSTIC_UPPER_BOUND_EXISTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
