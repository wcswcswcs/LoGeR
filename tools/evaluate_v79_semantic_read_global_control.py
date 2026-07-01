#!/usr/bin/env python3
"""Evaluate ACL2 v79 Phase2 semantic READ/global-attention runs.

The evaluator is deliberately conservative: a local ATE win by itself is kept
as diagnostic-only. A Phase2 gate pass requires a lower-is-better mechanism
metric, currently head-to-tail Sim(3) transfer or head/mid/tail scale CV, to
improve by at least 10% vs native and beat all required semantic/geometry/random
controls on the same metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (  # noqa: E402
    _eval_run,
    _finite,
    _load_kitti_gt,
    _safe_ratio_improvement,
)


DEFAULT_BASE_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase2_semantic_read_global_control/rollouts"
)
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_CANDIDATES = [
    "READ1_L07_SEMANTIC_LAYOUT_SELECT",
    "READ2_L13_SEMANTIC_VALUE_DAMP",
    "READ3_L13_STABLE_PROTECT",
    "READ4_L07_TO_L13_SEMANTIC_CONTRAST",
    "READ5_FRAME_L18_SEMANTIC_TAIL_STABILIZE",
]
DEFAULT_CONTROLS = [
    "READ6_GEOMETRY_ONLY_CONTROL",
    "READ7_LABEL_SHUFFLE",
    "READ8_CONFIDENCE_SHUFFLE",
    "READ9_SAME_READ_MASS_RANDOM",
    "READ10_GROUP_STRATIFIED_RANDOM",
]
LOCAL_KEY = "local_sim3_ate_rmse_m"
MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]
SECONDARY_KEYS = [
    "overlap3_to_future_pose_sim3_rmse_m",
]
LOWER_IS_BETTER_KEYS = [LOCAL_KEY] + MECHANISM_KEYS + SECONDARY_KEYS


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _parse_csv_names(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _control_values(rows_by_name: dict[str, dict[str, Any]], controls: list[str], key: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for name in controls:
        row = rows_by_name.get(name)
        if row is None:
            missing.append(name)
            continue
        val = _finite(row.get(key))
        if val is None:
            invalid.append(name)
            continue
        values[name] = val
    finite_vals = [float(v) for v in values.values()]
    return {
        "values": values,
        "missing_controls": missing,
        "invalid_control_values": invalid,
        "best_control": float(min(finite_vals)) if finite_vals else None,
        "all_controls_finite": bool(len(values) == len(controls)),
    }


def _compare_key(
    rows_by_name: dict[str, dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: list[str],
    key: str,
    min_improvement: float,
) -> dict[str, Any]:
    cand_row = rows_by_name.get(candidate)
    base_row = rows_by_name.get(baseline)
    cand_v = _finite(cand_row.get(key)) if cand_row is not None else None
    base_v = _finite(base_row.get(key)) if base_row is not None else None
    ctrl = _control_values(rows_by_name, controls, key)
    best_ctrl = _finite(ctrl.get("best_control"))
    ratio = _safe_ratio_improvement(base_v, cand_v)
    beats_controls = bool(cand_v is not None and best_ctrl is not None and ctrl["all_controls_finite"] and cand_v < best_ctrl)
    key_pass = bool(beats_controls and ratio is not None and ratio >= float(min_improvement))
    return {
        "candidate": cand_v,
        "baseline": base_v,
        "best_control": best_ctrl,
        "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
        "candidate_minus_best_control": (cand_v - best_ctrl) if cand_v is not None and best_ctrl is not None else None,
        "improvement_vs_baseline_ratio": ratio,
        "beats_all_required_controls": beats_controls,
        "min_improvement_required": float(min_improvement),
        "key_pass": key_pass,
        "control_detail": ctrl,
    }


def _build_candidate_decision(
    rows_by_name: dict[str, dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: list[str],
) -> dict[str, Any]:
    if candidate not in rows_by_name:
        return {
            "phase2_gate_pass": False,
            "candidate": candidate,
            "reason": f"missing_candidate:{candidate}",
        }
    if baseline not in rows_by_name:
        return {
            "phase2_gate_pass": False,
            "candidate": candidate,
            "reason": f"missing_baseline:{baseline}",
        }

    comparisons: dict[str, Any] = {}
    local_cmp = _compare_key(
        rows_by_name,
        candidate=candidate,
        baseline=baseline,
        controls=controls,
        key=LOCAL_KEY,
        min_improvement=0.05,
    )
    comparisons[LOCAL_KEY] = local_cmp
    mechanism_passes: list[str] = []
    secondary_passes: list[str] = []
    for key in MECHANISM_KEYS:
        cmp = _compare_key(
            rows_by_name,
            candidate=candidate,
            baseline=baseline,
            controls=controls,
            key=key,
            min_improvement=0.10,
        )
        comparisons[key] = cmp
        if cmp["key_pass"]:
            mechanism_passes.append(key)
    for key in SECONDARY_KEYS:
        cmp = _compare_key(
            rows_by_name,
            candidate=candidate,
            baseline=baseline,
            controls=controls,
            key=key,
            min_improvement=0.10,
        )
        comparisons[key] = cmp
        if cmp["key_pass"]:
            secondary_passes.append(key)

    local_support_pass = bool(local_cmp["key_pass"])
    phase2_gate_pass = bool(mechanism_passes)
    return {
        "phase2_gate_pass": phase2_gate_pass,
        "candidate": candidate,
        "baseline": baseline,
        "required_controls": controls,
        "local_support_pass": local_support_pass,
        "diagnostic_only_local_improvement": bool(local_support_pass and not phase2_gate_pass),
        "mechanism_metric_passes": mechanism_passes,
        "secondary_overlap_passes_not_phase2_gate": secondary_passes,
        "comparisons": comparisons,
        "rule": (
            "local ATE requires >=5% improvement vs native and must beat geometry/shuffle/random controls, "
            "but local-only wins are diagnostic-only. Phase2 pass requires head-to-tail Sim3 transfer or "
            "head/mid/tail scale CV to improve >=10% vs native and beat every required control on the same metric."
        ),
    }


def _evaluate_chunk(
    *,
    base_dir: Path,
    chunk: int,
    run_names: list[str],
    baseline: str,
    candidates: list[str],
    controls: list[str],
    gt_poses_all: np.ndarray,
    gt_pos_all: np.ndarray,
) -> dict[str, Any]:
    chunk_dir = base_dir / f"chunk{int(chunk):02d}"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name in run_names:
        run_dir = chunk_dir / name
        try:
            row = _eval_run(name, run_dir, gt_poses_all, gt_pos_all)
            row["chunk"] = int(chunk)
            rows.append(row)
        except Exception as exc:  # keep missing/failed runs auditable
            errors.append({"run": name, "run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    rows_by_name = {str(row["run"]): row for row in rows}
    decisions = {
        candidate: _build_candidate_decision(
            rows_by_name,
            candidate=candidate,
            baseline=baseline,
            controls=controls,
        )
        for candidate in candidates
    }
    return {
        "chunk": int(chunk),
        "chunk_dir": str(chunk_dir),
        "runs": rows,
        "evaluation_errors": errors,
        "decisions": decisions,
        "phase2_any_candidate_gate_pass": bool(any(dec.get("phase2_gate_pass") for dec in decisions.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunks", default="10")
    parser.add_argument("--baseline", default="READ0_NATIVE")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    candidates = args.candidate or DEFAULT_CANDIDATES
    controls = args.control or DEFAULT_CONTROLS
    run_names = [args.baseline] + candidates + controls
    # Keep order stable while removing duplicates.
    run_names = list(dict.fromkeys(run_names))
    out_json = args.out_json or args.base_dir / "phase2_semantic_read_gate_summary.json"
    out_csv = args.out_csv or args.base_dir / "phase2_semantic_read_metrics.csv"

    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    chunks_payload = [
        _evaluate_chunk(
            base_dir=args.base_dir,
            chunk=chunk,
            run_names=run_names,
            baseline=args.baseline,
            candidates=candidates,
            controls=controls,
            gt_poses_all=gt_poses_all,
            gt_pos_all=gt_pos_all,
        )
        for chunk in chunks
    ]
    rows = [row for chunk_payload in chunks_payload for row in chunk_payload["runs"]]
    payload = {
        "base_dir": str(args.base_dir),
        "gt": str(args.gt),
        "chunks": chunks_payload,
        "phase2_gate_pass": bool(any(chunk_payload["phase2_any_candidate_gate_pass"] for chunk_payload in chunks_payload)),
        "method_gate_claimed": False,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)
    print(json.dumps(_jsonable({"phase2_gate_pass": payload["phase2_gate_pass"], "chunks": [
        {
            "chunk": chunk_payload["chunk"],
            "phase2_any_candidate_gate_pass": chunk_payload["phase2_any_candidate_gate_pass"],
            "evaluation_errors": chunk_payload["evaluation_errors"],
            "candidate_gate_pass": {
                name: dec.get("phase2_gate_pass")
                for name, dec in chunk_payload["decisions"].items()
            },
        }
        for chunk_payload in chunks_payload
    ]}), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")


if __name__ == "__main__":
    main()
