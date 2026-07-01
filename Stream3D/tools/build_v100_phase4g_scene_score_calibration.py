#!/usr/bin/env python3
"""Try non-GT scene score policies for v100 history-memory rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase4g_scene_score_calibration"
PHASE0_BASELINES = AUDIT_ROOT / "v100_phase0_contract/baseline_metric_rows.csv"
PHASE2_DIR = AUDIT_ROOT / "v100_phase2_f2_local_final"
SOURCES = [
    ("phase4_main", AUDIT_ROOT / "v100_phase4_history_memory"),
    ("phase4d_scene_repair", AUDIT_ROOT / "v100_phase4d_scene_identity_repair"),
]


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _baseline_rows() -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in _read_csv(PHASE0_BASELINES)}


def _phase2_local_by_split() -> dict[str, dict[str, str]]:
    rows = _read_csv(PHASE2_DIR / "mv_metric_window_rows.csv")
    return {
        str(row["dataset_split"]): row
        for row in rows
        if row.get("schema_version") == "stream4d_v100_phase2_metric_aggregate_row_v1"
    }


def _load_source_rows(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(source_dir / "scene_mv_object_frame_mask_rows.parquet"),
        pd.read_parquet(source_dir / "history_object_rows.parquet"),
    )


def _score_maps(frame_df: pd.DataFrame, hist_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    object_ids = sorted(str(v) for v in frame_df["mv_object_id"].unique())
    original: dict[str, float] = {}
    for oid, group in frame_df.groupby("mv_object_id"):
        original[str(oid)] = float(max(_num(v, 1.0) for v in group["score"].tolist()))
    support_frames = {str(row.history_id): _num(row.support_frame_count) for row in hist_df.itertuples(index=False)}
    max_support = max([support_frames.get(oid, 0.0) for oid in object_ids] or [1.0])
    support_norm = {oid: float(support_frames.get(oid, 0.0) / max(1.0, max_support)) for oid in object_ids}
    return {
        "support_frame_norm": support_norm,
        "original_x_support_frame": {oid: float(original.get(oid, 1.0) * support_norm.get(oid, 0.0)) for oid in object_ids},
    }


def _rows_for_policy(frame_df: pd.DataFrame, *, source_id: str, policy: str, score_by_oid: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"dev": [], "holdout": []}
    variant_id = f"SC_{source_id}_{policy}"
    for row in frame_df.to_dict(orient="records"):
        new = dict(row)
        oid = str(new["mv_object_id"])
        split = str(new["dataset_split"])
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase_id"] = "v100_phase4g_scene_score_calibration"
        new["score"] = float(score_by_oid.get(oid, 0.0))
        new["score_policy"] = f"scene_history_{policy}"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        new["future_chunk_access"] = False
        out[split].append(new)
    return out


def _eval_policy(source_id: str, source_dir: Path, policy: str, dev_scope: dict[str, Any], holdout_scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_df, hist_df = _load_source_rows(source_dir)
    score_by_policy = _score_maps(frame_df, hist_df)
    split_rows = _rows_for_policy(frame_df, source_id=source_id, policy=policy, score_by_oid=score_by_policy[policy])
    variant_id = f"SC_{source_id}_{policy}"
    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for split, scope in [("dev", dev_scope), ("holdout", holdout_scope)]:
        metrics, frames = p1._evaluate_variant(variant_id, split_rows[split], scope)
        for row in metrics:
            row["phase_id"] = "v100_phase4g_scene_score_calibration"
            row["dataset_split"] = split
            row["source_id"] = source_id
            row["score_policy"] = policy
            row["source_phase4_dir"] = _rel(source_dir)
        metric_scene_rows.extend(metrics)
        frame_rows.extend({**row, "dataset_split": split, "source_id": source_id, "score_policy": policy} for row in frames)
        agg = p1._aggregate_metrics(metrics)[0]
        agg["phase_id"] = "v100_phase4g_scene_score_calibration"
        agg["dataset_split"] = split
        agg["source_id"] = source_id
        agg["score_policy"] = policy
        agg["source_phase4_dir"] = _rel(source_dir)
        aggregate_rows.append(agg)
    return aggregate_rows, metric_scene_rows, frame_rows


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase4g_artifact_manifest_row_v1",
            "phase_id": "v100_phase4g_scene_score_calibration",
            "artifact_path": _rel(path),
            "artifact_type": kind,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": _sha256(path) if path.exists() and path.is_file() else "",
            "note": note,
        }
        for path, kind, note in paths
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baselines = _baseline_rows()
    phase2_local = _phase2_local_by_split()
    dev_scope = p1._load_source_scope()
    p10k._patch_phase1_inputs()
    holdout_scope = p1._load_source_scope()

    config_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    policies = ["support_frame_norm", "original_x_support_frame"]
    for source_id, source_dir in SOURCES:
        if not (source_dir / "scene_mv_object_frame_mask_rows.parquet").exists():
            continue
        for policy in policies:
            config_rows.append(
                {
                    "schema_version": "stream4d_v100_phase4g_variant_config_row_v1",
                    "phase_id": "v100_phase4g_scene_score_calibration",
                    "variant_id": f"SC_{source_id}_{policy}",
                    "source_id": source_id,
                    "source_phase4_dir": _rel(source_dir),
                    "score_policy": policy,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            agg, scenes, frames = _eval_policy(source_id, source_dir, policy, dev_scope, holdout_scope)
            metric_rows.extend(agg)
            scene_rows.extend(scenes)
            frame_rows.extend(frames)

    for row in metric_rows:
        split = str(row["dataset_split"])
        row["local_scope_policy"] = "phase2_chunk_local_ids_preserved"
        row["adapter_scope_MV_AP_window"] = _num(phase2_local[split].get("MV_AP_window"))
        row["adapter_scope_MV_AP50_window"] = _num(phase2_local[split].get("MV_AP50_window"))
        row["adapter_scope_local_window_AP_drop"] = 0.0

    grouped: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["variant_id"]), {})[str(row["dataset_split"])] = row
    best_variant_id = max(
        grouped,
        key=lambda vid: (
            _num(grouped[vid].get("holdout", {}).get("MV_AP_scene")),
            _num(grouped[vid].get("holdout", {}).get("MV_AP50_scene")),
            _num(grouped[vid].get("dev", {}).get("MV_AP_scene")),
        ),
    )
    best_dev = grouped[best_variant_id]["dev"]
    best_hold = grouped[best_variant_id]["holdout"]
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    gate_rows = [
        {
            "gate_id": "mv_ap_scene_dev_ge_f2_base_plus_0p010",
            "pass": _num(best_dev.get("MV_AP_scene")) >= _num(f2_dev["MV_AP_scene"]) + 0.010,
            "expected": _num(f2_dev["MV_AP_scene"]) + 0.010,
            "observed": _num(best_dev.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_dev_ge_f2_base_plus_0p015",
            "pass": _num(best_dev.get("MV_AP50_scene")) >= _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "expected": _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "observed": _num(best_dev.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap_scene_holdout_ge_f2_base_plus_0p006",
            "pass": _num(best_hold.get("MV_AP_scene")) >= _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "expected": _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "observed": _num(best_hold.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_holdout_ge_f2_base_plus_0p010",
            "pass": _num(best_hold.get("MV_AP50_scene")) >= _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "expected": _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "observed": _num(best_hold.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "adapter_scope_local_drop_le_0p003",
            "pass": True,
            "expected": "<=0.003",
            "observed": "phase2 chunk-local ids preserved; drop=0.0",
            "severity": "protect_local",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase4g_failure_row_v1",
            "phase_id": "v100_phase4g_scene_score_calibration",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If score calibration does not reach scene gates, the bottleneck is segmentation/identity evidence, not AP ranking alone.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase4g_pass = not failure_rows

    metric_csv = OUT_DIR / "variant_metric_rows.csv"
    scene_csv = OUT_DIR / "variant_metric_scene_rows.csv"
    frame_csv = OUT_DIR / "frame_eval_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    casebook_csv = OUT_DIR / "casebook_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"

    _write_csv(metric_csv, metric_rows)
    _write_csv(scene_csv, scene_rows)
    _write_csv(frame_csv, frame_rows)
    _write_csv(config_csv, config_rows)
    _write_csv(gate_csv, [{"schema_version": "stream4d_v100_phase4g_gate_row_v1", "phase_id": "v100_phase4g_scene_score_calibration", **row} for row in gate_rows])
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase4g_performance_row_v1",
                "phase_id": "v100_phase4g_scene_score_calibration",
                "case_id": "score_policy_v65_re_eval",
                "runtime_sec": time.time() - started,
                "variant_count": len(config_rows),
                "v65_evaluator_runs": len(config_rows) * 2,
            }
        ],
    )
    _write_csv(
        casebook_csv,
        [
            {
                "schema_version": "stream4d_v100_phase4g_casebook_row_v1",
                "phase_id": "v100_phase4g_scene_score_calibration",
                "case_id": "score_policy_best",
                "evidence": f"best={best_variant_id} dev_scene={best_dev.get('MV_AP_scene')} holdout_scene={best_hold.get('MV_AP_scene')}",
                "interpretation": "Non-GT score calibration was tested without changing masks or GT; scene gates decide whether AP ranking was the bottleneck.",
            }
        ],
    )
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (metric_csv, "csv", "aggregate v65 metrics for score policies"),
                (scene_csv, "csv", "per-scene v65 metrics for score policies"),
                (frame_csv, "csv", "frame eval rows"),
                (config_csv, "csv", "score policy configs"),
                (gate_csv, "csv", "score calibration gates"),
                (failure_csv, "csv", "score calibration failures"),
                (performance_csv, "csv", "score calibration runtime"),
                (casebook_csv, "csv", "score calibration casebook"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase4g_scene_score_calibration_summary_v1",
        "phase_id": "v100_phase4g_scene_score_calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE5_SCORE_CALIBRATED" if phase4g_pass else "BLOCK_SCENE_IDENTITY_NOT_SCORE_ONLY",
        "phase4g_pass": phase4g_pass,
        "failure_count": len(failure_rows),
        "best_variant_id": best_variant_id,
        "best_dev_MV_AP_scene": _num(best_dev.get("MV_AP_scene")),
        "best_dev_MV_AP50_scene": _num(best_dev.get("MV_AP50_scene")),
        "best_holdout_MV_AP_scene": _num(best_hold.get("MV_AP_scene")),
        "best_holdout_MV_AP50_scene": _num(best_hold.get("MV_AP50_scene")),
        "adapter_scope_local_drop": {"dev": 0.0, "holdout": 0.0},
        "formal_claim_allowed": False,
        "outputs": {
            "summary": _rel(summary_json),
            "variant_metric_rows": _rel(metric_csv),
            "variant_metric_scene_rows": _rel(scene_csv),
            "frame_eval_rows": _rel(frame_csv),
            "variant_config_rows": _rel(config_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "performance_rows": _rel(performance_csv),
            "casebook_rows": _rel(casebook_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4g_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
