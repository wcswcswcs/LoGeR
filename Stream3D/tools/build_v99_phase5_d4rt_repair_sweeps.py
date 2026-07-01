#!/usr/bin/env python3
"""Repair sweeps for v99 Phase5 D4RT score weight and local2history threshold."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase5_d4rt_repair_sweeps"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE2_SUMMARY = PHASE2_DIR / "best_variant_summary.json"
PHASE5_DIR = AUDIT_ROOT / "v99_phase5_d4rt_anchor_verifier"
EPS_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
TAU_SWEEP = [0.20, 0.30, 0.40, 0.50, 0.60]


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
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


class DSU:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no rows for Phase2 best variant {variant}")
    return variant, rows


def _metrics_from_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        oid = str(row.get("mv_object_id"))
        if oid not in out:
            out[oid] = {
                "overlap": _num(row.get("d4rt_mean_anchor_overlap")),
                "conflict": _num(row.get("d4rt_conflict_count")),
            }
    return out


def _l2h_map(parent_rows: list[dict[str, Any]], tau: float) -> tuple[dict[str, str], int]:
    ids = sorted({str(row["mv_object_id"]) for row in parent_rows})
    dsu = DSU(ids)
    merge_count = 0
    for row in _read_csv(PHASE5_DIR / "local2history_merge_rows.csv"):
        a = str(row.get("mv_object_id_a", ""))
        b = str(row.get("mv_object_id_b", ""))
        if not a or not b:
            continue
        if _num(row.get("object_anchor_overlap")) >= tau and a in dsu.parent and b in dsu.parent:
            dsu.union(a, b)
            merge_count += 1
    return {oid: f"P5R_l2h_tau{tau:.2f}:{dsu.find(oid)}" for oid in ids}, merge_count


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase2_summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    phase5_summary = json.loads((PHASE5_DIR / "summary.json").read_text(encoding="utf-8"))
    parent_variant, parent_rows = _phase2_best_rows()
    scope = p1._load_source_scope()
    phase5_rows = _read_csv(PHASE5_DIR / "mv_object_frame_mask_rows.csv")
    real_rows = [row for row in phase5_rows if row.get("variant_id") == "P5_D1_anchor_boost_top20"]
    shuffled_rows = [row for row in phase5_rows if row.get("variant_id") == "P5_C1_shuffled_D4RT_anchor"]
    real_metrics = _metrics_from_rows(real_rows)
    shuffled_metrics = _metrics_from_rows(shuffled_rows)
    real_overlap_norm = _norm({oid: vals["overlap"] for oid, vals in real_metrics.items()})
    real_conflict_norm = _norm({oid: vals["conflict"] for oid, vals in real_metrics.items()})
    shuffled_overlap_norm = _norm({oid: vals["overlap"] for oid, vals in shuffled_metrics.items()})
    shuffled_conflict_norm = _norm({oid: vals["conflict"] for oid, vals in shuffled_metrics.items()})

    all_rows: list[dict[str, Any]] = []
    variant_meta: list[dict[str, Any]] = []
    for eps in EPS_SWEEP:
        for family, overlap_norm, conflict_norm in [
            ("real", real_overlap_norm, real_conflict_norm),
            ("shuffled", shuffled_overlap_norm, shuffled_conflict_norm),
        ]:
            for mode in ["boost", "veto", "boost_veto"]:
                variant = f"P5R_{family}_R20_{mode}_eps{eps:g}"
                for row in parent_rows:
                    oid = str(row["mv_object_id"])
                    score = _num(row.get("score"), 1.0)
                    if mode in {"boost", "boost_veto"}:
                        score += eps * overlap_norm.get(oid, 0.0)
                    if mode in {"veto", "boost_veto"}:
                        score -= eps * conflict_norm.get(oid, 0.0)
                    new = dict(row)
                    new["variant_id"] = variant
                    new["score"] = float(score)
                    new["score_policy"] = f"d4rt_repair_{family}_{mode}_eps{eps:g}"
                    new["phase5_repair_family"] = family
                    new["phase5_repair_mode"] = mode
                    new["phase5_repair_eps"] = eps
                    new["uses_gt_for_prediction"] = False
                    new["uses_future"] = False
                    all_rows.append(new)
                variant_meta.append(
                    {
                        "variant_id": variant,
                        "family": family,
                        "mode": mode,
                        "eps": eps,
                        "merge_tau": "",
                        "merge_count": "",
                    }
                )
    for tau in TAU_SWEEP:
        mapping, merge_count = _l2h_map(parent_rows, tau)
        variant = f"P5R_real_R20_local2history_tau{tau:.2f}"
        for row in parent_rows:
            oid = str(row["mv_object_id"])
            new = dict(row)
            new["variant_id"] = variant
            new["mv_object_id"] = mapping.get(oid, oid)
            new["object_id"] = new["mv_object_id"]
            new["score_policy"] = f"d4rt_repair_local2history_tau{tau:.2f}"
            new["object_id_policy"] = "d4rt_anchor_local2history_merge_tau_sweep"
            new["phase5_repair_family"] = "real"
            new["phase5_repair_mode"] = "local2history"
            new["phase5_repair_tau"] = tau
            new["phase5_repair_merge_count"] = merge_count
            new["uses_gt_for_prediction"] = False
            new["uses_future"] = False
            all_rows.append(new)
        variant_meta.append(
            {
                "variant_id": variant,
                "family": "real",
                "mode": "local2history",
                "eps": "",
                "merge_tau": tau,
                "merge_count": merge_count,
            }
        )

    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant_id"] for row in all_rows}):
        rows = [row for row in all_rows if row["variant_id"] == variant]
        metrics, frames = p1._evaluate_variant(variant, rows, scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)
    meta_by_variant = {row["variant_id"]: row for row in variant_meta}
    for row in aggregate_rows:
        row.update(meta_by_variant.get(row["variant_id"], {}))

    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    base_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    phase2_window = float(phase2_summary["best_MV_AP_window"])
    phase2_scene = float(phase2_summary["best_MV_AP_scene"])
    real_score_rows = [row for row in aggregate_rows if row.get("family") == "real" and row.get("mode") != "local2history"]
    shuffled_score_rows = [row for row in aggregate_rows if row.get("family") == "shuffled"]
    l2h_rows = [row for row in aggregate_rows if row.get("mode") == "local2history"]
    best_real_score = max(real_score_rows, key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"])))
    best_shuffled_score = max(shuffled_score_rows, key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"])))
    best_l2h = max(l2h_rows, key=lambda row: (float(row["MV_AP_scene"]), float(row["MV_AP_window"])))
    real_minus_shuffled_window = float(best_real_score["MV_AP_window"]) - float(best_shuffled_score["MV_AP_window"])
    real_minus_shuffled_scene = float(best_real_score["MV_AP_scene"]) - float(best_shuffled_score["MV_AP_scene"])
    score_repair_pass = bool(
        float(best_real_score["MV_AP_window"]) >= base_window + 0.005
        and float(best_real_score["MV_AP50_window"]) >= base_ap50 + 0.010
        and (real_minus_shuffled_window >= 0.005 or real_minus_shuffled_scene >= 0.010)
    )
    l2h_scene_candidate = bool(float(best_l2h["MV_AP_scene"]) > phase2_scene and float(best_l2h["MV_AP_window"]) >= phase2_window - 0.003)
    gate_rows = [
        {
            "gate_id": "score_lambda_repair_pass",
            "pass": score_repair_pass,
            "expected": "real score sweep passes local F2_base gate and real-shuffled margin",
            "observed": f"best_real={best_real_score['variant_id']} window={best_real_score['MV_AP_window']} scene={best_real_score['MV_AP_scene']}; best_shuffled={best_shuffled_score['variant_id']} window={best_shuffled_score['MV_AP_window']} scene={best_shuffled_score['MV_AP_scene']}; margins window={real_minus_shuffled_window} scene={real_minus_shuffled_scene}",
            "severity": "repair_required",
        },
        {
            "gate_id": "local2history_scene_candidate_without_large_window_drop",
            "pass": l2h_scene_candidate,
            "expected": f"MV_AP_scene>{phase2_scene} and MV_AP_window>={phase2_window - 0.003}",
            "observed": f"best_l2h={best_l2h['variant_id']} window={best_l2h['MV_AP_window']} scene={best_l2h['MV_AP_scene']}",
            "severity": "diagnostic_candidate",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "no Phase5 promotion from this sweep; keep D4RT diagnostic/local2history evidence unless a later pre-registered repair passes controls",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase5_d4rt_repair_sweep_summary_v1",
        "phase_id": "v99_phase5_d4rt_repair_sweeps",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_PHASE5_REPAIR_SWEEP" if score_repair_pass else "NO_GO_PHASE5_REPAIR_SWEEP",
        "parent_phase2_variant": parent_variant,
        "phase5_source_decision": phase5_summary.get("decision"),
        "best_real_score_variant": best_real_score["variant_id"],
        "best_real_score_MV_AP_window": float(best_real_score["MV_AP_window"]),
        "best_real_score_MV_AP50_window": float(best_real_score["MV_AP50_window"]),
        "best_real_score_MV_AP_scene": float(best_real_score["MV_AP_scene"]),
        "best_real_score_MV_AP50_scene": float(best_real_score["MV_AP50_scene"]),
        "best_shuffled_score_variant": best_shuffled_score["variant_id"],
        "best_shuffled_score_MV_AP_window": float(best_shuffled_score["MV_AP_window"]),
        "best_shuffled_score_MV_AP_scene": float(best_shuffled_score["MV_AP_scene"]),
        "real_minus_shuffled_MV_AP_window": real_minus_shuffled_window,
        "real_minus_shuffled_MV_AP_scene": real_minus_shuffled_scene,
        "best_l2h_variant": best_l2h["variant_id"],
        "best_l2h_MV_AP_window": float(best_l2h["MV_AP_window"]),
        "best_l2h_MV_AP_scene": float(best_l2h["MV_AP_scene"]),
        "best_l2h_window_delta_vs_phase2": float(best_l2h["MV_AP_window"]) - phase2_window,
        "best_l2h_scene_delta_vs_phase2": float(best_l2h["MV_AP_scene"]) - phase2_scene,
        "F2_base_MV_AP_window": base_window,
        "F2_base_MV_AP_scene": base_scene,
        "blocking_failure_count": len([row for row in failure_rows if row["severity"] == "repair_required"]),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if score_repair_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
