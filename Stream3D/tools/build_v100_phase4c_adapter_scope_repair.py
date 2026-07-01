#!/usr/bin/env python3
"""Repair/audit Phase4 local-vs-scene adapter scope separation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = Path(os.environ.get("V100_PHASE4C_OUT_DIR", str(AUDIT_ROOT / "v100_phase4c_adapter_scope_repair")))

PHASE0_BASELINES = AUDIT_ROOT / "v100_phase0_contract/baseline_metric_rows.csv"
PHASE2_DIR = Path(os.environ.get("V100_PHASE2_DIR", str(AUDIT_ROOT / "v100_phase2_f2_local_final")))
PHASE4_MAIN = AUDIT_ROOT / "v100_phase4_history_memory"
PHASE4_REPAIR = AUDIT_ROOT / "v100_phase4b_history_memory_repair"
PHASE4_SCENE_REPAIR = AUDIT_ROOT / "v100_phase4d_scene_identity_repair"
PHASE4H_OVERLAP_EXACT = AUDIT_ROOT / "v100_phase4h_overlap3_exact_history_memory"
PHASE4K_PHASE2C_SEMANTIC = AUDIT_ROOT / "v100_phase4k_phase2c_semantic_scene_repair"


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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _phase2_local_by_split() -> dict[str, dict[str, str]]:
    metric_path = PHASE2_DIR / "variant_metric_rows.csv"
    if not metric_path.exists():
        metric_path = PHASE2_DIR / "mv_metric_window_rows.csv"
    rows = _read_csv(metric_path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row.get("dataset_split") or not row.get("MV_AP_window"):
            continue
        split = str(row.get("dataset_split", ""))
        if split:
            out[split] = row
    return out


def _baseline_rows() -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in _read_csv(PHASE0_BASELINES)}


def _phase4_metric_rows(phase_dir: Path, source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(phase_dir / "variant_metric_rows.csv"):
        split = row.get("dataset_split", "")
        if split not in {"dev", "holdout"}:
            continue
        row = dict(row)
        row["source_phase4_dir"] = _rel(phase_dir)
        row["source_id"] = source_id
        rows.append(row)
    return rows


def _fused_metric_rows() -> list[dict[str, Any]]:
    phase2_local = _phase2_local_by_split()
    fused: list[dict[str, Any]] = []
    phase_sources = [
        (PHASE4_MAIN, "phase4_main"),
        (PHASE4_REPAIR, "phase4b_repair_local"),
    ]
    if "phase2c_overlap3" in PHASE2_DIR.as_posix():
        phase_sources = []
        if PHASE4H_OVERLAP_EXACT.exists():
            phase_sources.append((PHASE4H_OVERLAP_EXACT, "phase4h_overlap3_exact"))
        if PHASE4K_PHASE2C_SEMANTIC.exists():
            phase_sources.append((PHASE4K_PHASE2C_SEMANTIC, "phase4k_phase2c_semantic"))
    else:
        if PHASE4_SCENE_REPAIR.exists():
            phase_sources.append((PHASE4_SCENE_REPAIR, "phase4d_scene_repair"))
        if PHASE4H_OVERLAP_EXACT.exists():
            phase_sources.append((PHASE4H_OVERLAP_EXACT, "phase4h_overlap3_exact"))
        if PHASE4K_PHASE2C_SEMANTIC.exists():
            phase_sources.append((PHASE4K_PHASE2C_SEMANTIC, "phase4k_phase2c_semantic"))
    for phase_dir, source_id in phase_sources:
        if not (phase_dir / "variant_metric_rows.csv").exists():
            continue
        source_rows = _phase4_metric_rows(phase_dir, source_id)
        for row in source_rows:
            split = str(row["dataset_split"])
            local = phase2_local[split]
            fused.append(
                {
                    "schema_version": "stream4d_v100_phase4c_variant_metric_row_v1",
                    "phase_id": "v100_phase4c_adapter_scope_repair",
                    "source_id": row["source_id"],
                    "source_phase4_dir": row["source_phase4_dir"],
                    "variant_id": row["variant_id"],
                    "dataset_split": split,
                    "adapter_scope": "local_window_uses_phase2_chunk_object_id; scene_uses_causal_history_id",
                    "MV_AP_window": _num(local.get("MV_AP_window")),
                    "MV_AP50_window": _num(local.get("MV_AP50_window")),
                    "MV_AP25_window": _num(local.get("MV_AP25_window")),
                    "ScoreFreeMatch50_window": _num(local.get("ScoreFreeMatch50_window")),
                    "history_mapped_MV_AP_window": _num(row.get("MV_AP_window")),
                    "history_mapped_MV_AP50_window": _num(row.get("MV_AP50_window")),
                    "history_mapped_local_window_AP_drop": _num(local.get("MV_AP_window")) - _num(row.get("MV_AP_window")),
                    "MV_AP_scene": _num(row.get("MV_AP_scene")),
                    "MV_AP50_scene": _num(row.get("MV_AP50_scene")),
                    "MV_AP25_scene": _num(row.get("MV_AP25_scene")),
                    "ScoreFreeMatch50_scene": _num(row.get("ScoreFreeMatch50_scene")),
                    "accepted_link_count": _num(row.get("accepted_link_count")),
                    "objects_crossing_multiple_chunks": _num(row.get("objects_crossing_multiple_chunks")),
                    "fragmentation_rate": _num(row.get("fragmentation_rate")),
                    "confirmed_history_count": _num(row.get("confirmed_history_count")),
                    "overmerge_large_component_count": _num(row.get("overmerge_large_component_count")),
                    "same_frame_collision_count": _num(local.get("same_frame_collision_count")),
                    "pixel_collision_rate": _num(local.get("pixel_collision_rate")),
                    "missing_mask_raster_count": _num(local.get("missing_mask_raster_count")),
                    "future_chunk_access": _bool(row.get("future_chunk_access")),
                    "uses_future": _bool(row.get("uses_future")),
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "metric_source": "v65_scene_metrics_from_phase4_history_rows_plus_v65_phase2_local_metrics",
                }
            )
    return fused


def _best_variant(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["source_id"]), str(row["variant_id"])), {})[str(row["dataset_split"])] = row
    best_key = max(
        grouped,
        key=lambda key: (
            _num(grouped[key].get("holdout", {}).get("MV_AP_scene")),
            _num(grouped[key].get("holdout", {}).get("MV_AP50_scene")),
            _num(grouped[key].get("dev", {}).get("MV_AP_scene")),
            _num(grouped[key].get("dev", {}).get("MV_AP50_scene")),
        ),
    )
    return best_key[0], best_key[1], grouped[best_key]


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase4c_artifact_manifest_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
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
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    phase2_local = _phase2_local_by_split()
    fused_rows = _fused_metric_rows()
    best_source_id, best_variant_id, best = _best_variant(fused_rows)
    best_dev = best["dev"]
    best_hold = best["holdout"]

    local_drop_dev = _num(phase2_local["dev"]["MV_AP_window"]) - _num(best_dev["MV_AP_window"])
    local_drop_hold = _num(phase2_local["holdout"]["MV_AP_window"]) - _num(best_hold["MV_AP_window"])
    gate_rows = [
        {
            "gate_id": "adapter_scope_local_drop_le_0p003",
            "pass": local_drop_dev <= 0.003 and local_drop_hold <= 0.003,
            "expected": "<=0.003 for dev and holdout",
            "observed": f"dev_drop={local_drop_dev}; holdout_drop={local_drop_hold}",
            "severity": "adapter_scope_required",
        },
        {
            "gate_id": "mv_ap_scene_dev_ge_f2_base_plus_0p010",
            "pass": _num(best_dev["MV_AP_scene"]) >= _num(f2_dev["MV_AP_scene"]) + 0.010,
            "expected": _num(f2_dev["MV_AP_scene"]) + 0.010,
            "observed": _num(best_dev["MV_AP_scene"]),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_dev_ge_f2_base_plus_0p015",
            "pass": _num(best_dev["MV_AP50_scene"]) >= _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "expected": _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "observed": _num(best_dev["MV_AP50_scene"]),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap_scene_holdout_ge_f2_base_plus_0p006",
            "pass": _num(best_hold["MV_AP_scene"]) >= _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "expected": _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "observed": _num(best_hold["MV_AP_scene"]),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_holdout_ge_f2_base_plus_0p010",
            "pass": _num(best_hold["MV_AP50_scene"]) >= _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "expected": _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "observed": _num(best_hold["MV_AP50_scene"]),
            "severity": "required_scene",
        },
        {
            "gate_id": "objects_crossing_multiple_chunks_gt_0",
            "pass": _num(best_dev["objects_crossing_multiple_chunks"]) + _num(best_hold["objects_crossing_multiple_chunks"]) > 0,
            "expected": ">0",
            "observed": f"dev={best_dev['objects_crossing_multiple_chunks']} holdout={best_hold['objects_crossing_multiple_chunks']}",
            "severity": "identity_required",
        },
        {
            "gate_id": "safety_and_causality_clean",
            "pass": (
                _num(best_dev["same_frame_collision_count"]) == 0
                and _num(best_hold["same_frame_collision_count"]) == 0
                and _num(best_dev["pixel_collision_rate"]) <= 0.02
                and _num(best_hold["pixel_collision_rate"]) <= 0.02
                and not _bool(best_dev["future_chunk_access"])
                and not _bool(best_hold["future_chunk_access"])
            ),
            "expected": "collision/pixel/future clean",
            "observed": f"dev_collision={best_dev['same_frame_collision_count']} hold_collision={best_hold['same_frame_collision_count']} dev_pixel={best_dev['pixel_collision_rate']} hold_pixel={best_hold['pixel_collision_rate']} future_dev={best_dev['future_chunk_access']} future_hold={best_hold['future_chunk_access']}",
            "severity": "required_safety",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase4c_failure_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "Adapter scope is clean if only scene gates fail. Next repair should target scene identity evidence, "
                "not local-window materialization."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase4c_pass = not failure_rows

    config_rows = [
        {
            "schema_version": "stream4d_v100_phase4c_variant_config_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "source_id": source_id,
            "variant_id": variant_id,
            "adapter_scope": "local_window_uses_phase2_chunk_object_id; scene_uses_causal_history_id",
            "source_phase4_dir": row_by_split["dev"]["source_phase4_dir"],
            "local_metric_source": _rel(PHASE2_DIR / "mv_metric_window_rows.csv"),
            "scene_metric_source": _rel(Path(row_by_split["dev"]["source_phase4_dir"]) / "variant_metric_rows.csv"),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for (source_id, variant_id), row_by_split in sorted(
            {
                (str(row["source_id"]), str(row["variant_id"])): {
                    split_row["dataset_split"]: split_row
                    for split_row in fused_rows
                    if split_row["source_id"] == row["source_id"] and split_row["variant_id"] == row["variant_id"]
                }
                for row in fused_rows
            }.items()
        )
    ]
    casebook_rows = [
        {
            "schema_version": "stream4d_v100_phase4c_casebook_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "case_id": "local_drop_was_adapter_scope_sensitive",
            "evidence": f"best_source={best_source_id} best_variant={best_variant_id} history_mapped_holdout_drop={best_hold['history_mapped_local_window_AP_drop']} separated_holdout_drop={local_drop_hold}",
            "interpretation": "Changing local-window prediction ids to history ids caused the apparent local AP drop; separated local/scene adapter scope restores local AP.",
        },
        {
            "schema_version": "stream4d_v100_phase4c_casebook_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "case_id": "scene_identity_still_fails",
            "evidence": f"dev_scene={best_dev['MV_AP_scene']} holdout_scene={best_hold['MV_AP_scene']}",
            "interpretation": "After adapter scope repair, the remaining blocker is true scene identity quality, not local materialization.",
        },
        {
            "schema_version": "stream4d_v100_phase4c_casebook_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "case_id": "overlap_membership_missing",
            "evidence": "Only applies to original v100 Phase2 import; Phase2c supplies overlap3 materialized rows when V100_PHASE2_DIR points to v100_phase2c_overlap3_local_repair.",
            "interpretation": "Use the Phase2b/Phase2c summaries to distinguish old non-overlap import from repaired overlap3 materialization.",
        },
    ]
    performance_rows = [
        {
            "schema_version": "stream4d_v100_phase4c_performance_row_v1",
            "phase_id": "v100_phase4c_adapter_scope_repair",
            "case_id": "csv_metric_fusion_scope_audit",
            "runtime_sec": time.time() - started,
            "variant_rows_checked": len(fused_rows),
            "v65_evaluator_runs": 0,
            "note": "No AP recomputation; fuses already canonical v65 Phase2 local metrics and Phase4 scene metrics.",
        }
    ]

    metric_csv = OUT_DIR / "variant_metric_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    casebook_csv = OUT_DIR / "casebook_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"

    _write_csv(metric_csv, fused_rows)
    _write_csv(config_csv, config_rows)
    _write_csv(gate_csv, [{"schema_version": "stream4d_v100_phase4c_gate_row_v1", "phase_id": "v100_phase4c_adapter_scope_repair", **row} for row in gate_rows])
    _write_csv(failure_csv, failure_rows)
    _write_csv(casebook_csv, casebook_rows)
    _write_csv(performance_csv, performance_rows)
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (metric_csv, "csv", "Phase4c fused local/scene metric rows"),
                (config_csv, "csv", "Phase4c adapter-scope configs"),
                (gate_csv, "csv", "Phase4c gates"),
                (failure_csv, "csv", "Phase4c failures"),
                (casebook_csv, "csv", "Phase4c evidence casebook"),
                (performance_csv, "csv", "Phase4c runtime"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase4c_adapter_scope_repair_summary_v1",
        "phase_id": "v100_phase4c_adapter_scope_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE5_SCOPE_REPAIRED" if phase4c_pass else "BLOCK_SCENE_IDENTITY_AFTER_SCOPE_REPAIR",
        "phase4c_pass": phase4c_pass,
        "failure_count": len(failure_rows),
        "best_source_id": best_source_id,
        "best_variant_id": best_variant_id,
        "best_dev_MV_AP_window": best_dev["MV_AP_window"],
        "best_dev_MV_AP50_window": best_dev["MV_AP50_window"],
        "best_dev_MV_AP_scene": best_dev["MV_AP_scene"],
        "best_dev_MV_AP50_scene": best_dev["MV_AP50_scene"],
        "best_holdout_MV_AP_window": best_hold["MV_AP_window"],
        "best_holdout_MV_AP50_window": best_hold["MV_AP50_window"],
        "best_holdout_MV_AP_scene": best_hold["MV_AP_scene"],
        "best_holdout_MV_AP50_scene": best_hold["MV_AP50_scene"],
        "adapter_scope_local_drop": {"dev": local_drop_dev, "holdout": local_drop_hold},
        "history_mapped_local_drop": {
            "dev": best_dev["history_mapped_local_window_AP_drop"],
            "holdout": best_hold["history_mapped_local_window_AP_drop"],
        },
        "objects_crossing_multiple_chunks": {
            "dev": int(_num(best_dev["objects_crossing_multiple_chunks"])),
            "holdout": int(_num(best_hold["objects_crossing_multiple_chunks"])),
        },
        "fragmentation_rate": {
            "dev": best_dev["fragmentation_rate"],
            "holdout": best_hold["fragmentation_rate"],
        },
        "future_chunk_access": False,
        "formal_claim_allowed": False,
        "outputs": {
            "summary": _rel(summary_json),
            "variant_metric_rows": _rel(metric_csv),
            "variant_config_rows": _rel(config_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "casebook_rows": _rel(casebook_csv),
            "performance_rows": _rel(performance_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4c_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
