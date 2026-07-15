#!/usr/bin/env python3
"""Build v117 Stage3 internal-cue and memory-reliability readiness artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability"
OUT = RESULT_ROOT / "stage3_internal_reliability"
STAGE1 = RESULT_ROOT / "stage1_object_identity"
STAGE2 = RESULT_ROOT / "stage2_memory_provenance"
V115_STAGE2 = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control/stage2_alignment_cues"
V114_MISSING = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control/stage1_cue_bank/stage1_missing_internal_cue_report.md"

SEQS = ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))))


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([fnum(row.get(key)) for row in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]


def quantile_span(rows: list[dict[str, Any]], key: str) -> float:
    vals = finite_values(rows, key)
    if vals.size < 2:
        return 0.0
    return float(np.percentile(vals, 90) - np.percentile(vals, 10))


def norm_from_distribution(value: float, values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size < 2 or not math.isfinite(value):
        return 0.5
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    if abs(p90 - p10) < 1e-9:
        return 0.5
    return float(max(0.0, min(1.0, (value - p10) / (p90 - p10))))


def seq_from_case(row: dict[str, str]) -> str:
    case = row.get("case", "")
    match = re.search(r"kitti_(\d\d)", case)
    if match:
        return match.group(1)
    seq = row.get("seq", "")
    if seq in SEQS:
        return seq
    if "/" in seq:
        return seq.split("/")[0]
    return seq


def lingbot_candidate_and_reliability() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for seq in SEQS:
        for row in read_csv(STAGE1 / seq / "stage1_track_summary_rows.csv"):
            track_id = row.get("object_or_track_id", "")
            persistence = fnum(row.get("object_persistence"), 0.0)
            boundary_stability = fnum(row.get("boundary_stability"), 0.0)
            motion_risk = fnum(row.get("centroid_motion_risk"), 0.0)
            confidence = fnum(row.get("semantic_confidence"), 0.0)
            semantic_candidate = max(0.0, min(1.0, persistence * boundary_stability * (1.0 - motion_risk)))
            source_reliability = max(0.0, min(1.0, 0.45 * persistence + 0.35 * boundary_stability + 0.2 * confidence))
            unit_id = f"lingbot:{seq}:track:{track_id}"
            candidate_rows.append(
                {
                    "schema": "acl2_v117tf_stage3_candidate_update_row_v1",
                    "model": "LingBot",
                    "seq": seq,
                    "surface": "trajectory_admission_proxy",
                    "unit_id": unit_id,
                    "object_or_track_id": track_id,
                    "candidate_gain": semantic_candidate,
                    "candidate_gain_mode": "semantic_object_persistence_proxy_not_true_internal",
                    "alignment_available": False,
                    "residual_available": False,
                    "attention_entropy_available": False,
                    "update_pressure_available": False,
                    "source": rel(STAGE1 / seq / "stage1_track_summary_rows.csv"),
                }
            )
            reliability_rows.append(
                {
                    "schema": "acl2_v117tf_stage3_memory_reliability_row_v1",
                    "model": "LingBot",
                    "seq": seq,
                    "memory_family": "append_only_source_read_proxy",
                    "unit_id": unit_id,
                    "object_or_track_id": track_id,
                    "memory_reliability": source_reliability,
                    "reliability_mode": "append_only_source_read_object_persistence_proxy",
                    "fixed_reference_deviation_available": False,
                    "read_entropy_available": False,
                    "recent_update_pressure_available": False,
                    "source": rel(STAGE1 / seq / "stage1_track_summary_rows.csv"),
                }
            )
    return candidate_rows, reliability_rows


def hs_candidate_and_reliability() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    head_rows = [row for row in read_csv(V115_STAGE2 / "hs_head_reliability_rows.csv") if seq_from_case(row) in SEQS]
    gla_rows = [row for row in read_csv(V115_STAGE2 / "hs_gla_state_quality_rows.csv") if seq_from_case(row) in SEQS]
    mrt_rows = [row for row in read_csv(V115_STAGE2 / "hs_mrt_scale_safety_rows.csv") if seq_from_case(row) in SEQS]

    head_q = np.asarray([fnum(row.get("internal_head_q_std")) for row in head_rows], dtype=np.float64)
    head_gate = np.asarray([fnum(row.get("gate_std")) for row in head_rows], dtype=np.float64)
    gla_delta = np.asarray([fnum(row.get("state_delta_norm")) for row in gla_rows], dtype=np.float64)
    mrt_delta = np.asarray([abs(fnum(row.get("predicted_metric_scale_delta"))) for row in mrt_rows], dtype=np.float64)

    for idx, row in enumerate(head_rows):
        seq = seq_from_case(row)
        q_std = fnum(row.get("internal_head_q_std"))
        gate_std = fnum(row.get("gate_std"))
        risk = fnum(row.get("semantic_risk_mean"), 0.0)
        stable = fnum(row.get("semantic_stable_mean"), 0.0)
        candidate_gain = max(
            0.0,
            min(
                1.0,
                0.5 * norm_from_distribution(q_std, head_q)
                + 0.2 * norm_from_distribution(gate_std, head_gate)
                + 0.3 * sigmoid(stable - risk),
            ),
        )
        reliability = max(0.0, min(1.0, sigmoid(stable - risk + 0.5 * q_std)))
        unit_id = f"hs:{seq}:local_head:{row.get('chunk_idx','')}:{idx}"
        candidate_rows.append(
            {
                "schema": "acl2_v117tf_stage3_candidate_update_row_v1",
                "model": "HorizonStream",
                "seq": seq,
                "surface": "local_pose_head_probe",
                "unit_id": unit_id,
                "chunk_idx": row.get("chunk_idx", ""),
                "candidate_gain": candidate_gain,
                "candidate_gain_mode": "v115_head_probe_internal_std_proxy",
                "alignment_available": "head_probe_only",
                "residual_available": False,
                "attention_entropy_available": False,
                "update_pressure_available": False,
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_head_reliability_rows.csv")),
            }
        )
        reliability_rows.append(
            {
                "schema": "acl2_v117tf_stage3_memory_reliability_row_v1",
                "model": "HorizonStream",
                "seq": seq,
                "memory_family": "local_kv_head_probe",
                "unit_id": unit_id,
                "chunk_idx": row.get("chunk_idx", ""),
                "memory_reliability": reliability,
                "reliability_mode": "semantic_stable_minus_risk_plus_head_q_proxy",
                "fixed_reference_deviation_available": False,
                "read_entropy_available": False,
                "recent_update_pressure_available": False,
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_head_reliability_rows.csv")),
            }
        )

    for idx, row in enumerate(gla_rows):
        seq = seq_from_case(row)
        delta = fnum(row.get("state_delta_norm"))
        conv_norm = fnum(row.get("conv_state_norm"))
        stable = fnum(row.get("chunk_stable_mass_mean"), 0.0)
        dynamic = fnum(row.get("chunk_dynamic_mass_mean"), 0.0)
        boundary = fnum(row.get("chunk_boundary_mass_mean"), 0.0)
        candidate_gain = 0.55 * norm_from_distribution(delta, gla_delta) + 0.45 * sigmoid(stable - dynamic - boundary)
        reliability = max(0.0, min(1.0, sigmoid(stable - dynamic - boundary - 0.02 * conv_norm)))
        unit_id = f"hs:{seq}:gla_layer_chunk_band:{row.get('global_layer_idx','')}:{row.get('chunk_idx','')}:{idx}"
        candidate_rows.append(
            {
                "schema": "acl2_v117tf_stage3_candidate_update_row_v1",
                "model": "HorizonStream",
                "seq": seq,
                "surface": "gla_state_probe_chunk_band",
                "unit_id": unit_id,
                "chunk_idx": row.get("chunk_idx", ""),
                "candidate_gain": max(0.0, min(1.0, candidate_gain)),
                "candidate_gain_mode": "gla_state_delta_chunk_band_proxy",
                "alignment_available": False,
                "residual_available": "state_delta_norm_proxy",
                "attention_entropy_available": False,
                "update_pressure_available": False,
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_gla_state_quality_rows.csv")),
            }
        )
        reliability_rows.append(
            {
                "schema": "acl2_v117tf_stage3_memory_reliability_row_v1",
                "model": "HorizonStream",
                "seq": seq,
                "memory_family": "gla_state_chunk_band",
                "unit_id": unit_id,
                "chunk_idx": row.get("chunk_idx", ""),
                "memory_reliability": reliability,
                "reliability_mode": "chunk_semantic_state_norm_proxy",
                "fixed_reference_deviation_available": bool(str(row.get("state_ori_norm", "")).strip()),
                "read_entropy_available": False,
                "recent_update_pressure_available": False,
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_gla_state_quality_rows.csv")),
            }
        )

    for idx, row in enumerate(mrt_rows[:2000]):
        seq = seq_from_case(row)
        scale_delta = abs(fnum(row.get("predicted_metric_scale_delta"), 0.0))
        risk = fnum(row.get("chunk_semantic_risk"), 0.0)
        stable = fnum(row.get("chunk_stable_mass"), 0.0)
        unit_id = f"hs:{seq}:mrt_readout:{row.get('chunk_idx','')}:{idx}"
        candidate_rows.append(
            {
                "schema": "acl2_v117tf_stage3_candidate_update_row_v1",
                "model": "HorizonStream",
                "seq": seq,
                "surface": "mrt_readout_diagnostic",
                "unit_id": unit_id,
                "chunk_idx": row.get("chunk_idx", ""),
                "candidate_gain": 0.6 * norm_from_distribution(scale_delta, mrt_delta) + 0.4 * sigmoid(stable - risk),
                "candidate_gain_mode": "mrt_scale_delta_safety_diagnostic_only",
                "alignment_available": False,
                "residual_available": False,
                "attention_entropy_available": False,
                "update_pressure_available": False,
                "source": row.get("source_path", rel(V115_STAGE2 / "hs_mrt_scale_safety_rows.csv")),
            }
        )
    return candidate_rows, reliability_rows


def distribution_rows(candidate_rows: list[dict[str, Any]], reliability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("candidate", candidate_rows, "candidate_gain", "surface"),
        ("memory", reliability_rows, "memory_reliability", "memory_family"),
    ]
    for family, source_rows, metric, group_key in metric_specs:
        groups = sorted({(str(row.get("model", "")), str(row.get(group_key, ""))) for row in source_rows})
        for model, group in groups:
            vals = np.asarray(
                [fnum(row.get(metric)) for row in source_rows if str(row.get("model", "")) == model and str(row.get(group_key, "")) == group],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "group": group,
                    "metric": metric,
                    "count": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                    "p10_p90_span": float(np.percentile(vals, 90) - np.percentile(vals, 10)),
                }
            )
    return rows


def max_group_span(dist_rows: list[dict[str, Any]], family: str, metric: str) -> float:
    vals = [
        fnum(row.get("p10_p90_span"))
        for row in dist_rows
        if row.get("family") == family and row.get("metric") == metric
    ]
    vals_np = np.asarray(vals, dtype=np.float64)
    vals_np = vals_np[np.isfinite(vals_np)]
    return float(np.max(vals_np)) if vals_np.size else 0.0


def agreement_rows(candidate_rows: list[dict[str, Any]], reliability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rel_by_unit = {str(row.get("unit_id", "")): row for row in reliability_rows}
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        unit_id = str(row.get("unit_id", ""))
        rel_row = rel_by_unit.get(unit_id)
        if not rel_row:
            continue
        candidate = fnum(row.get("candidate_gain"), 0.0)
        reliability = fnum(rel_row.get("memory_reliability"), 0.0)
        rows.append(
            {
                "schema": "acl2_v117tf_stage3_candidate_reliability_agreement_row_v1",
                "model": row.get("model", ""),
                "seq": row.get("seq", ""),
                "surface": row.get("surface", ""),
                "unit_id": unit_id,
                "candidate_gain": candidate,
                "memory_reliability": reliability,
                "calibrated_gain": candidate * reliability,
                "semantic_internal_conflict": candidate >= 0.75 and reliability <= 0.35,
                "join_mode": "unit_id_exact",
            }
        )
    return rows


def lingbot_update_pressure_rows() -> list[dict[str, Any]]:
    traj_rows = read_csv(STAGE2 / "lingbot_trajectory_provenance_rows.csv")
    raw_cache: dict[str, list[int]] = {}
    keyframe_interval_cache: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in traj_rows:
        action_file_text = str(row.get("action_file", ""))
        action_file = resolve_path(action_file_text)
        cache_key = action_file.as_posix()
        if cache_key not in raw_cache:
            write_positions: list[int] = []
            keyframe_interval = 15
            if action_file.exists():
                with action_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        keyframe_interval = int(payload.get("keyframe_interval", keyframe_interval))
                        if bool(payload.get("final_is_keyframe")) and not bool(payload.get("skip_append")):
                            write_positions.append(int(payload.get("sample_position", -1)))
            raw_cache[cache_key] = sorted(v for v in write_positions if v >= 0)
            keyframe_interval_cache[cache_key] = keyframe_interval

        frame_id = int(fnum(row.get("retained_or_target_frame_id"), -1))
        write_positions = raw_cache[cache_key]
        writes_before = [pos for pos in write_positions if pos < frame_id]
        recent_window = 300
        recent_writes = [pos for pos in writes_before if frame_id - recent_window <= pos < frame_id]
        pressure = 1.0 - math.exp(-float(len(recent_writes)) / 10.0)
        rows.append(
            {
                "schema": "acl2_v117tf_stage3_lingbot_update_pressure_row_v1",
                "seq": row.get("seq", ""),
                "policy_id": row.get("policy_id", ""),
                "unit_id": row.get("unit_id", ""),
                "target_frame_id": frame_id,
                "writes_before_target": len(writes_before),
                "recent_window_frames": recent_window,
                "recent_write_count": len(recent_writes),
                "effective_update_pressure": pressure,
                "keyframe_interval": keyframe_interval_cache.get(cache_key, 15),
                "source_write_trace": rel(action_file) if action_file.exists() else action_file_text,
                "action_file_exists": action_file.exists(),
            }
        )
    return rows


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# v117 Stage3 Internal Reliability Readiness",
        "",
        f"- stage3_ready: `{summary['stage3_ready']}`",
        f"- candidate_gain_span: `{summary['candidate_gain_p10_p90_span']}`",
        f"- memory_reliability_span: `{summary['memory_reliability_p10_p90_span']}`",
        f"- memory_reliability_group_max_span: `{summary['memory_reliability_group_max_p10_p90_span']}`",
        f"- update_pressure_gate: `{summary['update_pressure_gate']}`",
        f"- fixed_reference_deviation_gate: `{summary['fixed_reference_deviation_gate']}`",
        f"- attention_alignment_gate: `{summary['attention_alignment_gate']}`",
        f"- semantic_provenance_join_gate: `{summary['semantic_provenance_join_gate']}`",
        "",
        "## Boundary",
        "",
        "Candidate distributions are non-constant and LingBot trajectory action-target provenance is available at frame-aggregate resolution.",
        "HorizonStream probes provide head/GQ/MRT diagnostics, not full attention entropy or token/channel-level fixed-reference state reliability.",
        "Stage3 remains blocked for runtime calibration unless update pressure, fixed-reference memory deviation, and attention/alignment gates are all auditable.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(STAGE2 / "stage2_memory_reliability_readiness.json")

    lb_candidate, lb_reliability = lingbot_candidate_and_reliability()
    hs_candidate, hs_reliability = hs_candidate_and_reliability()
    candidate_rows = lb_candidate + hs_candidate
    reliability_rows = lb_reliability + hs_reliability
    update_rows = lingbot_update_pressure_rows()
    dist_rows = distribution_rows(candidate_rows, reliability_rows)
    agree_rows = agreement_rows(candidate_rows, reliability_rows)

    candidate_span = quantile_span(candidate_rows, "candidate_gain")
    reliability_span = quantile_span(reliability_rows, "memory_reliability")
    memory_group_span = max_group_span(dist_rows, "memory", "memory_reliability")
    candidate_span_gate = candidate_span >= 0.15
    reliability_span_gate = reliability_span >= 0.15 or memory_group_span >= 0.15
    update_pressure_span = quantile_span(update_rows, "recent_write_count")
    update_pressure_gate = bool(update_rows) and update_pressure_span >= 1.0 and all(
        bool(row.get("action_file_exists")) for row in update_rows
    )
    fixed_reference_rows = [
        row for row in reliability_rows
        if row.get("model") == "HorizonStream" and row.get("memory_family") == "gla_state_chunk_band"
    ]
    fixed_reference_available = sum(1 for row in fixed_reference_rows if row.get("fixed_reference_deviation_available") is True)
    fixed_reference_coverage = (
        float(fixed_reference_available) / float(len(fixed_reference_rows)) if fixed_reference_rows else 0.0
    )
    fixed_reference_gate = fixed_reference_coverage >= 0.95
    attention_alignment_gate = False
    semantic_join_gate = bool(stage2.get("hs_local_kv_gate")) and bool(stage2.get("lingbot_anchor_patch_gate")) and bool(
        stage2.get("lingbot_trajectory_action_target_gate")
    )
    stage3_ready = (
        candidate_span_gate
        and reliability_span_gate
        and update_pressure_gate
        and fixed_reference_gate
        and attention_alignment_gate
        and semantic_join_gate
        and bool(stage2.get("stage2_ready"))
    )
    blockers: list[str] = []
    if not bool(stage2.get("stage2_ready")):
        blockers.append("Stage2 is not ready")
    if not reliability_span_gate:
        blockers.append("Memory reliability distribution remains too narrow for the Stage3 gate")
    if not update_pressure_gate:
        blockers.append("No auditable varying effective update-pressure trace from true writes is available")
    if not fixed_reference_gate:
        blockers.append("Fixed first-write reference deviation is not available for the relevant recurrent memory units")
    if not attention_alignment_gate:
        blockers.append("Attention entropy/alignment action path remains blocked for LingBot and incomplete for HorizonStream")
    if V114_MISSING.exists():
        blockers.append(f"Prior missing-cue report retained: {rel(V114_MISSING)}")

    summary = {
        "schema": "acl2_v117tf_stage3_internal_reliability_readiness_v1",
        "stage3_ready": stage3_ready,
        "stage2_ready_dependency": bool(stage2.get("stage2_ready")),
        "candidate_gain_p10_p90_span": candidate_span,
        "memory_reliability_p10_p90_span": reliability_span,
        "memory_reliability_group_max_p10_p90_span": memory_group_span,
        "candidate_gain_span_gate": candidate_span_gate,
        "memory_reliability_span_gate": reliability_span_gate,
        "update_pressure_gate": update_pressure_gate,
        "update_pressure_recent_write_count_p10_p90_span": update_pressure_span,
        "fixed_reference_deviation_gate": fixed_reference_gate,
        "fixed_reference_deviation_row_coverage": fixed_reference_coverage,
        "attention_alignment_gate": attention_alignment_gate,
        "semantic_provenance_join_gate": semantic_join_gate,
        "candidate_update_rows": len(candidate_rows),
        "memory_reliability_rows": len(reliability_rows),
        "update_pressure_rows": len(update_rows),
        "candidate_reliability_agreement_rows": len(agree_rows),
        "blockers": blockers,
        "outputs": {
            "candidate_update_rows": rel(OUT / "candidate_update_rows.csv"),
            "memory_reliability_rows": rel(OUT / "memory_reliability_rows.csv"),
            "update_pressure_rows": rel(OUT / "update_pressure_rows.csv"),
            "internal_cue_distribution": rel(OUT / "internal_cue_distribution.csv"),
            "candidate_reliability_agreement_rows": rel(OUT / "candidate_reliability_agreement_rows.csv"),
            "stage3_internal_readiness": rel(OUT / "stage3_internal_readiness.json"),
            "report": rel(OUT / "INTERNAL_RELIABILITY_READINESS_REPORT.md"),
            "blocked_report": rel(OUT / "STAGE3_INTERNAL_READINESS_BLOCKED.md"),
        },
    }
    write_csv(OUT / "candidate_update_rows.csv", candidate_rows)
    write_csv(OUT / "memory_reliability_rows.csv", reliability_rows)
    write_csv(OUT / "update_pressure_rows.csv", update_rows)
    write_csv(OUT / "internal_cue_distribution.csv", dist_rows)
    write_csv(OUT / "candidate_reliability_agreement_rows.csv", agree_rows)
    write_json(OUT / "stage3_internal_readiness.json", summary)
    write_text(OUT / "INTERNAL_RELIABILITY_READINESS_REPORT.md", report_text(summary))
    write_text(
        OUT / "STAGE3_INTERNAL_READINESS_BLOCKED.md",
        "# Stage3 Internal Readiness Blocked\n\n"
        "The generated proxy distributions are audit artifacts only. Runtime calibration is blocked because fixed-reference memory deviation "
        "and attention/alignment paths are not all auditable.\n\n"
        + "\n".join(f"- {item}" for item in blockers),
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
