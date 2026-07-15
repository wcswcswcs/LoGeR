#!/usr/bin/env python3
"""Build ACL2 v114-TF Stage1 semantic+internal evidence-quality cue artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"
STAGE1 = RESULT_ROOT / "stage1_cue_bank"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"

ROLE_TO_ID = {
    "dynamic": 0,
    "boundary_lowpurity": 1,
    "weak_context": 2,
    "stable_landmark": 3,
    "vegetation_repetitive": 4,
    "sky_lowobs": 5,
    "unknown_lowtrust": 6,
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def lingbot_rows() -> list[dict[str, Any]]:
    sources = [
        ("anchor", V112 / "stage2_memory_specific_cue_bank/anchor_memory_cue_rows.csv"),
        ("local", V112 / "stage2_memory_specific_cue_bank/local_window_token_cue_rows.csv"),
        ("trajectory", V112 / "stage2_memory_specific_cue_bank/trajectory_memory_cue_rows.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for surface, path in sources:
        for row in read_csv(path):
            stable = f(row.get("stable_landmark_mass"))
            dynamic = f(row.get("dynamic_mass"))
            boundary = f(row.get("boundary_mass"))
            weak = f(row.get("weak_context_mass"))
            trust = f(row.get("semantic_trust_mean"))
            purity = f(row.get("semantic_purity_mean"))
            semantic_risk = dynamic + boundary + 0.3 * weak + max(0.0, 1.0 - trust if math.isfinite(trust) else 0.0)
            semantic_support = stable + 0.2 * (trust if math.isfinite(trust) else 0.0)
            rows.append(
                {
                    "schema": "acl2_v114tf_stage1_lingbot_evidence_quality_row_v1",
                    "model": "LingBot",
                    "surface": surface,
                    "seq": row.get("seq", ""),
                    "frame_id": row.get("frame_id", row.get("current_frame", "")),
                    "context_type": row.get("context_type", "anchor_context" if surface == "anchor" else ""),
                    "token_type": row.get("token_type", row.get("context_token_type", "")),
                    "source_frame": row.get("source_frame", ""),
                    "source_frame_age": row.get("source_frame_age", ""),
                    "semantic_risk_proxy": semantic_risk,
                    "semantic_support_proxy": semantic_support,
                    "stable_landmark_mass": row.get("stable_landmark_mass", ""),
                    "dynamic_mass": row.get("dynamic_mass", ""),
                    "boundary_mass": row.get("boundary_mass", ""),
                    "weak_context_mass": row.get("weak_context_mass", ""),
                    "semantic_trust_mean": row.get("semantic_trust_mean", ""),
                    "semantic_purity_mean": row.get("semantic_purity_mean", ""),
                    "semantic_continuity": row.get("semantic_continuity_score", ""),
                    "internal_quality_proxy": row.get("Q_anchor_frame", row.get("Q_traj_frame", row.get("S_local", ""))),
                    "internal_attention_mass": "",
                    "internal_attention_entropy": "",
                    "internal_contradiction_proxy": "",
                    "missing_internal_attention": True,
                    "missing_internal_contradiction": True,
                    "evidence_quality_score_semantic_only": float(sigmoid(semantic_support - semantic_risk)),
                    "source_path": rel(path),
                }
            )
    return rows


def baseline_by_seq() -> dict[str, dict[str, str]]:
    out = {}
    for row in read_csv(V113 / "diagnostics/stage1_hs_baseline_metrics_rows.csv"):
        out[str(row.get("seq", ""))] = row
    return out


def horizon_rows() -> list[dict[str, Any]]:
    sem_root = V113 / "semantic_projection"
    base = baseline_by_seq()
    rows: list[dict[str, Any]] = []
    for seq in ["00", "01", "02", "05"]:
        role_path = sem_root / f"seq{seq}_role_ids.npy"
        risk_path = sem_root / f"seq{seq}_risk.npy"
        stable_path = sem_root / f"seq{seq}_stable.npy"
        conf_path = sem_root / f"seq{seq}_confidence.npy"
        if not (role_path.exists() and risk_path.exists() and stable_path.exists()):
            continue
        roles = np.load(role_path, mmap_mode="r")
        risk = np.load(risk_path, mmap_mode="r")
        stable = np.load(stable_path, mmap_mode="r")
        confidence = np.load(conf_path, mmap_mode="r") if conf_path.exists() else None
        frame_risk = np.asarray(np.mean(risk, axis=1), dtype=np.float32)
        frame_stable = np.asarray(np.mean(stable, axis=1), dtype=np.float32)
        frame_conf = np.asarray(np.mean(confidence, axis=1), dtype=np.float32) if confidence is not None else np.full_like(frame_risk, np.nan)
        prev_risk = np.concatenate([frame_risk[:1], frame_risk[:-1]])
        prev_stable = np.concatenate([frame_stable[:1], frame_stable[:-1]])
        continuity = 1.0 - np.clip(np.abs(frame_risk - prev_risk) + np.abs(frame_stable - prev_stable), 0.0, 1.0)
        base_row = base.get(seq, {})
        for frame_idx in range(int(risk.shape[0])):
            role_row = roles[frame_idx]
            total = float(max(role_row.size, 1))
            dynamic_mass = float(np.count_nonzero(role_row == ROLE_TO_ID["dynamic"]) / total)
            boundary_mass = float(np.count_nonzero(role_row == ROLE_TO_ID["boundary_lowpurity"]) / total)
            weak_mass = float(np.count_nonzero(role_row == ROLE_TO_ID["weak_context"]) / total)
            vegetation_mass = float(np.count_nonzero(role_row == ROLE_TO_ID["vegetation_repetitive"]) / total)
            sky_mass = float(np.count_nonzero(role_row == ROLE_TO_ID["sky_lowobs"]) / total)
            q_sem = float(sigmoid(frame_stable[frame_idx] - frame_risk[frame_idx] + 0.2 * frame_conf[frame_idx]))
            rows.append(
                {
                    "schema": "acl2_v114tf_stage1_horizonstream_evidence_quality_row_v1",
                    "model": "HorizonStream",
                    "surface": "local_value_path",
                    "seq": seq,
                    "frame_idx": frame_idx,
                    "patch_count": int(role_row.size),
                    "frame_dynamic_mass": dynamic_mass,
                    "frame_boundary_mass": boundary_mass,
                    "frame_weak_context_mass": weak_mass,
                    "frame_vegetation_mass": vegetation_mass,
                    "frame_sky_mass": sky_mass,
                    "frame_risk_mean": float(frame_risk[frame_idx]),
                    "frame_stable_mean": float(frame_stable[frame_idx]),
                    "semantic_trust_mean": float(frame_conf[frame_idx]) if math.isfinite(float(frame_conf[frame_idx])) else "",
                    "semantic_continuity_proxy": float(continuity[frame_idx]),
                    "internal_local_value_norm_available": "runtime_only",
                    "internal_local_q_proxy": "row_normalized_local_kv_value_norm_at_action_time",
                    "internal_entropy_available": False,
                    "internal_qk_available": False,
                    "internal_mrt_scale_available": "trace_only_for_mrt_rows",
                    "baseline_full_ATE_sim3_rmse": base_row.get("full_ATE_sim3_rmse", ""),
                    "baseline_rolling_ate_p90": base_row.get("rolling_ate_p90", ""),
                    "baseline_segment_scale_log_error_median_abs": base_row.get("segment_scale_log_error_median_abs", ""),
                    "baseline_missing": seq not in base,
                    "evidence_quality_score_semantic_only": q_sem,
                    "source_path": rel(sem_root),
                }
            )
    return rows


def distribution_rows(hs_rows: list[dict[str, Any]], lb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, source_rows, group_key in [
        ("HorizonStream", hs_rows, "seq"),
        ("LingBot", lb_rows, "surface"),
    ]:
        groups = sorted(set(str(r.get(group_key, "")) for r in source_rows))
        for group in groups:
            subset = [r for r in source_rows if str(r.get(group_key, "")) == group]
            for metric in [
                "frame_risk_mean",
                "frame_stable_mean",
                "evidence_quality_score_semantic_only",
                "semantic_risk_proxy",
                "semantic_support_proxy",
            ]:
                vals = np.asarray([f(r.get(metric)) for r in subset], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rows.append(
                    {
                        "model": model,
                        "group_key": group_key,
                        "group": group,
                        "metric": metric,
                        "count": int(vals.size),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "p05": float(np.percentile(vals, 5)),
                        "p50": float(np.percentile(vals, 50)),
                        "p95": float(np.percentile(vals, 95)),
                    }
                )
    return rows


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "candidate": "HS_LQ1_stable_rowmean_neutral_internal_confident",
            "surface": "HorizonStream local V",
            "semantic": "stable support centered per source row",
            "internal": "row-normalized local KV value norm confidence",
            "rowmean_neutral": True,
            "pilot": "00,02",
            "rationale": "directly targets v113 row-mean/generic confound while keeping stable support only where internal value confidence is high",
        },
        {
            "rank": 2,
            "candidate": "HS_LQ2_risk_suppress_internal_mismatch",
            "surface": "HorizonStream local V",
            "semantic": "risk centered per source row",
            "internal": "positive local KV value norm anomaly as mismatch/uncertainty proxy",
            "rowmean_neutral": True,
            "pilot": "00,02",
            "rationale": "dynamic/boundary suppression only when internal value path is anomalous, avoiding pure risk scalar repeat",
        },
        {
            "rank": 3,
            "candidate": "HS_LQ3_stable_plus_risk_internal_quality",
            "surface": "HorizonStream local V",
            "semantic": "stable and risk interaction",
            "internal": "confidence and anomaly from local KV value norm",
            "rowmean_neutral": True,
            "pilot": "00,02",
            "rationale": "combined positive/negative interaction gate if LQ1/LQ2 alone are too weak",
        },
        {
            "rank": 4,
            "candidate": "HS_LQ_CTRL_internal_confident_only",
            "surface": "control",
            "semantic": "none",
            "internal": "row-normalized local KV value norm only",
            "rowmean_neutral": True,
            "pilot": "00,02",
            "rationale": "tests whether internal-only explains the effect",
        },
        {
            "rank": 5,
            "candidate": "HS_LQ_CTRL_semantic_only_rowmean_neutral",
            "surface": "control",
            "semantic": "stable/risk centered",
            "internal": "none",
            "rowmean_neutral": True,
            "pilot": "00,02",
            "rationale": "tests whether semantic-only still suffices after row-mean neutralization",
        },
        {
            "rank": 6,
            "candidate": "HS_LQ_CTRL_rowmean_only_generic_scale",
            "surface": "control",
            "semantic": "row mean only",
            "internal": "none",
            "rowmean_neutral": False,
            "pilot": "00,02",
            "rationale": "explicit generic value-scale confound control",
        },
    ]


def main() -> None:
    STAGE1.mkdir(parents=True, exist_ok=True)
    lb_rows = lingbot_rows()
    hs_rows = horizon_rows()
    dist_rows = distribution_rows(hs_rows, lb_rows)
    candidates = candidate_rows()

    write_csv(STAGE1 / "stage1_lingbot_evidence_quality_rows.csv", lb_rows)
    write_csv(STAGE1 / "stage1_horizonstream_evidence_quality_rows.csv", hs_rows)
    write_csv(STAGE1 / "stage1_quality_score_distributions.csv", dist_rows)
    write_csv(STAGE1 / "stage1_candidate_surface_rank.csv", candidates)

    missing_report = """# Stage1 Missing Internal Cue Report

Stage1 is a cue-bank stage only; it does not claim geometry.

## HorizonStream

- Available offline: semantic role/risk/stable/trust per frame and per patch from v113 semantic projection.
- Available from v113 trace: bounded local/MRT/GLA probe rows, not full local/GLA tensor traces.
- Missing offline by design: local attention entropy, QK relevance, head-wise attention mass, and full GLA state-delta rows. v113 showed full local/GLA tensor tracing OOM on 22GB GPUs.
- Runtime-only internal proxy used for HS-LQ: row-normalized local KV value norm inside `apply_local_kv_value_action`; it is recorded by `HS_V114_ACTION_AUDIT_ROOT/hs_lq_action_gate_rows.csv` during action runs.

## LingBot

- Available from v112: memory-specific semantic cue rows for Anchor, Local Window, and Trajectory Memory.
- Missing or blocked: true Anchor source-span token attention/value hook, Local Window query-type index, trajectory retrieval hook, C1 retention/eviction hook, D1 write-admission hook.
- Consequence: LingBot Stage1 rows are frozen as cue-bank evidence only. No A2/L2/T6 runtime claim is allowed without hook repair and default-off parity.
"""
    write_text(STAGE1 / "stage1_missing_internal_cue_report.md", missing_report)

    summary = {
        "schema": "acl2_v114tf_stage1_cue_bank_summary_v1",
        "stage1_pass": bool(hs_rows and lb_rows and candidates),
        "lingbot_row_count": len(lb_rows),
        "horizonstream_row_count": len(hs_rows),
        "distribution_row_count": len(dist_rows),
        "candidate_row_count": len(candidates),
        "outputs": {
            "stage1_lingbot_evidence_quality_rows": rel(STAGE1 / "stage1_lingbot_evidence_quality_rows.csv"),
            "stage1_horizonstream_evidence_quality_rows": rel(STAGE1 / "stage1_horizonstream_evidence_quality_rows.csv"),
            "stage1_missing_internal_cue_report": rel(STAGE1 / "stage1_missing_internal_cue_report.md"),
            "stage1_quality_score_distributions": rel(STAGE1 / "stage1_quality_score_distributions.csv"),
            "stage1_candidate_surface_rank": rel(STAGE1 / "stage1_candidate_surface_rank.csv"),
        },
    }
    write_json(STAGE1 / "stage1_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
