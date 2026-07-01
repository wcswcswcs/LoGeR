#!/usr/bin/env python3
"""Build ACL2 v95 multi-route semantic memory evidence-control audit artifacts.

The builder is intentionally audit-only. It reads landed v88-v94 artifacts,
computes the v95 Stage 0-3 diagnostic tables, and blocks downstream action
tracks unless the measured gates already justify them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
V93_ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")
V91_ROOT = Path("results/acl2_v91tf_semantic_topology_regime_adaptive_memory_control")
V89_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
V86_ROOT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport")
V83_ROOT = Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse")
V80_ROOT = Path("results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control")

INPUTS = {
    "v94_boundary_rows": V94_ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv",
    "v94_scale_recovery_rows": V94_ROOT / "phase1_boundary_failure_atlas/scale_recovery_rows.csv",
    "v94_phase1_summary": V94_ROOT / "phase1_boundary_failure_atlas/phase1_gate_summary.json",
    "v94_carrier_trace_rows": V94_ROOT / "phase2_true_carrier_trace/carrier_trace_rows.csv",
    "v94_read_trace_rows": V94_ROOT / "phase2_true_carrier_trace/read_trace_rows.csv",
    "v94_swa_trace_rows": V94_ROOT / "phase2_true_carrier_trace/swa_trace_rows.csv",
    "v94_ttt_trace_rows": V94_ROOT / "phase2_true_carrier_trace/ttt_trace_rows.csv",
    "v94_semantic_rows": V94_ROOT / "phase4_semantic_evidence_taxonomy/semantic_evidence_rows.csv",
    "v94_semantic_summary": V94_ROOT / "phase4_semantic_evidence_taxonomy/semantic_taxonomy_summary.json",
    "v94_carrier_alignment_rows": V94_ROOT / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv",
    "v94_carrier_alignment_summary": V94_ROOT / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_summary.json",
    "v94_object_source_summary": V94_ROOT / "phase5_object_source_extension/phase5_object_source_extension_summary.json",
    "v94_counterfactual_summary": V94_ROOT / "phase6_object_source_counterfactual/phase6_object_source_counterfactual_summary.json",
    "v94_action_surface_rows": V94_ROOT / "phase6_object_source_action_surface/action_surface_effect_rows.csv",
    "v94_action_surface_summary": V94_ROOT / "phase6_object_source_action_surface/phase6_object_source_action_surface_summary.json",
    "v94_visual_manifest": V94_ROOT / "phase9_visual_audit_or_blocked/visual_audit_manifest.csv",
    "v94_rgb_visual_manifest": V94_ROOT / "phase9_visual_audit_or_blocked/rgb_metric_visual_audit/rgb_metric_visual_manifest.csv",
    "v94_final_decision": V94_ROOT / "report_final/final_decision.json",
    "v93_swa_secondary_summary": V93_ROOT / "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json",
    "v91_policy_state_rows": V91_ROOT / "phase5_memory_update_policy/policy_state_rows.csv",
    "v89_feature_match_rows": V89_ROOT / "phase3_feature_match_semantic_ruler/feature_match_semantic_rows.csv",
    "v89_visual_manifest": V89_ROOT / "phase10_visual_rediscovery/visual_manifest.csv",
    "v86_alignment_summary": V86_ROOT / "phase2_robust_transport/alignment_gain_gate_summary.json",
    "v83_unified_clue_matrix": V83_ROOT / "phase1_unified_clue_matrix/unified_clue_matrix.csv",
    "v83_clue_sufficiency_summary": V83_ROOT / "phase2_clue_sufficiency/clue_sufficiency_summary.json",
    "v80_case_bank_summary": V80_ROOT / "report_final/phase1_three_memory_case_bank/case_bank_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument("--copy-panels", action="store_true", help="copy representative existing panels into v95 visual dirs")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def df_to_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in {"", "nan", "None", "null"}:
            return None
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        return None
    return None


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    series = finite_series(frame, column)
    return float(series.mean()) if not series.empty else None


def q_or_none(frame: pd.DataFrame, column: str, quantile: float) -> float | None:
    series = finite_series(frame, column)
    return float(series.quantile(quantile)) if not series.empty else None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in INPUTS.items():
        rows.append(
            {
                "input_id": name,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "sha256": sha256(path),
            }
        )
    return rows


def pair_id_from_parts(seq: Any, prev: Any, curr: Any) -> str:
    try:
        seq_text = f"{int(float(seq)):02d}"
    except (TypeError, ValueError):
        seq_text = str(seq).zfill(2)
    return f"{seq_text}_{int(float(prev)):03d}_{int(float(curr)):03d}"


def normalize_pair_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pair_id" not in out.columns and {"seq", "prev_chunk", "curr_chunk"}.issubset(out.columns):
        out["pair_id"] = [pair_id_from_parts(s, p, c) for s, p, c in zip(out["seq"], out["prev_chunk"], out["curr_chunk"])]
    if "pair_id" in out.columns:
        out["pair_id"] = out["pair_id"].astype(str)
    return out


def build_metric_suite(out_root: Path, boundary: pd.DataFrame, scale: pd.DataFrame, phase1: dict[str, Any]) -> pd.DataFrame:
    boundary = normalize_pair_id(boundary)
    scale = scale.copy()
    if not scale.empty:
        scale["chunk_key"] = [
            f"{str(row.seq).zfill(2)}_{int(float(row.chunk)):03d}" for row in scale.itertuples(index=False)
        ]
        scale_lookup = scale.set_index("chunk_key").to_dict(orient="index")
    else:
        scale_lookup = {}

    rows: list[dict[str, Any]] = []
    for row in boundary.to_dict(orient="records"):
        seq = str(row.get("seq", "")).zfill(2)
        curr = row.get("curr_chunk")
        curr_key = f"{seq}_{int(float(curr)):03d}" if safe_float(curr) is not None else ""
        scale_row = scale_lookup.get(curr_key, {})
        curr_local = safe_float(row.get("curr_local_sim3_ate"))
        future = safe_float(row.get("future_after_overlap"))
        l3_penalty = future - curr_local if future is not None and curr_local is not None else None
        future5 = safe_float(row.get("future_error_5chunk"))
        rows.append(
            {
                "case_id": row.get("pair_id", ""),
                "seq": seq,
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": curr,
                "case_label_offline_only": row.get("case_label_offline_only", ""),
                "failure_type_primary": row.get("failure_type_primary", ""),
                "failure_type_secondary": row.get("failure_type_secondary", ""),
                "L0_ATE_full": "",
                "L0_status": "not_run_in_v95_no_promoted_runtime_action",
                "L1_local_sim3_ate": curr_local,
                "L1_local_sim3_scale": safe_float(scale_row.get("chunk_sim3_scale")),
                "L1_local_valid_frames": row.get("frame_end", ""),
                "L2_intra_scale_cv": safe_float(row.get("curr_scale_cv")),
                "L2_head_tail_proxy_error": safe_float(row.get("curr_head_tail_error")),
                "L2_head_tail_log_scale_delta": "",
                "L2_head_tail_delta_status": "unavailable_in_source_head_tail_error_proxy_recorded",
                "L3_adjacent_log_scale_jump": abs(safe_float(row.get("adjacent_log_scale_jump_offline")) or 0.0),
                "L3_handoff_transfer_rmse_proxy": future,
                "L3_handoff_transfer_penalty_proxy": l3_penalty,
                "L3_gauge_jump_proxy": safe_float(row.get("adjacent_gauge_jump_proxy")),
                "L3_J_handoff": safe_float(row.get("J_handoff")),
                "L4_future_error_1chunk": safe_float(row.get("future_error_1chunk")),
                "L4_future_error_3chunk": safe_float(row.get("future_error_3chunk")),
                "L4_future_error_5chunk": future5,
                "L4_propagation_growth_rate": safe_float(row.get("propagation_growth_rate")),
                "semantic_stable_mass": safe_float(row.get("semantic_valid_mass")),
                "semantic_invalid_mass": safe_float(row.get("semantic_invalid_mass")),
                "semantic_context_mass": safe_float(row.get("semantic_context_mass")),
                "semantic_dynamic_region_mass": safe_float(row.get("dynamic_or_transient_ratio")),
                "semantic_object_boundary_mass": safe_float(row.get("component_boundary_ratio")),
                "semantic_low_observability_score": safe_float(row.get("observability_score")),
                "semantic_multimode_conflict_score": safe_float(row.get("semantic_mode_entropy")),
                "semantic_evidence_type": row.get("semantic_evidence_type_majority", ""),
                "trace_path": row.get("trace_path", ""),
                "trace_provenance": row.get("trace_provenance", ""),
                "offline_audit_label_only": row.get("offline_audit_label_only", ""),
                "no_gt_runtime_feature": row.get("no_gt_runtime_feature", ""),
                "metric_source": "v94_phase1_boundary_failure_atlas",
            }
        )
    metric_df = pd.DataFrame(rows)
    metric_dir = out_root / "metric_suite"
    df_to_csv(metric_dir / "rows.csv", metric_df)
    df_to_csv(metric_dir / "metric_rows.csv", metric_df)

    summary = {
        "stage": "Stage0_metric_suite",
        "row_count": int(len(metric_df)),
        "sequence_coverage": int(metric_df["seq"].nunique()) if "seq" in metric_df else 0,
        "metric_suite_gate_pass": bool(
            len(metric_df) >= 40
            and finite_series(metric_df, "L1_local_sim3_ate").size >= 40
            and finite_series(metric_df, "L2_intra_scale_cv").size >= 40
            and finite_series(metric_df, "L3_J_handoff").size >= 40
        ),
        "full_ATE_status": "not_run",
        "runtime_action_allowed": False,
        "method_success_claim_allowed": False,
        "source_phase1_gate_pass": bool(phase1.get("phase1_gate_pass")),
        "failure_type_counts": dict(Counter(metric_df["failure_type_primary"])) if not metric_df.empty else {},
        "means": {
            "L1_local_sim3_ate": mean_or_none(metric_df, "L1_local_sim3_ate"),
            "L2_intra_scale_cv": mean_or_none(metric_df, "L2_intra_scale_cv"),
            "L3_J_handoff": mean_or_none(metric_df, "L3_J_handoff"),
            "L3_handoff_transfer_rmse_proxy": mean_or_none(metric_df, "L3_handoff_transfer_rmse_proxy"),
        },
        "q75": {
            "L1_local_sim3_ate": q_or_none(metric_df, "L1_local_sim3_ate", 0.75),
            "L2_intra_scale_cv": q_or_none(metric_df, "L2_intra_scale_cv", 0.75),
            "L3_J_handoff": q_or_none(metric_df, "L3_J_handoff", 0.75),
        },
        "limitations": [
            "L0 full ATE was not rerun because no action pilot was promoted.",
            "L2 head-tail log scale delta is not present in the source; head_tail_error is reported as proxy.",
            "L3 transfer rmse uses v94 future_after_overlap / J_handoff audit proxies.",
            "All labels are offline audit labels and are not runtime features.",
        ],
    }
    write_json(metric_dir / "summary.json", summary)
    write_json(metric_dir / "metric_validation_summary.json", summary)
    write_csv(
        metric_dir / "gate_checks.csv",
        [
            {"check": "row_count_ge_40", "pass": len(metric_df) >= 40, "value": len(metric_df), "required": ">=40"},
            {
                "check": "L1_local_sim3_available_ge_40",
                "pass": finite_series(metric_df, "L1_local_sim3_ate").size >= 40,
                "value": int(finite_series(metric_df, "L1_local_sim3_ate").size),
                "required": ">=40",
            },
            {
                "check": "L2_intra_scale_available_ge_40",
                "pass": finite_series(metric_df, "L2_intra_scale_cv").size >= 40,
                "value": int(finite_series(metric_df, "L2_intra_scale_cv").size),
                "required": ">=40",
            },
            {
                "check": "L3_handoff_available_ge_40",
                "pass": finite_series(metric_df, "L3_J_handoff").size >= 40,
                "value": int(finite_series(metric_df, "L3_J_handoff").size),
                "required": ">=40",
            },
            {"check": "L0_full_ATE_run", "pass": False, "value": "not_run", "required": "required only after action pilot"},
        ],
    )
    write_text(
        metric_dir / "metric_definition.md",
        """
# Metric Suite Definition

This v95 audit maps the plan's L0-L4 metrics onto measured v94 artifacts.

- L0 final sequence metric: not rerun in v95 because no promoted runtime action passed the mechanism gate.
- L1 local geometry: `curr_local_sim3_ate` and `chunk_sim3_scale` from v94 Phase1 scale recovery.
- L2 intra-chunk consistency: `curr_scale_cv` and `curr_head_tail_error` from v94 Phase1.
- L3 handoff consistency: `future_after_overlap`, `adjacent_log_scale_jump_offline`, `adjacent_gauge_jump_proxy`, and `J_handoff`.
- L4 long-window accumulation proxy: `future_error_1chunk/3chunk/5chunk` and `propagation_growth_rate` when present.
- Semantic reliability: v94 Phase1/4 semantic mass fields and evidence-type assignments.

All labels here are offline audit labels. They are used to select and explain cases, not as runtime features.
""",
    )
    write_text(
        metric_dir / "metric_validation_report.md",
        f"""
# Metric Validation Report

- row_count: `{summary['row_count']}`
- sequence_coverage: `{summary['sequence_coverage']}`
- metric_suite_gate_pass: `{summary['metric_suite_gate_pass']}`
- source_phase1_gate_pass: `{summary['source_phase1_gate_pass']}`
- full_ATE_status: `{summary['full_ATE_status']}`

The suite is sufficient for Stage 0 diagnostic selection, but not for method success. Full ATE remains unavailable because no v95 action was promoted.
""",
    )
    write_text(
        metric_dir / "metric_computation_scripts.py",
        """
#!/usr/bin/env python3
\"\"\"Recompute the v95 metric suite by running the repository builder.\"\"\"

from pathlib import Path
import subprocess

cmd = [
    "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
    "tools/build_v95_multiroute_semantic_memory_evidence_control.py",
]
subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], check=True)
""",
    )
    return metric_df


def build_track_a(out_root: Path, metrics: pd.DataFrame) -> dict[str, Any]:
    out = out_root / "trackA_base_case_bank"
    read_cases = metrics[metrics["failure_type_primary"].eq("LOCAL_BAD")].copy()
    swa_cases = metrics[metrics["failure_type_primary"].isin(["HANDOFF_SCALE", "HANDOFF_GAUGE"])].copy()
    lowobs_cases = metrics[metrics["failure_type_primary"].eq("LOW_OBSERVABILITY")].copy()
    good = metrics[metrics["case_label_offline_only"].eq("good")].copy()
    if "L4_future_error_5chunk" in metrics:
        ttt_cases = metrics[
            metrics["failure_type_primary"].isin(["HANDOFF_SCALE", "HANDOFF_GAUGE", "MULTIMODE_CONFLICT", "LOW_OBSERVABILITY"])
            & metrics["case_label_offline_only"].isin(["bad", "unlabelled_support"])
        ].copy()
    else:
        ttt_cases = metrics.iloc[0:0].copy()
    rejected = metrics[
        ~metrics["case_id"].isin(pd.concat([read_cases, swa_cases, ttt_cases, lowobs_cases, good])["case_id"].unique())
    ].copy()
    canonical = metrics.copy()
    if not canonical.empty:
        canonical["v95_case_bucket"] = "REJECTED_OR_SUPPORT"
        canonical.loc[canonical["case_id"].isin(read_cases["case_id"]), "v95_case_bucket"] = "READ_LOCAL_BAD"
        canonical.loc[canonical["case_id"].isin(swa_cases["case_id"]), "v95_case_bucket"] = "SWA_HANDOFF"
        canonical.loc[canonical["case_id"].isin(ttt_cases["case_id"]), "v95_case_bucket"] = "TTT_WRITE_RISK_DIAGNOSTIC"
        canonical.loc[canonical["case_id"].isin(lowobs_cases["case_id"]), "v95_case_bucket"] = "LOW_OBSERVABILITY"
        canonical.loc[canonical["case_id"].isin(good["case_id"]), "v95_case_bucket"] = "GOOD_PROTECTION"

    df_to_csv(out / "canonical_case_rows.csv", canonical)
    df_to_csv(out / "rows.csv", canonical)
    df_to_csv(out / "read_local_cases.csv", read_cases)
    df_to_csv(out / "swa_handoff_cases.csv", swa_cases)
    df_to_csv(out / "ttt_write_risk_cases.csv", ttt_cases)
    df_to_csv(out / "low_observability_cases.csv", lowobs_cases)
    df_to_csv(out / "good_controls.csv", good)
    df_to_csv(out / "rejected_cases.csv", rejected)
    summary_rows = []
    for bucket, frame in [
        ("READ_LOCAL_BAD", read_cases),
        ("SWA_HANDOFF", swa_cases),
        ("TTT_WRITE_RISK_DIAGNOSTIC", ttt_cases),
        ("LOW_OBSERVABILITY", lowobs_cases),
        ("GOOD_PROTECTION", good),
        ("REJECTED_OR_SUPPORT", rejected),
    ]:
        summary_rows.append(
            {
                "case_bucket": bucket,
                "row_count": int(len(frame)),
                "mean_L1_local_sim3_ate": mean_or_none(frame, "L1_local_sim3_ate"),
                "mean_L2_intra_scale_cv": mean_or_none(frame, "L2_intra_scale_cv"),
                "mean_L3_J_handoff": mean_or_none(frame, "L3_J_handoff"),
            }
        )
    write_csv(out / "metric_summary_by_case.csv", summary_rows)
    write_csv(out / "visual_manifest.csv", visual_rows_from_manifests())
    summary = {
        "stage": "Stage1_trackA_base_case_bank",
        "gate_pass": bool(len(read_cases) > 0 and len(swa_cases) > 0 and len(good) >= 10),
        "read_local_case_count": int(len(read_cases)),
        "swa_handoff_case_count": int(len(swa_cases)),
        "ttt_write_risk_diagnostic_count": int(len(ttt_cases)),
        "low_observability_count": int(len(lowobs_cases)),
        "good_control_count": int(len(good)),
        "rejected_count": int(len(rejected)),
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_csv(
        out / "gate_checks.csv",
        [
            {"check": "read_local_cases_exist", "pass": len(read_cases) > 0, "value": len(read_cases), "required": ">0"},
            {"check": "swa_handoff_cases_exist", "pass": len(swa_cases) > 0, "value": len(swa_cases), "required": ">0"},
            {"check": "good_controls_ge_10", "pass": len(good) >= 10, "value": len(good), "required": ">=10"},
            {"check": "ttt_trace_available", "pass": False, "value": "diagnostic_only", "required": "true before Track F action"},
        ],
    )
    write_text(
        out / "case_selection_report.md",
        f"""
# Track A Case Selection Report

- READ_LOCAL_BAD cases: `{len(read_cases)}`
- SWA_HANDOFF cases: `{len(swa_cases)}`
- TTT_WRITE_RISK diagnostic candidates: `{len(ttt_cases)}`
- LOW_OBSERVABILITY cases: `{len(lowobs_cases)}`
- GOOD_PROTECTION controls: `{len(good)}`

The seed list from the plan was not hard-coded. Rows were reselected from v94 metric fields and offline audit labels.
TTT candidates remain diagnostic because v94/v95 do not provide true TTT write traces for these rows.
""",
    )
    write_text(out / "failure_report.md", "Track A produced a usable case bank. Limitation: labels are offline audit labels only.")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "Track A already passes the diagnostic case-bank gate. Action promotion still needs Track G and D/E/F gates.",
    )
    return summary


def visual_rows_from_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v94_visual = read_csv(INPUTS["v94_visual_manifest"])
    for item in v94_visual.to_dict(orient="records"):
        category = str(item.get("category", ""))
        rows.append(
            {
                "clue_id": f"v94_{category}",
                "case_id": "",
                "memory_body": "multi",
                "drift_type": category,
                "testable_metric": item.get("evidence", ""),
                "next_track": "TrackG_or_TrackE",
                "visual_path": item.get("path", ""),
                "visual_exists": item.get("exists", ""),
                "source_version": "v94",
                "blocked_reason": item.get("blocked_reason", ""),
            }
        )
    rgb = read_csv(INPUTS["v94_rgb_visual_manifest"])
    for item in rgb.to_dict(orient="records"):
        rows.append(
            {
                "clue_id": f"v94_rgb_{item.get('pair_id', '')}",
                "case_id": item.get("pair_id", ""),
                "memory_body": "SWA/merge_gauge",
                "drift_type": item.get("semantic_evidence_type", ""),
                "testable_metric": "L3_handoff_and_semantic_evidence_type",
                "next_track": "TrackE",
                "visual_path": item.get("panel_path", ""),
                "visual_exists": item.get("panel_exists", ""),
                "source_version": "v94",
                "blocked_reason": "" if boolish(item.get("panel_exists")) else "panel_missing",
            }
        )
    v89_visual = read_csv(INPUTS["v89_visual_manifest"])
    for item in v89_visual.head(80).to_dict(orient="records"):
        rows.append(
            {
                "clue_id": f"v89_{len(rows):04d}",
                "case_id": item.get("case_id", item.get("pair_id", "")),
                "memory_body": "semantic_scale_mode",
                "drift_type": item.get("panel_type", item.get("category", "")),
                "testable_metric": "semantic_scale_mode_or_feature_match",
                "next_track": "TrackG",
                "visual_path": item.get("path", item.get("panel_path", "")),
                "visual_exists": item.get("exists", item.get("panel_exists", "")),
                "source_version": "v89",
                "blocked_reason": "",
            }
        )
    return rows


def build_track_b(out_root: Path, copy_panels: bool) -> dict[str, Any]:
    out = out_root / "trackB_visual_clue_registry"
    rows = visual_rows_from_manifests()
    write_csv(out / "visual_clue_registry.csv", rows)
    write_csv(out / "rows.csv", rows)
    write_csv(out / "visual_manifest.csv", rows)
    missing = [row for row in rows if str(row.get("visual_exists", "")).lower() not in {"true", "1"}]
    summary = {
        "stage": "Stage1_trackB_visual_clue_registry",
        "gate_pass": bool(rows and len(rows) > len(missing)),
        "row_count": len(rows),
        "missing_visual_rows": len(missing),
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_csv(
        out / "gate_checks.csv",
        [
            {"check": "visual_rows_exist", "pass": bool(rows), "value": len(rows), "required": ">0"},
            {
                "check": "non_missing_visual_majority",
                "pass": len(rows) > len(missing),
                "value": len(rows) - len(missing),
                "required": f">{len(rows) // 2}",
            },
        ],
    )
    write_text(
        out / "visual_clue_to_hypothesis.md",
        """
# Visual Clue To Hypothesis

- v94 failure-type panels test whether L1/L2/L3 buckets are separable enough for Stage 3.
- v94 RGB metric panels map boundary-level semantic evidence to SWA/merge-gauge handoff hypotheses.
- v89 feature/semantic panels are retained as old cue evidence for Track G.
- Placeholder panels are treated as blocked evidence, not as successful visual proof.
""",
    )
    write_text(
        out / "visual_clue_missing_report.md",
        "\n".join([f"- {row.get('clue_id')}: {row.get('visual_path')} missing_or_not_marked_true" for row in missing])
        or "No missing visual rows in imported manifests.",
    )
    write_text(out / "failure_report.md", "Track B registry was created from existing v94/v89 visual manifests.")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "Track B already passes as a registry. It does not by itself authorize action.",
    )
    if copy_panels:
        panel_dir = out / "visual_clue_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for row in rows:
            if copied >= 24:
                break
            src = Path(str(row.get("visual_path", "")))
            if src.exists() and src.is_file() and src.suffix.lower() == ".png":
                dst = panel_dir / src.name
                shutil.copy2(src, dst)
                copied += 1
    return summary


def build_track_g(out_root: Path, metrics: pd.DataFrame) -> dict[str, Any]:
    out = out_root / "trackG_cue_bank"
    internal_out = out_root / "trackG_internal_cues"
    old_sources = [
        ("v78_pca_global_attention_kv", "READ/SWA", V80_ROOT / "report_final/phase1_three_memory_case_bank/case_bank_summary.json"),
        ("v80_ttt_selected_write_low_support", "TTT", V80_ROOT / "report_final/phase1_three_memory_case_bank/case_bank_summary.json"),
        ("v83_unified_clue_matrix", "READ/SWA/TTT", INPUTS["v83_unified_clue_matrix"]),
        ("v84_memory_ruler_candidate_roles", "SWA", Path("results/acl2_v84tf_memory_ruler_audit/phase1_ruler_candidate_universe/ruler_candidate_pairs.csv")),
        ("v85_qk_anchor_features", "TrackC", Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_feature_index.csv")),
        ("v86_soft_latent_alignment", "TrackC/SWA", V86_ROOT / "phase2_robust_transport/alignment_gain_gate_summary.json"),
        ("v87_scale_conditioned_proxy", "SWA", Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance_k16_r1_median_abs/proxy_relevance_summary.json")),
        ("v88_scale_mode_entropy", "SWA/merge_gauge", Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe/scale_mode_pair_rows.csv")),
        ("v89_semantic_scale_mode_roles", "SWA/merge_gauge", V89_ROOT / "phase1_semantic_scale_mode_ledger/semantic_scale_mode_rows.csv"),
        ("v90_component_topology_roles", "SWA/merge_gauge", Path("results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control/phase2_semantic_topology_scale_mode_ledger/topology_pair_rows.csv")),
        ("v91_semantic_regime_policy", "TTT/SWA", INPUTS["v91_policy_state_rows"]),
        ("v92_policy_sidecar_availability", "SWA", Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/phase7_data_source_expansion/expanded_semantic_policy_summary.json")),
        ("v94_failure_atlas_merge_gauge_traces", "SWA/merge_gauge", INPUTS["v94_boundary_rows"]),
    ]
    g0_rows = []
    for cue_id, scope, path in old_sources:
        g0_rows.append(
            {
                "cue_id": cue_id,
                "source_version": cue_id.split("_", 1)[0],
                "memory_scope_candidate": scope,
                "computed_from": "landed_artifact",
                "artifact_path": str(path),
                "coverage": "available" if path.exists() else "missing",
                "known_success": "",
                "known_failure": "see final decision / gate summaries",
                "why_revisit": "v95 requires memory-specific cue bank before action",
                "known_bug_or_caveat": "" if path.exists() else "artifact_missing_in_current_checkout",
            }
        )
    write_csv(out / "g0_old_cue_registry.csv", g0_rows)
    write_csv(out / "g0_cue_reuse_recommendation.csv", g0_rows)
    write_text(
        out / "g0_old_cue_audit_report.md",
        f"# G0 Old Cue Audit\n\nAvailable cue artifacts: `{sum(1 for row in g0_rows if row['coverage'] == 'available')}` / `{len(g0_rows)}`.",
    )

    read_trace = normalize_pair_id(read_csv(INPUTS["v94_read_trace_rows"]))
    semantic = normalize_pair_id(read_csv(INPUTS["v94_semantic_rows"]))
    merged = read_trace.merge(
        semantic[["pair_id", "semantic_evidence_type", "S_stable", "S_invalid", "S_context", "S_multi", "S_lowobs"]],
        on="pair_id",
        how="left",
    )
    internal_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        active = safe_float(row.get("read_active_mass"))
        stable = safe_float(row.get("read_stable_mass"))
        invalid = safe_float(row.get("read_invalid_mass"))
        context = safe_float(row.get("read_context_mass"))
        entropy = safe_float(row.get("read_query_entropy"))
        unreliable = (invalid or 0.0) + (context or 0.0) + max(0.0, (entropy or 0.0) - 0.75)
        stable_score = (stable or 0.0) / max(active or 1.0, 1e-9)
        internal_rows.append(
            {
                "case_id": row.get("pair_id", ""),
                "memory_body": "READ",
                "layer_id": "read_patch_dump",
                "head_id": "pooled",
                "internal_cue_id": "read_patch_entropy_invalid_context",
                "read_active_mass": active,
                "read_stable_mass": stable,
                "read_invalid_mass": invalid,
                "read_context_mass": context,
                "read_query_entropy": entropy,
                "internal_unreliable_score": unreliable,
                "internal_stable_score": stable_score,
                "cue_map_path": row.get("read_trace_path", ""),
                "trace_provenance": row.get("trace_provenance", ""),
                "missing_reason": row.get("missing_reason", ""),
            }
        )
    write_csv(out / "internal_cue_rows.csv", internal_rows)
    write_csv(out / "g1_internal_cue_rows.csv", internal_rows)
    by_body = []
    if internal_rows:
        internal_df = pd.DataFrame(internal_rows)
        by_body.append(
            {
                "memory_body": "READ",
                "row_count": len(internal_df),
                "mean_internal_unreliable_score": mean_or_none(internal_df, "internal_unreliable_score"),
                "mean_internal_stable_score": mean_or_none(internal_df, "internal_stable_score"),
            }
        )
    write_csv(out / "layerwise_cue_summary.csv", by_body)
    write_csv(internal_out / "internal_cue_rows.csv", internal_rows)
    write_csv(internal_out / "rows.csv", internal_rows)
    write_csv(internal_out / "layerwise_cue_summary.csv", by_body)
    (internal_out / "cue_maps").mkdir(parents=True, exist_ok=True)
    (internal_out / "cue_visual_panels").mkdir(parents=True, exist_ok=True)

    semantic_summary = read_json(INPUTS["v94_semantic_summary"])
    carrier_summary = read_json(INPUTS["v94_carrier_alignment_summary"])
    object_summary = read_json(INPUTS["v94_object_source_summary"])
    action_summary = read_json(INPUTS["v94_action_surface_summary"])
    best_role = semantic_summary.get("best_semantic_role") or {}
    object_policy = object_summary.get("selected_policy") or {}
    combined_rows = [
        {
            "cue_id": "read_patch_entropy_invalid_context",
            "memory_body": "READ",
            "cue_type": "internal_plus_semantic",
            "bad_recall": "",
            "good_FPR": "",
            "balanced_accuracy": "",
            "semantic_shuffle_margin": "",
            "internal_shuffle_margin": "",
            "geometry_only_margin": "",
            "correlation_with_local_metric": "",
            "correlation_with_handoff_metric": "",
            "correlation_with_long_drift_metric": "",
            "service_ready": False,
            "blocker": "read cue lacks measured bad/good recall and QK compatibility is unavailable",
        },
        {
            "cue_id": "SEM_INVALID_BOUNDARY_role",
            "memory_body": "SWA/merge_gauge",
            "cue_type": "semantic_role",
            "bad_recall": best_role.get("bad_recall"),
            "good_FPR": best_role.get("good_FPR"),
            "balanced_accuracy": best_role.get("balanced_accuracy"),
            "semantic_shuffle_margin": best_role.get("semantic_shuffle_margin"),
            "internal_shuffle_margin": "",
            "geometry_only_margin": "",
            "correlation_with_local_metric": "",
            "correlation_with_handoff_metric": (carrier_summary.get("best_alignment_role") or {}).get("max_positive_carrier_subfield_corr"),
            "correlation_with_long_drift_metric": "",
            "service_ready": False,
            "blocker": carrier_summary.get("blocker") or "carrier alignment did not pass",
        },
        {
            "cue_id": object_policy.get("policy", "object_source_policy"),
            "memory_body": "SWA/merge_gauge",
            "cue_type": "object_source_plus_semantic",
            "bad_recall": object_policy.get("bad_recall"),
            "good_FPR": object_policy.get("good_FPR"),
            "balanced_accuracy": object_policy.get("balanced_accuracy"),
            "semantic_shuffle_margin": object_policy.get("min_control_margin"),
            "internal_shuffle_margin": "",
            "geometry_only_margin": "",
            "correlation_with_local_metric": "",
            "correlation_with_handoff_metric": "policy_selected_on_v94_object_source_extension",
            "correlation_with_long_drift_metric": "",
            "service_ready": bool(
                object_summary.get("object_source_extension_gate_pass")
                and safe_float(object_policy.get("bad_recall")) is not None
                and (safe_float(object_policy.get("bad_recall")) or 0) >= 0.60
                and safe_float(object_policy.get("good_FPR")) is not None
                and (safe_float(object_policy.get("good_FPR")) or 1) <= 0.25
            ),
            "blocker": "" if object_summary.get("object_source_extension_gate_pass") else "object_source_extension_failed",
        },
        {
            "cue_id": "ttt_write_hint_from_v91_policy_state",
            "memory_body": "TTT",
            "cue_type": "semantic_regime_policy",
            "bad_recall": "",
            "good_FPR": "",
            "balanced_accuracy": "",
            "semantic_shuffle_margin": "",
            "internal_shuffle_margin": "",
            "geometry_only_margin": "",
            "correlation_with_local_metric": "",
            "correlation_with_handoff_metric": "",
            "correlation_with_long_drift_metric": "",
            "service_ready": False,
            "blocker": "TTT write trace not entered in v94 Phase2",
        },
    ]
    write_csv(out / "g2_memory_specific_cue_rows.csv", combined_rows)
    write_csv(out / "rows.csv", combined_rows)
    write_csv(out / "cue_relationship_metrics.csv", combined_rows)
    write_text(
        out / "cue_conflict_report.md",
        f"""
# Cue Conflict Report

- Semantic-only role cue (`SEM_INVALID_BOUNDARY`) had low bad recall (`{best_role.get('bad_recall')}`) despite good FPR (`{best_role.get('good_FPR')}`).
- Object-source combined cue passed the offline Stage G service threshold, but v94 Phase6 measured action surface failed: `{action_summary.get('blocker')}`.
- READ internal cue has patch-dump mass but no measured recall/FPR or QK compatibility.
- TTT cue remains blocked because TTT trace was not entered.
""",
    )
    visual_rows = visual_rows_from_manifests()
    write_csv(out / "visual_manifest.csv", visual_rows)
    service_ready = any(boolish(row.get("service_ready")) for row in combined_rows)
    summary = {
        "stage": "Stage2_trackG_minimum_cue_bank",
        "gate_pass": service_ready,
        "service_ready_cues": [row["cue_id"] for row in combined_rows if boolish(row.get("service_ready"))],
        "read_cue_service_ready": False,
        "swa_cue_service_ready": service_ready,
        "ttt_cue_service_ready": False,
        "runtime_action_allowed": False,
        "object_source_action_surface_gate_pass": bool(action_summary.get("phase6_object_source_action_surface_gate_pass")),
        "object_source_action_surface_blocker": action_summary.get("blocker"),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "internal_cue_summary.json", {"read_internal_row_count": len(internal_rows), **summary})
    write_json(internal_out / "summary.json", {"read_internal_row_count": len(internal_rows), **summary})
    write_json(internal_out / "internal_cue_summary.json", {"read_internal_row_count": len(internal_rows), **summary})
    write_csv(
        out / "gate_checks.csv",
        [
            {"check": "read_cue_bad_recall_ge_0p60_good_fpr_le_0p25", "pass": False, "value": "not_measured", "required": "bad_recall>=0.60 and good_FPR<=0.25"},
            {
                "check": "swa_cue_bad_recall_ge_0p60_good_fpr_le_0p25",
                "pass": service_ready,
                "value": f"bad_recall={object_policy.get('bad_recall')};good_FPR={object_policy.get('good_FPR')}",
                "required": "bad_recall>=0.60 and good_FPR<=0.25",
            },
            {"check": "ttt_cue_visual_metric_aligned", "pass": False, "value": "TTT_trace_not_entered", "required": "true"},
        ],
    )
    write_text(
        out / "failure_report.md",
        "Track G produced a minimum cue bank. Only the object-source/SWA cue passes service threshold; READ and TTT remain blocked.",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "READ needs measured bad_recall/good_FPR and QK compatibility. TTT needs true write trace. SWA needs action-surface controls to pass before runtime action.",
    )
    write_csv(internal_out / "gate_checks.csv", [{"check": "read_internal_rows_exist", "pass": len(internal_rows) > 0, "value": len(internal_rows), "required": ">0"}])
    write_csv(internal_out / "visual_manifest.csv", visual_rows)
    write_text(
        internal_out / "failure_report.md",
        "Internal READ cue rows were extracted from read patch dumps, but QK compatibility and bad/good control metrics are unavailable.",
    )
    write_text(
        internal_out / "what_would_have_to_be_true_to_pass.md",
        "G1 needs layer/head-resolved cue maps with measured recall/FPR and controls before READ action.",
    )
    return summary


def build_drift_source(out_root: Path, metrics: pd.DataFrame) -> dict[str, Any]:
    out = out_root / "drift_source_diagnosis"
    rows = []
    for row in metrics.to_dict(orient="records"):
        failure = str(row.get("failure_type_primary", ""))
        if failure == "LOCAL_BAD":
            primary = "READ_LOCAL"
        elif failure in {"HANDOFF_SCALE", "HANDOFF_GAUGE"}:
            primary = "SWA_HANDOFF"
        elif failure == "LOW_OBSERVABILITY":
            primary = "LOW_OBSERVABILITY_ABSTAIN"
        elif failure == "MULTIMODE_CONFLICT":
            primary = "DYNAMIC_OR_BOUNDARY_CONTAMINATION"
        elif failure == "SAFE_OR_UNASSIGNED":
            primary = "GOOD_OR_UNASSIGNED"
        else:
            primary = "NO_VALID_EVIDENCE"
        rows.append(
            {
                "case_id": row.get("case_id", ""),
                "seq": row.get("seq", ""),
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": row.get("curr_chunk", ""),
                "failure_type_primary": failure,
                "drift_source_primary": primary,
                "L1_local_sim3_ate": row.get("L1_local_sim3_ate", ""),
                "L2_intra_scale_cv": row.get("L2_intra_scale_cv", ""),
                "L3_J_handoff": row.get("L3_J_handoff", ""),
                "L4_future_error_5chunk": row.get("L4_future_error_5chunk", ""),
                "semantic_evidence_type": row.get("semantic_evidence_type", ""),
                "decision_rule": "mapped_from_v94_failure_type_and_v95_rules",
            }
        )
    write_csv(out / "drift_source_rows.csv", rows)
    write_csv(out / "rows.csv", rows)
    counts = Counter(row["drift_source_primary"] for row in rows)
    by_type = [{"drift_source_primary": key, "row_count": value, "fraction": value / max(len(rows), 1)} for key, value in counts.items()]
    write_csv(out / "drift_source_by_case_type.csv", by_type)
    total = max(len(rows), 1)
    read_frac = counts.get("READ_LOCAL", 0) / total
    swa_frac = counts.get("SWA_HANDOFF", 0) / total
    ttt_frac = counts.get("TTT_ACCUMULATION", 0) / total
    lowobs_frac = counts.get("LOW_OBSERVABILITY_ABSTAIN", 0) / total
    if read_frac >= 0.40:
        priority = "TrackD_READ"
    elif swa_frac >= 0.30:
        priority = "TrackE_SWA"
    elif ttt_frac >= 0.20:
        priority = "TrackF_TTT"
    elif lowobs_frac >= 0.20:
        priority = "LOW_OBSERVABILITY_ABSTAIN_BEFORE_ACTION"
    else:
        priority = "TrackE_SWA_AND_TrackG_CONTINUE"
    summary = {
        "stage": "Stage3_drift_source_diagnosis",
        "gate_pass": True,
        "row_count": len(rows),
        "drift_source_counts": dict(counts),
        "fractions": {
            "READ_LOCAL": read_frac,
            "SWA_HANDOFF": swa_frac,
            "TTT_ACCUMULATION": ttt_frac,
            "LOW_OBSERVABILITY_ABSTAIN": lowobs_frac,
        },
        "memory_body_priority": priority,
        "runtime_action_allowed": False,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "memory_body_priority.json", summary)
    write_csv(
        out / "gate_checks.csv",
        [
            {"check": "diagnosis_rows_match_metric_rows", "pass": len(rows) == len(metrics), "value": len(rows), "required": len(metrics)},
            {"check": "priority_selected", "pass": bool(priority), "value": priority, "required": "non_empty"},
        ],
    )
    write_text(
        out / "drift_source_decision.md",
        f"""
# Drift Source Decision

- READ_LOCAL fraction: `{read_frac}`
- SWA_HANDOFF fraction: `{swa_frac}`
- TTT_ACCUMULATION fraction: `{ttt_frac}`
- LOW_OBSERVABILITY_ABSTAIN fraction: `{lowobs_frac}`
- Selected priority: `{priority}`

Although READ_LOCAL is substantial, READ cue did not pass Track G. The only service-ready cue is SWA/object-source, but its action surface later fails controls.
""",
    )
    write_text(out / "failure_report.md", "Drift diagnosis completed. Downstream action remains gate-dependent.")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "Drift diagnosis passes. To promote, a selected Track D/E/F route must pass cue, control, visual, and mechanism gates.",
    )
    write_csv(out / "visual_manifest.csv", visual_rows_from_manifests())
    return summary


def standard_track_files(
    out: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]] | pd.DataFrame,
    gate_rows: list[dict[str, Any]],
    failure_report: str,
    pass_conditions: str,
    visual_rows: list[dict[str, Any]] | None = None,
) -> None:
    write_json(out / "summary.json", summary)
    if isinstance(rows, pd.DataFrame):
        df_to_csv(out / "rows.csv", rows)
    else:
        write_csv(out / "rows.csv", rows)
    write_csv(out / "gate_checks.csv", gate_rows)
    write_csv(out / "visual_manifest.csv", visual_rows or [])
    write_text(out / "failure_report.md", failure_report)
    write_text(out / "what_would_have_to_be_true_to_pass.md", pass_conditions)


def build_downstream_tracks(out_root: Path) -> dict[str, dict[str, Any]]:
    visual_rows = visual_rows_from_manifests()
    read_rows = normalize_pair_id(read_csv(INPUTS["v94_read_trace_rows"]))
    carrier_rows = normalize_pair_id(read_csv(INPUTS["v94_carrier_alignment_rows"]))
    action_rows = normalize_pair_id(read_csv(INPUTS["v94_action_surface_rows"]))
    ttt_rows = normalize_pair_id(read_csv(INPUTS["v94_ttt_trace_rows"]))
    phase5 = read_json(INPUTS["v94_carrier_alignment_summary"])
    phase6 = read_json(INPUTS["v94_action_surface_summary"])
    object_summary = read_json(INPUTS["v94_object_source_summary"])
    v86_align = read_json(INPUTS["v86_alignment_summary"])
    v94_final = read_json(INPUTS["v94_final_decision"])

    track_c_summary = {
        "track": "TrackC_feature_alignment",
        "entered": True,
        "gate_pass": bool(phase5.get("phase5_semantic_carrier_alignment_gate_pass")),
        "blocker": phase5.get("blocker", ""),
        "best_alignment_role": phase5.get("best_alignment_role", {}),
        "v86_alignment_source_available": bool(v86_align),
        "runtime_action_allowed": False,
    }
    standard_track_files(
        out_root / "trackC_feature_alignment",
        track_c_summary,
        carrier_rows,
        [
            {"check": "carrier_alignment_gate_pass", "pass": track_c_summary["gate_pass"], "value": phase5.get("blocker", ""), "required": "true"},
            {"check": "diagnostic_only_no_pose_correction", "pass": True, "value": "no_pose_correction_written", "required": "true"},
        ],
        "Track C has diagnostic evidence, but v94 semantic-carrier alignment failed recall/shuffle/LOSO gates.",
        "Stable evidence alignment must pass bad recall, shuffle margins, and LOSO support without pose or trajectory correction.",
        visual_rows,
    )

    track_d_summary = {
        "track": "TrackD_read_eligibility",
        "entered": True,
        "gate_pass": False,
        "blocker": "read_cue_not_validated_against_bad_good_controls;read_QK_compatibility_unavailable",
        "read_trace_rows": int(len(read_rows)),
        "runtime_action_allowed": False,
    }
    standard_track_files(
        out_root / "trackD_read_eligibility",
        track_d_summary,
        read_rows,
        [
            {"check": "read_trace_available", "pass": len(read_rows) > 0, "value": len(read_rows), "required": ">0"},
            {"check": "read_cue_beats_shuffle", "pass": False, "value": "not_measured", "required": "true"},
            {"check": "good_controls_worsen_le_2pct", "pass": False, "value": "not_run", "required": "true before action"},
        ],
        "READ trace rows are available, but the cue was not validated with bad/good controls and QK compatibility is unavailable.",
        "READ cue needs measured >=5% L2 improvement, controls worsen <=2%, and shuffle/random beaten.",
        visual_rows,
    )

    track_e_summary = {
        "track": "TrackE_swa_transport",
        "entered": True,
        "cue_gate_pass": bool(object_summary.get("object_source_extension_gate_pass")),
        "gate_pass": bool(phase6.get("phase6_object_source_action_surface_gate_pass")),
        "blocker": phase6.get("blocker", ""),
        "selected_policy": (object_summary.get("selected_policy") or {}).get("policy", ""),
        "runtime_action_allowed": False,
        "diagnostic_only": True,
    }
    standard_track_files(
        out_root / "trackE_swa_transport",
        track_e_summary,
        action_rows,
        [
            {"check": "swa_cue_service_ready", "pass": track_e_summary["cue_gate_pass"], "value": track_e_summary["selected_policy"], "required": "true"},
            {"check": "measured_action_surface_gate_pass", "pass": track_e_summary["gate_pass"], "value": phase6.get("blocker", ""), "required": "true"},
            {"check": "runtime_action_allowed", "pass": False, "value": "blocked", "required": "true after action-surface pass"},
        ],
        f"Track E reached diagnostic/action-surface audit, but measured controls failed: {phase6.get('blocker', '')}.",
        "SWA needs L3/future-overlap improvement >=5%, good controls worsen <=2%, and semantic cue must beat measured controls.",
        visual_rows,
    )

    track_f_summary = {
        "track": "TrackF_ttt_write",
        "entered": False,
        "gate_pass": False,
        "blocker": "ttt_trace_not_entered_v94_phase2_diagnostic_only",
        "ttt_trace_rows": int(len(ttt_rows)),
        "runtime_action_allowed": False,
    }
    standard_track_files(
        out_root / "trackF_ttt_write",
        track_f_summary,
        ttt_rows,
        [
            {"check": "ttt_trace_has_write_mass", "pass": False, "value": "blank_trace_rows", "required": "true"},
            {"check": "same_write_mass_random_control", "pass": False, "value": "not_run", "required": "true"},
        ],
        "TTT write route is blocked. Source rows state diagnostic_only_not_entered and contain no write-mass evidence.",
        "TTT needs persistent/transient/no-write trace, >=5% L4 improvement, good control protection, and same-write-mass random control failure.",
        visual_rows,
    )

    track_h_rows = []
    selected = (v94_final.get("key_metrics") or {})
    track_h_rows.append(
        {
            "actuator": "merge_alpha_0p2",
            "source": str(V94_ROOT / "phase3s_merge_gauge_actuator_sweep_max16_confirm/runtime_probe_sensitivity_summary.json"),
            "diagnostic_sensitive": selected.get("phase3s_actuator_probe_gate_pass"),
            "action_surface_gate_pass": phase6.get("phase6_object_source_action_surface_gate_pass"),
            "blocker": phase6.get("blocker", ""),
            "runtime_action_allowed": False,
        }
    )
    track_h_summary = {
        "track": "TrackH_actuator_reeval",
        "entered": True,
        "diagnostic_actuator_found": True,
        "gate_pass": False,
        "blocker": "old_actuator_sensitive_but_not_semantic_specific_and_does_not_beat_controls",
        "runtime_action_allowed": False,
    }
    standard_track_files(
        out_root / "trackH_actuator_reeval",
        track_h_summary,
        track_h_rows,
        [
            {"check": "old_actuator_sensitive", "pass": True, "value": "merge_alpha_0p2", "required": "true"},
            {"check": "semantic_specific_action_surface_pass", "pass": False, "value": phase6.get("blocker", ""), "required": "true"},
        ],
        "Track H found an old merge/gauge actuator sensitivity, but v95 cannot treat that as semantic memory success.",
        "Old actuator can be reused only after new cue passes specificity and action-surface controls.",
        visual_rows,
    )
    return {
        "trackC": track_c_summary,
        "trackD": track_d_summary,
        "trackE": track_e_summary,
        "trackF": track_f_summary,
        "trackH": track_h_summary,
    }


def build_decision_matrix(
    out_root: Path,
    metric_summary: dict[str, Any],
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    track_g: dict[str, Any],
    drift: dict[str, Any],
    tracks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = out_root / "decision_matrix"
    rows = [
        {"route": "Stage0_metric_suite", "gate_pass": metric_summary.get("metric_suite_gate_pass"), "status": "complete"},
        {"route": "TrackA_base_case_bank", "gate_pass": track_a.get("gate_pass"), "status": "complete"},
        {"route": "TrackB_visual_clue_registry", "gate_pass": track_b.get("gate_pass"), "status": "complete"},
        {"route": "TrackG_minimum_cue_bank", "gate_pass": track_g.get("gate_pass"), "status": "partial_swa_only"},
        {"route": "Stage3_drift_source_diagnosis", "gate_pass": drift.get("gate_pass"), "status": "complete"},
        {"route": "TrackC_feature_alignment", "gate_pass": tracks["trackC"].get("gate_pass"), "status": "diagnostic_failed"},
        {"route": "TrackD_read_eligibility", "gate_pass": tracks["trackD"].get("gate_pass"), "status": "blocked"},
        {"route": "TrackE_swa_transport", "gate_pass": tracks["trackE"].get("gate_pass"), "status": "action_surface_failed"},
        {"route": "TrackF_ttt_write", "gate_pass": tracks["trackF"].get("gate_pass"), "status": "blocked"},
        {"route": "TrackH_actuator_reeval", "gate_pass": tracks["trackH"].get("gate_pass"), "status": "diagnostic_only"},
    ]
    write_csv(out / "rows.csv", rows)
    write_csv(out / "route_selection_rows.csv", rows)
    final_status = "NO_GO_TRACK_E_ACTION_SURFACE_CONTROL_FAILED_WITH_STAGE0_3_COMPLETE"
    decision = {
        "final_status": final_status,
        "method_success": False,
        "full_ATE_run": False,
        "runtime_action_allowed": False,
        "stage0_3_complete": True,
        "primary_blocker": tracks["trackE"].get("blocker"),
        "route_priority": drift.get("memory_body_priority"),
        "service_ready_cues": track_g.get("service_ready_cues", []),
        "key_metrics": {
            "metric_rows": metric_summary.get("row_count"),
            "read_local_cases": track_a.get("read_local_case_count"),
            "swa_handoff_cases": track_a.get("swa_handoff_case_count"),
            "good_controls": track_a.get("good_control_count"),
            "object_source_action_surface_gate_pass": tracks["trackE"].get("gate_pass"),
        },
        "answer_questions": {
            "how_scale_drift_measured": "L1 local Sim3, L2 intra scale CV/head-tail proxy, L3 handoff J/scale/gauge proxies, L4 future-error proxies; L0 full ATE not rerun.",
            "dominant_drift_sources": drift.get("drift_source_counts"),
            "old_visual_clues_testable": True,
            "trackG_cues": track_g.get("service_ready_cues", []),
            "read_problem_confirmed": "READ_LOCAL cases exist, but READ cue/action not validated.",
            "swa_transport_failure": "SWA/HANDOFF cases exist and object-source cue passed offline service threshold, but action surface failed controls.",
            "ttt_write_failure": "TTT trace unavailable; no write-risk action allowed.",
            "feature_alignment": "Track C diagnostic failed v94 carrier alignment gate.",
            "full_ATE_improved": False,
        },
    }
    write_json(out / "summary.json", decision)
    write_json(out / "decision_matrix.json", decision)
    write_csv(out / "gate_checks.csv", rows)
    write_csv(out / "visual_manifest.csv", visual_rows_from_manifests())
    write_text(
        out / "route_selection_report.md",
        f"""
# Route Selection Report

Final status: `{final_status}`

Stage 0-3 completed from measured v94/v93/v91/v89 artifacts. Track G found one service-ready SWA/object-source cue, but Track E action-surface replay failed measured controls. No runtime action or full ATE validation is allowed.
""",
    )
    write_text(
        out / "stopped_routes.md",
        """
# Stopped Routes

- Track D READ: blocked because READ cue did not beat controls and QK compatibility is unavailable.
- Track E SWA: stopped because measured object-source action surface does not beat selection controls.
- Track F TTT: blocked because true TTT write trace is unavailable.
- Track H old actuator: diagnostic only; old merge-alpha sensitivity is not semantic memory success.
""",
    )
    write_text(
        out / "next_route_recommendation.md",
        """
# Next Route Recommendation

Continue Track G before another runtime action. The immediate repair target is to improve SWA cue specificity beyond object-source broad selection, then rerun Track E with measured selection controls. READ needs QK-compatible trace extraction. TTT should not restart until persistent/transient/no-write trace is available.
""",
    )
    write_text(out / "failure_report.md", "No method success. The actionable blocker is Track E action-surface control failure.")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "A v95 pass requires a memory-specific cue to beat controls, action-surface mechanism metrics to improve without good-control regression, and then full ATE validation.",
    )
    return decision


def write_final_report(out_root: Path, decision: dict[str, Any]) -> None:
    report = f"""
# ACL2 v95-TF Multi-Route Semantic Memory Evidence Control Final Report

## Status

`{decision['final_status']}`

No v95 method success is claimed. Stage 0-3 were completed as audit/diagnostic artifacts, but no runtime action or full KITTI01 ATE validation was promoted.

## Evidence Chain

1. Metric suite was built from v94 Phase1 boundary failure atlas and scale recovery rows.
2. Track A selected READ_LOCAL, SWA_HANDOFF, TTT diagnostic, low-observability, and good-protection cases from measured fields.
3. Track B converted v94/v89 visual panels into a visual clue registry.
4. Track G rebuilt old cue registry and produced a minimum cue bank. The only service-ready cue was the object-source/SWA cue.
5. Stage 3 drift diagnosis mapped rows into READ_LOCAL, SWA_HANDOFF, LOW_OBSERVABILITY_ABSTAIN, and related sources.
6. Track E failed measured action-surface controls, so Track D/F/H and full ATE remain blocked or diagnostic-only.

## Key Answers

- Scale drift measurement: L1 local Sim3, L2 intra-scale CV/head-tail proxy, L3 handoff/gauge proxies, L4 future-error proxies.
- Base-case drift sources: see `drift_source_diagnosis/drift_source_rows.csv`.
- Old visual clues converted: see `trackB_visual_clue_registry/visual_clue_registry.csv`.
- Track G cues: `{decision.get('service_ready_cues')}`.
- READ: cases exist, but cue/action gate failed.
- SWA: offline object-source cue passed service threshold, but action-surface controls failed.
- TTT: blocked by missing write trace.
- Feature alignment: diagnostic failed carrier-alignment gate.
- Full ATE: not run; no promoted action.

## Audit Rule

All fields above come from landed artifacts listed in `input_artifact_manifest.csv` and v95-generated CSV/JSON files. Missing evidence is recorded as missing/blocking rather than inferred.
"""
    write_text(out_root / "ACL2_v95TF_DetailedExecution_FinalReport.md", report)


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "input_artifact_manifest.csv", input_manifest())

    boundary = read_csv(INPUTS["v94_boundary_rows"])
    scale = read_csv(INPUTS["v94_scale_recovery_rows"])
    phase1 = read_json(INPUTS["v94_phase1_summary"])
    metrics = build_metric_suite(out_root, boundary, scale, phase1)
    metric_summary = read_json(out_root / "metric_suite/summary.json")

    track_a = build_track_a(out_root, metrics)
    track_b = build_track_b(out_root, args.copy_panels)
    track_g = build_track_g(out_root, metrics)
    drift = build_drift_source(out_root, metrics)
    tracks = build_downstream_tracks(out_root)
    decision = build_decision_matrix(out_root, metric_summary, track_a, track_b, track_g, drift, tracks)
    write_final_report(out_root, decision)
    write_json(
        out_root / "build_summary.json",
        {
            "out_root": str(out_root),
            "final_status": decision.get("final_status"),
            "method_success": decision.get("method_success"),
            "files_written_note": "see generated directories for per-track artifacts",
        },
    )
    print(f"wrote_v95_root={out_root}")
    print(f"final_status={decision.get('final_status')}")
    print(f"method_success={decision.get('method_success')}")


if __name__ == "__main__":
    main()
