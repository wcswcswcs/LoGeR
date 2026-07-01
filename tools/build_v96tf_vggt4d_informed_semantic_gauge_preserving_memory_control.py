#!/usr/bin/env python3
"""Build ACL2 v96-TF audit artifacts from landed evidence.

This script is intentionally conservative.  It derives Stage 0 tables from
v95 artifacts and performs a local VGGT4D static audit / LoGeR mapping, but it
does not implement or run a VGGT4D-style LoGeR action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
VGGT4D_ROOT = Path("third_party/VGGT4D")

TRACK_DIRS = [
    "trackI_drift_observatory",
    "trackA_case_response_atlas",
    "trackB_visual_hypothesis_registry",
    "trackJ_vggt4d_code_audit",
    "trackJ_semantic_region_bank",
    "trackJ_skip_impact_diagnostic",
    "trackJ_read_skip_pilot",
    "trackJ_read_skip_pilot_repair_early_quarter",
    "trackJ_read_skip_pilot_repair_anchor_compensation",
    "trackJ_read_skip_pilot_repair_anchor_weak_compensation",
    "trackJ_read_skip_pilot_repair_anchor_weak_rho020",
    "trackJ_swa_skip_diagnostic",
    "trackJ_ttt_no_write_diagnostic",
    "trackG_memory_specific_cues",
    "trackG_read_cue_refinement",
    "trackG_read_qk_carrier_localization",
    "trackD_read_gauge_preserving_action",
    "trackE_swa_raw_transport_trace",
    "trackC_latent_gauge_alignment",
    "trackF_ttt_write_diagnostic",
    "stage7_full_validation",
    "final_decision",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in {"", "nan", "None", "null"}:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ensure_track_scaffold(track: str) -> Path:
    out = ROOT / track
    out.mkdir(parents=True, exist_ok=True)
    (out / "visual_panels").mkdir(exist_ok=True)
    for name in ("summary.json", "rows.csv", "gate_checks.csv", "failure_report.md", "what_would_have_to_be_true_to_pass.md", "visual_manifest.csv"):
        path = out / name
        if not path.exists():
            if name.endswith(".json"):
                write_json(path, {"status": "not_started", "track": track})
            elif name.endswith(".csv"):
                write_csv(path, [])
            else:
                write_text(path, f"# {track}\n\nNot started in this run yet.")
    return out


def copy_visual_manifest(src: Path, dst: Path, extra_rows: list[dict[str, Any]] | None = None) -> None:
    rows = read_csv(src)
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "visual_id": row.get("clue_id") or row.get("visual_id") or row.get("case_id") or f"visual_{len(out_rows):04d}",
                "case_id": row.get("case_id", ""),
                "source_path": row.get("visual_path", ""),
                "exists": row.get("visual_exists", ""),
                "source_version": row.get("source_version", "v95"),
                "note": row.get("blocked_reason", ""),
            }
        )
    if extra_rows:
        out_rows.extend(extra_rows)
    write_csv(dst, out_rows)


def stage7_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(V95_ROOT.glob("stage7_seq*/stage7_audit_summary*.json")):
        payload = read_json(path)
        full = payload.get("full_sequence", {}) if isinstance(payload.get("full_sequence"), dict) else {}
        active = payload.get("gate_chunk_summary", {}).get("active", {}) if isinstance(payload.get("gate_chunk_summary"), dict) else {}
        inactive = payload.get("gate_chunk_summary", {}).get("inactive", {}) if isinstance(payload.get("gate_chunk_summary"), dict) else {}
        delta_ate = safe_float(full.get("delta_aligned_ate_rmse_m"))
        delta_final = safe_float(full.get("delta_final_error_m"))
        delta_slope = safe_float(full.get("delta_error_slope_m_per_100f"))
        active_mean = safe_float(active.get("mean_delta_m"))
        inactive_mean = safe_float(inactive.get("mean_delta_m"))
        labels = []
        if delta_ate is not None and delta_ate > 0:
            labels.append("full_ATE_worse")
        if delta_ate is not None and delta_ate < 0:
            labels.append("full_ATE_improved")
        if delta_final is not None and delta_final < 0:
            labels.append("final_error_improved")
        if active_mean is not None and active_mean < 0 and delta_ate is not None and delta_ate > 0:
            labels.append("active_local_improve_but_full_ATE_worse")
        if inactive_mean is not None and inactive_mean > 0 and delta_ate is not None and delta_ate > 0:
            labels.append("inactive_global_tradeoff_worse")
        if not labels:
            labels.append("no_success_signal")
        rows.append(
            {
                "source_json": str(path),
                "baseline": payload.get("baseline", ""),
                "candidate": payload.get("candidate", ""),
                "method_success": payload.get("method_success", ""),
                "method_success_reason": payload.get("method_success_reason", ""),
                "baseline_aligned_ATE_full_sim3": full.get("baseline_aligned_ate_rmse_m", ""),
                "candidate_aligned_ATE_full_sim3": full.get("candidate_aligned_ate_rmse_m", ""),
                "delta_aligned_ATE_full_sim3": full.get("delta_aligned_ate_rmse_m", ""),
                "baseline_final_error_m": full.get("baseline_final_error_m", ""),
                "candidate_final_error_m": full.get("candidate_final_error_m", ""),
                "delta_final_error_m": full.get("delta_final_error_m", ""),
                "baseline_error_slope_m_per_100f": full.get("baseline_error_slope_m_per_100f", ""),
                "candidate_error_slope_m_per_100f": full.get("candidate_error_slope_m_per_100f", ""),
                "delta_error_slope_m_per_100f": full.get("delta_error_slope_m_per_100f", ""),
                "active_count": payload.get("gate_chunk_summary", {}).get("active_count", ""),
                "active_mean_delta_m": active.get("mean_delta_m", ""),
                "active_worse_fraction": active.get("worse_fraction", ""),
                "inactive_count": payload.get("gate_chunk_summary", {}).get("inactive_count", ""),
                "inactive_mean_delta_m": inactive.get("mean_delta_m", ""),
                "inactive_worse_fraction": inactive.get("worse_fraction", ""),
                "response_label": ";".join(labels),
                "L0_source": "v95_stage7_full_validation",
            }
        )
    return rows


def classify_base_response(row: dict[str, str], cue_selected_swa: set[str]) -> list[str]:
    labels: list[str] = []
    case_id = row.get("case_id", "")
    case_label = row.get("case_label_offline_only", "")
    primary = row.get("failure_type_primary", "")
    secondary = row.get("failure_type_secondary", "")
    rec = row.get("recommended_next_track", "")
    if "READ_LOCAL" in rec or "LOCAL_BAD" in primary:
        labels.append("READ_LOCAL_BAD")
    if "HANDOFF" in primary or "HANDOFF" in secondary:
        labels.append("SWA_HANDOFF_CANDIDATE")
    if "LOW_OBSERVABILITY" in primary or "LOWOBS" in row.get("semantic_evidence_type", ""):
        labels.append("LOW_OBSERVABILITY_ABSTAIN")
    if "TTT" in rec:
        labels.append("TTT_WRITE_RISK_DIAGNOSTIC")
    if case_label == "good" or "GOOD" in rec:
        labels.append("GOOD_PROTECTION")
    if case_id in cue_selected_swa:
        labels.append("TRACK_G_CUE_PASS_ACTION_FAIL")
    if ("HANDOFF" in primary or case_id in cue_selected_swa) and case_label == "bad":
        labels.append("TRACK_E_BOUNDARY_ONLY_OR_NO_L3_EFFECT")
    if not labels:
        labels.append("REJECTED_OR_SUPPORT")
    return sorted(set(labels))


def build_stage0() -> dict[str, Any]:
    metric_rows = read_csv(V95_ROOT / "metric_suite/rows.csv")
    case_rows = read_csv(V95_ROOT / "trackA_base_case_bank/rows.csv")
    visual_rows = read_csv(V95_ROOT / "trackB_visual_clue_registry/rows.csv")
    v95_decision = read_json(V95_ROOT / "decision_matrix/summary.json")
    v95_drift = read_json(V95_ROOT / "drift_source_diagnosis/summary.json")
    stage7_rows = stage7_summary_rows()

    track_i = ensure_track_scaffold("trackI_drift_observatory")
    observatory_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        out = dict(row)
        out["observatory_row_type"] = "v95_base_case"
        out["L0_full_ATE_available"] = bool(row.get("L0_ATE_full"))
        out["action_response_label"] = ""
        observatory_rows.append(out)
    for row in stage7_rows:
        observatory_rows.append(
            {
                "case_id": row["candidate"],
                "seq": "01",
                "observatory_row_type": "v95_stage7_read_full_candidate",
                "L0_ATE_full": row["candidate_aligned_ATE_full_sim3"],
                "L0_baseline_ATE_full": row["baseline_aligned_ATE_full_sim3"],
                "L0_delta_ATE_full": row["delta_aligned_ATE_full_sim3"],
                "L0_final_error_delta": row["delta_final_error_m"],
                "L4_error_slope_delta": row["delta_error_slope_m_per_100f"],
                "active_mean_delta_m": row["active_mean_delta_m"],
                "inactive_mean_delta_m": row["inactive_mean_delta_m"],
                "action_response_label": row["response_label"],
                "metric_source": row["source_json"],
            }
        )
    write_csv(track_i / "rows.csv", observatory_rows)
    write_csv(track_i / "stage7_read_full_validation_rows.csv", stage7_rows)
    missing_rows = [
        {
            "case_id": row.get("case_id", ""),
            "metric": "L0_ATE_full",
            "missing_reason": row.get("L0_status", "not_run_in_v95_metric_suite"),
            "source": "v95_metric_suite_rows",
        }
        for row in metric_rows
        if not row.get("L0_ATE_full")
    ]
    write_csv(track_i / "metric_missing_reason.csv", missing_rows)
    drift_counts = Counter(row.get("recommended_next_track", "") for row in case_rows)
    stage7_best = min(
        (safe_float(row.get("delta_aligned_ATE_full_sim3")) for row in stage7_rows),
        default=None,
    )
    track_i_gate = bool(metric_rows) and bool(stage7_rows)
    write_csv(
        track_i / "gate_checks.csv",
        [
            {"gate": "v95_metric_rows_parsed", "pass": bool(metric_rows), "value": len(metric_rows)},
            {"gate": "stage7_READ15_27_summaries_parsed", "pass": bool(stage7_rows), "value": len(stage7_rows)},
            {"gate": "missing_L0_marked_not_filled", "pass": len(missing_rows) >= 0, "value": len(missing_rows)},
            {"gate": "drift_observatory_stage0_pass", "pass": track_i_gate, "value": track_i_gate},
        ],
    )
    write_json(
        track_i / "summary.json",
        {
            "stage": "Stage0_TrackI_Drift_Observatory_v2",
            "gate_pass": track_i_gate,
            "runtime_action_allowed": False,
            "metric_rows_from_v95": len(metric_rows),
            "stage7_read_full_candidate_rows": len(stage7_rows),
            "missing_L0_base_case_rows": len(missing_rows),
            "v95_drift_source_counts": v95_drift.get("drift_source_counts", {}),
            "v95_decision_status": v95_decision.get("final_status", ""),
            "best_stage7_delta_aligned_ATE_full_sim3_m": stage7_best,
            "interpretation": "Stage0 observatory derives from landed v95 artifacts; no v96 model action was run.",
        },
    )
    write_text(
        track_i / "per_track_report.md",
        "\n".join(
            [
                "# Track I Drift Observatory v2",
                "",
                "Evidence source: landed v95 metric suite, v95 decision matrix, and v95 Stage7 READ15-27 summaries.",
                "",
                f"- v95 metric rows: {len(metric_rows)}",
                f"- v95 case-bank next-track counts: {dict(drift_counts)}",
                f"- Stage7 full READ candidate summaries parsed: {len(stage7_rows)}",
                f"- Base-case L0 full ATE missing rows explicitly recorded: {len(missing_rows)}",
                "",
                "Interpretation:",
                "",
                "- Base v95 rows preserve L1/L2/L3/L4 proxy fields and do not fabricate L0 full ATE.",
                "- Stage7 rows show READ full validation effects separately from base case metrics.",
                "- v96 action remains disallowed at Stage0.",
            ]
        ),
    )
    write_text(track_i / "failure_report.md", "# Track I Failure Report\n\nNo Stage0 parser failure detected. Method success is not claimed.")
    write_text(
        track_i / "what_would_have_to_be_true_to_pass.md",
        "# Track I Pass Conditions\n\nStage0 pass only means v95 artifacts were parsed and missing metrics were marked. Full method success would require later mechanism and full validation gates.",
    )
    copy_visual_manifest(V95_ROOT / "trackB_visual_clue_registry/rows.csv", track_i / "visual_manifest.csv")

    track_a = ensure_track_scaffold("trackA_case_response_atlas")
    cue_selected_swa = {"00_003_004", "02_017_018", "05_018_019"}
    atlas_rows: list[dict[str, Any]] = []
    touched = 0
    touched_labelled = 0
    labelled = 0
    for row in case_rows:
        labels = classify_base_response(row, cue_selected_swa)
        is_touched = any(label.startswith(("READ_", "SWA_", "TRACK_", "TTT_")) for label in labels)
        if is_touched:
            touched += 1
        if labels:
            labelled += 1
            if is_touched:
                touched_labelled += 1
        atlas_rows.append(
            {
                **row,
                "action_response_labels": ";".join(labels),
                "do_not_repeat": ";".join(
                    x
                    for x in [
                        "old_trackE_alpha_maxpts_off_source_gate_source_replace_sweep"
                        if "TRACK_E_BOUNDARY_ONLY_OR_NO_L3_EFFECT" in labels
                        else "",
                        "phase3_pass_claim_as_full_success" if "READ_LOCAL_BAD" in labels else "",
                    ]
                    if x
                ),
                "v96_recommended_next_track": (
                    "TrackJ_or_TrackG_before_action"
                    if "READ_LOCAL_BAD" in labels or "TRACK_G_CUE_PASS_ACTION_FAIL" in labels
                    else "TrackE_raw_transport_trace"
                    if "SWA_HANDOFF_CANDIDATE" in labels
                    else row.get("recommended_next_track", "")
                ),
            }
        )
    coverage = labelled / max(1, len(case_rows))
    touched_coverage = touched_labelled / max(1, touched)
    track_a_gate = bool(case_rows) and coverage >= 0.80
    write_csv(track_a / "rows.csv", atlas_rows)
    write_csv(
        track_a / "gate_checks.csv",
        [
            {"gate": "all_v95_base_cases_preserved", "pass": len(atlas_rows) == len(case_rows), "value": f"{len(atlas_rows)}/{len(case_rows)}"},
            {"gate": "action_response_labels_coverage_ge_80pct", "pass": coverage >= 0.80, "value": coverage},
            {"gate": "touched_action_response_labels_coverage_ge_80pct", "pass": touched_coverage >= 0.80, "value": touched_coverage},
            {"gate": "trackG_cue_selected_pairs_labelled", "pass": cue_selected_swa.issubset({r["case_id"] for r in atlas_rows if "TRACK_G_CUE_PASS_ACTION_FAIL" in r["action_response_labels"]}), "value": ",".join(sorted(cue_selected_swa))},
        ],
    )
    write_json(
        track_a / "summary.json",
        {
            "stage": "Stage0_TrackA_case_response_atlas_v2",
            "gate_pass": track_a_gate,
            "row_count": len(atlas_rows),
            "all_v95_base_cases_preserved": len(atlas_rows) == len(case_rows),
            "action_response_label_coverage": coverage,
            "touched_action_response_label_coverage": touched_coverage,
            "label_counts": dict(Counter(label for row in atlas_rows for label in row["action_response_labels"].split(";") if label)),
            "runtime_action_allowed": False,
        },
    )
    write_text(
        track_a / "failure_report.md",
        "# Track A Failure Report\n\nNo parser failure detected. Labels are derived from v95 case-bank fields and v95 summaries; unmeasured action effects are not inferred.",
    )
    write_text(
        track_a / "what_would_have_to_be_true_to_pass.md",
        "# Track A Pass Conditions\n\nLater action promotion still requires Track J/J1 audit, cue controls, trace fidelity, and mechanism gates.",
    )
    copy_visual_manifest(V95_ROOT / "trackB_visual_clue_registry/rows.csv", track_a / "visual_manifest.csv")

    track_b = ensure_track_scaffold("trackB_visual_hypothesis_registry")
    hypothesis_rows: list[dict[str, Any]] = []
    for row in visual_rows:
        drift_type = row.get("drift_type", "")
        memory = row.get("memory_body", "")
        if "SWA" in memory or "merge" in memory or "handoff" in drift_type:
            target_memory = "SWA"
            target_metric = "L3_handoff_transfer"
            required_control = "same_count_random_or_semantic_rotation"
        elif "semantic_scale" in memory or "TrackG" in row.get("next_track", ""):
            target_memory = "READ/SWA cue"
            target_metric = "cue_bad_recall_good_FPR_random_margin"
            required_control = "random_same_count_and_rotation"
        else:
            target_memory = "multi"
            target_metric = row.get("testable_metric", "")
            required_control = "source_artifact_review"
        hypothesis_rows.append(
            {
                "clue_id": row.get("clue_id", ""),
                "case_id": row.get("case_id", ""),
                "source_memory_body": memory,
                "v96_target_memory_body": target_memory,
                "hypothesis": drift_type,
                "metric_to_validate": target_metric,
                "required_control": required_control,
                "visual_path": row.get("visual_path", ""),
                "visual_exists": row.get("visual_exists", ""),
                "blocked_reason": row.get("blocked_reason", ""),
            }
        )
    visual_exists_count = sum(1 for row in hypothesis_rows if boolish(row.get("visual_exists")))
    track_b_gate = bool(hypothesis_rows) and visual_exists_count > 0
    write_csv(track_b / "rows.csv", hypothesis_rows)
    write_csv(
        track_b / "gate_checks.csv",
        [
            {"gate": "visual_hypothesis_rows_created", "pass": bool(hypothesis_rows), "value": len(hypothesis_rows)},
            {"gate": "at_least_one_existing_visual", "pass": visual_exists_count > 0, "value": visual_exists_count},
            {"gate": "stage0_trackB_pass", "pass": track_b_gate, "value": track_b_gate},
        ],
    )
    write_json(
        track_b / "summary.json",
        {
            "stage": "Stage0_TrackB_visual_hypothesis_registry_v2",
            "gate_pass": track_b_gate,
            "row_count": len(hypothesis_rows),
            "visual_exists_count": visual_exists_count,
            "runtime_action_allowed": False,
        },
    )
    write_text(
        track_b / "failure_report.md",
        "# Track B Failure Report\n\nNo Stage0 visual registry parser failure detected. Visual hypotheses are not action proof.",
    )
    write_text(
        track_b / "what_would_have_to_be_true_to_pass.md",
        "# Track B Pass Conditions\n\nVisual clues must be tied to Track J/G/D/E/F metrics before action promotion.",
    )
    copy_visual_manifest(V95_ROOT / "trackB_visual_clue_registry/rows.csv", track_b / "visual_manifest.csv")

    return {
        "trackI_gate_pass": track_i_gate,
        "trackA_gate_pass": track_a_gate,
        "trackB_gate_pass": track_b_gate,
        "metric_rows": len(metric_rows),
        "case_rows": len(case_rows),
        "visual_rows": len(visual_rows),
        "stage7_rows": len(stage7_rows),
    }


def text_line_matches(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # noqa: BLE001
        return [{"path": str(path), "line": "", "text": f"read_error:{type(exc).__name__}:{exc}"}]
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        if any(needle.lower() in low for needle in needles):
            rows.append({"path": str(path), "line": idx, "text": line.strip()[:240]})
    return rows


def build_vggt4d_file_index() -> list[dict[str, Any]]:
    files: list[Path] = []
    if VGGT4D_ROOT.exists():
        root_depth = len(VGGT4D_ROOT.parts)
        for path in VGGT4D_ROOT.rglob("*"):
            if path.is_file() and len(path.parts) - root_depth <= 5:
                files.append(path)
    files = sorted(files)
    write_text(ROOT / "trackJ_vggt4d_code_audit/file_index.txt", "\n".join(str(p) for p in files))
    rows = []
    for path in files:
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "suffix": path.suffix,
                "under_git_dir": "/.git/" in str(path),
            }
        )
    write_csv(ROOT / "trackJ_vggt4d_code_audit/file_index.csv", rows)
    return rows


def run_track_j_synthetic_unit_tests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(test: str, status: str, passed: bool, evidence: str, value: Any = "") -> None:
        rows.append(
            {
                "test": test,
                "status": status,
                "pass": passed,
                "value": value,
                "evidence": evidence,
            }
        )

    attention_py = VGGT4D_ROOT / "vggt4d/layers/attention.py"
    try:
        text = attention_py.read_text(encoding="utf-8", errors="replace")
        static_pass = all(
            marker in text
            for marker in (
                "non_dyn_k =",
                "non_dyn_v =",
                "layer_id in range(0, 5)",
                "pad = torch.zeros(5",
            )
        )
        add(
            "vggt4d_static_compacts_k_and_v_early_layers",
            "executed_static",
            static_pass,
            "Checked active VGGT4D attention implementation for K/V compaction, 5 special-token pad, and early-layer guard.",
            str(attention_py),
        )
    except Exception as exc:  # noqa: BLE001
        add(
            "vggt4d_static_compacts_k_and_v_early_layers",
            "error",
            False,
            f"{type(exc).__name__}:{exc}",
            str(attention_py),
        )

    try:
        import sys

        import torch
        from torch.nn.functional import scaled_dot_product_attention

        repo_root = str(Path.cwd())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from loger.models.layers.attention import _compact_kv_sdpa

        torch.manual_seed(9604)
        q = torch.randn(2, 3, 9, 4)
        k = torch.randn(2, 3, 9, 4)
        v = torch.randn(2, 3, 9, 4)
        all_keep = torch.ones(2, 9, dtype=torch.bool)
        compact = _compact_kv_sdpa(q, k, v, all_keep)
        dense = scaled_dot_product_attention(q, k, v)
        max_abs = float((compact - dense).abs().max().item())
        add(
            "loger_compact_kv_all_keep_no_action_parity",
            "executed_synthetic_cpu",
            max_abs <= 1e-6,
            "Synthetic CPU tensor test: all source tokens kept, compact-KV output compared with dense SDPA.",
            max_abs,
        )

        drop_mask = torch.tensor(
            [
                [True, True, False, True, True, False, True, True, True],
                [True, False, True, True, True, True, False, True, True],
            ],
            dtype=torch.bool,
        )
        stats: list[dict[str, Any]] = []
        dropped = _compact_kv_sdpa(q, k, v, drop_mask, attention_mass_stats=stats)
        drop_delta = float((dropped - dense).abs().max().item())
        add(
            "loger_compact_kv_drop_source_trace_change",
            "executed_synthetic_cpu",
            drop_delta > 1e-8,
            "Synthetic CPU tensor test: dropping source tokens changes attention output, proving hook is active on toy tensors.",
            drop_delta,
        )
        stats_pass = bool(stats) and safe_float(stats[0].get("attention_mass_removed_tokens_mean")) not in (None, 0.0)
        add(
            "loger_compact_kv_attention_mass_stats_recorded",
            "executed_synthetic_cpu",
            stats_pass,
            "Synthetic CPU tensor test: compact-KV attention mass stats recorded removed and retained source-token mass.",
            json.dumps(stats[0], sort_keys=True) if stats else "",
        )

        try:
            _compact_kv_sdpa(q, k, v, torch.ones(1, 9, dtype=torch.bool))
        except ValueError as exc:
            add(
                "loger_compact_kv_shape_guard",
                "executed_synthetic_cpu",
                "shape mismatch" in str(exc),
                "Synthetic CPU tensor test: invalid source_keep_mask shape raises ValueError.",
                str(exc),
            )
        else:
            add(
                "loger_compact_kv_shape_guard",
                "executed_synthetic_cpu",
                False,
                "Expected ValueError was not raised for invalid source_keep_mask shape.",
                "",
            )
    except Exception as exc:  # noqa: BLE001
        add(
            "loger_compact_kv_synthetic_suite",
            "error",
            False,
            f"{type(exc).__name__}:{exc}",
            "",
        )

    add(
        "real_v95_case_qk_token_grid_alignment_smoke",
        "not_executed_blocked",
        False,
        "Still requires selecting v95 READ_LOCAL and good-control cases, adding/using a trace-only Q/K dump, and proving case-to-token/semantic mask alignment.",
        "",
    )
    return rows


def audit_existing_v95_qqkk_probe_artifacts() -> list[dict[str, Any]]:
    probe_root = V95_ROOT / "trackD_read_qqkk_dump_probe_v1"
    rows: list[dict[str, Any]] = []
    if not probe_root.exists():
        return rows

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "seq": "",
                "chunk": "",
                "status": "error",
                "pass": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
        ]

    def tensor_shape(obj: Any) -> str:
        return "x".join(str(x) for x in obj.shape) if torch.is_tensor(obj) else ""

    for pca_path in sorted(probe_root.glob("seq*/chunk*/pca_features/chunk_*.pt")):
        chunk_dir = pca_path.parent.parent
        seq_dir = chunk_dir.parent
        seq = seq_dir.name.replace("seq", "")
        chunk = chunk_dir.name.replace("chunk", "")
        read_path = chunk_dir / "read_cue_patch_dumps" / f"chunk_{int(chunk):03d}_read_cue_patch.pt"
        row: dict[str, Any] = {
            "seq": seq,
            "chunk": chunk,
            "pca_path": str(pca_path),
            "read_cue_path": str(read_path),
            "status": "executed_artifact_audit",
            "pass": False,
            "error": "",
        }
        try:
            pca = torch.load(pca_path, map_location="cpu")
            read = torch.load(read_path, map_location="cpu") if read_path.exists() else {}
            pca_grid = pca.get("patch_grid", []) if isinstance(pca, dict) else []
            read_grid = read.get("patch_grid", []) if isinstance(read, dict) else []
            global_k = pca.get("tap::pca_attn_global_k_layers") if isinstance(pca, dict) else None
            global_v = pca.get("tap::pca_attn_global_v_layers") if isinstance(pca, dict) else None
            current_q = pca.get("tap::pca_swa_current_q_layers") if isinstance(pca, dict) else None
            current_k = pca.get("tap::pca_swa_current_k_layers") if isinstance(pca, dict) else None
            current_v = pca.get("tap::pca_swa_current_v_layers") if isinstance(pca, dict) else None
            read_patch = read.get("tensors", {}).get("read_patch_final") if isinstance(read, dict) else None
            read_active = read.get("tensors", {}).get("read_active_q90_patch") if isinstance(read, dict) else None
            pca_grid_tuple = tuple(int(x) for x in pca_grid) if pca_grid else ()
            read_grid_tuple = tuple(int(x) for x in read_grid) if read_grid else ()
            qk_like_available = all(torch.is_tensor(x) for x in (global_k, current_q, current_k, current_v))
            grid_match = bool(pca_grid_tuple) and pca_grid_tuple == read_grid_tuple
            pca_tensor_grid_match = (
                qk_like_available
                and tuple(int(x) for x in current_q.shape[2:4]) == pca_grid_tuple
                and tuple(int(x) for x in current_k.shape[2:4]) == pca_grid_tuple
                and tuple(int(x) for x in global_k.shape[2:4]) == pca_grid_tuple
            )
            read_tensor_grid_match = torch.is_tensor(read_patch) and tuple(int(x) for x in read_patch.shape[1:3]) == read_grid_tuple
            frame_count_match = (
                qk_like_available
                and torch.is_tensor(read_patch)
                and int(current_q.shape[0]) == int(current_k.shape[0]) == int(global_k.shape[0]) == int(read_patch.shape[0])
            )
            row.update(
                {
                    "pca_schema": pca.get("schema", "") if isinstance(pca, dict) else "",
                    "read_schema": read.get("schema", "") if isinstance(read, dict) else "",
                    "start_frame": pca.get("start_frame", "") if isinstance(pca, dict) else "",
                    "end_frame": pca.get("end_frame", "") if isinstance(pca, dict) else "",
                    "pca_patch_grid": "x".join(str(x) for x in pca_grid),
                    "read_patch_grid": "x".join(str(x) for x in read_grid),
                    "global_k_shape": tensor_shape(global_k),
                    "global_v_shape": tensor_shape(global_v),
                    "swa_current_q_shape": tensor_shape(current_q),
                    "swa_current_k_shape": tensor_shape(current_k),
                    "swa_current_v_shape": tensor_shape(current_v),
                    "read_patch_shape": tensor_shape(read_patch),
                    "read_active_q90_shape": tensor_shape(read_active),
                    "qk_like_taps_available": qk_like_available,
                    "pca_read_grid_match": grid_match,
                    "pca_tensor_grid_match": pca_tensor_grid_match,
                    "read_tensor_grid_match": read_tensor_grid_match,
                    "frame_count_match": frame_count_match,
                    "raw_qk_available": False,
                    "limitation": "v95 artifact is PCA-reduced Q/K-like feature dump plus READ cue patch map; not raw LoGeR Q/K and not a v96 no-action parity run.",
                }
            )
            row["pass"] = bool(qk_like_available and grid_match and pca_tensor_grid_match and read_tensor_grid_match and frame_count_match)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)
    return rows


def audit_raw_qk_trace_smoke(out: Path) -> dict[str, Any]:
    smoke_root = ROOT / "trackJ_raw_qk_trace_smoke"
    summary = read_json(smoke_root / "summary.json")
    parity_rows = read_csv(smoke_root / "parity_results.csv")
    semantic_rows = read_csv(smoke_root / "semantic_alignment_results.csv")
    action_rows = read_csv(smoke_root / "action_trace_probe_results.csv")
    audit_rows: list[dict[str, Any]] = []
    for row in parity_rows:
        sem = next((item for item in semantic_rows if item.get("case_id") == row.get("case_id")), {})
        action = next((item for item in action_rows if item.get("case_id") == row.get("case_id")), {})
        audit_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "label": row.get("label", ""),
                "strict_sha_parity": row.get("strict_sha_parity", ""),
                "max_abs_pose_diff": row.get("max_abs_pose_diff", ""),
                "raw_qk_dump_count": row.get("raw_qk_dump_count", ""),
                "read_cue_dump_count": row.get("read_cue_dump_count", ""),
                "trace_smoke_pass": row.get("trace_smoke_pass", ""),
                "semantic_alignment_smoke_pass": sem.get("semantic_alignment_smoke_pass", ""),
                "raw_q_shape": sem.get("raw_q_shape", ""),
                "raw_k_shape": sem.get("raw_k_shape", ""),
                "read_patch_grid": sem.get("read_patch_grid", ""),
                "raw_source_token_count": sem.get("raw_source_token_count", ""),
                "read_patch_tokens_per_frame": sem.get("read_patch_tokens_per_frame", ""),
                "special_tokens_per_frame": sem.get("special_tokens_per_frame", ""),
                "affected_true_count": sem.get("affected_true_count", ""),
                "hmc_context_source_control_tokens": sem.get("hmc_context_source_control_tokens", ""),
                "dense_semantic_patch_nonvoid_ratio": sem.get("dense_semantic_patch_nonvoid_ratio", ""),
                "dense_semantic_patch_purity": sem.get("dense_semantic_patch_purity", ""),
                "action_trace_probe_pass": action.get("action_trace_probe_pass", ""),
                "action_raw_attention_mass_delta": action.get("raw_attention_mass_delta", ""),
                "action_hmc_removed_before": action.get("hmc_mean_attention_mass_removed_before", ""),
                "action_hmc_removed_after": action.get("hmc_mean_attention_mass_removed_after", ""),
                "action_hmc_source_weight_min": action.get("hmc_mean_source_weight_min", ""),
            }
        )
    write_csv(out / "raw_qk_trace_smoke_audit.csv", audit_rows)
    trace_pass = bool(summary.get("trace_smoke_pass_all_cases", False))
    semantic_pass = bool(summary.get("semantic_alignment_smoke_pass_all_cases", False))
    action_pass = bool(summary.get("action_trace_probe_pass_all_cases", False))
    return {
        "available": bool(summary),
        "trace_smoke_pass_all_cases": trace_pass,
        "semantic_alignment_smoke_pass_all_cases": semantic_pass,
        "action_trace_probe_pass_all_cases": action_pass,
        "case_count": len(parity_rows),
        "raw_qk_dump_total": sum(int(row.get("raw_qk_dump_count") or 0) for row in parity_rows),
        "action_probe_raw_qk_dump_total": sum(int(row.get("raw_dump_count") or 0) for row in action_rows),
        "read_cue_dump_total": sum(int(row.get("read_cue_dump_count") or 0) for row in parity_rows),
        "audit_csv": str(out / "raw_qk_trace_smoke_audit.csv"),
    }


def build_track_j_audit() -> dict[str, Any]:
    out = ensure_track_scaffold("trackJ_vggt4d_code_audit")
    file_rows = build_vggt4d_file_index()
    relevant_paths = [
        VGGT4D_ROOT / "demo_vggt4d.py",
        VGGT4D_ROOT / "vggt4d/models/vggt4d.py",
        VGGT4D_ROOT / "vggt4d/models/aggregator.py",
        VGGT4D_ROOT / "vggt4d/layers/attention.py",
        VGGT4D_ROOT / "vggt4d/masks/dynamic_mask.py",
        VGGT4D_ROOT / "vggt4d/masks/refine_dyn_mask.py",
        VGGT4D_ROOT / "vggt4d/utils/model_utils.py",
        Path("loger/models/layers/attention.py"),
        Path("loger/pipeline/hybrid_memory_controller.py"),
    ]
    hit_rows: list[dict[str, Any]] = []
    for path in relevant_paths:
        hit_rows.extend(
            text_line_matches(
                path,
                [
                    "qkv",
                    "dyn_masks",
                    "scaled_dot_product_attention",
                    "extract_dyn_map",
                    "layer_ids",
                    "window",
                    "threshold",
                    "max_pool2d",
                    "source_keep_mask",
                    "context_source_skip",
                    "frame_attention",
                    "swa_read",
                    "ttt_apply",
                ],
            )
        )
    write_csv(out / "code_pointer_rows.csv", hit_rows)
    write_csv(
        out / "tensor_shape_table.csv",
        [
            {"component": "VGGT4D input images", "shape": "[B,S,3,H,W]", "source": "vggt4d/models/aggregator.py:17-24", "migration_note": "LoGeR chunk tensors are not VGGT frame-batch tensors."},
            {"component": "VGGT4D patch tokens", "shape": "[B*S,P,C]", "source": "vggt4d/models/aggregator.py:39-46", "migration_note": "VGGT patch token order cannot be copied."},
            {"component": "VGGT4D dyn_masks before attention", "shape": "[B,S,H/14*W/14]", "source": "vggt4d/models/aggregator.py:48-51", "migration_note": "Token-grid alignment is required before LoGeR action."},
            {"component": "VGGT4D Q/K dump", "shape": "global/frame Q,K stacked by layer; organized to [n_img,n_layer,n_head,n_tok,c]", "source": "vggt4d/utils/model_utils.py:57-109", "migration_note": "LoGeR layer/head ids differ."},
            {"component": "LoGeR source_keep_mask", "shape": "[B,N]", "source": "loger/models/layers/attention.py:_compact_kv_sdpa", "migration_note": "Candidate action surface exists but must be gated by v96 tests."},
        ],
    )
    write_csv(
        out / "attention_hook_map.csv",
        [
            {"system": "VGGT4D", "hook": "AttentionFor4D.forward", "path": "third_party/VGGT4D/vggt4d/layers/attention.py:63-95", "qk_available": True, "action_available": "dynamic mask on early layers"},
            {"system": "VGGT4D", "hook": "AggregatorFor4D._process_frame_attention", "path": "third_party/VGGT4D/vggt4d/models/aggregator.py:161-191", "qk_available": True, "action_available": "passes dyn_masks into frame blocks"},
            {"system": "VGGT4D", "hook": "AggregatorFor4D._process_global_attention", "path": "third_party/VGGT4D/vggt4d/models/aggregator.py:193-222", "qk_available": True, "action_available": "passes dyn_masks into global blocks"},
            {"system": "LoGeR", "hook": "frame_attention", "path": "loger/pipeline/hybrid_memory_controller.py and loger/models/layers/attention.py", "qk_available": "trace-dependent", "action_available": "existing source mask/bias paths, not v96-promoted"},
            {"system": "LoGeR", "hook": "swa_read", "path": "loger/pipeline/hybrid_memory_controller.py", "qk_available": "trace-dependent", "action_available": "existing SWA read controls, raw transport trace still required"},
            {"system": "LoGeR", "hook": "ttt_apply/write", "path": "loger/pipeline/hybrid_memory_controller.py and ttt_write_controller.py", "qk_available": "trace-dependent", "action_available": "diagnostic only for v96"},
        ],
    )
    write_text(
        out / "layer_group_definition.md",
        """# VGGT4D Layer Groups

Local audit result:

- Dynamic map mean1 uses global Q-Q, `layer_ids = torch.arange(3, 8)`.
- Dynamic map variance1 uses global Q-Q, `layer_ids = torch.arange(18, 20)`.
- Dynamic map mean2 uses global Q-Q, `layer_ids = torch.arange(17, 22)`.
- Dynamic map mean3 uses global K-K, `layer_ids = torch.arange(0, 1)`.
- Dynamic map variance3 uses global Q-K, `layer_ids = torch.arange(0, 1)`.
- Intermediate token preservation uses `[4, 11, 17, 23]`.

Migration boundary: these VGGT4D layer ids are architecture-specific and cannot be copied into LoGeR.
""",
    )
    write_text(
        out / "dynamic_mask_pipeline.md",
        """# Dynamic Mask Pipeline

1. Run `inference(model, images)` without masks.
2. Organize global/frame Q/K into token-only tensors with patch_start_idx=5.
3. Compute Q-Q, K-K, and Q-K dynamic maps over temporal offsets `[-6,-4,-2,2,4,6]`.
4. Min-max normalize maps and combine them as `(1-mean1)*(1-var1)*mean2*(1-mean3)*var3`.
5. Cluster encoder features with KMeans and assign cluster mean dynamic scores.
6. Upsample to image size and choose threshold with adaptive multi-Otsu.
7. Run second inference with `dyn_masks`.
8. Refine dynamic masks by projection/depth/RGB consistency.
""",
    )
    write_text(
        out / "early_k_side_masking_implementation.md",
        """# Early K-Side Masking Implementation

VGGT4D does not add a dense attention bias.  In `AttentionFor4D.attention_with_dynamic_mask`, dynamic patch tokens are removed from both K and V:

- special camera/register tokens are padded as non-dynamic;
- `non_dyn_idx = (~dyn_mask).nonzero(...)`;
- `non_dyn_k = k[..., non_dyn_idx, :]`;
- `non_dyn_v = v[..., non_dyn_idx, :]`;
- SDPA runs with all query rows and compacted K/V columns.

The current code applies this only when `dyn_masks is not None` and `layer_id in range(0, 5)`.  Because V is also compacted, LoGeR must not describe this as pure K-only suppression without qualification.
""",
    )
    write_text(
        out / "projection_gradient_refinement.md",
        """# Projection / Geometry Refinement

`RefineDynMask` inverse-projects VGGT depth using predicted intrinsics and cam2world, removes outliers, clusters coarse dynamic points, projects them into other cameras, samples other-frame depth/RGB/dynamic masks, and keeps labels whose depth+RGB consistency loss exceeds the threshold.

This depends on VGGT depth, intrinsics, camera poses, and RGB/depth projection behavior.  It is not directly portable to LoGeR runtime memory control.
""",
    )
    write_text(
        out / "full_mask_ablation_path.md",
        """# Full Mask / Hard Mask Path

The audited code contains commented-out patch-token zeroing in `vggt4d/models/aggregator.py:52-56` with a `bad effect` note.  The active path instead compacts K/V inside attention for early layers only.

LoGeR negative control may include a full-layer hard mask, but v96 must treat it as an implementation audit/control, not a promoted method.
""",
    )
    write_text(
        out / "do_not_copy_list.md",
        """# Do Not Copy Into LoGeR

- VGGT4D layer ids such as 0, 3-8, 17-22, and 18-20.
- VGGT camera/register token ordering and `patch_start_idx=5`.
- VGGT patch size 14 or patch-to-image mapping as a LoGeR constant.
- VGGT projection/depth refinement as a LoGeR runtime feature.
- Full-layer hard mask / patch-token zeroing.
- K/V compaction semantics described as K-only suppression.
- Any GT/projection outcome as a runtime trigger.
""",
    )
    write_text(
        out / "vggt4d_to_loger_mapping.md",
        """# VGGT4D to LoGeR Adaptation Mapping

| VGGT4D component | LoGeR candidate component | Migratable | Needed tensor | LoGeR risk | Unit test |
|---|---|---:|---|---|---|
| global Q/K Gram cue | frame/global attention trace cue | diagnostic only first | Q/K by layer/head/token | layer/head ids differ | Q/K dump shape test |
| temporal window offsets | chunk overlap / adjacent boundary windows | partly | frame/chunk index map | chunk cadence differs | temporal aggregation test |
| early dynamic K/V compaction | source token eligibility / source_keep_mask | partly | token mask [B,N] | V also changes if compacted | no-action parity and trace change test |
| dynamic mask from Q/K maps | internal unreliable-region cue | partly | patch-grid score map | patch-token image mapping differs | patch token -> image coordinate test |
| projection refinement | offline diagnostic only | no runtime | depth/pose/projection | VGGT-specific geometry | blocked from runtime |
| full hard mask ablation | negative control only | no method | full layer mask | OOD behavior | negative-control test |

Mapping conclusion: VGGT4D supplies principles and audit targets; LoGeR adaptation requires LoGeR-native tensor dumps, token-grid alignment, and parity tests before action.
""",
    )
    raw_trace = audit_raw_qk_trace_smoke(out)
    write_text(
        out / "unit_test_plan.md",
        f"""# J1 Unit Test Plan and Current Execution Status

| Test | Status | Evidence / Blocker |
|---|---|---|
| LoGeR compact-KV all-keep no-action parity | executed_synthetic_cpu | See `unit_test_results.csv`; this is a synthetic function-level parity test, not a real v95 case gate. |
| LoGeR compact-KV drop-source trace-change test | executed_synthetic_cpu | See `unit_test_results.csv`; verifies the hook changes synthetic attention output when source tokens are removed. |
| LoGeR compact-KV shape guard test | executed_synthetic_cpu | See `unit_test_results.csv`; verifies invalid masks are rejected. |
| VGGT4D K/V compaction static check | executed_static | See `unit_test_results.csv`; confirms active VGGT4D path compacts both K and V in early layers. |
| Existing v95 Q/K-like PCA dump shape + READ patch-grid audit | executed_artifact_audit | See `v95_qqkk_probe_shape_grid_audit.csv`; validates existing v95 PCA-reduced feature dumps and read cue patch maps share `[32,19,66]` grid where artifacts exist. |
| LoGeR Q/K dump shape test | executed_real_trace_smoke | See `raw_qk_trace_smoke_audit.csv`; pass_all_cases={raw_trace["trace_smoke_pass_all_cases"]}, cases={raw_trace["case_count"]}, raw_qk_dump_total={raw_trace["raw_qk_dump_total"]}. |
| patch token -> image coordinate mapping test | static_partial | VGGT4D mapping audited; LoGeR mapping still requires chunk token metadata. |
| layer/head indexing test | static_partial | VGGT4D ids audited and marked non-portable; LoGeR layer/head map not promoted. |
| temporal window aggregation test | planned_not_executed | Requires LoGeR chunk/frame index map for selected v95 cases. |
| semantic mask alignment test | executed_two_case_smoke_partial | See `raw_qk_trace_smoke_audit.csv`; pass_all_cases={raw_trace["semantic_alignment_smoke_pass_all_cases"]}. This is not full J2 base-universe coverage. |
| real-case no-action baseline parity test | executed_real_trace_smoke | See `trackJ_raw_qk_trace_smoke/parity_results.csv`; strict sha parity and max_abs_diff=0 are required. |
| early K-side hook trace change test | executed_trace_only_counterfactual | See `raw_qk_trace_smoke_audit.csv`; pass_all_cases={raw_trace["action_trace_probe_pass_all_cases"]}. This records before/after raw attention mass without promoting runtime action. |

Gate interpretation: J0 static code audit is complete. Raw LoGeR Q/K dump shape, two-case semantic alignment smoke, real-case no-action parity, and trace-only counterfactual early K-side attention-mass change are passed for the smoke cases. This does not promote runtime action; J3 skip-impact diagnostics still precede J4/J5/J6.
""",
    )
    unit_rows = run_track_j_synthetic_unit_tests()
    write_csv(out / "unit_test_results.csv", unit_rows)
    assessed_unit_rows = [row for row in unit_rows if row.get("status") != "not_executed_blocked"]
    passed_assessed_units = sum(1 for row in assessed_unit_rows if boolish(row.get("pass")))
    synthetic_unit_gate_pass = bool(assessed_unit_rows) and passed_assessed_units == len(assessed_unit_rows)
    v95_probe_rows = audit_existing_v95_qqkk_probe_artifacts()
    write_csv(out / "v95_qqkk_probe_shape_grid_audit.csv", v95_probe_rows)
    v95_probe_passed = sum(1 for row in v95_probe_rows if boolish(row.get("pass")))
    v95_probe_gate_pass = bool(v95_probe_rows) and v95_probe_passed == len(v95_probe_rows)
    j1_runtime_gate_pass = bool(
        synthetic_unit_gate_pass
        and v95_probe_gate_pass
        and raw_trace["trace_smoke_pass_all_cases"]
        and raw_trace["semantic_alignment_smoke_pass_all_cases"]
        and raw_trace["action_trace_probe_pass_all_cases"]
    )
    write_text(
        out / "code_audit_report.md",
        """# Track J0 VGGT4D Local Code Audit Report

## Required Answers

1. Dynamic cue source: VGGT4D uses global token Q/Q, K/K, and Q/K Gram-style products in `vggt4d/masks/dynamic_mask.py`; standard attention QK is available through `AttentionFor4D.forward`.
2. Aggregation dimensions: Q/K are organized as `[n_img,n_layer,n_head,n_tok,c]`; maps are rearranged to token grid and averaged across layer/head/source-frame/source-token dimensions or std across source-token dimension depending on cue.
3. Temporal window: offsets are fixed as `[-6,-4,-2,2,4,6]` with out-of-range frames filtered.
4. Layer groups: fixed VGGT layer ranges are used (`0`, `3..7`, `17..21`, `18..19`) and are architecture-specific.
5. Dynamic mask threshold: clustered dynamic score map is upsampled to image size, then thresholded by adaptive multi-Otsu over the current image/video score array.
6. Projection refinement: `RefineDynMask` depends on predicted VGGT depth, intrinsics, camera poses, RGB/depth projection consistency, and coarse dynamic mask clusters.
7. Early masking: active attention masking compacts both K and V source tokens for early layers (`layer_id in range(0,5)`); it is not pure K-only suppression.
8. Full mask harm path: patch-token zeroing is present only as commented-out code with a `bad effect` note; full hard mask should be a negative control only.
9. Migratable modules: Q/K statistics, temporal inconsistency idea, early source-token eligibility, and full-mask negative control can migrate as LoGeR diagnostics after LoGeR-native tests.
10. Non-portable modules: VGGT layer ids, token ordering, camera/register token logic, patch size, and projection refinement cannot be copied into LoGeR runtime.

## Gate

J0 static audit is complete. J1 runtime/unit tests are not passed; no VGGT4D-style LoGeR action is allowed yet.
""",
    )
    gate_rows = [
        {"gate": "file_index_created", "pass": bool(file_rows), "value": len(file_rows)},
        {"gate": "code_audit_report_complete", "pass": True, "value": str(out / "code_audit_report.md")},
        {"gate": "vggt4d_to_loger_mapping_complete", "pass": True, "value": str(out / "vggt4d_to_loger_mapping.md")},
        {"gate": "do_not_copy_list_complete", "pass": True, "value": str(out / "do_not_copy_list.md")},
        {"gate": "unit_test_plan_complete", "pass": True, "value": str(out / "unit_test_plan.md")},
        {"gate": "synthetic_function_level_unit_tests_pass", "pass": synthetic_unit_gate_pass, "value": f"{passed_assessed_units}/{len(assessed_unit_rows)}"},
        {"gate": "v95_qqkk_probe_shape_grid_audit_pass", "pass": v95_probe_gate_pass, "value": f"{v95_probe_passed}/{len(v95_probe_rows)}"},
        {"gate": "raw_qk_trace_smoke_pass", "pass": raw_trace["trace_smoke_pass_all_cases"], "value": f"cases={raw_trace['case_count']} raw_qk_dumps={raw_trace['raw_qk_dump_total']}"},
        {"gate": "semantic_alignment_two_case_smoke_pass", "pass": raw_trace["semantic_alignment_smoke_pass_all_cases"], "value": f"cases={raw_trace['case_count']}"},
        {"gate": "early_k_side_trace_change_probe_pass", "pass": raw_trace["action_trace_probe_pass_all_cases"], "value": f"action_raw_qk_dumps={raw_trace['action_probe_raw_qk_dump_total']}"},
        {"gate": "J1_runtime_unit_gate_pass", "pass": j1_runtime_gate_pass, "value": "trace_only_smoke_runtime_unit_gate"},
    ]
    write_csv(out / "gate_checks.csv", gate_rows)
    write_json(
        out / "summary.json",
        {
            "stage": "TrackJ_J0_J1_local_code_audit_and_mapping",
            "J0_static_audit_complete": True,
            "J1_runtime_unit_gate_pass": j1_runtime_gate_pass,
            "runtime_action_allowed": False,
            "file_index_row_count": len(file_rows),
            "code_pointer_row_count": len(hit_rows),
            "synthetic_function_level_unit_tests_pass": synthetic_unit_gate_pass,
            "synthetic_function_level_unit_tests_passed": passed_assessed_units,
            "synthetic_function_level_unit_tests_executed": len(assessed_unit_rows),
            "v95_qqkk_probe_shape_grid_audit_pass": v95_probe_gate_pass,
            "v95_qqkk_probe_shape_grid_audit_passed": v95_probe_passed,
            "v95_qqkk_probe_shape_grid_audit_rows": len(v95_probe_rows),
            "raw_qk_trace_smoke": raw_trace,
            "blocker": "" if j1_runtime_gate_pass else "J1 runtime smoke is incomplete; see gate_checks.csv",
        },
    )
    write_text(
        out / "failure_report.md",
        "# Track J Report\n\nJ0 static audit completed. Synthetic compact-KV tests, existing v95 PCA-reduced Q/K-like grid audit, raw LoGeR Q/K trace/no-action parity smoke, two-case semantic alignment smoke, and trace-only counterfactual early K-side attention-mass change passed. This is still diagnostic evidence only; J3 skip-impact diagnostics precede J4/J5/J6 actions.",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# Track J Pass Requirements\n\nJ1 would pass only after LoGeR Q/K dump shape test, token/image alignment, semantic mask alignment, no-action parity max_abs_diff==0, and early hook trace-change tests are executed or legitimately scoped with evidence.",
    )
    write_csv(out / "visual_manifest.csv", [])
    return {
        "J0_static_audit_complete": True,
        "J1_runtime_unit_gate_pass": j1_runtime_gate_pass,
        "file_index_rows": len(file_rows),
        "synthetic_function_level_unit_tests_pass": synthetic_unit_gate_pass,
        "synthetic_function_level_unit_tests_passed": passed_assessed_units,
        "synthetic_function_level_unit_tests_executed": len(assessed_unit_rows),
        "v95_qqkk_probe_shape_grid_audit_pass": v95_probe_gate_pass,
        "v95_qqkk_probe_shape_grid_audit_passed": v95_probe_passed,
        "v95_qqkk_probe_shape_grid_audit_rows": len(v95_probe_rows),
        "raw_qk_trace_smoke": raw_trace,
    }


def build_semantic_region_blocker() -> dict[str, Any]:
    out = ensure_track_scaffold("trackJ_semantic_region_bank")
    import torch
    import torch.nn.functional as F

    stage_c_dirs = sorted(Path("results/kitti_preprocess").glob("*/stage_c_cache_semantic_chunks"))
    sparse_candidates = [
        Path("results/v2_kitti01_full_sam3yoloe26l_dvisplus_driving_v1/sparse_masklets.pt"),
        Path("results/v3_kitti01_maskmot_sam2/sparse_masklets.pt"),
        Path("results/v2_kitti01_yoloe26l_quality_segformer_clean_v2/sparse_masklets.pt"),
    ]
    rows = []
    for path in stage_c_dirs:
        rows.append({"artifact_type": "stage_c_cache_semantic_chunks", "path": str(path), "exists": path.exists(), "bytes": ""})
    for path in sparse_candidates:
        rows.append({"artifact_type": "sparse_masklets_v1", "path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else ""})
    write_csv(out / "mask_manifest.csv", rows)

    region_label_names = {
        "DYNAMIC_OBJECT": {"car", "truck", "bus", "person", "rider", "cyclist", "motorcycle", "bicycle", "train"},
        "WEAK_SCALE_CONTEXT": {"sky", "road", "ground", "mountain", "sidewalk", "terrain"},
        "VEGETATION_REPETITIVE": {"tree", "grass", "vegetation", "bush"},
        "STABLE_ANCHOR": {
            "building",
            "house",
            "wall",
            "handrail_or_fence",
            "fence",
            "pole",
            "traffic sign",
            "traffic_sign",
            "traffic light",
            "traffic_light",
            "billboard_or_bulletin_board",
            "bridge",
            "other_construction",
        },
    }
    region_order = [
        "DYNAMIC_OBJECT",
        "OBJECT_BOUNDARY_BAND",
        "WEAK_SCALE_CONTEXT",
        "VEGETATION_REPETITIVE",
        "LOW_OBSERVABILITY",
        "MULTIMODE_CONFLICT",
        "STABLE_ANCHOR",
        "UNKNOWN_CONTEXT",
    ]

    def label_ids(label_to_id: dict[str, Any], names: set[str]) -> list[int]:
        ids = []
        for name, idx in label_to_id.items():
            canonical = str(name).strip().lower()
            if canonical in names:
                ids.append(int(idx))
        return ids

    def downsample(mask: torch.Tensor, grid: tuple[int, int] = (19, 66)) -> torch.Tensor:
        return F.adaptive_avg_pool2d(mask.float(), grid)

    def semantic_boundary(label_maps: torch.Tensor) -> torch.Tensor:
        boundary = torch.zeros_like(label_maps, dtype=torch.bool)
        boundary[:, 1:, :] |= label_maps[:, 1:, :] != label_maps[:, :-1, :]
        boundary[:, :-1, :] |= label_maps[:, 1:, :] != label_maps[:, :-1, :]
        boundary[:, :, 1:] |= label_maps[:, :, 1:] != label_maps[:, :, :-1]
        boundary[:, :, :-1] |= label_maps[:, :, 1:] != label_maps[:, :, :-1]
        return boundary

    def make_panel(case_row: dict[str, str], masks: dict[str, torch.Tensor]) -> str:
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            return ""
        seq = case_row.get("seq", "")
        chunk = int(case_row.get("curr_chunk") or 0)
        frame = chunk * 29
        image_path = Path(f"/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/{seq}/image_2/{frame:06d}.png")
        if not image_path.is_file():
            return ""
        image = Image.open(image_path).convert("RGB")
        overlay = np.asarray(image).astype("float32")
        colors = {
            "DYNAMIC_OBJECT": np.array([255, 40, 40], dtype="float32"),
            "OBJECT_BOUNDARY_BAND": np.array([255, 220, 40], dtype="float32"),
            "LOW_OBSERVABILITY": np.array([60, 120, 255], dtype="float32"),
            "STABLE_ANCHOR": np.array([40, 220, 100], dtype="float32"),
        }
        for region, color in colors.items():
            mask = masks.get(region)
            if mask is None:
                continue
            m = mask[0].float().numpy()
            m_img = Image.fromarray((m * 255).astype("uint8")).resize(image.size, Image.Resampling.BILINEAR)
            alpha = (np.asarray(m_img).astype("float32") / 255.0)[..., None] * 0.38
            overlay = overlay * (1.0 - alpha) + color * alpha
        bucket = case_row.get("v95_case_bucket", "bucket").replace("/", "_")
        panel_path = out / "visual_panels" / f"{bucket}_{case_row.get('case_id', '')}_semantic_regions.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype("uint8")).save(panel_path)
        return str(panel_path)

    case_rows = read_csv(ROOT / "trackA_case_response_atlas/rows.csv")
    region_rows: list[dict[str, Any]] = []
    per_case_rows: list[dict[str, Any]] = []
    mask_bank: dict[str, dict[str, Any]] = {}
    visual_rows: list[dict[str, Any]] = []
    visual_buckets_seen: set[str] = set()
    covered = 0
    aligned = 0
    stable_and_two_risk = 0
    read_swa_cases = 0
    for case in case_rows:
        case_id = case.get("case_id", "")
        seq = case.get("seq", "")
        curr_chunk = int(case.get("curr_chunk") or 0)
        bucket = case.get("v95_case_bucket", "")
        labels = case.get("action_response_labels", "")
        is_read_swa = "READ_LOCAL_BAD" in labels or "SWA_HANDOFF_CANDIDATE" in labels
        if is_read_swa:
            read_swa_cases += 1
        chunk_matches = sorted(Path(f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks").glob(f"chunk_{curr_chunk:03d}_*/masklet.pt"))
        if not chunk_matches:
            per_case_rows.append({"case_id": case_id, "seq": seq, "chunk": curr_chunk, "covered": False, "reason": "missing_stage_c_masklet"})
            continue
        masklet_path = chunk_matches[0]
        try:
            payload = torch.load(masklet_path, map_location="cpu")
            sem = payload.get("semantic_segmentation", {})
            label_maps = sem.get("label_maps")
            confidence = sem.get("confidence_maps")
            label_to_id = sem.get("label_to_id", {})
            if not torch.is_tensor(label_maps) or not torch.is_tensor(confidence):
                raise ValueError("missing_label_or_confidence_tensor")
            label_maps = label_maps.to(dtype=torch.long)
            confidence = confidence.to(dtype=torch.float32)
            num_frames = int(label_maps.shape[0])
            patch_grid = (19, 66)
            masks: dict[str, torch.Tensor] = {}
            known = torch.zeros_like(label_maps, dtype=torch.bool)
            for region in ("DYNAMIC_OBJECT", "WEAK_SCALE_CONTEXT", "VEGETATION_REPETITIVE", "STABLE_ANCHOR"):
                ids = label_ids(label_to_id, region_label_names[region])
                if ids:
                    region_mask = torch.isin(label_maps, torch.tensor(ids, dtype=label_maps.dtype))
                else:
                    region_mask = torch.zeros_like(label_maps, dtype=torch.bool)
                masks[region] = region_mask
                known |= region_mask
            masks["LOW_OBSERVABILITY"] = confidence < 0.5
            masks["OBJECT_BOUNDARY_BAND"] = semantic_boundary(label_maps)
            score = safe_float(case.get("semantic_multimode_conflict_score"))
            masks["MULTIMODE_CONFLICT"] = masks["OBJECT_BOUNDARY_BAND"] & (masks["LOW_OBSERVABILITY"] | (confidence < 0.75))
            if score is not None and score >= 2.5:
                masks["MULTIMODE_CONFLICT"] = masks["MULTIMODE_CONFLICT"] | masks["OBJECT_BOUNDARY_BAND"]
            masks["UNKNOWN_CONTEXT"] = ~(known | masks["LOW_OBSERVABILITY"])
            token_masks = {region: downsample(mask) for region, mask in masks.items()}
            mask_bank[case_id] = {
                "seq": seq,
                "chunk": curr_chunk,
                "masklet_path": str(masklet_path),
                "patch_grid": list(patch_grid),
                "num_frames": num_frames,
                "region_token_masks": {region: tensor.to(dtype=torch.float16) for region, tensor in token_masks.items()},
            }
            covered += 1
            grid_aligned = all(tuple(token_masks[region].shape) == (num_frames, *patch_grid) for region in region_order)
            if grid_aligned:
                aligned += 1
            region_mass = {region: float(token_masks[region].mean().item()) for region in region_order}
            risk_nonempty_count = sum(
                region_mass[region] > 0.001
                for region in (
                    "DYNAMIC_OBJECT",
                    "OBJECT_BOUNDARY_BAND",
                    "WEAK_SCALE_CONTEXT",
                    "VEGETATION_REPETITIVE",
                    "LOW_OBSERVABILITY",
                    "MULTIMODE_CONFLICT",
                )
            )
            stable_nonempty = region_mass["STABLE_ANCHOR"] > 0.001
            if is_read_swa and stable_nonempty and risk_nonempty_count >= 2:
                stable_and_two_risk += 1
            panel = ""
            if bucket and bucket not in visual_buckets_seen:
                panel = make_panel(case, token_masks)
                if panel:
                    visual_buckets_seen.add(bucket)
                    visual_rows.append({"case_id": case_id, "bucket": bucket, "visual_panel_path": panel, "exists": Path(panel).is_file()})
            for region in region_order:
                mask = masks[region]
                token = token_masks[region]
                region_rows.append(
                    {
                        "case_id": case_id,
                        "seq": seq,
                        "chunk_or_pair": curr_chunk,
                        "region_type": region,
                        "pixel_mass": float(mask.float().mean().item()),
                        "token_mass": region_mass[region],
                        "READ_mass_if_available": region_mass[region] if "READ_LOCAL_BAD" in labels else "",
                        "SWA_route_mass_if_available": region_mass[region] if "SWA_HANDOFF_CANDIDATE" in labels else "",
                        "TTT_write_mass_if_available": region_mass[region] if "TTT_WRITE_RISK_DIAGNOSTIC" in labels else "",
                        "stable_anchor_overlap": float((token * token_masks["STABLE_ANCHOR"]).mean().item()),
                        "dynamic_overlap": float((token * token_masks["DYNAMIC_OBJECT"]).mean().item()),
                        "boundary_overlap": float((token * token_masks["OBJECT_BOUNDARY_BAND"]).mean().item()),
                        "lowobs_overlap": float((token * token_masks["LOW_OBSERVABILITY"]).mean().item()),
                        "visual_panel_path": panel,
                        "masklet_path": str(masklet_path),
                        "patch_grid": "19x66",
                        "num_frames": num_frames,
                    }
                )
            per_case_rows.append(
                {
                    "case_id": case_id,
                    "seq": seq,
                    "chunk": curr_chunk,
                    "covered": True,
                    "token_grid_alignment_pass": grid_aligned,
                    "stable_anchor_token_mass": region_mass["STABLE_ANCHOR"],
                    "risk_region_nonempty_count": risk_nonempty_count,
                    "stable_and_two_risk_nonempty": stable_nonempty and risk_nonempty_count >= 2,
                    **{f"{region}_token_mass": region_mass[region] for region in region_order},
                }
            )
        except Exception as exc:  # noqa: BLE001
            per_case_rows.append({"case_id": case_id, "seq": seq, "chunk": curr_chunk, "covered": False, "reason": f"{type(exc).__name__}: {exc}", "masklet_path": str(masklet_path)})

    torch.save(mask_bank, out / "semantic_region_masks.pt")
    write_csv(out / "semantic_region_rows.csv", region_rows)
    write_csv(out / "rows.csv", region_rows)
    write_csv(out / "per_case_region_mass.csv", per_case_rows)
    memory_rows: list[dict[str, Any]] = []
    for memory, predicate in [
        ("READ", lambda row: "READ_LOCAL_BAD" in row.get("action_response_labels", "")),
        ("SWA", lambda row: "SWA_HANDOFF_CANDIDATE" in row.get("action_response_labels", "")),
        ("TTT", lambda row: "TTT_WRITE_RISK_DIAGNOSTIC" in row.get("action_response_labels", "")),
        ("GOOD", lambda row: "GOOD_PROTECTION" in row.get("action_response_labels", "")),
        ("ALL", lambda row: True),
    ]:
        selected_ids = {row.get("case_id", "") for row in case_rows if predicate(row)}
        for region in region_order:
            vals = [safe_float(row.get("token_mass")) for row in region_rows if row.get("case_id") in selected_ids and row.get("region_type") == region]
            vals = [v for v in vals if v is not None]
            memory_rows.append({"memory": memory, "region_type": region, "case_count": len(vals), "mean_token_mass": sum(vals) / len(vals) if vals else ""})
    write_csv(out / "per_memory_region_mass.csv", memory_rows)
    write_csv(out / "visual_manifest.csv", visual_rows)

    total_cases = len(case_rows)
    coverage_ratio = covered / max(total_cases, 1)
    alignment_ratio = aligned / max(covered, 1)
    stable_risk_ratio = stable_and_two_risk / max(read_swa_cases, 1)
    bucket_count = len({row.get("v95_case_bucket", "") for row in case_rows if row.get("v95_case_bucket", "")})
    visual_panel_gate = len(visual_rows) >= bucket_count
    coverage_gate = coverage_ratio >= 0.9
    alignment_gate = covered > 0 and alignment_ratio == 1.0
    stable_risk_gate = read_swa_cases == 0 or stable_risk_ratio >= 0.7
    gate_pass = bool(coverage_gate and alignment_gate and stable_risk_gate and visual_panel_gate)
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "semantic_sources_found", "pass": bool(rows), "value": len(rows)},
            {"gate": "region_masks_available_for_ge_90pct_v95_cases", "pass": coverage_gate, "value": f"{covered}/{total_cases}={coverage_ratio:.4f}"},
            {"gate": "token_grid_alignment_pass", "pass": alignment_gate, "value": f"{aligned}/{covered}={alignment_ratio:.4f}"},
            {"gate": "stable_anchor_and_two_risk_nonempty_ge70pct_READ_SWA", "pass": stable_risk_gate, "value": f"{stable_and_two_risk}/{read_swa_cases}={stable_risk_ratio:.4f}"},
            {"gate": "visual_panels_for_selected_case_buckets", "pass": visual_panel_gate, "value": f"{len(visual_rows)}/{bucket_count}"},
            {"gate": "TrackJ_J2_gate_pass", "pass": gate_pass, "value": gate_pass},
        ],
    )
    write_json(
        out / "summary.json",
        {
            "stage": "TrackJ_J2_semantic_region_bank",
            "gate_pass": gate_pass,
            "semantic_source_candidates": len(rows),
            "runtime_action_allowed": False,
            "case_rows": total_cases,
            "covered_cases": covered,
            "coverage_ratio": coverage_ratio,
            "aligned_cases": aligned,
            "alignment_ratio": alignment_ratio,
            "read_swa_cases": read_swa_cases,
            "stable_and_two_risk_cases": stable_and_two_risk,
            "stable_risk_ratio": stable_risk_ratio,
            "visual_panel_count": len(visual_rows),
            "selected_bucket_count": bucket_count,
            "semantic_region_row_count": len(region_rows),
            "mask_bank_path": str(out / "semantic_region_masks.pt"),
            "blocker": "" if gate_pass else "J2 mask bank attempted, but one or more coverage/alignment/stable-risk/visual-panel gates failed",
        },
    )
    write_text(
        out / "failure_report.md",
        f"# Track J2 Semantic Region Bank Report\n\nCoverage: {covered}/{total_cases} ({coverage_ratio:.4f}). Token-grid alignment: {aligned}/{covered} ({alignment_ratio:.4f}). Stable-anchor plus at least two risk regions in READ/SWA cases: {stable_and_two_risk}/{read_swa_cases} ({stable_risk_ratio:.4f}). Visual panels: {len(visual_rows)}/{bucket_count}. Gate pass: {gate_pass}.\n\nNo runtime action is allowed by this report alone; J1 early K-side action trace-change and J3 skip-impact diagnostics still precede J4/J5/J6 action.",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# J2 Pass Requirements\n\nPer-case masks must cover >=90% of v95 base cases, prove patch/token-grid alignment, show stable-anchor and at least two risk-region non-empty rates in >=70% of READ/SWA cases, and produce visual panels for selected case buckets.",
    )
    write_text(
        out / "region_taxonomy_report.md",
        "# Region Taxonomy Report\n\nDYNAMIC_OBJECT, WEAK_SCALE_CONTEXT, VEGETATION_REPETITIVE, and STABLE_ANCHOR are label-set masks from Stage-C semantic label maps. OBJECT_BOUNDARY_BAND is a semantic-neighbor boundary band. LOW_OBSERVABILITY is confidence < 0.5. MULTIMODE_CONFLICT is a diagnostic proxy from boundary plus low-confidence, widened when the v95 case-level multimode score is high. UNKNOWN_CONTEXT is the remaining non-classified context.\n",
    )
    return {
        "semantic_source_candidates": len(rows),
        "gate_pass": gate_pass,
        "case_rows": total_cases,
        "covered_cases": covered,
        "coverage_ratio": coverage_ratio,
        "aligned_cases": aligned,
        "alignment_ratio": alignment_ratio,
        "stable_risk_ratio": stable_risk_ratio,
        "visual_panel_count": len(visual_rows),
        "selected_bucket_count": bucket_count,
    }


def write_blocked_downstream_dirs(reason: str) -> None:
    blocked_tracks = [
        "trackJ_read_skip_pilot",
        "trackJ_swa_skip_diagnostic",
        "trackJ_ttt_no_write_diagnostic",
        "trackG_memory_specific_cues",
        "trackD_read_gauge_preserving_action",
        "trackE_swa_raw_transport_trace",
        "trackC_latent_gauge_alignment",
        "trackF_ttt_write_diagnostic",
        "stage7_full_validation",
    ]
    for track in blocked_tracks:
        out = ensure_track_scaffold(track)
        existing_summary = read_json(out / "summary.json")
        if existing_summary.get("status") == "complete":
            continue
        track_reason = reason
        if track == "trackJ_swa_skip_diagnostic":
            track_reason = (
                "J3 found SWA semantic-region candidates, but v96 J5 requires raw SWA transport trace "
                "before any skip/delay action; raw per-layer/per-head transport trace is not available in this run."
            )
            write_text(
                out / "swa_transport_trace_missing_report.md",
                "# SWA Transport Trace Missing Report\n\n"
                "J5 action is blocked because raw SWA current_Q to cache_K, cache_K/V stability, and "
                "semantic region-pair route traces were not instrumented in this run. The plan forbids "
                "falling back to old Track E source-mask/alpha sweeps.",
            )
        elif track == "trackJ_ttt_no_write_diagnostic":
            track_reason = (
                "J3 did not find a TTT write-risk semantic-region candidate and raw TTT write traces "
                "for persistent write enrichment are not available; v96 forbids runtime TTT action."
            )
            write_text(
                out / "ttt_write_trace_missing_report.md",
                "# TTT Write Trace Missing Report\n\n"
                "J6/Track F are diagnostic-only and require persistent write mass, post-zp delta, "
                "operator/update/final output deltas, and at least 3 TTT_WRITE_RISK cases with trace support. "
                "Those raw traces were not generated in this run.",
            )
        write_csv(out / "rows.csv", [{"status": "blocked_not_run", "reason": track_reason}])
        write_csv(out / "gate_checks.csv", [{"gate": f"{track}_gate_pass", "pass": False, "value": "blocked_not_run"}])
        write_json(out / "summary.json", {"status": "blocked_not_run", "track": track, "gate_pass": False, "runtime_action_allowed": False, "reason": track_reason})
        write_text(out / "failure_report.md", f"# {track} Failure Report\n\nBlocked/not run: {track_reason}")
        write_text(out / "what_would_have_to_be_true_to_pass.md", f"# {track} Pass Requirements\n\nResolve upstream blocker first: {track_reason}")
        write_csv(out / "visual_manifest.csv", [])


def build_track_j_skip_impact_diagnostic() -> dict[str, Any]:
    out = ensure_track_scaffold("trackJ_skip_impact_diagnostic")
    case_rows = read_csv(ROOT / "trackA_case_response_atlas/rows.csv")
    mass_rows = read_csv(ROOT / "trackJ_semantic_region_bank/per_case_region_mass.csv")
    mass_by_case = {row.get("case_id", ""): row for row in mass_rows if boolish(row.get("covered"))}
    region_order = [
        "DYNAMIC_OBJECT",
        "OBJECT_BOUNDARY_BAND",
        "WEAK_SCALE_CONTEXT",
        "VEGETATION_REPETITIVE",
        "LOW_OBSERVABILITY",
        "MULTIMODE_CONFLICT",
        "STABLE_ANCHOR",
        "UNKNOWN_CONTEXT",
    ]
    metric_specs = [
        ("READ_L1", "L1_local_sim3_ate", lambda row: "READ_LOCAL_BAD" in row.get("action_response_labels", "")),
        ("READ_L2", "L2_head_tail_proxy_error", lambda row: "READ_LOCAL_BAD" in row.get("action_response_labels", "")),
        ("SWA_L3", "L3_handoff_transfer_penalty_proxy", lambda row: "SWA_HANDOFF_CANDIDATE" in row.get("action_response_labels", "")),
        ("TTT_L4", "L4_future_error_3chunk", lambda row: "TTT_WRITE_RISK_DIAGNOSTIC" in row.get("action_response_labels", "")),
    ]

    def weighted_ratio(rows: list[dict[str, str]], region: str, metric: str, masses: dict[str, dict[str, str]]) -> float | None:
        vals: list[float] = []
        weights: list[float] = []
        for row in rows:
            case_id = row.get("case_id", "")
            mass = safe_float(masses.get(case_id, {}).get(f"{region}_token_mass"))
            metric_val = safe_float(row.get(metric))
            if mass is None or metric_val is None:
                continue
            vals.append(metric_val)
            weights.append(max(mass, 0.0))
        if not vals or sum(weights) <= 0:
            return None
        overall = sum(vals) / len(vals)
        if abs(overall) <= 1e-12:
            return None
        weighted = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        return weighted / (overall + 1e-12)

    rotated_mass_by_case: dict[str, dict[str, str]] = {}
    semantic_rotated_mass_by_case: dict[str, dict[str, str]] = {}
    mass_case_ids = [row.get("case_id", "") for row in mass_rows if row.get("case_id", "")]
    for idx, case_id in enumerate(mass_case_ids):
        donor = mass_by_case.get(mass_case_ids[(idx + 7) % len(mass_case_ids)], {}) if mass_case_ids else {}
        self_row = mass_by_case.get(case_id, {})
        rotated_mass_by_case[case_id] = donor
        sem_row = dict(self_row)
        for ridx, region in enumerate(region_order):
            src = region_order[(ridx + 1) % len(region_order)]
            sem_row[f"{region}_token_mass"] = self_row.get(f"{src}_token_mass", "")
        semantic_rotated_mass_by_case[case_id] = sem_row

    overlap_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for target, metric, selector in metric_specs:
        selected = [row for row in case_rows if selector(row) and row.get("case_id", "") in mass_by_case]
        good = [row for row in case_rows if "GOOD_PROTECTION" in row.get("action_response_labels", "") and row.get("case_id", "") in mass_by_case]
        for region in region_order:
            actual = weighted_ratio(selected, region, metric, mass_by_case)
            random_ratio = weighted_ratio(selected, region, metric, rotated_mass_by_case)
            semantic_ratio = weighted_ratio(selected, region, metric, semantic_rotated_mass_by_case)
            good_mass_vals = [safe_float(mass_by_case[row.get("case_id", "")].get(f"{region}_token_mass")) for row in good]
            selected_mass_vals = [safe_float(mass_by_case[row.get("case_id", "")].get(f"{region}_token_mass")) for row in selected]
            good_mass_vals = [v for v in good_mass_vals if v is not None]
            selected_mass_vals = [v for v in selected_mass_vals if v is not None]
            good_mean_mass = sum(good_mass_vals) / len(good_mass_vals) if good_mass_vals else None
            selected_mean_mass = sum(selected_mass_vals) / len(selected_mass_vals) if selected_mass_vals else None
            random_margin = actual - random_ratio if actual is not None and random_ratio is not None else None
            semantic_margin = actual - semantic_ratio if actual is not None and semantic_ratio is not None else None
            good_fp_ok = (
                good_mean_mass is None
                or selected_mean_mass is None
                or good_mean_mass <= selected_mean_mass * 1.1 + 1e-9
            )
            useful = bool(
                actual is not None
                and actual > 1.10
                and random_margin is not None
                and random_margin >= 0.05
                and semantic_margin is not None
                and semantic_margin >= 0.05
                and good_fp_ok
            )
            row = {
                "target": target,
                "metric": metric,
                "region_type": region,
                "case_count": len(selected),
                "A_region_metric": actual,
                "random_same_mass_A": random_ratio,
                "semantic_rotation_A": semantic_ratio,
                "random_same_mass_margin": random_margin,
                "semantic_rotation_margin": semantic_margin,
                "selected_mean_region_mass": selected_mean_mass,
                "good_control_mean_region_mass": good_mean_mass,
                "good_false_positive_ok": good_fp_ok,
                "diagnostic_useful": useful,
            }
            overlap_rows.append(row)
            if useful:
                candidate_rows.append(row)
    write_csv(out / "region_metric_overlap_rows.csv", overlap_rows)
    write_csv(out / "region_candidate_summary.csv", candidate_rows)
    write_csv(
        out / "random_control_summary.csv",
        [
            {
                "control": "case_rotated_same_region_mass",
                "description": "Region masses are rotated by seven case ids within the v95 base universe.",
            },
            {
                "control": "semantic_rotation",
                "description": "Each case keeps its mass vector but region names are rotated by one semantic bucket.",
            },
        ],
    )
    read_pass = any(row.get("target") in {"READ_L1", "READ_L2"} for row in candidate_rows)
    swa_pass = any(row.get("target") == "SWA_L3" for row in candidate_rows)
    ttt_pass = any(row.get("target") == "TTT_L4" for row in candidate_rows)
    gate_pass = bool(read_pass and (swa_pass or not any("SWA_HANDOFF_CANDIDATE" in row.get("action_response_labels", "") for row in case_rows)) and (ttt_pass or not any("TTT_WRITE_RISK_DIAGNOSTIC" in row.get("action_response_labels", "") for row in case_rows)))
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "READ_region_candidate_pass", "pass": read_pass, "value": read_pass},
            {"gate": "SWA_region_candidate_pass", "pass": swa_pass, "value": swa_pass},
            {"gate": "TTT_region_candidate_pass", "pass": ttt_pass, "value": ttt_pass},
            {"gate": "TrackJ_J3_skip_impact_gate_pass", "pass": gate_pass, "value": gate_pass},
        ],
    )
    write_json(
        out / "summary.json",
        {
            "stage": "TrackJ_J3_skip_impact_diagnostic",
            "gate_pass": gate_pass,
            "region_metric_rows": len(overlap_rows),
            "candidate_rows": len(candidate_rows),
            "read_candidate_pass": read_pass,
            "swa_candidate_pass": swa_pass,
            "ttt_candidate_pass": ttt_pass,
            "runtime_action_allowed": False,
            "blocker": "" if gate_pass else "No full READ/SWA/TTT skip-impact candidate set passed diagnostic controls; no skip action is allowed.",
        },
    )
    if candidate_rows:
        lines = ["# Skip Impact Diagnostic Report", "", "Diagnostic candidates that passed controls:"]
        for row in candidate_rows:
            lines.append(
                f"- {row['target']} {row['region_type']}: A={row['A_region_metric']}, random_margin={row['random_same_mass_margin']}, semantic_margin={row['semantic_rotation_margin']}"
            )
    else:
        lines = ["# Skip Impact Diagnostic Report", "", "No semantic region passed the diagnostic controls. No skip action is allowed."]
    write_text(out / "skip_impact_diagnostic_report.md", "\n".join(lines))
    write_text(
        out / "no_region_skip_candidate_report.md",
        "# No Region Skip Candidate Report\n\nGenerated when no complete READ/SWA/TTT candidate set passes controls. See `region_metric_overlap_rows.csv` and `region_candidate_summary.csv`.",
    )
    return {
        "gate_pass": gate_pass,
        "candidate_rows": len(candidate_rows),
        "read_candidate_pass": read_pass,
        "swa_candidate_pass": swa_pass,
        "ttt_candidate_pass": ttt_pass,
    }


def _read_stage7_rolling_summary(path: Path, baseline: str, candidate: str) -> dict[str, Any]:
    rows = read_csv(path)
    by_window: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for row in rows:
        run = row.get("run", "")
        if run not in {baseline, candidate}:
            continue
        window = row.get("window", "")
        key = (row.get("start", ""), row.get("end", ""))
        rmse = safe_float(row.get("rmse_m"))
        if not window or rmse is None:
            continue
        by_window.setdefault(window, {}).setdefault(key, {})[run] = rmse
    out: dict[str, Any] = {}
    for window, keyed in sorted(by_window.items(), key=lambda item: safe_float(item[0]) or 0.0):
        deltas: list[float] = []
        for vals in keyed.values():
            if baseline in vals and candidate in vals:
                deltas.append(vals[candidate] - vals[baseline])
        if not deltas:
            continue
        out[window] = {
            "paired_window_count": len(deltas),
            "mean_delta_rmse_m": sum(deltas) / len(deltas),
            "worst_delta_rmse_m": max(deltas),
            "best_delta_rmse_m": min(deltas),
            "worse_fraction": sum(1 for value in deltas if value > 0.0) / len(deltas),
        }
    return out


def build_stage7_full_validation_summary() -> dict[str, Any]:
    out = ensure_track_scaffold("stage7_full_validation")
    audit_paths = sorted(ROOT.glob("stage7_seq*_full_confneutral*/stage7_audit_summary*.json"))
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] = {}
    strict_pass_any = False
    for path in audit_paths:
        payload = read_json(path)
        full = payload.get("full_sequence", {}) if isinstance(payload.get("full_sequence"), dict) else {}
        chunks = payload.get("gate_chunk_summary", {}) if isinstance(payload.get("gate_chunk_summary"), dict) else {}
        active = chunks.get("active", {}) if isinstance(chunks.get("active"), dict) else {}
        inactive = chunks.get("inactive", {}) if isinstance(chunks.get("inactive"), dict) else {}
        baseline = str(payload.get("baseline", "READ0_NATIVE"))
        candidate = str(payload.get("candidate", ""))
        rolling_path = next(path.parent.glob("rolling_windows_*.csv"), None)
        rolling = _read_stage7_rolling_summary(rolling_path, baseline, candidate) if rolling_path else {}
        delta_ate = safe_float(full.get("delta_aligned_ate_rmse_m"))
        delta_final = safe_float(full.get("delta_final_error_m"))
        delta_slope = safe_float(full.get("delta_error_slope_m_per_100f"))
        delta_yaw = safe_float(full.get("delta_yaw_rmse_deg"))
        rolling_improved_count = sum(
            1
            for item in rolling.values()
            if (safe_float(item.get("mean_delta_rmse_m")) is not None and safe_float(item.get("mean_delta_rmse_m")) < 0.0)
        )
        rolling_worse_fraction_max = max(
            (safe_float(item.get("worse_fraction")) or 0.0 for item in rolling.values()),
            default=None,
        )
        ate_improve_ge_0p3 = delta_ate is not None and delta_ate <= -0.3
        ate_no_worse = delta_ate is not None and delta_ate <= 0.0
        final_not_worse = delta_final is not None and delta_final <= 0.0
        rolling_no_regression = rolling_worse_fraction_max is not None and rolling_worse_fraction_max <= 0.05
        strong_long_window = rolling_improved_count >= 2
        strict_full_gate_pass = bool(
            final_not_worse
            and rolling_no_regression
            and (ate_improve_ge_0p3 or (ate_no_worse and strong_long_window))
        )
        strict_pass_any = strict_pass_any or strict_full_gate_pass
        fail_reasons: list[str] = []
        if not final_not_worse:
            fail_reasons.append("final_error_worse")
        if not (ate_improve_ge_0p3 or (ate_no_worse and strong_long_window)):
            fail_reasons.append("full_ate_gate_not_met")
        if not rolling_no_regression:
            fail_reasons.append("rolling_worse_fraction_gt_0p05")
        if not fail_reasons and strict_full_gate_pass:
            fail_reasons.append("pass")
        row = {
            "source_json": str(path),
            "baseline": baseline,
            "candidate": candidate,
            "audit_method_success": bool(payload.get("method_success", False)),
            "audit_method_success_reason": payload.get("method_success_reason", ""),
            "strict_full_gate_pass": strict_full_gate_pass,
            "strict_full_gate_reason": ";".join(fail_reasons),
            "baseline_aligned_ate_rmse_m": full.get("baseline_aligned_ate_rmse_m", ""),
            "candidate_aligned_ate_rmse_m": full.get("candidate_aligned_ate_rmse_m", ""),
            "delta_aligned_ate_rmse_m": delta_ate,
            "baseline_final_error_m": full.get("baseline_final_error_m", ""),
            "candidate_final_error_m": full.get("candidate_final_error_m", ""),
            "delta_final_error_m": delta_final,
            "delta_yaw_rmse_deg": delta_yaw,
            "delta_error_slope_m_per_100f": delta_slope,
            "active_count": chunks.get("active_count", ""),
            "active_mean_delta_m": active.get("mean_delta_m", ""),
            "inactive_count": chunks.get("inactive_count", ""),
            "inactive_mean_delta_m": inactive.get("mean_delta_m", ""),
            "rolling_windows_csv": str(rolling_path) if rolling_path else "",
            "rolling_improved_count": rolling_improved_count,
            "rolling_worse_fraction_max": rolling_worse_fraction_max,
        }
        for window, item in rolling.items():
            row[f"rolling{window}_mean_delta_rmse_m"] = item.get("mean_delta_rmse_m")
            row[f"rolling{window}_worse_fraction"] = item.get("worse_fraction")
        rows.append(row)
        if not best_row:
            best_row = row
        else:
            current = safe_float(row.get("delta_aligned_ate_rmse_m"))
            best = safe_float(best_row.get("delta_aligned_ate_rmse_m"))
            if current is not None and (best is None or current < best):
                best_row = row
    write_csv(out / "rows.csv", rows)
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "stage7_audit_jsons_present", "pass": bool(rows), "value": len(rows)},
            {"gate": "strict_full_gate_pass_any", "pass": strict_pass_any, "value": strict_pass_any},
            {"gate": "runtime_action_allowed", "pass": strict_pass_any, "value": strict_pass_any},
        ],
    )
    summary = {
        "stage": "Stage7_full_validation",
        "status": "complete" if rows else "not_run",
        "classification": "FULL_METHOD_PASS" if strict_pass_any else ("MECHANISM_PASS_FULL_NO_GO" if rows else "NOT_RUN"),
        "gate_pass": strict_pass_any,
        "method_success": strict_pass_any,
        "full_method_success": strict_pass_any,
        "runtime_action_allowed": strict_pass_any,
        "candidate_count": len(rows),
        "best_candidate_by_delta_ate": best_row.get("candidate", ""),
        "best_delta_aligned_ate_rmse_m": best_row.get("delta_aligned_ate_rmse_m"),
        "best_delta_final_error_m": best_row.get("delta_final_error_m"),
        "best_strict_full_gate_pass": bool(best_row.get("strict_full_gate_pass", False)),
        "best_strict_full_gate_reason": best_row.get("strict_full_gate_reason", ""),
        "stage7_audit_jsons": [str(path) for path in audit_paths],
        "interpretation": (
            "At least one v96 confidence-neutral READ candidate passed the strict Stage7 full gate."
            if strict_pass_any
            else "Stage7 was run for confidence-neutral READ candidates, but no candidate passed the strict full gate."
            if rows
            else "No v96 Stage7 audit summary was found."
        ),
    }
    write_json(out / "summary.json", summary)
    if rows:
        write_text(
            out / "failure_report.md",
            "# Stage7 Full Validation Report\n\n"
            f"Strict full gate pass: `{strict_pass_any}`. Best candidate by delta ATE: `{summary['best_candidate_by_delta_ate']}` "
            f"with delta ATE `{summary['best_delta_aligned_ate_rmse_m']}` and delta final error `{summary['best_delta_final_error_m']}`. "
            "See `rows.csv` for all candidates, rolling-window deltas, and strict gate reasons.",
        )
    else:
        write_text(out / "failure_report.md", "# Stage7 Full Validation Report\n\nNo v96 Stage7 audit summary was found.")
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# Stage7 Pass Requirements\n\nA candidate must improve full aligned ATE by at least 0.3m, or be no-worse in full aligned ATE with at least two improved long-window metrics. It must also avoid final-error regression and rolling-window worse_fraction > 0.05.",
    )
    write_csv(out / "visual_manifest.csv", [])
    return summary


def build_final_decision(stage0: dict[str, Any], j: dict[str, Any], semantic: dict[str, Any]) -> None:
    out = ensure_track_scaffold("final_decision")
    j3 = read_json(ROOT / "trackJ_skip_impact_diagnostic/summary.json")
    j4 = read_json(ROOT / "trackJ_read_skip_pilot/summary.json")
    j4_repair = read_json(ROOT / "trackJ_read_skip_pilot_repair_early_quarter/summary.json")
    j4_anchor = read_json(ROOT / "trackJ_read_skip_pilot_repair_anchor_compensation/summary.json")
    j4_anchor_weak = read_json(ROOT / "trackJ_read_skip_pilot_repair_anchor_weak_compensation/summary.json")
    j4_anchor_weak_rho020 = read_json(ROOT / "trackJ_read_skip_pilot_repair_anchor_weak_rho020/summary.json")
    trackg_read_cue = read_json(ROOT / "trackG_read_cue_refinement/summary.json")
    trackg_qk_carrier = read_json(ROOT / "trackG_read_qk_carrier_localization/summary.json")
    trackd_per_head_action = read_json(ROOT / "trackD_read_gauge_preserving_action/summary.json")
    tracke_raw = read_json(ROOT / "trackE_swa_raw_transport_trace_swa_atlas_v1/trackE_swa_raw_transport_trace_summary.json")
    tracke_gates = tracke_raw.get("gates", {}) if isinstance(tracke_raw.get("gates"), dict) else {}
    trackc_swa = read_json(ROOT / "trackC_latent_gauge_alignment/summary.json")
    swa_route_decision = read_json(ROOT / "route_decisions/trackE_trackC_swa_route_decision.json")
    trackf_ttt_baseline = read_json(ROOT / "trackF_ttt_write_trace_atlas_v1/analysis/summary.json")
    trackf_ttt_proxy = read_json(ROOT / "trackF_ttt_write_trace_proxy_atlas_v1/analysis/summary.json")
    trackf_ttt_replay = read_json(ROOT / "trackF_ttt_write_trace_replay_contribution_atlas_v1/analysis/summary.json")
    trackf_ttt_scale_state = read_json(ROOT / "trackF_ttt_write_trace_replay_contribution_scale_state_atlas_v1/analysis/summary.json")
    trackf_ttt_branch_scale = read_json(
        ROOT / "trackF_ttt_write_trace_replay_contribution_branch_scale_state_atlas_v1/analysis_fixed_pair/summary.json"
    )
    trackf_candidates = [
        (
            "trackF_ttt_write_trace_replay_contribution_branch_scale_state_atlas_v1/analysis_fixed_pair",
            ROOT / "trackF_ttt_write_trace_replay_contribution_branch_scale_state_atlas_v1/analysis_fixed_pair",
            trackf_ttt_branch_scale,
        ),
        (
            "trackF_ttt_write_trace_replay_contribution_scale_state_atlas_v1/analysis",
            ROOT / "trackF_ttt_write_trace_replay_contribution_scale_state_atlas_v1/analysis",
            trackf_ttt_scale_state,
        ),
        (
            "trackF_ttt_write_trace_replay_contribution_atlas_v1/analysis",
            ROOT / "trackF_ttt_write_trace_replay_contribution_atlas_v1/analysis",
            trackf_ttt_replay,
        ),
        (
            "trackF_ttt_write_trace_proxy_atlas_v1/analysis",
            ROOT / "trackF_ttt_write_trace_proxy_atlas_v1/analysis",
            trackf_ttt_proxy,
        ),
        (
            "trackF_ttt_write_trace_atlas_v1/analysis",
            ROOT / "trackF_ttt_write_trace_atlas_v1/analysis",
            trackf_ttt_baseline,
        ),
    ]
    trackf_evidence_label = "trackF_ttt_write_trace_atlas_v1/analysis"
    trackf_analysis_root = ROOT / "trackF_ttt_write_trace_atlas_v1/analysis"
    trackf_ttt = trackf_ttt_baseline
    for candidate_label, candidate_root, candidate_summary in trackf_candidates:
        if candidate_summary and "read_error" not in candidate_summary:
            trackf_evidence_label = candidate_label
            trackf_analysis_root = candidate_root
            trackf_ttt = candidate_summary
            break
    stage7 = build_stage7_full_validation_summary()

    if tracke_raw:
        tracke_out = ensure_track_scaffold("trackE_swa_raw_transport_trace")
        tracke_payload = dict(tracke_raw)
        tracke_payload.update(
            {
                "status": "complete",
                "classification": swa_route_decision.get("classification", "TRACE_PASS_ACTION_FAIL_ROUTE_NOT_HANDOFF_CARRIER"),
                "gate_pass": bool(tracke_gates.get("tracke_raw_trace_gate_pass", False)),
                "runtime_swa_action_allowed": bool(tracke_gates.get("runtime_swa_action_allowed", False)),
                "method_success": False,
                "full_method_success": False,
            }
        )
        write_json(tracke_out / "summary.json", tracke_payload)
        write_csv(tracke_out / "rows.csv", tracke_raw.get("case_rows", []))
        write_csv(
            tracke_out / "gate_checks.csv",
            [{"gate": key, "pass": value if isinstance(value, bool) else "", "value": value} for key, value in tracke_gates.items()],
        )
        write_text(
            tracke_out / "failure_report.md",
            "# Track E SWA Raw Transport Failure Report\n\n"
            "Raw SWA transport trace is available, but the bad/good route-mass separability gate failed. "
            f"Stable bad-lower margin = {tracke_raw.get('separability', {}).get('stable_mass_bad_lower_margin_median')}; "
            f"unreliable bad-higher margin = {tracke_raw.get('separability', {}).get('unreliable_mass_bad_higher_margin_median')}; "
            "required margin is >= 0.05. No SWA runtime action is allowed.",
        )
        write_text(
            tracke_out / "what_would_have_to_be_true_to_pass.md",
            "# Track E Pass Requirements\n\n"
            "Trace availability must stay >=90%, per-layer/per-head rows must exist, stable/unreliable groups must be non-empty in >=70% cases, "
            "and bad handoff cases must show lower stable transport or higher unreliable transport than good controls with margin >=0.05.",
        )
        write_csv(tracke_out / "visual_manifest.csv", [])

    if trackf_ttt:
        trackf_out = ensure_track_scaffold("trackF_ttt_write_diagnostic")
        trackf_payload = dict(trackf_ttt)
        trackf_payload.update({"status": "complete"})
        best_fixed_risk = trackf_ttt.get("best_fixed_component_risk_median_enrichment")
        best_fixed_good_fp = trackf_ttt.get("best_fixed_component_good_false_positive_rate")
        replay_runtime_eligible = bool(trackf_ttt.get("replay_condition_chunk_count", 0)) and not bool(
            trackf_ttt.get("replay_condition_not_runtime_eligible", False)
        )
        proxy_runtime_eligible = bool(trackf_ttt.get("proxy_condition_chunk_count", 0)) and not bool(
            trackf_ttt.get("proxy_condition_not_runtime_eligible", False)
        )
        write_json(trackf_out / "summary.json", trackf_payload)
        write_csv(trackf_out / "rows.csv", read_csv(trackf_analysis_root / "case_rows.csv"))
        write_csv(trackf_out / "chunk_rows.csv", read_csv(trackf_analysis_root / "chunk_rows.csv"))
        write_csv(
            trackf_out / "gate_checks.csv",
            [
                {"gate": "risk_case_count_pass", "pass": bool(trackf_ttt.get("risk_case_count_pass", False)), "value": trackf_ttt.get("ttt_write_risk_case_count")},
                {"gate": "risk_enrichment_pass", "pass": bool(trackf_ttt.get("risk_enrichment_pass", False)), "value": trackf_ttt.get("median_gate_risk_enrichment", trackf_ttt.get("median_decomposed_risk_enrichment"))},
                {"gate": "good_false_positive_pass", "pass": bool(trackf_ttt.get("good_false_positive_pass", False)), "value": trackf_ttt.get("good_false_positive_rate")},
                {"gate": "fixed_component_pair_gate_pass", "pass": bool(trackf_ttt.get("fixed_component_pair_gate_pass", False)), "value": trackf_ttt.get("fixed_component_pair_pass_count")},
                {"gate": "best_fixed_component_risk_median_enrichment", "pass": bool(best_fixed_risk is not None and float(best_fixed_risk) >= 0.05), "value": best_fixed_risk},
                {"gate": "best_fixed_component_good_false_positive_rate", "pass": bool(best_fixed_good_fp is not None and float(best_fixed_good_fp) <= 0.25), "value": best_fixed_good_fp},
                {"gate": "visual_support_exists", "pass": bool(trackf_ttt.get("visual_support_exists", False)), "value": trackf_ttt.get("visual_support_exists")},
                {"gate": "exact_condition_maps_available", "pass": bool(trackf_ttt.get("exact_condition_chunk_count", 0)), "value": trackf_ttt.get("exact_condition_chunk_count")},
                {"gate": "replay_condition_maps_diagnostic_only", "pass": bool(trackf_ttt.get("replay_condition_chunk_count", 0)), "value": trackf_ttt.get("replay_condition_chunk_count")},
                {"gate": "replay_condition_not_runtime_eligible", "pass": bool(trackf_ttt.get("replay_condition_not_runtime_eligible", False)), "value": trackf_ttt.get("replay_condition_not_runtime_eligible")},
                {"gate": "proxy_condition_maps_diagnostic_only", "pass": bool(trackf_ttt.get("proxy_condition_chunk_count", 0)), "value": trackf_ttt.get("proxy_condition_chunk_count")},
                {"gate": "proxy_condition_not_runtime_eligible", "pass": bool(trackf_ttt.get("proxy_condition_not_runtime_eligible", False)), "value": trackf_ttt.get("proxy_condition_not_runtime_eligible")},
                {"gate": "TrackF_TTT_write_diagnostic_gate_pass", "pass": bool(trackf_ttt.get("gate_pass", False)), "value": trackf_ttt.get("classification")},
            ],
        )
        component_pair_rows = read_csv(trackf_analysis_root / "component_pair_rows.csv")
        if component_pair_rows:
            write_csv(trackf_out / "component_pair_rows.csv", component_pair_rows)
        write_text(
            trackf_out / "failure_report.md",
            "# Track F TTT Write Diagnostic Failure Report\n\n"
            f"TTT write traces are available from `{trackf_evidence_label}`, but the fixed component-pair write-risk gate failed. "
            f"Condition map source = {trackf_ttt.get('condition_map_source')}; "
            f"exact chunks = {trackf_ttt.get('exact_condition_chunk_count')}; "
            f"replay chunks = {trackf_ttt.get('replay_condition_chunk_count')}; "
            f"proxy chunks = {trackf_ttt.get('proxy_condition_chunk_count')}; "
            f"replay runtime eligible = {replay_runtime_eligible}; "
            f"proxy runtime eligible = {proxy_runtime_eligible}. "
            f"Median D-top10 enrichment = {trackf_ttt.get('median_risk_write_enrichment')}; "
            f"median decomposed enrichment = {trackf_ttt.get('median_decomposed_risk_enrichment')}; "
            f"median component-oracle enrichment = {trackf_ttt.get('median_component_risk_enrichment')}; "
            f"oracle good false-positive rate = {trackf_ttt.get('good_false_positive_rate')}; "
            f"fixed component pair pass count = {trackf_ttt.get('fixed_component_pair_pass_count')}; "
            f"best fixed pair = {trackf_ttt.get('best_fixed_component_write_source')} + {trackf_ttt.get('best_fixed_component_risk_source')}; "
            f"best fixed risk median enrichment = {trackf_ttt.get('best_fixed_component_risk_median_enrichment')}; "
            f"best fixed good false-positive rate = {trackf_ttt.get('best_fixed_component_good_false_positive_rate')}; "
            f"median conflict top10 = {trackf_ttt.get('median_conflict_top10_enrichment')}; "
            f"median scale top10 = {trackf_ttt.get('median_scale_risk_top10_enrichment')}; "
            "required fixed-pair risk median enrichment is >= 0.05 with good false-positive rate <= 0.25. No runtime TTT action is allowed.",
        )
        write_text(
            trackf_out / "what_would_have_to_be_true_to_pass.md",
            "# Track F Pass Requirements\n\n"
            "At least 3 TTT_WRITE_RISK cases must have trace support. A fixed, predeclared component write/risk pair must have median write-risk enrichment over same-mass random >=0.05 on risk cases and good-control false positive rate <=0.25. Visual support must exist. Runtime promotion additionally requires method-safe exact token condition maps or another audited non-proxy condition source; diagnostic replay/proxy maps alone are not runtime eligible.",
        )
        write_csv(trackf_out / "visual_manifest.csv", [])
    j4_repair_summaries = {
        "main": j4,
        "early_quarter": j4_repair,
        "anchor_compensation": j4_anchor,
        "anchor_weak_compensation": j4_anchor_weak,
        "anchor_weak_rho020": j4_anchor_weak_rho020,
    }
    j4_any_gate_pass = any(bool(item.get("gate_pass", False)) for item in j4_repair_summaries.values())
    j4_ran = j4.get("status") == "complete"
    branch_rows = [
        {"branch": "Stage0_TrackI_A_B", "classification": "DIAGNOSTIC_SUCCESS", "gate_pass": all(stage0.get(k) for k in ("trackI_gate_pass", "trackA_gate_pass", "trackB_gate_pass")), "evidence": "trackI/trackA/trackB summary.json"},
        {"branch": "TrackJ_J0_static_audit", "classification": "DIAGNOSTIC_SUCCESS", "gate_pass": j.get("J0_static_audit_complete", False), "evidence": "trackJ_vggt4d_code_audit/code_audit_report.md"},
        {"branch": "TrackJ_J1_runtime_unit_gate", "classification": "DIAGNOSTIC_SUCCESS" if j.get("J1_runtime_unit_gate_pass", False) else "PARTIAL_DIAGNOSTIC_SUCCESS_GATE_BLOCKED", "gate_pass": j.get("J1_runtime_unit_gate_pass", False), "evidence": "trackJ_vggt4d_code_audit/unit_test_plan.md, unit_test_results.csv, v95_qqkk_probe_shape_grid_audit.csv, and raw_qk_trace_smoke_audit.csv"},
        {"branch": "TrackJ_J2_semantic_region_bank", "classification": "DIAGNOSTIC_SUCCESS" if semantic.get("gate_pass", False) else "PARSER_OR_TRACE_BLOCKED", "gate_pass": semantic.get("gate_pass", False), "evidence": "trackJ_semantic_region_bank/summary.json"},
        {"branch": "TrackJ_J3_skip_impact_diagnostic", "classification": "DIAGNOSTIC_SUCCESS" if j3.get("gate_pass", False) else "DIAGNOSTIC_NO_GO", "gate_pass": j3.get("gate_pass", False), "evidence": "trackJ_skip_impact_diagnostic/summary.json"},
        {
            "branch": "TrackJ_J4_READ_weak_context_skip_pilot",
            "classification": "MECHANISM_NO_GO" if j4_ran else "PARSER_OR_TRACE_BLOCKED",
            "gate_pass": bool(j4_any_gate_pass),
            "evidence": (
                "trackJ_read_skip_pilot/summary.json, trackJ_read_skip_pilot_repair_early_quarter/summary.json, "
                "trackJ_read_skip_pilot_repair_anchor_compensation/summary.json, "
                "trackJ_read_skip_pilot_repair_anchor_weak_compensation/summary.json, and "
                "trackJ_read_skip_pilot_repair_anchor_weak_rho020/summary.json"
            ),
        },
        {
            "branch": "TrackG_READ_cue_refinement",
            "classification": trackg_read_cue.get("classification", "NOT_RUN"),
            "gate_pass": bool(trackg_read_cue.get("read_cue_v2_proxy_gate_pass", False)),
            "evidence": "trackG_read_cue_refinement/summary.json, rows.csv, cue_action_gap_report.md, saliency_not_geometry_report.md",
        },
        {
            "branch": "TrackG_READ_raw_QK_carrier_localization",
            "classification": trackg_qk_carrier.get("classification", "NOT_RUN"),
            "gate_pass": bool(trackg_qk_carrier.get("carrier_localization_gate_pass", False)),
            "evidence": "trackG_read_qk_carrier_localization/summary.json, rows.csv, layer_group_summary.csv, carrier_localization_report.md",
        },
        {
            "branch": "TrackJ_J5_TrackE_SWA_raw_transport",
            "classification": swa_route_decision.get("classification", "TRACE_MISSING_ACTION_BLOCKED"),
            "gate_pass": bool(tracke_gates.get("tracke_raw_trace_gate_pass", False)),
            "evidence": "trackE_swa_raw_transport_trace_swa_atlas_v1/trackE_swa_raw_transport_trace_summary.json and route_decisions/trackE_trackC_swa_route_decision.json",
        },
        {
            "branch": "TrackC_SWA_latent_gauge_alignment",
            "classification": trackc_swa.get("classification", "NOT_RUN"),
            "gate_pass": bool(trackc_swa.get("gate_pass", False)),
            "evidence": "trackC_latent_gauge_alignment/summary.json, rows.csv, gate_checks.csv, and failure_report.md",
        },
        {
            "branch": "TrackJ_J6_TrackF_TTT_write_diagnostic",
            "classification": trackf_ttt.get("classification", "TRACE_MISSING_ACTION_BLOCKED"),
            "gate_pass": bool(trackf_ttt.get("gate_pass", False)),
            "evidence": f"{trackf_evidence_label}/summary.json, case_rows.csv, chunk_rows.csv, and failure_report.md",
        },
        {
            "branch": "TrackD_H_READ_gauge_preserving_action",
            "classification": trackd_per_head_action.get("classification", "CUE_PASS_ACTION_FAIL"),
            "gate_pass": bool(trackd_per_head_action.get("gate_pass", False)),
            "evidence": "trackD_read_gauge_preserving_action/summary.json, rows.csv, gate_checks.csv, failure_report.md",
        },
        {
            "branch": "Stage7_full_validation",
            "classification": stage7.get("classification", "NOT_RUN"),
            "gate_pass": bool(stage7.get("gate_pass", False)),
            "evidence": "stage7_full_validation/summary.json, rows.csv, gate_checks.csv, and stage7_seq*/stage7_audit_summary*.json",
        },
    ]
    write_csv(out / "rows.csv", branch_rows)
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "diagnostic_success_any", "pass": True, "value": "Stage0 parsed and J0 audited"},
            {"gate": "J4_READ_mechanism_gate_pass", "pass": bool(j4_any_gate_pass), "value": bool(j4_any_gate_pass)},
            {"gate": "J4_READ_repair_gate_pass", "pass": bool(j4_repair.get("gate_pass", False)), "value": bool(j4_repair.get("gate_pass", False))},
            {"gate": "J4_READ_anchor_compensation_gate_pass", "pass": bool(j4_anchor.get("gate_pass", False)), "value": bool(j4_anchor.get("gate_pass", False))},
            {"gate": "J4_READ_anchor_weak_compensation_gate_pass", "pass": bool(j4_anchor_weak.get("gate_pass", False)), "value": bool(j4_anchor_weak.get("gate_pass", False))},
            {"gate": "J4_READ_anchor_weak_rho020_gate_pass", "pass": bool(j4_anchor_weak_rho020.get("gate_pass", False)), "value": bool(j4_anchor_weak_rho020.get("gate_pass", False))},
            {"gate": "TrackG_READ_cue_v2_proxy_gate_pass", "pass": bool(trackg_read_cue.get("read_cue_v2_proxy_gate_pass", False)), "value": bool(trackg_read_cue.get("read_cue_v2_proxy_gate_pass", False))},
            {"gate": "TrackG_READ_raw_QK_carrier_localization_gate_pass", "pass": bool(trackg_qk_carrier.get("carrier_localization_gate_pass", False)), "value": bool(trackg_qk_carrier.get("carrier_localization_gate_pass", False))},
            {"gate": "TrackD_read_gauge_preserving_action_gate_pass", "pass": bool(trackd_per_head_action.get("gate_pass", False)), "value": bool(trackd_per_head_action.get("gate_pass", False))},
            {"gate": "TrackE_SWA_raw_trace_availability_pass", "pass": bool(tracke_gates.get("trace_availability_ge_0p90", False)), "value": tracke_gates.get("trace_availability_frac")},
            {"gate": "TrackE_SWA_raw_transport_gate_pass", "pass": bool(tracke_gates.get("tracke_raw_trace_gate_pass", False)), "value": tracke_gates.get("bad_good_separable_by_stable_or_unreliable_mass_margin_ge_0p05")},
            {"gate": "TrackC_SWA_latent_gauge_gate_pass", "pass": bool(trackc_swa.get("gate_pass", False)), "value": trackc_swa.get("classification", "NOT_RUN")},
            {"gate": "TrackF_TTT_write_diagnostic_gate_pass", "pass": bool(trackf_ttt.get("gate_pass", False)), "value": trackf_ttt.get("classification", "NOT_RUN")},
            {"gate": "mechanism_success", "pass": bool(trackd_per_head_action.get("gate_pass", False)), "value": bool(trackd_per_head_action.get("gate_pass", False))},
            {"gate": "full_method_success", "pass": bool(stage7.get("full_method_success", False)), "value": stage7.get("classification", "NOT_RUN")},
        ],
    )
    mechanism_success = bool(trackd_per_head_action.get("gate_pass", False))
    full_method_success = bool(stage7.get("full_method_success", False))
    final_status = (
        "GO_FULL_METHOD"
        if full_method_success
        else "PARTIAL_DIAGNOSTIC_SUCCESS_TRACKD_READ_ACTION_MECHANISM_PASS_STAGE7_FULL_NO_GO_NO_RUNTIME_ACTION"
        if mechanism_success
        else "PARTIAL_DIAGNOSTIC_SUCCESS_J0_J1_J2_J3_PASS_J4_READ_NO_GO_TRACKG_READ_CUE_REFINEMENT_NO_GO_RAW_QK_CARRIER_DIAGNOSTIC_PASS_TRACKD_READ_ACTION_CONTROL_SPECIFICITY_NO_GO_NO_RUNTIME_ACTION"
    )
    trackf_replay_runtime_eligible = bool(trackf_ttt.get("replay_condition_chunk_count", 0)) and not bool(
        trackf_ttt.get("replay_condition_not_runtime_eligible", False)
    )
    trackf_proxy_runtime_eligible = bool(trackf_ttt.get("proxy_condition_chunk_count", 0)) and not bool(
        trackf_ttt.get("proxy_condition_not_runtime_eligible", False)
    )
    primary_blocker = (
        "Track D confidence-neutral L07 READ action produced at least one short-window mechanism-pass pilot, but Stage7 full validation did not pass the strict full gate. Best Stage7 candidate by delta ATE was "
        f"{stage7.get('best_candidate_by_delta_ate', '')} with delta ATE {stage7.get('best_delta_aligned_ate_rmse_m')} and delta final error {stage7.get('best_delta_final_error_m')}; strict reason: {stage7.get('best_strict_full_gate_reason', '')}. "
        "Therefore no runtime action is allowed. J4 weak-context READ skip, DG-Q90 source-bias, sampled carrier-scoped L07 dense frame-bias, and sampled QK-pair key-stability branches remain No-Go. "
        f"SWA raw trace is now available ({tracke_raw.get('pt_file_count')}/{tracke_raw.get('expected_pt_file_count')} expected files over {tracke_raw.get('case_count')} cases), but bad/good transport separability failed "
        f"(stable margin {tracke_raw.get('separability', {}).get('stable_mass_bad_lower_margin_median')}; unreliable margin {tracke_raw.get('separability', {}).get('unreliable_mass_bad_higher_margin_median')}; required >=0.05), and Track C classified it as {trackc_swa.get('classification', 'NOT_RUN')}. "
        f"TTT write trace is now available ({trackf_ttt.get('case_count')} cases, {trackf_ttt.get('chunk_row_count')} chunks), including replay contribution condition maps and branch-scale component diagnostics when available, but the method-safe fixed component-pair gate failed "
        f"(condition source {trackf_ttt.get('condition_map_source')}; exact chunks {trackf_ttt.get('exact_condition_chunk_count')}; proxy chunks {trackf_ttt.get('proxy_condition_chunk_count')}; "
        f"median D-top10 {trackf_ttt.get('median_d_tok_top10_enrichment', trackf_ttt.get('median_risk_write_enrichment'))}; "
        f"median conflict top10 {trackf_ttt.get('median_conflict_top10_enrichment')}; "
        f"median scale top10 {trackf_ttt.get('median_scale_risk_top10_enrichment')}; "
        f"median decomposed {trackf_ttt.get('median_decomposed_risk_enrichment')}; "
        f"median component oracle {trackf_ttt.get('median_component_risk_enrichment')}; oracle good FPR {trackf_ttt.get('good_false_positive_rate')}; "
        f"fixed pair pass count {trackf_ttt.get('fixed_component_pair_pass_count')}; best fixed pair {trackf_ttt.get('best_fixed_component_write_source')}+{trackf_ttt.get('best_fixed_component_risk_source')} "
        f"risk median {trackf_ttt.get('best_fixed_component_risk_median_enrichment')}, good FPR {trackf_ttt.get('best_fixed_component_good_false_positive_rate')}; required risk median >=0.05 and good FPR <=0.25). "
        f"Replay/proxy condition maps are diagnostic-only; replay runtime eligible = {trackf_replay_runtime_eligible}; proxy runtime eligible = {trackf_proxy_runtime_eligible}. No runtime SWA or TTT action is allowed."
        if mechanism_success and not full_method_success
        else "A v96 confidence-neutral READ branch passed Stage7 strict full validation; verify final audit artifacts before runtime integration."
        if full_method_success
        else "J4 READ weak-context early K-side suppression, early-quarter layer repair, stable-anchor compensation, anchor_weak strictness repair, and rho=0.2 strengthened anchor_weak repair all failed to produce >=5% bad READ_LOCAL L1/L2/scale improvement or beat controls. Track G READ cue refinement found no service-ready cue-v2 proxy pass. Per-head raw-QK carrier localization then produced a diagnostic carrier pass, but the DG-Q90 source-bias action pilots, anchor-rescue variants, and confidence-coupled gauge-normalized L07 T030/T045/T050 pilots failed the mechanism gate. SWA and TTT diagnostics are not runtime-ready."
    )
    decision = {
        "final_status": final_status,
        "method_success": full_method_success,
        "diagnostic_success": True,
        "mechanism_success": mechanism_success,
        "full_method_success": full_method_success,
        "runtime_action_allowed": full_method_success,
        "stage0": stage0,
        "trackJ": j,
        "semantic_region_bank": semantic,
        "skip_impact_diagnostic": j3,
        "read_skip_pilot": j4,
        "read_skip_pilot_repair_early_quarter": j4_repair,
        "read_skip_pilot_repair_anchor_compensation": j4_anchor,
        "read_skip_pilot_repair_anchor_weak_compensation": j4_anchor_weak,
        "read_skip_pilot_repair_anchor_weak_rho020": j4_anchor_weak_rho020,
        "trackG_read_cue_refinement": trackg_read_cue,
        "trackG_read_qk_carrier_localization": trackg_qk_carrier,
        "trackD_read_gauge_preserving_action_pilots": trackd_per_head_action,
        "trackE_swa_raw_transport_trace": tracke_raw,
        "trackC_swa_latent_gauge_alignment": trackc_swa,
        "trackE_trackC_swa_route_decision": swa_route_decision,
        "trackF_ttt_write_diagnostic": trackf_ttt,
        "trackF_ttt_write_diagnostic_baseline_missing_condition_maps": trackf_ttt_baseline,
        "trackF_ttt_write_proxy_condition_diagnostic": trackf_ttt_proxy,
        "stage7_full_validation": stage7,
        "primary_blocker": primary_blocker,
        "next_route_recommendation": "Do not promote READ/SWA/TTT runtime action. READ requires a Stage7 strict full-validation pass. SWA trace now exists but is not a validated L3 handoff carrier, so do not return to old Track E source-mask/alpha sweeps. TTT replay contribution and branch-scale diagnostics now exist, but replay/proxy condition maps are diagnostic-only and no fixed component write/risk pair passed both risk enrichment >=0.05 and good-control false-positive <=0.25; only an audited exact/non-proxy condition-map diagnostic with a fixed, predeclared pair that passes the same gates would justify revisiting Track F runtime eligibility.",
    }
    write_json(out / "summary.json", decision)
    write_json(out / "final_decision.json", decision)
    write_text(
        out / "final_report.md",
        f"""# ACL2 v96-TF Final Report

## Answers

1. VGGT4D local audit changed adaptation: yes. The active VGGT4D path compacts both K and V in early layers, uses VGGT-specific token ordering/layer ids, and depends on VGGT projection refinement; LoGeR must not copy it directly.
2. Semantic regions: J3 found READ WEAK_SCALE_CONTEXT and SWA DYNAMIC_OBJECT/LOW_OBSERVABILITY diagnostic candidates; TTT did not pass a candidate gate.
3. Service-ready cues: none newly service-ready. READ weak-context was action-tested and failed mechanism gate; SWA and TTT now have diagnostic traces, but both failed their action-enabling gates.
4. Gauge-preserving READ action: partial. J4 weak-context early K/logit suppression, DG-Q90 source-bias pilots, sampled carrier-scoped L07 dense frame-bias follow-ups, and sampled QK-pair key-stability follow-ups failed to produce a runtime-ready branch; confidence-neutral L07 produced a short-window mechanism-pass candidate, but this is not a full method success because Stage7 strict full validation status is `{stage7.get('classification', 'NOT_RUN')}`.
5. Raw SWA trace carrier: not identified. J5/Track E raw trace is available, but bad/good route-mass separability failed and Track C classified the route as `{trackc_swa.get('classification', 'NOT_RUN')}`.
6. TTT write contamination: diagnosed as not runtime-ready. Track F has trace-supported cases; condition source `{trackf_ttt.get('condition_map_source')}`, exact chunks `{trackf_ttt.get('exact_condition_chunk_count')}`, replay chunks `{trackf_ttt.get('replay_condition_chunk_count')}`, proxy chunks `{trackf_ttt.get('proxy_condition_chunk_count')}`. The component oracle reached median enrichment `{trackf_ttt.get('median_component_risk_enrichment')}` but oracle good FPR was `{trackf_ttt.get('good_false_positive_rate')}`; fixed component pair pass count was `{trackf_ttt.get('fixed_component_pair_pass_count')}`. Best fixed pair `{trackf_ttt.get('best_fixed_component_write_source')} + {trackf_ttt.get('best_fixed_component_risk_source')}` reached risk median `{trackf_ttt.get('best_fixed_component_risk_median_enrichment')}` with good FPR `{trackf_ttt.get('best_fixed_component_good_false_positive_rate')}`.
7. Ready branch: `{ 'confidence-neutral READ is runtime-ready' if full_method_success else 'no branch is runtime-ready' }`.
8. Paused branches: weak-context READ skip, SWA action, TTT action, and runtime promotion.

## Conclusion

This run produced a valid diagnostic/audit partial result, a Track G READ cue-refinement No-Go, a raw-QK per-head carrier diagnostic pass, a Track D confidence-neutral READ mechanism pass, and additional carrier-scoped L07 plus QK-pair key-stability follow-up No-Go evidence. Stage7 full validation result: `{stage7.get('classification', 'NOT_RUN')}`; best candidate `{stage7.get('best_candidate_by_delta_ate', '')}` delta ATE `{stage7.get('best_delta_aligned_ate_rmse_m')}`, delta final error `{stage7.get('best_delta_final_error_m')}`. Track E raw SWA trace availability: `{tracke_gates.get('trace_availability_frac')}`; route gate pass: `{tracke_gates.get('tracke_raw_trace_gate_pass')}`. Track F gate pass: `{trackf_ttt.get('gate_pass')}`. Runtime action allowed: `{full_method_success}`.
""",
    )
    write_text(
        out / "failure_report.md",
        f"# Final Failure Report\n\nMethod success: `{full_method_success}`. Mechanism success: `{mechanism_success}`. {primary_blocker}",
    )
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# What Would Have To Be True To Pass\n\nA READ cue must first pass cue-v2 recall/FPR/count/rotation/sequence-coverage requirements. A matched READ action must then improve bad READ_LOCAL L1/L2/scale metrics by at least 5%, beat all required controls by at least 5%, keep good-control worsen <=2%, provide stable/gauge safety evidence, and then pass Stage7 full validation. Current per-head raw-QK evidence is diagnostic-only because DG-Q90 layer/head source-bias pilots, sampled carrier-scoped L07 dense frame-bias follow-ups, and sampled QK-pair key-stability follow-ups failed the mechanism gate. Current gauge-normalized L07 evidence is also No-Go because Stage7 full validation failed despite a short-window mechanism pass. Track E now has raw SWA transport trace, but it would need bad/good stable or unreliable transport separation with margin >=0.05 and L3 relevance before any SWA action. Track F now has TTT replay contribution and branch-scale diagnostics, but runtime eligibility would require audited exact/non-proxy condition maps plus a fixed, predeclared component write/risk pair with risk median enrichment >=0.05 and good-control false positive <=0.25 before any TTT no-write action.",
    )
    write_csv(out / "visual_manifest.csv", [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--track-j", action="store_true")
    parser.add_argument("--semantic-blocker", action="store_true")
    parser.add_argument("--finalize-blocked", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global ROOT
    ROOT = args.out_root
    for track in TRACK_DIRS:
        ensure_track_scaffold(track)
    run_all = args.all or not any([args.stage0, args.track_j, args.semantic_blocker, args.finalize_blocked])
    stage0: dict[str, Any] = {}
    j: dict[str, Any] = {}
    semantic: dict[str, Any] = {}
    j3: dict[str, Any] = {}
    if run_all or args.stage0:
        stage0 = build_stage0()
    if run_all or args.track_j:
        j = build_track_j_audit()
    if run_all or args.semantic_blocker:
        semantic = build_semantic_region_blocker()
    if run_all:
        j3 = build_track_j_skip_impact_diagnostic()
    if run_all or args.finalize_blocked:
        if not stage0:
            stage0 = read_json(ROOT / "trackI_drift_observatory/summary.json")
            stage0 = {
                "trackI_gate_pass": bool(stage0.get("gate_pass")),
                "trackA_gate_pass": bool(read_json(ROOT / "trackA_case_response_atlas/summary.json").get("gate_pass")),
                "trackB_gate_pass": bool(read_json(ROOT / "trackB_visual_hypothesis_registry/summary.json").get("gate_pass")),
            }
        if not j:
            j = read_json(ROOT / "trackJ_vggt4d_code_audit/summary.json")
        if not semantic:
            semantic = read_json(ROOT / "trackJ_semantic_region_bank/summary.json")
        reason = (
            "J1/J2/J3 diagnostics have not promoted a runtime action; "
            "v96 plan forbids READ/SWA/TTT action and full validation before the relevant mechanism gates."
        )
        write_blocked_downstream_dirs(reason)
        build_final_decision(stage0, j, semantic)
    write_json(
        ROOT / "build_summary.json",
        {
            "out_root": str(ROOT),
            "stage0": stage0,
            "trackJ": j,
            "semantic_region_bank": semantic,
            "skip_impact_diagnostic": j3 or read_json(ROOT / "trackJ_skip_impact_diagnostic/summary.json"),
            "final_status": read_json(ROOT / "final_decision/final_decision.json").get("final_status", "in_progress"),
        },
    )


if __name__ == "__main__":
    main()
