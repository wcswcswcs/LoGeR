#!/usr/bin/env python3
"""Build live preflight artifacts for ACL2 v119 carrier-aware augmented plan.

The outputs are intentionally not final completion claims. They record current
evidence, missing dependencies, and the next plan-prescribed repair direction.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
OLD = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"

PLAN = ROOT / "docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_CarrierAwareAugmented_ExperimentPlan.md"
LBLOGICAL_TR_SUMMARY_CANDIDATES = [
    OUT / "stage2_lblogical_tr_semantic_repair/lblogical_tr_semantic_repair_summary_seq00.json",
    OUT / "stage2_lblogical_tr_pilot/lblogical_tr_pilot_summary_seq00.json",
]
LBLOGICAL_TR_COMPONENT_ABLATION_SUMMARY = OUT / "lblogical_tr_component_ablation_summary_seq00.json"
CLBP_MINIMECH_SUMMARY = OUT / "stage3_clbp_minimech/clbp_minimech_summary_seq00.json"
CLBL_MINIMECH_SUMMARY = OUT / "stage3_clbl_minimech/clbl_minimech_summary_seq00.json"
CLBA_MINIMECH_SUMMARY = OUT / "stage3_clba_minimech/clba_minimech_summary_seq00.json"
CLBM_MINIMECH_SUMMARY = OUT / "stage3_clbm_minimech/clbm_minimech_summary_seq00.json"
HS_DHS_LIVENESS_SMOKE_SUMMARY_CANDIDATES = [
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max128_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max96_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max64_global_mrt_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max64_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max48_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max32_global_mrt_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max32_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max24_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max16_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_max12_summary.json",
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_smoke_summary.json",
]
HS_DHS_CROSSSEQ_SUMMARY = (
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_liveness_crossseq_max32_global_mrt_summary.json"
)
HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY = (
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_lq5_strong_controls_max32_global_mrt_summary.json"
)
HS_DHS_LA4_STRONG_CONTROLS_SUMMARY = (
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_la4_strong_controls_max32_global_mrt_summary.json"
)
HS_CHS_EXPLICIT_LANE_SUMMARY = (
    OUT / "stage4_hs_dhs_liveness_smoke/dhs_chs_explicit_lane_smoke_summary.json"
)
HS_CHS_CARRIER_EVIDENCE_SUMMARY = OUT / "V119_CHS_CARRIER_EVIDENCE_ROWS_SUMMARY.json"
LBNORM_REAL_AR_SUMMARY = (
    OUT / "stage1_lbnorm_real_ar/summary/stage4_v119_lbnorm_ar_lingbot_ar_anchor_read_summary.json"
)
LBLR_LOGIT_SUMMARY = (
    OUT / "stage1_lblr_local_read_logit/summary/stage4_v119_lblr_logit_lingbot_ar_anchor_read_summary.json"
)
LBLR_VALUE_SUMMARY = (
    OUT / "stage1_lblr_local_read_value/summary/stage4_v119_lblr_value_lingbot_ar_anchor_read_summary.json"
)


ORIGINAL_BRANCHES = [
    ("LB-AI-FIX", "LingBot", "Anchor initialization", "old_v119", "LB-SCHED + SEM-V3"),
    ("LB-AR-FIX", "LingBot", "Anchor read", "old_v119", "LB-NORM + SEM-V3 + exact source/query span"),
    ("LB-LR", "LingBot", "Local read", "old_v119", "LB-NORM + SEM-V3 + exact source/query span"),
    ("LB-TA", "LingBot", "Trajectory admission", "old_v119", "LB-LOGICAL + SEM-V3"),
    ("LB-TR", "LingBot", "Trajectory retrieval", "old_v119", "LB-LOGICAL + SEM-V3"),
    ("LB-TE", "LingBot", "Retention / eviction", "old_v119", "LB-LOGICAL + SEM-V3"),
    ("LB-CT", "LingBot", "Compact context-token routing", "old_v119", "LB-LOGICAL + SEM-V3"),
    ("HS-PW", "HorizonStream", "Pre-write dual lane", "old_v119", "HS-KDA selected-layer write instrumentation"),
    ("HS-GR", "HorizonStream", "Direct retention / decay", "old_v119", "direct gamma/decay hook or 3 repairs"),
    ("HS-RR", "HorizonStream", "Readout routing", "old_v119", "persistent/transient lane trace"),
]

CARRIER_BRANCHES = [
    ("D-LB-A", "LingBot", "Anchor latent carrier", "track_d_latent", "5 granularities, >=6 controls"),
    ("D-LB-L", "LingBot", "Local latent carrier", "track_d_latent", "query/head/source, >=6 controls"),
    ("D-LB-T", "LingBot", "Trajectory latent carrier", "track_d_latent", "entry/token/page, >=6 controls"),
    ("D-LB-M", "LingBot", "Metric latent carrier", "track_d_latent", "anchor/compact/scale, >=5 controls"),
    ("C-LB-A", "LingBot", "Anchor Landmark Carrier", "track_c_explicit", "3 forms, >=6 controls"),
    ("C-LB-L", "LingBot", "Local role lanes", "track_c_explicit", "3 forms, >=6 controls"),
    ("C-LB-P", "LingBot", "Persistent trajectory carrier", "track_c_explicit", "4 forms, >=7 controls"),
    ("C-LB-M", "LingBot", "Metric/Gauge carrier", "track_c_explicit", "3 forms, >=5 controls"),
    ("D-HS-L", "HorizonStream", "Local head latent carrier", "track_d_latent", "head/layer/query, >=6 controls"),
    ("D-HS-G", "HorizonStream", "GLA lifetime latent carrier", "track_d_latent", "band/channel/low-rank, >=6 controls"),
    ("D-HS-M", "HorizonStream", "MRT metric latent carrier", "track_d_latent", "state direction/readout, >=5 controls"),
    ("C-HS-2L", "HorizonStream", "Two-lane explicit state", "track_c_explicit", "3 implementations, >=7 controls"),
    ("C-HS-3L", "HorizonStream", "Three-lane explicit state", "track_c_explicit", "3 implementations, >=7 controls"),
    ("C-HS-S", "HorizonStream", "Low-rank shadow carriers", "track_c_explicit", "ranks 8/16/32, >=7 controls"),
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def lblogical_tr_summary_path() -> Path:
    for path in LBLOGICAL_TR_SUMMARY_CANDIDATES:
        if path.is_file():
            return path
    return LBLOGICAL_TR_SUMMARY_CANDIDATES[-1]


def hs_dhs_liveness_smoke_summary_path() -> Path:
    first_existing: Path | None = None
    for path in HS_DHS_LIVENESS_SMOKE_SUMMARY_CANDIDATES:
        if path.is_file():
            if first_existing is None:
                first_existing = path
            payload = read_json(path)
            if payload.get("all_jobs_liveness_pass"):
                return path
    if first_existing is not None:
        return first_existing
    return HS_DHS_LIVENESS_SMOKE_SUMMARY_CANDIDATES[-1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def code_contains(path: str, needle: str) -> bool:
    return needle in read_text(ROOT / path)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def phase_s0_rows(generated_at: str) -> list[dict[str, Any]]:
    stage0 = read_json(OLD / "stage0/stage0_summary.json")
    semv3 = read_json(OLD / "stage1_semv3_sidecar/semv3_summary.json")
    lbnorm = read_json(OUT / "stage1_lbnorm/lbnorm_value_scaling_audit_summary.json")
    lbnorm_real_ar = read_json(LBNORM_REAL_AR_SUMMARY)
    lblr_logit = read_json(LBLR_LOGIT_SUMMARY)
    lblr_value = read_json(LBLR_VALUE_SUMMARY)
    lblogical = read_json(OUT / "stage1_lblogical/lblogical_gathered_kv_audit_summary.json")
    lblogical_tr_path = lblogical_tr_summary_path()
    lblogical_tr = read_json(lblogical_tr_path)
    lblogical_tr_component = read_json(LBLOGICAL_TR_COMPONENT_ABLATION_SUMMARY)
    hskda = read_json(OUT / "stage1_hskda/hskda_direct_probe_audit_summary.json")

    lbsched_code = (
        code_contains("third_party/lingbot-map/lingbot_map/models/gct_stream.py", "keyframe_schedule_mode")
        and code_contains("third_party/lingbot-map/lingbot_map/models/gct_stream.py", "global_frozen")
        and code_contains("third_party/lingbot-map/benchmark/methods/lingbot_map.py", "frozen_keyframe_indices_by_seq")
    )
    logical_code = (
        code_contains("third_party/lingbot-map/lingbot_map/models/gct_stream.py", "LogicalSpecialEntryTable")
        or code_contains("third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py", "LogicalSpecialEntryTable")
        or code_contains("third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py", "logical_special_entries")
        or code_contains("third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py", "gather_logical_special_kv")
        or code_contains("third_party/lingbot-map/lingbot_map/layers/attention.py", "logical_entry_id")
    )
    hs_kda_import = code_contains("third_party/HorizonStream/horizonstream/runtime/layers/attention.py", "KimiDeltaAttention")
    hs_direct_qkv = (
        code_contains("third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", "q_proj")
        or code_contains("third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", "k_proj")
        or code_contains("third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", "v_proj")
    )
    hs_state_proxy = code_contains("third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", "trace_gla_update")

    lblogical_status = "FORM_A_B_C_SYNTHETIC_AUDIT_PASS_RUNTIME_PENDING"
    lblogical_boundary = "logical per-frame entry table plus Form A subrange mask, Form B gathered KV, and Form C one-frame reference pass synthetic audit; KITTI runtime rows pending"
    lblogical_next = "run LB-TA/TR/TE/CT candidate/control rows with logical entry provenance joined to SEM-V3"
    if lblogical_tr.get("runtime_worker_eval_all_pass"):
        lblogical_status = "RUNTIME_PILOT_COMPLETE_GEOMETRY_GATE_FAIL"
        lblogical_boundary = (
            "seq00 LB-TR runtime rows complete and candidate logical evidence is present, "
            "but candidate worsened ATE vs default/page controls; no terminal carrier pass"
        )
        lblogical_next = (
            "repair blocker by restricting logical special injection or adding semantic/role gating; "
            "rerun candidate plus default/page/random/reverse controls and require geometry gate pass"
        )
    if lblogical_tr_component:
        lblogical_status = "COMPONENT_ABLATION_COMPLETE_NO_SEMANTIC_MARGIN_CLAIM_CLOSED"
        lblogical_boundary = (
            "seq00 LB-TR component ablation completed per plan 34.3; correct semantic candidate "
            "failed to beat default/page, internal-only, provenance-shuffle, role-swap fallback, "
            "reverse, and random controls; semantic claim is closed for this logical retrieval branch"
        )
        lblogical_next = (
            "do not sweep stable/risk coefficients; proceed to Track C explicit carrier construction "
            "or another preregistered non-LB-TR branch with fresh candidate/control evidence"
        )

    lbnorm_status = "PRIMITIVE_REPAIRED_SYNTHETIC_AUDIT_PASS_REAL_RUNTIME_PENDING"
    lbnorm_evidence = rel(OUT / "stage1_lbnorm/lbnorm_value_scaling_audit_summary.json")
    lbnorm_gate = bool(lbnorm.get("audit_pass", False))
    lbnorm_boundary = "synthetic hook audit only; no real candidate/control runtime statistics yet"
    lbnorm_next = "run AR/LR value-scaling candidate and matched controls with arithmetic_mean_1"
    if lbnorm_real_ar:
        lbnorm_decision = str(lbnorm_real_ar.get("stage4_v119_lbnorm_ar_decision", ""))
        real_complete = bool(lbnorm_real_ar.get("complete", False))
        real_fidelity = bool(lbnorm_real_ar.get("action_fidelity", False))
        real_norm = str(lbnorm_real_ar.get("value_weight_normalization", ""))
        if real_complete and real_fidelity and real_norm == "arithmetic_mean_1":
            lbnorm_status = "REAL_AR_RUNTIME_COMPLETE_ARITHMETIC_MEAN_1_NO_GLOBAL_SUCCESS_CLAIM"
            lbnorm_gate = True
            lbnorm_boundary = (
                f"real AR value-scaling runtime rows completed with arithmetic_mean_1; "
                f"decision={lbnorm_decision}; this closes AR normalization/runtime evidence only, "
                "not LB-LR or global carrier success"
            )
            lbnorm_next = "record LB-AR runtime boundary, then run LB-LR value-scaling rows or move to another preregistered branch"
        else:
            lbnorm_status = "REAL_AR_RUNTIME_ATTEMPTED_BLOCKED_OR_FIDELITY_FAIL"
            lbnorm_gate = False
            lbnorm_boundary = (
                f"real AR value-scaling runtime attempted; complete={real_complete}, "
                f"action_fidelity={real_fidelity}, value_weight_normalization={real_norm}, "
                f"decision={lbnorm_decision}"
            )
            lbnorm_next = "repair real AR runtime/action fidelity before treating LB-NORM as more than synthetic"
        lbnorm_evidence = rel(LBNORM_REAL_AR_SUMMARY)

    rows = [
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "Stage0-canonicalization",
            "status": "PASS_FROM_REPRESENTATION_REPAIR_REFERENCE",
            "evidence": rel(OLD / "stage0/stage0_summary.json"),
            "gate_pass": bool_text(stage0.get("stage0_gate_pass", False)),
            "current_plan_boundary": "historical reference imported; not a carrier-aware completion claim",
            "next_required_action": "use as frozen baseline/reference only",
        },
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "LB-SCHED",
            "status": "CODE_PRESENT_HISTORICAL_PARITY_ARTIFACTS_PRESENT",
            "evidence": rel(OLD / "stage1_lbsched_parity"),
            "gate_pass": bool_text(lbsched_code),
            "current_plan_boundary": "global_frozen support present; carrier-aware anchor qualification still pending",
            "next_required_action": "use frozen schedule for any D-LB-A / C-LB-A run and reject confounded rows",
        },
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "LB-NORM",
            "status": lbnorm_status,
            "evidence": lbnorm_evidence,
            "gate_pass": bool_text(lbnorm_gate),
            "current_plan_boundary": lbnorm_boundary,
            "next_required_action": lbnorm_next,
        },
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "SEM-V3",
            "status": "READY_FROM_REPRESENTATION_REPAIR_REFERENCE",
            "evidence": rel(OLD / "stage1_semv3_sidecar/semv3_summary.json"),
            "gate_pass": bool_text(semv3.get("semv3_ready", False) and semv3.get("prefix_leakage_gate", False)),
            "current_plan_boundary": "SEM-V3 sidecar ready for 00/02; must still hash/join runtime rows in new carrier evidence",
            "next_required_action": "join semantic_sidecar_hash into carrier evidence rows and runtime action rows",
        },
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "LB-LOGICAL",
            "status": lblogical_status,
            "evidence": rel(OUT / "stage1_lblogical/lblogical_gathered_kv_audit_summary.json"),
            "gate_pass": bool_text(logical_code and lblogical.get("audit_pass", False)),
            "current_plan_boundary": lblogical_boundary,
            "next_required_action": lblogical_next,
            "selected_physical_page_ids": lblogical.get("selected_physical_page_ids", ""),
            "selected_token_count": lblogical.get("selected_token_count", ""),
            "form_a_backend": lblogical.get("form_a_backend", ""),
            "logical_backend": lblogical.get("logical_backend", ""),
            "one_frame_page_backend": lblogical.get("one_frame_page_backend", ""),
            "form_a_form_b_max_abs_diff": lblogical.get("form_a_form_b_max_abs_diff", ""),
            "form_b_form_c_max_abs_diff": lblogical.get("form_b_form_c_max_abs_diff", ""),
            "runtime_logical_action_trace_row_count": lblogical.get("runtime_logical_action_trace_row_count", ""),
            "runtime_logical_read_trace_row_count": lblogical.get("runtime_logical_read_trace_row_count", ""),
            "runtime_logical_selected_entry_counts": json.dumps(
                lblogical.get("runtime_logical_selected_entry_counts", []), sort_keys=True
            ),
            "runtime_pilot_summary": rel(lblogical_tr_path),
            "component_ablation_summary": rel(LBLOGICAL_TR_COMPONENT_ABLATION_SUMMARY)
            if lblogical_tr_component
            else "",
            "runtime_worker_eval_all_pass": bool_text(lblogical_tr.get("runtime_worker_eval_all_pass", False)),
            "candidate_runtime_logical_evidence_pass": bool_text(
                lblogical_tr.get("candidate_runtime_logical_evidence_pass", False)
            ),
            "carrier_route_terminal_pass": bool_text(lblogical_tr.get("carrier_route_terminal_pass", False)),
            "candidate_ate": lblogical_tr.get("candidate_ate", ""),
            "default_ate": lblogical_tr.get("default_ate", ""),
            "page_control_ate": lblogical_tr.get("page_control_ate", ""),
            "random_control_ate": lblogical_tr.get("random_control_ate", ""),
            "reverse_control_ate": lblogical_tr.get("reverse_control_ate", ""),
            "candidate_ate_improvement_pct_vs_default": lblogical_tr.get(
                "candidate_ate_improvement_pct_vs_default", ""
            ),
            "candidate_ate_improvement_pct_vs_random_control": lblogical_tr.get(
                "candidate_ate_improvement_pct_vs_random_control", ""
            ),
            "component_ablation_terminal_pass": bool_text(
                lblogical_tr_component.get("component_ablation_terminal_pass", False)
            )
            if lblogical_tr_component
            else "",
            "candidate_beats_default": bool_text(
                lblogical_tr_component.get("comparison_variants", {})
                .get("tr0_default_no_policy", {})
                .get("candidate_beats_this", False)
            )
            if lblogical_tr_component
            else "",
            "candidate_beats_internal_ablation": bool_text(
                lblogical_tr_component.get("comparison_variants", {})
                .get("tr8_logical_internal_qk_topk2", {})
                .get("candidate_beats_this", False)
            )
            if lblogical_tr_component
            else "",
            "candidate_beats_provenance_shuffle_control": bool_text(
                lblogical_tr_component.get("comparison_variants", {})
                .get("tr9_logical_provenance_shuffle_qk_topk2", {})
                .get("candidate_beats_this", False)
            )
            if lblogical_tr_component
            else "",
        },
        {
            "schema": "acl2_v119tf_carrier_aware_phase_s0_repair_status_v1",
            "generated_at_utc": generated_at,
            "repair_id": "HS-KDA",
            "status": "REPAIR_1_DIRECT_QKV_DECAY_SYNTHETIC_AUDIT_PASS_RUNTIME_PENDING",
            "evidence": rel(OUT / "stage1_hskda/hskda_direct_probe_audit_summary.json"),
            "gate_pass": bool_text(hs_kda_import and hs_direct_qkv and hskda.get("audit_pass", False)),
            "current_plan_boundary": "direct q/k/v and decay primitive pass; direct gamma unavailable; sequence-level selected-layer carrier rows pending",
            "next_required_action": "Run HorizonStream selected-layer candidate/control rows; close HS-GR if direct gamma remains unavailable and decay-only evidence is insufficient",
            "kda_import_present": bool_text(hs_kda_import),
            "state_delta_proxy_present": bool_text(hs_state_proxy),
            "direct_qkv_or_gamma_trace_present": bool_text(hs_direct_qkv),
            "direct_qkv_statuses": json.dumps(hskda.get("direct_qkv_statuses", []), sort_keys=True),
            "direct_decay_statuses": json.dumps(hskda.get("direct_decay_statuses", []), sort_keys=True),
            "direct_gamma_statuses": json.dumps(hskda.get("direct_gamma_statuses", []), sort_keys=True),
        },
    ]
    return rows


def branch_rows(generated_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lblogical_tr_path = lblogical_tr_summary_path()
    lblogical_tr = read_json(lblogical_tr_path)
    lblogical_tr_component = read_json(LBLOGICAL_TR_COMPONENT_ABLATION_SUMMARY)
    clbp = read_json(CLBP_MINIMECH_SUMMARY)
    clbl = read_json(CLBL_MINIMECH_SUMMARY)
    clba = read_json(CLBA_MINIMECH_SUMMARY)
    clbm = read_json(CLBM_MINIMECH_SUMMARY)
    hs_dhs_path = hs_dhs_liveness_smoke_summary_path()
    hs_dhs = read_json(hs_dhs_path)
    hs_dhs_crossseq = read_json(HS_DHS_CROSSSEQ_SUMMARY)
    hs_lq5_controls = read_json(HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY)
    hs_la4_controls = read_json(HS_DHS_LA4_STRONG_CONTROLS_SUMMARY)
    hs_chs = read_json(HS_CHS_EXPLICIT_LANE_SUMMARY)
    hs_chs_evidence = read_json(HS_CHS_CARRIER_EVIDENCE_SUMMARY)
    lbnorm_real_ar = read_json(LBNORM_REAL_AR_SUMMARY)
    lblr_logit = read_json(LBLR_LOGIT_SUMMARY)
    lblr_value = read_json(LBLR_VALUE_SUMMARY)
    hs_chs_branch = {
        row.get("branch"): row
        for row in hs_chs.get("branches", [])
        if isinstance(row, dict) and row.get("branch")
    }
    hs_dhs_branch_stats = hs_dhs.get("branch_stats", {}) if isinstance(hs_dhs.get("branch_stats", {}), dict) else {}
    hs_dhs_crossseq_branch = (
        hs_dhs_crossseq.get("branch_summary", {}) if isinstance(hs_dhs_crossseq.get("branch_summary", {}), dict) else {}
    )
    for branch, model, target, branch_set, requirement in ORIGINAL_BRANCHES + CARRIER_BRANCHES:
        dependency_blocker = ""
        terminal_status = "PENDING_PHASE_S1_CARRIER_AWARE_EXECUTION"
        evidence = ""
        if branch.startswith("D-LB-T") or branch in {"LB-TA", "LB-TR", "LB-TE", "LB-CT", "C-LB-P", "C-LB-L", "C-LB-A", "C-LB-M"}:
            dependency_blocker = "LB-LOGICAL Form A/B/C primitive pass; runtime carrier evidence pending"
            if lblogical_tr.get("runtime_worker_eval_all_pass") and branch in {"LB-TR", "D-LB-T"}:
                terminal_status = "RUNTIME_PILOT_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
                evidence = rel(lblogical_tr_path)
                dependency_blocker = (
                    "seq00 logical TR candidate has runtime logical evidence but worsens ATE vs default/page; "
                    "requires blocker repair and rerun with matched controls"
                )
            if lblogical_tr_component and branch in {"LB-TR", "D-LB-T"}:
                terminal_status = "COMPONENT_ABLATION_COMPLETE_NO_GO_NO_SEMANTIC_MARGIN"
                evidence = rel(LBLOGICAL_TR_COMPONENT_ABLATION_SUMMARY)
                dependency_blocker = (
                    "plan 34.3 ablations completed; correct semantic candidate has no margin over "
                    "matched controls, so this semantic claim is closed and Track C/non-LB-TR branches remain"
                )
            elif lblogical_tr_component and branch == "C-LB-P":
                dependency_blocker = (
                    "Track C persistent trajectory carrier is now the plan-prescribed explicit-construction "
                    "continuation after LB-TR/D-LB-T semantic-margin No-Go"
                )
            elif lblogical_tr_component and branch == "C-LB-L":
                dependency_blocker = (
                    "Track C local role-lane carrier is a plan-prescribed explicit-construction continuation "
                    "after LB-TR/D-LB-T semantic-margin No-Go"
                )
            elif lblogical_tr_component and branch == "C-LB-A":
                dependency_blocker = (
                    "Track C anchor landmark carrier is a plan-prescribed explicit-construction continuation "
                    "after LB-TR/D-LB-T semantic-margin No-Go"
                )
            elif lblogical_tr_component and branch == "C-LB-M":
                dependency_blocker = (
                    "Track C metric/gauge carrier is a plan-prescribed explicit-construction continuation "
                    "after LB-TR/D-LB-T semantic-margin No-Go"
                )
            if clbp and branch == "C-LB-P":
                terminal_status = "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
                evidence = rel(CLBP_MINIMECH_SUMMARY)
                dependency_blocker = (
                    f"seq00 C-LB-P minimum-mechanism matrix ran {clbp.get('variant_count', '')} variants "
                    f"with {clbp.get('form_count', '')} forms and {clbp.get('control_count', '')} controls; "
                    f"best candidate {clbp.get('best_candidate_variant', '')} ATE={clbp.get('best_candidate_ate', '')} "
                    f"does not beat default/page ATE={clbp.get('default_ate', '')}; full v119 completion remains open"
                )
            if clbl and branch == "C-LB-L":
                terminal_status = "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
                if clbl.get("clbl_minimech_terminal_pass") or clbl.get("explicit_minimech_terminal_pass"):
                    terminal_status = "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                evidence = rel(CLBL_MINIMECH_SUMMARY)
                dependency_blocker = (
                    f"seq00 C-LB-L minimum-mechanism matrix ran {clbl.get('variant_count', '')} variants "
                    f"with {clbl.get('form_count', '')} forms and {clbl.get('control_count', '')} controls; "
                    f"best candidate {clbl.get('best_candidate_variant', '')} ATE={clbl.get('best_candidate_ate', '')}; "
                    "full v119 completion remains open pending cross-sequence and remaining branch gates"
                )
            if clba and branch == "C-LB-A":
                terminal_status = "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
                if clba.get("clba_minimech_terminal_pass") or clba.get("explicit_minimech_terminal_pass"):
                    terminal_status = "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                evidence = rel(CLBA_MINIMECH_SUMMARY)
                dependency_blocker = (
                    f"seq00 C-LB-A minimum-mechanism matrix ran {clba.get('variant_count', '')} variants "
                    f"with {clba.get('form_count', '')} forms and {clba.get('control_count', '')} controls; "
                    f"best candidate {clba.get('best_candidate_variant', '')} ATE={clba.get('best_candidate_ate', '')}; "
                    "full v119 completion remains open pending cross-sequence and remaining branch gates"
                )
            if clbm and branch == "C-LB-M":
                terminal_status = "SEQ00_MINIMECH_COMPLETE_NO_GO_METRIC_SCALE_GATE_MISSING_OR_GEOMETRY_FAIL"
                if clbm.get("clbm_minimech_terminal_pass") or clbm.get("explicit_minimech_terminal_pass"):
                    terminal_status = "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                evidence = rel(CLBM_MINIMECH_SUMMARY)
                dependency_blocker = (
                    f"seq00 C-LB-M minimum-mechanism matrix ran {clbm.get('variant_count', '')} variants "
                    f"with {clbm.get('form_count', '')} forms and {clbm.get('control_count', '')} controls; "
                    f"best candidate {clbm.get('best_candidate_variant', '')} ATE={clbm.get('best_candidate_ate', '')}; "
                    f"metric_scale_support_gate={clbm.get('metric_scale_support_gate', '')}; "
                    "full v119 completion remains open pending scale/gauge evidence, cross-sequence and remaining branch gates"
                )
        elif branch.startswith("D-HS") or branch.startswith("C-HS") or branch.startswith("HS-"):
            dependency_blocker = "HS-KDA direct q/k/v/decay primitive pass; sequence-level carrier branch evidence pending"
            if branch.startswith("C-HS") and branch in hs_chs_branch:
                chs_row = hs_chs_branch[branch]
                terminal_status = str(chs_row.get("current_status", "SMOKE_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_OR_CAPACITY_CONTROL_FAIL"))
                evidence = rel(HS_CHS_EXPLICIT_LANE_SUMMARY)
                dependency_blocker = (
                    f"seq00 max12/global_mrt {branch} explicit-lane smoke ran "
                    f"{chs_row.get('job_count', '')} jobs; best_candidate="
                    f"{chs_row.get('best_candidate_case', '')} ATE={chs_row.get('best_candidate_ate', '')}; "
                    f"best_control={chs_row.get('best_control_case', '')} ATE={chs_row.get('best_control_ate', '')}; "
                    f"candidate_beats_all_controls={bool_text(chs_row.get('candidate_beats_all_controls', False))}. "
                    "This closes only the smoke/control boundary, not full sequence or cross-sequence validation."
                )
            elif hs_dhs and branch.startswith("D-HS"):
                stats = hs_dhs_branch_stats.get(branch, {})
                evidence = rel(hs_dhs_path)
                if stats.get("jobs") and stats.get("liveness_pass_jobs") == stats.get("jobs"):
                    terminal_status = "LIVENESS_SMOKE_COMPLETE_FULL_VALIDATION_PENDING"
                    crossseq_note = ""
                    if hs_dhs_crossseq_branch.get(branch):
                        crossseq_note = (
                            f"; crossseq max32/global_mrt partial rows available for {branch} "
                            f"in {rel(HS_DHS_CROSSSEQ_SUMMARY)}"
                        )
                    if branch == "D-HS-M" and hs_lq5_controls.get("strong_control_failure"):
                        terminal_status = "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_FAIL"
                        crossseq_note = (
                            f"; D-HS-M LQ5 strong controls in {rel(HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY)} "
                            "show semantic-only rowmean-neutral control beats the candidate on seq00/seq02"
                        )
                    if branch == "D-HS-L" and hs_la4_controls.get("any_fail_controls"):
                        terminal_status = "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_CONTROL_ROBUSTNESS_FAIL"
                        crossseq_note = (
                            f"; D-HS-L LA4 strong controls in {rel(HS_DHS_LA4_STRONG_CONTROLS_SUMMARY)} "
                            "show role-rotation dynamic/stable control beats the candidate on seq02 and repeat"
                        )
                    dependency_blocker = (
                        f"seq00 max_frames={hs_dhs.get('max_frames', '')} D-HS liveness smoke produced "
                        f"{stats.get('liveness_pass_jobs', 0)}/{stats.get('jobs', 0)} liveness-pass rows; "
                        "not full-sequence and not a terminal carrier success"
                        f"{crossseq_note}"
                    )
                else:
                    terminal_status = "LIVENESS_SMOKE_ATTEMPTED_BLOCKED_OR_INCOMPLETE"
                    dependency_blocker = (
                        f"seq00 max_frames={hs_dhs.get('max_frames', '')} D-HS liveness smoke produced "
                        f"{stats.get('liveness_pass_jobs', 0)}/{stats.get('jobs', 0)} liveness-pass rows; "
                        "inspect runner logs and repair missing runtime/action/trace evidence before promotion"
                    )
            elif hs_dhs and branch.startswith("HS-"):
                evidence = rel(hs_dhs_path)
                dependency_blocker = (
                    "D-HS selected-layer liveness smoke exists; old HS-PW/HS-GR/HS-RR branch-specific "
                    "full candidate/control rows remain pending"
                )
        elif branch in {"LB-AR-FIX", "LB-LR"}:
            dependency_blocker = "LB-NORM synthetic pass only; real runtime controls pending"
            if lbnorm_real_ar and branch == "LB-AR-FIX":
                evidence = rel(LBNORM_REAL_AR_SUMMARY)
                decision = str(lbnorm_real_ar.get("stage4_v119_lbnorm_ar_decision", ""))
                terminal_status = "REAL_AR_RUNTIME_COMPLETE_NO_GLOBAL_SUCCESS_CLAIM"
                dependency_blocker = (
                    "real AR arithmetic_mean_1 value-scaling candidate/control rows are present; "
                    f"decision={decision}; complete={bool_text(lbnorm_real_ar.get('complete', False))}; "
                    f"action_fidelity={bool_text(lbnorm_real_ar.get('action_fidelity', False))}; "
                    f"candidate_better_all_controls={bool_text(lbnorm_real_ar.get('candidate_better_all_controls', False))}; "
                    f"baseline_gate={bool_text(lbnorm_real_ar.get('baseline_gate', False))}. "
                    "This closes the AR runtime normalization boundary only; LB-LR and global carrier gates remain open."
                )
            elif lbnorm_real_ar and branch == "LB-LR":
                evidence_paths = []
                if lblr_logit:
                    evidence_paths.append(rel(LBLR_LOGIT_SUMMARY))
                if lblr_value:
                    evidence_paths.append(rel(LBLR_VALUE_SUMMARY))
                if evidence_paths:
                    evidence = ";".join(evidence_paths)
                logit_decision = str(lblr_logit.get("stage4_v119_lblr_logit_decision", "")) if lblr_logit else ""
                value_decision = str(lblr_value.get("stage4_v119_lblr_value_decision", "")) if lblr_value else ""
                logit_complete = bool(lblr_logit.get("complete", False)) if lblr_logit else False
                value_complete = bool(lblr_value.get("complete", False)) if lblr_value else False
                logit_fidelity = bool(lblr_logit.get("action_fidelity", False)) if lblr_logit else False
                value_fidelity = bool(lblr_value.get("action_fidelity", False)) if lblr_value else False
                if lblr_logit and lblr_value:
                    terminal_status = "REAL_LR_RUNTIME_COMPLETE_NO_GLOBAL_SUCCESS_CLAIM"
                    dependency_blocker = (
                        "LB-LR logit/value runtime summaries exist; "
                        f"logit_decision={logit_decision}; value_decision={value_decision}; "
                        f"logit_complete={bool_text(logit_complete)}; value_complete={bool_text(value_complete)}; "
                        f"logit_action_fidelity={bool_text(logit_fidelity)}; value_action_fidelity={bool_text(value_fidelity)}. "
                        "This records the local-read runtime boundary only; global carrier gates remain open."
                    )
                elif lblr_logit or lblr_value:
                    terminal_status = "REAL_LR_RUNTIME_PARTIAL_FORM_PENDING"
                    dependency_blocker = (
                        f"LB-LR partial runtime summary exists; logit_present={bool_text(bool(lblr_logit))}; "
                        f"value_present={bool_text(bool(lblr_value))}; both intervention forms are required before branch closure."
                    )
                else:
                    dependency_blocker = (
                        "LB-NORM real AR arithmetic_mean_1 rows exist, but Local Read value-scaling rows remain pending"
                    )
        elif branch in {"D-LB-A", "D-LB-M", "C-LB-A", "C-LB-M", "LB-AI-FIX"}:
            dependency_blocker = "carrier liveness/provenance qualification pending"
        rows.append(
            {
                "schema": "acl2_v119tf_carrier_aware_branch_completion_live_row_v1",
                "generated_at_utc": generated_at,
                "matrix_scope": "live_progress_not_final_completion_matrix",
                "branch": branch,
                "model": model,
                "target": target,
                "branch_set": branch_set,
                "mandatory_requirement": requirement,
                "terminal": False,
                "terminal_status": terminal_status,
                "allowed_terminal_status": "",
                "dependency_blocker_or_next_gate": dependency_blocker,
                "evidence": evidence,
                "missing_required_variant_count": "",
                "unmatched_control_count": "",
                "not_run_count": "",
                "unexplained_pending_count": "",
                "global_goal_achieved": False,
            }
        )
    return rows


def qualification_rows(generated_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clbp = read_json(CLBP_MINIMECH_SUMMARY)
    clbl = read_json(CLBL_MINIMECH_SUMMARY)
    clba = read_json(CLBA_MINIMECH_SUMMARY)
    clbm = read_json(CLBM_MINIMECH_SUMMARY)
    hs_dhs_path = hs_dhs_liveness_smoke_summary_path()
    hs_dhs = read_json(hs_dhs_path)
    hs_dhs_crossseq = read_json(HS_DHS_CROSSSEQ_SUMMARY)
    hs_lq5_controls = read_json(HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY)
    hs_la4_controls = read_json(HS_DHS_LA4_STRONG_CONTROLS_SUMMARY)
    hs_chs = read_json(HS_CHS_EXPLICIT_LANE_SUMMARY)
    hs_chs_evidence = read_json(HS_CHS_CARRIER_EVIDENCE_SUMMARY)
    hs_chs_branch = {
        row.get("branch"): row
        for row in hs_chs.get("branches", [])
        if isinstance(row, dict) and row.get("branch")
    }
    hs_dhs_branch_stats = hs_dhs.get("branch_stats", {}) if isinstance(hs_dhs.get("branch_stats", {}), dict) else {}
    hs_dhs_crossseq_branch = (
        hs_dhs_crossseq.get("branch_summary", {}) if isinstance(hs_dhs_crossseq.get("branch_summary", {}), dict) else {}
    )
    for branch, model, target, branch_set, requirement in CARRIER_BRANCHES:
        geometry_activity_gate = "pending"
        addressability_gate = "pending"
        role_consistency_gate = "pending"
        semantic_specificity_gate = "pending"
        cross_sequence_direction_gate = "pending"
        runtime_method_gate = "pending"
        current_status = "PENDING_PHASE_S1_CARRIER_INSTRUMENTATION_AND_LIVENESS"
        truthfulness_boundary = "no carrier metrics have been measured in this matrix yet"
        if branch == "C-LB-P" and clbp:
            geometry_activity_gate = "fail_seq00_minimech_best_candidate_not_better_than_default"
            addressability_gate = "seq00_runtime_trace_present"
            role_consistency_gate = "seq00_controls_present_role_swap_falls_back_no_logical_reads"
            semantic_specificity_gate = "fail_no_margin_over_default_page_or_role_swap"
            cross_sequence_direction_gate = "pending_not_run"
            runtime_method_gate = "pass_worker_eval_all_rc0" if clbp.get("runtime_worker_eval_all_pass") else "fail"
            current_status = "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
            truthfulness_boundary = str(clbp.get("truthfulness_boundary", "C-LB-P seq00 only; no full completion claim"))
        if branch == "C-LB-L" and clbl:
            geometry_activity_gate = (
                "pass_seq00_minimech_best_candidate_better_than_default"
                if clbl.get("best_candidate_geometry_pass_vs_default_1pct")
                else "fail_seq00_minimech_best_candidate_not_better_than_default"
            )
            addressability_gate = "seq00_runtime_trace_present"
            role_consistency_gate = "seq00_controls_present_role_swap_reverse_random_shuffle"
            semantic_specificity_gate = (
                "pass_best_candidate_beats_primary_controls"
                if clbl.get("best_candidate_beats_all_primary_controls")
                else "fail_no_margin_over_default_page_or_primary_controls"
            )
            cross_sequence_direction_gate = "pending_not_run"
            runtime_method_gate = "pass_worker_eval_artifacts" if clbl.get("runtime_worker_eval_all_pass") else "fail"
            current_status = (
                "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                if (clbl.get("clbl_minimech_terminal_pass") or clbl.get("explicit_minimech_terminal_pass"))
                else "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
            )
            truthfulness_boundary = str(clbl.get("truthfulness_boundary", "C-LB-L seq00 only; no full completion claim"))
        if branch == "C-LB-A" and clba:
            geometry_activity_gate = (
                "pass_seq00_minimech_best_candidate_better_than_default"
                if clba.get("best_candidate_geometry_pass_vs_default_1pct")
                else "fail_seq00_minimech_best_candidate_not_better_than_default"
            )
            addressability_gate = "seq00_runtime_trace_present"
            role_consistency_gate = "seq00_controls_present_role_swap_reverse_random_shuffle"
            semantic_specificity_gate = (
                "pass_best_candidate_beats_primary_controls"
                if clba.get("best_candidate_beats_all_primary_controls")
                else "fail_no_margin_over_default_page_or_primary_controls"
            )
            cross_sequence_direction_gate = "pending_not_run"
            runtime_method_gate = "pass_worker_eval_artifacts" if clba.get("runtime_worker_eval_all_pass") else "fail"
            current_status = (
                "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                if (clba.get("clba_minimech_terminal_pass") or clba.get("explicit_minimech_terminal_pass"))
                else "SEQ00_MINIMECH_COMPLETE_NO_GO_GEOMETRY_GATE_FAIL"
            )
            truthfulness_boundary = str(clba.get("truthfulness_boundary", "C-LB-A seq00 only; no full completion claim"))
        if branch == "C-LB-M" and clbm:
            geometry_activity_gate = (
                "pass_seq00_minimech_best_candidate_better_than_default"
                if clbm.get("best_candidate_geometry_pass_vs_default_1pct")
                else "fail_seq00_minimech_best_candidate_not_better_than_default"
            )
            addressability_gate = "seq00_runtime_trace_present"
            role_consistency_gate = "seq00_controls_present_metric_negative_controls"
            semantic_specificity_gate = (
                "fail_metric_scale_support_not_measured"
                if clbm.get("metric_scale_support_gate")
                else (
                    "pass_best_candidate_beats_primary_controls"
                    if clbm.get("best_candidate_beats_all_primary_controls")
                    else "fail_no_margin_over_default_page_or_primary_controls"
                )
            )
            cross_sequence_direction_gate = "pending_not_run"
            runtime_method_gate = "pass_worker_eval_artifacts" if clbm.get("runtime_worker_eval_all_pass") else "fail"
            current_status = (
                "SEQ00_MINIMECH_PASS_FULL_VALIDATION_PENDING"
                if (clbm.get("clbm_minimech_terminal_pass") or clbm.get("explicit_minimech_terminal_pass"))
                else "SEQ00_MINIMECH_COMPLETE_NO_GO_METRIC_SCALE_GATE_MISSING_OR_GEOMETRY_FAIL"
            )
            truthfulness_boundary = str(clbm.get("truthfulness_boundary", "C-LB-M seq00 only; no full completion claim"))
        if branch.startswith("D-HS") and hs_dhs:
            stats = hs_dhs_branch_stats.get(branch, {})
            jobs = int(stats.get("jobs", 0) or 0)
            pass_jobs = int(stats.get("liveness_pass_jobs", 0) or 0)
            candidate_pass = int(stats.get("candidate_liveness_pass_jobs", 0) or 0)
            control_pass = int(stats.get("control_liveness_pass_jobs", 0) or 0)
            geometry_activity_gate = (
                "smoke_eval_ran_ate_recorded_not_full_sequence_gate" if pass_jobs > 0 else "fail_no_smoke_eval_pass"
            )
            addressability_gate = (
                "pass_smoke_expected_action_audit_rows_present" if candidate_pass > 0 else "fail_candidate_action_audit_missing"
            )
            role_consistency_gate = (
                "partial_matched_control_liveness_present" if control_pass > 0 else "fail_matched_control_liveness_missing"
            )
            semantic_specificity_gate = "pending_small_frame_smoke_not_semantic_specificity_gate"
            cross_sequence_direction_gate = "pending_not_run"
            if branch in hs_dhs_crossseq_branch:
                pairs = hs_dhs_crossseq_branch.get(branch, [])
                if any(
                    item.get("beats_control_both_seqs") and item.get("beats_baseline_both_seqs")
                    for item in pairs
                ):
                    cross_sequence_direction_gate = "partial_seq00_seq02_max32_global_mrt_direction_consistent_candidate_present"
                else:
                    cross_sequence_direction_gate = "fail_or_inconclusive_seq00_seq02_max32_global_mrt_direction"
            if branch == "D-HS-M" and hs_lq5_controls.get("strong_control_failure"):
                semantic_specificity_gate = "fail_lq5_semantic_only_rowmean_neutral_control_beats_candidate_seq00_seq02"
                cross_sequence_direction_gate = "fail_lq5_strong_controls_override_initial_value_random_direction"
                current_status = "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_FAIL"
            if branch == "D-HS-L" and hs_la4_controls.get("any_fail_controls"):
                semantic_specificity_gate = "fail_la4_role_rotation_control_beats_candidate_seq02_repeat"
                cross_sequence_direction_gate = "fail_la4_strong_controls_override_initial_small_positive_direction"
                current_status = "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_CONTROL_ROBUSTNESS_FAIL"
            runtime_method_gate = (
                "pass_smoke_real_pipeline_rc0_trace_rows" if pass_jobs == jobs and jobs > 0 else "fail_or_incomplete_smoke_runtime_trace"
            )
            if current_status not in {
                "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_FAIL",
                "CROSSSEQ_STRONG_CONTROL_COMPLETE_NO_GO_CONTROL_ROBUSTNESS_FAIL",
            }:
                current_status = (
                    "LIVENESS_SMOKE_COMPLETE_FULL_VALIDATION_PENDING"
                    if pass_jobs == jobs and jobs > 0
                    else "LIVENESS_SMOKE_ATTEMPTED_BLOCKED_OR_INCOMPLETE"
                )
            truthfulness_boundary = str(
                hs_dhs.get(
                    "truthfulness_boundary",
                    "D-HS liveness smoke only; no full completion claim",
                )
            )
        if branch.startswith("C-HS") and branch in hs_chs_branch:
            chs_row = hs_chs_branch[branch]
            all_liveness = bool(chs_row.get("all_liveness_pass", False))
            candidate_beats = bool(chs_row.get("candidate_beats_all_controls", False))
            geometry_activity_gate = (
                "smoke_eval_ran_ate_recorded_not_full_sequence_gate" if all_liveness else "fail_no_smoke_eval_pass"
            )
            addressability_gate = (
                "missing_object_level_addressability_not_measured_in_chs_evidence_rows"
                if hs_chs_evidence
                else ("pass_explicit_lane_action_audit_rows_present" if all_liveness else "fail_lane_action_audit_missing")
            )
            role_consistency_gate = (
                "partial_explicit_lane_forms_and_representation_controls_present"
                if all_liveness
                else "fail_explicit_lane_controls_missing"
            )
            semantic_specificity_gate = (
                "pass_smoke_candidate_beats_controls"
                if candidate_beats
                else "fail_best_capacity_or_generic_control_beats_or_matches_candidate"
            )
            cross_sequence_direction_gate = "pending_not_run"
            runtime_method_gate = (
                "pass_smoke_real_pipeline_rc0_trace_rows" if all_liveness else "fail_or_incomplete_smoke_runtime_trace"
            )
            current_status = str(
                chs_row.get("current_status", "SMOKE_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_OR_CAPACITY_CONTROL_FAIL")
            )
            truthfulness_boundary = str(
                chs_row.get(
                    "truthfulness_boundary",
                    "C-HS explicit-lane smoke only; no full completion claim",
                )
            )
        rows.append(
            {
                "schema": "acl2_v119tf_carrier_qualification_live_row_v1",
                "generated_at_utc": generated_at,
                "matrix_scope": "live_progress_not_final_qualification_matrix",
                "branch": branch,
                "model": model,
                "target": target,
                "branch_set": branch_set,
                "mandatory_requirement": requirement,
                "geometry_activity_gate": geometry_activity_gate,
                "addressability_gate": addressability_gate,
                "role_consistency_gate": role_consistency_gate,
                "semantic_specificity_gate": semantic_specificity_gate,
                "cross_sequence_direction_gate": cross_sequence_direction_gate,
                "runtime_method_gate": runtime_method_gate,
                "current_status": current_status,
                "truthfulness_boundary": truthfulness_boundary,
            }
        )
    return rows


def report_text(summary: dict[str, Any], s0_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v119TF Carrier-Aware Augmented Preflight Report",
        "",
        "This is a live preflight artifact, not a final completion claim.",
        "",
        "## Summary",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- global_goal_achieved: {summary['global_goal_achieved']}",
        f"- original_branch_count: {summary['original_branch_count']}",
        f"- carrier_branch_count: {summary['carrier_branch_count']}",
        "",
        "## Phase S0 Status",
        "",
        "| repair | status | gate_pass | next_required_action |",
        "|---|---|---:|---|",
    ]
    for row in s0_rows:
        lines.append(
            f"| {row['repair_id']} | {row['status']} | {row['gate_pass']} | {row['next_required_action']} |"
        )
    lines.extend(
        [
            "",
            "## Truthfulness Boundary",
            "",
            "Rows marked pending are not terminal states. They are included to make the remaining execution surface explicit; they do not satisfy the plan's final completion audit.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    clbp = read_json(CLBP_MINIMECH_SUMMARY)
    clbl = read_json(CLBL_MINIMECH_SUMMARY)
    clba = read_json(CLBA_MINIMECH_SUMMARY)
    clbm = read_json(CLBM_MINIMECH_SUMMARY)
    hs_dhs_path = hs_dhs_liveness_smoke_summary_path()
    hs_dhs = read_json(hs_dhs_path)
    hs_dhs_crossseq = read_json(HS_DHS_CROSSSEQ_SUMMARY)
    hs_lq5_controls = read_json(HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY)
    hs_la4_controls = read_json(HS_DHS_LA4_STRONG_CONTROLS_SUMMARY)
    hs_chs = read_json(HS_CHS_EXPLICIT_LANE_SUMMARY)
    hs_chs_evidence = read_json(HS_CHS_CARRIER_EVIDENCE_SUMMARY)
    lbnorm_real_ar = read_json(LBNORM_REAL_AR_SUMMARY)
    lblr_logit = read_json(LBLR_LOGIT_SUMMARY)
    lblr_value = read_json(LBLR_VALUE_SUMMARY)
    s0_rows = phase_s0_rows(generated_at)
    branches = branch_rows(generated_at)
    qual = qualification_rows(generated_at)
    next_actions = [
        "Close the current LB-TR/D-LB-T semantic logical retrieval claim; do not sweep stable/risk coefficients.",
        "Start Track C explicit carrier construction, beginning with the preregistered LingBot persistent/local/anchor carrier rows.",
        "Run HorizonStream HS-KDA selected-layer candidate/control rows with direct q/k/v/decay probe enabled.",
        "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
        "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
    ]
    if clbp:
        next_actions = [
            "Record C-LB-P seq00 minimum-mechanism matrix as No-Go for geometry gate; do not claim full v119 completion.",
            "Continue Track C explicit construction on a different preregistered LingBot branch such as C-LB-L or C-LB-A, with matched controls.",
            "Run HorizonStream HS-KDA selected-layer candidate/control rows with direct q/k/v/decay probe enabled.",
            "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
            "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
        ]
    if clbl:
        next_actions = [
            "Record C-LB-L seq00 minimum-mechanism matrix from its summary; do not claim full v119 completion.",
            "Continue Track C explicit construction on another preregistered LingBot branch such as C-LB-A or C-LB-M, unless C-LB-L summary requires a concrete bug repair.",
            "Run HorizonStream HS-KDA selected-layer candidate/control rows with direct q/k/v/decay probe enabled.",
            "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
            "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
        ]
    if clba:
        next_actions = [
            "Record C-LB-A seq00 minimum-mechanism matrix from its summary; do not claim full v119 completion.",
            "Continue Track C explicit construction on another preregistered LingBot branch such as C-LB-M, unless C-LB-A summary requires a concrete bug repair.",
            "Run HorizonStream HS-KDA selected-layer candidate/control rows with direct q/k/v/decay probe enabled.",
            "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
            "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
        ]
    if clbm:
        next_actions = [
            "Record C-LB-M seq00 minimum-mechanism matrix from its summary; do not claim full v119 completion without metric/scale evidence.",
            "Continue remaining preregistered Track D-HS / C-HS branches or add the plan-required anchor-calibrated SE3/scale-jump metric before any metric-carrier success claim.",
            "Run HorizonStream HS-KDA selected-layer candidate/control rows with direct q/k/v/decay probe enabled.",
            "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
            "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
        ]
    if hs_dhs:
        if hs_dhs.get("all_jobs_liveness_pass"):
            next_actions = [
                "Record D-HS seq00 small-frame liveness smoke as partial instrumentation evidence only; do not claim full v119 completion.",
                "Promote D-HS-L/G/M to full-sequence or longer-window selected-layer candidate/control rows with the same action-audit and trace evidence schema.",
                "Continue C-HS explicit lane branches or old HS-PW/HS-GR/HS-RR branch-specific controls if D-HS full rows fail their carrier gates.",
                "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows.",
                "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
            ]
            if hs_dhs_crossseq:
                next_actions = [
                    "Record D-HS seq00/seq02 max32 global_mrt cross-sequence smoke as partial direction evidence only; do not claim full v119 completion.",
                    "Prioritize D-HS-M LQ5 secondary and D-HS-L LA4 for stronger controls or memory-safe longer-window validation; D-HS-G/GQ4 remain inconclusive.",
                    "Do not retry max64 directly without memory repair; current max64 full/global_mrt runs hit OOM.",
                    "Continue C-HS explicit lane branches or old HS-PW/HS-GR/HS-RR controls if D-HS stronger validation fails.",
                    "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows and build Carrier Evidence Row schema.",
                ]
            if hs_lq5_controls:
                next_actions = [
                    "Record D-HS-M LQ5 seq00/seq02 strong-control matrix as No-Go for semantic-specific carrier success; semantic-only rowmean-neutral control beats the candidate.",
                    "Do not promote D-HS-M LQ5 without a new preregistered repair and stronger controls; D-HS-L LA4 remains only a small-effect partial smoke signal.",
                    "Do not retry max64 directly without memory repair; current max64 full/global_mrt runs hit OOM.",
                    "Continue C-HS explicit lane branches or old HS-PW/HS-GR/HS-RR controls, or run real LB-NORM arithmetic_mean_1 value-scaling rows.",
                    "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
                ]
            if hs_la4_controls:
                next_actions = [
                    "Record D-HS-L LA4 seq00/seq02 strong-control matrix as No-Go for strict control robustness; role-rotation dynamic/stable control beats the candidate on seq02 repeat.",
                    "Record D-HS-M LQ5 seq00/seq02 strong-control matrix as No-Go for semantic-specific carrier success; semantic-only rowmean-neutral control beats the candidate.",
                    "Do not promote D-HS-L/D-HS-M without new preregistered repairs and stronger controls.",
                    "Continue C-HS explicit lane branches or old HS-PW/HS-GR/HS-RR controls, or run real LB-NORM arithmetic_mean_1 value-scaling rows.",
                    "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
                ]
            if hs_chs:
                next_actions = [
                    "Record C-HS-2L/3L/S seq00 max12/global_mrt explicit-lane smoke matrices as No-Go for semantic-specific/capacity-control robustness; best controls beat or match the candidates.",
                    "Do not promote C-HS explicit lanes from this smoke boundary; any retry needs a new preregistered mechanism repair, not strength/gain sweeping.",
                    "Continue old HS-PW/HS-GR/HS-RR branch-specific controls or real LB-NORM arithmetic_mean_1 value-scaling rows, because v119 all-branch completion is still false.",
                    "Add Carrier Evidence Row integration for C-HS lane audit rows if pursuing a broader counterfactual matrix.",
                    "Do not retry max64 D-HS directly without memory repair; current max64 full/global_mrt rows hit OOM.",
                ]
            if hs_chs and hs_chs_evidence:
                next_actions = [
                    "Record C-HS Carrier Evidence Rows as audit-only lane evidence; 44 rows were built from exact hs_chs_lane_action rows.",
                    "Do not treat the C-HS evidence rows as object-level semantic addressability: source_track_ids, provenance_entropy, dominant_track_fraction, addressability_score, semantic_sidecar_hash, future_leakage, and scale sensitivity remain unmeasured.",
                    "Do not promote C-HS explicit lanes from this smoke boundary; best capacity/generic controls beat or match the candidates.",
                    "Continue old HS-PW/HS-GR/HS-RR branch-specific controls or real LB-NORM arithmetic_mean_1 value-scaling rows, because v119 all-branch completion is still false.",
                    "Do not retry max64 D-HS directly without memory repair; current max64 full/global_mrt rows hit OOM.",
                ]
        else:
            next_actions = [
                "Inspect D-HS liveness smoke failures and repair the missing runtime, expected action-audit, or trace-row evidence before promotion.",
                "Do not claim HorizonStream carrier success from partial or failed liveness rows.",
                "If D-HS hook/action rows are blocked, follow plan 34.7 by using HS-KDA direct q/k/v/decay evidence to choose a different selected layer or switch to C-HS explicit lanes.",
                "Run real LB-NORM arithmetic_mean_1 value-scaling candidate/control rows after the current HS blocker has an evidence-backed boundary.",
                "Build Carrier Evidence Row schema from exact runtime provenance rather than page/frame aggregates.",
            ]
    if lbnorm_real_ar:
        next_actions = [
            "Record LB-NORM real AR arithmetic_mean_1 runtime rows as complete but No-Go for LB-AR promotion; candidate fails baseline and matched-control gates.",
            "Do not promote LB-AR-FIX from LB-NORM AR: complete/action_fidelity are true, but candidate_better_all_controls=false and baseline_gate=false.",
            "If continuing LB-NORM, run the still-pending LB-LR value-scaling candidate/control rows; otherwise continue old HS-PW/HS-GR/HS-RR branch-specific controls.",
            "Keep C-HS Carrier Evidence Rows as audit-only lane evidence, not object-level semantic addressability proof.",
            "Do not retry max64 D-HS directly without memory repair; current max64 full/global_mrt rows hit OOM.",
        ]
    if lblr_logit or lblr_value:
        next_actions = [
            "Record LB-LR local-window logit/value routing runtime summaries; do not promote unless both forms are complete, action-fidelity true, and controls/baseline gates pass.",
            "If only one LB-LR form exists, finish the missing form before branch closure.",
            "If both LB-LR forms are complete but control/baseline gates fail, keep LB-LR as No-Go and continue old HS-PW/HS-GR/HS-RR branch-specific controls.",
            "Keep C-HS Carrier Evidence Rows as audit-only lane evidence, not object-level semantic addressability proof.",
            "Do not retry max64 D-HS directly without memory repair; current max64 full/global_mrt rows hit OOM.",
        ]
    summary = {
        "schema": "acl2_v119tf_carrier_aware_augmented_preflight_summary_v1",
        "generated_at_utc": generated_at,
        "git_head": git_head(),
        "plan": rel(PLAN),
        "result_root": rel(OUT),
        "old_representation_repair_root": rel(OLD),
        "global_goal_achieved": False,
        "completion_claim": "not_complete_live_preflight_only",
        "original_branch_count": len(ORIGINAL_BRANCHES),
        "carrier_branch_count": len(CARRIER_BRANCHES),
        "phase_s0_rows": len(s0_rows),
        "phase_s0_gate_pass_count": sum(1 for row in s0_rows if row.get("gate_pass") == "true"),
        "phase_s0_missing_or_runtime_pending_count": sum(
            1
            for row in s0_rows
            if row.get("status") not in {
                "PASS_FROM_REPRESENTATION_REPAIR_REFERENCE",
                "CODE_PRESENT_HISTORICAL_PARITY_ARTIFACTS_PRESENT",
                "READY_FROM_REPRESENTATION_REPAIR_REFERENCE",
            }
        ),
        "clbp_minimech_summary": rel(CLBP_MINIMECH_SUMMARY) if clbp else "",
        "clbp_minimech_terminal_pass": bool_text(clbp.get("clbp_minimech_terminal_pass", False)) if clbp else "",
        "clbp_best_candidate_variant": clbp.get("best_candidate_variant", "") if clbp else "",
        "clbp_best_candidate_ate": clbp.get("best_candidate_ate", "") if clbp else "",
        "clbp_default_ate": clbp.get("default_ate", "") if clbp else "",
        "clbl_minimech_summary": rel(CLBL_MINIMECH_SUMMARY) if clbl else "",
        "clbl_minimech_terminal_pass": bool_text(clbl.get("clbl_minimech_terminal_pass", False)) if clbl else "",
        "clbl_best_candidate_variant": clbl.get("best_candidate_variant", "") if clbl else "",
        "clbl_best_candidate_ate": clbl.get("best_candidate_ate", "") if clbl else "",
        "clbl_default_ate": clbl.get("default_ate", "") if clbl else "",
        "clba_minimech_summary": rel(CLBA_MINIMECH_SUMMARY) if clba else "",
        "clba_minimech_terminal_pass": bool_text(clba.get("clba_minimech_terminal_pass", False)) if clba else "",
        "clba_best_candidate_variant": clba.get("best_candidate_variant", "") if clba else "",
        "clba_best_candidate_ate": clba.get("best_candidate_ate", "") if clba else "",
        "clba_default_ate": clba.get("default_ate", "") if clba else "",
        "clbm_minimech_summary": rel(CLBM_MINIMECH_SUMMARY) if clbm else "",
        "clbm_minimech_terminal_pass": bool_text(clbm.get("clbm_minimech_terminal_pass", False)) if clbm else "",
        "clbm_best_candidate_variant": clbm.get("best_candidate_variant", "") if clbm else "",
        "clbm_best_candidate_ate": clbm.get("best_candidate_ate", "") if clbm else "",
        "clbm_default_ate": clbm.get("default_ate", "") if clbm else "",
        "clbm_metric_scale_support_gate": clbm.get("metric_scale_support_gate", "") if clbm else "",
        "hs_dhs_liveness_smoke_summary": rel(hs_dhs_path) if hs_dhs else "",
        "hs_dhs_all_jobs_returncode_zero": bool_text(hs_dhs.get("all_jobs_returncode_zero", False)) if hs_dhs else "",
        "hs_dhs_all_jobs_liveness_pass": bool_text(hs_dhs.get("all_jobs_liveness_pass", False)) if hs_dhs else "",
        "hs_dhs_job_count": hs_dhs.get("job_count", "") if hs_dhs else "",
        "hs_dhs_liveness_fail_count": hs_dhs.get("liveness_fail_count", "") if hs_dhs else "",
        "hs_dhs_truthfulness_boundary": hs_dhs.get("truthfulness_boundary", "") if hs_dhs else "",
        "hs_dhs_crossseq_summary": rel(HS_DHS_CROSSSEQ_SUMMARY) if hs_dhs_crossseq else "",
        "hs_dhs_crossseq_primary_insight": hs_dhs_crossseq.get("primary_insight", "") if hs_dhs_crossseq else "",
        "hs_dhs_lq5_strong_controls_summary": rel(HS_DHS_LQ5_STRONG_CONTROLS_SUMMARY)
        if hs_lq5_controls
        else "",
        "hs_dhs_lq5_strong_control_failure": bool_text(hs_lq5_controls.get("strong_control_failure", False))
        if hs_lq5_controls
        else "",
        "hs_dhs_lq5_primary_blocker": hs_lq5_controls.get("primary_blocker", "") if hs_lq5_controls else "",
        "hs_dhs_la4_strong_controls_summary": rel(HS_DHS_LA4_STRONG_CONTROLS_SUMMARY)
        if hs_la4_controls
        else "",
        "hs_dhs_la4_candidate_beats_all_controls": bool_text(
            hs_la4_controls.get("candidate_beats_all_controls_all_seqs", False)
        )
        if hs_la4_controls
        else "",
        "hs_dhs_la4_primary_blocker": hs_la4_controls.get("primary_blocker", "") if hs_la4_controls else "",
        "hs_chs_explicit_lane_summary": rel(HS_CHS_EXPLICIT_LANE_SUMMARY) if hs_chs else "",
        "hs_chs_all_jobs_liveness_pass": bool_text(hs_chs.get("all_jobs_liveness_pass", False)) if hs_chs else "",
        "hs_chs_branch_statuses": json.dumps(
            {
                row.get("branch"): {
                    "current_status": row.get("current_status"),
                    "best_candidate_case": row.get("best_candidate_case"),
                    "best_candidate_ate": row.get("best_candidate_ate"),
                    "best_control_case": row.get("best_control_case"),
                    "best_control_ate": row.get("best_control_ate"),
                    "candidate_beats_all_controls": row.get("candidate_beats_all_controls"),
                }
                for row in hs_chs.get("branches", [])
                if isinstance(row, dict) and row.get("branch")
            },
            sort_keys=True,
        )
        if hs_chs
        else "",
        "hs_chs_carrier_evidence_summary": rel(HS_CHS_CARRIER_EVIDENCE_SUMMARY) if hs_chs_evidence else "",
        "hs_chs_carrier_evidence_rows": hs_chs_evidence.get("row_count", "") if hs_chs_evidence else "",
        "hs_chs_carrier_evidence_missing_not_inferred_fields": json.dumps(
            hs_chs_evidence.get("missing_not_inferred_fields", []), sort_keys=True
        )
        if hs_chs_evidence
        else "",
        "lbnorm_real_ar_summary": rel(LBNORM_REAL_AR_SUMMARY) if lbnorm_real_ar else "",
        "lbnorm_real_ar_decision": lbnorm_real_ar.get("stage4_v119_lbnorm_ar_decision", "")
        if lbnorm_real_ar
        else "",
        "lbnorm_real_ar_complete": lbnorm_real_ar.get("complete", "") if lbnorm_real_ar else "",
        "lbnorm_real_ar_action_fidelity": lbnorm_real_ar.get("action_fidelity", "") if lbnorm_real_ar else "",
        "lbnorm_real_ar_value_weight_normalization": lbnorm_real_ar.get("value_weight_normalization", "")
        if lbnorm_real_ar
        else "",
        "lblr_logit_summary": rel(LBLR_LOGIT_SUMMARY) if lblr_logit else "",
        "lblr_logit_decision": lblr_logit.get("stage4_v119_lblr_logit_decision", "") if lblr_logit else "",
        "lblr_logit_complete": lblr_logit.get("complete", "") if lblr_logit else "",
        "lblr_logit_action_fidelity": lblr_logit.get("action_fidelity", "") if lblr_logit else "",
        "lblr_value_summary": rel(LBLR_VALUE_SUMMARY) if lblr_value else "",
        "lblr_value_decision": lblr_value.get("stage4_v119_lblr_value_decision", "") if lblr_value else "",
        "lblr_value_complete": lblr_value.get("complete", "") if lblr_value else "",
        "lblr_value_action_fidelity": lblr_value.get("action_fidelity", "") if lblr_value else "",
        "next_actions": next_actions,
        "outputs": {
            "phase_s0_repair_status": rel(OUT / "V119_PHASE_S0_REPAIR_STATUS.csv"),
            "branch_completion_matrix": rel(OUT / "V119_CARRIER_BRANCH_COMPLETION_MATRIX.csv"),
            "carrier_qualification_matrix": rel(OUT / "V119_CARRIER_QUALIFICATION_MATRIX.csv"),
            "summary": rel(OUT / "V119_CARRIER_AWARE_PREFLIGHT_AUDIT.json"),
            "report": rel(OUT / "V119_CARRIER_AWARE_PREFLIGHT_REPORT.md"),
        },
    }
    write_csv(OUT / "V119_PHASE_S0_REPAIR_STATUS.csv", s0_rows)
    write_csv(OUT / "V119_CARRIER_BRANCH_COMPLETION_MATRIX.csv", branches)
    write_csv(OUT / "V119_CARRIER_QUALIFICATION_MATRIX.csv", qual)
    write_json(OUT / "V119_CARRIER_AWARE_PREFLIGHT_AUDIT.json", summary)
    write_text(OUT / "V119_CARRIER_AWARE_PREFLIGHT_REPORT.md", report_text(summary, s0_rows))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
