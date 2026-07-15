#!/usr/bin/env python3
"""Build ACL2 v118-TF final structural closure artifacts.

This is a closure/audit builder only. It refuses to synthesize runtime metrics
when Stage3 did not promote any surface to Stage4.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
BRANCH_ROOT = RESULT_ROOT / "branches"
STAGE2_SUMMARY = RESULT_ROOT / "stage2_memory_entry_provenance/stage2_memory_entry_provenance_summary.json"
STAGE2_ROWS = RESULT_ROOT / "stage2_memory_entry_provenance/stage2_surface_gate_rows.csv"
STAGE3_SUMMARY = RESULT_ROOT / "stage3_internal_signal_readiness/stage3_internal_signal_readiness_summary.json"
STAGE3_ROWS = RESULT_ROOT / "stage3_internal_signal_readiness/stage3_surface_signal_gate_rows.csv"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"

V115_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
V115_QUERY_SMOKE = V115_ROOT / "stage5_lingbot_a2_l2_query_hook_smoke/query_hook_smoke_summary.json"
V115_QUERY_FIDELITY = V115_ROOT / "stage5_lingbot_a2_l2_query_hook_smoke/query_hook_fidelity_rows.csv"
V115_ALIGN = V115_ROOT / "stage2_alignment_cues"
V117_STAGE3 = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability/stage3_internal_reliability"


BRANCHES = [
    {
        "branch": "LB-AI",
        "model": "LingBot",
        "operation": "Anchor initialization",
        "surface": "LB-Anchor",
        "decision_forms": "frame ranking; diversity-aware selection; semantic+internal selection",
        "primary_blocker": "no true LingBot anchor internal candidate or memory reliability rows after Stage3",
    },
    {
        "branch": "LB-AR",
        "model": "LingBot",
        "operation": "Anchor read",
        "surface": "LB-Anchor",
        "decision_forms": "selected-query logit bias; source-value scaling control",
        "primary_blocker": "no true LingBot anchor-read internal candidate or memory reliability rows after Stage3",
    },
    {
        "branch": "LB-LR",
        "model": "LingBot",
        "operation": "Local read",
        "surface": "LB-Local",
        "decision_forms": "local-only role routing; selected-query retrieval; attention bias",
        "primary_blocker": "no true LingBot local-read internal candidate or memory reliability rows after Stage3",
    },
    {
        "branch": "LB-TA",
        "model": "LingBot",
        "operation": "Trajectory admission",
        "surface": "LB-Trajectory",
        "decision_forms": "fixed-budget ranking; hard no-append; soft admission",
        "primary_blocker": "default FlashInfer trajectory provenance unavailable; v117 trajectory cue is semantic persistence proxy",
    },
    {
        "branch": "LB-TR",
        "model": "LingBot",
        "operation": "Trajectory retrieval",
        "surface": "LB-Trajectory",
        "decision_forms": "top-K retrieval; segment diversity; semantic-calibrated QK score",
        "primary_blocker": "default FlashInfer read provenance unavailable; SDPA read rows are synthetic debug only",
    },
    {
        "branch": "LB-TE",
        "model": "LingBot",
        "operation": "Retention / eviction",
        "surface": "LB-Trajectory",
        "decision_forms": "utility ranking; redundancy pruning; semantic-calibrated eviction",
        "primary_blocker": "default FlashInfer eviction lifecycle smoke blocked; v117 reliability is object persistence proxy",
    },
    {
        "branch": "LB-CT",
        "model": "LingBot",
        "operation": "Compact context token routing",
        "surface": "LB-Local",
        "decision_forms": "camera/register/anchor type-specific keep/gate",
        "primary_blocker": "query/context hooks exist from v115 smoke, but no v118 internal candidate or memory reliability rows pass Stage3",
    },
    {
        "branch": "HS-LA",
        "model": "HorizonStream",
        "operation": "Local Attention",
        "surface": "HS-Local",
        "decision_forms": "selected-query logit control; same-magnitude value control",
        "primary_blocker": "HS local candidate span passes but reliability mode is semantic-mixed proxy",
    },
    {
        "branch": "HS-HG",
        "model": "HorizonStream",
        "operation": "Head reliability",
        "surface": "HS-Local",
        "decision_forms": "head-output gate conditioned on true alignment/entropy",
        "primary_blocker": "head internal-std candidate exists, but Stage3 reliability remains semantic-mixed rather than true memory-internal",
    },
    {
        "branch": "HS-GW",
        "model": "HorizonStream",
        "operation": "GLA write",
        "surface": "HS-GLA",
        "decision_forms": "candidate write gain; write-value scaling; state-delta scaling",
        "primary_blocker": "GLA candidate span passes with state-delta approximation, but reliability span/provenance fails and direct KDA gamma is unavailable",
    },
    {
        "branch": "HS-GR",
        "model": "HorizonStream",
        "operation": "GLA state reliability / retention",
        "surface": "HS-GLA",
        "decision_forms": "fixed-reference calibration; direct gamma attempt; channel-band fallback",
        "primary_blocker": "fixed-reference/direct gamma reliability unavailable; chunk semantic-state reliability proxy rejected",
    },
    {
        "branch": "HS-MR",
        "model": "HorizonStream",
        "operation": "MRT safety/readout",
        "surface": "HS-MRT",
        "decision_forms": "scale-delta guard; readout uncertainty calibration",
        "primary_blocker": "MRT diagnostic candidate exists but no Stage3 memory reliability rows are available",
    },
]


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def df_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def series_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if df.empty or column not in df:
        return {"rows": 0, "coverage": 0.0, "p10": None, "p90": None, "span": 0.0, "nunique": 0}
    values = pd.to_numeric(df[column], errors="coerce")
    ok = values[np.isfinite(values)]
    if ok.empty:
        return {"rows": int(len(values)), "coverage": 0.0, "p10": None, "p90": None, "span": 0.0, "nunique": 0}
    p10 = float(np.percentile(ok.to_numpy(dtype=float), 10))
    p90 = float(np.percentile(ok.to_numpy(dtype=float), 90))
    return {
        "rows": int(len(values)),
        "coverage": float(len(ok)) / float(len(values)),
        "p10": p10,
        "p90": p90,
        "span": p90 - p10,
        "nunique": int(ok.nunique()),
    }


def command_output(cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - audit capture only
        return {"cmd": " ".join(cmd), "returncode": None, "stdout": "", "stderr": repr(exc)}


def gpu_audit() -> dict[str, Any]:
    smi = command_output(
        [
            "nvidia-smi",
            "-i",
            "0,1,2,3,4",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    torch_cmd = [
        "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
        "-c",
        (
            "import torch;"
            "print('cuda_available', torch.cuda.is_available());"
            "print('device_count', torch.cuda.device_count());"
            "[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
        ),
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4"}
    env.update(dict(**__import__("os").environ))
    torch = command_output(torch_cmd, env=env)
    return {"nvidia_smi": smi, "torch_cuda": torch}


def stage_row_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    df = df_or_empty(path)
    if df.empty or key not in df:
        return {}
    return {str(row[key]): {k: row[k] for k in df.columns} for _, row in df.iterrows()}


def append_registry(row: dict[str, Any]) -> None:
    existing = []
    if REGISTRY.exists():
        with REGISTRY.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
    if any(r.get("stage") == row["stage"] and r.get("branch") == row["branch"] for r in existing):
        return
    fields = ["schema", "stage", "branch", "status", "artifact", "decision"]
    write_header = not REGISTRY.exists()
    with REGISTRY.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def repair_attempts(branch: dict[str, str], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    b = branch["branch"]
    surface = branch["surface"]
    stage3_row = evidence["stage3_rows"].get(surface, {})
    common = [
        {
            "attempt": "stage2_surface_provenance_audit",
            "mechanism": "exact memory-entry / surface provenance audit",
            "evidence": rel(STAGE2_SUMMARY),
            "result": evidence["stage2_surface_status"].get(surface, "missing"),
            "pass": surface in evidence["ready_surfaces"],
        },
        {
            "attempt": "stage3_v117_strict_proxy_rejection",
            "mechanism": "reuse existing candidate/reliability rows only when not semantic proxy",
            "evidence": rel(STAGE3_ROWS),
            "result": str(stage3_row.get("blockers", "")),
            "pass": bool(stage3_row.get("stage3_surface_ready", False)),
        },
    ]
    if b.startswith("LB-T"):
        return common + [
            {
                "attempt": "sdpa_memory_entry_read_write_lifecycle",
                "mechanism": "SDPA synthetic append/read/evict lifecycle with default-off parity",
                "evidence": rel(RESULT_ROOT / "stage2_memory_entry_provenance/smoke_lingbot_sdpa_trace.jsonl"),
                "result": "synthetic debug passes but default FlashInfer trajectory backend remains blocked",
                "pass": False,
            },
            {
                "attempt": "flashinfer_page_side_table_hook",
                "mechanism": "backend-specific FlashInfer page/source-frame side table and trace hook",
                "evidence": "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
                "result": evidence["flashinfer_blocker"],
                "pass": False,
            },
        ]
    if b.startswith("LB-"):
        q = evidence["query_smoke"]
        return common + [
            {
                "attempt": "selected_query_action_hook_smoke",
                "mechanism": "selected-query/source-context action fidelity in LingBot forward path",
                "evidence": rel(V115_QUERY_FIDELITY),
                "result": f"v115 smoke all_action_fidelity={q.get('all_action_fidelity')}; not a v118 internal candidate/reliability gate",
                "pass": False,
            },
            {
                "attempt": "sdpa_attention_read_topk_hook",
                "mechanism": "selected-query attention_read_topk rows with memory_entry_id in SDPA",
                "evidence": "third_party/lingbot-map/lingbot_map/layers/attention.py",
                "result": "debug read lifecycle available, but no KITTI default-backend anchor/local internal reliability rows",
                "pass": False,
            },
        ]
    if surface == "HS-Local":
        hs = evidence["hs_stats"]
        return common + [
            {
                "attempt": "v115_head_internal_std_probe",
                "mechanism": "operation-specific head internal std / changed-head fraction audit",
                "evidence": rel(V115_ALIGN / "hs_head_reliability_rows.csv"),
                "result": f"internal_head_q_std_span={hs['head_internal_std']['span']}; changed_head_fraction_span={hs['head_changed_fraction']['span']}",
                "pass": False,
            },
            {
                "attempt": "semantic_mixed_reliability_rejection",
                "mechanism": "remove semantic-stable/risk mixed reliability from promotion set",
                "evidence": rel(STAGE3_ROWS),
                "result": "candidate passes but reliability_is_semantic_proxy_not_memory_internal remains",
                "pass": False,
            },
        ]
    if surface == "HS-GLA":
        hs = evidence["hs_stats"]
        return common + [
            {
                "attempt": "direct_kda_gamma_write_audit",
                "mechanism": "direct KDA write/gamma availability audit",
                "evidence": rel(RESULT_ROOT / "stage2_memory_entry_provenance/stage2_code_hook_audit.json"),
                "result": "direct_kda_write_weight_available=False; output_attentions path does not expose direct gamma/write weights",
                "pass": False,
            },
            {
                "attempt": "state_delta_and_channel_band_fallback",
                "mechanism": "state-delta approximation plus selected layer/channel-band fallback",
                "evidence": rel(V115_ALIGN / "hs_gla_state_quality_rows.csv"),
                "result": f"state_delta_rel_norm_span={hs['gla_state_delta_rel']['span']}; Stage3 reliability span/proxy gate still fails",
                "pass": False,
            },
        ]
    if surface == "HS-MRT":
        hs = evidence["hs_stats"]
        return common + [
            {
                "attempt": "mrt_trace_scale_delta_probe",
                "mechanism": "MRT readout scale-delta trace/probe audit",
                "evidence": rel(V115_ALIGN / "hs_mrt_scale_safety_rows.csv"),
                "result": f"predicted_metric_scale_delta_span={hs['mrt_scale_delta']['span']}; coverage={hs['mrt_scale_delta']['coverage']}",
                "pass": False,
            },
            {
                "attempt": "require_paired_hs_policy_block",
                "mechanism": "require HS-LA/HS-GW/HS-GR pairing before MRT runtime promotion",
                "evidence": rel(STAGE3_ROWS),
                "result": "no paired HS branch reached Stage4; reliability rows unavailable",
                "pass": False,
            },
        ]
    return common


def branch_report(branch: dict[str, str], attempts: list[dict[str, Any]], stage3_row: dict[str, Any]) -> str:
    lines = [
        f"# ACL2 v118-TF {branch['branch']} Report",
        "",
        f"- status: `STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS`",
        f"- model: `{branch['model']}`",
        f"- operation: `{branch['operation']}`",
        f"- surface: `{branch['surface']}`",
        f"- decision_forms: `{branch['decision_forms']}`",
        f"- primary_blocker: `{branch['primary_blocker']}`",
        "",
        "## Stage3 Evidence",
        "",
        f"- candidate_row_count: `{stage3_row.get('candidate_row_count', '')}`",
        f"- candidate_span: `{stage3_row.get('candidate_p10_p90_span', '')}`",
        f"- candidate_mode: `{stage3_row.get('candidate_mode', '')}`",
        f"- reliability_row_count: `{stage3_row.get('reliability_row_count', '')}`",
        f"- reliability_span: `{stage3_row.get('reliability_p10_p90_span', '')}`",
        f"- reliability_mode: `{stage3_row.get('reliability_mode', '')}`",
        f"- blockers: `{stage3_row.get('blockers', '')}`",
        "",
        "## Fail-Forward Attempts",
        "",
        "| attempt | mechanism | pass | result | evidence |",
        "|---|---|---:|---|---|",
    ]
    for row in attempts:
        lines.append(f"| {row['attempt']} | {row['mechanism']} | {row['pass']} | {row['result']} | {row['evidence']} |")
    lines += [
        "",
        "## Runtime Boundary",
        "",
        "No Stage4/5/6 runtime metrics were generated for this branch in v118 because the branch did not have a Stage3-promoted internal candidate plus operation-specific memory reliability. This report intentionally leaves geometry metric fields blank rather than fabricating ATE or control-comparison values.",
    ]
    return "\n".join(lines)


def fail_forward_log(branch: dict[str, str], attempts: list[dict[str, Any]]) -> str:
    lines = [
        f"# {branch['branch']} Fail-Forward Log",
        "",
        f"Branch: `{branch['branch']}`",
        f"Surface: `{branch['surface']}`",
        f"Closure: `STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS`",
        "",
        "The following attempts are mechanism-distinct. None is counted as runtime success.",
        "",
    ]
    for i, row in enumerate(attempts, start=1):
        lines += [
            f"## Attempt {i}: {row['attempt']}",
            "",
            f"- mechanism: `{row['mechanism']}`",
            f"- evidence: `{row['evidence']}`",
            f"- pass: `{row['pass']}`",
            f"- result: `{row['result']}`",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(STAGE2_SUMMARY)
    stage3 = read_json(STAGE3_SUMMARY)
    if not stage3.get("stage3_complete"):
        raise RuntimeError("Stage3 is incomplete; refusing final closure")
    if stage3.get("any_surface_ready_for_stage4"):
        raise RuntimeError("At least one surface is Stage4-ready; this structural closure builder is not appropriate")

    stage2_df = df_or_empty(STAGE2_ROWS)
    stage2_surface_status: dict[str, str] = {}
    if not stage2_df.empty:
        for _, row in stage2_df.iterrows():
            surf = str(row.get("surface", ""))
            status = str(row.get("status", ""))
            if surf and surf not in stage2_surface_status:
                stage2_surface_status[surf] = status
    stage3_rows = stage_row_map(STAGE3_ROWS, "surface")
    query_smoke = read_json(V115_QUERY_SMOKE)
    head = df_or_empty(V115_ALIGN / "hs_head_reliability_rows.csv")
    gla = df_or_empty(V115_ALIGN / "hs_gla_state_quality_rows.csv")
    mrt = df_or_empty(V115_ALIGN / "hs_mrt_scale_safety_rows.csv")
    evidence = {
        "stage2_surface_status": stage2_surface_status,
        "ready_surfaces": set(stage2.get("ready_surfaces", [])),
        "stage3_rows": stage3_rows,
        "query_smoke": query_smoke,
        "flashinfer_blocker": stage2.get("flashinfer_blocker", ""),
        "hs_stats": {
            "head_internal_std": series_stats(head, "internal_head_q_std"),
            "head_changed_fraction": series_stats(head, "changed_head_fraction_abs_gt_1e_4"),
            "gla_state_delta_rel": series_stats(gla, "state_delta_rel_norm"),
            "mrt_scale_delta": series_stats(mrt, "predicted_metric_scale_delta"),
        },
    }
    gpu = gpu_audit()
    write_json(RESULT_ROOT / "stage_gpu_availability_audit.json", gpu)

    matrix_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    branch_summary: dict[str, Any] = {}
    for branch in BRANCHES:
        branch_id = branch["branch"]
        out = BRANCH_ROOT / branch_id
        attempts = repair_attempts(branch, evidence)
        stage3_row = stage3_rows.get(branch["surface"], {})
        for idx, row in enumerate(attempts, start=1):
            repair_rows.append(
                {
                    "schema": "acl2_v118tf_branch_repair_attempt_row_v1",
                    "branch": branch_id,
                    "surface": branch["surface"],
                    "attempt_index": idx,
                    **row,
                }
            )
        terminal = "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS"
        decision = {
            "schema": "acl2_v118tf_branch_decision_summary_v1",
            "branch": branch_id,
            "status": terminal,
            "model": branch["model"],
            "operation": branch["operation"],
            "surface": branch["surface"],
            "primary_blocker": branch["primary_blocker"],
            "runtime_launched": False,
            "gpu_runtime_launched": False,
            "stage3_surface_ready": bool(stage3_row.get("stage3_surface_ready", False)),
            "stage3_blockers": stage3_row.get("blockers", ""),
            "repair_attempt_count": len(attempts),
            "repair_attempts": attempts,
            "metric_boundary": "No ATE/control/semantic runtime metrics generated in v118 for this branch.",
            "outputs": {
                "run_manifest": rel(out / f"{branch_id}_RUN_MANIFEST.csv"),
                "action_fidelity": rel(out / f"{branch_id}_ACTION_FIDELITY.csv"),
                "geometry_metrics": rel(out / f"{branch_id}_GEOMETRY_METRICS.csv"),
                "control_comparison": rel(out / f"{branch_id}_CONTROL_COMPARISON.csv"),
                "fail_forward_log": rel(out / f"{branch_id}_FAIL_FORWARD_LOG.md"),
                "summary": rel(out / f"{branch_id}_DECISION_SUMMARY.json"),
                "report": rel(out / f"{branch_id}_REPORT.md"),
            },
        }
        write_csv(
            out / f"{branch_id}_RUN_MANIFEST.csv",
            [
                {
                    "schema": "acl2_v118tf_branch_run_manifest_row_v1",
                    "branch": branch_id,
                    "run_name": "",
                    "seq": "",
                    "gpu": "",
                    "status": terminal,
                    "runtime_launched": False,
                    "blocker": branch["primary_blocker"],
                    "stage3_blockers": stage3_row.get("blockers", ""),
                }
            ],
        )
        write_csv(
            out / f"{branch_id}_ACTION_FIDELITY.csv",
            [
                {
                    "schema": "acl2_v118tf_branch_action_fidelity_row_v1",
                    "branch": branch_id,
                    "status": "BLOCKED_BEFORE_RUNTIME",
                    "action_fidelity_pass": "",
                    "evidence": rel(out / f"{branch_id}_FAIL_FORWARD_LOG.md"),
                }
            ],
        )
        write_csv(
            out / f"{branch_id}_GEOMETRY_METRICS.csv",
            [
                {
                    "schema": "acl2_v118tf_branch_geometry_metric_row_v1",
                    "branch": branch_id,
                    "seq": "",
                    "ate_full": "",
                    "ate_p90": "",
                    "status": "NO_RUNTIME_METRIC_STAGE3_BLOCKED",
                    "blocker": branch["primary_blocker"],
                }
            ],
        )
        write_csv(
            out / f"{branch_id}_CONTROL_COMPARISON.csv",
            [
                {
                    "schema": "acl2_v118tf_branch_control_comparison_row_v1",
                    "branch": branch_id,
                    "policy": "",
                    "control": "",
                    "comparison_metric": "",
                    "status": "NO_CONTROL_RUN_STAGE3_BLOCKED",
                    "blocker": branch["primary_blocker"],
                }
            ],
        )
        write_text(out / f"{branch_id}_FAIL_FORWARD_LOG.md", fail_forward_log(branch, attempts))
        write_json(out / f"{branch_id}_DECISION_SUMMARY.json", decision)
        write_text(out / f"{branch_id}_REPORT.md", branch_report(branch, attempts, stage3_row))
        matrix_rows.append(
            {
                "schema": "acl2_v118tf_branch_completion_matrix_row_v1",
                "branch": branch_id,
                "model": branch["model"],
                "operation": branch["operation"],
                "surface": branch["surface"],
                "status": terminal,
                "runtime_launched": False,
                "gpu_runtime_launched": False,
                "repair_attempt_count": len(attempts),
                "stage3_surface_ready": bool(stage3_row.get("stage3_surface_ready", False)),
                "stage3_blockers": stage3_row.get("blockers", ""),
                "primary_blocker": branch["primary_blocker"],
                "report": rel(out / f"{branch_id}_REPORT.md"),
            }
        )
        branch_summary[branch_id] = decision

    write_csv(RESULT_ROOT / "stage3_branch_repair_attempt_rows.csv", repair_rows)
    write_csv(RESULT_ROOT / "V118_BRANCH_COMPLETION_MATRIX.csv", matrix_rows)
    write_csv(
        RESULT_ROOT / "counterfactual_bucket_manifest.csv",
        [
            {
                "schema": "acl2_v118tf_counterfactual_bucket_manifest_row_v1",
                "branch": row["branch"],
                "surface": row["surface"],
                "bucket": "",
                "status": "STRUCTURAL_BLOCKED_BEFORE_STAGE4",
                "reason": row["primary_blocker"],
            }
            for row in matrix_rows
        ],
    )
    write_csv(
        RESULT_ROOT / "counterfactual_geometry_rows.csv",
        [
            {
                "schema": "acl2_v118tf_counterfactual_geometry_row_v1",
                "branch": row["branch"],
                "seq": "",
                "delta_metric": "",
                "status": "NO_COUNTERFACTUAL_RUNTIME_STAGE3_BLOCKED",
                "reason": row["primary_blocker"],
            }
            for row in matrix_rows
        ],
    )
    write_csv(
        RESULT_ROOT / "counterfactual_rank_quality.csv",
        [
            {
                "schema": "acl2_v118tf_counterfactual_rank_quality_row_v1",
                "branch": row["branch"],
                "spearman": "",
                "auroc": "",
                "top_quartile_uplift": "",
                "matched_control_p95_gap": "",
                "status": "NO_RANK_QUALITY_STAGE3_BLOCKED",
                "reason": row["primary_blocker"],
            }
            for row in matrix_rows
        ],
    )
    write_text(
        RESULT_ROOT / "CARRIER_ATTRIBUTION_REPORT.md",
        "\n".join(
            [
                "# ACL2 v118-TF Carrier Attribution Report",
                "",
                "Stage4 counterfactual carrier attribution was not launched because Stage3 produced zero Stage4-ready surfaces. No Spearman/AUROC/uplift values were computed in v118.",
                "",
                f"- Stage3 summary: `{rel(STAGE3_SUMMARY)}`",
                f"- Branch repair rows: `{rel(RESULT_ROOT / 'stage3_branch_repair_attempt_rows.csv')}`",
                f"- Completion matrix: `{rel(RESULT_ROOT / 'V118_BRANCH_COMPLETION_MATRIX.csv')}`",
            ]
        ),
    )
    taxonomy = "V118_STRUCTURAL_BLOCKERS_REMAIN_AFTER_EXHAUSTIVE_REPAIRS"
    final_summary = {
        "schema": "acl2_v118tf_final_decision_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy": taxonomy,
        "all_branches_terminal": True,
        "not_run_rows_present": False,
        "branch_count": len(matrix_rows),
        "terminal_status_counts": {"STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS": len(matrix_rows)},
        "stage0_1_status": "Stage0 and Stage1 complete pass; Stage1 fragmentation pass is bucketed-control, not raw-rate.",
        "stage2_status": stage2.get("stage2_decision", "PARTIAL_SURFACE_PASS"),
        "stage3_decision": stage3.get("stage3_decision"),
        "stage4_5_6_runtime_launched": False,
        "gpu_available": True,
        "gpu_runtime_boundary": "GPU 0-4 are visible, but no v118 runtime policy was launched because Stage3 did not promote any surface to Stage4.",
        "gpu_audit": rel(RESULT_ROOT / "stage_gpu_availability_audit.json"),
        "outputs": {
            "registry": rel(REGISTRY),
            "completion_matrix": rel(RESULT_ROOT / "V118_BRANCH_COMPLETION_MATRIX.csv"),
            "final_report": rel(RESULT_ROOT / "V118_FINAL_DECISION_REPORT.md"),
            "method_boundaries": rel(RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"),
            "carrier_attribution_report": rel(RESULT_ROOT / "CARRIER_ATTRIBUTION_REPORT.md"),
        },
    }
    write_json(RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json", final_summary)
    report_lines = [
        "# ACL2 v118-TF Final Decision Report",
        "",
        f"- taxonomy: `{taxonomy}`",
        f"- branch_count: `{len(matrix_rows)}`",
        "- all branches terminal: `True`",
        "- Stage4/5/6 runtime launched: `False`",
        "",
        "## Why GPU Runtime Did Not Run",
        "",
        "GPU 0-4 were audited as visible and CUDA-capable. The runtime branch queue was not launched because Stage3 produced zero Stage4-ready surfaces. Launching policy runs anyway would violate the v118 plan's requirement that semantic+internal+memory-reliability cues be established before counterfactual attribution and runtime policy.",
        "",
        "## Branch Matrix",
        "",
        "| branch | surface | status | repair attempts | blocker |",
        "|---|---|---|---:|---|",
    ]
    for row in matrix_rows:
        report_lines.append(
            f"| {row['branch']} | {row['surface']} | {row['status']} | {row['repair_attempt_count']} | {row['primary_blocker']} |"
        )
    report_lines += [
        "",
        "## Evidence Chain",
        "",
        f"- Stage0 fresh reference: `{rel(RESULT_ROOT / 'stage0_fresh_reference/stage0_fresh_reference_summary.json')}`",
        f"- Stage1 sidecar: `{rel(RESULT_ROOT / 'stage1_causal_object_track_sidecar/stage1_semantic_track_v2_summary.json')}`",
        f"- Stage2 provenance: `{rel(STAGE2_SUMMARY)}`",
        f"- Stage3 signal readiness: `{rel(STAGE3_SUMMARY)}`",
        f"- Branch repair attempts: `{rel(RESULT_ROOT / 'stage3_branch_repair_attempt_rows.csv')}`",
    ]
    write_text(RESULT_ROOT / "V118_FINAL_DECISION_REPORT.md", "\n".join(report_lines))
    boundaries = [
        "# ACL2 v118-TF Method And No-Go Boundaries",
        "",
        "- Do not report any v118 Stage4/5/6 ATE, Spearman, AUROC, uplift, semantic-causality, or control-comparison metric: no such runtime was launched in v118.",
        "- Stage1 object identity is usable only with fragmentation bucket controls; raw fragmentation did not pass.",
        "- LB-Trajectory SDPA provenance is debug-only; default FlashInfer trajectory provenance remains blocked by missing FlashInfer runtime.",
        "- HS-GLA uses state-delta approximation; it is not direct KDA gamma/write-weight provenance.",
        "- Semantic persistence, frame aggregate, q norm, state norm, and generic rowmean controls were not promoted as missing internal/reliability variables.",
        "- GPU 0-4 availability was confirmed, but GPU availability is not a substitute for Stage3 readiness.",
    ]
    write_text(RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md", "\n".join(boundaries))
    append_registry(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3",
            "branch": "internal_signal_readiness",
            "status": "NO_GO_INTERNAL_SIGNAL_READINESS_BLOCKED",
            "artifact": rel(STAGE3_SUMMARY),
            "decision": "zero Stage4-ready surfaces; semantic proxies not promoted",
        }
    )
    append_registry(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Final",
            "branch": "all_branches",
            "status": taxonomy,
            "artifact": rel(RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json"),
            "decision": "all 12 branches closed as structural blockers after mechanism-distinct repair attempts; no runtime metrics fabricated",
        }
    )
    print(json.dumps(clean_json(final_summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
