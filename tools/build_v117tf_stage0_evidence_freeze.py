#!/usr/bin/env python3
"""Build ACL2 v117-TF Stage0 evidence-freeze artifacts.

This is a read-only audit. It does not run geometry experiments. It records
which raw reference artifacts are currently readable, which metrics are only
available through derived v116 freeze artifacts, and whether Stage0 is ready
for runtime action.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v117tf_same_space_semantic_memory_reliability"
OUT = RESULT_ROOT / "stage0_evidence_freeze"

V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
V111 = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
V114 = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"
V115 = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
V116 = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"

V116_STAGE0 = V116 / "stage0_evidence_freeze"

REQUIRED_RAW_ARTIFACTS = [
    ("v110_b1_full_metrics", V110 / "stage4_full_00_01_02_05_validation/full_metric_rows.csv"),
    ("v110_b1_action_fidelity", V110 / "stage4_full_00_01_02_05_validation/action_fidelity_rows.csv"),
    ("v111_a1_metric_summary", V111 / "batch_a_a1_anchor_selection/a1_metric_summary.json"),
    ("v111_a1_full_metrics", V111 / "batch_a_a1_anchor_selection/full_metric_rows.csv"),
    ("v112_stage0_summary", V112 / "stage0_evidence_freeze/stage0_summary.json"),
    ("v113_hs_value_decision", V113 / "diagnostics/stage6_action_decision_summary.json"),
    ("v113_hs_projection_seq00_risk", V113 / "semantic_projection/seq00_risk.npy"),
    ("v113_hs_projection_seq02_risk", V113 / "semantic_projection/seq02_risk.npy"),
    ("v114_stage0_summary", V114 / "stage0_evidence_freeze/stage0_summary.json"),
    ("v114_hs_lq_decision", V114 / "diagnostics/stage_hs_lq_decision_summary.json"),
    ("v115_stage0_summary", V115 / "stage0_evidence_freeze/stage0_summary.json"),
    ("v115_hook_audit_summary", V115 / "stage1_hook_audit/stage1_hook_audit_summary.json"),
    ("v115_l2_query_full_summary", V115 / "stage5_lingbot_a2_l2_query_full_pilot_00_02/query_full_metric_summary.json"),
    ("v115_l2_control_summary", V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json"),
    ("v116_final_decision_summary", V116 / "V116_FINAL_DECISION_SUMMARY.md"),
    ("v116_continuation_decision", V116 / "carrier_diagnosis/CONTINUATION_GQ6_CONTINUITY_SEMINT_CONTROL_DECISION.md"),
]

DERIVED_REFERENCE_ARTIFACTS = [
    ("v116_stage0_artifact_manifest", V116_STAGE0 / "STAGE0_ARTIFACT_MANIFEST.csv"),
    ("v116_stage0_baseline_boundary_rows", V116_STAGE0 / "STAGE0_BASELINE_BOUNDARY_ROWS.csv"),
    ("v116_stage0_summary", V116_STAGE0 / "STAGE0_EVIDENCE_FREEZE_SUMMARY.json"),
    ("v116_task1_policy_summary", V116 / "task1_ab/TASK1_POLICY_SUMMARY.csv"),
    ("v116_task1_control_policy_summary", V116 / "task1_ab_controls/TASK1_CONTROL_POLICY_SUMMARY.csv"),
    ("v116_task1_control_decision", V116 / "task1_ab_controls/TASK1_CONTROL_DECISION_SUMMARY.json"),
    ("v116_carrier_repair_summary", V116 / "carrier_diagnosis/CARRIER_REPAIR_DIAGNOSTIC_SUMMARY.json"),
    ("v116_semantic_cue_readiness", V116 / "carrier_diagnosis/SEMANTIC_CUE_REDESIGN_READINESS.json"),
]

CODE_LOCI = [
    ("lingbot_wrapper", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "LingBot runtime wrapper and action surface"),
    ("lingbot_attention", ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py", "LingBot attention / local read code"),
    ("horizonstream_semantic_runtime", ROOT / "third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", "HorizonStream semantic action hooks"),
    ("horizonstream_model", ROOT / "third_party/HorizonStream/horizonstream/models/horizonstream.py", "HorizonStream model wiring"),
    ("horizonstream_runner", ROOT / "tools/run_v115tf_hs_deterministic_pipeline.py", "Deterministic HorizonStream runner reused by v116"),
    ("v116_task1_metrics", ROOT / "tools/build_v116tf_task1_ab_metrics.py", "LingBot B1/A1 metric builder reference"),
    ("v116_task1_control_metrics", ROOT / "tools/build_v116tf_task1_ab_control_metrics.py", "LingBot matched-control metric builder reference"),
    ("v116_stage0_builder", ROOT / "tools/build_v116tf_stage0_evidence_freeze.py", "Prior Stage0 freeze builder reference"),
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def artifact_row(artifact_id: str, path: Path, required: bool, artifact_class: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "path": rel(path),
        "required": required,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else "",
    }


def artifact_rows() -> list[dict[str, Any]]:
    rows = [artifact_row(artifact_id, path, True, "raw_reference") for artifact_id, path in REQUIRED_RAW_ARTIFACTS]
    rows.extend(artifact_row(artifact_id, path, False, "derived_reference") for artifact_id, path in DERIVED_REFERENCE_ARTIFACTS)
    return rows


def code_loci_rows() -> list[dict[str, Any]]:
    return [
        {
            "locus_id": locus_id,
            "path": rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else "",
            "purpose": purpose,
        }
        for locus_id, path, purpose in CODE_LOCI
    ]


def process_rows() -> list[dict[str, Any]]:
    markers = (
        "v117",
        "v116",
        "v115",
        "horizonstream",
        "lingbot",
        "run_v",
        "build_v",
    )
    ignore = (
        "build_v117tf_stage0_evidence_freeze.py",
        "ps -eo pid,ppid,user,stat,etime,cmd",
        "rg -i",
    )
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,user,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[1:]:
        if not any(marker in line.lower() for marker in markers):
            continue
        if any(marker in line for marker in ignore):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        rows.append(
            {
                "row_type": "process",
                "gpu_index": "",
                "pid": parts[0],
                "ppid": parts[1],
                "user": parts[2],
                "stat": parts[3],
                "etime": parts[4],
                "cmd": parts[5],
                "memory_used_mib": "",
                "memory_total_mib": "",
                "gpu_utilization_pct": "",
            }
        )
    return rows


def gpu_rows() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [
            {
                "row_type": "gpu_query_error",
                "gpu_index": "",
                "pid": "",
                "ppid": "",
                "user": "",
                "stat": "",
                "etime": "",
                "cmd": proc.stderr.strip(),
                "memory_used_mib": "",
                "memory_total_mib": "",
                "gpu_utilization_pct": "",
            }
        ]
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "row_type": "gpu",
                "gpu_index": parts[0],
                "pid": "",
                "ppid": "",
                "user": "",
                "stat": "",
                "etime": "",
                "cmd": parts[1],
                "memory_used_mib": parts[2],
                "memory_total_mib": parts[3],
                "gpu_utilization_pct": parts[4],
            }
        )
    return rows


def reference_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in read_csv(V116_STAGE0 / "STAGE0_BASELINE_BOUNDARY_ROWS.csv"):
        rows.append(
            {
                "metric_id": f"derived_v116_stage0_{row.get('source_version', '')}_{row.get('candidate_or_fact', '')}",
                "model": row.get("model", ""),
                "source_version": row.get("source_version", ""),
                "candidate_or_fact": row.get("candidate_or_fact", ""),
                "seq_scope": "as_reported_in_source",
                "median_full_rel": row.get("median_full_rel", ""),
                "mean_full_rel": row.get("mean_full_rel", ""),
                "improved_seq_count": row.get("improved_seq_count", ""),
                "max_harm": row.get("max_harm", ""),
                "semantic_status": row.get("semantic_status", ""),
                "source_readability": "derived_v116_stage0_original_path_missing_or_unchecked",
                "source_path": row.get("source_path", ""),
                "v117_use": "historical boundary only; raw artifact must be restored for Stage0 raw pass",
            }
        )

    for row in read_csv(V116 / "task1_ab/TASK1_POLICY_SUMMARY.csv"):
        if row.get("policy_id") in {"AB0_B1_semantic_only_reference", "AB_CTRL_A1_default_first8_plus_B1"}:
            rows.append(
                {
                    "metric_id": f"v116_task1_{row.get('policy_id')}",
                    "model": "LingBot",
                    "source_version": "v116TF",
                    "candidate_or_fact": row.get("policy_id", ""),
                    "seq_scope": "00,02",
                    "median_full_rel": row.get("median_full_rel", ""),
                    "mean_full_rel": row.get("mean_full_rel", ""),
                    "improved_seq_count": row.get("improved_seq_count", ""),
                    "max_harm": row.get("max_harm", ""),
                    "semantic_status": "v116_geometry_liveness_control_matched_later",
                    "source_readability": "raw_v116_metric_artifact_readable",
                    "source_path": rel(V116 / "task1_ab/TASK1_POLICY_SUMMARY.csv"),
                    "v117_use": "pilot-only B1 liveness and action-count reference; not semantic proof",
                }
            )

    control_decision = read_json(V116 / "task1_ab_controls/TASK1_CONTROL_DECISION_SUMMARY.json")
    if control_decision:
        rows.append(
            {
                "metric_id": "v116_task1_b1_best_control",
                "model": "LingBot",
                "source_version": "v116TF",
                "candidate_or_fact": control_decision.get("best_b1_control_policy_id", ""),
                "seq_scope": "00,02",
                "median_full_rel": control_decision.get("best_b1_control_median_full_rel", ""),
                "mean_full_rel": "",
                "improved_seq_count": "",
                "max_harm": "",
                "semantic_status": control_decision.get("task_status", ""),
                "source_readability": "raw_v116_metric_artifact_readable",
                "source_path": rel(V116 / "task1_ab_controls/TASK1_CONTROL_DECISION_SUMMARY.json"),
                "v117_use": "B1 semantic selector is control matched; v117 must use stronger same-budget provenance controls",
            }
        )

    carrier = read_json(V116 / "carrier_diagnosis/CARRIER_REPAIR_DIAGNOSTIC_SUMMARY.json")
    for row in carrier.get("branches", []) if isinstance(carrier.get("branches"), list) else []:
        agg = row.get("vs_fresh_noaction", {}).get("aggregate", {})
        rows.append(
            {
                "metric_id": f"v116_carrier_{row.get('branch', row.get('name', 'unknown'))}",
                "model": "HorizonStream",
                "source_version": "v116TF",
                "candidate_or_fact": row.get("branch", row.get("name", "")),
                "seq_scope": "00,02",
                "median_full_rel": agg.get("median_full_ATE_rel_improvement", ""),
                "mean_full_rel": "",
                "improved_seq_count": "",
                "max_harm": agg.get("max_full_ATE_harm_rel", ""),
                "semantic_status": "generic_carrier_safety_diagnostic_only",
                "source_readability": "raw_v116_metric_artifact_readable",
                "source_path": rel(V116 / "carrier_diagnosis/CARRIER_REPAIR_DIAGNOSTIC_SUMMARY.json"),
                "v117_use": "generic rowmean + MRT tight is a mandatory HS control",
            }
        )

    return rows


def forbidden_text() -> str:
    return """# v117 Stage0 Forbidden Repeats

- Do not run frame-level stable/risk role-mass strength sweeps.
- Do not run HS-HG mild/medium/sparse head-output gates as a substitute for same-space reliability.
- Do not run L2T class-only risk/stable token bias.
- Do not treat B1 semantic selector as proven semantic method.
- Do not call rowmean+MRT a semantic method.
- Do not use continuity-only sidecar directly as semantic action.
- Do not replace fixed first-write memory reference with moving-average state reference.
- Do not add controls only after promoting an action.
- Do not use source-frame aggregate provenance to claim token-level same-space causality.
- Do not use GT error, SLAM, external depth, or output post-processing as runtime cue.
"""


def allowed_text() -> str:
    return """# v117 Stage0 Allowed Carriers

- LingBot B1 no-append/cache-admission carrier: allowed only as fixed-budget action surface with same-count, same-bucket, track/instance/reliability/internal controls.
- LingBot A1 anchor selection: allowed as clean semantic anchor reference and later persistent-landmark role test.
- LingBot local read: allowed only when semantic role, object persistence, selected-query alignment, entropy, and residual are all auditable.
- HorizonStream local value carrier: allowed only with same-magnitude, row-mean, track-shuffle, instance-shuffle, reliability-shuffle, internal-shuffle, reverse, and rowmean generic+MRT controls.
- HorizonStream GLA update: allowed only if Stage2 maps auditable state units and Stage3 fixed-reference reliability is non-constant.
- MRT tight scale-delta guard: allowed as safety layer only, never as semantic claim by itself.
"""


def missing_text(missing: list[dict[str, Any]], derived: list[dict[str, Any]]) -> str:
    lines = [
        "# v117 Stage0 Missing Artifacts",
        "",
        "Stage0 requires raw historical reference artifacts. The files below are missing in the current checkout.",
        "Derived v116 freeze artifacts are readable and useful for orientation, but they are not treated as a raw-reference pass.",
        "",
        "## Missing Required Raw Artifacts",
        "",
        "| artifact_id | path |",
        "|---|---|",
    ]
    for row in missing:
        lines.append(f"| {row['artifact_id']} | `{row['path']}` |")
    if not missing:
        lines.append("| none | none |")
    lines += [
        "",
        "## Readable Derived References",
        "",
        "| artifact_id | path | exists |",
        "|---|---|---:|",
    ]
    for row in derived:
        lines.append(f"| {row['artifact_id']} | `{row['path']}` | `{row['exists']}` |")
    lines += [
        "",
        "## Repair Attempts Performed In This Turn",
        "",
        "- Listed current `results/` roots: only v114, v115, v116 and preprocessing roots were present.",
        "- Searched current workspace for v110/v111/v112/v113 exact result paths and key metric filenames; no raw result files were found.",
        "- Started broader `/mnt/data/users/chengshun.wang` `find` searches for missing roots and metric files, then interrupted them after they exceeded the bounded search window without output.",
        "",
        "## Consequence",
        "",
        "Stage0 raw-reference gate is not passed. Runtime action that claims v117 object-level semantic causality should not start until raw artifacts are restored or the reference metrics are regenerated as artifact repair.",
    ]
    return "\n".join(lines)


def report_text(summary: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v117-TF Stage0 Evidence Freeze",
        "",
        f"- stage0_complete: `{summary['stage0_complete']}`",
        f"- raw_reference_artifacts_readable: `{summary['raw_reference_artifacts_readable']}`",
        f"- derived_frozen_metrics_readable: `{summary['derived_frozen_metrics_readable']}`",
        f"- no_stale_worker: `{summary['no_stale_worker']}`",
        f"- code_loci_readable: `{summary['code_loci_readable']}`",
        f"- required_missing_count: `{summary['required_missing_count']}`",
        "",
        "## Reference Rows",
        "",
        "| metric_id | model | source | candidate_or_fact | median_full_rel | semantic_status | source_readability |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in metrics:
        lines.append(
            "| {metric_id} | {model} | {source_version} | {candidate_or_fact} | {median_full_rel} | {semantic_status} | {source_readability} |".format(
                **{k: str(row.get(k, "")).replace("|", "/") for k in row}
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "Stage0 is not passed because raw v110R/v111/v112/v113 reference artifacts are missing in the current checkout.",
        "The readable v116 derived artifacts are retained as orientation evidence only and are labeled as derived in `stage0_reference_metrics.csv`.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    artifacts = artifact_rows()
    code_rows = code_loci_rows()
    gpu_proc_rows = gpu_rows() + process_rows()
    metrics = reference_metric_rows()

    missing_required = [row for row in artifacts if row["required"] and not row["exists"]]
    derived_rows = [row for row in artifacts if row["artifact_class"] == "derived_reference"]
    stale_processes = [row for row in gpu_proc_rows if row["row_type"] == "process"]
    raw_reference_artifacts_readable = not missing_required
    derived_frozen_metrics_readable = bool(metrics) and any(
        row["source_readability"].startswith("derived_v116_stage0") for row in metrics
    )
    no_stale_worker = not stale_processes
    code_loci_readable = all(row["exists"] for row in code_rows)
    b1_a1_hs_exact_match = raw_reference_artifacts_readable

    summary = {
        "schema": "acl2_v117tf_stage0_evidence_freeze_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": rel(RESULT_ROOT),
        "stage0_complete": raw_reference_artifacts_readable and no_stale_worker and code_loci_readable and b1_a1_hs_exact_match,
        "raw_reference_artifacts_readable": raw_reference_artifacts_readable,
        "derived_frozen_metrics_readable": derived_frozen_metrics_readable,
        "no_stale_worker": no_stale_worker,
        "code_loci_readable": code_loci_readable,
        "b1_a1_hs_carrier_baseline_exact_match": b1_a1_hs_exact_match,
        "required_missing_count": len(missing_required),
        "missing_required_artifacts": missing_required,
        "stale_process_count": len(stale_processes),
        "stage0_blocker": "" if raw_reference_artifacts_readable else "RAW_REFERENCE_ARTIFACTS_MISSING",
        "next_repair_direction": "restore missing v110R/v111/v112/v113 artifact roots or regenerate those reference metrics as artifact repair before runtime action",
        "outputs": {
            "frozen_facts": rel(OUT / "stage0_frozen_facts.json"),
            "reference_metrics": rel(OUT / "stage0_reference_metrics.csv"),
            "forbidden_repeats": rel(OUT / "stage0_forbidden_repeats.md"),
            "allowed_carriers": rel(OUT / "stage0_allowed_carriers.md"),
            "code_loci": rel(OUT / "stage0_code_loci.csv"),
            "missing_artifacts": rel(OUT / "stage0_missing_artifacts.md"),
            "process_gpu_audit": rel(OUT / "stage0_process_gpu_audit.csv"),
            "report": rel(OUT / "STAGE0_EVIDENCE_FREEZE_REPORT.md"),
        },
    }

    write_json(OUT / "stage0_frozen_facts.json", summary)
    write_csv(OUT / "stage0_reference_artifacts.csv", artifacts)
    write_csv(OUT / "stage0_reference_metrics.csv", metrics)
    write_text(OUT / "stage0_forbidden_repeats.md", forbidden_text())
    write_text(OUT / "stage0_allowed_carriers.md", allowed_text())
    write_csv(OUT / "stage0_code_loci.csv", code_rows)
    write_text(OUT / "stage0_missing_artifacts.md", missing_text(missing_required, derived_rows))
    write_csv(OUT / "stage0_process_gpu_audit.csv", gpu_proc_rows)
    write_text(OUT / "STAGE0_EVIDENCE_FREEZE_REPORT.md", report_text(summary, metrics))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
