#!/usr/bin/env python3
"""Build ACL2 v114-TF Stage0 frozen facts and forbidden-repeat artifacts.

This script is read-only with respect to prior experiment outputs. It freezes
the v112 LingBot and v113 HorizonStream boundaries before v114 action runs.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"

V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"


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


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stale_process_rows() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    markers = (
        "third_party/HorizonStream",
        "horizonstream",
        "HorizonStream",
        "lingbot",
        "LingBot",
        "v113hs",
        "v114tf",
        "run_v111",
        "run_v112",
    )
    self_markers = (
        "build_v114tf_stage0_evidence_freeze.py",
        "ps -eo pid,ppid,stat,etime,cmd",
        "rg -i",
        "grep",
        "tee results/acl2_v114tf",
    )
    out: list[str] = []
    for line in proc.stdout.splitlines():
        if not any(marker in line for marker in markers):
            continue
        if any(marker in line for marker in self_markers):
            continue
        out.append(line)
    return out


def v113_baseline_rows() -> list[dict[str, Any]]:
    rows = read_csv(V113 / "diagnostics/stage1_hs_baseline_metrics_rows.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "model": "HorizonStream",
                "source_version": "v113-HS",
                "policy_or_variant": "baseline_default_no_loop",
                "seq": row.get("seq", ""),
                "full_ATE_sim3_rmse": row.get("full_ATE_sim3_rmse", ""),
                "rolling_ate_p90": row.get("rolling_ate_p90", ""),
                "segment_scale_log_error_median_abs": row.get("segment_scale_log_error_median_abs", ""),
                "adjacent_log_scale_jump_p90_abs": row.get("adjacent_log_scale_jump_p90_abs", ""),
                "source_path": rel(V113 / "diagnostics/stage1_hs_baseline_metrics_rows.csv"),
                "note": "v113 baseline exists for 00/02/05 only; v114 must add KITTI01 before full 00/01/02/05 validation",
            }
        )
    v112_summary = read_json(V112 / "stage0_evidence_freeze/stage0_summary.json")
    for key, label in [
        ("b1_reference", "B1_semantic_only"),
        ("a1_reference", "A1_low_dynamic_from_first32"),
        ("f19_reference", "F19_dynamic_or_special_admitted_high_risk_else_weak_context"),
    ]:
        ref = v112_summary.get(key, {}) if isinstance(v112_summary, dict) else {}
        out.append(
            {
                "model": "LingBot",
                "source_version": "v112 freeze of earlier references",
                "policy_or_variant": label,
                "seq": "00,01,02,05",
                "median_full_rel_improvement": ref.get("median_full_rel", ""),
                "mean_full_rel_improvement": ref.get("mean_full_rel", ""),
                "improved_seq_count": ref.get("improved_seq_count", ""),
                "max_harm": ref.get("max_harm", ""),
                "source_path": ref.get("source", rel(V112 / "stage0_evidence_freeze/stage0_summary.json")),
                "note": "reference boundary only; not a v114 action result",
            }
        )
    return out


def causality_failure_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v112_final = read_json(V112 / "stage8_final_decision/final_decision_summary.json")
    if v112_final:
        rows.append(
            {
                "model": "LingBot",
                "source_version": "v112",
                "failure_id": "H1_T4_same_schedule_mask_shuffle_match",
                "taxonomy": v112_final.get("final_taxonomy", ""),
                "candidate": "H1/T4 trajectory lifetime/mask candidates",
                "matched_control": "same-schedule mask-shuffle",
                "candidate_metric": v112_final.get("key_metrics", {}).get("stage6_h1_semantic_lifetime_soft_raw_median_full_rel", ""),
                "control_metric": v112_final.get("key_metrics", {}).get("stage7_h1_best_control_median_full_rel", ""),
                "margin": v112_final.get("key_metrics", {}).get("stage7_h1_candidate_minus_best_control", ""),
                "source_path": rel(V112 / "stage8_final_decision/final_decision_summary.json"),
                "lesson": "trajectory mask shape/schedule can explain geometry gain; v114 T6/T7 must add internal quality or retrieval/write hooks",
            }
        )
    v113_decision = read_json(V113 / "diagnostics/stage6_action_decision_summary.json")
    if v113_decision:
        target = v113_decision.get("target", {})
        for control in v113_decision.get("control_matches_or_exceeds_target", []):
            control_row = next((r for r in v113_decision.get("rows", []) if r.get("name") == control), {})
            rows.append(
                {
                    "model": "HorizonStream",
                    "source_version": "v113-HS",
                    "failure_id": "HS_geometry_pass_semantic_causality_fail",
                    "taxonomy": "HS_GEOMETRY_PASS_SEMANTIC_CAUSALITY_FAIL",
                    "candidate": target.get("name", ""),
                    "matched_control": control,
                    "candidate_median_full_ATE_rel": target.get("median_full_ATE_rel_improvement", ""),
                    "control_median_full_ATE_rel": control_row.get("median_full_ATE_rel_improvement", ""),
                    "candidate_median_rolling_p90_rel": target.get("median_rolling_p90_rel_improvement", ""),
                    "control_median_rolling_p90_rel": control_row.get("median_rolling_p90_rel_improvement", ""),
                    "source_path": rel(V113 / "diagnostics/stage6_action_decision_summary.json"),
                    "lesson": "semantic shuffle/same-count random can reproduce or exceed the gain; v114 HS-LQ must neutralize row mean and condition on internal local quality",
                }
            )
    return rows


def hook_blocker_rows() -> list[dict[str, Any]]:
    paths = [
        V112 / "stage1_hook_traceability_audit/A2_ANCHOR_SOURCE_SPAN_BLOCKED.md",
        V112 / "stage1_hook_traceability_audit/QUERY_TYPE_INDEX_BLOCKED.md",
        V112 / "stage1_hook_traceability_audit/T5_RETRIEVAL_HOOK_BLOCKED.md",
        V112 / "stage1_hook_traceability_audit/C1_D1_HOOK_BLOCKED_REPORT.md",
        V112 / "stage1_hook_traceability_audit/H2_HEAD_OUTPUT_HOOK_BLOCKED.md",
        V113 / "audit/hs_hook_contract_gla.md",
        V113 / "audit/hs_hook_contract_local.md",
        V113 / "audit/hs_hook_contract_mrt.md",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        rows.append(
            {
                "path": rel(path),
                "exists": path.exists(),
                "surface": path.stem,
                "summary": " ".join(text.splitlines()[:3])[:500] if text else "",
                "v114_implication": (
                    "blocked_or_contract_only; must repair hook before action claim"
                    if "BLOCKED" in path.name or "blocked" in text.lower()
                    else "contract exists; verify parity before extending"
                ),
            }
        )
    return rows


def artifact_status_rows() -> list[dict[str, Any]]:
    required = [
        V112 / "stage0_evidence_freeze/stage0_summary.json",
        V112 / "stage8_final_decision/final_decision_summary.json",
        V112 / "stage9_a2_anchor_source_bias_proxy_pilot_00_02/A2_TOKEN_LEVEL_CUE_BLOCKED.md",
        V113 / "diagnostics/stage1_hs_baseline_metrics_summary.json",
        V113 / "diagnostics/hs_semantic_projection_summary.json",
        V113 / "diagnostics/hs_noop_trace_parity_summary.json",
        V113 / "diagnostics/hs_influence_kernel_summary.json",
        V113 / "diagnostics/stage6_action_decision_summary.json",
        V113 / "semantic_projection/seq00_risk.npy",
        V113 / "semantic_projection/seq02_risk.npy",
        V113 / "semantic_projection/seq05_risk.npy",
    ]
    return [
        {
            "path": rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else "",
            "required_for": "stage0_freeze",
        }
        for path in required
    ]


def main() -> None:
    STAGE0.mkdir(parents=True, exist_ok=True)
    stale_rows = stale_process_rows()

    forbidden = [
        "Pure semantic scalar gate without internal quality.",
        "Same schedule/mask shape but claiming semantic causality.",
        "Generic row-mean / value magnitude perturbation presented as semantic.",
        "HorizonStream local stable boost without segment-scale guard.",
        "LingBot H1/T4 same schedule without stronger causality control.",
        "Anchor A2 frame-uniform proxy presented as token-level anchor source management.",
        "HorizonStream GLA gamma modulation claimed without direct gamma hook.",
        "Local Window scientific No-Go without implementing safe local read hook.",
    ]
    write_text(
        STAGE0 / "stage0_forbidden_repeats.md",
        "# Stage0 Forbidden Repeats\n\n" + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(forbidden)),
    )
    write_text(
        STAGE0 / "stage0_next_allowed_surfaces.md",
        """# Stage0 Next Allowed Surfaces

Priority order follows the v114 plan:

1. HorizonStream HS-LQ row-mean neutral semantic+internal local gate.
2. LingBot A2 true token-level Anchor Context read hook, only after source-span/default-off parity.
3. LingBot L2 Local Window query-type-specific read hook, only after local source/query span audit.
4. LingBot T6/T7 trajectory admission/retrieval with internal relevance or contradiction, not same schedule mask replay.
5. HorizonStream HS-GQ/HS-MQ safety only after local path gate and scale diagnostics.

No full `00/01/02/05` HorizonStream claim is allowed until KITTI01 baseline exists.
""",
    )
    write_csv(STAGE0 / "stage0_known_good_geometry_baselines.csv", v113_baseline_rows())
    write_csv(STAGE0 / "stage0_semantic_causality_failures.csv", causality_failure_rows())
    write_csv(STAGE0 / "stage0_hook_blockers.csv", hook_blocker_rows())
    write_csv(STAGE0 / "stage0_artifact_manifest.csv", artifact_status_rows())

    if stale_rows:
        write_text(
            STAGE0 / "STALE_WORKER_BLOCKER.md",
            "# STALE_WORKER_BLOCKER\n\n"
            "Stage0 detected existing LingBot/HorizonStream-related workers. Per v114 plan, no action run should start until these are resolved.\n\n"
            + "\n".join(f"- `{line}`" for line in stale_rows),
        )

    missing_required = [row["path"] for row in artifact_status_rows() if not row["exists"]]
    summary = {
        "schema": "acl2_v114tf_stage0_evidence_freeze_summary_v1",
        "stage0_pass": not stale_rows and not missing_required,
        "no_stale_worker": not stale_rows,
        "pending_worker_rows": stale_rows,
        "required_missing_artifacts": missing_required,
        "forbidden_repeat_count": len(forbidden),
        "known_good_geometry_baseline_rows": len(v113_baseline_rows()),
        "semantic_causality_failure_rows": len(causality_failure_rows()),
        "hook_blocker_rows": len(hook_blocker_rows()),
        "outputs": {
            "stage0_frozen_facts": rel(STAGE0 / "stage0_frozen_facts.json"),
            "forbidden_repeats": rel(STAGE0 / "stage0_forbidden_repeats.md"),
            "known_good_geometry_baselines": rel(STAGE0 / "stage0_known_good_geometry_baselines.csv"),
            "semantic_causality_failures": rel(STAGE0 / "stage0_semantic_causality_failures.csv"),
            "hook_blockers": rel(STAGE0 / "stage0_hook_blockers.csv"),
            "next_allowed_surfaces": rel(STAGE0 / "stage0_next_allowed_surfaces.md"),
            "artifact_manifest": rel(STAGE0 / "stage0_artifact_manifest.csv"),
        },
        "v113_hs_baseline_missing_seq01": True,
        "priority1": "HS-LQ-1: add KITTI01 baseline, then run row-mean-neutral semantic+internal local gates and controls on KITTI 00/02",
    }
    frozen = {
        "summary": summary,
        "v112_final_decision": read_json(V112 / "stage8_final_decision/final_decision_summary.json"),
        "v113_baseline_summary": read_json(V113 / "diagnostics/stage1_hs_baseline_metrics_summary.json").get("aggregate", {}),
        "v113_stage6_decision": read_json(V113 / "diagnostics/stage6_action_decision_summary.json"),
        "stage0_artifact_manifest": artifact_status_rows(),
    }
    write_json(STAGE0 / "stage0_frozen_facts.json", frozen)
    write_json(STAGE0 / "stage0_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
