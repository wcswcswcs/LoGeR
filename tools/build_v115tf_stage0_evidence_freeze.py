#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
V114 = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
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
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def process_rows() -> list[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,stat,etime,cmd"], cwd=ROOT, text=True, capture_output=True, check=False)
    markers = ("HorizonStream", "horizonstream", "LingBot", "lingbot", "v113hs", "v114tf", "v115tf")
    self_markers = ("build_v115tf_stage0_evidence_freeze.py", "ps -eo", "rg ", "grep ")
    rows = []
    for line in proc.stdout.splitlines():
        if any(m in line for m in markers) and not any(m in line for m in self_markers):
            rows.append(line)
    return rows


def artifact_status() -> list[dict[str, Any]]:
    required = [
        V112 / "stage0_evidence_freeze/stage0_summary.json",
        V112 / "stage8_final_decision/final_decision_summary.json",
        V113 / "diagnostics/stage1_hs_baseline_metrics_summary.json",
        V113 / "diagnostics/hs_semantic_projection_summary.json",
        V113 / "diagnostics/hs_noop_trace_parity_summary.json",
        V113 / "diagnostics/stage6_action_decision_summary.json",
        V113 / "semantic_projection/seq00_risk.npy",
        V113 / "semantic_projection/seq01_risk.npy",
        V113 / "semantic_projection/seq02_risk.npy",
        V113 / "semantic_projection/seq05_risk.npy",
        V114 / "diagnostics/stage_hs_lq_decision_summary.json",
        V114 / "diagnostics/stage_hs_lq_decision_rows.csv",
    ]
    return [
        {
            "path": rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else "",
        }
        for path in required
    ]


def known_geometry_controls() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V113 / "diagnostics/stage1_hs_baseline_metrics_rows.csv"):
        rows.append(
            {
                "model": "HorizonStream",
                "source_version": "v113",
                "policy_or_variant": "baseline_default_no_loop",
                "seq": row.get("seq", ""),
                "full_ATE_sim3_rmse": row.get("full_ATE_sim3_rmse", ""),
                "rolling_ate_p90": row.get("rolling_ate_p90", ""),
                "segment_scale_log_error_median_abs": row.get("segment_scale_log_error_median_abs", ""),
                "source_path": rel(V113 / "diagnostics/stage1_hs_baseline_metrics_rows.csv"),
                "v115_usage": "baseline reference only",
            }
        )
    for row in read_csv(V114 / "diagnostics/stage_hs_lq_decision_rows.csv"):
        if row.get("kind") in {"control", "control_full"} or "rowmean" in row.get("label", ""):
            rows.append(
                {
                    "model": "HorizonStream",
                    "source_version": "v114",
                    "policy_or_variant": row.get("label", ""),
                    "seq": row.get("candidate_name", ""),
                    "median_full_ATE_rel_improvement": row.get("median_full_ATE_rel_improvement", ""),
                    "median_rolling_p90_rel_improvement": row.get("median_rolling_p90_rel_improvement", ""),
                    "max_full_ATE_harm_rel": row.get("max_full_ATE_harm_rel", ""),
                    "segment_scale_not_worse_all": row.get("segment_scale_not_worse_all", ""),
                    "strict_pass": row.get("v114_strict_pilot_pass", ""),
                    "source_path": rel(V114 / "diagnostics/stage_hs_lq_decision_rows.csv"),
                    "v115_usage": "known confound/control; not semantic causality",
                }
            )
    return rows


def semantic_failures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    v112_final = read_json(V112 / "stage8_final_decision/final_decision_summary.json")
    if v112_final:
        out.append(
            {
                "model": "LingBot",
                "source_version": "v112",
                "failure_id": "same_schedule_or_shuffle_control_match",
                "taxonomy": v112_final.get("final_taxonomy", ""),
                "source_path": rel(V112 / "stage8_final_decision/final_decision_summary.json"),
                "v115_lesson": "Non-B1/H1 schedule action must have a real hook and causality controls.",
            }
        )
    v113_decision = read_json(V113 / "diagnostics/stage6_action_decision_summary.json")
    if v113_decision:
        out.append(
            {
                "model": "HorizonStream",
                "source_version": "v113",
                "failure_id": "semantic_shuffle_or_same_count_matches_value_gate",
                "taxonomy": "HS_GEOMETRY_PASS_SEMANTIC_CAUSALITY_FAIL",
                "target": v113_decision.get("target", {}).get("name", ""),
                "matched_controls": ",".join(v113_decision.get("control_matches_or_exceeds_target", [])),
                "source_path": rel(V113 / "diagnostics/stage6_action_decision_summary.json"),
                "v115_lesson": "Do not use local value-scaling or row-count perturbation as semantic causality.",
            }
        )
    v114_decision = read_json(V114 / "diagnostics/stage_hs_lq_decision_summary.json")
    if v114_decision:
        out.append(
            {
                "model": "HorizonStream",
                "source_version": "v114",
                "failure_id": "semantic_internal_lq_failed_generic_rowmean_passed",
                "taxonomy": v114_decision.get("final_taxonomy", ""),
                "decision_reason": v114_decision.get("decision_reason", ""),
                "source_path": rel(V114 / "diagnostics/stage_hs_lq_decision_summary.json"),
                "v115_lesson": "Move from local value scaling to query/head/state action surfaces.",
            }
        )
    return out


def main() -> None:
    STAGE0.mkdir(parents=True, exist_ok=True)
    status = artifact_status()
    missing = [row for row in status if not row["exists"]]
    facts = {
        "schema": "acl2_v115tf_stage0_frozen_facts_v1",
        "result_root": rel(RESULT_ROOT),
        "artifact_status": status,
        "missing_required_artifacts": missing,
        "active_process_rows": process_rows(),
        "v112_final_decision": read_json(V112 / "stage8_final_decision/final_decision_summary.json"),
        "v113_baseline_summary": read_json(V113 / "diagnostics/stage1_hs_baseline_metrics_summary.json"),
        "v113_decision_summary": read_json(V113 / "diagnostics/stage6_action_decision_summary.json"),
        "v114_decision_summary": read_json(V114 / "diagnostics/stage_hs_lq_decision_summary.json"),
    }
    write_json(STAGE0 / "stage0_frozen_facts.json", facts)
    write_csv(STAGE0 / "stage0_known_geometry_controls.csv", known_geometry_controls())
    write_csv(STAGE0 / "stage0_semantic_failures.csv", semantic_failures())
    write_text(
        STAGE0 / "stage0_forbidden_repeats.md",
        """# Stage0 Forbidden Repeats

1. Pure semantic scalar gate without internal quality.
2. Local KV value scaling or row-mean magnitude perturbation claimed as semantic causality.
3. v114 generic rowmean + MRT safety gate claimed as a v115 semantic method.
4. Same schedule, same count, same magnitude, or semantic shuffle controls omitted after a geometry pass.
5. LingBot B1/H1 schedule-only reuse without a non-B1/H1 hook audit.
6. Attention-logit/full-attention claims when fused SDPA hides the probability map.
7. Pose-level parity failures written as full Stage1 pass.
""",
    )
    write_text(
        STAGE0 / "stage0_allowed_new_surfaces.md",
        """# Stage0 Allowed New Surfaces

1. HS-HG: local pose-query per-head output gate, row-mean neutral when possible.
2. HS-LA: local query/logit action only if attention weights/logits are explicitly materialized and parity passes.
3. HS-GQ: pre-GLA/state update gain using semantic + internal novelty, with bounded state norm evidence.
4. MRT: safety layer only after a base candidate passes; never the primary semantic method.
5. LingBot A2/L2/T7 or other non-B1/H1 surfaces only after source-span/action fidelity audit.
""",
    )
    if missing:
        write_text(
            STAGE0 / "V115_REFERENCE_ARTIFACT_MISSING.md",
            "# V115 Reference Artifact Missing\n\n"
            + "\n".join(f"- `{row['path']}`" for row in missing)
            + "\n",
        )
    summary = {
        "schema": "acl2_v115tf_stage0_summary_v1",
        "stage0_complete": not missing,
        "missing_count": len(missing),
        "known_geometry_control_count": len(known_geometry_controls()),
        "semantic_failure_count": len(semantic_failures()),
        "outputs": {
            "stage0_frozen_facts": rel(STAGE0 / "stage0_frozen_facts.json"),
            "stage0_known_geometry_controls": rel(STAGE0 / "stage0_known_geometry_controls.csv"),
            "stage0_semantic_failures": rel(STAGE0 / "stage0_semantic_failures.csv"),
            "stage0_forbidden_repeats": rel(STAGE0 / "stage0_forbidden_repeats.md"),
            "stage0_allowed_new_surfaces": rel(STAGE0 / "stage0_allowed_new_surfaces.md"),
        },
    }
    write_json(STAGE0 / "stage0_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

