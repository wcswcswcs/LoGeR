#!/usr/bin/env python3
"""Lock C9 memory-behavior evidence for ACL2 v76-TF Phase 0."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (
    C9_REPEAT_DIR,
    PREPROCESS_ROOT,
    V45_CLEAN_REGISTRY,
    V45_INTERACTION,
    V45_LEDGER,
    V45_ROOT,
    V46B_REGISTRY,
    V52_ROOT,
    V64_ROOT,
    V65_ROOT,
    V74_ROOT,
    V76_ROOT,
    artifact_status,
    boolish,
    count_nonzero_csv,
    ensure_dir,
    first_row,
    read_csv,
    read_json,
    read_text,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


COMPONENT_NOTES = {
    "tri_replay": "Exact C9 write-side tri-replay removal delta. Treat as C9-specific knockout evidence, not clean positive-only proof.",
    "tri_gamma_chunk_map": "Exact C9 chunk gamma map delta. Chunk-id policy is forbidden for v76 deployment.",
    "commit_ema": "Exact C9 chunk-specific commit EMA delta. Needs training-free global/runtime bridge before deployment.",
    "native_mix": "Exact C9 native-mix delta.",
    "swa_overlap_replace": "Exact C9 SWA overlap-source replacement delta.",
    "read_beta_map": "Exact C9 read beta chunk map delta. Chunk-id policy is forbidden for v76 deployment.",
    "fixed_tri_gamma_best": "Best fixed gamma substitute in v45 C9-clean rows.",
    "fixed_ema_best": "Best fixed/global EMA substitute in v45 C9-clean rows.",
    "fixed_read_beta": "Fixed read beta substitute in v45 C9-clean rows.",
}


FORBIDDEN_CLAIMS = [
    "C9 is a deployable general policy before removing absolute chunk-id dependence.",
    "C9 knockout rows answer clean H35 positive-only component effects.",
    "v64 adaptive TTT negatives disprove C9 tri-replay.",
    "Semantic residual lambda sweeps solve v76 without an active tri-replay actuator.",
    "KITTI09 prefix artifacts are full KITTI09 validation.",
    "Any GT-oracle or chunk-tuned decision rule is training-free deployment evidence.",
]


def _metric_summary() -> Dict[str, Any]:
    clean_rows = read_csv(V45_CLEAN_REGISTRY)
    v46b_rows = read_csv(V46B_REGISTRY)
    c9 = first_row(clean_rows, "name", "F0")
    h35 = first_row(v46b_rows, "row", "F000_NONE")
    return {
        "c9_repeat": {
            "source": rel(V45_CLEAN_REGISTRY),
            "row": "F0",
            "available": c9 is not None,
            "ATE_full": safe_float(c9.get("ATE_full")) if c9 else None,
            "frames": safe_float(c9.get("frames")) if c9 else None,
            "hmc_rows": safe_float(c9.get("hmc_rows")) if c9 else None,
            "status": c9.get("status") if c9 else None,
        },
        "h35_clean": {
            "source": rel(V46B_REGISTRY),
            "row": "F000_NONE",
            "available": h35 is not None,
            "ATE_full": safe_float(h35.get("ATE_full")) if h35 else None,
            "frames": safe_float(h35.get("frames")) if h35 else None,
            "hmc_rows": safe_float(h35.get("hmc_rows")) if h35 else None,
            "status": h35.get("status") if h35 else None,
            "no_chunk_policy_pass": boolish(h35.get("no_chunk_policy_pass")) if h35 else False,
        },
    }


def _component_rows() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in read_csv(V45_LEDGER):
        component = str(row.get("component") or "")
        out.append({
            "component": component,
            "effect_delta_vs_C9": row.get("effect_delta_vs_C9"),
            "source_artifact": rel(V45_LEDGER),
            "evidence_kind": "exact_C9_knockout_or_substitute",
            "v76_use": "prioritize actuator/bridge audit" if component in {"tri_replay", "tri_gamma_chunk_map", "commit_ema"} else "supporting prior",
            "note": COMPONENT_NOTES.get(component, ""),
        })
    return out


def _cache_status(seq: str) -> Dict[str, Any]:
    semantic_dir = PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks"
    radio_candidates = sorted((PREPROCESS_ROOT / seq).glob("radio_sidecar_chunks*"))
    return {
        "sequence": seq,
        "semantic_cache": rel(semantic_dir),
        "semantic_cache_exists": semantic_dir.exists(),
        "semantic_cache_file_count": sum(1 for p in semantic_dir.iterdir() if p.is_file()) if semantic_dir.exists() else 0,
        "radio_sidecar_dirs": [rel(path) for path in radio_candidates],
        "radio_sidecar_dir_count": len(radio_candidates),
    }


def _extract_config_lines(path: Path) -> List[str]:
    text = read_text(path)
    hits: List[str] = []
    needles = (
        "TTT_WRITE",
        "READ_BETA",
        "SWA_OVERLAP",
        "COMMIT_EMA",
        "tri_replay",
        "commit_ema",
        "gradient_reversal",
        "read_beta",
    )
    for line in text.splitlines():
        if any(needle in line for needle in needles):
            hits.append(line.strip())
    return hits[:120]


def _registry_prior(path: Path, label: str) -> Dict[str, Any]:
    rows = read_csv(path)
    ate_values = [safe_float(row.get("ATE_full")) for row in rows]
    ate_values = [value for value in ate_values if value is not None]
    return {
        "prior": label,
        "source_artifact": rel(path),
        "available": path.exists(),
        "row_count": len(rows),
        "best_ATE_full": min(ate_values) if ate_values else None,
        "worst_ATE_full": max(ate_values) if ate_values else None,
        "note": "Imported historical diagnostic; not a v76 current-run metric.",
    }


def _semantic_prior_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in (
        ("v45_sem1_vs_c9", V45_ROOT / "phase5_semantic_read/report_R1/sem1_vs_c9/full_online_registry.csv"),
        ("v45_sem2_vs_c9clean", V45_ROOT / "phase5_semantic_read/report_R1/sem2_vs_c9clean/full_online_registry.csv"),
        ("v45_sem4_vs_c9", V45_ROOT / "phase5_semantic_read/report_R1/sem4_vs_c9/full_online_registry.csv"),
        ("v52_c9_attribution", V52_ROOT / "positive_only_factorial_table.csv"),
    ):
        rows.append(_registry_prior(path, label))
    for label, path in (
        ("v52_summary", V52_ROOT / "phase1_attribution_summary.json"),
        ("v64_summary", V64_ROOT / "v64_summary.json"),
        ("v65_summary", V65_ROOT / "v65_summary.json"),
        ("v74_final_report", V74_ROOT / "report_final/v74tf_final_report.md"),
        ("v74_no_go_report", V74_ROOT / "report_final/v74tf_no_go_report.md"),
    ):
        status = artifact_status(path)
        rows.append({
            "prior": label,
            "source_artifact": status["path"],
            "available": status["exists"],
            "row_count": "",
            "best_ATE_full": "",
            "worst_ATE_full": "",
            "note": "Imported historical report/summary; inspect source for detailed fields.",
        })
    return rows


def _write_chunk_policy_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    audit = read_json(C9_REPEAT_DIR / "chunk_id_policy_audit.json")
    config_lines = _extract_config_lines(C9_REPEAT_DIR / "effective_config.yaml")
    lines = [
        "# v76 Phase 0 C9 Chunk-Id Policy Audit",
        "",
        "This file is generated from existing C9 repeat artifacts. It is an audit lock, not a new experiment.",
        "",
        "## Artifact",
        "",
        f"- C9 repeat dir: `{rel(C9_REPEAT_DIR)}`",
        f"- chunk audit JSON exists: `{(C9_REPEAT_DIR / 'chunk_id_policy_audit.json').exists()}`",
        f"- effective config exists: `{(C9_REPEAT_DIR / 'effective_config.yaml').exists()}`",
        "",
        "## Deployment Boundary",
        "",
        "- Absolute chunk-id maps from C9 are historical diagnosis only.",
        "- v76 deployment candidates must use fixed training-free semantic/runtime rules.",
        "- Any row using chunk-specific C9 maps must be reported as C9 experience, not as a deployable v76 rule.",
        "",
        "## Gate Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Config Lines", ""])
    if config_lines:
        lines.extend(f"- `{line}`" for line in config_lines)
    else:
        lines.append("- No matching C9/TTT/read/SWA config lines found in effective_config.yaml.")
    lines.extend(["", "## Raw Chunk Audit", "", "```json"])
    lines.append(str(audit if audit is not None else {}))
    lines.extend(["```", ""])
    write_text(out_dir / "c9_chunk_id_policy_audit.md", "\n".join(lines))


def run(out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    metrics = _metric_summary()
    component_rows = _component_rows()
    semantic_prior_rows = _semantic_prior_rows()
    cache_rows = [_cache_status("01"), _cache_status("09")]
    config_lines = _extract_config_lines(C9_REPEAT_DIR / "effective_config.yaml")
    post_zp = count_nonzero_csv(C9_REPEAT_DIR / "post_zeropower_delta_norm.csv")

    c9_knockout_available = bool(component_rows) and V45_INTERACTION.exists()
    semantic_cache_available = all(row["semantic_cache_exists"] and row["semantic_cache_file_count"] > 0 for row in cache_rows)
    ttt_config_recoverable = bool(config_lines) and any("tri_replay" in line or "TTT_WRITE" in line for line in config_lines)
    summary = {
        "c9_locked_repeat_metric_available": bool(metrics["c9_repeat"]["available"]),
        "h35_clean_metric_available": bool(metrics["h35_clean"]["available"]),
        "c9_component_knockout_rows_available": c9_knockout_available,
        "dense_semantic_cache_available": semantic_cache_available,
        "ttt_tri_replay_configs_recoverable": ttt_config_recoverable,
        "phase0_gate_pass": bool(
            metrics["c9_repeat"]["available"]
            and metrics["h35_clean"]["available"]
            and c9_knockout_available
            and semantic_cache_available
            and ttt_config_recoverable
        ),
    }
    usable = {
        "metrics": metrics,
        "artifact_status": {
            "c9_repeat_dir": artifact_status(C9_REPEAT_DIR, kind="directory"),
            "v45_component_ledger": artifact_status(V45_LEDGER),
            "v45_interaction_matrix": artifact_status(V45_INTERACTION),
            "v46b_registry": artifact_status(V46B_REGISTRY),
            "v52_summary": artifact_status(V52_ROOT / "phase1_attribution_summary.json"),
            "v64_summary": artifact_status(V64_ROOT / "v64_summary.json"),
            "v65_summary": artifact_status(V65_ROOT / "v65_summary.json"),
            "v74_final_report": artifact_status(V74_ROOT / "report_final/v74tf_final_report.md"),
        },
        "semantic_cache_status": cache_rows,
        "post_zeropower_delta_norm": post_zp,
        "phase0_gate": summary,
    }

    write_csv(out_dir / "c9_component_necessity.csv", component_rows)
    write_csv(out_dir / "semantic_read_ttt_combo_prior.csv", semantic_prior_rows)
    write_csv(out_dir / "semantic_cache_status.csv", cache_rows)
    write_json(out_dir / "usable_artifacts.json", usable)
    write_json(out_dir / "phase0_summary.json", summary)
    write_text(out_dir / "forbidden_claims.md", "# v76 Forbidden Claims\n\n" + "\n".join(f"- {claim}" for claim in FORBIDDEN_CLAIMS) + "\n")
    _write_chunk_policy_report(out_dir, summary)
    return {"out_dir": rel(out_dir), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase0_c9_experience_lock"))
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)
    if not result["phase0_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
