#!/usr/bin/env python3
"""Aggregate ACL2 v76-TF semantic tri-replay/read/EMA/SWA evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (  # noqa: E402
    V45_ROOT,
    V74_ROOT,
    V76_ROOT,
    boolish,
    ensure_dir,
    first_row,
    read_csv,
    read_json,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


SEM4_REGISTRY = V45_ROOT / "phase5_semantic_read/report_R1/sem4_vs_c9/full_online_registry.csv"
SEM4_CHUNK_AUDIT = V45_ROOT / "phase3_c23_support/rollouts/V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST/chunk_id_policy_audit.json"
TTT_SMOKE_SUMMARY = V74_ROOT / "phase5_harmful_no_persistent_ttt_dynamic_lowstable_top4/ttt_write_online_smoke_summary.json"
READ_SMOKE_SUMMARY = V74_ROOT / "report_final/phase4_extra_nA_online_read_smoke_top4_lam010_all_beta010/radio_read_online_smoke_summary.json"
SWA_SMOKE_SUMMARY = V74_ROOT / "phase5_component_leave_one_out_swa_turnoff_top4/radio_swa_online_smoke_summary.json"
PHASE6_PRECHECK = V74_ROOT / "report_final/phase6_online_controller_smoke_after_ttt_no_persistent_phase5_repair/online_smoke_precheck.json"


def _json_dict(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _sem4_row() -> Dict[str, Any]:
    rows = read_csv(SEM4_REGISTRY)
    f0 = first_row(rows, "name", "F0")
    sem1 = first_row(rows, "name", "SEM1")
    sem4 = first_row(rows, "name", "SEM4")
    audit = _json_dict(SEM4_CHUNK_AUDIT)
    f0_ate = safe_float(f0.get("ATE_full")) if f0 else None
    sem4_ate = safe_float(sem4.get("ATE_full")) if sem4 else None
    sem1_ate = safe_float(sem1.get("ATE_full")) if sem1 else None
    has_chunk_policy = any(
        boolish(audit.get(key))
        for key in (
            "has_read_beta_frame_chunks",
            "has_tri_gamma_chunk_map",
            "has_tri_replay_chunk_params",
            "has_commit_ema_chunks",
        )
    )
    return {
        "phase": "phase3_semantic_tri_replay",
        "artifact": rel(SEM4_REGISTRY),
        "candidate": "V45_SEM4_C23_RESID_PLUS_ADAPTIVE_TRI_BEST",
        "available": sem4 is not None,
        "hook_or_action_active": bool(sem4 is not None),
        "gate_pass": bool(sem4 is not None and f0_ate is not None and sem4_ate is not None and f0_ate - sem4_ate >= 0.3 and not has_chunk_policy),
        "historical_positive_gain_m": f0_ate - sem4_ate if f0_ate is not None and sem4_ate is not None else None,
        "read_only_gain_m": f0_ate - sem1_ate if f0_ate is not None and sem1_ate is not None else None,
        "candidate_ATE_full": sem4_ate,
        "reference_ATE_full": f0_ate,
        "chunk_policy_free": not has_chunk_policy,
        "reason": "SEM4 improves C9 historical full-run ATE, but uses C9 chunk maps, so it is not a deployable v76 semantic trigger.",
    }


def _smoke_row(phase: str, path: Path, gate_key: str, candidate_key: str) -> Dict[str, Any]:
    payload = _json_dict(path)
    details = payload.get("chunk_details")
    best_improvement = None
    worst_improvement = None
    if isinstance(details, list):
        improvements = [
            safe_float(item.get("candidate_improvement_m"))
            for item in details
            if isinstance(item, dict)
        ]
        improvements = [value for value in improvements if value is not None]
        if improvements:
            best_improvement = max(improvements)
            worst_improvement = min(improvements)
    return {
        "phase": phase,
        "artifact": rel(path),
        "candidate": payload.get("phase", candidate_key),
        "available": path.exists(),
        "hook_or_action_active": bool(payload.get("candidate_hook_active_chunks")),
        "gate_pass": bool(payload.get(gate_key)),
        "rows": payload.get("rows"),
        "candidate_chunks": ",".join(str(x) for x in payload.get("candidate_chunks", [])) if isinstance(payload.get("candidate_chunks"), list) else "",
        "candidate_hook_active_chunks": ",".join(str(x) for x in payload.get("candidate_hook_active_chunks", [])) if isinstance(payload.get("candidate_hook_active_chunks"), list) else "",
        "candidate_pass_chunks": ",".join(str(x) for x in payload.get("candidate_pass_chunks", [])) if isinstance(payload.get("candidate_pass_chunks"), list) else "",
        "best_candidate_improvement_m": best_improvement,
        "worst_candidate_improvement_m": worst_improvement,
        "gate_rule": payload.get("gate_rule"),
    }


def _phase6_row() -> Dict[str, Any]:
    payload = _json_dict(PHASE6_PRECHECK)
    return {
        "phase": "phase6_online_controller_guard",
        "artifact": rel(PHASE6_PRECHECK),
        "candidate": payload.get("action_family", "phase6_guard"),
        "available": PHASE6_PRECHECK.exists(),
        "hook_or_action_active": payload.get("enable_v74tf_memory_control"),
        "gate_pass": bool(payload.get("gate_pass")),
        "online_smoke_precondition_pass": bool(payload.get("online_smoke_precondition_pass")),
        "status": payload.get("status"),
        "reason": payload.get("note"),
    }


def run(out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = [
        _sem4_row(),
        _smoke_row("phase3_semantic_ttt_write_controls", TTT_SMOKE_SUMMARY, "ttt_write_online_gate_pass", "v74_ttt_write_smoke"),
        _smoke_row("phase4_read_ttt_handshake_controls", READ_SMOKE_SUMMARY, "read_online_gate_pass", "v74_read_smoke"),
        _smoke_row("phase6_swa_tri_handoff_controls", SWA_SMOKE_SUMMARY, "swa_online_gate_pass", "v74_swa_smoke"),
        _phase6_row(),
    ]
    by_phase = {str(row["phase"]): row for row in rows}
    summary = {
        "phase3_historical_sem4_positive_but_chunk_mapped": bool(
            safe_float(by_phase["phase3_semantic_tri_replay"].get("historical_positive_gain_m")) is not None
            and safe_float(by_phase["phase3_semantic_tri_replay"].get("historical_positive_gain_m")) >= 0.3
            and not by_phase["phase3_semantic_tri_replay"].get("chunk_policy_free")
        ),
        "phase3_semantic_tri_replay_gate_pass": bool(by_phase["phase3_semantic_tri_replay"].get("gate_pass"))
        or bool(by_phase["phase3_semantic_ttt_write_controls"].get("gate_pass")),
        "phase4_read_ttt_handshake_gate_pass": bool(by_phase["phase4_read_ttt_handshake_controls"].get("gate_pass")),
        "phase5_commit_ema_bridge_gate_pass": False,
        "phase5_commit_ema_bridge_reason": "C9 commit EMA knockout is positive in Phase0/2, but no training-free semantic commit-EMA bridge online gate passed in available artifacts.",
        "phase6_swa_tri_handoff_gate_pass": bool(by_phase["phase6_swa_tri_handoff_controls"].get("gate_pass")),
        "phase6_online_controller_guard_pass": bool(by_phase["phase6_online_controller_guard"].get("gate_pass")),
    }
    summary["phase3_to_phase6_gate_pass"] = bool(
        summary["phase3_semantic_tri_replay_gate_pass"]
        and summary["phase4_read_ttt_handshake_gate_pass"]
        and summary["phase5_commit_ema_bridge_gate_pass"]
        and summary["phase6_swa_tri_handoff_gate_pass"]
        and summary["phase6_online_controller_guard_pass"]
    )
    summary["primary_blocker"] = (
        "training-free semantic/RADIO online actions activate, but do not beat geometry/shuffle/random controls or Phase6 preconditions"
        if not summary["phase3_to_phase6_gate_pass"]
        else "none"
    )
    write_csv(out_dir / "phase3_6_semantic_control_evidence.csv", rows)
    write_json(out_dir / "phase3_6_semantic_control_summary.json", summary)
    _write_report(out_dir, rows, summary)
    return {"out_dir": rel(out_dir), **summary}


def _write_report(out_dir: Path, rows: List[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# v76 Phase 3-6 Semantic Control Evidence",
        "",
        "This report imports completed SEM4/v74/v70 artifacts and keeps historical positives separate from deployable gates.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Evidence Rows",
        "",
        "| phase | candidate | available | hook/action active | gate | key note |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in rows:
        note = row.get("reason") or row.get("gate_rule") or ""
        lines.append(
            f"| `{row.get('phase')}` | `{row.get('candidate')}` | {row.get('available')} | "
            f"{row.get('hook_or_action_active')} | {row.get('gate_pass')} | {note} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- SEM4 is useful C9-informed historical evidence but is chunk-map contaminated for v76 deployment.",
        "- TTT-write, READ, and SWA online smoke hooks activate, but none passes its control gate.",
        "- Phase6 online controller remains blocked by the plan precondition guard.",
        "",
    ])
    write_text(out_dir / "phase3_6_semantic_control_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase3_6_semantic_control_evidence"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)
    if args.strict and not result["phase3_to_phase6_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
