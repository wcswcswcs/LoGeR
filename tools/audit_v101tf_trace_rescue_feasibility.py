#!/usr/bin/env python3
"""Audit whether existing runtime traces can rescue v101 strict blockers.

The strict-action frontier says action is blocked by missing instance identity,
query/head controls, write/cache/current materialization, and Q2 true-stage.
This script checks the already-produced runtime probe traces for substitute
evidence.  It reports what is present and what remains unavailable; it does not
convert passthrough traces into action authorization.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            rows.append({"_line_no": line_no, "_read_error": f"{type(exc).__name__}:{exc}"})
            continue
        if isinstance(payload, dict):
            payload["_line_no"] = line_no
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key, "")) for key in keys})


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out


def hook_identity_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    native_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    for path in sorted(ROOT.glob("outcomeD_*/**/hmc_hook_identity_check.json")):
        payload = read_json(path)
        row = {
            "artifact": rel(path),
            "status": payload.get("status", ""),
            "chunk_attention_hook": payload.get("chunk_attention_hook", ""),
            "frame_attention_hook": payload.get("frame_attention_hook", ""),
            "swa_read_hook": payload.get("swa_read_hook", ""),
            "ttt_apply_hook": payload.get("ttt_apply_hook", ""),
            "identity_ttt_update": payload.get("identity_ttt_update", ""),
            "native_ttt_update": payload.get("native_ttt_update", ""),
            "identity_hook_paths_nonempty": bool(payload.get("identity_hook_paths")),
            "implemented_paths_nonempty": bool(payload.get("implemented_paths")),
            "claim_level": "hook_wiring_or_passthrough_only_no_instance_identity",
        }
        rows.append(row)
        status_counts[str(row["status"])] += 1
        native_counts[str(row["native_ttt_update"])] += 1
        identity_counts[str(row["identity_ttt_update"])] += 1
    summary = {
        "hook_identity_file_count": len(rows),
        "hook_identity_status_counts": dict(status_counts),
        "hook_identity_native_ttt_update_counts": dict(native_counts),
        "hook_identity_identity_ttt_update_counts": dict(identity_counts),
        "identity_hook_paths_nonempty_file_count": sum(1 for row in rows if row["identity_hook_paths_nonempty"]),
        "implemented_paths_nonempty_file_count": sum(1 for row in rows if row["implemented_paths_nonempty"]),
    }
    return rows, summary


def trace_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = Counter()
    for path in sorted(list(ROOT.glob("outcomeD_*/**/hmc_control_summary.jsonl")) + list(ROOT.glob("outcomeD_*/**/hook_effect_summary.jsonl"))):
        payload_rows = iter_jsonl(path)
        for payload in payload_rows:
            hook_summary = payload.get("hook_effect_summary")
            if hook_summary is None:
                hook_summary = payload.get("control_trace", {}).get("hook_effect_summary", {})
            site_counts = Counter()
            for site, metrics in hook_summary.items() if isinstance(hook_summary, dict) else []:
                if not isinstance(metrics, dict):
                    continue
                if metrics.get("attention_mass_available") is True:
                    site_counts["attention_mass_available_true_sites"] += 1
                site_counts["query_soft_applied_total"] += int(num(metrics.get("num_swa_prev_ttt_anchor_query_soft_applied")))
                site_counts["query_soft_selected_queries_max"] = max(
                    site_counts["query_soft_selected_queries_max"],
                    int(num(metrics.get("max_swa_prev_ttt_anchor_query_soft_selected_queries"))),
                )
                site_counts["source_gate_applied_total"] += int(num(metrics.get("num_source_gate_applied")))
                site_counts["stable_gate_applied_total"] += int(num(metrics.get("num_swa_prev_ttt_stable_anchor_gate_applied")))
                site_counts["semantic_boost_applied_total"] += int(num(metrics.get("num_semantic_anchor_boost_applied")))
                site_counts["attention_sites_seen"] += 1
            write_debug_available = payload.get("probe_ttt_write_debug_available") is True
            write_native_cosine_count = int(num(payload.get("probe_ttt_write_native_cosine_count")))
            implemented_paths_nonempty = bool(payload.get("implemented_paths"))
            identity_paths_nonempty = bool(payload.get("identity_hook_paths"))
            row = {
                "artifact": rel(path),
                "line_no": payload.get("_line_no", ""),
                "chunk_idx": payload.get("chunk_idx", ""),
                "hybrid_memory_mode": payload.get("hybrid_memory_mode", ""),
                "attention_sites_seen": site_counts["attention_sites_seen"],
                "attention_mass_available_true_sites": site_counts["attention_mass_available_true_sites"],
                "query_soft_applied_total": site_counts["query_soft_applied_total"],
                "query_soft_selected_queries_max": site_counts["query_soft_selected_queries_max"],
                "source_gate_applied_total": site_counts["source_gate_applied_total"],
                "stable_gate_applied_total": site_counts["stable_gate_applied_total"],
                "semantic_boost_applied_total": site_counts["semantic_boost_applied_total"],
                "write_debug_available": write_debug_available,
                "write_native_cosine_count": write_native_cosine_count,
                "implemented_paths_nonempty": implemented_paths_nonempty,
                "identity_hook_paths_nonempty": identity_paths_nonempty,
                "claim_level": "trace_summary_no_action_rescue",
            }
            rows.append(row)
            for key in [
                "attention_sites_seen",
                "attention_mass_available_true_sites",
                "query_soft_applied_total",
                "source_gate_applied_total",
                "stable_gate_applied_total",
                "semantic_boost_applied_total",
            ]:
                aggregate[key] += int(row[key])
            aggregate["trace_jsonl_line_count"] += 1
            aggregate["write_debug_available_line_count"] += int(write_debug_available)
            aggregate["write_native_cosine_positive_line_count"] += int(write_native_cosine_count > 0)
            aggregate["implemented_paths_nonempty_line_count"] += int(implemented_paths_nonempty)
            aggregate["identity_hook_paths_nonempty_line_count"] += int(identity_paths_nonempty)
    summary = {
        "trace_jsonl_file_count": len({row["artifact"] for row in rows}),
        "trace_jsonl_line_count": aggregate["trace_jsonl_line_count"],
        "trace_attention_sites_seen": aggregate["attention_sites_seen"],
        "attention_mass_available_true_site_count": aggregate["attention_mass_available_true_sites"],
        "query_soft_applied_total": aggregate["query_soft_applied_total"],
        "source_gate_applied_total": aggregate["source_gate_applied_total"],
        "stable_gate_applied_total": aggregate["stable_gate_applied_total"],
        "semantic_boost_applied_total": aggregate["semantic_boost_applied_total"],
        "write_debug_available_line_count": aggregate["write_debug_available_line_count"],
        "write_native_cosine_positive_line_count": aggregate["write_native_cosine_positive_line_count"],
        "implemented_paths_nonempty_line_count": aggregate["implemented_paths_nonempty_line_count"],
        "identity_hook_paths_nonempty_line_count": aggregate["identity_hook_paths_nonempty_line_count"],
    }
    return rows, summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v101 Trace Rescue Feasibility",
        "",
        "This audit checks whether existing outcomeD runtime traces can substitute for the strict-action prerequisites.",
        "It does not authorize action.",
        "",
        "## Summary",
        "",
        f"- hook_identity_file_count: {summary['hook_identity_file_count']}",
        f"- trace_jsonl_file_count: {summary['trace_jsonl_file_count']}",
        f"- trace_jsonl_line_count: {summary['trace_jsonl_line_count']}",
        f"- identity_hook_paths_nonempty_file_count: {summary['identity_hook_paths_nonempty_file_count']}",
        f"- implemented_paths_nonempty_file_count: {summary['implemented_paths_nonempty_file_count']}",
        f"- attention_mass_available_true_site_count: {summary['attention_mass_available_true_site_count']}",
        f"- query_soft_applied_total: {summary['query_soft_applied_total']}",
        f"- source_gate_applied_total: {summary['source_gate_applied_total']}",
        f"- stable_gate_applied_total: {summary['stable_gate_applied_total']}",
        f"- semantic_boost_applied_total: {summary['semantic_boost_applied_total']}",
        f"- write_debug_available_line_count: {summary['write_debug_available_line_count']}",
        f"- write_native_cosine_positive_line_count: {summary['write_native_cosine_positive_line_count']}",
        f"- trace_rescue_available: {summary['trace_rescue_available']}",
        "",
        "## Decision",
        "",
        "The traces prove read-path hook wiring and identity passthrough coverage, but they do not provide strict instance identity, query/head control margins, write/cache/current materialization, or Q2 true-stage evidence.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    hook_rows, hook_summary = hook_identity_rows()
    trace_detail_rows, trace_summary = trace_rows()
    strict = read_json(FINAL / "strict_action_frontier_summary.json")

    trace_rescue_available = (
        hook_summary["identity_hook_paths_nonempty_file_count"] > 0
        or hook_summary["implemented_paths_nonempty_file_count"] > 0
        or trace_summary["attention_mass_available_true_site_count"] > 0
        or trace_summary["query_soft_applied_total"] > 0
        or trace_summary["source_gate_applied_total"] > 0
        or trace_summary["stable_gate_applied_total"] > 0
        or trace_summary["semantic_boost_applied_total"] > 0
        or trace_summary["write_debug_available_line_count"] > 0
        or trace_summary["write_native_cosine_positive_line_count"] > 0
    )
    summary = {
        "schema": "acl2_v101_trace_rescue_feasibility_v1",
        **hook_summary,
        **trace_summary,
        "strict_frontier_missing_prereq_counts": strict.get("missing_prereq_counts", {}),
        "trace_rescue_available": trace_rescue_available,
        "strict_instance_identity_rescued": hook_summary["identity_hook_paths_nonempty_file_count"] > 0,
        "query_head_controls_rescued": trace_summary["query_soft_applied_total"] > 0
        or trace_summary["attention_mass_available_true_site_count"] > 0,
        "write_cache_current_chain_rescued": trace_summary["write_debug_available_line_count"] > 0
        or trace_summary["write_native_cosine_positive_line_count"] > 0,
        "q2_true_stage_rescued": False,
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "full_method_success": False,
        "blocked_reason": "Existing outcomeD traces are identity-passthrough/read-path diagnostics; they do not rescue strict instance identity, query-head controls, write/cache/current materialization, or Q2 true-stage.",
        "claim": "Trace rescue audit only; no runtime/full action is authorized.",
    }
    write_rows(FINAL / "trace_rescue_hook_identity_rows.csv", hook_rows)
    write_rows(FINAL / "trace_rescue_jsonl_rows.csv", trace_detail_rows)
    write_json(FINAL / "trace_rescue_feasibility_summary.json", summary)
    write_report(FINAL / "trace_rescue_feasibility_report.md", summary)
    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
