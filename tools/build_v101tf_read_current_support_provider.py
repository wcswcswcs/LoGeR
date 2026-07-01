#!/usr/bin/env python3
"""Build ACL2 v101 DH4 READ current-support provider diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
OUT = ROOT / "trackDH4_read_current_support_refresh_provider"
TRACK_U = ROOT / "trackU_true_current_support"
TRACK_W = ROOT / "trackW_anchor_memory_role"
TRACK_S2 = ROOT / "trackS2_anchor_state_estimator"
TRACK_T = ROOT / "trackT_drift_target_relabel"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def quantile(values: list[Any], q: float) -> float:
    vals = sorted(f(v) for v in values if math.isfinite(f(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id", "")), str(row.get("anchor_id", ""))


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        k = key(row)
        if k[0] and k[1] and k not in out:
            out[k] = row
    return out


def main() -> None:
    support_rows = by_key(read_rows(TRACK_U / "anchor_current_support_rows.csv"))
    role_rows = by_key(read_rows(TRACK_W / "anchor_role_rows.csv"))
    state_rows = by_key(read_rows(TRACK_S2 / "anchor_state_rows.csv"))
    target_rows = {row.get("case_id", ""): row for row in read_rows(TRACK_T / "target_universe_v101.csv")}
    rows: list[dict[str, Any]] = []
    for k, support in sorted(support_rows.items()):
        case_id, anchor_id = k
        role = role_rows.get(k, {})
        state = state_rows.get(k, {})
        target = target_rows.get(case_id, {})
        s_read = f(support.get("S_read"))
        s_feat = f(support.get("S_feat"))
        read_support = mean([s_read, s_feat])
        rows.append(
            {
                "case_id": case_id,
                "boundary_id": support.get("boundary_id", ""),
                "anchor_id": anchor_id,
                "semantic_label": support.get("semantic_label", ""),
                "target_taxonomy": target.get("target_taxonomy", support.get("target_taxonomy", "")),
                "S_read": support.get("S_read", ""),
                "S_feat": support.get("S_feat", ""),
                "READ_current_support_proxy": read_support,
                "S_cur_combined": support.get("S_cur_combined", ""),
                "R_same": support.get("R_same", ""),
                "query_hit_max": support.get("query_hit_max", ""),
                "role": role.get("role", ""),
                "allowed_READ_behavior": role.get("allowed_READ_behavior", ""),
                "state_status": state.get("state_status", ""),
                "K_anchor": state.get("K_anchor", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", support.get("L3_handoff_transfer_penalty_proxy", "")),
                "provider_claim": "READ same-space hidden/current support provider only",
                "runtime_action_allowed": False,
            }
        )

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    case_rows: list[dict[str, Any]] = []
    for case_id, parts in sorted(by_case.items()):
        target = target_rows.get(case_id, {})
        case_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
                "anchor_count": len(parts),
                "READ_current_support_mean": mean([row["READ_current_support_proxy"] for row in parts]),
                "S_read_mean": mean([row["S_read"] for row in parts]),
                "S_feat_mean": mean([row["S_feat"] for row in parts]),
                "context_read_frac": mean([1.0 if row["allowed_READ_behavior"] == "context_read" else 0.0 for row in parts]),
                "candidate_read_frac": mean([1.0 if row["allowed_READ_behavior"] == "read_as_candidate_evidence" else 0.0 for row in parts]),
            }
        )

    support_values = [row["READ_current_support_mean"] for row in case_rows]
    low_support_threshold = quantile(support_values, 0.25)
    safe_cases = [row for row in case_rows if row["target_taxonomy"] == "SAFE_GOOD"]
    handoff_cases = [row for row in case_rows if row["target_taxonomy"] == "HANDOFF_SCALE_GAUGE_TARGET"]
    safe_low_support_fpr = (
        sum(1 for row in safe_cases if f(row["READ_current_support_mean"]) <= low_support_threshold) / len(safe_cases)
        if safe_cases
        else math.nan
    )
    handoff_low_support_recall = (
        sum(1 for row in handoff_cases if f(row["READ_current_support_mean"]) <= low_support_threshold) / len(handoff_cases)
        if handoff_cases
        else math.nan
    )
    corr = pearson([row["READ_current_support_mean"] for row in case_rows], [row["L3_handoff_transfer_penalty_proxy"] for row in case_rows])
    role_counts = Counter(row["role"] for row in rows)
    gate = (
        len(case_rows) >= 28
        and math.isfinite(safe_low_support_fpr)
        and safe_low_support_fpr <= 0.30
        and math.isfinite(corr)
        and corr < 0.0
        and False
    )
    blockers = [
        "READ provider is same-space/proxy support only; no READ full action or refresh pilot was run.",
        "No baseline improvement over U/Q2 true-stage can be claimed because Q2 true-stage is blocked.",
        "Track T has only one clean HANDOFF target, so provider recall is not sequence-covered.",
    ]
    summary = {
        "schema": "acl2_v101_trackDH4_read_current_support_provider_v1",
        "status": "complete_diagnostic_provider_only",
        "gate_pass": gate,
        "provider_row_count": len(rows),
        "case_count": len(case_rows),
        "safe_good_case_count": len(safe_cases),
        "handoff_target_case_count": len(handoff_cases),
        "low_read_support_threshold_q25": low_support_threshold,
        "safe_good_low_read_support_fpr": safe_low_support_fpr,
        "handoff_low_read_support_recall": handoff_low_support_recall,
        "READ_current_support_mean_corr_L3": corr,
        "role_counts": dict(role_counts),
        "runtime_action_allowed": False,
        "blockers": blockers,
        "claim": "READ is evaluated only as a current-support provider; no READ runtime action or refresh success is claimed.",
    }
    write_rows(OUT / "read_provider_anchor_rows.csv", rows)
    write_rows(OUT / "read_provider_case_rows.csv", case_rows)
    write_json(OUT / "DH4_summary.json", summary)
    write_json(OUT / "blocked_summary.json", {**summary, "run_allowed": False})
    write_rows(
        OUT / "gate_checks.csv",
        [
            {"gate": "case_count_ge_28", "pass": len(case_rows) >= 28, "observed": len(case_rows)},
            {"gate": "safe_good_low_read_support_fpr_le_0p30", "pass": math.isfinite(safe_low_support_fpr) and safe_low_support_fpr <= 0.30, "observed": safe_low_support_fpr},
            {"gate": "READ_current_support_corr_L3_negative", "pass": math.isfinite(corr) and corr < 0.0, "observed": corr},
            {"gate": "Q2_true_stage_available", "pass": False, "observed": False},
            {"gate": "runtime_action_allowed", "pass": False, "observed": False},
        ],
    )
    write_rows(
        OUT / "not_run_manifest.csv",
        [
            {
                "track": "DH4",
                "not_run": False,
                "diagnostic_run": True,
                "runtime_action_run": False,
                "reason": "; ".join(blockers),
                "planned_outputs": "READ provider diagnostics and DH4_summary.json",
            }
        ],
    )
    write_text(
        OUT / "provider_report.md",
        "# Track DH4 READ Current-Support Provider\n\n"
        f"- Provider rows: {summary['provider_row_count']}\n"
        f"- Case count: {summary['case_count']}\n"
        f"- Safe-good low READ-support FPR: {summary['safe_good_low_read_support_fpr']}\n"
        f"- HANDOFF low READ-support recall: {summary['handoff_low_read_support_recall']}\n"
        f"- Corr(READ_current_support_mean, L3): {summary['READ_current_support_mean_corr_L3']}\n"
        f"- Gate pass: {summary['gate_pass']}\n\n"
        "Blockers:\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
    )
    write_text(OUT / "failure_report.md", "\n".join(f"- {item}" for item in blockers))
    write_text(
        OUT / "what_would_have_to_be_true_to_pass.md",
        "READ provider must improve U/Q2 diagnostics without READ full action, and Q2 true-stage must be available for comparison.",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
