#!/usr/bin/env python3
"""Audit v81S true SWA action-route smokes.

This script only summarizes existing v78 Phase9 runner outputs reused by v81S.
It does not recompute trajectories or promote the old v78 gate as a v81S method
gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_action_route_smoke"
)
DEFAULT_CASE_BANK = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS2_swa_good_bad_pair_bank/swa_good_bad_pair_bank.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS5_swa_action_route_audit"
)

CANDIDATES = {
    "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST": {
        "short": "source_replace_stable_v",
        "control": "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
        "family": "swa_source_replace",
    },
    "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_LAST": {
        "short": "source_gate_role_negative_v",
        "control": "P9_7_SOURCE_GATE_ROLE_NEGATIVE_RANDOM_SAME_MASS_V_LAST",
        "family": "swa_source_gate",
    },
}
BASELINE = "P9_0_NATIVE"
METRIC_KEYS = {
    "head10_to_tail10_pose_sim3_rmse_m": "head_tail",
    "overlap3_to_future_pose_sim3_rmse_m": "future_after_overlap",
    "scale_cv_head_mid_tail_pose_sim3": "scale_cv",
    "local_sim3_ate_rmse_m": "local_ate",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _improvement(base: Any, cand: Any) -> float | None:
    b = _safe_float(base)
    c = _safe_float(cand)
    if b is None or c is None or abs(b) < 1e-12:
        return None
    return (b - c) / abs(b)


def _chunk_from_run_dir(value: Any) -> int | None:
    match = re.search(r"chunk(\d+)", str(value))
    return int(match.group(1)) if match else None


def _seq_from_root(root: Path) -> str:
    match = re.match(r"seq(\d\d)", root.name)
    return match.group(1) if match else ""


def _load_case_bank(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (str(row.get("seq", "")), int(row.get("curr_chunk", "")))
            except ValueError:
                continue
            out[key] = dict(row)
    return out


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple, set)):
                    out[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _rows_by_run(runs: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in runs:
        chunk = _chunk_from_run_dir(row.get("run_dir"))
        run = str(row.get("run", ""))
        if chunk is not None and run:
            out[(chunk, run)] = row
    return out


def _route_file_count(row: Mapping[str, Any]) -> int:
    run_dir = Path(str(row.get("run_dir", "")))
    return len(list((run_dir / "swa_overlap_feature_maps").glob("*.pt"))) if run_dir.is_dir() else 0


def _action_applied(row: Mapping[str, Any], family: str) -> bool:
    if family == "swa_source_replace":
        return bool(
            (_safe_float(row.get("phase9_swa_overlap_source_replace_applied_sum")) or 0.0) > 0.0
            and (_safe_float(row.get("phase9_swa_mean_overlap_source_replace_alpha")) or 0.0) > 0.0
        )
    if family == "swa_source_gate":
        return bool(
            (_safe_float(row.get("phase9_swa_overlap_source_gate_applied_sum")) or 0.0) > 0.0
            and (_safe_float(row.get("phase9_swa_mean_overlap_source_gate_delta")) or 0.0) > 0.0
        )
    return False


def _summarize_root(root: Path, case_bank: Mapping[tuple[str, int], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = _read_json(root / "phase9_swa_cache_value_metrics.json")
    decision = _read_json(root / "phase9_swa_cache_value_decision.json")
    seq = _seq_from_root(root)
    runs = metrics.get("runs") if isinstance(metrics.get("runs"), list) else []
    by_run = _rows_by_run(runs)
    chunks = sorted({chunk for chunk, _ in by_run})
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        base = by_run.get((chunk, BASELINE), {})
        bank_row = case_bank.get((seq, chunk), {})
        for candidate, spec in CANDIDATES.items():
            cand = by_run.get((chunk, candidate), {})
            control = by_run.get((chunk, str(spec["control"])), {})
            if not cand:
                rows.append(
                    {
                        "root": str(root),
                        "seq": seq,
                        "chunk": chunk,
                        "case_type": bank_row.get("case_type", "unknown"),
                        "candidate": candidate,
                        "candidate_short": spec["short"],
                        "missing_candidate": True,
                    }
                )
                continue
            metric_payload: dict[str, dict[str, Any]] = {}
            any_bad_metric_signal = False
            for metric_key, short in METRIC_KEYS.items():
                imp = _improvement(base.get(metric_key), cand.get(metric_key))
                control_value = _safe_float(control.get(metric_key))
                cand_value = _safe_float(cand.get(metric_key))
                beats_control = bool(cand_value is not None and control_value is not None and cand_value < control_value)
                threshold = 0.10 if short in {"future_after_overlap", "scale_cv"} else 0.05
                metric_pass = bool(imp is not None and imp >= threshold and beats_control)
                any_bad_metric_signal = bool(any_bad_metric_signal or metric_pass)
                metric_payload[short] = {
                    "baseline": _safe_float(base.get(metric_key)),
                    "candidate": cand_value,
                    "control": control_value,
                    "improvement_vs_baseline_ratio": imp,
                    "beats_control": beats_control,
                    "v81s_metric_threshold": threshold,
                    "v81s_metric_pass": metric_pass,
                }
            route_files = _route_file_count(cand)
            action_applied = _action_applied(cand, str(spec["family"]))
            rows.append(
                {
                    "root": str(root),
                    "seq": seq,
                    "chunk": chunk,
                    "case_type": bank_row.get("case_type", "unknown"),
                    "candidate": candidate,
                    "candidate_short": spec["short"],
                    "control": spec["control"],
                    "family": spec["family"],
                    "route_file_count": route_files,
                    "route_mask_present": route_files > 0,
                    "action_applied": action_applied,
                    "phase9_gate_pass": (decision.get("decisions") or {}).get(candidate, {}).get("phase9_gate_pass"),
                    "phase9_action_fidelity_pass": (decision.get("decisions") or {}).get(candidate, {}).get("action_fidelity_pass"),
                    "v81s_any_metric_signal": any_bad_metric_signal,
                    "metrics": metric_payload,
                    "head_tail_improvement_vs_baseline_ratio": metric_payload["head_tail"]["improvement_vs_baseline_ratio"],
                    "future_after_overlap_improvement_vs_baseline_ratio": metric_payload["future_after_overlap"]["improvement_vs_baseline_ratio"],
                    "scale_cv_improvement_vs_baseline_ratio": metric_payload["scale_cv"]["improvement_vs_baseline_ratio"],
                    "head_tail_beats_control": metric_payload["head_tail"]["beats_control"],
                    "future_after_overlap_beats_control": metric_payload["future_after_overlap"]["beats_control"],
                    "scale_cv_beats_control": metric_payload["scale_cv"]["beats_control"],
                }
            )
    root_summary = {
        "root": str(root),
        "seq": seq,
        "metrics_exists": bool(metrics),
        "decision_exists": bool(decision),
        "phase9_any_gate_pass": decision.get("phase9_any_gate_pass"),
        "chunk_count": len(chunks),
        "candidate_rows": len(rows),
        "route_file_count": sum(int(row.get("route_file_count") or 0) for row in rows),
    }
    return rows, root_summary


def _median(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if _safe_float(v) is not None]
    return float(median(vals)) if vals else None


def _candidate_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    cr = [row for row in rows if row.get("candidate") == candidate and not row.get("missing_candidate")]
    bad = [row for row in cr if row.get("case_type") == "bad"]
    good = [row for row in cr if row.get("case_type") == "good"]
    good_worse_ratios = []
    for row in good:
        for key in ("head_tail_improvement_vs_baseline_ratio", "future_after_overlap_improvement_vs_baseline_ratio"):
            value = _safe_float(row.get(key))
            if value is not None:
                good_worse_ratios.append(-value)
    return {
        "candidate": candidate,
        "candidate_short": CANDIDATES[candidate]["short"],
        "row_count": len(cr),
        "bad_row_count": len(bad),
        "good_row_count": len(good),
        "seq_coverage": sorted({str(row.get("seq")) for row in cr if row.get("seq")}),
        "route_mask_rows": sum(1 for row in cr if row.get("route_mask_present")),
        "action_applied_rows": sum(1 for row in cr if row.get("action_applied")),
        "phase9_gate_pass_rows": sum(1 for row in cr if row.get("phase9_gate_pass")),
        "v81s_metric_signal_rows": sum(1 for row in cr if row.get("v81s_any_metric_signal")),
        "bad_head_tail_median_improvement_vs_baseline_ratio": _median(row.get("head_tail_improvement_vs_baseline_ratio") for row in bad),
        "bad_future_after_overlap_median_improvement_vs_baseline_ratio": _median(row.get("future_after_overlap_improvement_vs_baseline_ratio") for row in bad),
        "bad_scale_cv_median_improvement_vs_baseline_ratio": _median(row.get("scale_cv_improvement_vs_baseline_ratio") for row in bad),
        "good_max_worsen_ratio_over_head_future": max(good_worse_ratios) if good_worse_ratios else None,
        "good_no_worse_le_2pct": bool(good_worse_ratios and max(good_worse_ratios) <= 0.02),
        "phaseS5_candidate_gate_pass": False,
        "candidate_gate_blocker": (
            "route/action fidelity is present, but no candidate satisfies v81S bad improvement thresholds "
            "while preserving good cases and beating controls."
        ),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ACL2 v81S SWA Action Route Audit",
        "",
        "## Decision",
        "",
        f"- phaseS3_true_route_smoke_present: `{summary['gate']['phaseS3_true_route_smoke_present']}`",
        f"- phaseS5_action_fidelity_pass: `{summary['gate']['phaseS5_action_fidelity_pass']}`",
        f"- phaseS5_geometry_metric_gate_pass: `{summary['gate']['phaseS5_geometry_metric_gate_pass']}`",
        f"- phaseS5_gate_pass: `{summary['gate']['phaseS5_gate_pass']}`",
        f"- decision: `{summary['decision']}`",
        "",
        "## Candidate Summary",
        "",
    ]
    for cand in summary["candidate_summaries"]:
        lines.append(
            "- {short}: rows=`{rows}`, bad=`{bad}`, good=`{good}`, action_applied=`{applied}`, "
            "metric_signal=`{signal}`, bad_head_median=`{head}`, bad_future_median=`{future}`, "
            "bad_scale_median=`{scale}`, good_max_worse=`{good_worse}`".format(
                short=cand["candidate_short"],
                rows=cand["row_count"],
                bad=cand["bad_row_count"],
                good=cand["good_row_count"],
                applied=cand["action_applied_rows"],
                signal=cand["v81s_metric_signal_rows"],
                head=cand["bad_head_tail_median_improvement_vs_baseline_ratio"],
                future=cand["bad_future_after_overlap_median_improvement_vs_baseline_ratio"],
                scale=cand["bad_scale_cv_median_improvement_vs_baseline_ratio"],
                good_worse=cand["good_max_worsen_ratio_over_head_future"],
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- root: `{summary['root']}`",
            f"- row_csv: `{summary['row_csv']}`",
            f"- root_csv: `{summary['root_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-bank", type=Path, default=DEFAULT_CASE_BANK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    case_bank = _load_case_bank(args.case_bank)
    roots = sorted(path for path in args.root.glob("seq*_v78runner_smoke") if path.is_dir())
    route_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    for root in roots:
        rows, root_summary = _summarize_root(root, case_bank)
        route_rows.extend(rows)
        root_rows.append(root_summary)

    out_json = args.out_dir / "swa_action_route_audit_summary.json"
    out_rows = args.out_dir / "swa_action_route_rows.csv"
    out_roots = args.out_dir / "swa_action_route_roots.csv"
    out_report = args.out_dir / "swa_action_route_audit_report.md"

    complete_roots = [row for row in root_rows if row.get("metrics_exists") and row.get("decision_exists")]
    candidate_rows = [row for row in route_rows if not row.get("missing_candidate")]
    action_fidelity_pass = bool(candidate_rows and all(row.get("route_mask_present") and row.get("action_applied") for row in candidate_rows))
    geometry_metric_pass = any(row.get("v81s_any_metric_signal") for row in candidate_rows)
    candidate_summaries = [_candidate_summary(route_rows, candidate) for candidate in CANDIDATES]
    summary = {
        "schema": "acl2_v81s_swa_action_route_audit_v1",
        "root": str(args.root),
        "case_bank": str(args.case_bank),
        "out_dir": str(args.out_dir),
        "row_csv": str(out_rows),
        "root_csv": str(out_roots),
        "report": str(out_report),
        "root_count": len(roots),
        "complete_root_count": len(complete_roots),
        "route_row_count": len(candidate_rows),
        "route_mask_row_count": sum(1 for row in candidate_rows if row.get("route_mask_present")),
        "route_file_count": sum(int(row.get("route_file_count") or 0) for row in candidate_rows),
        "seq_coverage": sorted({str(row.get("seq")) for row in candidate_rows if row.get("seq")}),
        "bad_row_count": sum(1 for row in candidate_rows if row.get("case_type") == "bad"),
        "good_row_count": sum(1 for row in candidate_rows if row.get("case_type") == "good"),
        "phase9_any_gate_pass_roots": sum(1 for row in root_rows if row.get("phase9_any_gate_pass")),
        "candidate_summaries": candidate_summaries,
        "gate": {
            "route_smoke_roots_complete": bool(roots and len(complete_roots) == len(roots)),
            "phaseS3_true_route_smoke_present": any(row.get("route_mask_present") for row in candidate_rows),
            "phaseS5_action_fidelity_pass": action_fidelity_pass,
            "phaseS5_geometry_metric_gate_pass": geometry_metric_pass,
            "phaseS5_good_protection_present": any(summary_row["good_row_count"] > 0 for summary_row in candidate_summaries),
            "phaseS5_gate_pass": False,
        },
        "decision": "No-Go_swa_route_action_fidelity_present_but_geometry_gate_failed",
        "notes": [
            "These are v78 Phase9 SWA source replace/gate actions reused as v81S route-smoke evidence, not the full SWA7 method.",
            "A real route map/action signal is present, but metric improvements are too small or fail controls.",
            "Per v81S failure handling, the next path is merge/gauge fallback or rediscovery, not alpha sweeping.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_rows, route_rows)
    _write_csv(out_roots, root_rows)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_report.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
