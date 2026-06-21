#!/usr/bin/env python3
"""Phase 4 fixed action-family oracle aggregation for ACL2 v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import load_json, read_csv, safe_float, utc_now, write_csv, write_json, write_text
from v74tf_common import REPORT_ROOT, median


def _split_paths(text: str) -> list[Path]:
    return [Path(part.strip()) for part in str(text).split(",") if part.strip()]


def _load_oracle_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv(path):
            row = dict(row)
            row["source_oracle_csv"] = str(path)
            rows.append(row)
    return rows


def _load_seq09_prefix_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, dict):
            out.append(
                {
                    "summary_json": str(path),
                    "source_label": payload.get("source_label"),
                    "pair_files": payload.get("pair_files"),
                    "rows": payload.get("rows"),
                    "oracle_action_gate_pass": bool(payload.get("oracle_action_gate_pass")),
                    "counts": payload.get("counts"),
                    "mean_best_mechanism_improvement": payload.get("mean_best_mechanism_improvement"),
                    "mean_raw_overlap_improvement_ratio": payload.get("mean_raw_overlap_improvement_ratio"),
                    "note": "prefix-only KITTI09 diagnostic; not full validation and not a deployment gate pass",
                }
            )
    return out


def _family(row: dict[str, Any]) -> str:
    filt = str(row.get("fit_semantic_filter", "")).lower()
    action = str(row.get("damped_action_family", row.get("action_family", ""))).lower()
    if any(key in filt for key in ("vertical_static", "ground_static", "stable")):
        return "U-A2_stable_evidence_refresh"
    if any(key in filt for key in ("dynamic", "sky", "vegetation", "void", "lowtrust")):
        return "O-A3_semantic_robust_kernel_diagnostic"
    if "se3" in action or "sim3" in action:
        return "U-A1_native_transform_amplification_proxy"
    return "unknown_action_family"


def _control_kind(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("source_label", "source_oracle_csv", "fit_semantic_filter")).lower()
    for key in ("label_shuffled", "confidence_shuffled", "same_weight", "same_spatial", "same_anchor", "random", "geometry"):
        if key in text:
            return key
    if "all" == str(row.get("fit_semantic_filter", "")).lower():
        return "geometry_or_all_points"
    return "candidate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-results", default="", help="Comma-separated overlap_pair_action_oracle_results.csv paths.")
    parser.add_argument("--seq09-prefix-oracle-summaries", default="", help="Optional comma-separated KITTI09 prefix oracle summary JSON paths.")
    parser.add_argument("--failure-mode-csv", type=Path, default=REPORT_ROOT / "phase3_failure_mode_rules" / "failure_mode_by_seq_chunk.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase4_action_family_oracle")
    args = parser.parse_args()

    paths = _split_paths(args.oracle_results)
    rows = _load_oracle_rows(paths)
    seq09_prefix_summaries = _load_seq09_prefix_summaries(_split_paths(args.seq09_prefix_oracle_summaries))
    mode_rows = read_csv(args.failure_mode_csv)
    mode_by_chunk = {}
    for row in mode_rows:
        if str(row.get("seq", "")).zfill(2) == "01":
            try:
                mode_by_chunk[int(row.get("chunk_id", -1))] = row.get("failure_mode_rule")
            except (TypeError, ValueError):
                pass
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            chunk = int(row.get("curr_chunk", row.get("chunk_id", -1)))
        except (TypeError, ValueError):
            chunk = -1
        result = dict(row)
        result["seq"] = "01"
        result["chunk_id"] = chunk
        result["failure_mode_rule"] = mode_by_chunk.get(chunk, "UNKNOWN")
        result["v74tf_action_family"] = _family(result)
        result["control_kind"] = _control_kind(result)
        result["positive_by_v74tf_gate"] = bool(str(result.get("oracle_action_gate_pass")).lower() == "true")
        result["J_scale_improvement_proxy"] = safe_float(result.get("best_mechanism_improvement"))
        out_rows.append(result)
    summary_rows: list[dict[str, Any]] = []
    families = sorted({row.get("v74tf_action_family", "unknown_action_family") for row in out_rows}) or ["no_direct_oracle_rows"]
    for family in families:
        fam_rows = [row for row in out_rows if row.get("v74tf_action_family") == family]
        candidate_rows = [row for row in fam_rows if row.get("control_kind") == "candidate"]
        if not candidate_rows and family == "no_direct_oracle_rows":
            candidate_rows = []
        positives = sorted({int(row.get("chunk_id")) for row in candidate_rows if row.get("positive_by_v74tf_gate") and str(row.get("chunk_id", "")).lstrip("-").isdigit()})
        control_rows = [row for row in fam_rows if row.get("control_kind") != "candidate"]
        median_imp = median(row.get("J_scale_improvement_proxy") for row in candidate_rows)
        control_median = median(row.get("J_scale_improvement_proxy") for row in control_rows)
        beats_controls = bool(control_rows and median_imp is not None and control_median is not None and median_imp > control_median)
        pass_01 = bool((len(positives) >= 4 or (median_imp is not None and median_imp >= 0.05)) and beats_controls)
        summary_rows.append(
            {
                "seq": "01",
                "v74tf_action_family": family,
                "candidate_rows": len(candidate_rows),
                "control_rows": len(control_rows),
                "positive_chunks": ",".join(str(x) for x in positives),
                "positive_chunk_count": len(positives),
                "median_J_scale_improvement_proxy": median_imp,
                "control_median_improvement_proxy": control_median,
                "beats_controls": beats_controls,
                "phase4_01_gate_pass": pass_01,
                "gate_note": "Requires >=4 positive chunks or median proxy >=5%, and beats available controls.",
            }
        )
    phase4_01 = any(row.get("phase4_01_gate_pass") for row in summary_rows)
    summary = {
        "schema": "acl2_v74tf_phase4_action_family_oracle_v1",
        "created_at": utc_now(),
        "oracle_result_paths": [str(path) for path in paths],
        "oracle_rows": len(out_rows),
        "family_rows": summary_rows,
        "phase4_01_gate_pass": phase4_01,
        "phase4_09_gate_pass": False,
        "phase4_09_prefix_diagnostic": seq09_prefix_summaries,
        "phase4_gate_pass": False,
        "blocked_reason": (
            "No direct oracle rows were supplied."
            if not out_rows
            else (
                "KITTI09 prefix action-family oracle did not pass deployment gate and remains prefix-only; controls/full validation missing."
                if seq09_prefix_summaries
                else "KITTI09 action-family validation is missing or controls do not pass; no v74-TF online promotion."
            )
        ),
        "training_free_compliance": "Aggregates fixed diagnostic/action oracle rows only; no policy is trained.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "action_family_results.csv", out_rows)
    write_csv(args.out_dir / "positive_chunks_by_action.csv", summary_rows)
    write_json(args.out_dir / "action_family_summary_by_seq.json", summary)
    lines = [
        "# v74-TF Phase 4 No-Go Reasons By Action",
        "",
        f"- oracle_rows: `{len(out_rows)}`",
        f"- phase4_01_gate_pass: `{phase4_01}`",
        f"- phase4_09_gate_pass: `False`",
        f"- phase4_09_prefix_summaries: `{len(seq09_prefix_summaries)}`",
        f"- blocked_reason: `{summary['blocked_reason']}`",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['v74tf_action_family']}: positives `{row['positive_chunk_count']}`, "
            f"median_proxy `{row['median_J_scale_improvement_proxy']}`, beats_controls `{row['beats_controls']}`, "
            f"gate `{row['phase4_01_gate_pass']}`"
        )
    write_text(args.out_dir / "no_go_reasons_by_action.md", "\n".join(lines) + "\n")
    print({"out_dir": str(args.out_dir), "phase4_gate_pass": summary["phase4_gate_pass"], "blocked_reason": summary["blocked_reason"]})


if __name__ == "__main__":
    main()
