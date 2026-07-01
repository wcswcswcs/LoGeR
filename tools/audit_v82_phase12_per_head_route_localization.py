#!/usr/bin/env python3
"""Audit v82 Phase12 per-head SWA route-localization diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_per_head_route_localization"
)
DEFAULT_ROUTE_ROOT = DEFAULT_ROOT / "route_dump"

SEQ_ROOT_NAMES = {
    "00": "seq00_attention_mass_diagnostic",
    "01": "seq01_attention_mass_diagnostic",
    "02": "seq02_attention_mass_diagnostic",
    "05": "seq05_attention_mass_diagnostic",
    "00_semantic": "seq00_semantic_samegroup_attention_mass",
    "01_semantic": "seq01_semantic_samegroup_attention_mass",
    "02_semantic": "seq02_semantic_samegroup_attention_mass",
    "05_semantic": "seq05_semantic_samegroup_attention_mass",
    "00_semantic_head15": "seq00_semantic_samegroup_head15_attention_mass",
    "01_semantic_head15": "seq01_semantic_samegroup_head15_attention_mass",
    "02_semantic_head15": "seq02_semantic_samegroup_head15_attention_mass",
    "05_semantic_head15": "seq05_semantic_samegroup_head15_attention_mass",
}

CASE_INFO = {
    "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST": {
        "route_group": "all_head",
        "control_kind": "actual",
        "head_indices": "",
        "semantic_candidate": False,
    },
    "P9_35_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_MASS_AUDIT_LAST": {
        "route_group": "all_head",
        "control_kind": "same_mass_random",
        "head_indices": "",
        "semantic_candidate": False,
    },
    "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST": {
        "route_group": "head6",
        "control_kind": "actual",
        "head_indices": "6",
        "semantic_candidate": False,
    },
    "P9_37_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEAD6_LAST": {
        "route_group": "head6",
        "control_kind": "same_mass_random",
        "head_indices": "6",
        "semantic_candidate": False,
    },
    "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST": {
        "route_group": "heads0_6_8",
        "control_kind": "actual",
        "head_indices": "0,6,8",
        "semantic_candidate": False,
    },
    "P9_39_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_HEADS0_6_8_LAST": {
        "route_group": "heads0_6_8",
        "control_kind": "same_mass_random",
        "head_indices": "0,6,8",
        "semantic_candidate": False,
    },
    "P9_42_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_all_head",
        "control_kind": "actual",
        "head_indices": "",
        "semantic_candidate": True,
    },
    "P9_43_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_RANDOM_SAME_MASS_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_all_head",
        "control_kind": "same_mass_random",
        "head_indices": "",
        "semantic_candidate": False,
    },
    "P9_44_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_SHUFFLED_SEMANTIC_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_all_head",
        "control_kind": "shuffled_semantic",
        "head_indices": "",
        "semantic_candidate": False,
    },
    "P9_45_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_HEAD15_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_head15",
        "control_kind": "actual",
        "head_indices": "15",
        "semantic_candidate": True,
    },
    "P9_46_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_RANDOM_SAME_MASS_HEAD15_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_head15",
        "control_kind": "same_mass_random",
        "head_indices": "15",
        "semantic_candidate": False,
    },
    "P9_47_ATTENTION_BIAS_SEMANTIC_SAMEGROUP_STABLE_AGREEMENT_SHUFFLED_SEMANTIC_HEAD15_MASS_AUDIT_LAST": {
        "route_group": "semantic_samegroup_head15",
        "control_kind": "shuffled_semantic",
        "head_indices": "15",
        "semantic_candidate": False,
    },
}

SCALAR_KEYS = [
    "num_swa_overlap_bias_applied",
    "mean_abs_bias",
    "max_abs_bias",
    "mean_swa_overlap_attention_mass_selected_before",
    "mean_swa_overlap_attention_mass_selected_after",
    "mean_swa_overlap_attention_mass_selected_lift",
    "mean_swa_overlap_attention_mass_source_before",
    "mean_swa_overlap_attention_mass_source_after",
    "mean_swa_overlap_attention_mass_source_lift",
    "mean_swa_overlap_attention_mass_selected_head_max_before",
    "mean_swa_overlap_attention_mass_selected_head_max_after",
    "mean_swa_overlap_attention_mass_selected_head_max_lift",
    "mean_attention_mass_query_sample_tokens",
    "mean_attention_mass_removed_tokens",
    "mean_attention_mass_retained_tokens",
]

VECTOR_KEYS = [
    "swa_overlap_attention_mass_selected_lift_by_head",
    "swa_overlap_attention_mass_source_lift_by_head",
    "swa_overlap_attention_mass_selected_before_by_head",
    "swa_overlap_attention_mass_selected_after_by_head",
    "swa_overlap_attention_mass_source_before_by_head",
    "swa_overlap_attention_mass_source_after_by_head",
]

GEOMETRY_KEYS = [
    "ate_rmse_m",
    "rel_t_error_percent",
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _avg_vectors(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    width = len(vectors[0])
    if width <= 0 or any(len(v) != width for v in vectors):
        return None
    return [float(sum(v[i] for v in vectors) / len(vectors)) for i in range(width)]


def _load_hmc_stats(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                swa = (
                    payload.get("control_trace", {})
                    .get("hook_effect_summary", {})
                    .get("swa_read", {})
                )
                if isinstance(swa, dict):
                    rows.append(swa)

    available_rows = [row for row in rows if bool(row.get("attention_mass_available"))]
    out: dict[str, Any] = {
        "hmc_rows": int(len(rows)),
        "attention_mass_available_rows": int(len(available_rows)),
        "attention_mass_available": bool(available_rows),
    }
    for key in SCALAR_KEYS:
        vals = [_finite(row.get(key)) for row in available_rows]
        vals = [v for v in vals if v is not None]
        out[key] = _mean(vals)

    for key in VECTOR_KEYS:
        vectors: list[list[float]] = []
        for row in available_rows:
            raw = row.get(key)
            if isinstance(raw, list):
                vals = [_finite(v) for v in raw]
                if vals and all(v is not None for v in vals):
                    vectors.append([float(v) for v in vals if v is not None])
        avg = _avg_vectors(vectors)
        out[key] = json.dumps(avg) if avg is not None else ""
        out[f"{key}_available"] = bool(avg is not None)
        out[f"{key}_width"] = int(len(avg)) if avg is not None else 0

    top_heads = [
        int(row.get("swa_overlap_attention_mass_selected_top_head_by_lift"))
        for row in available_rows
        if row.get("swa_overlap_attention_mass_selected_top_head_by_lift") is not None
    ]
    top_counts = Counter(top_heads)
    out["selected_top_head_by_lift_mode"] = (
        int(top_counts.most_common(1)[0][0]) if top_counts else ""
    )
    out["selected_top_head_by_lift_counts"] = json.dumps(
        {str(k): int(v) for k, v in sorted(top_counts.items())},
        sort_keys=True,
    )
    return out


def _load_geometry_rows(seq_root: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(seq_root / "phase9_swa_cache_value_metrics.csv")
    return {str(row.get("run", "")): row for row in rows}


def _parse_seq_root(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--seq-root entries must be SEQ=PATH")
    seq, path = text.split("=", 1)
    seq = seq.strip()
    if not seq:
        raise argparse.ArgumentTypeError("empty seq in --seq-root")
    return seq, Path(path)


def _job_rows(seq: str, seq_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(seq_root / "phase9_swa_cache_value_run_manifest.json")
    jobs = manifest.get("jobs") or []
    geom_by_run = _load_geometry_rows(seq_root)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        case = str(job.get("case", ""))
        info = CASE_INFO.get(case, {
            "route_group": "other",
            "control_kind": "actual",
            "head_indices": "",
            "semantic_candidate": False,
        })
        run_dir = Path(str(job.get("out_dir", "")))
        hmc_path = run_dir / "hmc_state_hash.jsonl"
        stats = _load_hmc_stats(hmc_path)
        geom = geom_by_run.get(case, {})
        row: dict[str, Any] = {
            "seq": seq,
            "chunk": int(job.get("chunk", -1)),
            "case": case,
            "route_group": info["route_group"],
            "control_kind": info["control_kind"],
            "is_same_mass_control": bool(info["control_kind"] == "same_mass_random"),
            "is_shuffled_semantic_control": bool(info["control_kind"] == "shuffled_semantic"),
            "is_semantic_candidate": bool(info["semantic_candidate"]),
            "head_indices": info["head_indices"],
            "returncode": job.get("returncode"),
            "skipped": bool(job.get("skipped", False)),
            "gpu": job.get("gpu", ""),
            "out_dir": str(run_dir),
            "hmc_state_hash": str(hmc_path),
            "trajectory": str(run_dir / "01.txt"),
            **stats,
        }
        for key in GEOMETRY_KEYS:
            row[key] = geom.get(key, "")
        rows.append(row)
    return rows, {
        "seq": seq,
        "root": str(seq_root),
        "planned_jobs": int(len(jobs)),
        "completed_jobs": int(sum(1 for job in jobs if job.get("returncode") is not None)),
        "failed_jobs": [
            {
                "chunk": int(job.get("chunk", -1)),
                "case": str(job.get("case", "")),
                "returncode": job.get("returncode"),
            }
            for job in jobs
            if job.get("returncode") not in (None, 0)
        ],
        "metrics_csv_available": bool((seq_root / "phase9_swa_cache_value_metrics.csv").exists()),
        "decision_json_available": bool((seq_root / "phase9_swa_cache_value_decision.json").exists()),
        "decision": _read_json(seq_root / "phase9_swa_cache_value_decision.json"),
    }


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[(
            str(row["seq"]),
            int(row["chunk"]),
            str(row["route_group"]),
            str(row["control_kind"]),
        )] = row
    out: list[dict[str, Any]] = []
    control_kinds = sorted(
        {str(row["control_kind"]) for row in rows if str(row.get("control_kind")) != "actual"}
    )
    for (seq, chunk, group, control_kind), actual in sorted(by_key.items()):
        if control_kind != "actual" or group == "other":
            continue
        for kind in control_kinds:
            control = by_key.get((seq, chunk, group, kind))
            if control is None:
                continue
            paired: dict[str, Any] = {
                "seq": seq,
                "chunk": int(chunk),
                "route_group": group,
                "control_kind": kind,
                "actual_case": actual.get("case"),
                "control_case": control.get("case") if control else "",
                "actual_returncode": actual.get("returncode"),
                "control_returncode": control.get("returncode") if control else "",
                "pair_complete": bool(
                    actual.get("returncode") == 0
                    and control is not None
                    and control.get("returncode") == 0
                ),
                "actual_attention_mass_available": bool(actual.get("attention_mass_available")),
                "control_attention_mass_available": bool(control and control.get("attention_mass_available")),
                "actual_is_semantic_candidate": bool(actual.get("is_semantic_candidate")),
            }
            for key in [
                "mean_swa_overlap_attention_mass_selected_lift",
                "mean_swa_overlap_attention_mass_source_lift",
                "mean_swa_overlap_attention_mass_selected_head_max_lift",
                "mean_abs_bias",
                "max_abs_bias",
                *GEOMETRY_KEYS,
            ]:
                av = _finite(actual.get(key))
                cv = _finite(control.get(key)) if control else None
                paired[f"actual_{key}"] = av
                paired[f"control_{key}"] = cv
                paired[f"actual_minus_control_{key}"] = av - cv if av is not None and cv is not None else None
            paired["actual_selected_top_head_by_lift_mode"] = actual.get("selected_top_head_by_lift_mode", "")
            paired["control_selected_top_head_by_lift_mode"] = (
                control.get("selected_top_head_by_lift_mode", "") if control else ""
            )
            out.append(paired)
    return out


def _summarize(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], seq_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    complete_rows = [row for row in rows if row.get("returncode") == 0]
    failed_rows = [row for row in rows if row.get("returncode") not in (None, 0)]
    per_head_array_rows = [
        row for row in complete_rows if bool(row.get("swa_overlap_attention_mass_selected_lift_by_head_available"))
    ]
    summary: dict[str, Any] = {
        "schema": "acl2_v82_phase12_per_head_route_localization_audit_v1",
        "planned_jobs": int(len(rows)),
        "completed_jobs": int(len(complete_rows)),
        "failed_jobs": int(len(failed_rows)),
        "failed_job_details": [
            {"seq": row.get("seq"), "chunk": row.get("chunk"), "case": row.get("case"), "returncode": row.get("returncode")}
            for row in failed_rows
        ],
        "attention_mass_available_jobs": int(sum(1 for row in complete_rows if row.get("attention_mass_available"))),
        "per_head_array_available_jobs": int(len(per_head_array_rows)),
        "expected_case_count_per_chunk": int(len(CASE_INFO)),
        "seq_summaries": seq_summaries,
        "per_head_observability_gate_pass": bool(
            rows
            and len(complete_rows) == len(rows)
            and len(per_head_array_rows) == len(rows)
            and all(row.get("attention_mass_available") for row in complete_rows)
        ),
        "semantic_candidate_available_in_this_audit": False,
        "semantic_shuffle_control_available_in_this_audit": False,
        "method_promotion_gate_pass": False,
        "method_promotion_blocker": (
            "This audit reports route localization and controls, but does not promote a method. "
            "Promotion still requires a semantic candidate to beat same-mass and shuffled-semantic controls "
            "on the configured geometry gates with good-case protection."
        ),
    }
    group_summary: dict[str, Any] = {}
    for group in sorted({str(pair.get("route_group")) for pair in pairs}):
        group_pairs = [pair for pair in pairs if pair.get("route_group") == group and pair.get("pair_complete")]
        selected_diff = [
            _finite(pair.get("actual_minus_control_mean_swa_overlap_attention_mass_selected_lift"))
            for pair in group_pairs
        ]
        selected_diff = [v for v in selected_diff if v is not None]
        source_diff = [
            _finite(pair.get("actual_minus_control_mean_swa_overlap_attention_mass_source_lift"))
            for pair in group_pairs
        ]
        source_diff = [v for v in source_diff if v is not None]
        top_counts = Counter(
            str(pair.get("actual_selected_top_head_by_lift_mode"))
            for pair in group_pairs
            if str(pair.get("actual_selected_top_head_by_lift_mode")) != ""
        )
        group_summary[group] = {
            "pairs": int(len(group_pairs)),
            "same_mass_random_pairs": int(sum(1 for pair in group_pairs if pair.get("control_kind") == "same_mass_random")),
            "shuffled_semantic_pairs": int(sum(1 for pair in group_pairs if pair.get("control_kind") == "shuffled_semantic")),
            "selected_lift_actual_gt_control_count": int(sum(1 for v in selected_diff if v > 0.0)),
            "selected_lift_median_actual_minus_control": _median(selected_diff),
            "source_lift_actual_gt_control_count": int(sum(1 for v in source_diff if v > 0.0)),
            "source_lift_median_actual_minus_control": _median(source_diff),
            "actual_selected_top_head_mode_counts": dict(sorted(top_counts.items())),
        }
    summary["group_summary"] = group_summary
    summary["same_mass_control_pairs_complete"] = int(
        sum(1 for pair in pairs if pair.get("pair_complete") and pair.get("control_kind") == "same_mass_random")
    )
    summary["shuffled_semantic_control_pairs_complete"] = int(
        sum(1 for pair in pairs if pair.get("pair_complete") and pair.get("control_kind") == "shuffled_semantic")
    )
    summary["same_mass_control_gate_pass"] = bool(
        pairs
        and summary["same_mass_control_pairs_complete"] == len(
            [pair for pair in pairs if pair.get("control_kind") == "same_mass_random"]
        )
    )
    summary["shuffled_semantic_control_available_in_this_audit"] = bool(
        summary["shuffled_semantic_control_pairs_complete"] > 0
    )
    summary["semantic_candidate_available_in_this_audit"] = bool(
        any(row.get("is_semantic_candidate") for row in complete_rows)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    parser.add_argument(
        "--seq-root",
        action="append",
        type=_parse_seq_root,
        help="Optional SEQ=PATH override. Can be repeated.",
    )
    args = parser.parse_args()

    seq_roots = {
        seq: args.route_root / root_name
        for seq, root_name in SEQ_ROOT_NAMES.items()
    }
    if args.seq_root:
        seq_roots.update(dict(args.seq_root))

    rows: list[dict[str, Any]] = []
    seq_summaries: list[dict[str, Any]] = []
    for seq, seq_root in sorted(seq_roots.items()):
        seq_rows, seq_summary = _job_rows(seq, seq_root)
        rows.extend(seq_rows)
        seq_summaries.append(seq_summary)

    pairs = _paired_rows(rows)
    summary = _summarize(rows, pairs, seq_summaries)

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_root / "per_head_route_rows.csv"
    pairs_path = args.out_root / "per_head_route_paired_controls.csv"
    summary_path = args.out_root / "per_head_route_localization_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(pairs_path, pairs)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_rows={rows_path}")
    print(f"wrote_pairs={pairs_path}")
    print(f"wrote_summary={summary_path}")


if __name__ == "__main__":
    main()
