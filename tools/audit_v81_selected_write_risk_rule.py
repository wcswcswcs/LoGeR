#!/usr/bin/env python3
"""Audit v81 selected-write risk rule without running any actuator."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable


DEFAULT_ROWS = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
)
DEFAULT_VISUAL_REVIEW = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase2_long_window_visual_confirmation/visual_review.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase3_selected_write_risk_rule"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f(row: dict[str, str], key: str) -> float | None:
    try:
        text = row.get(key)
        return None if text in (None, "") else float(text)
    except ValueError:
        return None


def q25(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 4:
        return sorted(values)[0]
    return statistics.quantiles(values, n=4)[0]


def profile_strict(row: dict[str, Any]) -> bool:
    low = bool(row["selected_low_support_ratio"] >= 0.50 or row["selected_low_support_mass"] >= 50)
    return low and row["continuous_low_support_cluster_len"] >= 3 and row["downstream_harm"] and not row["good_protected"]


def profile_visual_cluster2(row: dict[str, Any]) -> bool:
    return (
        row["selected_low_support_ratio"] >= 0.50
        and row["continuous_low_support_cluster_len"] >= 2
        and row["downstream_harm"]
        and not row["good_protected"]
    )


def profile_direction_guarded(row: dict[str, Any]) -> bool:
    return (
        row["selected_low_support_ratio"] >= 0.50
        and row["downstream_harm"]
        and not row["good_protected"]
        and row["visual_regime_status"] != "context_dominant_regime"
    )


def profile_seq02_cluster_diagnostic(row: dict[str, Any]) -> bool:
    return (
        row["seq"] == "02"
        and row["chunk_start"] <= 66
        and row["chunk_end"] >= 62
        and row["selected_low_support_ratio"] >= 0.50
        and row["downstream_harm"]
        and not row["good_protected"]
    )


PROFILES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "R0_plan_strict": profile_strict,
    "R1_visual_cluster2_ratio_guard": profile_visual_cluster2,
    "R2_direction_guarded_no_context": profile_direction_guarded,
    "R3_seq02_cluster_diagnostic_only": profile_seq02_cluster_diagnostic,
}


def confusion(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tp = sum(row[field] and row["label_bad"] for row in rows)
    fn = sum((not row[field]) and row["label_bad"] for row in rows)
    fp = sum(row[field] and not row["label_bad"] for row in rows)
    tn = sum((not row[field]) and not row["label_bad"] for row in rows)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    seqs = sorted({row["seq"] for row in rows if row[field]})
    seq02_cluster_rows = [
        row for row in rows
        if row["seq"] == "02" and row["chunk_start"] <= 66 and row["chunk_end"] >= 62
    ]
    seq02_positive = sum(row[field] for row in seq02_cluster_rows)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "bad_recall": recall,
        "good_false_positive_rate": fpr,
        "positive_seqs": seqs,
        "seq_coverage": len(seqs),
        "seq02_62_70_positive": seq02_positive,
        "seq02_62_70_total": len(seq02_cluster_rows),
        "seq02_62_70_mostly_positive": seq02_positive >= max(1, int(0.6 * len(seq02_cluster_rows))),
        "gate_pass": recall >= 0.60 and fpr <= 0.25 and len(seqs) >= 3 and seq02_positive >= max(1, int(0.6 * len(seq02_cluster_rows))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--visual-review", type=Path, default=DEFAULT_VISUAL_REVIEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = read_csv(args.rows)
    review = {row["window_id"]: row for row in read_csv(args.visual_review)}
    baseline_vals = [value for row in rows if (value := f(row, "baseline_abs_error_mean")) is not None]
    protection_q25 = q25(baseline_vals)
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        rv = review.get(row["window_id"], {})
        base = f(row, "baseline_abs_error_mean")
        item: dict[str, Any] = {
            "window_id": row["window_id"],
            "seq": row["seq"],
            "chunk_start": int(row["chunk_start"]),
            "chunk_end": int(row["chunk_end"]),
            "case_type": row["case_type"],
            "label_bad": row["case_type"] == "bad",
            "selected_low_support_ratio": f(row, "selected_low_support_ratio") or 0.0,
            "selected_low_support_mass": f(row, "selected_low_support_mass") or 0.0,
            "selected_runtime_mass": f(row, "selected_runtime_mass") or 0.0,
            "continuous_low_support_cluster_len": int(float(row.get("continuous_low_support_cluster_len") or 0)),
            "downstream_direction": row.get("selected_minus_control_downstream_direction") or "unknown",
            "downstream_harm": row.get("selected_minus_control_downstream_direction") == "harmful",
            "baseline_abs_error_mean": base,
            "good_protected": bool(base is not None and protection_q25 is not None and base <= protection_q25),
            "selected_write_alignment_status": rv.get("selected_write_alignment_status", ""),
            "support_status": rv.get("support_status", ""),
            "visual_regime_status": rv.get("visual_regime_status", ""),
        }
        for name, fn in PROFILES.items():
            item[name] = bool(fn(item))
        audit_rows.append(item)

    matrices = {name: confusion(audit_rows, name) for name in PROFILES}
    best_name = max(PROFILES, key=lambda name: (matrices[name]["gate_pass"], matrices[name]["bad_recall"], -matrices[name]["good_false_positive_rate"]))
    best_matrix = matrices[best_name]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "selected_write_risk_rows.csv", audit_rows)
    write_json(
        args.out_dir / "bad_good_confusion_matrix.json",
        {
            "schema": "acl2_v81_phase3_selected_write_risk_rule_v1",
            "baseline_abs_error_q25_good_protection": protection_q25,
            "profiles": matrices,
            "best_profile": best_name,
            "best_profile_gate_pass": best_matrix["gate_pass"],
            "risk_rule_can_enter_action": best_matrix["gate_pass"],
            "decision": "pass_to_ttt_action" if best_matrix["gate_pass"] else "block_ttt_action_refine_or_route_type_b",
        },
    )
    best_field = best_name
    write_csv(args.out_dir / "false_positive_good_cases.csv", [row for row in audit_rows if row[best_field] and not row["label_bad"]])
    write_csv(args.out_dir / "false_negative_bad_cases.csv", [row for row in audit_rows if (not row[best_field]) and row["label_bad"]])
    report = [
        "# ACL2 v81 Phase3 Selected-Write Risk Rule Audit",
        "",
        "No actuator was run in this phase.",
        "",
        f"Best profile: `{best_name}`",
        f"Gate pass: `{best_matrix['gate_pass']}`",
        f"Bad recall: `{best_matrix['bad_recall']}`",
        f"Good false-positive rate: `{best_matrix['good_false_positive_rate']}`",
        f"Seq coverage: `{best_matrix['seq_coverage']}`",
        f"Seq02 62-70 positive: `{best_matrix['seq02_62_70_positive']}/{best_matrix['seq02_62_70_total']}`",
        "",
        "Interpretation:",
        "",
        "- Strict selected-write risk catches the seq02 cluster subset but misses many bad long windows that look like Type-B merge/overlap/context regimes.",
        "- The repaired visual profiles improve only modestly and still fail the global v81 Phase3 gate.",
        "- Therefore selected-write low-support remains a diagnostic/routing signal, not a TTT actuator rule.",
        "",
    ]
    (args.out_dir / "rule_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"best_profile": best_name, **best_matrix}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
