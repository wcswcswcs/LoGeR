#!/usr/bin/env python3
"""Evaluate v94 Phase5 object-source extension candidates with controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
V93_OBJECT_ROWS = Path(
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier"
    "/phase1_object_identity_row_join/object_identity_row_join.csv"
)
V93_OBJECT_SUMMARY = V93_OBJECT_ROWS.parent / "object_identity_source_summary.json"
V93_OBJECT_AUDIT = V93_OBJECT_ROWS.parent / "object_identity_join_audit.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[int(pos)]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def stable_key(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def rotate(pred: list[bool], amount: int) -> list[bool]:
    if not pred:
        return []
    amount %= len(pred)
    return pred[amount:] + pred[:amount]


def regime_rotate(rows: list[dict[str, Any]], pred: list[bool], amount: int) -> list[bool]:
    out = [False] * len(pred)
    by_seq: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_seq.setdefault(str(row.get("seq")), []).append(idx)
    for indices in by_seq.values():
        values = [pred[idx] for idx in indices]
        values = rotate(values, amount)
        for idx, value in zip(indices, values):
            out[idx] = value
    return out


def object_order_rotate(rows: list[dict[str, Any]], pred: list[bool], amount: int) -> list[bool]:
    order = sorted(range(len(rows)), key=lambda idx: (f(rows[idx].get("object_boundary_ratio")), idx))
    values = [pred[idx] for idx in order]
    values = rotate(values, amount)
    out = [False] * len(pred)
    for idx, value in zip(order, values):
        out[idx] = value
    return out


def metric(rows: list[dict[str, Any]], pred: list[bool], policy: str, kind: str) -> dict[str, Any]:
    labelled = [idx for idx, row in enumerate(rows) if row.get("case_label_offline_only") in {"bad", "good"}]
    bad = [idx for idx in labelled if rows[idx].get("case_label_offline_only") == "bad"]
    good = [idx for idx in labelled if rows[idx].get("case_label_offline_only") == "good"]
    bad_hits = [idx for idx in bad if pred[idx]]
    good_hits = [idx for idx in good if pred[idx]]
    bad_recall = len(bad_hits) / len(bad) if bad else 0.0
    good_fpr = len(good_hits) / len(good) if good else 0.0
    positives = [idx for idx in labelled if pred[idx]]
    return {
        "policy": policy,
        "kind": kind,
        "positive_rows": len(positives),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + 1.0 - good_fpr),
        "loso_positive_folds": len({rows[idx].get("seq") for idx in positives}),
        "positive_sequences": ",".join(sorted({str(rows[idx].get("seq")) for idx in positives})),
        "bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in bad_hits),
        "good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in good_hits),
    }


def pred_or(*preds: list[bool]) -> list[bool]:
    return [any(values) for values in zip(*preds)]


def pred_and(*preds: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*preds)]


def build_atoms(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, list[bool]]:
    return {
        "GLOBAL_CROSS_GE_Q75": [
            f(row.get("boundary_global_cross_ratio")) >= thresholds["boundary_global_cross_ratio_q75"] for row in rows
        ],
        "GLOBAL_NEW_GE_Q75": [
            f(row.get("boundary_new_id_ratio")) >= thresholds["boundary_new_id_ratio_q75"] for row in rows
        ],
        "OBJ_BOUNDARY_GE_Q75": [
            f(row.get("object_boundary_ratio")) >= thresholds["object_boundary_ratio_q75"] for row in rows
        ],
        "RADIO_BOUNDARY_GE_Q75": [
            f(row.get("radio_boundary_mean")) >= thresholds["radio_boundary_mean_q75"] for row in rows
        ],
        "SEM_INVALID": [row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY" for row in rows],
        "SEM_LOWOBS": [row.get("semantic_evidence_type") == "SEM_LOWOBS_ABSTAIN" for row in rows],
        "SEM_WEAK_CONTEXT": [row.get("semantic_evidence_type") == "SEM_WEAK_CONTEXT" for row in rows],
    }


def eval_policy(rows: list[dict[str, Any]], pred: list[bool], policy: str, kind: str) -> dict[str, Any]:
    actual = metric(rows, pred, policy, kind)
    controls = {
        "same_count_rot7": metric(rows, rotate(pred, 7), f"{policy}__same_count_rot7", "control"),
        "object_order_rot5": metric(rows, object_order_rotate(rows, pred, 5), f"{policy}__object_order_rot5", "control"),
        "regime_rot1": metric(rows, regime_rotate(rows, pred, 1), f"{policy}__regime_rot1", "control"),
        "regime_rot2": metric(rows, regime_rotate(rows, pred, 2), f"{policy}__regime_rot2", "control"),
    }
    for name, ctrl in controls.items():
        actual[f"{name}_balanced_accuracy"] = ctrl["balanced_accuracy"]
        actual[f"{name}_margin"] = actual["balanced_accuracy"] - ctrl["balanced_accuracy"]
    margins = [actual[f"{name}_margin"] for name in controls]
    actual["min_control_margin"] = min(margins)
    actual["object_source_extension_gate_pass"] = bool(
        actual["bad_recall"] >= 0.60
        and actual["good_FPR"] <= 0.25
        and actual["loso_positive_folds"] >= 3
        and actual["min_control_margin"] >= 0.05
    )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--object-identity-rows", type=Path, default=V93_OBJECT_ROWS)
    parser.add_argument("--object-source-summary", type=Path, default=V93_OBJECT_SUMMARY)
    parser.add_argument("--object-source-audit", type=Path, default=V93_OBJECT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase5_object_source_extension")
    args = parser.parse_args()

    phase5_rows = read_csv_rows(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    phase5_labelled = [row for row in phase5_rows if row.get("case_label_offline_only") in {"bad", "good"}]
    object_rows = read_csv_rows(args.object_identity_rows)
    object_by_pair = {row["pair_id"]: row for row in object_rows}
    joined: list[dict[str, Any]] = []
    missing_pairs = []
    for row in phase5_labelled:
        obj = object_by_pair.get(str(row.get("pair_id")))
        if obj is None:
            missing_pairs.append(str(row.get("pair_id")))
            obj = {}
        out = dict(row)
        for key, value in obj.items():
            if key not in out:
                out[key] = value
            else:
                out[f"object_source_{key}"] = value
        joined.append(out)

    thresholds = {
        "boundary_global_cross_ratio_q75": quantile([f(row.get("boundary_global_cross_ratio")) for row in object_rows], 0.75),
        "boundary_new_id_ratio_q75": quantile([f(row.get("boundary_new_id_ratio")) for row in object_rows], 0.75),
        "object_boundary_ratio_q75": quantile([f(row.get("object_boundary_ratio")) for row in object_rows], 0.75),
        "radio_boundary_mean_q75": quantile([f(row.get("radio_boundary_mean")) for row in object_rows], 0.75),
    }
    atoms = build_atoms(joined, thresholds)

    policy_preds: dict[str, tuple[str, list[bool]]] = {}
    for name, pred in atoms.items():
        policy_preds[name] = ("atom", pred)
    for name_a, name_b in [
        ("GLOBAL_CROSS_GE_Q75", "SEM_INVALID"),
        ("GLOBAL_NEW_GE_Q75", "SEM_INVALID"),
        ("OBJ_BOUNDARY_GE_Q75", "GLOBAL_CROSS_GE_Q75"),
        ("OBJ_BOUNDARY_GE_Q75", "GLOBAL_NEW_GE_Q75"),
        ("RADIO_BOUNDARY_GE_Q75", "SEM_INVALID"),
    ]:
        policy_preds[f"{name_a}_OR_{name_b}"] = ("or2", pred_or(atoms[name_a], atoms[name_b]))
    for names in [
        ("GLOBAL_CROSS_GE_Q75", "SEM_INVALID", "SEM_LOWOBS"),
        ("GLOBAL_NEW_GE_Q75", "SEM_INVALID", "SEM_LOWOBS"),
        ("OBJ_BOUNDARY_GE_Q75", "GLOBAL_CROSS_GE_Q75", "SEM_INVALID"),
        ("OBJ_BOUNDARY_GE_Q75", "GLOBAL_NEW_GE_Q75", "SEM_INVALID"),
        ("RADIO_BOUNDARY_GE_Q75", "GLOBAL_CROSS_GE_Q75", "SEM_INVALID"),
        ("GLOBAL_CROSS_GE_Q75", "SEM_INVALID", "SEM_WEAK_CONTEXT"),
    ]:
        policy_preds["_OR_".join(names)] = ("or3", pred_or(*(atoms[name] for name in names)))

    policy_rows = [eval_policy(joined, pred, name, kind) for name, (kind, pred) in policy_preds.items()]
    policy_rows = sorted(
        policy_rows,
        key=lambda row: (
            bool(row["object_source_extension_gate_pass"]),
            f(row["bad_recall"]),
            -f(row["good_FPR"]),
            f(row["min_control_margin"]),
            f(row["loso_positive_folds"]),
        ),
        reverse=True,
    )
    passing = [row for row in policy_rows if row["object_source_extension_gate_pass"]]
    selected = passing[0] if passing else policy_rows[0]
    selected_pred = policy_preds[selected["policy"]][1]
    selected_rows = []
    for row, is_pos in zip(joined, selected_pred):
        out = {
            "pair_id": row.get("pair_id"),
            "seq": row.get("seq"),
            "case_label_offline_only": row.get("case_label_offline_only"),
            "selected_policy": selected["policy"],
            "selected_policy_positive": bool(is_pos),
            "semantic_evidence_type": row.get("semantic_evidence_type"),
            "boundary_global_cross_ratio": row.get("boundary_global_cross_ratio"),
            "boundary_new_id_ratio": row.get("boundary_new_id_ratio"),
            "object_boundary_ratio": row.get("object_boundary_ratio"),
            "same_global_object_id": row.get("same_global_object_id"),
            "radio_boundary_mean": row.get("radio_boundary_mean"),
            "carrier_error_merge_residual_after_abs": row.get("carrier_error_merge_residual_after_abs"),
            "carrier_error_abs_log_scale_jump_runtime": row.get("carrier_error_abs_log_scale_jump_runtime"),
        }
        selected_rows.append(out)

    object_summary = read_json(args.object_source_summary) if args.object_source_summary.exists() else {}
    object_audit = read_json(args.object_source_audit) if args.object_source_audit.exists() else {}
    summary = {
        "phase": "Phase5_object_source_extension",
        "diagnostic_only": True,
        "method_promoted": False,
        "object_source_extension_gate_pass": bool(passing),
        "selected_policy": selected,
        "passing_policy_count": len(passing),
        "policies_evaluated": len(policy_rows),
        "joined_labelled_rows": len(joined),
        "missing_object_source_pairs": missing_pairs,
        "thresholds": thresholds,
        "object_identity_source_pass": object_audit.get("object_identity_source_pass"),
        "object_identity_labelled_coverage": object_summary.get("object_identity_labelled_coverage"),
        "radio_labelled_coverage": object_summary.get("radio_labelled_coverage"),
        "source_scope_counts": object_summary.get("source_scope_counts"),
        "counterfactual_allowed": bool(passing),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "Object-source extension is an offline next-route diagnostic. Phase6 counterfactual is still required before runtime action.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "object_source_policy_metrics.csv", policy_rows)
    write_csv(args.out_dir / "selected_policy_rows.csv", selected_rows)
    write_json(args.out_dir / "phase5_object_source_extension_summary.json", summary)
    (args.out_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# ACL2 v94 Phase5 Object-Source Extension",
                "",
                f"- object_source_extension_gate_pass: `{summary['object_source_extension_gate_pass']}`",
                f"- selected_policy: `{selected['policy']}`",
                f"- bad_recall: `{selected['bad_recall']}`",
                f"- good_FPR: `{selected['good_FPR']}`",
                f"- LOSO: `{selected['loso_positive_folds']}`",
                f"- min_control_margin: `{selected['min_control_margin']}`",
                "",
                "This is not a runtime method. Phase6 counterfactual remains mandatory.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"object_source_extension_gate_pass={summary['object_source_extension_gate_pass']}")
    print(f"selected_policy={selected['policy']}")
    print(f"selected_bad_recall={selected['bad_recall']}")
    print(f"selected_good_FPR={selected['good_FPR']}")
    print(f"selected_loso_positive_folds={selected['loso_positive_folds']}")
    print(f"selected_min_control_margin={selected['min_control_margin']}")
    print(f"passing_policy_count={len(passing)}")
    print("runtime_action_allowed=False")


if __name__ == "__main__":
    main()
