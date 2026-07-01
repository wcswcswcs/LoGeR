#!/usr/bin/env python3
"""Diagnostic-only cue-gated upper bound for ACL2 v95 Track H READ action.

This audit does not execute a runtime controller.  It combines already measured
READ action rows with a cue bank to estimate whether a cue-gated READ policy is
worth implementing: cue-selected target cases use the measured candidate row,
and unselected cases fall back to READ0_NATIVE.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v80_semantic_read_existing_actuator_control import (  # noqa: E402
    BASELINE,
    DEFAULT_CONTROLS,
    LOWER_IS_BETTER_KEYS,
    _candidate_decision,
    _jsonable,
    _parse_csv_text,
    _seq_norm,
)


DEFAULT_EVAL_METRICS = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackH_read_old_v79_l07_action_surface_full_v1/eval_metrics.csv"
)
DEFAULT_CUE_METRICS = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackD_read_qqkk_patch_tensor_eval_full_v2/patch_tensor_candidate_metrics.csv"
)
DEFAULT_TARGET_CSV = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackD_read_ggsmd_action_targets_v1/target_cases.csv"
)
DEFAULT_CANDIDATE = "READ1_EXISTING_L07_LAYOUT_SELECT"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _split_csv(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _case_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["seq"]), str(row["case_type"]), int(row["chunk"]))


def _target_case_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (_seq_norm(row["seq"]), str(row["case_type"]), int(row["chunk_id"]))


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return cleaned[:120].strip("_") or "cue"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    out[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _select_targets(path: Path, *, seqs: set[str], case_types: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(path):
        seq = _seq_norm(row["seq"])
        case_type = str(row["case_type"])
        if seqs and seq not in seqs:
            continue
        if case_types and case_type not in case_types:
            continue
        cur = dict(row)
        cur["seq"] = seq
        cur["case_type"] = case_type
        cur["chunk_id"] = int(row["chunk_id"])
        out.append(cur)
    return out


def _index_eval_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not row.get("seq") or not row.get("case_type") or not row.get("chunk") or not row.get("run"):
            continue
        indexed.setdefault(_case_key(row), {})[str(row["run"])] = row
    return indexed


def _cue_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_csv(args.cue_metrics_csv)
    requested = set(args.cue_id or [])
    selected: list[dict[str, Any]] = []
    for row in rows:
        cue_id = str(row.get("cue_id", ""))
        if requested and cue_id not in requested:
            continue
        if args.candidate_gate_only and not _truthy(row.get("candidate_gate_pass")):
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no cue rows selected")
    return selected


def _evaluate_cue(
    *,
    cue: dict[str, Any],
    targets: list[dict[str, Any]],
    indexed_eval: dict[tuple[str, str, int], dict[str, dict[str, Any]]],
    candidate_run: str,
    controls: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_pair_ids = set(_split_csv(cue.get("selected_pair_ids")))
    synth_name = f"{candidate_run}__CUE_GATE_{_sanitize(str(cue['cue_id']))}"
    decision_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    selected_target_ids: list[str] = []
    selected_counts: dict[str, int] = {"bad": 0, "good": 0}

    required_runs = [BASELINE, candidate_run] + controls
    for target in targets:
        key = _target_case_key(target)
        case_id = str(target["case_id"])
        case_type = str(target["case_type"])
        case_rows = indexed_eval.get(key, {})
        for run in [BASELINE] + controls:
            row = case_rows.get(run)
            if row is None:
                missing.append({"case_id": case_id, "run": run, "reason": "missing_eval_row"})
                continue
            decision_rows.append(dict(row))
        cue_selected = case_id in selected_pair_ids
        source_run = candidate_run if cue_selected else BASELINE
        source = case_rows.get(source_run)
        if source is None:
            missing.append({"case_id": case_id, "run": source_run, "reason": "missing_eval_row"})
            continue
        synth = dict(source)
        synth["run"] = synth_name
        synth["diagnostic_cue_id"] = cue["cue_id"]
        synth["diagnostic_source_run"] = source_run
        synth["diagnostic_cue_selected"] = str(bool(cue_selected)).lower()
        synth["diagnostic_case_id"] = case_id
        decision_rows.append(synth)
        detailed_rows.append(synth)
        if cue_selected:
            selected_target_ids.append(case_id)
            selected_counts[case_type] = selected_counts.get(case_type, 0) + 1

    missing_required: list[dict[str, Any]] = []
    for target in targets:
        key = _target_case_key(target)
        case_id = str(target["case_id"])
        case_rows = indexed_eval.get(key, {})
        for run in required_runs:
            if run not in case_rows:
                missing_required.append({"case_id": case_id, "run": run})

    decision = _candidate_decision(decision_rows, candidate=synth_name, controls=controls)
    summary: dict[str, Any] = {
        "cue_id": cue["cue_id"],
        "candidate_run": candidate_run,
        "synthetic_candidate": synth_name,
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "runtime_gate_claimed": False,
        "target_count": len(targets),
        "cue_bank_selected_count": cue.get("selected_count"),
        "cue_bank_bad_recall": cue.get("bad_recall"),
        "cue_bank_good_FPR": cue.get("good_FPR"),
        "cue_bank_balanced_accuracy": cue.get("balanced_accuracy"),
        "cue_bank_candidate_gate_pass": _truthy(cue.get("candidate_gate_pass")),
        "cue_bank_selected_pair_ids": cue.get("selected_pair_ids"),
        "selected_target_count": len(selected_target_ids),
        "selected_target_bad_count": selected_counts.get("bad", 0),
        "selected_target_good_count": selected_counts.get("good", 0),
        "selected_target_ids": selected_target_ids,
        "missing_eval_row_count": len(missing),
        "missing_eval_rows": missing,
        "missing_required_row_count": len(missing_required),
        "decision": decision,
        "diagnostic_phase3_gate_pass": bool(decision.get("phase3_existing_actuator_gate_pass")),
        "bad_metric_passes": decision.get("bad_metric_passes", []),
        "good_safety_pass": decision.get("good_safety_pass"),
        "rule": (
            "Diagnostic-only upper bound: cue-selected target cases reuse measured candidate-run metrics; "
            "unselected target cases reuse READ0_NATIVE metrics. This is not a runtime controller result."
        ),
    }
    for key in ["J_short_eval_proxy"] + LOWER_IS_BETTER_KEYS:
        comp = decision["comparisons"][key]
        prefix = f"{key}."
        for field in [
            "bad_baseline_median",
            "bad_candidate_median",
            "bad_improvement_vs_baseline_ratio",
            "good_baseline_median",
            "good_candidate_median",
            "good_worsen_ratio",
            "bad_candidate_beats_all_controls",
            "threshold",
            "bad_key_pass",
        ]:
            summary[prefix + field] = comp.get(field)
    return summary, detailed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-metrics-csv", type=Path, default=DEFAULT_EVAL_METRICS)
    parser.add_argument("--cue-metrics-csv", type=Path, default=DEFAULT_CUE_METRICS)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--cue-id", action="append", default=[])
    parser.add_argument("--candidate-gate-only", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-detail-csv", type=Path, default=None)
    args = parser.parse_args()

    controls = args.control or DEFAULT_CONTROLS
    seqs = {_seq_norm(seq) for seq in _parse_csv_text(args.seqs)}
    case_types = {str(case_type).strip() for case_type in _parse_csv_text(args.case_types)}
    eval_rows = _read_csv(args.eval_metrics_csv)
    indexed_eval = _index_eval_rows(eval_rows)
    targets = _select_targets(args.target_csv, seqs=seqs, case_types=case_types)
    cues = _cue_rows(args)

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for cue in cues:
        summary, detail_rows = _evaluate_cue(
            cue=cue,
            targets=targets,
            indexed_eval=indexed_eval,
            candidate_run=args.candidate_run,
            controls=controls,
        )
        summaries.append(summary)
        details.extend(detail_rows)

    summaries.sort(
        key=lambda row: (
            not bool(row.get("diagnostic_phase3_gate_pass")),
            -(float(row.get("J_short_eval_proxy.bad_improvement_vs_baseline_ratio") or -999.0)),
            str(row.get("cue_id")),
        )
    )
    payload = {
        "eval_metrics_csv": str(args.eval_metrics_csv),
        "cue_metrics_csv": str(args.cue_metrics_csv),
        "target_csv": str(args.target_csv),
        "candidate_run": args.candidate_run,
        "controls": controls,
        "target_count": len(targets),
        "cue_count": len(cues),
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "runtime_gate_claimed": False,
        "any_diagnostic_phase3_gate_pass": bool(any(row["diagnostic_phase3_gate_pass"] for row in summaries)),
        "summaries": summaries,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.out_csv, summaries)
    if args.out_detail_csv is not None:
        _write_csv(args.out_detail_csv, details)
    print(
        json.dumps(
            _jsonable(
                {
                    "diagnostic_only": True,
                    "runtime_action_allowed": False,
                    "cue_count": len(cues),
                    "any_diagnostic_phase3_gate_pass": payload["any_diagnostic_phase3_gate_pass"],
                    "top_cue_id": summaries[0]["cue_id"] if summaries else None,
                    "top_gate_pass": summaries[0]["diagnostic_phase3_gate_pass"] if summaries else None,
                    "top_selected_target_count": summaries[0]["selected_target_count"] if summaries else None,
                    "top_bad_metric_passes": summaries[0]["bad_metric_passes"] if summaries else None,
                    "top_good_safety_pass": summaries[0]["good_safety_pass"] if summaries else None,
                    "out_json": str(args.out_json),
                    "out_csv": str(args.out_csv),
                    "out_detail_csv": str(args.out_detail_csv) if args.out_detail_csv is not None else None,
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
