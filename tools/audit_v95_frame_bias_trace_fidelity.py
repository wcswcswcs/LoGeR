#!/usr/bin/env python3
"""Audit sampled frame-bias attention-mass trace fidelity for ACL2 v95 Track D."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_TARGET_CSV = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackD_read_ggsmd_action_targets_v1/target_cases.csv"
)


def _parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _seq_norm(value: Any) -> str:
    return f"{int(str(value).strip()):02d}"


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(statistics.median(vals))


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _select_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    seqs = {_seq_norm(seq) for seq in _parse_csv_text(args.seqs)}
    case_types = {str(case_type).strip() for case_type in _parse_csv_text(args.case_types)}
    max_per_bucket = int(args.max_targets_per_case_type_per_seq)
    max_total = int(args.max_targets_total)
    counts: dict[tuple[str, str], int] = {}
    selected: list[dict[str, Any]] = []
    with args.target_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = _seq_norm(row.get("seq") or row.get("sequence"))
            case_type = str(row.get("case_type", "")).strip()
            if seqs and seq not in seqs:
                continue
            if case_types and case_type not in case_types:
                continue
            bucket = (seq, case_type)
            if max_per_bucket > 0 and counts.get(bucket, 0) >= max_per_bucket:
                continue
            target = dict(row)
            target["seq"] = seq
            target["case_type"] = case_type
            target["chunk_id"] = int(row["chunk_id"])
            selected.append(target)
            counts[bucket] = counts.get(bucket, 0) + 1
            if max_total > 0 and len(selected) >= max_total:
                break
    if not selected:
        raise ValueError(f"no targets selected from {args.target_csv}")
    return selected


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return None


def _extract_frame_summary(payload: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if payload is None:
        return {}, "missing_hmc_state_hash"
    frame = (
        payload.get("control_trace", {})
        .get("hook_effect_summary", {})
        .get("frame_attention", {})
    )
    if not isinstance(frame, dict):
        return {}, "missing_frame_attention_summary"
    return frame, "ok"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _run_summary(rows: list[dict[str, Any]], run: str, case_type: str) -> dict[str, Any]:
    subset = [row for row in rows if row.get("run") == run and row.get("case_type") == case_type]
    available = [row for row in subset if bool(row.get("trace_available"))]

    def vals(key: str) -> list[float]:
        out: list[float] = []
        for row in available:
            value = _finite(row.get(key))
            if value is not None:
                out.append(value)
        return out

    pos_lift = vals("mean_frame_bias_positive_pair_mass_lift")
    neg_lift = vals("mean_frame_bias_negative_pair_mass_lift")
    return {
        "run": run,
        "case_type": case_type,
        "row_count": len(subset),
        "trace_available_count": len(available),
        "trace_available_fraction": float(len(available) / max(len(subset), 1)),
        "positive_lift_median": _median(pos_lift),
        "positive_lift_mean": _mean(pos_lift),
        "negative_lift_median": _median(neg_lift),
        "negative_lift_mean": _mean(neg_lift),
        "positive_pair_fraction_median": _median(vals("mean_frame_bias_positive_pair_fraction")),
        "negative_pair_fraction_median": _median(vals("mean_frame_bias_negative_pair_fraction")),
        "mean_abs_bias_median": _median(vals("mean_abs_bias")),
        "attention_mass_available_fraction": float(
            sum(1 for row in subset if bool(row.get("attention_mass_available"))) / max(len(subset), 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--max-targets-per-case-type-per-seq", type=int, default=0)
    parser.add_argument("--max-targets-total", type=int, default=0)
    parser.add_argument("--candidate", default="READ12_GG_SMD_A1B1G1")
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--baseline", default="READ0_NATIVE")
    parser.add_argument("--action-eval-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    controls = args.control or ["READ7_EXISTING_QK_GEOM_QKVAR_CONTROL", "READ10_EXISTING_QK_RANDOM_SAME_MASS"]
    run_names = list(dict.fromkeys([args.baseline, args.candidate] + controls))
    targets = _select_targets(args)
    rows: list[dict[str, Any]] = []
    for target in targets:
        seq = str(target["seq"])
        case_type = str(target["case_type"])
        chunk = int(target["chunk_id"])
        for run in run_names:
            run_dir = args.base_dir / f"seq{seq}" / f"chunk{chunk:03d}_{case_type}" / run
            hmc_path = run_dir / "hmc_state_hash.jsonl"
            payload = _read_first_jsonl(hmc_path)
            frame, status = _extract_frame_summary(payload)
            row: dict[str, Any] = {
                "seq": seq,
                "case_type": case_type,
                "chunk": chunk,
                "run": run,
                "run_dir": str(run_dir),
                "hmc_state_hash": str(hmc_path),
                "trace_status": status,
                "trace_available": bool(frame.get("attention_mass_available", False)),
            }
            for key, value in frame.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
            rows.append(row)

    run_summaries: dict[str, dict[str, Any]] = {}
    for run in run_names:
        for case_type in sorted({str(t["case_type"]) for t in targets}):
            run_summaries[f"{run}:{case_type}"] = _run_summary(rows, run, case_type)

    cand_bad = run_summaries.get(f"{args.candidate}:bad", {})
    cand_good = run_summaries.get(f"{args.candidate}:good", {})
    pos_bad = _finite(cand_bad.get("positive_lift_median"))
    neg_bad = _finite(cand_bad.get("negative_lift_median"))
    pos_good = _finite(cand_good.get("positive_lift_median"))
    neg_good = _finite(cand_good.get("negative_lift_median"))
    trace_fidelity_gate_pass = bool(
        cand_bad.get("trace_available_fraction") == 1.0
        and cand_good.get("trace_available_fraction") == 1.0
        and pos_bad is not None
        and pos_bad > 0.0
        and neg_bad is not None
        and neg_bad < 0.0
        and pos_good is not None
        and pos_good > 0.0
        and neg_good is not None
        and neg_good < 0.0
    )

    action_gate = None
    if args.action_eval_json is not None and args.action_eval_json.exists():
        action_payload = json.loads(args.action_eval_json.read_text(encoding="utf-8"))
        action_gate = bool(action_payload.get("phase3_existing_actuator_gate_pass", False))

    payload = {
        "base_dir": str(args.base_dir),
        "target_csv": str(args.target_csv),
        "selected_target_count": len(targets),
        "run_names": run_names,
        "row_count": len(rows),
        "run_summaries": run_summaries,
        "trace_fidelity_gate_pass": trace_fidelity_gate_pass,
        "trace_fidelity_rule": (
            "candidate bad/good rows must all expose frame attention mass stats; "
            "positive-bias pair mass median lift > 0 and negative-bias pair mass median lift < 0."
        ),
        "action_eval_json": str(args.action_eval_json) if args.action_eval_json else None,
        "action_mechanism_gate_pass": action_gate,
        "action_coupling_blocker": bool(trace_fidelity_gate_pass and action_gate is False),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = args.out_dir / "rows.csv"
    out_summary = args.out_dir / "summary.json"
    _write_csv(out_rows, rows)
    out_summary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({
        "out_dir": args.out_dir,
        "row_count": len(rows),
        "trace_fidelity_gate_pass": trace_fidelity_gate_pass,
        "action_mechanism_gate_pass": action_gate,
        "action_coupling_blocker": payload["action_coupling_blocker"],
    }), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
