#!/usr/bin/env python3
"""Phase 3 fixed failure-mode rule assignment for ACL2 v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import finite_quantile, read_csv, safe_float, utc_now, write_csv, write_json
from v74tf_common import REPORT_ROOT


def _q(rows: list[dict[str, Any]], key: str, q: float) -> float | None:
    return finite_quantile((row.get(key) for row in rows), q)


def _mass(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    vals = [safe_float(row.get(key)) for key in keys]
    finite = [float(v) for v in vals if v is not None]
    return sum(finite) if finite else None


def _assign(row: dict[str, Any], th: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    stable = _mass(row, ("stable_structure_ratio", "radio_static_mean", "radio_interior_mean"))
    harm = _mass(row, ("dynamic_thing_ratio", "lowtrust_stuff_ratio", "sky_context_ratio", "radio_dynamic_mean", "radio_lowtrust_mean", "radio_boundary_mean"))
    context = _mass(row, ("road_context_ratio", "sky_context_ratio", "lowtrust_stuff_ratio"))
    residual = safe_float(row.get("raw_overlap_residual_rmse") or row.get("raw_overlap_residual"))
    trans = safe_float(row.get("merge_transform_translation_norm"))
    boundary = safe_float(row.get("boundary_jump"))
    gram = safe_float(row.get("global_k_layer5_gram_motion"))
    radio_stability = safe_float(row.get("radio_temporal_stability_mean"))
    debug = {
        "stable_mass": stable,
        "harm_mass": harm,
        "context_mass": context,
        "raw_overlap_residual": residual,
        "merge_transform_translation_norm": trans,
        "boundary_jump": boundary,
        "short_geom_instability": gram,
        "radio_temporal_stability_mean": radio_stability,
    }
    if stable is None and harm is None and context is None and residual is None and trans is None:
        return "UNKNOWN", ["missing_runtime_rule_features"], debug
    type_u = bool(
        trans is not None
        and th.get("trans_p25") is not None
        and trans <= th["trans_p25"]
        and boundary is not None
        and th.get("boundary_p75") is not None
        and boundary >= th["boundary_p75"]
        and stable is not None
        and th.get("stable_p50") is not None
        and stable >= th["stable_p50"]
        and (residual is None or th.get("residual_p90") is None or residual <= th["residual_p90"])
    )
    type_o = bool(
        residual is not None
        and th.get("residual_p75") is not None
        and residual >= th["residual_p75"]
        and harm is not None
        and th.get("harm_p75") is not None
        and harm >= th["harm_p75"]
        and (stable is None or th.get("stable_p50") is None or stable <= th["stable_p50"])
    )
    type_l = bool(
        context is not None
        and th.get("context_p75") is not None
        and context >= th["context_p75"]
        and (stable is None or th.get("stable_p50") is None or stable <= th["stable_p50"])
        and (radio_stability is None or th.get("radio_stability_p50") is None or radio_stability <= th["radio_stability_p50"])
    )
    type_n = bool(
        gram is not None
        and th.get("gram_p75") is not None
        and gram >= th["gram_p75"]
        and not type_o
        and (stable is None or th.get("stable_p50") is None or stable <= th["stable_p50"])
    )
    fired = [name for name, ok in (("U", type_u), ("O", type_o), ("L", type_l), ("N", type_n)) if ok]
    if type_l:
        return "L", fired, debug
    if type_o and type_u:
        return ("O" if residual is not None and th.get("residual_p75") is not None and residual >= th["residual_p75"] else "U"), fired, debug
    if type_o:
        return "O", fired, debug
    if type_u:
        return "U", fired, debug
    if type_n:
        return "N", fired, debug
    return "NATIVE", fired, debug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, default=REPORT_ROOT / "phase1_multiseq_scale_drift_ledger" / "scale_drift_ledger.csv")
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase3_failure_mode_rules")
    args = parser.parse_args()

    rows = read_csv(args.ledger_csv)
    out_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    summary_by_seq: dict[str, Any] = {}
    for seq in sorted({str(row.get("seq", "")) for row in rows}):
        seq_rows = [row for row in rows if str(row.get("seq", "")) == seq]
        th = {
            "trans_p25": _q(seq_rows, "merge_transform_translation_norm", 0.25),
            "boundary_p75": _q(seq_rows, "boundary_jump", 0.75),
            "residual_p75": _q(seq_rows, "raw_overlap_residual_rmse", 0.75),
            "residual_p90": _q(seq_rows, "raw_overlap_residual_rmse", 0.90),
            "gram_p75": _q(seq_rows, "global_k_layer5_gram_motion", 0.75),
            "stable_p50": _q([{"stable": _mass(row, ("stable_structure_ratio", "radio_static_mean", "radio_interior_mean"))} for row in seq_rows], "stable", 0.50),
            "harm_p75": _q([{"harm": _mass(row, ("dynamic_thing_ratio", "lowtrust_stuff_ratio", "sky_context_ratio", "radio_dynamic_mean", "radio_lowtrust_mean", "radio_boundary_mean"))} for row in seq_rows], "harm", 0.75),
            "context_p75": _q([{"context": _mass(row, ("road_context_ratio", "sky_context_ratio", "lowtrust_stuff_ratio"))} for row in seq_rows], "context", 0.75),
            "radio_stability_p50": _q(seq_rows, "radio_temporal_stability_mean", 0.50),
        }
        counts: dict[str, int] = {}
        for row in seq_rows:
            mode, fired, debug = _assign(row, th)
            counts[mode] = counts.get(mode, 0) + 1
            base = {
                "seq": seq,
                "chunk_id": row.get("chunk_id"),
                "is_target_chunk": row.get("is_target_chunk"),
                "failure_mode_rule": mode,
                "rules_fired": ",".join(fired),
                "rule_source": "fixed_no_gt_runtime_feature_quantiles",
                "action_family_hint": {"U": "under_refresh", "O": "over_refresh", "L": "low_observability", "N": "nonuniform", "NATIVE": "native", "UNKNOWN": "blocked"}.get(mode, "blocked"),
            }
            out_rows.append({**base, **debug})
            debug_rows.append({**base, **debug, **{f"threshold_{k}": v for k, v in th.items()}})
        unknown = counts.get("UNKNOWN", 0)
        non_native_modes = {key for key, value in counts.items() if value > 0 and key not in {"UNKNOWN", "NATIVE"}}
        limit = 0.40 if seq == "01" else 0.50
        gate = bool(seq_rows and unknown / len(seq_rows) <= limit and len(non_native_modes) >= 1)
        summary_by_seq[seq] = {
            "rows": len(seq_rows),
            "mode_counts": counts,
            "unknown_fraction": None if not seq_rows else float(unknown / len(seq_rows)),
            "thresholds": th,
            "gate_pass": gate,
            "gate_rule": f"unknown_fraction <= {limit} and at least one non-native mode fires.",
        }
    summary = {
        "schema": "acl2_v74tf_phase3_failure_mode_rules_v1",
        "created_at": utc_now(),
        "summary_by_seq": summary_by_seq,
        "phase3_gate_pass": bool(summary_by_seq.get("01", {}).get("gate_pass")) and bool(summary_by_seq.get("09", {}).get("gate_pass")),
        "training_free_compliance": "Fixed quantile rules from no-GT runtime/proxy columns; no classifier is trained.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "failure_mode_by_seq_chunk.csv", out_rows)
    write_csv(args.out_dir / "rule_trigger_debug.csv", debug_rows)
    write_json(args.out_dir / "failure_mode_summary_by_seq.json", summary)
    print({"out_dir": str(args.out_dir), "phase3_gate_pass": summary["phase3_gate_pass"], "summary_by_seq": summary_by_seq})


if __name__ == "__main__":
    main()

