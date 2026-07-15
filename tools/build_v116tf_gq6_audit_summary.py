#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v == v]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": mean(vals),
        "median": median(vals),
        "max": max(vals),
    }


def summarize_case(case_dir: Path) -> dict[str, Any]:
    state_rows = read_csv_rows(case_dir / "hs_gq_state_action_rows.csv")
    token_rows = read_csv_rows(case_dir / "hs_gq_action_gate_rows.csv")
    changed = [row.get("changed_state") == "True" for row in state_rows]
    first_chunk = [row.get("first_chunk_no_prior") == "True" for row in state_rows]
    return {
        "case_dir": str(case_dir),
        "state_action_rows": len(state_rows),
        "pre_gla_token_action_rows": len(token_rows),
        "changed_state_rows": int(sum(changed)),
        "first_chunk_no_prior_rows": int(sum(first_chunk)),
        "changed_state_fraction_non_first": (
            None
            if len(state_rows) == int(sum(first_chunk))
            else float(sum(c for c, first in zip(changed, first_chunk) if not first) / max(1, len(state_rows) - int(sum(first_chunk))))
        ),
        "scope_counts": {scope: sum(1 for row in state_rows if row.get("scope") == scope) for scope in sorted({row.get("scope") for row in state_rows})},
        "global_layer_counts": {
            str(layer): sum(1 for row in state_rows if row.get("global_layer_idx") == layer)
            for layer in sorted({row.get("global_layer_idx") for row in state_rows})
        },
        "state_delta_gate": stats([v for v in (as_float(row.get("state_delta_gate")) for row in state_rows) if v is not None]),
        "state_delta_rel_norm_raw": stats(
            [v for v in (as_float(row.get("state_delta_rel_norm_raw")) for row in state_rows) if v is not None]
        ),
        "state_delta_rel_norm_after": stats(
            [v for v in (as_float(row.get("state_delta_rel_norm_after")) for row in state_rows) if v is not None]
        ),
        "semantic_risk_mean": stats([v for v in (as_float(row.get("semantic_risk_mean")) for row in state_rows) if v is not None]),
        "semantic_stable_mean": stats([v for v in (as_float(row.get("semantic_stable_mean")) for row in state_rows) if v is not None]),
        "delta_pressure": stats([v for v in (as_float(row.get("delta_pressure")) for row in state_rows) if v is not None]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v116 GQ6 state-action audit rows.")
    parser.add_argument("--results-root", default="results/acl2_v116tf_fast_semantic_causal_memory_influence")
    parser.add_argument("--case-glob", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    diagnostics = root / "diagnostics"
    cases = sorted(p for p in diagnostics.glob(args.case_glob) if p.is_dir())
    rows = [summarize_case(case) for case in cases]
    payload = {
        "results_root": str(root),
        "case_glob": args.case_glob,
        "case_count": len(cases),
        "cases": rows,
        "all_cases_have_state_action_rows": bool(rows) and all(row["state_action_rows"] > 0 for row in rows),
        "any_pre_gla_token_action_rows": any(row["pre_gla_token_action_rows"] > 0 for row in rows),
        "all_cases_layer23_only": bool(rows)
        and all(set(row["global_layer_counts"].keys()).issubset({"23"}) for row in rows if row["state_action_rows"] > 0),
    }
    out_path = diagnostics / f"{args.output_prefix}_gq6_audit_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
