#!/usr/bin/env python3
"""Formalize the v94 Phase3S merge-alpha actuator as a repaired Phase3 gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_PROBE = ROOT / "phase3s_merge_gauge_actuator_sweep_max16_confirm"
DEFAULT_OUT = ROOT / "phase3_formal_merge_alpha_sensitivity"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary_path = args.probe_root / "runtime_probe_sensitivity_summary.json"
    manifest_path = args.probe_root / "runtime_probe_manifest.json"
    variant_path = args.probe_root / "runtime_probe_variant_summary.csv"
    effect_path = args.probe_root / "runtime_probe_effect_rows.csv"

    summary = read_json(summary_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    selected = summary.get("selected_candidate_summary") or {}
    if not isinstance(selected, dict):
        selected = {}

    bad_improvement = f(selected.get("bad_median_I_J_runtime_proxy"))
    good_median_worsen = f(selected.get("good_median_worsen_runtime_proxy"))
    good_max_worsen = f(selected.get("good_max_worsen_runtime_proxy"))
    failed_count = int(summary.get("runtime_probe_failed_count") or 0)
    job_count = int(summary.get("runtime_probe_job_count") or manifest.get("job_count") or 0)
    completed_count = int(manifest.get("completed_count") or job_count)
    candidate = str(summary.get("selected_candidate_variant") or "")

    checks = {
        "runtime_probe_executed": bool(summary.get("runtime_probe_executed")),
        "runtime_probe_failed_count_eq_0": failed_count == 0,
        "runtime_probe_completed_all_jobs": bool(manifest.get("all_completed", completed_count == job_count)),
        "selected_candidate_present": candidate != "",
        "selected_candidate_is_merge_alpha": candidate.startswith("merge_alpha_"),
        "bad_improvement_gate_ge_0p05": bool(selected.get("bad_improvement_gate_ge_0p05"))
        and bad_improvement >= 0.05,
        "good_median_worsen_gate_le_0p02": bool(selected.get("good_median_worsen_gate_le_0p02"))
        and good_median_worsen <= 0.02,
        "good_catastrophic_worsen_absent_le_0p02": bool(
            selected.get("good_catastrophic_worsen_absent_le_0p02")
        )
        and good_max_worsen <= 0.02,
        "selected_candidate_beats_control": bool(summary.get("selected_candidate_beats_control")),
        "sequence_coverage_ge_3": bool(selected.get("sequence_coverage_ge_3")),
        "trajectory_rows_complete": bool(selected.get("trajectory_rows_complete")),
        "handoff_transfer_rows_complete": bool(selected.get("handoff_transfer_rows_complete")),
    }
    gate_pass = all(checks.values())
    blockers = [name for name, passed in checks.items() if not passed]

    gate = {
        "phase": "Phase3_formal_merge_alpha_sensitivity",
        "entered": True,
        "phase3_repaired_gate_pass": gate_pass,
        "phase3_original_gate_replaced": False,
        "formalization_scope": "Phase3S measured runtime actuator; original trace-only Phase3 remains recorded separately",
        "selected_carrier_body": "merge_gauge" if gate_pass else "",
        "selected_actuator_variant": candidate,
        "selected_actuator_description": "semantic_merge_blend_alpha=0.2" if candidate == "merge_alpha_0p2" else candidate,
        "target_count": summary.get("target_count"),
        "runtime_probe_job_count": job_count,
        "runtime_probe_completed_count": completed_count,
        "runtime_probe_failed_count": failed_count,
        "metric_row_count": summary.get("metric_row_count"),
        "effect_row_count": summary.get("effect_row_count"),
        "bad_rows": selected.get("bad_rows"),
        "good_rows": selected.get("good_rows"),
        "sequence_coverage": selected.get("sequence_coverage"),
        "bad_median_I_J_runtime_proxy": bad_improvement,
        "good_median_worsen_runtime_proxy": good_median_worsen,
        "good_max_worsen_runtime_proxy": good_max_worsen,
        "carrier_state_delta_nonzero_rows": selected.get("carrier_state_delta_nonzero_rows"),
        "runtime_probe_trajectory_available_rows": selected.get("runtime_probe_trajectory_available_rows"),
        "handoff_transfer_available_rows": selected.get("handoff_transfer_available_rows"),
        "selected_candidate_beats_control": summary.get("selected_candidate_beats_control"),
        "checks": checks,
        "blocker": "" if gate_pass else ";".join(blockers),
        "phase4_semantic_taxonomy_allowed": gate_pass,
        "phase5_semantic_carrier_alignment_allowed": False,
        "counterfactual_allowed": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "evidence_files": {
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "variant_summary": str(variant_path),
            "effect_rows": str(effect_path),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "phase3_formal_gate_summary.json", gate)
    if variant_path.exists():
        write_csv(args.out_dir / "phase3_formal_variant_summary.csv", read_csv_rows(variant_path))
    if effect_path.exists():
        rows = read_csv_rows(effect_path)
        selected_rows = [row for row in rows if row.get("variant") == candidate]
        write_csv(args.out_dir / "phase3_formal_selected_effect_rows.csv", selected_rows)

    print(f"phase3_repaired_gate_pass={gate_pass}")
    print(f"selected_carrier_body={gate['selected_carrier_body']}")
    print(f"selected_actuator_variant={candidate}")
    print(f"bad_median_I_J_runtime_proxy={bad_improvement}")
    print(f"good_median_worsen_runtime_proxy={good_median_worsen}")
    print(f"good_max_worsen_runtime_proxy={good_max_worsen}")
    print(f"blocker={gate['blocker']}")


if __name__ == "__main__":
    main()
