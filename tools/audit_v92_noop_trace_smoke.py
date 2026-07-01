#!/usr/bin/env python3
"""Summarize v92 Phase2 native no-op merge/gauge trace smoke repair runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, safe_float, write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, pair_id, read_jsonl


DEFAULT_ROOT = ROOT / "phase2_boundary_trace_ledger/noop_trace_smoke"
DEFAULT_OUT = ROOT / "phase2_boundary_trace_ledger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _pair_from_smoke_root(path: Path) -> tuple[str, int, int, str]:
    name = path.name
    seq = ""
    curr = -1
    if name.startswith("seq") and "_chunk" in name:
        seq_part, rest = name.split("_chunk", 1)
        seq = seq_part.replace("seq", "")
        digits = ""
        for char in rest:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            curr = int(digits)
    prev = curr - 1 if curr >= 0 else -1
    return seq.zfill(2), prev, curr, pair_id(seq, prev, curr)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seq_root in sorted(args.smoke_root.iterdir()) if args.smoke_root.exists() else []:
        if not seq_root.is_dir():
            continue
        seq, prev, curr, pid = _pair_from_smoke_root(seq_root)
        manifest_path = seq_root / "phase9_swa_cache_value_run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        jobs = manifest.get("jobs", []) if isinstance(manifest, dict) else []
        job = jobs[0] if jobs else {}
        run_dir = seq_root / f"chunk{curr:02d}" / "P9_0_NATIVE"
        trace_path = run_dir / "merge_state_trace.jsonl"
        metrics_path = seq_root / "phase9_swa_cache_value_metrics.csv"
        run_log = run_dir / "run.log"
        trace_rows = read_jsonl(trace_path)
        target = None
        for item in trace_rows:
            if int(item.get("chunk_idx") or -1) == curr:
                target = item
        if target is None and trace_rows:
            target = trace_rows[-1]
        target = target or {}
        manifest_returncode = job.get("returncode")
        completed_inferred = bool(trace_path.exists() and metrics_path.exists() and run_log.exists())
        rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "pair_id": pid,
                "manifest_returncode": manifest_returncode,
                "completed_inferred_from_artifacts": completed_inferred,
                "duration_sec": job.get("duration_sec"),
                "gpu": job.get("gpu"),
                "manifest_path": str(manifest_path),
                "metrics_path": str(metrics_path),
                "run_dir": str(run_dir),
                "run_log": str(run_log),
                "trace_path": str(trace_path),
                "trace_exists": trace_path.exists(),
                "trace_rows": len(trace_rows),
                "target_trace_found": bool(target),
                "trace_schema": target.get("schema", ""),
                "transform_kind": target.get("transform_kind", ""),
                "transform_reason": target.get("transform_reason", ""),
                "transform_scale_value": target.get("transform_scale_value", ""),
                "transform_trans_norm": target.get("transform_trans_norm", ""),
                "transform_rot_trace": target.get("transform_rot_trace", ""),
                "has_non_identity_scale": bool(abs((safe_float(target.get("transform_scale_value")) or 1.0) - 1.0) > 1e-9),
                "has_nonzero_translation": bool(abs(safe_float(target.get("transform_trans_norm")) or 0.0) > 1e-9),
                "has_residual_fields": any("residual" in str(key).lower() for key in target.keys()),
            }
        )
    completed = sum(1 for row in rows if row.get("completed_inferred_from_artifacts"))
    trace_rows = sum(1 for row in rows if row.get("trace_exists"))
    non_identity = sum(1 for row in rows if row.get("has_non_identity_scale") or row.get("has_nonzero_translation"))
    residual = sum(1 for row in rows if row.get("has_residual_fields"))
    summary = {
        "phase": "Phase2_native_noop_trace_smoke_repair_audit",
        "smoke_root": str(args.smoke_root),
        "job_count": len(rows),
        "completed_jobs": completed,
        "failed_jobs": len(rows) - completed,
        "trace_file_count": trace_rows,
        "non_identity_transform_rows": non_identity,
        "residual_field_rows": residual,
        "all_completed": completed == len(rows) and len(rows) > 0,
        "repair_outcome": "true_trace_smoke_completed_but_native_trace_identity_only_no_boundary_update_norm_or_residual",
        "phase2_gate_repaired": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_csv(args.out_dir / "noop_trace_smoke_rows.csv", rows)
    write_json(args.out_dir / "noop_trace_smoke_summary.json", summary)
    print(f"job_count={summary['job_count']}")
    print(f"completed_jobs={summary['completed_jobs']}")
    print(f"failed_jobs={summary['failed_jobs']}")
    print(f"trace_file_count={summary['trace_file_count']}")
    print(f"non_identity_transform_rows={summary['non_identity_transform_rows']}")
    print(f"residual_field_rows={summary['residual_field_rows']}")
    print(f"phase2_gate_repaired={summary['phase2_gate_repaired']}")


if __name__ == "__main__":
    main()
