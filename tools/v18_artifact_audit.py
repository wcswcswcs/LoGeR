#!/usr/bin/env python3
"""Audit ACL2 v18 true-action instrumentation artifacts.

The audit is deliberately conservative: it only reports files that exist on
disk and simple integrity checks that can be verified from those files. It does
not infer that a true-action candidate is deployable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import torch


REQUIRED = [
    "post_zp_delta_before_after.pt",
    "per_layer_branch_post_zp_delta.pt",
    "per_token_to_post_zp_contribution_summary.pt",
    "basis_vector_bank.pt",
    "basis_projection_coefficients.csv",
    "W_long_short_tensor_summary.pt",
    "W_short_apply_history.jsonl",
    "overlap_geometry_replay_target.pt",
    "overlap_geometry_replay_debug.jsonl",
    "window_scale_proxy.jsonl",
    "candidate_commit_manifest.csv",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"json_decode_error": line[:200]})
    return rows


def _pt_chunk_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return 0
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    return len(chunks) if isinstance(chunks, list) else 0


def _finite_csv_rows(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    numeric_fields = [
        "native_delta_norm",
        "committed_delta_norm",
        "action_delta_norm",
        "cos_committed_to_continuity_basis",
        "cos_action_to_continuity_basis",
    ]
    for row in rows:
        for key in numeric_fields:
            text = row.get(key)
            if text in {None, ""}:
                continue
            try:
                value = float(text)
            except ValueError:
                return False
            if math.isnan(value) or math.isinf(value):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for run_dir in sorted(p for p in trace_root.iterdir() if p.is_dir()) if trace_root.exists() else []:
        present = [name for name in REQUIRED if (run_dir / name).exists()]
        coeff_rows = _read_csv(run_dir / "basis_projection_coefficients.csv")
        row = {
            "run_name": run_dir.name,
            "trace_dir": str(run_dir),
            "required_present": len(present),
            "required_total": len(REQUIRED),
            "artifact_coverage": len(present) / float(len(REQUIRED)),
            "post_zp_chunks": _pt_chunk_count(run_dir / "post_zp_delta_before_after.pt"),
            "basis_bank_chunks": _pt_chunk_count(run_dir / "basis_vector_bank.pt"),
            "overlap_target_chunks": _pt_chunk_count(run_dir / "overlap_geometry_replay_target.pt"),
            "basis_projection_rows": len(coeff_rows),
            "basis_projection_no_nan": _finite_csv_rows(coeff_rows),
            "window_scale_proxy_rows": len(_read_jsonl(run_dir / "window_scale_proxy.jsonl")),
            "W_short_history_rows": len(_read_jsonl(run_dir / "W_short_apply_history.jsonl")),
            "present_files": ",".join(present),
            "gate_pass": len(present) >= math.ceil(0.9 * len(REQUIRED)) and _finite_csv_rows(coeff_rows),
        }
        rows.append(row)

    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "v18_artifact_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "v18_artifact_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pass_count = sum(1 for row in rows if row.get("gate_pass"))
    summary = {
        "trace_root": str(trace_root),
        "runs": len(rows),
        "gate_pass_runs": pass_count,
        "all_runs_gate_pass": bool(rows) and pass_count == len(rows),
        "required_files": REQUIRED,
    }
    (out_dir / "v18_artifact_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ACL2 v18 Artifact Audit",
        "",
        f"Trace root: `{trace_root}`",
        f"Runs audited: `{len(rows)}`",
        f"Gate-pass runs: `{pass_count}`",
        "",
        "| Run | Coverage | Coeff rows | Window proxy rows | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_name']}` | {float(row['artifact_coverage']):.3f} | "
            f"{row['basis_projection_rows']} | {row['window_scale_proxy_rows']} | "
            f"{row['gate_pass']} |"
        )
    (out_dir / "v18_artifact_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
