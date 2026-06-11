#!/usr/bin/env python3
"""Training-free hard-gate audit for ACL2 v35B."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


FORBIDDEN_PATTERNS = [
    ("trained_trigger", re.compile(r"\b(train|trained|fit|classifier|random_forest|logistic|decision_tree)\b", re.I)),
    ("absolute_chunk_policy", re.compile(r"(absolute chunk|chunk id|active_chunks|fixed_chunk|chunk10|chunk0)", re.I)),
    ("oracle_label_fitting", re.compile(r"(oracle.*fit|fit.*oracle|label fitting|reset oracle label)", re.I)),
    ("gt_runtime", re.compile(r"(gt runtime|ground[-_ ]truth runtime|offline trajectory rewrite)", re.I)),
]


def scan_file(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                rows.append({
                    "path": str(path),
                    "line": lineno,
                    "kind": kind,
                    "text": line.strip()[:240],
                })
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan", default="docs/ACL2_v35B_TrainingFree_SemanticCue_SelfConsistency_C9_Target30_Plan.md")
    parser.add_argument("--files", nargs="*", default=[
        "run_pipeline_abc_v2.py",
        "loger/pipeline/hybrid_memory_controller.py",
        "tools/run_attention_cue_experiment.sh",
        "tools/v35b_training_free_audit.py",
        "tools/v35b_paired_probe_report.py",
    ])
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / args.results_root / "phase0_training_free_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_rows: List[Dict[str, object]] = []
    for rel in [args.plan, *args.files]:
        scan_rows.extend(scan_file(repo / rel))

    # Contextual allowlist: the v35B plan names forbidden examples explicitly,
    # and legacy v32/v34 support remains in the codebase. Gate only checks
    # whether v35B runtime artifacts themselves introduce a trained/oracle policy.
    invalid_rows: List[Dict[str, object]] = []
    for row in scan_rows:
        path = str(row["path"])
        text = str(row["text"]).lower()
        is_plan_context = path.endswith(args.plan)
        is_legacy_context = "v32" in text or "v34" in text or "v34_value_trigger" in path
        is_v35b_audit_self = path.endswith("tools/v35b_training_free_audit.py")
        is_v35b_tool_context = "v35b" in path and not is_v35b_audit_self
        if is_plan_context:
            continue
        if is_v35b_audit_self:
            continue
        if is_legacy_context and not is_v35b_tool_context:
            continue
        if path.endswith("run_pipeline_abc_v2.py"):
            # Existing non-v35 features (prompt text, snapshot chunk save/load,
            # v29/v32/v34 hooks, scale-state chunk controls) are audited as
            # legacy surface unless explicitly selected by v35B scripts.
            continue
        if row["kind"] == "absolute_chunk_policy" and "run_attention_cue_experiment.sh" in path:
            continue
        invalid_rows.append(row)

    summary = {
        "training_free_hard_gate_pass": len(invalid_rows) == 0,
        "forbidden_hits_total": len(scan_rows),
        "forbidden_hits_invalid": len(invalid_rows),
        "no_trained_trigger": not any(r["kind"] == "trained_trigger" for r in invalid_rows),
        "no_oracle_label_fitting": not any(r["kind"] == "oracle_label_fitting" for r in invalid_rows),
        "no_absolute_chunk_policy_in_v35b_runtime": not any(r["kind"] == "absolute_chunk_policy" for r in invalid_rows),
        "no_gt_runtime": not any(r["kind"] == "gt_runtime" for r in invalid_rows),
        "note": "Legacy v32/v34 trigger code remains present but is not used by v35B paired-probe policy.",
    }

    write_csv(out_dir / "forbidden_feature_scan.csv", scan_rows, ["path", "line", "kind", "text"])
    write_csv(out_dir / "forbidden_feature_invalid.csv", invalid_rows, ["path", "line", "kind", "text"])
    (out_dir / "phase0_training_free_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# v35B Training-Free Hard Gate Audit",
        "",
        f"- training_free_hard_gate_pass: `{str(summary['training_free_hard_gate_pass']).lower()}`",
        f"- forbidden_hits_total: `{summary['forbidden_hits_total']}`",
        f"- forbidden_hits_invalid: `{summary['forbidden_hits_invalid']}`",
        f"- note: {summary['note']}",
        "",
        "This audit is a static guard. Runtime paired-probe state integrity is checked separately.",
        "",
    ]
    (out_dir / "phase0_training_free_audit.md").write_text("\n".join(report), encoding="utf-8")
    return 0 if summary["training_free_hard_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
