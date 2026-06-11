#!/usr/bin/env python3
"""Audit v36B H0B attention-mass feasibility from landed artifacts and code.

This intentionally does not fabricate attention mass.  It checks whether the
landed rollout summaries contain the requested mass fields, and whether the
current source-removal implementation matches the VGGT4D source-only K/V
compaction pattern.  True before/after attention mass requires q/k or attention
probability tensors that are not present in the v36B landed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_MASS_FIELDS = [
    "attention_mass_removed_before",
    "attention_mass_removed_after",
    "attention_mass_retained_before",
    "attention_mass_retained_after",
    "attention_mass_to_structure_before",
    "attention_mass_to_structure_after",
    "attention_mass_to_dynamic_before",
    "attention_mass_to_dynamic_after",
    "attention_mass_to_semantic_group_before",
    "attention_mass_to_semantic_group_after",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="replace")


def _count_mass_fields(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {field: 0 for field in REQUIRED_MASS_FIELDS}
    for row in rows:
        for field in REQUIRED_MASS_FIELDS:
            if field in row:
                counts[field] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-context-jsonl",
        action="append",
        default=[],
        help="context_skip_summary.jsonl files from representative v36B rows.",
    )
    parser.add_argument(
        "--sample-hmc-jsonl",
        action="append",
        default=[],
        help="hmc_state_hash.jsonl files from representative v36B rows.",
    )
    args = parser.parse_args()

    repo = args.repo_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vggt_attn = repo / "third_party/VGGT4D/vggt4d/layers/attention.py"
    loger_attn = repo / "loger/models/layers/attention.py"
    loger_pi3 = repo / "loger/models/pi3.py"

    code_summary = {
        "vggt4d_attention_file": str(vggt_attn),
        "loger_attention_file": str(loger_attn),
        "loger_pi3_file": str(loger_pi3),
        "vggt4d_uses_non_dyn_kv_subset": _contains(vggt_attn, "non_dyn_idx = (~dyn_mask).nonzero"),
        "vggt4d_uses_sdpa_on_subset_kv": _contains(vggt_attn, "F.scaled_dot_product_attention(qb, non_dyn_k, non_dyn_v)"),
        "loger_has_compact_kv_sdpa": _contains(loger_attn, "def _compact_kv_sdpa"),
        "loger_compact_kv_uses_keep_indices": _contains(loger_attn, "idx = torch.nonzero(source_keep_mask[b]"),
        "loger_compact_kv_uses_sdpa_on_subset_kv": _contains(loger_attn, "scaled_dot_product_attention(qb, kb, vb)"),
        "loger_context_skip_builds_source_keep_mask": _contains(loger_pi3, '"source_keep_mask": source_keep_mask'),
        "loger_context_skip_preserves_protected_tokens": _contains(loger_pi3, "keep = ((~skip) | protected).detach()"),
        "loger_context_skip_records_counts_not_attention_mass": _contains(loger_pi3, '"source_skip_tokens": source_skip_tokens'),
    }
    code_summary["source_removal_semantics_match_vggt4d"] = bool(
        code_summary["vggt4d_uses_non_dyn_kv_subset"]
        and code_summary["vggt4d_uses_sdpa_on_subset_kv"]
        and code_summary["loger_has_compact_kv_sdpa"]
        and code_summary["loger_compact_kv_uses_keep_indices"]
        and code_summary["loger_compact_kv_uses_sdpa_on_subset_kv"]
        and code_summary["loger_context_skip_builds_source_keep_mask"]
    )

    artifact_rows: List[Dict[str, Any]] = []
    total_records = 0
    total_mass_records = 0
    for raw in list(args.sample_context_jsonl) + list(args.sample_hmc_jsonl):
        path = Path(raw)
        rows = _read_jsonl(path)
        total_records += len(rows)
        counts = _count_mass_fields(rows)
        any_mass = any(v > 0 for v in counts.values())
        if any_mass:
            total_mass_records += sum(1 for row in rows if any(field in row for field in REQUIRED_MASS_FIELDS))
        artifact_rows.append({
            "path": str(path),
            "exists": path.exists(),
            "records": len(rows),
            "any_attention_mass_field": any_mass,
            **counts,
        })

    summary = {
        "results_root": str(args.results_root),
        "out_dir": str(out_dir),
        "required_mass_fields": REQUIRED_MASS_FIELDS,
        "sample_files_checked": len(artifact_rows),
        "sample_records_checked": total_records,
        "sample_records_with_any_attention_mass": total_mass_records,
        "landed_attention_mass_available": bool(total_mass_records > 0),
        "source_removal_semantics_match_vggt4d": bool(code_summary["source_removal_semantics_match_vggt4d"]),
        "h0b_gate_pass": False,
        "blocker": (
            "landed artifacts contain source skip counts/keep ratios but not q/k, "
            "attention probabilities, or before/after attention mass fields"
        ),
        "safe_conclusion": (
            "compact_kv source-removal semantics match VGGT4D's source-only K/V subset pattern, "
            "but true attention-mass removed/retained cannot be reconstructed from current landed artifacts"
        ),
        "required_repair_before_claiming_attention_mass": [
            "instrument attention layer to compute or sample qk softmax mass before/after source removal",
            "write attention_mass_removed_before/after and retained/group mass fields into hmc trace",
            "rerun at least H0C or representative H1 source-skip rows",
        ],
        "code_summary": code_summary,
    }

    _write_csv(out_dir / "h0b_artifact_field_audit.csv", artifact_rows)
    (out_dir / "h0b_attention_mass_feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md = [
        "# v36B H0B Attention-Mass Feasibility Audit",
        "",
        "## Code Check",
        "",
        f"- VGGT4D source-removal pattern found: `{code_summary['vggt4d_uses_non_dyn_kv_subset'] and code_summary['vggt4d_uses_sdpa_on_subset_kv']}`.",
        f"- LoGeR compact_kv source-removal pattern found: `{code_summary['loger_compact_kv_uses_keep_indices'] and code_summary['loger_compact_kv_uses_sdpa_on_subset_kv']}`.",
        f"- Source-removal semantics match VGGT4D: `{summary['source_removal_semantics_match_vggt4d']}`.",
        "",
        "## Artifact Check",
        "",
        f"- Sample files checked: `{summary['sample_files_checked']}`.",
        f"- Sample records checked: `{summary['sample_records_checked']}`.",
        f"- Records with attention-mass fields: `{summary['sample_records_with_any_attention_mass']}`.",
        "",
        "## Decision",
        "",
        "H0B attention-mass explainability remains incomplete. Current landed v36B artifacts support source-count / keep-ratio claims, not attention-mass causality claims.",
        "",
        "Required repair before claiming attention mass:",
        "",
    ]
    for item in summary["required_repair_before_claiming_attention_mass"]:
        md.append(f"- {item}")
    md.append("")
    (out_dir / "h0b_attention_mass_feasibility_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
