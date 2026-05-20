#!/usr/bin/env python3
"""Write a categorized YAML config diff report for v15 reproducibility gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


READ_KEYS = ("read_", "beta_", "frame_bias", "fast_cue", "enable_frame_read", "enable_swa_read")
SWA_KEYS = ("swa_", "enable_swa_")
TTT_KEYS = ("ttt_", "hmc_", "prior_", "mp_", "write_", "commit")
EVAL_KEYS = ("output_", "hybrid_debug_jsonl", "prior_debug_jsonl", "start_frame", "end_frame", "save_", "load_")


def _load(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return data


def _has_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) or prefix in key for prefix in prefixes)


def _row(key: str, old: Any, new: Any) -> Dict[str, Any]:
    return {
        "field_name": key,
        "old_value": old,
        "new_value": new,
        "source": "hmc_config.yaml",
        "is_ttt_related": _has_prefix(key, TTT_KEYS),
        "is_read_related": _has_prefix(key, READ_KEYS),
        "is_swa_related": _has_prefix(key, SWA_KEYS),
        "is_eval_related": _has_prefix(key, EVAL_KEYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    old = _load(old_path)
    new = _load(new_path)
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            rows.append(_row(key, old.get(key), new.get(key)))

    out = {
        "old_config": str(old_path),
        "new_config": str(new_path),
        "diff_count": len(rows),
        "diffs": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if args.out_md:
        lines = [
            "# v15 Config Diff Report",
            "",
            f"old = `{old_path}`",
            f"new = `{new_path}`",
            "",
            "| Field | Old | New | TTT | Read | SWA | Eval |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['field_name']}` | `{row['old_value']}` | `{row['new_value']}` | "
                f"{str(row['is_ttt_related']).lower()} | {str(row['is_read_related']).lower()} | "
                f"{str(row['is_swa_related']).lower()} | {str(row['is_eval_related']).lower()} |"
            )
        Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
