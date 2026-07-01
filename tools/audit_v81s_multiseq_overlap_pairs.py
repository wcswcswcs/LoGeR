#!/usr/bin/env python3
"""Audit v81S S1 multi-sequence overlap-pair gate outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair"
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    rows_path = args.root / "overlap_pairs_summary_by_seq.csv"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    rows = _read_rows(rows_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["overlap_pair_files_written"] = int(float(row.get("overlap_pair_files_written") or 0))
        out["semantic_label_projected_pair_ratio"] = _float_or_none(row.get("semantic_label_projected_pair_ratio"))
        out["median_saved_pairs_per_overlap"] = _float_or_none(row.get("median_saved_pairs_per_overlap"))
        out["median_raw_residual_rmse"] = _float_or_none(row.get("median_raw_residual_rmse"))
        out["swa_action_allowed"] = _bool(row.get("swa_action_allowed"))
        normalized.append(out)

    allowed = [row["seq"] for row in normalized if bool(row.get("swa_action_allowed"))]
    gate = {
        "phaseS1_gate_pass": bool(len(allowed) >= 3 and "01" in allowed and any(seq in allowed for seq in ("00", "02", "05"))),
        "swa_action_allowed_seqs": allowed,
        "swa_action_allowed_seq_count": len(allowed),
        "includes_seq01": "01" in allowed,
        "includes_one_of_00_02_05": any(seq in allowed for seq in ("00", "02", "05")),
    }
    audit = {
        "schema": "acl2_v81s_multiseq_overlap_pairs_audit_v1",
        "root": str(args.root),
        "gate": gate,
        "rows": normalized,
    }
    out_path = args.root / "phaseS1_overlap_pair_audit_summary.json"
    out_path.write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({"gate": gate, "out_path": out_path}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
