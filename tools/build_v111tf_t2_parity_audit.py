#!/usr/bin/env python3
"""Audit ACL2 v111TF T2 context-token hook parity."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T2 = RESULT_ROOT / "batch_t_t2_context_token_ablation"
WORKSPACE = T2 / "workspace"
RAW_ACTION = T2 / "raw_action"
V105_BASE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
SEQUENCES = ("00", "01", "02", "05")

PARITY_PAIRS = [
    (
        "default_off_vs_v105_baseline",
        "T2_no_action_mask_all1_default_off",
        "v105_baseline",
    ),
    (
        "new_all_context_vs_legacy_all_special",
        "T2_default_context_tokens",
        "T2_default_context_tokens_legacy_context_only",
    ),
    (
        "new_anchor_only_vs_legacy_anchor_special",
        "T2_anchor_only",
        "T2_anchor_only_for_high_risk_else_default",
    ),
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def method_dir(seq: str, policy_id: str) -> Path:
    if policy_id == "v105_baseline":
        return V105_BASE / seq / "lingbot_map_stream_default"
    return WORKSPACE / f"kitti_v111tf_t2_fullseq_{seq}" / seq / f"lingbot_map_v111tf_t2_{policy_id}_{seq}"


def compare_file(a: Path, b: Path) -> tuple[bool, str, str, float | str]:
    if not a.exists() or not b.exists():
        return False, str(a.exists()), str(b.exists()), ""
    arr_a = np.loadtxt(a)
    arr_b = np.loadtxt(b)
    if arr_a.shape != arr_b.shape:
        return False, str(arr_a.shape), str(arr_b.shape), ""
    return True, str(arr_a.shape), str(arr_b.shape), float(np.max(np.abs(arr_a - arr_b)))


def parity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        for pair_id, policy_a, policy_b in PARITY_PAIRS:
            root_a = method_dir(seq, policy_a)
            root_b = method_dir(seq, policy_b)
            for artifact in ("traj.txt", "intrinsics.txt"):
                comparable, shape_a, shape_b, diff = compare_file(root_a / artifact, root_b / artifact)
                rows.append(
                    {
                        "schema": "acl2_v111tf_t2_parity_row_v1",
                        "seq": seq,
                        "pair_id": pair_id,
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "artifact": artifact,
                        "path_a": rel(root_a / artifact),
                        "path_b": rel(root_b / artifact),
                        "comparable": comparable,
                        "shape_a": shape_a,
                        "shape_b": shape_b,
                        "max_abs_diff": diff,
                        "parity_pass": comparable and diff == 0.0,
                    }
                )
    return rows


def action_rows() -> list[dict[str, Any]]:
    config_by_key = {
        (row["policy_id"], row["seq"]): row
        for row in read_csv(T2 / "action_config_rows.csv")
    }
    rows: list[dict[str, Any]] = []
    for (policy_id, seq), cfg in sorted(config_by_key.items()):
        action_file = Path(cfg["action_file"])
        loaded = []
        if action_file.exists():
            loaded = [json.loads(line) for line in action_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        forced = [
            row for row in loaded
            if row.get("forced_context_only") or row.get("forced_anchor_only")
        ]
        masks = sorted({str(row.get("token_type_mask", "")) for row in loaded if str(row.get("token_type_mask", ""))})
        modes = sorted({str(row.get("context_only_special_mode", "")) for row in forced})
        expected_selected = int(float(cfg.get("selected_count", 0) or 0))
        rows.append(
            {
                "schema": "acl2_v111tf_t2_action_mask_audit_row_v1",
                "seq": seq,
                "policy_id": policy_id,
                "policy_family": cfg.get("policy_family", ""),
                "stage4_action_mode": cfg.get("stage4_action_mode", ""),
                "expected_selected_count": expected_selected,
                "action_file": rel(action_file),
                "action_file_exists": action_file.exists(),
                "action_log_rows": len(loaded),
                "forced_action_rows": len(forced),
                "context_only_special_modes": ";".join(modes),
                "token_type_masks": ";".join(masks),
                "forced_count_matches_expected": action_file.exists() and len(forced) == expected_selected,
            }
        )
    return rows


def main() -> None:
    p_rows = parity_rows()
    a_rows = action_rows()
    comparable = [row for row in p_rows if row["comparable"]]
    missing = [row for row in p_rows if not row["comparable"]]
    summary = {
        "schema": "acl2_v111tf_t2_parity_summary_v1",
        "parity_rows": len(p_rows),
        "completed_parity_rows": len(comparable),
        "missing_or_uncomparable_rows": len(missing),
        "completed_parity_pass": bool(comparable) and all(bool(row["parity_pass"]) for row in comparable),
        "parity_complete_all_sequences": len(missing) == 0,
        "action_mask_audit_rows": len(a_rows),
        "completed_action_mask_rows": sum(1 for row in a_rows if row["action_file_exists"]),
        "completed_action_mask_counts_pass": all(
            bool(row["forced_count_matches_expected"])
            for row in a_rows
            if row["action_file_exists"]
        ),
        "outputs": {
            "parity_rows": rel(T2 / "t2_parity_rows.csv"),
            "action_mask_audit_rows": rel(T2 / "t2_action_mask_audit_rows.csv"),
            "summary": rel(T2 / "t2_parity_summary.json"),
        },
    }
    write_csv(T2 / "t2_parity_rows.csv", p_rows)
    write_csv(T2 / "t2_action_mask_audit_rows.csv", a_rows)
    write_json(T2 / "t2_parity_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
