#!/usr/bin/env python3
"""Synthetic audit for v119 LB-NORM anchor source value normalization."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from lingbot_map.layers.attention import _v118_anchor_source_value_scaling


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented/stage1_lbnorm"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_action_row(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one action row in {path}, got {len(rows)}")
    return rows[0]


def run_case(mode: str) -> dict[str, Any]:
    action_path = OUT / f"synthetic_{mode}.jsonl"
    if action_path.exists():
        action_path.unlink()
    os.environ["ACL2_V105_STAGE4_ACTION_FILE"] = str(action_path)
    os.environ["ACL2_V105_STAGE4_ACTION_LABEL"] = f"V119_LBNORM_SYNTHETIC_{mode.upper()}"
    os.environ["ACL2_V105_GCA_TRACE_DATASET"] = "synthetic"
    os.environ["ACL2_V105_GCA_TRACE_SEQ"] = "00"
    os.environ["ACL2_V105_GCA_TRACE_METHOD"] = f"synthetic_{mode}"

    # two cached scale-reference frames; patch tokens are offsets 6..9 in each frame
    v_full = torch.ones((1, 2, 20, 3), dtype=torch.float32)
    kv_cache = {
        "_anchor_source_value_enabled": True,
        "_anchor_source_value_weight_map": {0: 2.0, 1: 1.0},
        "_anchor_source_value_token_roles": ["patch"],
        "_anchor_source_value_context_roles": ["scale_reference_context"],
        "_anchor_source_value_token_weight_root": "",
        "_anchor_source_value_token_weight_mode": "",
        "_anchor_source_value_weight_normalization": mode,
        "_v107_source_frames_0": [0, 1],
    }
    scaled = _v118_anchor_source_value_scaling(
        kv_cache=kv_cache,
        global_idx=0,
        v_full=v_full,
        q_seq_len=10,
        special_prefix_tokens=0,
        tokens_per_frame=10,
        cached_frames=2,
        scale_frames=8,
        num_register_tokens=4,
        current_frame_start=2,
    )
    row = read_action_row(action_path)
    patch_values = torch.cat([scaled[:, :, 6:10, :].reshape(-1), scaled[:, :, 16:20, :].reshape(-1)])
    return {
        "case": mode,
        "action_row": rel(action_path),
        "weight_mean": float(row["weight_mean"]),
        "weight_std": float(row["weight_std"]),
        "weight_l1": float(row["weight_l1"]),
        "weight_l2": float(row["weight_l2"]),
        "weight_delta_l1": float(row["weight_delta_l1"]),
        "weight_delta_l2": float(row["weight_delta_l2"]),
        "changed_value_token_count": int(row["changed_value_token_count"]),
        "target_value_token_count": int(row["target_value_token_count"]),
        "normalization_scale": float(row["normalization_scale"]),
        "value_weight_normalization": str(row["value_weight_normalization"]),
        "observed_scaled_patch_mean": float(patch_values.mean().item()),
        "observed_scaled_patch_std": float(patch_values.std(unbiased=False).item()),
        "mean_abs_diff_from_1": abs(float(row["weight_mean"]) - 1.0),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case("legacy_geometric_mean_1"), run_case("arithmetic_mean_1")]
    summary = {
        "schema": "acl2_v119tf_stage1_lbnorm_value_scaling_audit_v1",
        "rows": rows,
        "legacy_mean_gt_1": rows[0]["weight_mean"] > 1.0,
        "arithmetic_mean_exact_pass": rows[1]["mean_abs_diff_from_1"] <= 1e-6,
        "arithmetic_changed_token_count": rows[1]["changed_value_token_count"],
        "outputs": {
            "rows": rel(OUT / "lbnorm_value_scaling_audit_rows.csv"),
            "summary": rel(OUT / "lbnorm_value_scaling_audit_summary.json"),
        },
    }
    summary["audit_pass"] = bool(summary["legacy_mean_gt_1"] and summary["arithmetic_mean_exact_pass"])
    write_csv(OUT / "lbnorm_value_scaling_audit_rows.csv", rows)
    write_json(OUT / "lbnorm_value_scaling_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["audit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
