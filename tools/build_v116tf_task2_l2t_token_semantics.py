#!/usr/bin/env python3
"""Build compact per-patch semantic tensors for ACL2 v116 Task2 L2T."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "task2_l2t"
TOKEN_ROOT = OUT / "token_semantics"
V108_STAGE2 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank"
TOKEN_ROWS = V108_STAGE2 / "token_semantic_rows.csv"
STAGE2_SUMMARY = V108_STAGE2 / "stage2_summary.json"
SEQUENCES = ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_stage2_summary() -> dict[str, Any]:
    return json.loads(STAGE2_SUMMARY.read_text(encoding="utf-8"))


def role_channels(row: dict[str, str]) -> tuple[float, float, float, float, float, float, int]:
    role = str(row.get("semantic_role", ""))
    trust = max(0.0, min(1.0, safe_float(row.get("semantic_trust"))))
    boundary = max(0.0, min(1.0, safe_float(row.get("semantic_boundary_risk"))))
    confidence = max(0.0, min(1.0, safe_float(row.get("semantic_confidence"))))

    dynamic = trust if role == "dynamic_transient" else 0.0
    stable = trust if role == "stable_structure" else 0.0
    weak = trust if role in {"vegetation_weak_context", "ground_or_road_weak", "sky_or_lowobs"} else 0.0
    lowtrust = max(1.0 - trust, trust if role == "unknown_lowtrust" else 0.0)
    role_id = {
        "dynamic_transient": 1,
        "semantic_boundary": 2,
        "unknown_lowtrust": 3,
        "vegetation_weak_context": 4,
        "ground_or_road_weak": 5,
        "sky_or_lowobs": 6,
        "stable_structure": 7,
        "road_boundary_or_layout": 8,
    }.get(role, 0)
    return dynamic, boundary, lowtrust, weak, stable, confidence, role_id


def main() -> int:
    summary = load_stage2_summary()
    frame_universe = {str(k): int(v) for k, v in dict(summary["frame_universe_by_seq"]).items()}
    patch_count = int(summary["patch_grid_h"]) * int(summary["patch_grid_w"])
    patch_start_idx = int(summary["patch_start_idx"])
    token_end = patch_start_idx + patch_count - 1

    TOKEN_ROOT.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for seq in SEQUENCES:
        frame_count = frame_universe[seq]
        arrays[seq] = {
            "dynamic": np.zeros((frame_count, patch_count), dtype=np.float32),
            "boundary": np.zeros((frame_count, patch_count), dtype=np.float32),
            "lowtrust": np.zeros((frame_count, patch_count), dtype=np.float32),
            "weak": np.zeros((frame_count, patch_count), dtype=np.float32),
            "stable": np.zeros((frame_count, patch_count), dtype=np.float32),
            "confidence": np.zeros((frame_count, patch_count), dtype=np.float32),
            "role_id": np.zeros((frame_count, patch_count), dtype=np.uint8),
            "filled": np.zeros((frame_count, patch_count), dtype=np.bool_),
        }

    seq_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    duplicate_count = 0
    out_of_scope_count = 0
    out_of_range_count = 0
    with TOKEN_ROWS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = str(row.get("seq_id", ""))
            if seq not in arrays:
                continue
            frame = int(float(row.get("frame_id", -1)))
            token_id = int(float(row.get("token_id", -1)))
            patch_idx = token_id - patch_start_idx
            if frame < 0 or frame >= arrays[seq]["filled"].shape[0] or patch_idx < 0 or patch_idx >= patch_count:
                out_of_range_count += 1
                continue
            if token_id < patch_start_idx or token_id > token_end:
                out_of_scope_count += 1
                continue
            if bool(arrays[seq]["filled"][frame, patch_idx]):
                duplicate_count += 1
            dynamic, boundary, lowtrust, weak, stable, confidence, role_id = role_channels(row)
            arrays[seq]["dynamic"][frame, patch_idx] = dynamic
            arrays[seq]["boundary"][frame, patch_idx] = boundary
            arrays[seq]["lowtrust"][frame, patch_idx] = lowtrust
            arrays[seq]["weak"][frame, patch_idx] = weak
            arrays[seq]["stable"][frame, patch_idx] = stable
            arrays[seq]["confidence"][frame, patch_idx] = confidence
            arrays[seq]["role_id"][frame, patch_idx] = role_id
            arrays[seq]["filled"][frame, patch_idx] = True
            seq_counts[seq] += 1
            role_counts[str(row.get("semantic_role", ""))] += 1
            label_counts[str(row.get("label_name", ""))] += 1

    output_files: dict[str, dict[str, str]] = {}
    coverage_rows: list[dict[str, Any]] = []
    for seq, seq_arrays in arrays.items():
        output_files[seq] = {}
        filled = seq_arrays["filled"]
        for name, value in seq_arrays.items():
            path = TOKEN_ROOT / f"seq{seq}_{name}.npy"
            np.save(path, value)
            output_files[seq][name] = rel(path)
        coverage = float(filled.mean()) if filled.size else 0.0
        frame_coverage = np.mean(filled.all(axis=1)) if filled.shape[0] else 0.0
        coverage_rows.append(
            {
                "seq": seq,
                "frame_count": int(filled.shape[0]),
                "patch_count": int(filled.shape[1]),
                "filled_token_count": int(filled.sum()),
                "expected_token_count": int(filled.size),
                "token_coverage": coverage,
                "all_patch_frame_coverage": float(frame_coverage),
                "dynamic_mean": float(seq_arrays["dynamic"].mean()),
                "boundary_mean": float(seq_arrays["boundary"].mean()),
                "lowtrust_mean": float(seq_arrays["lowtrust"].mean()),
                "weak_mean": float(seq_arrays["weak"].mean()),
                "stable_mean": float(seq_arrays["stable"].mean()),
            }
        )

    csv_path = TOKEN_ROOT / "TOKEN_SEMANTIC_COVERAGE.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields: list[str] = []
        for row in coverage_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    report_lines = [
        "# ACL2 v116 Task2 L2T Token Semantic Tensor Build",
        "",
        f"source: `{rel(TOKEN_ROWS)}`",
        f"patch_start_idx: `{patch_start_idx}`",
        f"patch_count: `{patch_count}`",
        f"token_range: `{patch_start_idx}..{token_end}`",
        "",
        "## Coverage",
        "",
        "| seq | frame_count | filled_token_count | expected_token_count | token_coverage | all_patch_frame_coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage_rows:
        report_lines.append(
            f"| {row['seq']} | {row['frame_count']} | {row['filled_token_count']} | "
            f"{row['expected_token_count']} | {row['token_coverage']} | {row['all_patch_frame_coverage']} |"
        )
    report_lines += [
        "",
        "## Channel Mapping",
        "",
        "- `dynamic`: semantic_role == `dynamic_transient`, weighted by `semantic_trust`.",
        "- `boundary`: `semantic_boundary_risk`.",
        "- `lowtrust`: max(`1 - semantic_trust`, `semantic_trust` when role is `unknown_lowtrust`).",
        "- `weak`: semantic_role in `vegetation_weak_context`, `ground_or_road_weak`, `sky_or_lowobs`, weighted by `semantic_trust`.",
        "- `stable`: semantic_role == `stable_structure`, weighted by `semantic_trust`.",
        "",
        "These tensors are only cue materialization. No L2T geometry result is produced by this build.",
    ]
    report_path = TOKEN_ROOT / "TOKEN_SEMANTIC_BUILD_REPORT.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    out_summary = {
        "schema": "acl2_v116tf_task2_l2t_token_semantic_build_summary_v1",
        "source_token_rows": rel(TOKEN_ROWS),
        "stage2_summary": rel(STAGE2_SUMMARY),
        "sequences": list(SEQUENCES),
        "patch_grid_h": int(summary["patch_grid_h"]),
        "patch_grid_w": int(summary["patch_grid_w"]),
        "patch_count": patch_count,
        "patch_start_idx": patch_start_idx,
        "token_range": [patch_start_idx, token_end],
        "seq_row_counts": dict(seq_counts),
        "role_counts": dict(role_counts),
        "top_labels": dict(label_counts.most_common(30)),
        "duplicate_count": duplicate_count,
        "out_of_scope_count": out_of_scope_count,
        "out_of_range_count": out_of_range_count,
        "coverage_rows": coverage_rows,
        "outputs": {
            "token_root": rel(TOKEN_ROOT),
            "coverage": rel(csv_path),
            "report": rel(report_path),
            "arrays": output_files,
        },
    }
    write_json(TOKEN_ROOT / "TOKEN_SEMANTIC_BUILD_SUMMARY.json", out_summary)
    print(json.dumps(clean_json(out_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
