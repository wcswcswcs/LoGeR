#!/usr/bin/env python3
"""Summarize v118 Stage3-R14 LingBot FlashInfer internal-read signal smoke."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE2 = RESULT_ROOT / "stage2_memory_entry_provenance"
OUT = RESULT_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe"
TRACE = STAGE2 / "smoke_lingbot_flashinfer_trace.jsonl"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv_rows(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = finite_values(rows, key)
    total = len(rows)
    if total == 0 or values.size == 0:
        return {
            f"{key}_coverage": 0.0,
            f"{key}_p10": None,
            f"{key}_p90": None,
            f"{key}_span": 0.0,
            f"{key}_finite_count": int(values.size),
        }
    p10, p90 = np.percentile(values, [10, 90])
    return {
        f"{key}_coverage": float(values.size) / float(total),
        f"{key}_p10": float(p10),
        f"{key}_p90": float(p90),
        f"{key}_span": float(p90 - p10),
        f"{key}_finite_count": int(values.size),
    }


def report_text(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ACL2 v118-TF Stage3-R14 LingBot FlashInfer Internal Signal Probe",
            "",
            f"- stage3_r14_decision: `{summary['stage3_r14_decision']}`",
            f"- synthetic_internal_signal_smoke_pass: `{summary['synthetic_internal_signal_smoke_pass']}`",
            f"- full_lingbot_runtime_ready_for_stage4: `{summary['full_lingbot_runtime_ready_for_stage4']}`",
            f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
            "",
            "## Evidence",
            "",
            f"- trace: `{summary['input_trace']}`",
            f"- read_rows: `{summary['read_row_count']}`",
            f"- read_rows_with_internal_qk: `{summary['read_rows_with_internal_qk']}`",
            f"- trajectory_special_read_rows: `{summary['trajectory_special_read_rows']}`",
            f"- qk_relevance_cosine_span: `{summary['all_read_signal_stats']['qk_relevance_cosine_span']}`",
            f"- qk_relevance_softmax_span: `{summary['all_read_signal_stats']['qk_relevance_softmax_span']}`",
            "",
            "## Boundary",
            "",
            "R14 verifies that the default FlashInfer page manager can emit page-level QK/read-utility signal fields under v118 trace mode. It is a synthetic hook/provenance smoke only. It does not provide full KITTI 00/02 LingBot internal candidate/reliability rows and it does not reopen Stage4 by itself.",
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(TRACE)
    read_rows = [row for row in rows if row.get("operation_type") == "read_visible_page"]
    qk_rows = [row for row in read_rows if row.get("internal_signal_source") == "flashinfer_online_page_qk_summary"]
    trajectory_rows = [row for row in qk_rows if row.get("memory_family") == "trajectory_special"]
    probe_rows = [
        {
            "seq": row.get("seq"),
            "memory_family": row.get("memory_family"),
            "source_frame_id": row.get("source_frame_id"),
            "memory_entry_id": row.get("memory_entry_id"),
            "qk_relevance_cosine": row.get("qk_relevance_cosine"),
            "qk_relevance_rank": row.get("qk_relevance_rank"),
            "qk_relevance_softmax": row.get("qk_relevance_softmax"),
            "read_entropy_normalized": row.get("read_entropy_normalized"),
            "visible_page_count": row.get("visible_page_count"),
            "read_count": row.get("read_count"),
            "internal_signal_source": row.get("internal_signal_source"),
        }
        for row in qk_rows
    ]
    all_stats = {}
    trajectory_stats = {}
    for key in ("qk_relevance_cosine", "qk_relevance_softmax", "read_entropy_normalized", "read_count"):
        all_stats.update(stats(qk_rows, key))
        trajectory_stats.update(stats(trajectory_rows, key))

    smoke_pass = bool(read_rows and qk_rows and len(qk_rows) == len(read_rows) and trajectory_rows)
    summary = {
        "schema": "acl2_v118tf_stage3_r14_lingbot_flashinfer_internal_signal_probe_summary_v1",
        "stage3_r14_decision": (
            "SYNTHETIC_INTERNAL_SIGNAL_HOOK_PASS_FULL_LINGBOT_RUNTIME_PENDING"
            if smoke_pass
            else "NO_GO_FLASHINFER_INTERNAL_SIGNAL_HOOK_MISSING"
        ),
        "synthetic_internal_signal_smoke_pass": smoke_pass,
        "full_lingbot_runtime_ready_for_stage4": False,
        "global_goal_achieved": False,
        "input_trace": rel(TRACE),
        "read_row_count": len(read_rows),
        "read_rows_with_internal_qk": len(qk_rows),
        "trajectory_special_read_rows": len(trajectory_rows),
        "all_read_signal_stats": all_stats,
        "trajectory_special_signal_stats": trajectory_stats,
        "outputs": {
            "probe_rows": rel(OUT / "stage3_r14_flashinfer_internal_signal_rows.csv"),
            "summary": rel(OUT / "stage3_r14_lingbot_flashinfer_internal_signal_probe_summary.json"),
            "report": rel(OUT / "STAGE3_R14_LINGBOT_FLASHINFER_INTERNAL_SIGNAL_PROBE_REPORT.md"),
        },
        "modified_code_under_test": [
            "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
            "tools/build_v118tf_stage2_memory_entry_provenance.py",
        ],
        "boundary": (
            "Synthetic hook pass only; full LingBot default-backend 00/02 internal candidate/reliability "
            "runtime remains pending before LB-TA/LB-TR/LB-TE can enter Stage4."
        ),
    }
    write_csv(OUT / "stage3_r14_flashinfer_internal_signal_rows.csv", probe_rows)
    write_json(OUT / "stage3_r14_lingbot_flashinfer_internal_signal_probe_summary.json", summary)
    write_text(OUT / "STAGE3_R14_LINGBOT_FLASHINFER_INTERNAL_SIGNAL_PROBE_REPORT.md", report_text(summary))
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage3-R14",
            "surface_or_branch": "LB-Trajectory",
            "status": summary["stage3_r14_decision"],
            "artifact": rel(OUT / "stage3_r14_lingbot_flashinfer_internal_signal_probe_summary.json"),
            "notes": "Default FlashInfer trace now emits synthetic page-level QK/read-utility fields; full LingBot runtime still pending",
        }
    )
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
