#!/usr/bin/env python3
"""Run v107 Phase4 transactional-admission variants under explicit gate override."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from tools.run_v107_phase3_override_probation_sweep import (
    DEFAULT_CONFIG,
    DEFAULT_REFERENCE,
    PYTHON,
    ROOT,
    RUNNER,
    command_for_variant,
    label_metrics,
    metric_summary,
    read_json,
    rel,
    sha256_file,
    write_json,
)


def base_phase4_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "T0_immediate_commit_currentcode_area20k",
            "description": "Current-code immediate commit control; transaction disabled.",
            "args": {},
        },
        {
            "variant_id": "T1_pending2_delay4_area20k",
            "description": "Transactional admission: commit when at least 2 pending masks or delay reaches 4 frames.",
            "args": {
                "--birth-transaction-enabled": None,
                "--birth-transaction-min-pending": "2",
                "--birth-transaction-max-delay-frames": "4",
                "--birth-transaction-min-total-area": "0",
                "--birth-transaction-immediate-area": "0",
            },
        },
        {
            "variant_id": "T1_pending4_delay6_area20k",
            "description": "Transactional admission: larger pending-count trigger, capped by 6-frame delay.",
            "args": {
                "--birth-transaction-enabled": None,
                "--birth-transaction-min-pending": "4",
                "--birth-transaction-max-delay-frames": "6",
                "--birth-transaction-min-total-area": "0",
                "--birth-transaction-immediate-area": "0",
            },
        },
        {
            "variant_id": "T2_area80000_delay6_area20k",
            "description": "Transactional admission: area-sum benefit proxy trigger at 80k px, capped by 6-frame delay.",
            "args": {
                "--birth-transaction-enabled": None,
                "--birth-transaction-min-pending": "9999",
                "--birth-transaction-max-delay-frames": "6",
                "--birth-transaction-min-total-area": "80000",
                "--birth-transaction-immediate-area": "0",
            },
        },
        {
            "variant_id": "T3_area80000_high60000_delay6_area20k",
            "description": "T2 plus high-value immediate path for masks at least 60k px.",
            "args": {
                "--birth-transaction-enabled": None,
                "--birth-transaction-min-pending": "9999",
                "--birth-transaction-max-delay-frames": "6",
                "--birth-transaction-min-total-area": "80000",
                "--birth-transaction-immediate-area": "60000",
            },
        },
    ]


def high_immediate_repair_variants() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in (50000, 40000, 30000, 25000):
        out.append(
            {
                "variant_id": f"T3_repair_area80000_high{threshold}_delay6_area20k",
                "description": (
                    "Phase4 repair: area-sum transaction trigger plus lower high-value immediate threshold "
                    f"({threshold} px)."
                ),
                "args": {
                    "--birth-transaction-enabled": None,
                    "--birth-transaction-min-pending": "9999",
                    "--birth-transaction-max-delay-frames": "6",
                    "--birth-transaction-min-total-area": "80000",
                    "--birth-transaction-immediate-area": str(threshold),
                },
            }
        )
    return out


def phase4_variants(mode: str) -> list[dict[str, Any]]:
    if mode == "base":
        return base_phase4_variants()
    if mode == "high-immediate-repair":
        return high_immediate_repair_variants()
    if mode == "base-plus-repair":
        return base_phase4_variants() + high_immediate_repair_variants()
    raise ValueError(f"unsupported variant mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--frame-start", type=int, default=4160)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=90)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--large-region-area", type=int, default=20000)
    parser.add_argument(
        "--variant-mode",
        choices=["base", "high-immediate-repair", "base-plus-repair"],
        default="base",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    phase4 = output_root / "phase4_override"
    phase4.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root

    env = os.environ.copy()
    if str(args.gpu).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    reference_metric = metric_summary(reference_root / "summary.json")
    rows: list[dict[str, Any]] = [
        {
            "variant_id": "P0_current_v106_reference",
            "description": "Prior v106 reference artifact; not rerun in this sweep.",
            "status": "reference_only",
            "command": [],
            "returncode": 0,
            "run_root": rel(reference_root),
            **reference_metric,
        }
    ]
    run_records: list[dict[str, Any]] = []

    for variant in phase4_variants(str(args.variant_mode)):
        cmd = command_for_variant(
            output_root=output_root,
            config=config,
            scene_id=str(args.scene_id),
            frame_start=int(args.frame_start),
            frame_stride=int(args.frame_stride),
            frame_count=int(args.frame_count),
            gpu=str(args.gpu),
            seed=int(args.seed),
            variant=variant,
        )
        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        elapsed = float(time.time() - started)
        run_root = output_root / str(variant["variant_id"]) / "v106_stateful_sam2_rolling_scene_stream"
        summary_path = run_root / "summary.json"
        stdout_path = phase4 / f"{variant['variant_id']}.stdout.txt"
        stderr_path = phase4 / f"{variant['variant_id']}.stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        run_record = {
            "variant_id": str(variant["variant_id"]),
            "description": str(variant["description"]),
            "command": cmd,
            "returncode": int(proc.returncode),
            "elapsed_sec": elapsed,
            "stdout_path": rel(stdout_path),
            "stderr_path": rel(stderr_path),
            "run_root": rel(run_root),
            "summary_path": rel(summary_path) if summary_path.exists() else "",
        }
        run_records.append(run_record)
        if proc.returncode != 0 or not summary_path.exists():
            rows.append(
                {
                    "variant_id": str(variant["variant_id"]),
                    "description": str(variant["description"]),
                    "status": "run_failed",
                    **run_record,
                }
            )
            continue
        ref_metrics = label_metrics(
            reference_root=reference_root,
            candidate_root=run_root,
            large_region_area=int(args.large_region_area),
        )
        metric_path = phase4 / f"{variant['variant_id']}.reference_metrics.json"
        write_json(metric_path, ref_metrics)
        rows.append(
            {
                "variant_id": str(variant["variant_id"]),
                "description": str(variant["description"]),
                "status": "completed",
                **run_record,
                **metric_summary(summary_path, ref_metrics),
                "reference_metric_path": rel(metric_path),
            }
        )

    ref_recon = float(reference_metric.get("reconsolidate_runtime_sec") or 0.0)
    ref_calls = float(reference_metric.get("reconsolidate_call_count") or 0.0)
    ref_wall = float(reference_metric.get("wrapper_wall_time_sec") or 0.0)
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in {"completed", "reference_only"}:
            scored_rows.append(row)
            continue
        recon = float(row.get("reconsolidate_runtime_sec") or 0.0)
        calls = float(row.get("reconsolidate_call_count") or 0.0)
        wall = float(row.get("wrapper_wall_time_sec") or 0.0)
        metrics = row.get("reference_metrics", {}) or {}
        row["reconsolidation_runtime_reduction_vs_p0"] = (
            float((ref_recon - recon) / ref_recon) if ref_recon > 0 else None
        )
        row["reconsolidation_count_reduction_vs_p0"] = (
            float((ref_calls - calls) / ref_calls) if ref_calls > 0 else None
        )
        row["wall_reduction_vs_p0"] = float((ref_wall - wall) / ref_wall) if ref_wall > 0 else None
        row["phase4_proxy_gate"] = {
            "exact_parity_gate_ignored": True,
            "reconsolidation_runtime_reduction_ge_30pct": bool(
                row["reconsolidation_runtime_reduction_vs_p0"] is not None
                and row["reconsolidation_runtime_reduction_vs_p0"] >= 0.30
            ),
            "reconsolidation_count_reduction_ge_30pct": bool(
                row["reconsolidation_count_reduction_vs_p0"] is not None
                and row["reconsolidation_count_reduction_vs_p0"] >= 0.30
            ),
            "wall_reduction_ge_15pct": bool(
                row["wall_reduction_vs_p0"] is not None and row["wall_reduction_vs_p0"] >= 0.15
            ),
            "large_region_recall_ge_0_98": bool(
                metrics.get("large_region_recall_proxy") is not None
                and float(metrics["large_region_recall_proxy"]) >= 0.98
            ),
            "tor_0_5_ge_0_95": bool(
                metrics.get("tor_0_5_proxy") is not None and float(metrics["tor_0_5_proxy"]) >= 0.95
            ),
            "note": "Exact SAM2 parity is treated as a diagnostic only; reference-fidelity proxies are still reported.",
        }
        scored_rows.append(row)

    df_rows = []
    for row in scored_rows:
        metrics = row.get("reference_metrics", {}) or {}
        df_rows.append(
            {
                "variant_id": row.get("variant_id"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "wrapper_wall_time_sec": row.get("wrapper_wall_time_sec"),
                "reconsolidate_call_count": row.get("reconsolidate_call_count"),
                "reconsolidate_runtime_sec": row.get("reconsolidate_runtime_sec"),
                "stream_add_masks_admitted_mask_count": row.get("stream_add_masks_admitted_mask_count"),
                "birth_transaction_commit_count": row.get("birth_transaction_commit_count"),
                "birth_transaction_committed_mask_count": row.get("birth_transaction_committed_mask_count"),
                "birth_transaction_commit_runtime_sec": row.get("birth_transaction_commit_runtime_sec"),
                "birth_transaction_reconsolidate_call_count": row.get("birth_transaction_reconsolidate_call_count"),
                "birth_transaction_max_queue_mask_count": row.get("birth_transaction_max_queue_mask_count"),
                "birth_transaction_max_queue_frame_count": row.get("birth_transaction_max_queue_frame_count"),
                "birth_transaction_max_delay_frames_observed": row.get("birth_transaction_max_delay_frames_observed"),
                "foreground_recall_vs_reference": metrics.get("foreground_recall_vs_reference"),
                "foreground_precision_vs_reference": metrics.get("foreground_precision_vs_reference"),
                "large_region_recall_proxy": metrics.get("large_region_recall_proxy"),
                "tor_0_5_proxy": metrics.get("tor_0_5_proxy"),
                "merge_error_proxy_rate": metrics.get("merge_error_proxy_rate"),
                "reconsolidation_runtime_reduction_vs_p0": row.get("reconsolidation_runtime_reduction_vs_p0"),
                "reconsolidation_count_reduction_vs_p0": row.get("reconsolidation_count_reduction_vs_p0"),
                "wall_reduction_vs_p0": row.get("wall_reduction_vs_p0"),
            }
        )
    csv_path = phase4 / "phase4_override_transaction_variant_table.csv"
    pd.DataFrame(df_rows).to_csv(csv_path, index=False)
    table_json = phase4 / "phase4_override_transaction_variant_table.json"
    write_json(table_json, {"rows": df_rows, "row_count": len(df_rows)})

    completed = [row for row in scored_rows if row.get("status") == "completed"]
    candidates = [
        row
        for row in completed
        if (row.get("phase4_proxy_gate") or {}).get("reconsolidation_runtime_reduction_ge_30pct")
        and (row.get("phase4_proxy_gate") or {}).get("wall_reduction_ge_15pct")
        and (row.get("phase4_proxy_gate") or {}).get("large_region_recall_ge_0_98")
        and (row.get("phase4_proxy_gate") or {}).get("tor_0_5_ge_0_95")
    ]
    best = None
    if completed:
        best = max(
            completed,
            key=lambda row: (
                float((row.get("reference_metrics") or {}).get("tor_0_5_proxy") or -1.0),
                float(row.get("reconsolidation_runtime_reduction_vs_p0") or -999.0),
                float(row.get("wall_reduction_vs_p0") or -999.0),
            ),
        )
    decision = "PASS_PHASE4_OVERRIDE_PROXY_GATE" if candidates else "NO_GO_PHASE4_OVERRIDE_TRANSACTION_PROXY_GATE"
    summary = {
        "schema_version": "stream4d_v107_phase4_override_transaction_sweep_summary_v1",
        "created_unix_time": time.time(),
        "override_reason": "User explicitly requested ignoring the exact parity gate and continuing.",
        "exact_parity_gate_handling": "ignored_as_hard_gate_tracked_as_diagnostic",
        "variant_mode": str(args.variant_mode),
        "phase2_gate_status": "NO_GO_PHASE2_SAM2_MEMORY_PARITY_FAILED_BUT_OVERRIDDEN",
        "phase3_gate_status": "NO_GO_PHASE3_OVERRIDE_PROBATION_PROXY_GATE_BUT_OVERRIDDEN",
        "scene_id": str(args.scene_id),
        "frame_start": int(args.frame_start),
        "frame_stride": int(args.frame_stride),
        "frame_count": int(args.frame_count),
        "config": {"path": rel(config), "sha256": sha256_file(config)},
        "reference_run_root": {"path": rel(reference_root), "summary_sha256": sha256_file(reference_root / "summary.json")},
        "runner": {"path": rel(RUNNER), "sha256": sha256_file(RUNNER)},
        "run_records": run_records,
        "rows": scored_rows,
        "variant_table_csv": rel(csv_path),
        "variant_table_json": rel(table_json),
        "completed_variant_count": int(len(completed)),
        "failed_variant_count": int(sum(1 for row in scored_rows if row.get("status") == "run_failed")),
        "candidate_count": int(len(candidates)),
        "candidate_variant_ids": [str(row["variant_id"]) for row in candidates],
        "best_variant_id": str(best["variant_id"]) if best else "",
        "decision": decision,
    }
    summary_path = phase4 / "phase4_override_transaction_sweep_summary.json"
    write_json(summary_path, summary)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase4_override_transaction_sweep_run_v1",
            "summary": rel(summary_path),
            "decision": decision,
            "completed_variant_count": summary["completed_variant_count"],
            "candidate_count": summary["candidate_count"],
            "best_variant_id": summary["best_variant_id"],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "summary": str(summary_path),
                "decision": decision,
                "completed_variant_count": summary["completed_variant_count"],
                "candidate_count": summary["candidate_count"],
                "best_variant_id": summary["best_variant_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["completed_variant_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
