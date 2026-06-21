#!/usr/bin/env python3
"""Batch-scan current v78 five-chunk TTT baselines.

This diagnostic launcher runs the existing single-window v78 TTT smoke on many
candidate five-chunk windows, then aggregates LW0 native and LW1 TTT baseline
metrics. It does not change method behavior or claim promotion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_TABLE = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "bad_window_selection/v1_existing_trajectories/bad_5chunk_window_table.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti_acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase4_ttt_current_baseline_window_scan_v1"
)
DEFAULT_SINGLE_RUNNER = Path("tools/run_v78_ttt_long_window_regime_action_smoke.py")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_ratio(native: Any, ttt: Any) -> float | None:
    n = _finite(native)
    t = _finite(ttt)
    if n is None or t is None or abs(n) < 1e-12:
        return None
    return float((n - t) / abs(n))


def _parse_gpu_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--gpus must contain at least one GPU id")
    return out


def _expand_window_chunks(text: str) -> str:
    chunks = [p.strip() for p in str(text).split("-") if p.strip()]
    if len(chunks) == 2:
        a, b = int(chunks[0]), int(chunks[1])
        if b < a:
            raise ValueError(f"bad window chunk range: {text!r}")
        chunks = [str(i) for i in range(a, b + 1)]
    if len(chunks) != 5:
        raise ValueError(f"expected exactly 5 chunks, got {text!r}")
    ints = [int(c) for c in chunks]
    if ints != list(range(min(ints), max(ints) + 1)):
        raise ValueError(f"window chunks must be contiguous: {text!r}")
    return "-".join(str(i) for i in ints)


def _compact_window(window_chunks: str) -> str:
    ints = [int(p) for p in str(window_chunks).split("-")]
    if len(ints) == 5 and ints == list(range(ints[0], ints[0] + 5)):
        return f"{ints[0]}-{ints[-1]}"
    return "-".join(str(i) for i in ints)


def _parse_window_spec(spec: str) -> dict[str, Any]:
    if ":" not in spec:
        raise ValueError(f"window spec must be SEQ:START-END or SEQ:C0-C1-C2-C3-C4, got {spec!r}")
    seq, chunk_text = spec.split(":", 1)
    seq = seq.strip().zfill(2)
    window_chunks = _expand_window_chunks(chunk_text)
    chunks = [int(p) for p in window_chunks.split("-")]
    return {
        "seq": seq,
        "window_chunks": window_chunks,
        "chunk_range": _compact_window(window_chunks),
        "start_chunk_id": int(chunks[0]),
        "end_chunk_id": int(chunks[-1]),
        "source": "manual",
    }


def _read_old_table(path: Path, max_windows: int, exclude: set[tuple[str, str]]) -> list[dict[str, Any]]:
    if max_windows <= 0:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set(exclude)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = str(row.get("sequence") or "").zfill(2)
            window_chunks = _expand_window_chunks(str(row.get("window_chunks") or ""))
            key = (seq, window_chunks)
            if key in seen:
                continue
            seen.add(key)
            chunks = [int(p) for p in window_chunks.split("-")]
            rows.append(
                {
                    "seq": seq,
                    "window_chunks": window_chunks,
                    "chunk_range": _compact_window(window_chunks),
                    "start_chunk_id": int(chunks[0]),
                    "end_chunk_id": int(chunks[-1]),
                    "source": "old_bad_5chunk_table",
                    "old_rank_metric_window5_joint_sim3_rmse_m": _finite(
                        row.get("window5_joint_sim3_rmse_m")
                    ),
                    "old_run": row.get("run"),
                    "old_trajectory": row.get("trajectory"),
                }
            )
            if len(rows) >= max_windows:
                break
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_child_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("case")): dict(row) for row in csv.DictReader(handle)}


def _build_command(args: argparse.Namespace, window: dict[str, Any], gpu: int, out_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(args.single_runner),
        "--seq",
        str(window["seq"]),
        "--chunks",
        str(window["chunk_range"]),
        "--cases",
        "LW0_READPATH_NATIVE,LW1_TTT_SEMANTIC_BASE",
        "--baseline",
        "LW1_TTT_SEMANTIC_BASE",
        "--native-baseline",
        "LW0_READPATH_NATIVE",
        "--gpus",
        str(gpu),
        "--output-root",
        str(out_dir),
        "--disable-ttt-compile",
        str(int(args.disable_ttt_compile)),
        "--min-coverage",
        str(args.min_coverage),
    ]
    if bool(args.skip_existing):
        cmd.append("--skip-existing")
    return cmd


def _window_output_dir(output_root: Path, window: dict[str, Any]) -> Path:
    return output_root / f"seq{window['seq']}_chunks{window['chunk_range'].replace('-', '_')}"


def _run_window(job: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(job["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = out_dir / "window_launcher.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    start_t = time.time()
    with launcher_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=job["workdir"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job["returncode"] = int(proc.returncode)
    job["duration_sec"] = float(time.time() - start_t)
    job["launcher_log"] = str(launcher_log)
    return job


def _aggregate(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        out_dir = Path(job["output_root"])
        by_case = _read_child_metrics(out_dir / "long_window_ttt_regime_action_metrics.csv")
        native = by_case.get("LW0_READPATH_NATIVE", {})
        ttt = by_case.get("LW1_TTT_SEMANTIC_BASE", {})
        native_rmse = _finite(native.get("window5_joint_sim3_rmse_m"))
        ttt_rmse = _finite(ttt.get("window5_joint_sim3_rmse_m"))
        row = {
            "seq": job["seq"],
            "window_chunks": job["window_chunks"],
            "chunk_range": job["chunk_range"],
            "source": job.get("source"),
            "old_rank_metric_window5_joint_sim3_rmse_m": job.get(
                "old_rank_metric_window5_joint_sim3_rmse_m"
            ),
            "old_run": job.get("old_run"),
            "gpu": job["gpu"],
            "returncode": int(job.get("returncode") or 0),
            "duration_sec": job.get("duration_sec"),
            "native_window5_joint_sim3_rmse_m": native_rmse,
            "ttt_window5_joint_sim3_rmse_m": ttt_rmse,
            "ttt_minus_native_rmse_m": None
            if native_rmse is None or ttt_rmse is None
            else float(ttt_rmse - native_rmse),
            "ttt_improvement_vs_native_ratio": _safe_ratio(native_rmse, ttt_rmse),
            "native_window5_subchunk_scale_cv": _finite(native.get("window5_subchunk_scale_cv")),
            "ttt_window5_subchunk_scale_cv": _finite(ttt.get("window5_subchunk_scale_cv")),
            "native_memory_ttt_mean_rel_diff_mean": _finite(
                native.get("memory_ttt_mean_rel_diff_mean")
            ),
            "ttt_memory_ttt_mean_rel_diff_mean": _finite(ttt.get("memory_ttt_mean_rel_diff_mean")),
            "output_root": str(out_dir),
            "metrics_csv": str(out_dir / "long_window_ttt_regime_action_metrics.csv"),
            "decision_json": str(out_dir / "long_window_ttt_regime_action_decision.json"),
            "launcher_log": job.get("launcher_log"),
        }
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: _finite(row.get("ttt_window5_joint_sim3_rmse_m")) or -math.inf,
        reverse=True,
    )
    ok_rows = [row for row in rows if int(row.get("returncode") or 0) == 0]
    summary = {
        "schema": "acl2_v78_ttt_current_baseline_window_scan_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "promotion_blocker": (
            "current-baseline target discovery only; any action still requires held-out "
            "five-chunk improvement before promotion"
        ),
        "args": _jsonable(vars(args)),
        "num_windows": len(rows),
        "num_ok": len(ok_rows),
        "all_jobs_ok": bool(len(ok_rows) == len(rows)),
        "rows_ranked_by_ttt_window5_joint_sim3_rmse_m": ranked,
        "top_current_bad_windows": ranked[: int(args.top_k_report)],
    }
    _write_csv(args.output_root / "current_baseline_window_scan_rows.csv", ranked)
    (args.output_root / "current_baseline_window_scan_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default="", help="Comma-separated specs like 02:18-22,00:139-143.")
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_OLD_TABLE)
    parser.add_argument(
        "--old-table-top-k",
        type=int,
        default=0,
        help="Append this many unique windows from the old bad_5chunk table.",
    )
    parser.add_argument(
        "--exclude-windows",
        default="",
        help="Comma-separated SEQ:WINDOW specs to exclude from old-table selection.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--single-runner", type=Path, default=DEFAULT_SINGLE_RUNNER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--disable-ttt-compile", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--top-k-report", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpus = _parse_gpu_list(args.gpus)

    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for spec in [p.strip() for p in str(args.windows or "").split(",") if p.strip()]:
        item = _parse_window_spec(spec)
        key = (item["seq"], item["window_chunks"])
        if key not in seen:
            seen.add(key)
            windows.append(item)

    exclude: set[tuple[str, str]] = set(seen)
    for spec in [p.strip() for p in str(args.exclude_windows or "").split(",") if p.strip()]:
        item = _parse_window_spec(spec)
        exclude.add((item["seq"], item["window_chunks"]))

    if int(args.old_table_top_k) > 0:
        old_rows = _read_old_table(args.candidate_csv, int(args.old_table_top_k), exclude)
        for item in old_rows:
            key = (item["seq"], item["window_chunks"])
            if key not in seen:
                seen.add(key)
                windows.append(item)

    if not windows:
        raise ValueError("no windows requested; pass --windows or --old-table-top-k > 0")

    jobs: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        out_dir = _window_output_dir(args.output_root, window)
        metrics_csv = out_dir / "long_window_ttt_regime_action_metrics.csv"
        skipped = bool(args.skip_existing and metrics_csv.is_file())
        gpu = int(gpus[idx % len(gpus)])
        cmd = _build_command(args, window, gpu, out_dir)
        jobs.append(
            {
                **window,
                "gpu": gpu,
                "output_root": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "workdir": str(args.workdir),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            }
        )

    manifest = {
        "schema": "acl2_v78_ttt_current_baseline_window_scan_manifest_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "args": _jsonable(vars(args)),
        "jobs": jobs,
    }
    manifest_path = args.output_root / "current_baseline_window_scan_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"planned_windows={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        print(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2, sort_keys=True))
        return

    completed: list[dict[str, Any]] = [job for job in jobs if job["skipped"]]
    run_jobs = [job for job in jobs if not job["skipped"]]
    jobs_by_gpu: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def run_gpu_queue(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in queue:
            result = _run_window(job)
            results.append(result)
            print(
                f"finished seq={result['seq']} chunks={result['chunk_range']} "
                f"gpu={result['gpu']} returncode={result['returncode']} "
                f"duration_sec={result['duration_sec']:.1f}"
            )
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(gpus))) as pool:
        futures = [pool.submit(run_gpu_queue, queue) for queue in jobs_by_gpu.values() if queue]
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())

    by_key = {(str(job["seq"]), str(job["window_chunks"])): job for job in completed}
    ordered_completed = [by_key[(str(job["seq"]), str(job["window_chunks"]))] for job in jobs]
    run_summary = {
        "schema": "acl2_v78_ttt_current_baseline_window_scan_run_summary_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "jobs": ordered_completed,
        "all_jobs_ok": bool(all(int(job.get("returncode") or 0) == 0 for job in ordered_completed)),
    }
    run_summary_path = args.output_root / "current_baseline_window_scan_run_summary.json"
    run_summary_path.write_text(
        json.dumps(_jsonable(run_summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _aggregate(args, ordered_completed)
    print(
        json.dumps(
            {
                "all_jobs_ok": summary["all_jobs_ok"],
                "rows_csv": str(args.output_root / "current_baseline_window_scan_rows.csv"),
                "summary_json": str(args.output_root / "current_baseline_window_scan_summary.json"),
                "top_current_bad_windows": summary["top_current_bad_windows"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
