#!/usr/bin/env python3
"""Run Video Masklet v2 non-regression audits and dense review sheets.

This wrapper deliberately combines machine-readable gates with step=1 visual
artifacts. It is not a replacement for manual frame-by-frame review; it makes
that review reproducible and prevents a local fix from silently regressing
older high-risk windows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent


DEFAULT_KITTI_WINDOWS = (
    "early_car:0-30,"
    "local75_76:70-80,"
    "road_sign_081_098:81-98,"
    "short_track_347_358:347-358,"
    "gapfill_430_440:430-440,"
    "sign_689_707:689-707,"
    "sign_852_905:852-905,"
    "edge_static_999_1067:999-1067"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--cache_root", default="", help="Optional Stage-C cache root for proposal-to-final coverage.")
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--windows", default=DEFAULT_KITTI_WINDOWS)
    parser.add_argument("--frames_per_sheet", type=int, default=36)
    parser.add_argument("--thing_labels", default="car,person,traffic sign,pole")
    parser.add_argument("--stuff_labels", default="road,sky,vegetation,grass,fence,building,billboard_or_bulletin_board,pole,handrail_or_fence")
    parser.add_argument("--max_thing_adjacent", type=int, default=0)
    parser.add_argument("--max_thing_gap", type=int, default=0)
    parser.add_argument("--max_thing_low_conflict_same", type=int, default=0)
    parser.add_argument("--max_stuff_adjacent", type=int, default=-1, help="<0 disables stuff temporal gate failure.")
    parser.add_argument("--coverage_labels", default="car,traffic sign,person,pole")
    parser.add_argument("--coverage_states", default="confirmed,tentative")
    parser.add_argument("--coverage_min_conf", type=float, default=0.0)
    parser.add_argument("--allow_command_failures", type=int, default=0)
    return parser.parse_args()


def parse_windows(raw: str) -> List[Tuple[str, int, int]]:
    windows: List[Tuple[str, int, int]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, span = item.split(":", 1)
        else:
            name, span = f"window_{len(windows):02d}", item
        if "-" not in span:
            raise ValueError(f"Window must be name:start-end, got {item!r}")
        start_s, end_s = span.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"Invalid window {item!r}: end < start")
        windows.append((name.strip() or f"window_{len(windows):02d}", start, end))
    return windows


def run_cmd(cmd: List[str], cwd: Path, log_path: Path, allow_fail: bool) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{TOOLS_ROOT}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nlog={log_path}")
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "log": str(log_path),
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def frame_chunks(start: int, end: int, size: int) -> List[List[int]]:
    frames = list(range(int(start), int(end) + 1))
    size = max(int(size), 1)
    return [frames[i : i + size] for i in range(0, len(frames), size)]


def render_dense_sheets(args: argparse.Namespace, out_dir: Path, allow_fail: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, start, end in parse_windows(args.windows):
        for page_idx, frames in enumerate(frame_chunks(start, end, int(args.frames_per_sheet))):
            frame_arg = ",".join(str(v) for v in frames)
            for only_things, suffix in ((1, "things"), (0, "all")):
                output_jpg = out_dir / "dense_sheets" / f"{name}_{suffix}_p{page_idx:02d}_{frames[0]:04d}_{frames[-1]:04d}.jpg"
                cmd = [
                    sys.executable,
                    str(TOOLS_ROOT / "render_sparse_track_id_contact.py"),
                    "--input_pt",
                    args.candidate_pt,
                    "--input_video",
                    args.input_video,
                    "--output_jpg",
                    str(output_jpg),
                    "--frames",
                    frame_arg,
                    "--start_frame",
                    str(int(args.start_frame)),
                    "--processing_max_side",
                    str(int(args.processing_max_side)),
                    "--cols",
                    "6",
                    "--only_things",
                    str(int(only_things)),
                ]
                log = out_dir / "logs" / f"render_{name}_{suffix}_p{page_idx:02d}.log"
                result = run_cmd(cmd, REPO_ROOT, log, allow_fail)
                rows.append(
                    {
                        "window": name,
                        "view": suffix,
                        "page": int(page_idx),
                        "frames": [int(frames[0]), int(frames[-1])],
                        "output_jpg": str(output_jpg),
                        "returncode": int(result["returncode"]),
                        "log": str(log),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allow_fail = bool(int(args.allow_command_failures))
    commands: List[Dict[str, Any]] = []

    thing_dir = out_dir / "temporal_thing"
    commands.append(
        run_cmd(
            [
                sys.executable,
                str(TOOLS_ROOT / "audit_sparse_id_temporal_consistency.py"),
                "--input_pt",
                args.candidate_pt,
                "--output_dir",
                str(thing_dir),
                "--labels",
                args.thing_labels,
                "--same_frame_min_mask_iou",
                "0.25",
                "--same_frame_min_mask_containment",
                "0.70",
                "--max_rows",
                "20000",
            ],
            REPO_ROOT,
            out_dir / "logs" / "temporal_thing.log",
            allow_fail,
        )
    )

    stuff_dir = out_dir / "temporal_stuff"
    commands.append(
        run_cmd(
            [
                sys.executable,
                str(TOOLS_ROOT / "audit_sparse_id_temporal_consistency.py"),
                "--input_pt",
                args.candidate_pt,
                "--output_dir",
                str(stuff_dir),
                "--labels",
                args.stuff_labels,
                "--same_frame_min_mask_iou",
                "0.25",
                "--same_frame_min_mask_containment",
                "0.70",
                "--max_rows",
                "20000",
            ],
            REPO_ROOT,
            out_dir / "logs" / "temporal_stuff.log",
            allow_fail,
        )
    )

    provenance_dir = out_dir / "provenance"
    commands.append(
        run_cmd(
            [
                sys.executable,
                str(TOOLS_ROOT / "audit_sparse_semantic_provenance.py"),
                "--input_pt",
                args.candidate_pt,
                "--output_dir",
                str(provenance_dir),
                "--focus_labels",
                f"{args.thing_labels},{args.stuff_labels}",
                "--metrics_json",
                str(args.metrics_json),
            ],
            REPO_ROOT,
            out_dir / "logs" / "provenance.log",
            allow_fail,
        )
    )

    coverage_summary: Dict[str, Any] = {"enabled": False}
    if str(args.cache_root).strip():
        coverage_dir = out_dir / "proposal_to_final"
        commands.append(
            run_cmd(
                [
                    sys.executable,
                    str(TOOLS_ROOT / "audit_proposal_to_final_coverage.py"),
                    "--cache_root",
                    args.cache_root,
                    "--sparse_pt",
                    args.candidate_pt,
                    "--output_dir",
                    str(coverage_dir),
                    "--labels",
                    args.coverage_labels,
                    "--states",
                    args.coverage_states,
                    "--min_conf",
                    str(float(args.coverage_min_conf)),
                ],
                REPO_ROOT,
                out_dir / "logs" / "proposal_to_final.log",
                allow_fail,
            )
        )
        coverage_summary = load_json(coverage_dir / "summary.json")
        coverage_summary["enabled"] = True

    dense_sheets = render_dense_sheets(args, out_dir, allow_fail)

    thing_summary = load_json(thing_dir / "summary.json")
    stuff_summary = load_json(stuff_dir / "summary.json")
    provenance_summary = load_json(provenance_dir / "provenance_summary.json")
    failures: List[str] = []
    if int(thing_summary.get("adjacent_switch_candidate_count", 0)) > int(args.max_thing_adjacent):
        failures.append("thing_adjacent_switch_exceeds_gate")
    if int(thing_summary.get("gap_handoff_candidate_count", 0)) > int(args.max_thing_gap):
        failures.append("thing_gap_handoff_exceeds_gate")
    if int(thing_summary.get("same_frame_low_conflict_candidate_count", 0)) > int(args.max_thing_low_conflict_same):
        failures.append("thing_low_conflict_duplicate_exceeds_gate")
    if int(args.max_stuff_adjacent) >= 0 and int(stuff_summary.get("adjacent_switch_candidate_count", 0)) > int(args.max_stuff_adjacent):
        failures.append("stuff_adjacent_switch_exceeds_gate")
    for command in commands:
        if int(command["returncode"]) != 0:
            failures.append("subcommand_failed")
            break
    for sheet in dense_sheets:
        if int(sheet["returncode"]) != 0:
            failures.append("dense_sheet_render_failed")
            break

    summary = {
        "candidate_pt": str(args.candidate_pt),
        "input_video": str(args.input_video),
        "start_frame": int(args.start_frame),
        "windows": [
            {"name": name, "start": int(start), "end": int(end)}
            for name, start, end in parse_windows(args.windows)
        ],
        "gate_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "thing_temporal_summary": thing_summary,
        "stuff_temporal_summary": stuff_summary,
        "provenance_summary": provenance_summary,
        "coverage_summary": coverage_summary,
        "dense_sheets": dense_sheets,
        "commands": commands,
    }
    (out_dir / "nonregression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Video Masklet Non-regression Gate",
        "",
        f"candidate: `{args.candidate_pt}`",
        f"status: `{summary['gate_status']}`",
        "",
        "## Thing temporal",
        "",
        f"- adjacent: {thing_summary.get('adjacent_switch_candidate_count')}",
        f"- same-frame low-conflict: {thing_summary.get('same_frame_low_conflict_candidate_count')}",
        f"- gap: {thing_summary.get('gap_handoff_candidate_count')}",
        "",
        "## Stuff temporal",
        "",
        f"- adjacent: {stuff_summary.get('adjacent_switch_candidate_count')}",
        f"- same-frame low-conflict: {stuff_summary.get('same_frame_low_conflict_candidate_count')}",
        f"- gap: {stuff_summary.get('gap_handoff_candidate_count')}",
        "",
        "## Dense sheets",
        "",
    ]
    for sheet in dense_sheets:
        report_lines.append(
            f"- {sheet['window']} {sheet['view']} p{sheet['page']:02d} "
            f"frames {sheet['frames'][0]}-{sheet['frames'][1]}: `{sheet['output_jpg']}`"
        )
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- {failure}" for failure in failures)
    (out_dir / "nonregression_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
