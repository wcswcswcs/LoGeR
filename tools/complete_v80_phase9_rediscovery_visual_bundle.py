#!/usr/bin/env python3
"""Complete the v80 Phase9 rediscovery visual bundle from existing artifacts.

This script does not create new model evidence. It copies existing visual
artifacts into the plan-required Phase9 layout, records provenance for every
file, and audits that the required panel groups are non-empty and readable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_REDISCOVERY_ROOT = REPORT_ROOT / "phase10_seq01_error_ttt_semantic_alignment_rediscovery_20260622_2030"
DEFAULT_PHASE2_DIRECT_ROOT = REPORT_ROOT / "phase2_direct_hook_enhanced_visual_review"
DEFAULT_GEOMETRY_BRIDGE_ROOT = REPORT_ROOT / "phase9_seq01_ref055_geometry_delta_bridge"
DEFAULT_FALSE_POSITIVE_ROOT = REPORT_ROOT / "phase9_seq01_boundary09_semantic_false_positive_merge_gauge_audit"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_phase9_rediscovery_visual_completion_20260622_2305"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rediscovery-root", type=Path, default=DEFAULT_REDISCOVERY_ROOT)
    parser.add_argument("--phase2-direct-root", type=Path, default=DEFAULT_PHASE2_DIRECT_ROOT)
    parser.add_argument("--geometry-bridge-root", type=Path, default=DEFAULT_GEOMETRY_BRIDGE_ROOT)
    parser.add_argument("--false-positive-root", type=Path, default=DEFAULT_FALSE_POSITIVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--merge-boundary-chunks", default="8,10,12")
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _png_size(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() != ".png" or not path.exists():
        return None, None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _copy_file(src: Path, dst: Path, group: str, source_kind: str) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "group": group,
        "source_kind": source_kind,
        "source_path": str(src),
        "dest_path": str(dst),
        "source_exists": src.exists(),
        "copied": False,
        "bytes": 0,
        "width": None,
        "height": None,
        "ok": False,
    }
    if not src.exists() or not src.is_file():
        return row
    shutil.copy2(src, dst)
    width, height = _png_size(dst)
    size = dst.stat().st_size
    is_image = dst.suffix.lower() in {".png", ".jpg", ".jpeg"}
    row.update(
        {
            "copied": True,
            "bytes": size,
            "width": width,
            "height": height,
            "ok": bool(size > 0 and (not is_image or (width and height))),
        }
    )
    return row


def _copy_group_files(
    source_files: list[tuple[Path, str]],
    out_dir: Path,
    group: str,
    source_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src, dst_name in source_files:
        rows.append(_copy_file(src, out_dir / group / dst_name, group, source_kind))
    return rows


def _existing_pngs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.png") if path.is_file())


def _visual_sources(args: argparse.Namespace) -> dict[str, list[tuple[Path, str, str]]]:
    short = [
        (
            args.phase2_direct_root / "short_direct_hook_panels" / name,
            name,
            "phase2_direct_short_qk_panel",
        )
        for name in ("01_chunk009_bad.png", "01_chunk010_bad.png", "01_chunk013_bad.png")
    ]
    mid: list[tuple[Path, str, str]] = [
        (
            args.rediscovery_root / "mid_swa_qkv_visual_panels" / "frame_000232_chunk_008_swa_ttt_alignment.png",
            "frame_000232_chunk_008_swa_ttt_alignment.png",
            "phase10_mid_swa_ttt_panel",
        )
    ]
    mid.extend(
        (
            args.phase2_direct_root / "mid_direct_hook_panels" / name,
            name,
            "phase2_direct_mid_swa_panel",
        )
        for name in ("01_pair008_009_bad.png", "01_pair010_011_bad.png", "01_pair013_014_bad.png")
    )
    long = [
        (path, path.name, "phase10_long_ttt_branch_panel")
        for path in _existing_pngs(args.rediscovery_root / "long_ttt_branch_visual_panels")
    ]

    merge: list[tuple[Path, str, str]] = []
    for chunk in _parse_chunks(args.merge_boundary_chunks):
        plot_dir = args.geometry_bridge_root / f"chunk{chunk:02d}" / "plots"
        merge.append(
            (
                plot_dir / "error_over_frame.png",
                f"chunk{chunk:02d}_error_over_frame.png",
                "geometry_delta_bridge_error_map_proxy",
            )
        )
        merge.append(
            (
                plot_dir / "trajectory_error_map_xz.png",
                f"chunk{chunk:02d}_trajectory_error_map_xz.png",
                "geometry_delta_bridge_error_map_proxy",
            )
        )

    return {
        "short_qk_pair_visual_panels": short,
        "mid_swa_qkv_visual_panels": mid,
        "long_ttt_branch_visual_panels": long,
        "merge_boundary_visual_panels": merge,
    }


def _copy_required_docs(args: argparse.Namespace) -> list[dict[str, Any]]:
    docs = [
        (args.rediscovery_root / "failed_case_to_visual_question.csv", "failed_case_to_visual_question.csv"),
        (args.rediscovery_root / "visual_review.csv", "visual_review.csv"),
        (args.rediscovery_root / "visual_insight.md", "visual_insight.md"),
        (args.rediscovery_root / "new_semantic_memory_hypothesis_bank.md", "new_semantic_memory_hypothesis_bank.md"),
        (args.rediscovery_root / "canary_error_ttt_semantic_alignment_rows.csv", "canary_error_ttt_semantic_alignment_rows.csv"),
        (args.false_positive_root / "semantic_false_positive_report.md", "merge_boundary_visual_panels/semantic_false_positive_report.md"),
        (args.false_positive_root / "semantic_false_positive_rows.csv", "merge_boundary_visual_panels/semantic_false_positive_rows.csv"),
    ]
    return [
        _copy_file(src, args.out_dir / dst_name, "required_docs", "phase9_rediscovery_or_boundary_report")
        for src, dst_name in docs
    ]


def _group_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        group = str(row["group"])
        out.setdefault(group, {"count": 0, "ok": 0})
        out[group]["count"] += 1
        if row.get("ok"):
            out[group]["ok"] += 1
    return out


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v80 Phase9 Rediscovery Visual Completion",
        "",
        f"- gate_pass: `{summary['gate_pass']}`",
        f"- visual_audit_gate_pass: `{summary['visual_audit_gate_pass']}`",
        f"- diagnostic_only: `{summary['diagnostic_only']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        f"- v80_goal_achieved: `{summary['v80_goal_achieved']}`",
        "",
        "## Required Groups",
        "",
        "| group | copied | ok |",
        "|---|---:|---:|",
    ]
    for group, counts in summary["group_counts"].items():
        lines.append(f"| {group} | {counts['count']} | {counts['ok']} |")
    lines.extend(
        [
            "",
            "## Provenance Note",
            "",
            "Merge-boundary panels are copied from existing geometry-delta bridge error-map plots. They are visual diagnostics for boundary/gauge failure, not new runtime evidence.",
            "",
            "## Files",
            "",
            "| group | source_kind | ok | dest | source |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['source_kind']} | {row['ok']} | {row['dest_path']} | {row['source_path']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for group, entries in _visual_sources(args).items():
        for src, dst_name, source_kind in entries:
            rows.append(_copy_file(src, args.out_dir / group / dst_name, group, source_kind))
    rows.extend(_copy_required_docs(args))

    group_counts = _group_counts(rows)
    required_panel_groups = [
        "short_qk_pair_visual_panels",
        "mid_swa_qkv_visual_panels",
        "long_ttt_branch_visual_panels",
        "merge_boundary_visual_panels",
    ]
    missing_groups = [
        group for group in required_panel_groups if group_counts.get(group, {}).get("ok", 0) <= 0
    ]
    bad_files = [row for row in rows if not row.get("ok")]
    required_docs_ok = group_counts.get("required_docs", {}).get("ok", 0) >= 5
    gate_pass = not missing_groups and required_docs_ok and not bad_files
    summary = {
        "schema": "acl2_v80_phase9_rediscovery_visual_completion_v1",
        "out_dir": args.out_dir,
        "rediscovery_root": args.rediscovery_root,
        "phase2_direct_root": args.phase2_direct_root,
        "geometry_bridge_root": args.geometry_bridge_root,
        "false_positive_root": args.false_positive_root,
        "gate_pass": gate_pass,
        "visual_audit_gate_pass": gate_pass,
        "required_phase9_panel_sets_present": not missing_groups,
        "missing_required_panel_groups": missing_groups,
        "bad_file_count": len(bad_files),
        "bad_files": bad_files,
        "required_docs_ok": required_docs_ok,
        "group_counts": group_counts,
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "merge_boundary_source_note": (
            "merge_boundary_visual_panels use existing geometry-delta bridge error-over-frame and trajectory error-map plots"
        ),
    }

    _write_csv(args.out_dir / "visual_manifest.csv", rows)
    _write_json(args.out_dir / "visual_integrity_audit.json", summary)
    _write_json(args.out_dir / "rediscovery_visual_completion_summary.json", summary)
    _write_report(args.out_dir / "visual_completion_report.md", summary, rows)
    print(json.dumps(_jsonable({"out_dir": args.out_dir, **summary}), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
