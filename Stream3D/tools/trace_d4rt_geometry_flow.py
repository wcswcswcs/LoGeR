from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.object_tube_io import MergeGeometryError, TubeRecord, assert_merge_geometry_valid


REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET_FILES = [
    "stream4d_native/d4rt_scene_builder.py",
    "stream4d_native/self_stitch.py",
    "stream4d_native/sim3.py",
    "geometry_provider/d4rt_carrier_provider.py",
    "tools/run_v23_d4rt_reconstruction_quality_audit.py",
    "tools/export_v21_3_occupancy_carrier_cache.py",
]

SUPPLEMENTAL_FILES = [
    "stream4d_native/chunk_alignment.py",
    "stream4d_native/occupancy_dense_tracker.py",
    "stream4d_native/occupancy_state.py",
    "stream4d_native/measurement_bank.py",
    "stream4d_native/tube_cover.py",
    "stream4d_native/signed_tube_graph.py",
    "stream4d_native/tube_partition.py",
    "stream4d_native/tube_memory.py",
    "stream4d_native/object_tube_io.py",
    "stream4d/d4rt_adapter.py",
    "stream4d/carrier_store.py",
    "stream4d/carrier_sampler.py",
    "geometry_provider/base.py",
    "geometry_provider/common.py",
    "tools/run_v21_3_stream3d_provider_replacement.py",
    "tests/test_native_chunking_and_sim3.py",
    "tests/test_v21_3_geometry_provider.py",
    "tests/test_native_occupancy_and_builder.py",
    "tests/test_v24_scale_consistency.py",
]

STREAM4D_MERGE_FILES = [
    "stream4d/evidence_graph.py",
    "stream4d/object_memory.py",
    "stream4d/measurement_bank.py",
]


def _rel(path: str) -> Path:
    return REPO_ROOT / path


def _exists_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        full = _rel(path)
        rows.append({"path": path, "present": full.exists(), "bytes": full.stat().st_size if full.exists() else 0})
    return rows


def _grep_file(path: str, patterns: list[str]) -> list[dict[str, Any]]:
    full = _rel(path)
    if not full.exists():
        return []
    text = full.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: list[dict[str, Any]] = []
    compiled = [(name, re.compile(pattern)) for name, pattern in zip(patterns, patterns)]
    for lineno, line in enumerate(text, start=1):
        for name, regex in compiled:
            if regex.search(line):
                hits.append({"path": path, "line": lineno, "pattern": name, "text": line.strip()[:240]})
    return hits


def _scan_static() -> dict[str, Any]:
    target_rows = _exists_rows(TARGET_FILES)
    supplemental_rows = _exists_rows(SUPPLEMENTAL_FILES)
    merge_rows = _exists_rows(STREAM4D_MERGE_FILES)
    missing_native_merge = [
        row["path"]
        for row in supplemental_rows
        if row["path"].startswith("stream4d_native/")
        and row["path"].split("/", 1)[1]
        in {"measurement_bank.py", "tube_cover.py", "signed_tube_graph.py", "tube_partition.py", "tube_memory.py"}
        and not row["present"]
    ]
    geometry_hits: list[dict[str, Any]] = []
    for path in TARGET_FILES + STREAM4D_MERGE_FILES:
        geometry_hits.extend(
            _grep_file(
                path,
                [
                    r"xyz_canonical",
                    r"xyz_local",
                    r"xyz_ref",
                    r"carrier_id",
                    r"persistent_tube_id",
                    r"eval_gt_sim3",
                    r"eval[-_]?sim3",
                    r"ScanNet",
                    r"load_depth",
                    r"load_pose",
                ],
            )
        )
    builder_text = _rel("stream4d_native/d4rt_scene_builder.py").read_text(encoding="utf-8", errors="replace")
    suspicious_index_matching = bool(re.search(r"prev_tubes\s*\[\s*i\s*\]|curr_tubes\s*\[\s*i\s*\]|range\s*\(\s*count\s*\)", builder_text))
    return {
        "target_files": target_rows,
        "supplemental_files": supplemental_rows,
        "stream4d_merge_files": merge_rows,
        "missing_native_merge_modules": missing_native_merge,
        "native_merge_modules_present": len(missing_native_merge) == 0,
        "geometry_keyword_hits": geometry_hits,
        "builder_suspicious_index_matching": suspicious_index_matching,
        "mask_merge_audit_status": "complete" if len(missing_native_merge) == 0 else "blocked_by_missing_native_merge_modules",
    }


def _record(
    *,
    tube_id: int,
    chunk_id: int,
    submap_id: int,
    coordinate_frame: str,
    alignment_source: str,
    allow_metric_merge: bool,
    pass_gate: bool,
) -> TubeRecord:
    return TubeRecord(
        tube_id=tube_id,
        persistent_tube_id=tube_id,
        chunk_id=chunk_id,
        submap_id=submap_id,
        source_frame_global=0,
        source_xy=(0, 0),
        source_uv=(0.0, 0.0),
        target_frames_global=np.asarray([0, 1], dtype=np.int64),
        uv=np.zeros((2, 2), dtype=np.float32),
        visibility=np.ones((2,), dtype=np.float32),
        confidence=np.ones((2,), dtype=np.float32),
        xyz_local=np.zeros((2, 3), dtype=np.float32),
        xyz_ref0=np.zeros((2, 3), dtype=np.float32),
        xyz_canonical=np.zeros((2, 3), dtype=np.float32) if coordinate_frame == "d4rt_canonical" else None,
        T_chunk_to_canonical={"scale": 1.0, "rot": np.eye(3).tolist(), "trans": [0.0, 0.0, 0.0]},
        alignment_quality={"pass_gate": bool(pass_gate)},
        coordinate_frame=coordinate_frame,
        scale_status="canonical" if coordinate_frame == "d4rt_canonical" else "chunk_local",
        allow_metric_merge=bool(allow_metric_merge),
        alignment_source=alignment_source,
        transform_id=f"trace_{tube_id}",
    )


def _run_guard_trace() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = [
        {
            "case": "same_chunk_local_allowed",
            "a": _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="chunk_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=2, chunk_id=0, submap_id=0, coordinate_frame="chunk_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "expect_pass": True,
        },
        {
            "case": "cross_chunk_canonical_self_sim3_allowed",
            "a": _record(tube_id=3, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=4, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True),
            "expect_pass": True,
        },
        {
            "case": "cross_chunk_local_blocked",
            "a": _record(tube_id=5, chunk_id=0, submap_id=0, coordinate_frame="chunk_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=6, chunk_id=1, submap_id=0, coordinate_frame="chunk_local", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True),
            "expect_pass": False,
        },
        {
            "case": "cross_chunk_ref0_blocked",
            "a": _record(tube_id=7, chunk_id=0, submap_id=0, coordinate_frame="ref0_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=8, chunk_id=1, submap_id=0, coordinate_frame="ref0_local", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True),
            "expect_pass": False,
        },
        {
            "case": "cross_submap_blocked",
            "a": _record(tube_id=9, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=10, chunk_id=1, submap_id=1, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True),
            "expect_pass": False,
        },
        {
            "case": "eval_aligned_blocked",
            "a": _record(tube_id=11, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="eval_gt_sim3", allow_metric_merge=False, pass_gate=True),
            "b": _record(tube_id=12, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True),
            "expect_pass": False,
        },
        {
            "case": "weak_alignment_blocked",
            "a": _record(tube_id=13, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True),
            "b": _record(tube_id=14, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=False, pass_gate=False),
            "expect_pass": False,
        },
    ]
    events: list[dict[str, Any]] = [
        {
            "event_type": "create_geometry",
            "tensor_name": "xyz_local",
            "coordinate_frame": "chunk_local",
            "chunk_id": 0,
            "submap_id": 0,
            "transform_id": None,
            "alignment_source": "none",
            "used_for": "tube_extraction",
            "is_method_path": True,
            "is_eval_only": False,
        },
        {
            "event_type": "create_geometry",
            "tensor_name": "xyz_ref0",
            "coordinate_frame": "ref0_local",
            "chunk_id": 0,
            "submap_id": 0,
            "transform_id": None,
            "alignment_source": "none",
            "used_for": "self_stitch",
            "is_method_path": True,
            "is_eval_only": False,
        },
        {
            "event_type": "transform_geometry",
            "tensor_name": "xyz_canonical",
            "coordinate_frame": "d4rt_canonical",
            "chunk_id": 1,
            "submap_id": 0,
            "transform_id": "trace_4",
            "alignment_source": "d4rt_self_sim3",
            "used_for": "mask_merge",
            "is_method_path": True,
            "is_eval_only": False,
        },
    ]
    for case in cases:
        try:
            event = assert_merge_geometry_valid(case["a"], case["b"], case["case"])
            event["case"] = case["case"]
            event["expected_pass"] = bool(case["expect_pass"])
            event["unexpected"] = not bool(case["expect_pass"])
        except MergeGeometryError as exc:
            try:
                event = json.loads(str(exc))
            except json.JSONDecodeError:
                event = {"guard_pass": False, "guard_reason": str(exc)}
            event["case"] = case["case"]
            event["expected_pass"] = bool(case["expect_pass"])
            event["unexpected"] = bool(case["expect_pass"])
        frame = event.get("coordinate_frame_used", "unknown")
        tensor_name = {
            "chunk_local": "xyz_local",
            "ref0_local": "xyz_ref0",
            "d4rt_canonical": "xyz_canonical",
            "eval_scannet": "xyz_eval_aligned",
        }.get(str(frame), "unknown")
        event.update(
            {
                "event_type": "read_geometry_for_merge",
                "tensor_name": tensor_name,
                "coordinate_frame": frame,
                "chunk_id": [event.get("chunk_i"), event.get("chunk_j")],
                "submap_id": [event.get("submap_i"), event.get("submap_j")],
                "transform_id": [event.get("transform_i"), event.get("transform_j")],
                "used_for": "mask_merge",
                "is_method_path": True,
                "is_eval_only": event.get("guard_reason") == "eval_aligned_geometry_forbidden",
            }
        )
        events.append(event)
    guard_events = [event for event in events if event.get("event_type") == "read_geometry_for_merge"]
    summary = {
        "runtime_event_count": len(events),
        "runtime_case_count": len(guard_events),
        "runtime_guard_pass_count": sum(1 for event in guard_events if event.get("guard_pass")),
        "runtime_guard_block_count": sum(1 for event in guard_events if not event.get("guard_pass")),
        "unexpected_guard_result_count": sum(1 for event in guard_events if event.get("unexpected")),
        "read_geometry_for_merge_count": len(guard_events),
        "cross_chunk_local_xyz_merge_count": sum(
            1
            for event in guard_events
            if event.get("chunk_i") != event.get("chunk_j") and event.get("tensor_name") == "xyz_local" and event.get("guard_pass")
        ),
        "eval_aligned_merge_count": sum(1 for event in guard_events if event.get("is_eval_only") and event.get("guard_pass")),
        "cross_submap_metric_merge_count": sum(
            1
            for event in guard_events
            if event.get("submap_i") != event.get("submap_j") and event.get("guard_pass")
        ),
        "unknown_coordinate_merge_count": sum(1 for event in guard_events if event.get("coordinate_frame") == "unknown"),
    }
    return events, summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_static_md(path: Path, static: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v24 Geometry Flow Static Trace",
        "",
        "Diagnostic-only static trace. Missing files are recorded instead of inferred.",
        "",
        "## Required Target Files",
        "| path | present | bytes |",
        "| --- | ---: | ---: |",
    ]
    for row in static["target_files"]:
        lines.append(f"| `{row['path']}` | {row['present']} | {row['bytes']} |")
    lines.extend(["", "## Supplemental Files", "| path | present | bytes |", "| --- | ---: | ---: |"])
    for row in static["supplemental_files"]:
        lines.append(f"| `{row['path']}` | {row['present']} | {row['bytes']} |")
    lines.extend(["", "## Merge Audit Status", ""])
    lines.append(f"- `mask_merge_audit_status`: `{static['mask_merge_audit_status']}`")
    lines.append(f"- `native_merge_modules_present`: `{static['native_merge_modules_present']}`")
    lines.append(f"- `missing_native_merge_modules`: `{', '.join(static['missing_native_merge_modules']) or 'none'}`")
    lines.append(f"- `builder_suspicious_index_matching`: `{static['builder_suspicious_index_matching']}`")
    lines.extend(["", "## Geometry Keyword Hits", "| path | line | pattern | text |", "| --- | ---: | --- | --- |"])
    for hit in static["geometry_keyword_hits"]:
        safe = str(hit["text"]).replace("|", "\\|")
        lines.append(f"| `{hit['path']}` | {hit['line']} | `{hit['pattern']}` | `{safe}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, static: dict[str, Any], runtime: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v24 Geometry Flow Summary",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| target files present | {sum(1 for row in static['target_files'] if row['present'])} / {len(static['target_files'])} |",
        f"| supplemental files present | {sum(1 for row in static['supplemental_files'] if row['present'])} / {len(static['supplemental_files'])} |",
        f"| native merge modules present | {static['native_merge_modules_present']} |",
        f"| missing native merge modules | {len(static['missing_native_merge_modules'])} |",
        f"| builder suspicious index matching | {static['builder_suspicious_index_matching']} |",
        f"| runtime guard cases | {runtime['runtime_case_count']} |",
        f"| runtime guard pass | {runtime['runtime_guard_pass_count']} |",
        f"| runtime guard block | {runtime['runtime_guard_block_count']} |",
        f"| unexpected guard results | {runtime['unexpected_guard_result_count']} |",
        "",
        "## Decision",
        "",
        f"`mask_merge_audit_status={static['mask_merge_audit_status']}`.",
    ]
    if static["missing_native_merge_modules"]:
        lines.append(
            "The v24 native mask/object merge modules requested by the plan are missing, so this run can audit and test self-stitch/provider-side geometry guards but cannot claim full native mask merge integration."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/audit/v24_geometry_flow")
    parser.add_argument("--dry-run-probe5", action="store_true")
    parser.add_argument("--assert-merge-geometry-valid", action="store_true")
    parser.add_argument("--assert-no-eval-aligned-read-before-export", action="store_true")
    args = parser.parse_args()

    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    static = _scan_static()
    runtime_events, runtime_summary = _run_guard_trace()
    metadata = {
        "diagnostic_only": True,
        "method_result": False,
        "dry_run_probe5": bool(args.dry_run_probe5),
        "assert_merge_geometry_valid": bool(args.assert_merge_geometry_valid),
        "assert_no_eval_aligned_read_before_export": bool(args.assert_no_eval_aligned_read_before_export),
        "static": static,
        "runtime_summary": runtime_summary,
    }
    _write_static_md(output_root / "geometry_flow_static.md", static)
    _write_jsonl(output_root / "geometry_flow_runtime.jsonl", runtime_events)
    _write_summary_md(output_root / "geometry_flow_summary.md", static, runtime_summary)
    (output_root / "geometry_flow_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    guard_root = REPO_ROOT / "outputs/audit/v24_merge_geometry_guard"
    _write_jsonl(guard_root / "merge_events.jsonl", runtime_events)
    (guard_root / "merge_guard_summary.json").write_text(json.dumps(runtime_summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.assert_merge_geometry_valid and int(runtime_summary["unexpected_guard_result_count"]) != 0:
        return 2
    if args.assert_no_eval_aligned_read_before_export:
        eval_pass = [
            event
            for event in runtime_events
            if event.get("alignment_source") == "eval_gt_sim3" and event.get("guard_pass")
        ]
        if eval_pass:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
