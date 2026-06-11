from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "infer_track_3d.py",
    "eval_track3d_in_worldtrack.py",
    "run_eval_worldtrack.sh",
    "vis/build_like_demo.py",
    "vis/build_like_demo_for_worldtrack.py",
    "src/eval/tasks.py",
)
REQUIRED_HELPERS = (
    "_make_anchor_clip_indices",
    "_infer_tracks",
    "_encode_model_memory",
    "_model_clip_frames",
    "_run_model_for_queries",
    "_umeyama_sim3",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _stream3d_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _stream3d_root().parent


def _resolve_opend4rt_root(raw: str | Path) -> tuple[Path, bool]:
    requested = Path(raw)
    if not requested.is_absolute():
        requested = _repo_root() / requested
    if requested.exists():
        return requested.resolve(), False
    fallback = _repo_root() / "Open-d4rt"
    if fallback.exists():
        return fallback.resolve(), True
    return requested.resolve(), False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def audit(opend4rt_root: Path, requested_root_missing: bool) -> dict[str, Any]:
    stream3d_root = _stream3d_root()
    notes_path = stream3d_root / "stream4d_native" / "OPEND4RT_SOURCE_NOTES.md"
    file_rows = [
        {"path": item, "exists": bool((opend4rt_root / item).exists())}
        for item in REQUIRED_FILES
    ]
    all_text = "\n".join(_read(opend4rt_root / item) for item in REQUIRED_FILES)
    helper_rows = [
        {"helper": helper, "present_in_opend4rt": helper in all_text}
        for helper in REQUIRED_HELPERS
    ]
    notes_text = _read(notes_path)
    note_rows = [
        {"helper": helper, "mentioned_in_notes": helper in notes_text or helper.removeprefix("_") in notes_text}
        for helper in REQUIRED_HELPERS
    ]
    native_builder = _read(stream3d_root / "stream4d_native" / "d4rt_scene_builder.py")
    occupancy = _read(stream3d_root / "stream4d_native" / "occupancy_dense_tracker.py")
    chunking = _read(stream3d_root / "stream4d_native" / "chunk_alignment.py")
    summary = {
        "opend4rt_root": str(opend4rt_root),
        "requested_root_missing_used_fallback": bool(requested_root_missing),
        "opend4rt_required_files_present": bool(all(row["exists"] for row in file_rows)),
        "opend4rt_required_helpers_present": bool(all(row["present_in_opend4rt"] for row in helper_rows)),
        "opend4rt_source_notes_present": bool(notes_path.exists()),
        "opend4rt_source_notes_mentions_helpers": bool(all(row["mentioned_in_notes"] for row in note_rows)),
        "chunk_size_policy_pass": "read_checkpoint_clip_frames" in chunking and "temporal_chunk_size" in native_builder,
        "occupancy_primary_path_present": "query_d4rt_tubes_with_spatiotemporal_occupancy" in native_builder and "SpatioTemporalOccupancyState" in occupancy,
        "opend4rt_helpers_reused_or_ported": int(sum(row["present_in_opend4rt"] for row in helper_rows)),
    }
    return {"summary": summary, "files": file_rows, "helpers": helper_rows, "notes": note_rows}


def _write(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# OpenD4RT Source Alignment Audit", "", "## Summary", ""]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Required Files", "", "| file | exists |", "|---|---:|"])
    for row in payload["files"]:
        lines.append(f"| `{row['path']}` | {row['exists']} |")
    lines.extend(["", "## Required Helpers", "", "| helper | present | notes mention |", "|---|---:|---:|"])
    notes = {row["helper"]: row["mentioned_in_notes"] for row in payload["notes"]}
    for row in payload["helpers"]:
        lines.append(f"| `{row['helper']}` | {row['present_in_opend4rt']} | {notes.get(row['helper'], False)} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stream4D native source alignment with OpenD4RT.")
    parser.add_argument("--opend4rt-root", required=True)
    parser.add_argument("--output", default="outputs/audit/v21_3_phaseA/opend4rt_source_alignment.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root, fallback = _resolve_opend4rt_root(args.opend4rt_root)
    payload = audit(root, fallback)
    output = Path(args.output)
    if not output.is_absolute():
        output = _stream3d_root() / output
    _write(output, payload)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))
    summary = payload["summary"]
    if not (
        summary["opend4rt_required_files_present"]
        and summary["opend4rt_required_helpers_present"]
        and summary["opend4rt_source_notes_present"]
        and summary["chunk_size_policy_pass"]
        and summary["occupancy_primary_path_present"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
