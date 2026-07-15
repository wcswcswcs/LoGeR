#!/usr/bin/env python3
"""Aggregate Stream4D v107 Phase1 scope artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return {"path": rel(path), "sha256": sha256_file(path), "json": read_json(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase0-gate", required=True)
    parser.add_argument("--scene0050-label-summary", required=True)
    parser.add_argument("--scene0011-label-summary", required=True)
    parser.add_argument("--short48-label-summary", required=True)
    parser.add_argument("--live-rawlogit-summary", required=True)
    parser.add_argument("--failed-prethreshold-summary", default="")
    parser.add_argument("--nohook-parity", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    phase1 = output_root / "phase1"
    phase1.mkdir(parents=True, exist_ok=True)

    phase0_gate = artifact(args.phase0_gate)
    scene0050 = artifact(args.scene0050_label_summary)
    scene0011 = artifact(args.scene0011_label_summary)
    short48 = artifact(args.short48_label_summary)
    live = artifact(args.live_rawlogit_summary)
    failed = artifact(args.failed_prethreshold_summary) if args.failed_prethreshold_summary else None
    nohook = artifact(args.nohook_parity) if args.nohook_parity else None

    label_summaries = [scene0050, scene0011, short48]
    label_parity_pass = all(item["json"].get("label_exact_parity_pass") is True for item in label_summaries)
    scope_reference_items_pass = all(
        item["json"].get("frame_count", 0) > 0 and item["json"].get("object_frame_row_count", 0) > 0
        for item in label_summaries
    )
    live_pass = live["json"].get("label_exact_parity_pass") is True and live["json"].get("raw_logit_row_count", 0) > 0

    summary = {
        "schema_version": "stream4d_v107_phase1_scope_summary_v1",
        "created_unix_time": time.time(),
        "phase0_gate": {k: v for k, v in phase0_gate.items() if k != "json"},
        "scope_items": {
            "scene0050_full90_label_instrumentation": {
                "artifact": {k: v for k, v in scene0050.items() if k != "json"},
                "decision": scene0050["json"].get("decision"),
                "frame_count": scene0050["json"].get("frame_count"),
                "object_frame_row_count": scene0050["json"].get("object_frame_row_count"),
                "object_lifecycle_count": scene0050["json"].get("object_lifecycle_count"),
                "label_exact_parity_pass": scene0050["json"].get("label_exact_parity_pass"),
            },
            "scene0011_full90_label_instrumentation": {
                "artifact": {k: v for k, v in scene0011.items() if k != "json"},
                "decision": scene0011["json"].get("decision"),
                "frame_count": scene0011["json"].get("frame_count"),
                "object_frame_row_count": scene0011["json"].get("object_frame_row_count"),
                "object_lifecycle_count": scene0011["json"].get("object_lifecycle_count"),
                "label_exact_parity_pass": scene0011["json"].get("label_exact_parity_pass"),
            },
            "scene0050_short48_allmemory_label_instrumentation": {
                "artifact": {k: v for k, v in short48.items() if k != "json"},
                "decision": short48["json"].get("decision"),
                "frame_count": short48["json"].get("frame_count"),
                "object_frame_row_count": short48["json"].get("object_frame_row_count"),
                "object_lifecycle_count": short48["json"].get("object_lifecycle_count"),
                "label_exact_parity_pass": short48["json"].get("label_exact_parity_pass"),
            },
            "scene0050_full90_live_rawlogit_trace": {
                "artifact": {k: v for k, v in live.items() if k != "json"},
                "decision": live["json"].get("decision"),
                "raw_logit_row_count": live["json"].get("raw_logit_row_count"),
                "label_exact_parity_pass": live["json"].get("label_exact_parity_pass"),
                "pixel_mismatch_count": live["json"].get("pixel_mismatch_count"),
                "wall_time_sec": live["json"].get("wall_time_sec"),
            },
        },
        "blocker_repair_evidence": {
            "prethreshold_hook_failed": None
            if failed is None
            else {
                "artifact": {k: v for k, v in failed.items() if k != "json"},
                "decision": failed["json"].get("decision"),
                "pixel_mismatch_count": failed["json"].get("pixel_mismatch_count"),
                "raw_logit_row_count": failed["json"].get("raw_logit_row_count"),
            },
            "nohook_replay_exact": None
            if nohook is None
            else {
                "artifact": {k: v for k, v in nohook.items() if k != "json"},
                "label_exact_parity_pass": nohook["json"].get("label_exact_parity_pass"),
                "bad_frame_count": nohook["json"].get("bad_frame_count"),
                "pixel_mismatch_count": nohook["json"].get("pixel_mismatch_count"),
            },
            "repair": (
                "Moved STREAM_INFER_TRACE_HOOK after masks/ids are materialized in infer_stream_frame. "
                "The pre-threshold hook failed exact parity; no-hook replay passed; post-mask hook passed."
            ),
        },
        "phase1_label_parity_pass": bool(label_parity_pass),
        "phase1_scope_reference_items_pass": bool(scope_reference_items_pass),
        "phase1_live_rawlogit_trace_pass": bool(live_pass),
        "phase1_full_gate_pass": bool(label_parity_pass and scope_reference_items_pass and live_pass),
        "warnings": [
            "Scene0011 and short48 instrumentation are label/RGB-derived traces; raw-logit live trace was run on scene0050 full90.",
            "SAM2 pooled appearance embeddings were not extracted; RGB mask mean/std descriptors are present in object_frame_rows.",
        ],
    }
    summary["decision"] = (
        "PASS_PHASE1_REFERENCE_INSTRUMENTATION_WITH_WARNINGS"
        if summary["phase1_full_gate_pass"]
        else "NO_GO_PHASE1_SCOPE_INCOMPLETE"
    )
    out_path = phase1 / "phase1_scope_summary.json"
    write_json(out_path, summary)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase1_scope_summary_run_v1",
            "phase1_scope_summary": rel(out_path),
            "decision": summary["decision"],
            "phase1_full_gate_pass": summary["phase1_full_gate_pass"],
        },
    )
    print(json.dumps({"output_root": str(output_root), "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["phase1_full_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
