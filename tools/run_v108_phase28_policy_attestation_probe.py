#!/usr/bin/env python3
"""Probe v108 lower-level policy behavior around visual acceptance attestation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Stream3D") not in sys.path:
    sys.path.insert(1, str(ROOT / "Stream3D"))

from Stream3D.stream4d_v108.growth_repair import GrowthRepairPlanner  # noqa: E402
from Stream3D.stream4d_v108.lifecycle import DelayedAdmissionPolicy  # noqa: E402


ACCEPTED = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="phase28_policy_attestation_probe")
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    policy = DelayedAdmissionPolicy()
    component_stats = {
        "area_px": 1000,
        "edge_touch_count": 0,
        "bbox_area_frac": 0.05,
        "area_frac": 0.04,
        "bbox_extent": 0.8,
    }
    watcher_stats = {
        "visible_frame_count": 3,
        "mean_iou_to_previous_visible": 0.75,
    }
    physical_support_stats = {
        "physical_anchor_ready": True,
        "geometry_available": True,
        "projected_positive_count": 3,
        "conflict_diagnostics": {
            "positive_negative_conflict_count": 0,
            "positive_cluster_outlier_count": 0,
        },
        "target_support": {
            "core_depth_valid_fraction": 1.0,
        },
    }
    lifecycle = policy.evaluate(
        frame_id=4460,
        global_object_id=84,
        component_stats=component_stats,
        watcher_stats=watcher_stats,
        physical_support_stats=physical_support_stats,
        visual_review_status=ACCEPTED,
    )

    planner = GrowthRepairPlanner()
    growth = planner.suggest_from_shadow_stats(
        frame_id=4460,
        global_object_id=84,
        visible=True,
        edge_touch_count=0,
        area_ratio_to_history=1.0,
        bbox_area_fraction=0.05,
        visual_review_status=ACCEPTED,
    )

    rows = [
        {
            "probe": "delayed_admission_policy",
            "visual_review_status": ACCEPTED,
            "durable_memory_allowed": bool(lifecycle.durable_memory_allowed),
            "output_state": str(lifecycle.output_state.value),
            "reasons": list(lifecycle.reasons),
            "user_attestation_verified": getattr(lifecycle, "user_attestation_verified", "field_absent"),
        },
        {
            "probe": "growth_repair_planner",
            "visual_review_status": ACCEPTED,
            "durable_memory_allowed": bool(growth.durable_memory_allowed),
            "action": str(growth.action),
            "reason": str(growth.reason),
            "user_attestation_verified": getattr(growth, "user_attestation_verified", "field_absent"),
        },
    ]
    rows_json = output_root / "phase28_policy_attestation_probe_rows.json"
    rows_csv = output_root / "phase28_policy_attestation_probe_rows.csv"
    write_json(rows_json, {"schema_version": "stream4d_v108_phase28_policy_attestation_probe_rows_v1", "records": rows})
    write_csv(rows_csv, rows)

    blocked_without_attestation = (
        bool(lifecycle.durable_memory_allowed) is False
        and "explicit_user_attestation_not_verified_for_durable_memory" in set(lifecycle.reasons)
        and str(growth.action) == "keep_output_probation_until_user_attestation"
    )
    summary = {
        "schema_version": "stream4d_v108_phase28_policy_attestation_probe_summary_v1",
        "label": str(args.label),
        "status": "POLICY_ATTESTATION_GUARD_PRESENT" if blocked_without_attestation else "POLICY_ATTESTATION_GUARD_MISSING",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "accepted_status_used": ACCEPTED,
        "rows_json": rel(rows_json),
        "rows_json_sha256": sha256_file(rows_json),
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "lifecycle_durable_memory_allowed": bool(lifecycle.durable_memory_allowed),
        "lifecycle_reasons": list(lifecycle.reasons),
        "growth_action": str(growth.action),
        "growth_durable_memory_allowed": bool(growth.durable_memory_allowed),
        "growth_reason": str(growth.reason),
        "note": (
            "This probe uses a synthetic accepted visual_review_status string. It is not a real user "
            "visual review and must not be used as acceptance evidence."
        ),
    }
    summary_path = output_root / "phase28_policy_attestation_probe_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "status": summary["status"],
                "lifecycle_durable_memory_allowed": bool(summary["lifecycle_durable_memory_allowed"]),
                "growth_action": summary["growth_action"],
                "growth_durable_memory_allowed": bool(summary["growth_durable_memory_allowed"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
