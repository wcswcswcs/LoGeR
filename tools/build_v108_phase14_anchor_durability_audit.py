#!/usr/bin/env python3
"""Build a visual-first physical-anchor durability audit from Phase14 controls.

This is a posthoc audit helper. It does not infer final quality from metrics and
does not admit durable memory. It aligns existing visual-review artifacts across
baseline/random-geometry/appearance/no-LingBot controls so a reviewer can inspect
whether a good-looking output-plane candidate is actually supported by physical
anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def index_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    index_path = resolve_path(str(summary.get("phase12_visual_review_index_json", "")))
    if not index_path.is_file():
        return []
    payload = read_json(index_path)
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return [dict(row) for row in records]


def first_event_record(records: list[dict[str, Any]], event_index: int) -> dict[str, Any] | None:
    matches = [row for row in records if int(row.get("event_index", -1)) == int(event_index)]
    if not matches:
        return None
    priority = {
        "probation_attempt": 0,
        "confirm": 1,
        "shadow_output": 2,
        "source_identity_mapping": 3,
    }
    matches.sort(key=lambda row: priority.get(str(row.get("record_type", "")), 99))
    return matches[0]


def read_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def title_bar(image: np.ndarray, title: str) -> np.ndarray:
    pad = 48
    out = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    out[pad:] = image
    cv2.putText(out, title[:90], (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2, cv2.LINE_AA)
    return out


def scale_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * (height / image.shape[0]))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_panel(visuals: list[tuple[str, Path]], panel_path: Path) -> str:
    loaded: list[tuple[str, np.ndarray]] = []
    for title, path in visuals:
        image = read_image(path)
        if image is not None:
            loaded.append((title, image))
    if not loaded:
        return ""
    target_h = max(image.shape[0] for _, image in loaded)
    titled = [title_bar(scale_to_height(image, target_h), title) for title, image in loaded]
    gap = np.full((titled[0].shape[0], 18, 3), 255, dtype=np.uint8)
    panel = titled[0]
    for image in titled[1:]:
        panel = np.concatenate([panel, gap, image], axis=1)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return rel(panel_path)


def summarize_record(prefix: str, row: dict[str, Any] | None, out: dict[str, Any]) -> None:
    if row is None:
        out[f"{prefix}_present"] = False
        return
    out[f"{prefix}_present"] = True
    for key in [
        "event_index",
        "frame_id",
        "live_obj_id",
        "reference_obj_id",
        "record_type",
        "primary_visual_kind",
        "source_g3_selected_variant",
        "source_g3_record_skip_reason",
        "event_specific_visual_used",
        "visual_review_status",
        "visual_path",
        "visual_sha256",
        "source_event_visual_path",
        "source_event_visual_sha256",
        "generated_final_label_panel_path",
        "generated_final_label_panel_sha256",
    ]:
        out[f"{prefix}_{key}"] = row.get(key, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--random-geometry-summary", required=True)
    parser.add_argument("--appearance-only-summary", required=True)
    parser.add_argument("--no-lingbot-summary", required=True)
    parser.add_argument("--event-index", action="append", type=int, required=True)
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {
        "baseline": read_json(resolve_path(args.baseline_summary)),
        "random_geometry": read_json(resolve_path(args.random_geometry_summary)),
        "appearance_only": read_json(resolve_path(args.appearance_only_summary)),
        "no_lingbot": read_json(resolve_path(args.no_lingbot_summary)),
    }
    records = {name: index_records(summary) for name, summary in summaries.items()}

    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, str]] = []
    for event_index in args.event_index:
        selected = {name: first_event_record(items, event_index) for name, items in records.items()}
        row: dict[str, Any] = {
            "event_index": int(event_index),
            "metrics_are_diagnostic_only": True,
            "visual_decision_required": True,
            "durable_acceptance_status": "USER_REVIEW_PENDING",
            "anchor_durability_evidence_status": "INSUFFICIENT_FOR_DURABLE_MEMORY",
            "reason": (
                "Aligned controls are for visual audit only. A good baseline candidate is not durable "
                "unless physical anchors remain visually valid and random/appearance controls do not "
                "explain the same result."
            ),
        }
        for prefix, selected_row in selected.items():
            summarize_record(prefix, selected_row, row)

        visuals: list[tuple[str, Path]] = []
        for prefix in ["baseline", "random_geometry", "appearance_only"]:
            selected_row = selected.get(prefix)
            if selected_row is None:
                continue
            visual_text = str(selected_row.get("source_event_visual_path") or selected_row.get("visual_path") or "")
            if not visual_text:
                continue
            visual_path = resolve_path(visual_text)
            if visual_path.is_file():
                visuals.append((f"{prefix} event{event_index:03d}", visual_path))
        if visuals:
            panel_path = output_root / "visual_checks" / f"event{event_index:03d}_prompt_control_panel.png"
            panel_rel = make_panel(visuals, panel_path)
            if panel_rel:
                row["prompt_control_panel_path"] = panel_rel
                row["prompt_control_panel_sha256"] = sha256_file(resolve_path(panel_rel))
                panel_rows.append(
                    {
                        "event_index": str(event_index),
                        "panel_path": panel_rel,
                        "panel_sha256": row["prompt_control_panel_sha256"],
                    }
                )
        rows.append(row)

    rows_csv = output_root / "anchor_durability_rows.csv"
    rows_json = output_root / "anchor_durability_rows.json"
    summary_path = output_root / "anchor_durability_summary.json"
    markdown_path = output_root / "anchor_durability_summary.md"
    write_csv(rows_csv, rows)
    write_json(rows_json, rows)

    summary = {
        "schema_version": "stream4d_v108_phase14_anchor_durability_audit_v1",
        "status": "ANCHOR_DURABILITY_AUDIT_USER_REVIEW_PENDING",
        "metrics_are_diagnostic_only": True,
        "durable_acceptance_emitted": False,
        "event_count": len(rows),
        "event_indices": [int(x) for x in args.event_index],
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "rows_json": rel(rows_json),
        "rows_json_sha256": sha256_file(rows_json),
        "panel_count": len(panel_rows),
        "panels": panel_rows,
        "input_summaries": {
            "baseline": rel(resolve_path(args.baseline_summary)),
            "random_geometry": rel(resolve_path(args.random_geometry_summary)),
            "appearance_only": rel(resolve_path(args.appearance_only_summary)),
            "no_lingbot": rel(resolve_path(args.no_lingbot_summary)),
        },
        "note": (
            "This audit aligns visual evidence across controls. It does not inspect images automatically "
            "and does not prove durable memory acceptance."
        ),
    }
    write_json(summary_path, summary)

    lines = [
        "# v108 Phase14 Physical-Anchor Durability Audit",
        "",
        "Status: ANCHOR_DURABILITY_AUDIT_USER_REVIEW_PENDING",
        "",
        "Metrics and row counts are diagnostic only. Durable memory is not accepted by this artifact.",
        "",
        f"Rows: `{rel(rows_csv)}`",
        f"Rows JSON: `{rel(rows_json)}`",
        "",
        "## Events",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### event{int(row['event_index']):03d}",
                "",
                f"- durable_acceptance_status: `{row['durable_acceptance_status']}`",
                f"- anchor_durability_evidence_status: `{row['anchor_durability_evidence_status']}`",
                f"- baseline_visual: `{row.get('baseline_visual_path', '')}`",
                f"- random_geometry_visual: `{row.get('random_geometry_visual_path', '')}`",
                f"- appearance_only_visual: `{row.get('appearance_only_visual_path', '')}`",
                f"- no_lingbot_visual: `{row.get('no_lingbot_visual_path', '')}`",
                f"- prompt_control_panel: `{row.get('prompt_control_panel_path', '')}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_json"] = rel(summary_path)
    summary["markdown"] = rel(markdown_path)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    write_json(summary_path, summary)

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
