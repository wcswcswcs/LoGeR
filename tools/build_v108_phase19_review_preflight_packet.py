#!/usr/bin/env python3
"""Build a visual review and shadow transaction preflight packet for v108."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Stream3D") not in sys.path:
    sys.path.insert(1, str(ROOT / "Stream3D"))

from Stream3D.stream4d_v108.transaction_manager import (  # noqa: E402
    Sam2MemoryMutationRequest,
    SparseTransactionScheduler,
)


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


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return float(default)
        return float(text)
    except Exception:
        return float(default)


def asset(path_value: Any) -> dict[str, Any]:
    text = str(path_value or "").strip()
    if not text:
        return {"path": "", "exists": False, "sha256": ""}
    path = resolve_path(text)
    if not path.is_file():
        return {"path": rel(path), "exists": False, "sha256": ""}
    return {"path": rel(path), "exists": True, "sha256": sha256_file(path)}


def copy_asset(path_value: Any, output_dir: Path, prefix: str) -> dict[str, Any]:
    item = asset(path_value)
    if not item["exists"]:
        return item
    src = resolve_path(item["path"])
    dst = output_dir / f"{prefix}_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": rel(dst), "exists": True, "sha256": sha256_file(dst), "source_path": item["path"]}


def load_confirm_visuals(g3_records_csv: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if not g3_records_csv.is_file():
        return out
    for row in read_csv_rows(g3_records_csv):
        if str(row.get("record_type", "")) != "confirm":
            continue
        event_index = parse_int(row.get("event_index"), -1)
        if event_index < 0:
            continue
        out[event_index] = row
    return out


def title_bar(image: np.ndarray, text: str) -> np.ndarray:
    bar = 42
    out = np.full((image.shape[0] + bar, image.shape[1], 3), 255, dtype=np.uint8)
    out[bar:] = image
    cv2.putText(out, text[:110], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10, 10, 10), 2, cv2.LINE_AA)
    return out


def fit_panel(path_value: Any, title: str, *, target_h: int = 420, max_w: int = 1100) -> np.ndarray:
    item = asset(path_value)
    if not item["exists"]:
        canvas = np.full((target_h, min(max_w, 760), 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "missing image", (30, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        return title_bar(canvas, title)
    image = cv2.imread(str(resolve_path(item["path"])), cv2.IMREAD_COLOR)
    if image is None:
        canvas = np.full((target_h, min(max_w, 760), 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "unreadable image", (30, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        return title_bar(canvas, title)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(float(target_h) / float(image.shape[0]), float(max_w) / float(image.shape[1]), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return title_bar(image, title)


def make_contact_sheet(
    *,
    prompt_path: str,
    confirm_path: str,
    final_panel_path: str,
    out_path: Path,
    event_index: int,
    frame_id: int,
    live_obj_id: int,
) -> dict[str, Any]:
    panels = [
        fit_panel(prompt_path, f"event {event_index} probation prompt"),
        fit_panel(confirm_path, f"event {event_index} confirm prompt"),
        fit_panel(final_panel_path, f"final online/reference panel f{frame_id} live{live_obj_id}", max_w=1500),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=0)
        padded.append(panel)
    sheet = np.concatenate(padded, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return {"path": rel(out_path), "exists": True, "sha256": sha256_file(out_path)}


def select_review_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ready: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    for row in rows:
        if parse_bool(row.get("durable_memory_allowed_except_user_review")):
            ready.append(row)
        else:
            blocked.append(row)
    ready.sort(key=lambda row: (parse_int(row.get("event_index"), 999999), parse_int(row.get("frame_id"), 999999)))
    blocked.sort(key=lambda row: (parse_int(row.get("event_index"), 999999), parse_int(row.get("frame_id"), 999999)))
    return ready, blocked


def build_candidate_rows(
    *,
    lifecycle_rows: list[dict[str, str]],
    confirm_by_event: dict[int, dict[str, str]],
    output_root: Path,
    source_label: str,
) -> tuple[list[dict[str, Any]], list[Sam2MemoryMutationRequest]]:
    images_dir = output_root / "review_images"
    ready, blocked = select_review_rows(lifecycle_rows)
    candidate_rows: list[dict[str, Any]] = []
    requests: list[Sam2MemoryMutationRequest] = []
    for index, row in enumerate([*ready, *blocked]):
        event_index = parse_int(row.get("event_index"), -1)
        frame_id = parse_int(row.get("frame_id"), -1)
        live_obj_id = parse_int(row.get("live_obj_id"), -1)
        confirm_row = confirm_by_event.get(event_index, {})
        confirm_path = str(confirm_row.get("all_prompt_visual_path") or confirm_row.get("visual_path") or "")
        prompt_copy = copy_asset(row.get("visual_path"), images_dir, f"{source_label}_event{event_index:03d}_prompt")
        confirm_copy = copy_asset(confirm_path, images_dir, f"{source_label}_event{event_index:03d}_confirm")
        final_copy = copy_asset(
            row.get("generated_final_label_panel_path"),
            images_dir,
            f"{source_label}_event{event_index:03d}_final",
        )
        contact_sheet = make_contact_sheet(
            prompt_path=prompt_copy.get("path") or row.get("visual_path", ""),
            confirm_path=confirm_copy.get("path") or confirm_path,
            final_panel_path=final_copy.get("path") or row.get("generated_final_label_panel_path", ""),
            out_path=images_dir / f"{source_label}_event{event_index:03d}_review_contact_sheet.png",
            event_index=event_index,
            frame_id=frame_id,
            live_obj_id=live_obj_id,
        )
        ready_except_user = parse_bool(row.get("durable_memory_allowed_except_user_review"))
        visual_status = str(row.get("visual_review_status", "USER_REVIEW_PENDING"))
        preflight_status = (
            "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
            if ready_except_user and visual_status == "USER_REVIEW_PENDING"
            else "STRUCTURAL_BLOCKED_OR_NOT_SELECTED"
        )
        if ready_except_user:
            requests.append(
                Sam2MemoryMutationRequest(
                    frame_id=frame_id,
                    global_object_id=live_obj_id,
                    sam2_runtime_object_id=live_obj_id,
                    mutation="durable_admission_after_explicit_user_visual_acceptance",
                    prompt_count=parse_int(row.get("physical_projected_positive_count_diagnostic_only"), 0),
                    evidence_status="AWAITING_USER_VISUAL_ACCEPTANCE",
                )
            )
        candidate_rows.append(
            {
                "source_label": source_label,
                "row_index": int(index),
                "scene_id": row.get("scene_id", ""),
                "event_index": int(event_index),
                "record_type": row.get("record_type", ""),
                "frame_id": int(frame_id),
                "live_obj_id": int(live_obj_id),
                "reference_obj_id": parse_int(row.get("reference_obj_id"), -1),
                "visual_review_status": visual_status,
                "preflight_status": preflight_status,
                "durable_memory_allowed": parse_bool(row.get("durable_memory_allowed")),
                "durable_memory_allowed_except_user_review": bool(ready_except_user),
                "durable_memory_mutation_request_emitted": False,
                "durable_memory_block_reasons": row.get("durable_memory_block_reasons", ""),
                "durable_memory_block_reasons_except_user_review": row.get(
                    "durable_memory_block_reasons_except_user_review", ""
                ),
                "temporal_visible_frame_count": parse_int(row.get("temporal_visible_frame_count_diagnostic_only"), 0),
                "temporal_mean_iou_to_previous_visible": parse_float(
                    row.get("temporal_mean_iou_to_previous_visible_diagnostic_only"), -1.0
                ),
                "temporal_last_iou_to_previous_visible": parse_float(
                    row.get("temporal_last_iou_to_previous_visible_diagnostic_only"), -1.0
                ),
                "physical_support_source_mapping_found": parse_bool(row.get("physical_support_source_mapping_found")),
                "physical_anchor_ready": parse_bool(row.get("physical_anchor_ready_diagnostic_only")),
                "physical_projected_positive_count": parse_int(
                    row.get("physical_projected_positive_count_diagnostic_only"), 0
                ),
                "prompt_visual": prompt_copy,
                "confirm_visual": confirm_copy,
                "final_panel": final_copy,
                "review_contact_sheet": contact_sheet,
                "quality_decision_rule": "Only explicit user visual review may accept durable memory.",
            }
        )
    return candidate_rows, requests


def write_markdown(path: Path, *, summary: dict[str, Any], candidate_rows: list[dict[str, Any]], batch_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v108 Phase19 Review Preflight Packet",
        "",
        f"status: {summary['status']}",
        "metrics_are_diagnostic_only: true",
        "durable_memory_mutation_request_emitted: false",
        "acceptance_rule: explicit user visual acceptance is required before durable SAM2 memory mutation",
        "",
        "## Summary",
        "",
        f"- ready_except_user_count: {summary['ready_except_user_count']}",
        f"- baseline_candidate_count: {summary['baseline_candidate_count']}",
        f"- control_candidate_count: {summary['control_candidate_count']}",
        f"- shadow_transaction_batch_count: {summary['shadow_transaction_batch_count']}",
        "",
        "## Review Candidates",
        "",
    ]
    for row in candidate_rows:
        if row["source_label"] == "random_geometry" and row["preflight_status"] != "READY_EXCEPT_USER_VISUAL_ACCEPTANCE":
            # Keep the control concise in markdown while preserving all rows in CSV/JSON.
            control_note = "control"
        else:
            control_note = "candidate"
        lines.extend(
            [
                f"### {control_note}: {row['source_label']} event {row['event_index']} live {row['live_obj_id']}",
                "",
                f"- preflight_status: {row['preflight_status']}",
                f"- visual_review_status: {row['visual_review_status']}",
                f"- durable_memory_allowed_except_user_review: {row['durable_memory_allowed_except_user_review']}",
                f"- durable_memory_block_reasons_except_user_review: {row['durable_memory_block_reasons_except_user_review']}",
                f"- temporal_visible_frame_count: {row['temporal_visible_frame_count']}",
                f"- temporal_mean_iou_to_previous_visible: {row['temporal_mean_iou_to_previous_visible']}",
                f"- physical_anchor_ready: {row['physical_anchor_ready']}",
                f"- physical_projected_positive_count: {row['physical_projected_positive_count']}",
                f"- review_contact_sheet: {row['review_contact_sheet']['path']}",
                "",
            ]
        )
    lines.extend(["## Shadow Transaction Preflight", ""])
    for row in batch_rows:
        lines.extend(
            [
                f"- batch_id: {row.get('batch_id')}",
                f"  request_count: {row.get('request_count')}",
                f"  global_object_ids: {row.get('global_object_ids')}",
                f"  shadow_only: {row.get('shadow_only')}",
                f"  reason: {row.get('reason')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--random-geometry-root", required=True)
    parser.add_argument("--max-requests-per-batch", type=int, default=4)
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
    baseline_root = resolve_path(args.baseline_root)
    control_root = resolve_path(args.random_geometry_root)

    baseline_rows = read_csv_rows(baseline_root / "phase12_lifecycle_admission_rows.csv")
    control_rows = read_csv_rows(control_root / "phase12_lifecycle_admission_rows.csv")
    baseline_summary = json.loads((baseline_root / "phase12_full_online_summary.json").read_text(encoding="utf-8"))
    control_summary = json.loads((control_root / "phase12_full_online_summary.json").read_text(encoding="utf-8"))
    baseline_confirm = load_confirm_visuals(
        resolve_path(baseline_summary["records_csv"])
        if str(baseline_summary.get("records_csv", "")).strip()
        else baseline_root / "missing.csv"
    )
    control_confirm = load_confirm_visuals(
        resolve_path(control_summary["records_csv"])
        if str(control_summary.get("records_csv", "")).strip()
        else control_root / "missing.csv"
    )

    baseline_candidate_rows, baseline_requests = build_candidate_rows(
        lifecycle_rows=baseline_rows,
        confirm_by_event=baseline_confirm,
        output_root=output_root,
        source_label="baseline",
    )
    control_candidate_rows, control_requests = build_candidate_rows(
        lifecycle_rows=control_rows,
        confirm_by_event=control_confirm,
        output_root=output_root,
        source_label="random_geometry",
    )
    # Controls are recorded but intentionally excluded from durable preflight requests.
    requests = baseline_requests
    scheduler = SparseTransactionScheduler(max_requests_per_batch=int(args.max_requests_per_batch))
    batches = scheduler.build_batches(requests)
    batch_rows = [row for batch in batches for row in batch.as_rows()]

    candidate_rows = [*baseline_candidate_rows, *control_candidate_rows]
    ready_count = sum(1 for row in candidate_rows if row["preflight_status"] == "READY_EXCEPT_USER_VISUAL_ACCEPTANCE")
    rows_csv = output_root / "phase19_review_candidate_rows.csv"
    rows_json = output_root / "phase19_review_candidate_rows.json"
    batch_csv = output_root / "phase19_shadow_transaction_preflight_rows.csv"
    batch_json = output_root / "phase19_shadow_transaction_preflight_rows.json"
    write_csv(rows_csv, candidate_rows)
    write_json(rows_json, {"schema_version": "stream4d_v108_phase19_review_candidate_rows_v1", "records": candidate_rows})
    write_csv(batch_csv, batch_rows)
    write_json(
        batch_json,
        {
            "schema_version": "stream4d_v108_phase19_shadow_transaction_preflight_rows_v1",
            "records": batch_rows,
            "shadow_only": True,
            "durable_memory_mutation_request_emitted": False,
        },
    )

    summary_path = output_root / "phase19_review_preflight_summary.json"
    markdown_path = output_root / "phase19_review_preflight_packet.md"
    summary = {
        "schema_version": "stream4d_v108_phase19_review_preflight_summary_v1",
        "status": "USER_REVIEW_PENDING_TRANSACTION_PREFLIGHT_ONLY",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "baseline_root": rel(baseline_root),
        "baseline_summary": rel(baseline_root / "phase12_full_online_summary.json"),
        "baseline_summary_sha256": sha256_file(baseline_root / "phase12_full_online_summary.json"),
        "random_geometry_root": rel(control_root),
        "random_geometry_summary": rel(control_root / "phase12_full_online_summary.json"),
        "random_geometry_summary_sha256": sha256_file(control_root / "phase12_full_online_summary.json"),
        "baseline_candidate_count": int(len(baseline_candidate_rows)),
        "control_candidate_count": int(len(control_candidate_rows)),
        "ready_except_user_count": int(ready_count),
        "ready_except_user_event_indices": [
            int(row["event_index"])
            for row in candidate_rows
            if row["preflight_status"] == "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
        ],
        "ready_except_user_global_object_ids": [
            int(row["live_obj_id"])
            for row in candidate_rows
            if row["preflight_status"] == "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
        ],
        "shadow_transaction_batch_count": int(len(batches)),
        "shadow_transaction_request_count": int(len(requests)),
        "durable_memory_mutation_request_emitted": False,
        "acceptance_rule": "Only explicit user visual acceptance can convert a preflight candidate into a durable SAM2 memory mutation request.",
        "metrics_are_diagnostic_only": True,
        "review_candidate_rows_csv": rel(rows_csv),
        "review_candidate_rows_csv_sha256": sha256_file(rows_csv),
        "review_candidate_rows_json": rel(rows_json),
        "review_candidate_rows_json_sha256": sha256_file(rows_json),
        "shadow_transaction_preflight_rows_csv": rel(batch_csv),
        "shadow_transaction_preflight_rows_csv_sha256": sha256_file(batch_csv),
        "shadow_transaction_preflight_rows_json": rel(batch_json),
        "shadow_transaction_preflight_rows_json_sha256": sha256_file(batch_json),
        "markdown": rel(markdown_path),
    }
    write_markdown(markdown_path, summary=summary, candidate_rows=candidate_rows, batch_rows=batch_rows)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": summary["status"],
                "ready_except_user_count": int(ready_count),
                "shadow_transaction_request_count": int(len(requests)),
                "durable_memory_mutation_request_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
