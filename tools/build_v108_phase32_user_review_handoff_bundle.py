#!/usr/bin/env python3
"""Build a minimal visual-first user-review handoff bundle for v108.

This tool packages the current Phase19 ready contact sheets and pending review
manifest template. It does not mark any row as accepted, does not run Phase20,
does not construct transaction preflight rows, and does not apply SAM2 memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import sys
import time
import zipfile
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v108_phase20_user_review_activation_preflight import (  # noqa: E402
    contact_sheet_pair,
    jsonable,
    load_phase19_candidates,
    parse_int,
    read_json,
    rel,
    resolve_path,
    row_key,
    sha256_file,
    write_json,
    write_template_manifest,
)


STATUS = "USER_VISUAL_REVIEW_HANDOFF_READY_PENDING_USER_DECISION"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def png_size(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
    except Exception:
        pass
    return 0, 0


def copy_ready_contact_sheet(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    source_path_text, expected_sha = contact_sheet_pair(row)
    source_path = resolve_path(source_path_text)
    event_index = parse_int(row.get("event_index"), -1)
    frame_id = parse_int(row.get("frame_id"), -1)
    live_obj_id = parse_int(row.get("live_obj_id"), -1)
    copied_name = f"event{event_index:03d}_frame{frame_id:06d}_live{live_obj_id:04d}_review_contact_sheet.png"
    copied_path = output_dir / copied_name
    copied_path.parent.mkdir(parents=True, exist_ok=True)

    source_exists = source_path.is_file()
    current_sha = sha256_file(source_path) if source_exists else ""
    if source_exists:
        shutil.copy2(source_path, copied_path)
    copy_exists = copied_path.is_file()
    copy_sha = sha256_file(copied_path) if copy_exists else ""
    width, height = png_size(copied_path) if copy_exists else (0, 0)
    return {
        "scene_id": str(row.get("scene_id", "")),
        "event_index": event_index,
        "frame_id": frame_id,
        "live_obj_id": live_obj_id,
        "reference_obj_id": parse_int(row.get("reference_obj_id"), -1),
        "preflight_status": str(row.get("preflight_status", "")),
        "visual_review_status": str(row.get("visual_review_status", "")),
        "source_contact_sheet": source_path_text,
        "source_contact_sheet_exists": bool(source_exists),
        "expected_contact_sheet_sha256": expected_sha,
        "current_source_contact_sheet_sha256": current_sha,
        "source_hash_matches_phase19": bool(source_exists and current_sha == expected_sha),
        "copied_contact_sheet": rel(copied_path) if copy_exists else "",
        "copied_contact_sheet_sha256": copy_sha,
        "copied_hash_matches_source": bool(copy_exists and copy_sha == current_sha and current_sha == expected_sha),
        "copied_contact_sheet_width": int(width),
        "copied_contact_sheet_height": int(height),
        "durable_memory_mutation_request_emitted": False,
        "transaction_preflight_constructed": False,
        "sam2_memory_mutation_applied": False,
    }


def write_html(path: Path, rows: list[dict[str, Any]], *, manifest_template: Path) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Stream4D v108 User Review Handoff</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f2528}",
        "main{max-width:1180px;margin:auto}",
        "section{margin:0 0 28px 0;padding:18px;background:white;border:1px solid #d8ddd9}",
        "img{max-width:100%;height:auto;border:1px solid #c9d0cc}",
        "code{background:#eef1ef;padding:2px 4px}",
        ".meta{font-size:13px;color:#4d5953;line-height:1.45}",
        "</style></head><body><main>",
        "<h1>Stream4D v108 User Review Handoff</h1>",
        "<p class=\"meta\">Status: USER_REVIEW_PENDING. No visual acceptance is recorded in this bundle.</p>",
        f"<p class=\"meta\">Pending manifest template: <code>{escape(rel(manifest_template))}</code></p>",
    ]
    for row in rows:
        copied_path = Path(row["copied_contact_sheet"])
        parts.extend(
            [
                "<section>",
                f"<h2>event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}</h2>",
                f"<p class=\"meta\">source: <code>{escape(str(row['source_contact_sheet']))}</code></p>",
                f"<p class=\"meta\">sha256: <code>{escape(str(row['expected_contact_sheet_sha256']))}</code></p>",
                f"<p class=\"meta\">hash match: {str(row['source_hash_matches_phase19']).lower()}</p>",
                f"<img src=\"{escape(copied_path.as_posix())}\" alt=\"event {row['event_index']} review contact sheet\">",
                "</section>",
            ]
        )
    parts.extend(["</main></body></html>"])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v108 Phase32 User Review Handoff",
        "",
        f"status: {summary['status']}",
        "visual_review_status: USER_REVIEW_PENDING",
        "durable_memory_mutation_request_emitted: false",
        "transaction_preflight_constructed: false",
        "sam2_memory_mutation_applied: false",
        "metrics_are_diagnostic_only: true",
        "",
        "## Ready Contact Sheets",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}",
                "",
                f"- copied_contact_sheet: {row['copied_contact_sheet']}",
                f"- source_contact_sheet: {row['source_contact_sheet']}",
                f"- expected_contact_sheet_sha256: {row['expected_contact_sheet_sha256']}",
                f"- source_hash_matches_phase19: {row['source_hash_matches_phase19']}",
                f"- copied_hash_matches_source: {row['copied_hash_matches_source']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Verification",
            "",
            f"- manifest_template: {summary['manifest_template']}",
            f"- phase31_verifier_command: {summary['phase31_verifier_command_file']}",
            f"- handoff_html: {summary['handoff_html']}",
            f"- handoff_zip: {summary['handoff_zip']}",
            "",
            "This handoff does not mark acceptance. A real user decision must be supplied separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(zip_path: Path, files: list[Path]) -> list[str]:
    entries: list[str] = []
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(files):
            if not path.is_file():
                continue
            arc = path.relative_to(zip_path.parent).as_posix()
            zf.write(path, arc)
            entries.append(arc)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase19-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase31-root", default="")
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

    phase19_root = resolve_path(args.phase19_root)
    phase31_root = resolve_path(args.phase31_root) if str(args.phase31_root).strip() else None
    ready_rows = load_phase19_candidates(phase19_root)
    ready_rows.sort(key=lambda row: (parse_int(row.get("event_index"), 999999), parse_int(row.get("frame_id"), 999999)))

    images_dir = output_root / "review_images"
    handoff_rows = [copy_ready_contact_sheet(row, images_dir) for row in ready_rows]
    manifest_template = output_root / "phase32_user_review_manifest_template_PENDING_ONLY.json"
    template_payload = write_template_manifest(manifest_template, ready_rows)

    rows_json = output_root / "phase32_user_review_handoff_rows.json"
    rows_csv = output_root / "phase32_user_review_handoff_rows.csv"
    html_path = output_root / "phase32_user_review_handoff.html"
    markdown_path = output_root / "phase32_user_review_handoff.md"
    command_path = output_root / "phase32_phase31_verifier_command.txt"
    summary_path = output_root / "phase32_user_review_handoff_summary.json"
    bundle_zip = output_root / "phase32_user_review_handoff_bundle.zip"

    command = (
        f"{sys.executable} tools/run_v108_phase31_review_manifest_verifier.py "
        f"--phase19-root {rel(phase19_root)} "
        "--review-manifest <PATH_TO_REAL_USER_FILLED_MANIFEST> "
        f"--output-root {rel(output_root / 'phase31_verifier_on_user_manifest')}"
    )
    command_path.write_text(command + "\n", encoding="utf-8")

    write_csv(rows_csv, handoff_rows)
    write_json(rows_json, {"schema_version": "stream4d_v108_phase32_user_review_handoff_rows_v1", "records": handoff_rows})

    phase31_no_manifest_summary = ""
    phase31_no_manifest_status = ""
    phase31_no_manifest_sha256 = ""
    if phase31_root is not None:
        candidate = phase31_root / "no_manifest" / "phase31_review_manifest_verifier_summary.json"
        if candidate.is_file():
            payload = read_json(candidate)
            phase31_no_manifest_summary = rel(candidate)
            phase31_no_manifest_status = str(payload.get("status", ""))
            phase31_no_manifest_sha256 = sha256_file(candidate)

    preliminary_summary = {
        "schema_version": "stream4d_v108_phase32_user_review_handoff_summary_v1",
        "status": STATUS,
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase19_root": rel(phase19_root),
        "phase19_candidate_rows_json": rel(phase19_root / "phase19_review_candidate_rows.json"),
        "phase19_candidate_rows_json_sha256": sha256_file(phase19_root / "phase19_review_candidate_rows.json"),
        "phase31_no_manifest_summary": phase31_no_manifest_summary,
        "phase31_no_manifest_summary_sha256": phase31_no_manifest_sha256,
        "phase31_no_manifest_status": phase31_no_manifest_status,
        "ready_row_count": int(len(handoff_rows)),
        "ready_event_indices": [int(row["event_index"]) for row in handoff_rows],
        "ready_live_obj_ids": [int(row["live_obj_id"]) for row in handoff_rows],
        "all_source_contact_sheets_current_hash_match": all(row["source_hash_matches_phase19"] for row in handoff_rows),
        "all_copied_contact_sheets_hash_match": all(row["copied_hash_matches_source"] for row in handoff_rows),
        "manifest_template": rel(manifest_template),
        "manifest_template_sha256": sha256_file(manifest_template),
        "template_record_count": int(len(template_payload.get("records", []))),
        "handoff_rows_csv": rel(rows_csv),
        "handoff_rows_csv_sha256": sha256_file(rows_csv),
        "handoff_rows_json": rel(rows_json),
        "handoff_rows_json_sha256": sha256_file(rows_json),
        "phase31_verifier_command_file": rel(command_path),
        "phase31_verifier_command_file_sha256": sha256_file(command_path),
        "handoff_html": rel(html_path),
        "handoff_markdown": rel(markdown_path),
        "handoff_zip": rel(bundle_zip),
        "durable_memory_mutation_request_emitted": False,
        "transaction_preflight_constructed": False,
        "sam2_memory_mutation_applied": False,
        "visual_acceptance_claimed_by_codex": False,
        "metrics_are_diagnostic_only": True,
        "acceptance_rule": (
            "This bundle only presents current ready contact sheets. It does not mark acceptance; "
            "a real user-filled manifest must pass Phase31 before Phase20 may run."
        ),
    }
    write_html(html_path, handoff_rows, manifest_template=manifest_template)
    preliminary_summary["handoff_html_sha256"] = sha256_file(html_path)
    preliminary_summary["handoff_markdown_sha256"] = ""
    preliminary_summary["handoff_zip_sha256"] = ""
    preliminary_summary["handoff_zip_entry_count"] = 0
    write_markdown(markdown_path, handoff_rows, preliminary_summary)
    preliminary_summary["handoff_markdown_sha256"] = sha256_file(markdown_path)

    zip_files = [
        *images_dir.glob("*"),
        manifest_template,
        rows_json,
        rows_csv,
        html_path,
        markdown_path,
        command_path,
        output_root / "last_command.txt",
    ]
    entries = make_zip(bundle_zip, zip_files)
    preliminary_summary["handoff_zip_sha256"] = sha256_file(bundle_zip)
    preliminary_summary["handoff_zip_entry_count"] = int(len(entries))
    preliminary_summary["handoff_zip_entries"] = entries
    preliminary_summary["note"] = (
        "The bundle zip excludes this summary JSON so the summary can record the final bundle zip hash."
    )
    write_json(summary_path, preliminary_summary)

    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": STATUS,
                "ready_row_count": int(len(handoff_rows)),
                "ready_event_indices": [int(row["event_index"]) for row in handoff_rows],
                "all_source_contact_sheets_current_hash_match": bool(
                    preliminary_summary["all_source_contact_sheets_current_hash_match"]
                ),
                "all_copied_contact_sheets_hash_match": bool(preliminary_summary["all_copied_contact_sheets_hash_match"]),
                "handoff_zip": rel(bundle_zip),
                "handoff_zip_sha256": preliminary_summary["handoff_zip_sha256"],
                "durable_memory_mutation_request_emitted": False,
                "transaction_preflight_constructed": False,
                "sam2_memory_mutation_applied": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
