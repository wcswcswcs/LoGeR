#!/usr/bin/env python3
"""Build a read-only v108 user visual-review handoff packet.

The packet makes the current Phase19 ready rows easy to inspect, but it does
not create durable acceptance and does not touch the Phase20 transaction path.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
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
    rel,
    required_acceptance_attestation,
    resolve_path,
    sha256_file,
    write_json,
    write_template_manifest,
)


def copy_contact_sheet(source_text: str, assets_dir: Path, event_index: int) -> dict[str, Any]:
    source = resolve_path(source_text)
    if not source.is_file():
        return {
            "source_path": rel(source),
            "asset_path": "",
            "exists": False,
            "sha256": "",
            "copy_sha256_matches_source": False,
        }
    dst = assets_dir / f"event{event_index:03d}_{source.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    source_sha = sha256_file(source)
    copy_sha = sha256_file(dst)
    return {
        "source_path": rel(source),
        "source_sha256": source_sha,
        "asset_path": rel(dst),
        "asset_sha256": copy_sha,
        "exists": True,
        "sha256": copy_sha,
        "copy_sha256_matches_source": bool(source_sha == copy_sha),
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v108 Phase32 User Review Handoff",
        "",
        f"status: {summary['status']}",
        "metrics_are_diagnostic_only: true",
        "durable_memory_mutation_request_emitted: false",
        "transaction_preflight_constructed: false",
        "sam2_memory_mutation_applied: false",
        "",
        "## Required User Gate",
        "",
        "Explicit user visual acceptance is still required for every current ready row before Phase20 may run.",
        "This handoff is not acceptance and is intentionally template/pending only.",
        "",
        "## Ready Rows",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}",
                "",
                f"- scene_id: {row['scene_id']}",
                f"- reference_obj_id: {row['reference_obj_id']}",
                f"- preflight_status: {row['preflight_status']}",
                f"- visual_review_status: {row['visual_review_status']}",
                f"- contact_sheet_source: {row['contact_sheet_source']}",
                f"- contact_sheet_sha256: {row['contact_sheet_sha256']}",
                f"- local_handoff_copy: {row['handoff_asset_path']}",
                f"- local_handoff_copy_sha256: {row['handoff_asset_sha256']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Manifest",
            "",
            f"- pending_manifest: {summary['pending_manifest']}",
            f"- pending_manifest_sha256: {summary['pending_manifest_sha256']}",
            f"- required_ready_evidence_fingerprint_sha256: {summary['required_ready_evidence_fingerprint_sha256']}",
            "",
            "## Boundary",
            "",
            "Codex visual notes, schema checks, templates, pending rows, and this handoff packet are not durable acceptance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        asset = html.escape(row["handoff_asset_rel_to_html"])
        title = html.escape(f"event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}")
        sha = html.escape(row["contact_sheet_sha256"])
        source = html.escape(row["contact_sheet_source"])
        cards.append(
            f"""
      <section class="case">
        <h2>{title}</h2>
        <dl>
          <dt>source</dt><dd>{source}</dd>
          <dt>sha256</dt><dd><code>{sha}</code></dd>
          <dt>status</dt><dd>USER_REVIEW_PENDING</dd>
        </dl>
        <img src="{asset}" alt="{title} contact sheet">
      </section>
"""
        )
    body = "\n".join(cards)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stream4D v108 Phase32 User Review Handoff</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111; background: #fff; }}
    header {{ max-width: 1100px; margin-bottom: 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .status {{ font-weight: 700; color: #7a3b00; }}
    .case {{ margin: 28px 0 44px; border-top: 1px solid #ddd; padding-top: 20px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    dl {{ display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 6px 12px; max-width: 1200px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    img {{ display: block; max-width: 100%; height: auto; border: 1px solid #ccc; }}
  </style>
</head>
<body>
  <header>
    <h1>Stream4D v108 Phase32 User Review Handoff</h1>
    <p class="status">{html.escape(summary['status'])}</p>
    <p>This page is a read-only handoff for visual review. It is not user acceptance, does not run Phase20, and does not emit durable SAM2 memory mutation.</p>
    <p>Required evidence fingerprint: <code>{html.escape(summary['required_ready_evidence_fingerprint_sha256'])}</code></p>
  </header>
{body}
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase19-root", required=True)
    parser.add_argument("--output-root", required=True)
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
    ready_rows = load_phase19_candidates(phase19_root)
    assets_dir = output_root / "review_assets"
    rows: list[dict[str, Any]] = []
    all_assets_match = True
    for row in ready_rows:
        event_index = parse_int(row.get("event_index"), -1)
        contact_path, contact_sha = contact_sheet_pair(row)
        copied = copy_contact_sheet(contact_path, assets_dir, event_index)
        all_assets_match = bool(all_assets_match and copied.get("copy_sha256_matches_source"))
        asset_path = resolve_path(copied.get("asset_path", "")) if copied.get("asset_path") else Path("")
        try:
            asset_rel_to_html = asset_path.resolve().relative_to(output_root.resolve()).as_posix()
        except Exception:
            asset_rel_to_html = str(copied.get("asset_path", ""))
        rows.append(
            {
                "scene_id": str(row.get("scene_id", "")),
                "event_index": event_index,
                "frame_id": parse_int(row.get("frame_id"), -1),
                "live_obj_id": parse_int(row.get("live_obj_id"), -1),
                "reference_obj_id": parse_int(row.get("reference_obj_id"), -1),
                "preflight_status": str(row.get("preflight_status", "")),
                "visual_review_status": str(row.get("visual_review_status", "USER_REVIEW_PENDING")),
                "contact_sheet_source": contact_path,
                "contact_sheet_sha256": contact_sha,
                "handoff_asset_path": copied.get("asset_path", ""),
                "handoff_asset_sha256": copied.get("asset_sha256", ""),
                "handoff_asset_rel_to_html": asset_rel_to_html,
                "copy_sha256_matches_source": bool(copied.get("copy_sha256_matches_source")),
            }
        )

    pending_manifest = output_root / "phase32_pending_user_review_manifest.json"
    template_payload = write_template_manifest(pending_manifest, ready_rows)
    required_attestation = required_acceptance_attestation(ready_rows)
    rows_json = output_root / "phase32_user_review_handoff_rows.json"
    summary_path = output_root / "phase32_user_review_handoff_summary.json"
    markdown_path = output_root / "phase32_user_review_handoff.md"
    html_path = output_root / "phase32_user_review_handoff.html"

    write_json(rows_json, {"schema_version": "stream4d_v108_phase32_user_review_handoff_rows_v1", "records": rows})
    summary: dict[str, Any] = {
        "schema_version": "stream4d_v108_phase32_user_review_handoff_summary_v1",
        "status": "USER_REVIEW_HANDOFF_READY_PENDING_USER_DECISION",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase19_root": rel(phase19_root),
        "phase19_candidate_rows_json": rel(phase19_root / "phase19_review_candidate_rows.json"),
        "phase19_candidate_rows_json_sha256": sha256_file(phase19_root / "phase19_review_candidate_rows.json"),
        "ready_row_count": int(len(ready_rows)),
        "ready_events": [int(row["event_index"]) for row in rows],
        "ready_frames": [int(row["frame_id"]) for row in rows],
        "ready_live_obj_ids": [int(row["live_obj_id"]) for row in rows],
        "pending_manifest": rel(pending_manifest),
        "pending_manifest_sha256": sha256_file(pending_manifest),
        "pending_manifest_record_count": int(len(template_payload.get("records", []))),
        "required_acceptance_attestation": required_attestation,
        "required_ready_evidence_fingerprint_sha256": required_attestation[
            "ready_evidence_fingerprint_sha256"
        ],
        "rows_json": rel(rows_json),
        "rows_json_sha256": sha256_file(rows_json),
        "asset_copy_count": int(len(rows)),
        "all_asset_copy_hashes_match_source": bool(all_assets_match),
        "durable_memory_mutation_request_emitted": False,
        "transaction_preflight_constructed": False,
        "sam2_memory_mutation_applied": False,
        "phase20_preflight_may_be_run": False,
        "visual_acceptance_claimed_by_codex": False,
        "metrics_are_diagnostic_only": True,
        "acceptance_rule": (
            "This handoff is pending-only. Only a real user-supplied exact-scope manifest "
            "with VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY for every ready row and the required "
            "top-level attestation may be passed through Phase31 and then Phase20."
        ),
        "markdown": rel(markdown_path),
        "html": rel(html_path),
    }
    write_markdown(markdown_path, summary, rows)
    write_html(html_path, summary, rows)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    summary["html_sha256"] = sha256_file(html_path)
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": summary["status"],
                "ready_row_count": int(len(ready_rows)),
                "pending_manifest": rel(pending_manifest),
                "html": rel(html_path),
                "durable_memory_mutation_request_emitted": False,
                "phase20_preflight_may_be_run": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
