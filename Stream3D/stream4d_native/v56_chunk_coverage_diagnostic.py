from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_bool, read_csv, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def build_v56_chunk_coverage_diagnostic(
    *,
    chunk_role_rows_path: str | Path = "outputs/audit/v55_chunk_roles/chunk_role_rows.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    native_carrier_rows_path: str | Path = "outputs/audit/v55_native_carrier_materialization_q4096_l11/objectlet_native_carrier_rows.csv",
    c3_update_rows_path: str | Path = "outputs/audit/v56_reuse_core_update_C3_e1_boundary_uv/history_update_rows.csv",
) -> dict[str, Any]:
    role_rows = read_csv(_project(chunk_role_rows_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    local_summary = read_json(_project(local_summary_path))
    c3_update_rows = read_csv(_project(c3_update_rows_path))
    best_variant = str(local_summary.get("best_method_variant") or "")
    best_objectlet_by_chunk = Counter(
        str(row.get("chunk_id"))
        for row in objectlet_rows
        if str(row.get("variant")) == best_variant
    )
    c3_confirmed_by_chunk = Counter(
        str(row.get("chunk_id"))
        for row in c3_update_rows
        if str(row.get("update_state")) == "confirmed_update"
    )
    c3_no_evidence_by_chunk = Counter(
        str(row.get("chunk_id"))
        for row in c3_update_rows
        if str(row.get("update_state")) == "occluded_or_absent"
    )
    native_objectlets_by_chunk: Counter[str] = Counter()
    native_valid_observed_by_chunk: Counter[str] = Counter()
    with _project(native_carrier_rows_path).open(newline="", encoding="utf-8") as handle:
        seen_objectlets: set[tuple[str, str]] = set()
        seen_valid_objectlets: set[tuple[str, str]] = set()
        for row in csv.DictReader(handle):
            chunk_id = str(row.get("chunk_id") or "")
            objectlet_id = str(row.get("objectlet_id") or "")
            key = (chunk_id, objectlet_id)
            if key not in seen_objectlets:
                native_objectlets_by_chunk[chunk_id] += 1
                seen_objectlets.add(key)
            if (
                key not in seen_valid_objectlets
                and parse_bool(row.get("visible"))
                and parse_bool(row.get("valid_uv"))
                and str(row.get("observed_mask_id") or "") not in {"", "0"}
            ):
                native_valid_observed_by_chunk[chunk_id] += 1
                seen_valid_objectlets.add(key)

    rows: list[dict[str, Any]] = []
    for row in sorted(role_rows, key=lambda item: (str(item.get("scene")), str(item.get("chunk_id")))):
        chunk_id = str(row.get("chunk_id"))
        best_objectlets = best_objectlet_by_chunk.get(chunk_id, 0)
        c3_updates = c3_confirmed_by_chunk.get(chunk_id, 0)
        rows.append(
            {
                "scene": row.get("scene"),
                "chunk_id": chunk_id,
                "role": row.get("role"),
                "best_objectlet_count": best_objectlets,
                "native_objectlet_count": native_objectlets_by_chunk.get(chunk_id, 0),
                "native_valid_observed_objectlet_count": native_valid_observed_by_chunk.get(chunk_id, 0),
                "c3_confirmed_update_count": c3_updates,
                "c3_no_evidence_history_count": c3_no_evidence_by_chunk.get(chunk_id, 0),
                "has_best_objectlets": best_objectlets > 0,
                "has_c3_confirmed_update": c3_updates > 0,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )

    evidence_rows = [row for row in rows if row["role"] in {"bridge", "update"}]
    bridge_rows = [row for row in rows if row["role"] == "bridge"]
    update_rows = [row for row in rows if row["role"] == "update"]
    summary = {
        "phase": "v56_chunk_coverage_diagnostic",
        "created_at": utc_now(),
        "best_variant": best_variant,
        "chunk_count": len(rows),
        "bridge_chunk_count": len(bridge_rows),
        "update_chunk_count": len(update_rows),
        "evidence_chunk_count": len(evidence_rows),
        "evidence_chunks_with_best_objectlets": sum(1 for row in evidence_rows if row["has_best_objectlets"]),
        "evidence_chunks_without_best_objectlets": sum(1 for row in evidence_rows if not row["has_best_objectlets"]),
        "evidence_chunks_with_c3_confirmed_update": sum(1 for row in evidence_rows if row["has_c3_confirmed_update"]),
        "bridge_chunks_without_best_objectlets": [
            row["chunk_id"] for row in bridge_rows if not row["has_best_objectlets"]
        ],
        "total_best_objectlets_in_evidence_chunks": sum(int(row["best_objectlet_count"]) for row in evidence_rows),
        "total_c3_confirmed_updates": sum(int(row["c3_confirmed_update_count"]) for row in evidence_rows),
        "total_c3_no_evidence_history_count": sum(int(row["c3_no_evidence_history_count"]) for row in evidence_rows),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    return {"summary": summary, "rows": rows}


def write_v56_chunk_coverage_diagnostic(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "chunk_coverage_summary.json", payload["summary"])
    write_csv(out / "chunk_coverage_rows.csv", payload["rows"])
