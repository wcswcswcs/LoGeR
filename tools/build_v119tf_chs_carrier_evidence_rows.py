#!/usr/bin/env python3
"""Build C-HS Carrier Evidence Rows from explicit-lane runtime audits.

This builder only uses measured HorizonStream C-HS smoke artifacts. Fields that
were not measured by the current small-window run are left empty and listed in
the summary; they are not inferred.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
SUMMARY_JSON = OUT / "stage4_hs_dhs_liveness_smoke/dhs_chs_explicit_lane_smoke_summary.json"
EVIDENCE_CSV = OUT / "V119_CHS_CARRIER_EVIDENCE_ROWS.csv"
EVIDENCE_PARQUET = OUT / "V119_CHS_CARRIER_EVIDENCE_ROWS.parquet"
SUMMARY_OUT = OUT / "V119_CHS_CARRIER_EVIDENCE_ROWS_SUMMARY.json"


FIELDS = [
    "schema",
    "generated_at_utc",
    "run_id",
    "model",
    "sequence",
    "frame_id",
    "query_frame_id",
    "carrier_family",
    "carrier_id",
    "parent_physical_carrier_id",
    "carrier_unit_type",
    "layer_id",
    "head_id",
    "channel_or_rank_id",
    "token_start",
    "token_end",
    "chunk_idx",
    "chunk_start",
    "chunk_end",
    "source_frame_ids",
    "source_track_ids",
    "source_role_distribution",
    "source_frame_count",
    "source_object_count",
    "provenance_entropy",
    "dominant_track_fraction",
    "addressability_score",
    "memory_role_candidate",
    "internal_alignment",
    "innovation_residual",
    "redundancy",
    "read_relevance",
    "read_entropy",
    "read_utility",
    "write_norm",
    "retention_or_gamma",
    "MRT_or_scale_sensitivity",
    "geometry_intervention_budget",
    "semantic_sidecar_hash",
    "prefix_end_frame",
    "future_leakage_audit_pass",
    "fragmentation_bucket",
    "track_confidence",
    "action",
    "control",
    "case_id",
    "role",
    "ate_rmse",
    "lane_form",
    "calibration_mode",
    "intervention_form",
    "representation_control",
    "first_chunk_no_prior",
    "changed_state",
    "stored_lane_count",
    "state_delta_norm_raw",
    "state_delta_norm_after",
    "state_delta_rel_norm_raw",
    "state_delta_rel_norm_after",
    "semantic_risk_mean",
    "semantic_stable_mean",
    "transient_assignment",
    "persistent_assignment",
    "metric_assignment",
    "truthfulness_boundary",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_summary() -> dict[str, Any]:
    if not SUMMARY_JSON.is_file():
        return {}
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def source_frame_ids(start: Any, end: Any) -> str:
    try:
        s = int(float(start))
        e = int(float(end))
    except (TypeError, ValueError):
        return ""
    if e <= s:
        return ""
    return ",".join(str(idx) for idx in range(s, e))


def role_distribution(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "semantic_risk_mean": row.get("semantic_risk_mean", ""),
            "semantic_stable_mean": row.get("semantic_stable_mean", ""),
        },
        sort_keys=True,
    )


def build_rows(summary: dict[str, Any], generated_at: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        if not isinstance(case, dict):
            continue
        audit_path_raw = str(case.get("audit_path", ""))
        if not audit_path_raw:
            continue
        audit_path = ROOT / audit_path_raw
        audit_rows = read_csv(audit_path)
        for idx, audit in enumerate(audit_rows):
            chunk_start = audit.get("chunk_start", "")
            chunk_end = audit.get("chunk_end", "")
            branch = str(case.get("branch", ""))
            lane_form = audit.get("lane_form", "")
            carrier_id = (
                f"{branch}:layer{audit.get('global_layer_idx', '')}:"
                f"chunk{audit.get('chunk_idx', '')}:{lane_form}:row{idx}"
            )
            rank = audit.get("shadow_rank", "")
            if not rank or str(rank) == "0":
                rank = ""
            row = {
                "schema": "acl2_v119tf_chs_carrier_evidence_row_v1",
                "generated_at_utc": generated_at,
                "run_id": f"{case.get('case_id', '')}_seq{case.get('seq', '')}_max12_global_mrt",
                "model": "HorizonStream",
                "sequence": case.get("seq", ""),
                "frame_id": "",
                "query_frame_id": "",
                "carrier_family": branch,
                "carrier_id": carrier_id,
                "parent_physical_carrier_id": f"gla_recurrent_state_layer_{audit.get('global_layer_idx', '')}",
                "carrier_unit_type": "gla_recurrent_state_lane_sidecar",
                "layer_id": audit.get("global_layer_idx", ""),
                "head_id": "",
                "channel_or_rank_id": rank,
                "token_start": "",
                "token_end": "",
                "chunk_idx": audit.get("chunk_idx", ""),
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "source_frame_ids": source_frame_ids(chunk_start, chunk_end),
                "source_track_ids": "",
                "source_role_distribution": role_distribution(audit),
                "source_frame_count": "",
                "source_object_count": "",
                "provenance_entropy": "",
                "dominant_track_fraction": "",
                "addressability_score": "",
                "memory_role_candidate": lane_form,
                "internal_alignment": "",
                "innovation_residual": audit.get("delta_pressure", ""),
                "redundancy": "",
                "read_relevance": "",
                "read_entropy": "",
                "read_utility": "",
                "write_norm": audit.get("state_delta_norm_after", ""),
                "retention_or_gamma": "",
                "MRT_or_scale_sensitivity": "",
                "geometry_intervention_budget": audit.get("stored_lane_count", ""),
                "semantic_sidecar_hash": "",
                "prefix_end_frame": chunk_end,
                "future_leakage_audit_pass": "",
                "fragmentation_bucket": "",
                "track_confidence": "",
                "action": case.get("action", ""),
                "control": case.get("control", ""),
                "case_id": case.get("case_id", ""),
                "role": case.get("role", ""),
                "ate_rmse": case.get("ate_rmse", ""),
                "lane_form": lane_form,
                "calibration_mode": audit.get("calibration_mode", ""),
                "intervention_form": audit.get("intervention_form", ""),
                "representation_control": audit.get("representation_control", ""),
                "first_chunk_no_prior": audit.get("first_chunk_no_prior", ""),
                "changed_state": audit.get("changed_state", ""),
                "stored_lane_count": audit.get("stored_lane_count", ""),
                "state_delta_norm_raw": audit.get("state_delta_norm_raw", ""),
                "state_delta_norm_after": audit.get("state_delta_norm_after", ""),
                "state_delta_rel_norm_raw": audit.get("state_delta_rel_norm_raw", ""),
                "state_delta_rel_norm_after": audit.get("state_delta_rel_norm_after", ""),
                "semantic_risk_mean": audit.get("semantic_risk_mean", ""),
                "semantic_stable_mean": audit.get("semantic_stable_mean", ""),
                "transient_assignment": audit.get("transient_assignment", ""),
                "persistent_assignment": audit.get("persistent_assignment", ""),
                "metric_assignment": audit.get("metric_assignment", ""),
                "truthfulness_boundary": (
                    "C-HS seq00 max12/global_mrt lane audit row; object provenance, "
                    "track ids, addressability, dense geometry, and scale sensitivity are not measured here"
                ),
            }
            out.append(row)
    return out


def maybe_write_parquet(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"pandas_unavailable: {exc}"
    try:
        frame = pd.DataFrame(rows, columns=FIELDS)
        frame.to_parquet(EVIDENCE_PARQUET, index=False)
        return True, rel(EVIDENCE_PARQUET)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"parquet_write_failed: {exc}"


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = load_summary()
    rows = build_rows(summary, generated_at)
    write_csv(EVIDENCE_CSV, rows)
    parquet_written, parquet_status = maybe_write_parquet(rows)
    missing_measured_fields = [
        "source_track_ids",
        "source_object_count",
        "provenance_entropy",
        "dominant_track_fraction",
        "addressability_score",
        "semantic_sidecar_hash",
        "future_leakage_audit_pass",
        "MRT_or_scale_sensitivity",
    ]
    payload = {
        "schema": "acl2_v119tf_chs_carrier_evidence_rows_summary_v1",
        "generated_at_utc": generated_at,
        "source_summary": rel(SUMMARY_JSON),
        "evidence_csv": rel(EVIDENCE_CSV),
        "evidence_parquet": rel(EVIDENCE_PARQUET) if parquet_written else "",
        "parquet_status": parquet_status,
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "carrier_families": sorted({row["carrier_family"] for row in rows}),
        "missing_not_inferred_fields": missing_measured_fields,
        "truthfulness_boundary": (
            "Rows are exact C-HS lane action audit evidence from seq00 max12/global_mrt smoke. "
            "They do not provide object-level addressability, dense geometry, scale, or cross-sequence proof."
        ),
    }
    SUMMARY_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
