#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R27 holdout cue-prep artifacts for seq01/05."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE1_SCRIPT = ROOT / "tools/build_v118tf_stage1_causal_object_track_sidecar.py"
spec = importlib.util.spec_from_file_location("acl2_v118tf_stage1_sidecar", STAGE1_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"unable to load Stage1 sidecar script: {STAGE1_SCRIPT}")
stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1)

RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r27_holdout_cue_prep"
OUT = STAGE / "summary"
TRACE_DIR = RESULT_ROOT / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
SEQS = ("01", "05")
DEFAULT_ANCHOR_COUNT = 8
BUFFER_SIZE = 32
SCALE_FRAME_COUNT = 8


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def top_counts(values: list[Any], limit: int = 5) -> str:
    counts = Counter("" if pd.isna(value) else str(value) for value in values)
    counts.pop("", None)
    return ";".join(f"{key}:{value}" for key, value in counts.most_common(limit))


def frame_support_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not prefix_rows:
        return []
    df = pd.DataFrame(prefix_rows)
    rows: list[dict[str, Any]] = []
    for (seq, frame_id), group in df.groupby(["seq", "frame_id"], sort=True):
        persistence = pd.to_numeric(group["semantic_persistence_prefix"], errors="coerce")
        confidence = pd.to_numeric(group["semantic_confidence_prefix"], errors="coerce")
        area = pd.to_numeric(group["current_area_ratio"], errors="coerce")
        mask_quality = pd.to_numeric(group["current_mask_quality"], errors="coerce")
        best_idx = persistence.fillna(-1).idxmax()
        best = group.loc[best_idx]
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r27_holdout_frame_semantic_support_row_v1",
                "seq": str(seq),
                "frame_id": int(frame_id),
                "visible_track_rows": int(len(group)),
                "unique_track_count": int(group["track_id"].nunique()),
                "max_semantic_persistence_prefix": safe_float(persistence.max()),
                "mean_semantic_persistence_prefix": safe_float(persistence.mean()),
                "max_semantic_confidence_prefix": safe_float(confidence.max()),
                "mean_semantic_confidence_prefix": safe_float(confidence.mean()),
                "sum_current_area_ratio": safe_float(area.sum()),
                "mean_current_mask_quality": safe_float(mask_quality.mean()),
                "best_track_id_by_semantic_persistence": str(best.get("track_id", "")),
                "best_track_role": str(best.get("current_role", "")),
                "best_track_label": str(best.get("current_label", "")),
                "top_roles": top_counts(group["current_role"].tolist()),
                "top_labels": top_counts(group["current_label"].tolist()),
            }
        )
    return rows


def trace_stats(seq: str) -> dict[str, Any]:
    path = TRACE_DIR / f"seq{seq}_flashinfer_trace.jsonl"
    stats: dict[int, dict[str, Any]] = {}
    total = 0
    read_rows = 0
    qk_rows = 0
    local_image_patch_read_rows = 0
    row_type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                total += 1
                row_type = str(payload.get("row_type", payload.get("operation_type", "")))
                family = str(payload.get("memory_family", ""))
                row_type_counts[row_type] += 1
                family_counts[family] += 1
                is_read = payload.get("row_type") == "read" or payload.get("operation_type") == "read_visible_page"
                if is_read:
                    read_rows += 1
                if payload.get("internal_signal_source") == "flashinfer_online_page_qk_summary":
                    qk_rows += 1
                if payload.get("row_type") != "read":
                    continue
                if payload.get("memory_family") != "local":
                    continue
                if payload.get("token_type") != "image_patch":
                    continue
                local_image_patch_read_rows += 1
                frame_raw = payload.get("source_frame_id")
                if frame_raw is None:
                    continue
                frame = int(frame_raw)
                if not (DEFAULT_ANCHOR_COUNT <= frame < BUFFER_SIZE):
                    continue
                bucket = stats.setdefault(
                    frame,
                    {
                        "frame_id": frame,
                        "read_rows": 0,
                        "sum_qk_cosine": 0.0,
                    },
                )
                bucket["read_rows"] += 1
                bucket["sum_qk_cosine"] += fnum(payload.get("qk_relevance_cosine"))
    for bucket in stats.values():
        n = int(bucket["read_rows"])
        bucket["mean_qk_cosine"] = bucket["sum_qk_cosine"] / n if n else 0.0
    eligible = sorted(stats)
    qk_coverage = qk_rows / read_rows if read_rows else 0.0
    return {
        "seq": seq,
        "trace": rel(path),
        "trace_exists": path.exists(),
        "trace_size_bytes": path.stat().st_size if path.exists() else 0,
        "total_trace_rows": total,
        "read_rows": read_rows,
        "qk_source_rows": qk_rows,
        "qk_coverage_over_reads": qk_coverage,
        "local_image_patch_read_rows": local_image_patch_read_rows,
        "first32_nondefault_eligible_frame_count": len(eligible),
        "first32_nondefault_eligible_frames": eligible,
        "first32_nondefault_ready": len(eligible) >= SCALE_FRAME_COUNT,
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "memory_family_counts": dict(sorted(family_counts.items())),
    }


def report_text(summary: dict[str, Any]) -> str:
    lines = [
        "# ACL2 v118-TF Stage4-R27 Holdout Cue Prep",
        "",
        f"- stage4_r27_cue_prep_decision: `{summary['stage4_r27_cue_prep_decision']}`",
        f"- holdout_cue_ready_for_r26_rule_validation: `{summary['holdout_cue_ready_for_r26_rule_validation']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | trace rows | qk coverage | eligible first32 frames | support rows | parity violations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["seq_summaries"]:
        trace = row["trace_stats"]
        lines.append(
            f"| {row['seq']} | {trace['total_trace_rows']} | {trace['qk_coverage_over_reads']} | "
            f"{trace['first32_nondefault_eligible_frame_count']} | {row['frame_support_row_count']} | "
            f"{row['future_leakage_violation_count']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            summary["boundary"],
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_prefix: list[dict[str, Any]] = []
    all_running_summary: list[dict[str, Any]] = []
    all_support: list[dict[str, Any]] = []
    all_parity: list[dict[str, Any]] = []
    all_violations: list[dict[str, Any]] = []
    seq_summaries: list[dict[str, Any]] = []

    for seq in SEQS:
        obs_by_track, obs_meta = stage1.build_observations(seq)
        prefix_rows, running_summary_rows, meta = stage1.build_prefix_rows(seq, obs_by_track)
        parity_rows, violation_rows = stage1.prefix_parity_rows(seq, obs_by_track, prefix_rows)
        support_rows = frame_support_rows(prefix_rows)
        trace = trace_stats(seq)
        support_frame_ids = {int(row["frame_id"]) for row in support_rows}
        first32_support_coverage = (
            sum(1 for frame in range(BUFFER_SIZE) if frame in support_frame_ids) / float(BUFFER_SIZE)
            if support_rows
            else 0.0
        )
        qk_values = []
        for frame in trace["first32_nondefault_eligible_frames"]:
            qk_values.append(frame)
        seq_summaries.append(
            {
                "seq": seq,
                "track_count": obs_meta["track_count"],
                "chunk_count": obs_meta["chunk_count"],
                "prefix_row_count": len(prefix_rows),
                "running_summary_row_count": len(running_summary_rows),
                "frame_support_row_count": len(support_rows),
                "first32_support_coverage": first32_support_coverage,
                "prefix_parity_row_count": len(parity_rows),
                "future_leakage_violation_count": len(violation_rows),
                "trace_stats": trace,
                "cue_ready": bool(
                    trace["trace_exists"]
                    and trace["qk_coverage_over_reads"] == 1.0
                    and trace["first32_nondefault_ready"]
                    and first32_support_coverage == 1.0
                    and not violation_rows
                ),
            }
        )
        all_prefix.extend(prefix_rows)
        all_running_summary.extend(running_summary_rows)
        all_support.extend(support_rows)
        all_parity.extend(parity_rows)
        all_violations.extend(violation_rows)

    holdout_cue_ready = bool(seq_summaries and all(row["cue_ready"] for row in seq_summaries))
    decision = (
        "HOLDOUT_CUES_READY_FOR_PRE_REGISTERED_R26_RULE_VALIDATION"
        if holdout_cue_ready
        else "NO_GO_HOLDOUT_CUE_PREP_INCOMPLETE"
    )
    prefix_path = OUT / "stage4_r27_holdout_object_track_prefix_rows.parquet"
    pd.DataFrame(all_prefix).to_parquet(prefix_path, index=False)
    write_csv(OUT / "stage4_r27_holdout_running_summary_rows.csv", all_running_summary)
    write_csv(OUT / "stage4_r27_holdout_frame_semantic_support_rows.csv", all_support)
    write_csv(OUT / "stage4_r27_holdout_prefix_parity_rows.csv", all_parity)
    write_csv(OUT / "stage4_r27_holdout_future_leakage_violation_rows.csv", all_violations)

    summary = {
        "schema": "acl2_v118tf_stage4_r27_holdout_cue_prep_summary_v1",
        "stage4_r27_cue_prep_decision": decision,
        "holdout_cue_ready_for_r26_rule_validation": holdout_cue_ready,
        "global_goal_achieved": False,
        "holdout_sequences": list(SEQS),
        "seq_summaries": seq_summaries,
        "coverage_reference_note": (
            "v117 Stage1 coverage CSV only lists 00/02 in this workspace, so 01/05 cue prep does not reuse that "
            "coverage gate as pass evidence; it instead records direct Stage-C chunk availability, prefix parity, "
            "frame support rows, and default FlashInfer trace coverage for the holdout cue gate."
        ),
        "boundary": (
            "R27 cue prep generates fresh 01/05 semantic frame support and default FlashInfer internal read cue "
            "tables only. It does not select anchors from holdout ATE and does not claim global success. If ready, "
            "the next step is to apply the already specified R26 calibration rule to these holdout cue fields."
        ),
        "outputs": {
            "summary": rel(OUT / "stage4_r27_holdout_cue_prep_summary.json"),
            "report": rel(OUT / "STAGE4_R27_HOLDOUT_CUE_PREP_REPORT.md"),
            "prefix_rows": rel(prefix_path),
            "running_summary_rows": rel(OUT / "stage4_r27_holdout_running_summary_rows.csv"),
            "frame_support_rows": rel(OUT / "stage4_r27_holdout_frame_semantic_support_rows.csv"),
            "prefix_parity_rows": rel(OUT / "stage4_r27_holdout_prefix_parity_rows.csv"),
            "future_leakage_violation_rows": rel(OUT / "stage4_r27_holdout_future_leakage_violation_rows.csv"),
        },
    }
    write_json(OUT / "stage4_r27_holdout_cue_prep_summary.json", summary)
    (OUT / "STAGE4_R27_HOLDOUT_CUE_PREP_REPORT.md").write_text(report_text(summary), encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R27-CuePrep",
            "surface_or_branch": "LB-AI",
            "status": decision,
            "artifact": rel(OUT / "stage4_r27_holdout_cue_prep_summary.json"),
            "notes": "Fresh 01/05 holdout semantic frame support plus default FlashInfer internal cue prep; no ATE-based anchor selection or global success claim",
        }
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
