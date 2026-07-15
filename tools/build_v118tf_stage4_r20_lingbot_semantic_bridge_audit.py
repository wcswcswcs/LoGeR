#!/usr/bin/env python3
"""Audit v118 Stage4 LingBot semantic bridge readiness.

This script intentionally separates two facts that are easy to conflate:

1. Stage1 can provide frame-level semantic/track support for source frames.
2. Current FlashInfer trajectory-special actions operate on special pages, not
   token/object-exact semantic entries.

The output is therefore an audit artifact and repair guide, not a semantic
success claim.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r20_lingbot_semantic_bridge_audit"
OUT = STAGE / "summary"
STAGE1 = RESULT_ROOT / "stage1_causal_object_track_sidecar"
STAGE2 = RESULT_ROOT / "stage2_memory_entry_provenance"
V117_PROVENANCE = (
    ROOT
    / "results/acl2_v117tf_same_space_semantic_memory_reliability"
    / "stage2_memory_provenance/lingbot_trajectory_provenance_rows.csv"
)

TRACE_INPUTS = [
    {
        "label": "R16_default_flashinfer_seq00",
        "seq": "00",
        "policy": "DEFAULT_FLASHINFER",
        "path": RESULT_ROOT
        / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full/seq00_flashinfer_trace.jsonl",
    },
    {
        "label": "R16_default_flashinfer_seq02",
        "seq": "02",
        "policy": "DEFAULT_FLASHINFER",
        "path": RESULT_ROOT
        / "stage3_r14_lingbot_flashinfer_internal_signal_probe/runtime_full/seq02_flashinfer_trace.jsonl",
    },
    {
        "label": "R19_qk_topk2_seq00",
        "seq": "00",
        "policy": "TR1_QK_TOPK",
        "path": RESULT_ROOT
        / "stage4_r19_lingbot_tr_topk_calibration/runtime_full_thread8/traces/"
        / "lingbot_map_stream_flashinfer_v118_r19_tr1_qk_topk2_seq00.jsonl",
    },
    {
        "label": "R19_qk_topk2_seq02",
        "seq": "02",
        "policy": "TR1_QK_TOPK",
        "path": RESULT_ROOT
        / "stage4_r19_lingbot_tr_topk_calibration/runtime_full_thread8/traces/"
        / "lingbot_map_stream_flashinfer_v118_r19_tr1_qk_topk2_seq02.jsonl",
    },
    {
        "label": "R19_random_topk2_seq00",
        "seq": "00",
        "policy": "TR1_RANDOM_TOPK",
        "path": RESULT_ROOT
        / "stage4_r19_lingbot_tr_topk_calibration/runtime_full_thread8/traces/"
        / "lingbot_map_stream_flashinfer_v118_r19_tr1_random_topk2_seq00.jsonl",
    },
    {
        "label": "R19_random_topk2_seq02",
        "seq": "02",
        "policy": "TR1_RANDOM_TOPK",
        "path": RESULT_ROOT
        / "stage4_r19_lingbot_tr_topk_calibration/runtime_full_thread8/traces/"
        / "lingbot_map_stream_flashinfer_v118_r19_tr1_random_topk2_seq02.jsonl",
    },
]

DIRECT_SEMANTIC_KEYS = {
    "track_id",
    "object_id",
    "instance_id",
    "dominant_track_id",
    "semantic_persistence",
    "semantic_persistence_prefix",
    "dominant_role",
    "dominant_label",
    "semantic_role",
    "semantic_label",
    "current_role",
    "current_label",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def top_counts(values: list[Any], limit: int = 5) -> str:
    counts = Counter("" if pd.isna(v) else str(v) for v in values)
    counts.pop("", None)
    return ";".join(f"{key}:{value}" for key, value in counts.most_common(limit))


def load_stage1_frame_support() -> tuple[dict[tuple[str, int], dict], list[dict]]:
    prefix_path = STAGE1 / "object_track_prefix_rows.parquet"
    df = pd.read_parquet(prefix_path)
    support: dict[tuple[str, int], dict] = {}
    rows: list[dict] = []
    for (seq, frame_id), group in df.groupby(["seq", "frame_id"], sort=True):
        g = group.copy()
        persistence = pd.to_numeric(g["semantic_persistence_prefix"], errors="coerce")
        confidence = pd.to_numeric(g["semantic_confidence_prefix"], errors="coerce")
        area = pd.to_numeric(g["current_area_ratio"], errors="coerce")
        mask_quality = pd.to_numeric(g["current_mask_quality"], errors="coerce")
        best_idx = persistence.fillna(-1).idxmax()
        best = g.loc[best_idx]
        row = {
            "schema": "acl2_v118tf_stage4_r20_frame_semantic_support_row_v1",
            "seq": str(seq),
            "frame_id": int(frame_id),
            "visible_track_rows": int(len(g)),
            "unique_track_count": int(g["track_id"].nunique()),
            "max_semantic_persistence_prefix": safe_float(persistence.max()),
            "mean_semantic_persistence_prefix": safe_float(persistence.mean()),
            "max_semantic_confidence_prefix": safe_float(confidence.max()),
            "mean_semantic_confidence_prefix": safe_float(confidence.mean()),
            "sum_current_area_ratio": safe_float(area.sum()),
            "mean_current_mask_quality": safe_float(mask_quality.mean()),
            "best_track_id_by_semantic_persistence": str(best.get("track_id", "")),
            "best_track_role": str(best.get("current_role", "")),
            "best_track_label": str(best.get("current_label", "")),
            "top_roles": top_counts(g["current_role"].tolist()),
            "top_labels": top_counts(g["current_label"].tolist()),
        }
        support[(str(seq), int(frame_id))] = row
        rows.append(row)
    return support, rows


def load_stage2_surface_gate() -> dict[str, dict]:
    path = STAGE2 / "stage2_surface_gate_rows.csv"
    if not path.exists():
        return {}
    by_surface: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            surface = row.get("surface", "")
            if not surface:
                continue
            by_surface.setdefault(surface, {"rows": [], "all_gate_pass": True})
            by_surface[surface]["rows"].append(row)
            by_surface[surface]["all_gate_pass"] = by_surface[surface]["all_gate_pass"] and str(row.get("gate_pass")) == "True"
    return by_surface


def parse_pages(text: Any) -> list[int]:
    if not text:
        return []
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def has_stage1_support(support: dict[tuple[str, int], dict], seq: str, frame_id: Any) -> bool:
    try:
        return (seq, int(frame_id)) in support
    except (TypeError, ValueError):
        return False


def trace_audit_row(trace: dict, support: dict[tuple[str, int], dict]) -> dict:
    path = Path(trace["path"])
    seq = str(trace["seq"])
    row: dict[str, Any] = {
        "schema": "acl2_v118tf_stage4_r20_trace_semantic_bridge_audit_row_v1",
        "label": trace["label"],
        "seq": seq,
        "policy": trace["policy"],
        "trace": rel(path),
        "trace_exists": path.exists(),
    }
    if not path.exists():
        row.update(
            {
                "total_rows": 0,
                "trajectory_append_rows": 0,
                "trajectory_append_stage1_supported_rows": 0,
                "trajectory_append_stage1_support_coverage": 0.0,
                "trajectory_read_rows": 0,
                "trajectory_read_stage1_supported_rows": 0,
                "trajectory_read_stage1_support_coverage": 0.0,
                "local_patch_action_rows": 0,
                "local_patch_action_stage1_supported_rows": 0,
                "local_patch_action_stage1_support_coverage": 0.0,
                "direct_semantic_field_rows": 0,
                "trajectory_exact_token_object_provenance_available": False,
                "stage1_frame_support_available": False,
            }
        )
        return row

    row_type_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    trajectory_token_types: Counter[str] = Counter()
    direct_semantic_field_rows = 0

    trajectory_append_rows = 0
    trajectory_append_supported = 0
    trajectory_read_rows = 0
    trajectory_read_supported = 0
    local_patch_action_rows = 0
    local_patch_action_supported = 0

    action_rows = 0
    changed_action_rows = 0
    selected_pages_with_frame_map = 0
    selected_pages_without_frame_map = 0
    selected_page_frame_refs = 0
    selected_page_frame_refs_supported = 0

    special_page_frames: dict[int, list[int]] = defaultdict(list)
    special_page_frame_sets: dict[int, set[int]] = defaultdict(set)
    special_page_append_segments: Counter[int] = Counter()
    local_action_row_types = {"append", "read", "eviction", "rollback"}

    total_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            total_rows += 1
            row_type = str(payload.get("row_type", ""))
            memory_family = str(payload.get("memory_family", ""))
            row_type_counts[row_type] += 1
            family_counts[memory_family or "<none>"] += 1
            if DIRECT_SEMANTIC_KEYS & set(payload):
                direct_semantic_field_rows += 1

            if memory_family == "trajectory_special":
                trajectory_token_types[str(payload.get("token_type", ""))] += 1
                if row_type == "append":
                    trajectory_append_rows += 1
                    if has_stage1_support(support, seq, payload.get("source_frame_id")):
                        trajectory_append_supported += 1
                    page = payload.get("cache_position_or_page")
                    frame = payload.get("source_frame_id")
                    try:
                        page_i = int(page)
                        frame_i = int(frame)
                    except (TypeError, ValueError):
                        pass
                    else:
                        special_page_append_segments[page_i] += 1
                        if frame_i not in special_page_frame_sets[page_i]:
                            special_page_frame_sets[page_i].add(frame_i)
                            special_page_frames[page_i].append(frame_i)
                elif row_type == "read":
                    trajectory_read_rows += 1
                    if has_stage1_support(support, seq, payload.get("source_frame_id")):
                        trajectory_read_supported += 1

            if memory_family in {"local", "anchor"} and str(payload.get("token_type")) == "image_patch" and row_type in local_action_row_types:
                local_patch_action_rows += 1
                if has_stage1_support(support, seq, payload.get("source_frame_id")):
                    local_patch_action_supported += 1

            if row_type == "action" and str(payload.get("action_branch")) == "LB-TR":
                action_rows += 1
                if payload.get("action_changed_visible_table") is True:
                    changed_action_rows += 1
                selected = parse_pages(payload.get("selected_pages_preview"))
                for page in selected:
                    frames = special_page_frames.get(page, [])
                    if frames:
                        selected_pages_with_frame_map += 1
                    else:
                        selected_pages_without_frame_map += 1
                    selected_page_frame_refs += len(frames)
                    selected_page_frame_refs_supported += sum(1 for frame in frames if (seq, int(frame)) in support)

    page_frame_counts = [len(frames) for frames in special_page_frames.values()]
    row.update(
        {
            "total_rows": total_rows,
            "row_type_counts": dict(sorted(row_type_counts.items())),
            "memory_family_counts": dict(sorted(family_counts.items())),
            "trajectory_token_type_counts": dict(sorted(trajectory_token_types.items())),
            "trajectory_append_rows": trajectory_append_rows,
            "trajectory_append_stage1_supported_rows": trajectory_append_supported,
            "trajectory_append_stage1_support_coverage": ratio(trajectory_append_supported, trajectory_append_rows),
            "trajectory_read_rows": trajectory_read_rows,
            "trajectory_read_stage1_supported_rows": trajectory_read_supported,
            "trajectory_read_stage1_support_coverage": ratio(trajectory_read_supported, trajectory_read_rows),
            "local_patch_action_rows": local_patch_action_rows,
            "local_patch_action_stage1_supported_rows": local_patch_action_supported,
            "local_patch_action_stage1_support_coverage": ratio(local_patch_action_supported, local_patch_action_rows),
            "lb_tr_action_rows": action_rows,
            "lb_tr_changed_action_rows": changed_action_rows,
            "selected_special_pages_with_frame_map": selected_pages_with_frame_map,
            "selected_special_pages_without_frame_map": selected_pages_without_frame_map,
            "selected_special_page_member_frame_refs": selected_page_frame_refs,
            "selected_special_page_member_frame_refs_stage1_supported": selected_page_frame_refs_supported,
            "selected_special_page_member_frame_refs_stage1_support_coverage": ratio(
                selected_page_frame_refs_supported,
                selected_page_frame_refs,
            ),
            "special_pages_observed": len(special_page_frames),
            "special_page_member_frame_count_min": min(page_frame_counts) if page_frame_counts else 0,
            "special_page_member_frame_count_median": median(page_frame_counts) if page_frame_counts else 0,
            "special_page_member_frame_count_max": max(page_frame_counts) if page_frame_counts else 0,
            "special_page_append_segment_count_max": max(special_page_append_segments.values()) if special_page_append_segments else 0,
            "direct_semantic_field_rows": direct_semantic_field_rows,
            "stage1_frame_support_available": trajectory_append_supported > 0 or local_patch_action_supported > 0,
            "trajectory_exact_token_object_provenance_available": False,
            "trajectory_exact_token_object_blocker": (
                "trajectory_special rows expose source_frame_id/page/token_type but no track/object/semantic fields; "
                "LB-TR action rows select special pages, while each special page may contain many source frames"
            ),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def v117_boundary() -> dict:
    if not V117_PROVENANCE.exists():
        return {"path": rel(V117_PROVENANCE), "exists": False}
    modes: Counter[str] = Counter()
    token_types: Counter[str] = Counter()
    rows = 0
    with V117_PROVENANCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            modes[row.get("provenance_mode", "")] += 1
            token_types[row.get("token_type", "")] += 1
    return {
        "path": rel(V117_PROVENANCE),
        "exists": True,
        "rows": rows,
        "provenance_modes": dict(sorted(modes.items())),
        "token_types": dict(sorted(token_types.items())),
        "classification": "historical_frame_aggregate_source_distribution_not_actual_flashinfer_memory_entry_policy",
    }


def report_text(summary: dict, trace_rows: list[dict]) -> str:
    lines = [
        "# Stage4-R20 LingBot Semantic Bridge Audit",
        "",
        f"- decision: `{summary['stage4_r20_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- exact_tr_semantic_policy_ready: `{summary['exact_tr_semantic_policy_ready']}`",
        f"- frame_supported_diagnostic_ready: `{summary['frame_supported_diagnostic_ready']}`",
        f"- local_patch_te_semantic_repair_ready: `{summary['local_patch_te_semantic_repair_ready']}`",
        "",
        "## Trace Coverage",
        "",
        "| label | trajectory append coverage | trajectory read coverage | local patch coverage | special pages | frames/page max | direct semantic rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trace_rows:
        lines.append(
            "| {label} | {append:.6f} | {read:.6f} | {local:.6f} | {pages} | {max_frames} | {direct} |".format(
                label=row["label"],
                append=row["trajectory_append_stage1_support_coverage"],
                read=row["trajectory_read_stage1_support_coverage"],
                local=row["local_patch_action_stage1_support_coverage"],
                pages=row["special_pages_observed"],
                max_frames=row["special_page_member_frame_count_max"],
                direct=row["direct_semantic_field_rows"],
            )
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
    support, frame_rows = load_stage1_frame_support()
    stage2_surfaces = load_stage2_surface_gate()
    trace_rows = [trace_audit_row(trace, support) for trace in TRACE_INPUTS]

    all_traces_exist = all(row["trace_exists"] for row in trace_rows)
    append_coverages = [row["trajectory_append_stage1_support_coverage"] for row in trace_rows if row["trajectory_append_rows"]]
    read_coverages = [row["trajectory_read_stage1_support_coverage"] for row in trace_rows if row["trajectory_read_rows"]]
    local_coverages = [row["local_patch_action_stage1_support_coverage"] for row in trace_rows if row["local_patch_action_rows"]]
    direct_semantic_rows = sum(int(row["direct_semantic_field_rows"]) for row in trace_rows)
    frame_supported_diagnostic_ready = bool(append_coverages) and min(append_coverages) >= 0.95
    local_patch_te_semantic_repair_ready = (
        bool(local_coverages)
        and min(local_coverages) >= 0.95
        and stage2_surfaces.get("LB-Local", {}).get("all_gate_pass") is True
    )
    exact_tr_semantic_policy_ready = False

    if not all_traces_exist:
        decision = "SEMANTIC_BRIDGE_AUDIT_INCOMPLETE_MISSING_TRACE"
    elif frame_supported_diagnostic_ready and not exact_tr_semantic_policy_ready:
        decision = "FRAME_SUPPORTED_BRIDGE_PASS_EXACT_TRAJECTORY_SEMANTIC_POLICY_BLOCKED"
    else:
        decision = "SEMANTIC_BRIDGE_FRAME_SUPPORT_INCOMPLETE"

    summary = {
        "schema": "acl2_v118tf_stage4_r20_lingbot_semantic_bridge_audit_summary_v1",
        "stage4_r20_decision": decision,
        "global_goal_achieved": False,
        "all_traces_exist": all_traces_exist,
        "trace_count": len(trace_rows),
        "stage1_frame_support_row_count": len(frame_rows),
        "stage1_frame_support_seq_counts": dict(Counter(row["seq"] for row in frame_rows)),
        "min_trajectory_append_stage1_support_coverage": min(append_coverages) if append_coverages else 0.0,
        "min_trajectory_read_stage1_support_coverage": min(read_coverages) if read_coverages else 0.0,
        "min_local_patch_action_stage1_support_coverage": min(local_coverages) if local_coverages else 0.0,
        "direct_semantic_field_rows": direct_semantic_rows,
        "frame_supported_diagnostic_ready": frame_supported_diagnostic_ready,
        "exact_tr_semantic_policy_ready": exact_tr_semantic_policy_ready,
        "local_patch_te_semantic_repair_ready": local_patch_te_semantic_repair_ready,
        "stage2_lb_local_gate_pass": stage2_surfaces.get("LB-Local", {}).get("all_gate_pass", False),
        "stage2_lb_trajectory_gate_pass": stage2_surfaces.get("LB-Trajectory", {}).get("all_gate_pass", False),
        "v117_boundary": v117_boundary(),
        "boundary": (
            "Stage1 semantic sidecar can support source_frame_id joins for real FlashInfer rows, "
            "but current LB-TR trajectory_special runtime rows do not expose track/object/semantic fields "
            "and LB-TR actions select special pages that may pack many source frames. Therefore this audit "
            "permits only a frame-supported diagnostic or a local-patch TE repair attempt; it does not permit "
            "a TR3/TE3 semantic success claim for trajectory-special provenance."
        ),
        "outputs": {
            "summary": rel(OUT / "stage4_r20_lingbot_semantic_bridge_audit_summary.json"),
            "trace_rows": rel(OUT / "stage4_r20_lingbot_semantic_bridge_trace_rows.csv"),
            "frame_support_rows": rel(OUT / "stage4_r20_frame_semantic_support_rows.csv"),
            "report": rel(OUT / "STAGE4_R20_LINGBOT_SEMANTIC_BRIDGE_AUDIT_REPORT.md"),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "stage4_r20_lingbot_semantic_bridge_trace_rows.csv", trace_rows)
    write_csv(OUT / "stage4_r20_frame_semantic_support_rows.csv", frame_rows)
    write_json(OUT / "stage4_r20_lingbot_semantic_bridge_audit_summary.json", summary)
    (OUT / "STAGE4_R20_LINGBOT_SEMANTIC_BRIDGE_AUDIT_REPORT.md").write_text(
        report_text(summary, trace_rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
