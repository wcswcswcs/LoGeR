#!/usr/bin/env python3
"""Build mandatory v81 Phase11 visual rediscovery bundle.

This script copies or derives panels from real v81 artifacts. It does not
invent missing QK/TTT/SWA evidence; unavailable evidence is recorded as a
question for the next mechanism design.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/report_final")
PHASE1_ROWS = REPORT_ROOT / "phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
PHASE2_ROOT = REPORT_ROOT / "phase2_long_window_visual_confirmation"
PHASE3_SUMMARY = REPORT_ROOT / "phase3_selected_write_risk_rule/bad_good_confusion_matrix.json"
PHASE4_ROOT = REPORT_ROOT / "phase4_read_swa_confirmation"
PHASE6_ROOT = REPORT_ROOT / "phase6_merge_boundary_typeb_rescue"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase11_rediscovery"

QUESTION_FIELDS = [
    "priority",
    "window_id",
    "case_type",
    "failed_gate",
    "failure_evidence",
    "visual_question",
    "required_next_hook_or_artifact",
    "source_panel_group",
    "source_panel",
    "candidate_hypothesis",
]
REVIEW_FIELDS = [
    "group",
    "dest_path",
    "source_path",
    "source_kind",
    "status",
    "review_note",
    "bytes",
    "width",
    "height",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or [])
    if not fieldnames:
        for row in rows_list:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _png_size(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() != ".png" or not path.is_file():
        return None, None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _copy_panel(src: Path, out_dir: Path, group: str, dest_name: str, source_kind: str, note: str) -> dict[str, Any]:
    dst = out_dir / group / dest_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "group": group,
        "dest_path": str(dst),
        "source_path": str(src),
        "source_kind": source_kind,
        "status": "missing",
        "review_note": note,
        "bytes": 0,
        "width": None,
        "height": None,
    }
    if not src.is_file():
        return row
    shutil.copy2(src, dst)
    width, height = _png_size(dst)
    size = dst.stat().st_size
    ok = bool(size > 0 and (dst.suffix.lower() != ".png" or (width and height)))
    row.update({"status": "ok" if ok else "invalid_image", "bytes": size, "width": width, "height": height})
    return row


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _copy_core_panels(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qk_windows = [
        "seq02_chunks062_066",
        "seq02_chunks066_070",
        "seq01_chunks006_010",
        "seq05_chunks081_085",
    ]
    for window in qk_windows:
        rows.append(
            _copy_panel(
                PHASE2_ROOT / "selected_write_vs_random_panels" / f"{window}_selected_support.png",
                out_dir,
                "qk_pair_panels",
                f"{window}_selected_support.png",
                "phase2_selected_write_support_panel",
                "Selected-write support vs random visual proxy; diagnostic only.",
            )
        )

    read_pairs = [
        ("seq01_chunks006_010", "chunk008"),
        ("seq01_chunks008_012", "chunk010"),
        ("seq05_chunks081_085", "chunk081"),
    ]
    swa_pairs = [
        ("seq01_chunks006_010", "chunk008"),
        ("seq01_chunks008_012", "chunk010"),
        ("seq02_chunks062_066", "chunk066"),
        ("seq05_chunks081_085", "chunk083"),
    ]
    for window, chunk in read_pairs:
        rows.append(
            _copy_panel(
                PHASE4_ROOT / "read_confirmation_maps" / f"{window}_{chunk}_read.png",
                out_dir,
                "swa_kv_handoff_panels",
                f"{window}_{chunk}_read.png",
                "phase4_read_confirmation_proxy_map",
                "READ confirmation proxy map; Phase4 did not prove action-ready alignment.",
            )
        )
    for window, chunk in swa_pairs:
        rows.append(
            _copy_panel(
                PHASE4_ROOT / "swa_confirmation_maps" / f"{window}_{chunk}_swa.png",
                out_dir,
                "swa_kv_handoff_panels",
                f"{window}_{chunk}_swa.png",
                "phase4_swa_confirmation_proxy_map",
                "SWA confirmation proxy map; used to inspect READ/SWA mismatch.",
            )
        )

    ttt_windows = [
        ("seq02_chunks062_066", "bad"),
        ("seq02_chunks066_070", "bad"),
        ("seq02_chunks041_045", "good"),
        ("seq05_chunks006_010", "false_positive"),
    ]
    for window, case_type in ttt_windows:
        rows.append(
            _copy_panel(
                PHASE2_ROOT / "long_window_panels" / f"{window}_{case_type}.png",
                out_dir,
                "ttt_write_less_panels",
                f"{window}_{case_type}.png",
                "phase2_long_window_panel",
                "Long-window visual context for TTT write-less questions; no Phase5 action claim.",
            )
        )

    merge_windows = [
        ("seq01_chunks005_009", "bad"),
        ("seq01_chunks006_010", "bad"),
        ("seq01_chunks007_011", "bad"),
        ("seq01_chunks008_012", "bad"),
        ("seq01_chunks009_013", "bad"),
        ("seq05_chunks081_085", "bad"),
    ]
    for window, case_type in merge_windows:
        rows.append(
            _copy_panel(
                PHASE2_ROOT / "long_window_panels" / f"{window}_{case_type}.png",
                out_dir,
                "merge_boundary_panels",
                f"{window}_{case_type}.png",
                "phase2_typeb_long_window_panel",
                "Type-B merge-boundary long-window panel.",
            )
        )
        rows.append(
            _copy_panel(
                PHASE2_ROOT / "downstream_direction_panels" / f"{window}_downstream_direction.png",
                out_dir,
                "merge_boundary_panels",
                f"{window}_downstream_direction.png",
                "phase2_downstream_direction_panel",
                "Type-B downstream direction diagnostic panel.",
            )
        )
    return rows


def _try_make_phase6_plot(out_dir: Path) -> dict[str, Any]:
    decisions = _read_csv(PHASE6_ROOT / "targeted_overlap_outlier_smoke/v81_typeb_overlap_outlier_decisions.csv")
    dst = out_dir / "merge_boundary_panels" / "phase6_overlap_outlier_improvement_summary.png"
    row = {
        "group": "merge_boundary_panels",
        "dest_path": str(dst),
        "source_path": str(PHASE6_ROOT / "targeted_overlap_outlier_smoke/v81_typeb_overlap_outlier_decisions.csv"),
        "source_kind": "phase6_evaluator_plot",
        "status": "missing",
        "review_note": "Chunk-level improvement plot from Phase6 evaluator CSV.",
        "bytes": 0,
        "width": None,
        "height": None,
    }
    if not decisions:
        return row
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency fallback
        row["status"] = f"matplotlib_unavailable:{exc}"
        return row
    chunks = [int(r["chunk"]) for r in decisions]
    head = [_safe_float(r.get("head_tail_improvement_vs_baseline_ratio")) or 0.0 for r in decisions]
    overlap = [_safe_float(r.get("overlap_improvement_vs_baseline_ratio")) or 0.0 for r in decisions]
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    x = list(range(len(chunks)))
    width = 0.36
    ax.bar([v - width / 2 for v in x], head, width=width, label="head-tail")
    ax.bar([v + width / 2 for v in x], overlap, width=width, label="overlap-future")
    ax.axhline(0.05, color="crimson", linewidth=1.2, linestyle="--", label="5% gate")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chunks])
    ax.set_xlabel("seq01 target chunk")
    ax.set_ylabel("improvement vs native baseline ratio")
    ax.set_title("v81 Phase6 targeted overlap-outlier smoke")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)
    width_px, height_px = _png_size(dst)
    size = dst.stat().st_size
    row.update({"status": "ok", "bytes": size, "width": width_px, "height": height_px})
    return row


def _build_questions() -> list[dict[str, Any]]:
    phase1 = _read_csv(PHASE1_ROWS)
    phase6_typeb = _read_csv(PHASE6_ROOT / "typeb_case_rows.csv")
    phase3 = _read_json(PHASE3_SUMMARY)
    best_profile_name = str(phase3.get("best_profile", ""))
    best_profile_metrics = {}
    profiles = phase3.get("profiles")
    if isinstance(profiles, dict):
        value = profiles.get(best_profile_name)
        if isinstance(value, dict):
            best_profile_metrics = value
    rows: list[dict[str, Any]] = []
    seq02_bad = [row for row in phase1 if row.get("seq") == "02" and row.get("case_type") == "bad"][:2]
    for row in seq02_bad:
        window = row.get("window_id", "")
        rows.append(
            {
                "priority": "P0",
                "window_id": window,
                "case_type": row.get("case_type"),
                "failed_gate": "Phase3 recall/coverage and Phase4 READ/SWA alignment",
                "failure_evidence": (
                    f"selected_low_support_ratio={row.get('selected_low_support_ratio')}; "
                    f"cluster_len={row.get('continuous_low_support_cluster_len')}; "
                    f"phase3_best_profile={best_profile_name}; "
                    f"phase3_best_recall={best_profile_metrics.get('bad_recall')}"
                ),
                "visual_question": "Which selected-write low-support patches are downstream-harmful and also READ/SWA confirmed, rather than merely co-located with bad geometry?",
                "required_next_hook_or_artifact": "READ_SWA_TTT_ALIGNMENT_LOG with random same-mass alignment baseline",
                "source_panel_group": "qk_pair_panels",
                "source_panel": f"{window}_selected_support.png",
                "candidate_hypothesis": "Selected-write risk is a local cluster detector, not sufficient as a write actuator without alignment confirmation.",
            }
        )
    for row in phase6_typeb[:4]:
        window = row.get("window_id", "")
        rows.append(
            {
                "priority": "P0",
                "window_id": window,
                "case_type": "bad",
                "failed_gate": "Phase6 Type-B overlap-outlier smoke",
                "failure_evidence": (
                    f"future={row.get('downstream_future_consistency')}; "
                    f"radio_boundary={row.get('radio_boundary_mean')}; "
                    "Phase6 head_tail_pass_count=0 overlap_pass_count=0"
                ),
                "visual_question": "Is the failure carried by merge/gauge boundary state rather than token-level selected-write support?",
                "required_next_hook_or_artifact": "MERGE_BOUNDARY_OUTLIER_WEIGHT plus direct non-GT gauge/scale state trace, with good-pair protection",
                "source_panel_group": "merge_boundary_panels",
                "source_panel": f"{window}_bad.png",
                "candidate_hypothesis": "Type-B seq01 windows need direct merge/gauge state controller; overlap-outlier downweight is too weak or no-op under current guards.",
            }
        )
    rows.append(
        {
            "priority": "P1",
            "window_id": "seq05_chunks081_085",
            "case_type": "bad",
            "failed_gate": "Phase6 support-map coverage",
            "failure_evidence": "Type-B but support_map_exists=false; scale_cv=0.40164490393358376",
            "visual_question": "Does seq05 share the same merge/gauge failure as seq01, or is this a scale-side observability case?",
            "required_next_hook_or_artifact": "seq05 overlap support map plus scale-side-state confidence trace",
            "source_panel_group": "merge_boundary_panels",
            "source_panel": "seq05_chunks081_085_bad.png",
            "candidate_hypothesis": "Coverage gap blocks method validation; support-map generation must cover non-seq01 Type-B rows before action promotion.",
        }
    )
    rows.append(
        {
            "priority": "P1",
            "window_id": "multi_seq_good_false_positive",
            "case_type": "good_or_false_positive",
            "failed_gate": "good-pair protection missing in Phase6",
            "failure_evidence": "Phase6 targeted smoke was seq01 bad-only; good_pair_coverage=false",
            "visual_question": "Which good/false-positive windows would be harmed by a merge/gauge boundary controller?",
            "required_next_hook_or_artifact": "good-pair merge-boundary controls with J_mid worsens <=2%",
            "source_panel_group": "ttt_write_less_panels",
            "source_panel": "seq05_chunks006_010_false_positive.png",
            "candidate_hypothesis": "Any deployable v81 controller must be validated on good/false-positive windows, not only bad canary chunks.",
        }
    )
    return rows


def _write_text_reports(out_dir: Path, audit: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]) -> None:
    insight = [
        "# ACL2 v81 Phase11 Visual Insight",
        "",
        f"status: `{audit.get('status')}`",
        f"visual_audit_gate_pass: `{audit.get('visual_audit_gate_pass')}`",
        "method_gate_claimed: `false`",
        "",
        "## Evidence Chain",
        "",
        "- Phase1/2 confirmed a real long-window case bank and visual panels.",
        "- Phase3 selected-write risk rule failed recall/coverage; best profile stayed seq02-local.",
        "- Phase4 READ/SWA proxy maps had nonzero mass but low alignment and no random alignment baseline.",
        "- Phase6 Type-B seq01 smoke ran real overlap-outlier jobs, but did not pass controls.",
        "",
        "## Main Insight",
        "",
        "v81 currently separates into two failure bodies: seq02 selected-write clusters are visible but not action-confirmed, while seq01 Type-B merge/gauge failures are not explained by selected-write and are not rescued by existing overlap-outlier downweight.",
        "",
        "## No-Go Boundary",
        "",
        "A final method success claim is not supported. The next valid mechanism must introduce a direct merge/gauge or scale-side-state controller with READ/SWA/random alignment and good-pair protection, rather than another scalar qscale or selected-write threshold sweep.",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(insight) + "\n", encoding="utf-8")

    hypotheses = [
        "# ACL2 v81 New Hypothesis Bank",
        "",
        "## H1 Direct Merge/Gauge State Controller",
        "Use a non-GT boundary-local scale/gauge state trace and update it only when semantic/geometric overlap evidence beats random and good-pair protection holds.",
        "",
        "## H2 READ/SWA/TTT Alignment First",
        "Do not use selected-write clusters as actions until READ/SWA/TTT alignment exceeds a random same-mass baseline on multiple sequences.",
        "",
        "## H3 Type-B Coverage Expansion",
        "Generate support maps for seq01 center11 and seq05 center83 before any Phase6 promotion; current Type-B smoke covers only 4 eligible seq01 chunks.",
        "",
        "## H4 Good-Pair Merge Protection",
        "Evaluate merge-boundary actions on good/false-positive windows with a <=2% worsen gate before claiming method progress.",
        "",
        "## H5 Overlap-Outlier No-Op Diagnosis",
        "Current overlap-outlier downweight is near-no-op or loses to controls under native-overlap rejection; inspect merge_state_trace for applied weight mass before tuning any scalar.",
    ]
    (out_dir / "new_hypothesis_bank.md").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")

    report = [
        "# ACL2 v81 Phase11 Rediscovery Report",
        "",
        f"Gate pass: `{audit.get('gate_pass')}`",
        f"Status: `{audit.get('status')}`",
        "",
        "## Questions",
        "",
    ]
    for row in questions:
        report.append(f"- {row['priority']} {row['window_id']}: {row['visual_question']}")
    (out_dir / "rediscovery_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _group_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        group = str(row["group"])
        out.setdefault(group, {"count": 0, "ok": 0})
        out[group]["count"] += 1
        if row.get("status") == "ok":
            out[group]["ok"] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _copy_core_panels(args.out_dir)
    rows.append(_try_make_phase6_plot(args.out_dir))
    questions = _build_questions()
    _write_csv(args.out_dir / "failed_long_window_to_visual_question.csv", questions, QUESTION_FIELDS)
    _write_csv(args.out_dir / "visual_review.csv", rows, REVIEW_FIELDS)
    group_counts = _group_counts(rows)
    required_groups = ["qk_pair_panels", "swa_kv_handoff_panels", "ttt_write_less_panels", "merge_boundary_panels"]
    required_phase11_panel_sets_present = all(group_counts.get(group, {}).get("ok", 0) > 0 for group in required_groups)
    questions_ok = bool(questions)
    visual_audit_gate_pass = bool(required_phase11_panel_sets_present and questions_ok)
    audit = {
        "status": "rediscovery_required",
        "gate_pass": visual_audit_gate_pass,
        "visual_audit_gate_pass": visual_audit_gate_pass,
        "required_phase11_panel_sets_present": required_phase11_panel_sets_present,
        "required_groups": required_groups,
        "group_counts": group_counts,
        "question_count": len(questions),
        "review_rows": len(rows),
        "method_gate_claimed": False,
        "v81_goal_achieved": False,
        "diagnostic_only": True,
        "artifact_note": "Phase11 bundle is a visual rediscovery/audit handoff, not a runtime method success claim.",
    }
    _write_text_reports(args.out_dir, audit, questions)
    _write_json(args.out_dir / "visual_integrity_audit.json", audit)
    print(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
