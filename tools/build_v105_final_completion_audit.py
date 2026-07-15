#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json_any(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for _ in path.open("rb"))


def _casebook_decision(summary: dict[str, Any]) -> dict[str, Any]:
    gates = summary.get("gates")
    if not isinstance(gates, dict):
        return {}
    decision = gates.get("casebook_final_decision")
    return decision if isinstance(decision, dict) else {}


def _visual_audit_rows(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        rows = []
    bad_rows = [
        {
            "scene_id": row.get("scene_id"),
            "variant_id": row.get("variant_id"),
            "decoded_frame_count": row.get("decoded_frame_count"),
            "expected_frame_count": row.get("expected_frame_count"),
            "sheet_group_count": row.get("sheet_group_count"),
            "expected_sheet_group_count": row.get("expected_sheet_group_count"),
            "sheet_groups_cover_expected_frames": row.get("sheet_groups_cover_expected_frames"),
        }
        for row in rows
        if isinstance(row, dict) and not bool(row.get("sheet_groups_cover_expected_frames"))
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path),
        "video_count": payload.get("video_count"),
        "frames_per_sheet": payload.get("frames_per_sheet"),
        "all_videos_decode_expected_frames": payload.get("all_videos_decode_expected_frames"),
        "all_sheet_groups_cover_expected_frames": payload.get("all_sheet_groups_cover_expected_frames"),
        "row_count": len(rows),
        "bad_row_count": len(bad_rows),
        "bad_rows": bad_rows,
        "sheet_group_jpg_count": sum(1 for _ in root.glob("**/sheet_groups/*.jpg")),
    }


def _audit_split_root(name: str, root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    decision = _casebook_decision(summary)
    visual_assessment_files = sorted(str(p) for p in root.rglob("*visual_assessment*.json"))
    return {
        "split_name": name,
        "root": str(root),
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "summary_sha256": _sha256(summary_path),
        "status": summary.get("status"),
        "cache_read_count": summary.get("cache_read_count"),
        "failure_count": summary.get("failure_count"),
        "casebook_status": decision.get("status"),
        "casebook_method_success": decision.get("method_success"),
        "casebook_full_dev_pass": decision.get("full_dev_pass"),
        "casebook_holdout_pass": decision.get("holdout_pass"),
        "casebook_visual_artifacts_pass": decision.get("visual_artifacts_pass"),
        "visual_assessment_file_count": len(visual_assessment_files),
        "visual_assessment_files": visual_assessment_files,
        "visual_audits": {
            "sgq_local": _visual_audit_rows(root, "sgq_local/full_frame_visual_audit/full_frame_visual_audit.json"),
            "local2history": _visual_audit_rows(root, "local2history/full_frame_visual_audit/full_frame_visual_audit.json"),
            "baselines": _visual_audit_rows(root, "baselines/full_frame_visual_audit/full_frame_visual_audit.json"),
        },
    }


def _collect_mv_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("mv_scene_window_metric_records.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            data = payload.get("rows", [])
        else:
            data = payload
        if not isinstance(data, list):
            data = []
        for row in data:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "path": str(path),
                    "split_name": row.get("split_name"),
                    "scene_id": row.get("scene_id"),
                    "variant_id": row.get("variant_id"),
                    "MV_AP_window": row.get("MV_AP_window"),
                    "MV_AP_scene": row.get("MV_AP_scene"),
                    "expected_stride_frame_count": row.get("expected_stride_frame_count"),
                    "available_mask_frame_count": row.get("available_mask_frame_count"),
                    "missing_expected_prediction_frame_count": row.get("missing_expected_prediction_frame_count"),
                    "full_scene_prediction_complete": row.get("full_scene_prediction_complete"),
                }
            )
    return rows


def _mv_scene_completion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_rows = [
        row
        for row in rows
        if str(row.get("metric_scope", "")).startswith("MV_AP_scene")
        or row.get("MV_AP_scene") is not None
    ]
    incomplete = [row for row in scene_rows if not bool(row.get("full_scene_prediction_complete"))]
    return {
        "scene_row_count": len(scene_rows),
        "scene_incomplete_row_count": len(incomplete),
        "scene_rows_complete": bool(scene_rows) and len(incomplete) == 0,
        "incomplete_scene_rows_first20": incomplete[:20],
    }


def _audit_core_visual_assessment(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "exists": False,
            "all_core_records_reviewed": False,
            "evidence": "No --core-visual-assessment-root was provided.",
        }
    records_path = root / "visual_assessment_records.json"
    summary_path = root / "visual_assessment_summary.json"
    manual_summary_path = root / "visual_assessment_manual_review_summary.json"
    records_payload = _read_json_any(records_path)
    records = records_payload if isinstance(records_payload, list) else []
    pending = sum(row.get("verdict") == "PENDING_MANUAL_REVIEW" for row in records if isinstance(row, dict))
    pass_rows = [
        row
        for row in records
        if isinstance(row, dict) and str(row.get("verdict", "")).startswith("PASS_CORE_VISUAL_REVIEW")
    ]
    hard_fail_rows = [
        {
            "split_name": row.get("split_name"),
            "source_kind": row.get("source_kind"),
            "scene_id": row.get("scene_id"),
            "sheet_group_index": row.get("sheet_group_index"),
            "verdict": row.get("verdict"),
        }
        for row in records
        if isinstance(row, dict)
        and row.get("verdict") not in {"PENDING_MANUAL_REVIEW"}
        and not str(row.get("verdict", "")).startswith("PASS_CORE_VISUAL_REVIEW")
    ]
    blank_count = sum(bool(row.get("blank_frame_present")) for row in records if isinstance(row, dict))
    unreadable_count = sum(not bool(row.get("all_frames_readable")) for row in records if isinstance(row, dict))
    return {
        "root": str(root),
        "exists": root.exists(),
        "records_path": str(records_path),
        "records_exists": records_path.exists(),
        "records_sha256": _sha256(records_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "manual_summary_path": str(manual_summary_path),
        "manual_summary_sha256": _sha256(manual_summary_path),
        "record_count": len(records),
        "pending_manual_review_count": pending,
        "pass_count": len(pass_rows),
        "hard_fail_row_count": len(hard_fail_rows),
        "hard_fail_rows": hard_fail_rows[:20],
        "blank_frame_present_count": blank_count,
        "unreadable_record_count": unreadable_count,
        "all_core_records_reviewed": bool(records) and pending == 0 and len(pass_rows) == len(records),
    }


def _audit_boundary_visual_assessment(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None:
        return {
            "exists": False,
            "manual_review_complete": False,
            "continuous_scene_level_id_claim": False,
            "evidence": "No --boundary-visual-assessment-summary was provided.",
        }
    summary = _read_json(summary_path)
    return {
        "path": str(summary_path),
        "exists": summary_path.exists(),
        "sha256": _sha256(summary_path),
        "manual_review_complete": bool(summary.get("manual_review_complete")),
        "record_count": summary.get("record_count"),
        "pass_no_obvious_switch_count": summary.get("pass_no_obvious_switch_count"),
        "potential_identity_failure_count": summary.get("potential_identity_failure_count"),
        "hard_identity_failure_count": summary.get("hard_identity_failure_count"),
        "uncertain_identity_witness_count": summary.get("uncertain_identity_witness_count"),
        "hard_or_potential_failure_count": summary.get("hard_or_potential_failure_count"),
        "failure_or_uncertain_count": summary.get("failure_or_uncertain_count"),
        "continuous_scene_level_id_claim": bool(summary.get("continuous_scene_level_id_claim")),
        "manual_review_conclusion": summary.get("manual_review_conclusion"),
        "hard_identity_failures": summary.get("hard_identity_failures", [])[:20],
        "potential_identity_failures": summary.get("potential_identity_failures", [])[:20],
        "uncertain_identity_witnesses": summary.get("uncertain_identity_witnesses", [])[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Stream4D v105 final completion audit without claiming success.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--holdout-root", required=True)
    parser.add_argument("--fullscene-summary", required=True)
    parser.add_argument("--mv-root", required=True)
    parser.add_argument("--complete-scene-mv-root", default=None)
    parser.add_argument("--fullscene-l2h-stitch-summary", default=None)
    parser.add_argument("--core-visual-assessment-root", default=None)
    parser.add_argument("--boundary-visual-assessment-summary", default=None)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    plan = Path(args.plan)
    dev_root = Path(args.dev_root)
    holdout_root = Path(args.holdout_root)
    fullscene_summary_path = Path(args.fullscene_summary)
    mv_root = Path(args.mv_root)
    complete_scene_mv_root = Path(args.complete_scene_mv_root) if args.complete_scene_mv_root else None
    fullscene_l2h_stitch_summary_path = Path(args.fullscene_l2h_stitch_summary) if args.fullscene_l2h_stitch_summary else None
    core_visual_root = Path(args.core_visual_assessment_root) if args.core_visual_assessment_root else None
    boundary_visual_summary_path = (
        Path(args.boundary_visual_assessment_summary) if args.boundary_visual_assessment_summary else None
    )
    output_root = Path(args.output_root)

    dev = _audit_split_root("dev", dev_root)
    holdout = _audit_split_root("holdout", holdout_root)
    fullscene = _read_json(fullscene_summary_path)
    mv_rows = _collect_mv_rows(mv_root)
    complete_scene_mv_rows = _collect_mv_rows(complete_scene_mv_root) if complete_scene_mv_root else mv_rows
    mv_scene_completion = _mv_scene_completion(complete_scene_mv_rows)
    fullscene_l2h_stitch = _read_json(fullscene_l2h_stitch_summary_path) if fullscene_l2h_stitch_summary_path else {}
    core_visual_assessment = _audit_core_visual_assessment(core_visual_root)
    boundary_visual_assessment = _audit_boundary_visual_assessment(boundary_visual_summary_path)

    visual_roots = [dev, holdout]
    all_visual_sheet_groups_cover = all(
        bool(audit.get("all_sheet_groups_cover_expected_frames"))
        for root in visual_roots
        for audit in root.get("visual_audits", {}).values()
    )
    any_manual_visual_assessment = bool(core_visual_assessment.get("all_core_records_reviewed")) or any(
        int(root.get("visual_assessment_file_count") or 0) > 0 for root in visual_roots
    )
    full_scene_prediction_complete = bool(mv_scene_completion.get("scene_rows_complete"))
    split_visual_protocols_complete = bool(dev.get("casebook_full_dev_pass")) and bool(holdout.get("casebook_holdout_pass"))
    cold_cache_ok = dev.get("cache_read_count") == 0 and holdout.get("cache_read_count") == 0
    failure_free = dev.get("failure_count") == 0 and holdout.get("failure_count") == 0
    fullscene_video_complete = bool(fullscene.get("all_scene_videos_complete"))
    fullscene_continuous_identity = bool(fullscene.get("continuous_scene_level_id_claim"))
    if fullscene_l2h_stitch:
        fullscene_video_complete = fullscene_video_complete and bool(fullscene_l2h_stitch.get("all_complete_scene_videos"))
        fullscene_continuous_identity = fullscene_continuous_identity or bool(fullscene_l2h_stitch.get("continuous_scene_level_id_claim"))
    if boundary_visual_assessment.get("exists"):
        fullscene_continuous_identity = fullscene_continuous_identity and bool(
            boundary_visual_assessment.get("continuous_scene_level_id_claim")
        )
    user_final_visual_confirmation = False

    blockers: list[dict[str, Any]] = []
    if not split_visual_protocols_complete:
        blockers.append(
            {
                "requirement": "dev_and_holdout_visual_protocols",
                "status": "incomplete",
                "evidence": {
                    "dev_full_dev_pass": dev.get("casebook_full_dev_pass"),
                    "holdout_holdout_pass": holdout.get("casebook_holdout_pass"),
                },
            }
        )
    if not all_visual_sheet_groups_cover:
        blockers.append({"requirement": "8_frame_sheet_groups_cover_expected_frames", "status": "incomplete"})
    if not any_manual_visual_assessment:
        blockers.append(
            {
                "requirement": "manual_visual_assessment_records",
                "status": "missing_or_incomplete",
                "evidence": core_visual_assessment,
            }
        )
    if not full_scene_prediction_complete:
        blockers.append(
            {
                "requirement": "complete_scene_mv_predictions",
                "status": "incomplete",
                "evidence": {
                    "complete_scene_mv_root": str(complete_scene_mv_root) if complete_scene_mv_root else str(mv_root),
                    **mv_scene_completion,
                },
            }
        )
    if not fullscene_continuous_identity:
        blockers.append(
            {
                "requirement": "continuous_full_scene_identity",
                "status": "not_proven",
                "evidence": {
                    "fullscene_summary": str(fullscene_summary_path),
                    "all_scene_videos_complete": fullscene_video_complete,
                    "continuous_scene_level_id_claim": fullscene.get("continuous_scene_level_id_claim"),
                    "chunk_windowed_internal": fullscene.get("chunk_windowed_internal"),
                    "not_claimed": fullscene.get("not_claimed"),
                    "l2h_stitch_summary": str(fullscene_l2h_stitch_summary_path) if fullscene_l2h_stitch_summary_path else None,
                    "l2h_stitch_candidate": {
                        "id_only_stitch_candidate": fullscene_l2h_stitch.get("id_only_stitch_candidate"),
                        "complete_scene_prediction_candidate": fullscene_l2h_stitch.get("complete_scene_prediction_candidate"),
                        "continuous_scene_level_id_claim": fullscene_l2h_stitch.get("continuous_scene_level_id_claim"),
                        "total_boundary_count": fullscene_l2h_stitch.get("total_boundary_count"),
                        "total_weak_boundary_count": fullscene_l2h_stitch.get("total_weak_boundary_count"),
                        "claim_boundary": fullscene_l2h_stitch.get("claim_boundary"),
                    }
                    if fullscene_l2h_stitch
                    else None,
                    "boundary_visual_assessment": boundary_visual_assessment,
                },
            }
        )
    blockers.append(
        {
            "requirement": "user_final_visual_confirmation",
            "status": "not_available_to_codex",
            "evidence": "User final confirmation cannot be generated by this audit script.",
        }
    )

    achieved = bool(
        split_visual_protocols_complete
        and cold_cache_ok
        and failure_free
        and all_visual_sheet_groups_cover
        and any_manual_visual_assessment
        and full_scene_prediction_complete
        and fullscene_continuous_identity
        and user_final_visual_confirmation
    )

    records = [
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "plan_read_current_file",
            "status": "evidence_recorded",
            "evidence": {"line_count": _count_lines(plan), "sha256": _sha256(plan), "path": str(plan)},
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "dev_visual_protocol",
            "status": "pass" if bool(dev.get("casebook_full_dev_pass")) else "incomplete",
            "evidence": dev,
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "holdout_visual_protocol",
            "status": "pass" if bool(holdout.get("casebook_holdout_pass")) else "incomplete",
            "evidence": holdout,
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "manual_visual_assessment_records",
            "status": "pass_core_64f_only" if bool(core_visual_assessment.get("all_core_records_reviewed")) else "missing_or_incomplete",
            "evidence": core_visual_assessment,
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "mv_scene_window_diagnostics",
            "status": "complete_rows" if full_scene_prediction_complete else "window_only_scene_incomplete",
            "evidence": {
                "mv_root": str(mv_root),
                "complete_scene_mv_root": str(complete_scene_mv_root) if complete_scene_mv_root else str(mv_root),
                "row_count": len(mv_rows),
                "complete_scene_row_count": mv_scene_completion.get("scene_row_count"),
                "complete_scene_incomplete_row_count": mv_scene_completion.get("scene_incomplete_row_count"),
                "scene_completion": mv_scene_completion,
            },
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "fullscene_l2h_stitch_candidate",
            "status": "candidate_complete_masks_identity_not_proven" if fullscene_l2h_stitch else "missing",
            "evidence": {
                "summary_path": str(fullscene_l2h_stitch_summary_path) if fullscene_l2h_stitch_summary_path else None,
                "summary_sha256": _sha256(fullscene_l2h_stitch_summary_path) if fullscene_l2h_stitch_summary_path else None,
                **fullscene_l2h_stitch,
            }
            if fullscene_l2h_stitch
            else {},
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "fullscene_boundary_manual_review",
            "status": (
                "no_go_identity_failure"
                if boundary_visual_assessment.get("manual_review_complete")
                and int(boundary_visual_assessment.get("failure_or_uncertain_count") or 0) > 0
                else "pass"
                if boundary_visual_assessment.get("continuous_scene_level_id_claim")
                else "missing_or_incomplete"
            ),
            "evidence": boundary_visual_assessment,
        },
        {
            "schema_version": "stream4d_v105_completion_requirement_row_v1",
            "requirement": "complete_scene_video_artifact",
            "status": "video_complete_identity_not_proven" if fullscene_video_complete else "incomplete",
            "evidence": {"summary_path": str(fullscene_summary_path), "summary_sha256": _sha256(fullscene_summary_path), **fullscene},
        },
    ]

    summary = {
        "schema_version": "stream4d_v105_final_completion_audit_v1",
        "achieved": achieved,
        "honest_status": "partial_diagnostic_or_no_go" if not achieved else "complete",
        "dev_root": str(dev_root),
        "holdout_root": str(holdout_root),
        "fullscene_summary": str(fullscene_summary_path),
        "fullscene_l2h_stitch_summary": str(fullscene_l2h_stitch_summary_path) if fullscene_l2h_stitch_summary_path else None,
        "boundary_visual_assessment_summary": str(boundary_visual_summary_path) if boundary_visual_summary_path else None,
        "mv_root": str(mv_root),
        "complete_scene_mv_root": str(complete_scene_mv_root) if complete_scene_mv_root else None,
        "core_visual_assessment_root": str(core_visual_root) if core_visual_root else None,
        "plan": {"path": str(plan), "line_count": _count_lines(plan), "sha256": _sha256(plan)},
        "cold_cache_ok": cold_cache_ok,
        "failure_free": failure_free,
        "split_visual_protocols_complete": split_visual_protocols_complete,
        "all_visual_sheet_groups_cover": all_visual_sheet_groups_cover,
        "manual_visual_assessment_present": any_manual_visual_assessment,
        "manual_visual_assessment_core_pass": bool(core_visual_assessment.get("all_core_records_reviewed")),
        "core_visual_assessment": core_visual_assessment,
        "boundary_visual_assessment_present": bool(boundary_visual_assessment.get("exists")),
        "boundary_manual_review_complete": bool(boundary_visual_assessment.get("manual_review_complete")),
        "boundary_hard_identity_failure_count": boundary_visual_assessment.get("hard_identity_failure_count"),
        "boundary_hard_or_potential_failure_count": boundary_visual_assessment.get("hard_or_potential_failure_count"),
        "boundary_failure_or_uncertain_count": boundary_visual_assessment.get("failure_or_uncertain_count"),
        "boundary_visual_assessment": boundary_visual_assessment,
        "full_scene_prediction_complete": full_scene_prediction_complete,
        "fullscene_video_complete": fullscene_video_complete,
        "fullscene_continuous_identity_proven": fullscene_continuous_identity,
        "user_final_visual_confirmation": user_final_visual_confirmation,
        "mv_row_count": len(mv_rows),
        "mv_incomplete_row_count": sum(not bool(row.get("full_scene_prediction_complete")) for row in mv_rows),
        "complete_scene_mv_scene_row_count": mv_scene_completion.get("scene_row_count"),
        "complete_scene_mv_incomplete_scene_row_count": mv_scene_completion.get("scene_incomplete_row_count"),
        "fullscene_l2h_stitch_candidate_present": bool(fullscene_l2h_stitch),
        "fullscene_l2h_stitch_total_weak_boundary_count": fullscene_l2h_stitch.get("total_weak_boundary_count") if fullscene_l2h_stitch else None,
        "blockers": blockers,
        "records_json": str(output_root / "completion_audit_records.json"),
    }
    _write_json(output_root / "completion_audit_records.json", records)
    _write_json(output_root / "completion_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
