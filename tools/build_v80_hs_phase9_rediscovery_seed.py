#!/usr/bin/env python3
"""Build a partial Phase9 rediscovery seed from a v80 HS bridge summary."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument("--swa-ttt-summary", type=Path, default=None)
    parser.add_argument("--short-qk-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _label_text(row: dict[str, Any], key: str) -> str:
    labels = row.get(key, [])
    if not isinstance(labels, list):
        return ""
    return ";".join(f"{x.get('label_name')}:{float(x.get('fraction', 0.0)):.3f}" for x in labels[:5])


def _copy_if_exists(src_text: str | None, dst_dir: Path) -> Path | None:
    if not src_text:
        return None
    src = Path(src_text)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _case_id(row: dict[str, Any]) -> str:
    return f"frame{int(row['frame']):06d}_chunk{int(row['primary_chunk_id']):03d}_bridge"


def _write_failed_questions(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "frame",
        "chunk",
        "delta_error_vs_baseline_m",
        "delta_error_vs_control_m",
        "visual_question",
        "visual_panel_png",
    ]
    with (out_dir / "failed_case_to_visual_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            panel_name = Path(str(row.get("visual_panel_png") or "")).name
            writer.writerow(
                {
                    "case_id": _case_id(row),
                    "frame": row["frame"],
                    "chunk": row["primary_chunk_id"],
                    "delta_error_vs_baseline_m": row.get("delta_error_vs_baseline_m"),
                    "delta_error_vs_control_m": row.get("delta_error_vs_control_m"),
                    "visual_question": (
                        "Does READ/world error overlap contain a stable semantic carrier, "
                        "or is it dominated by risky sky/tree while TTT action writes elsewhere?"
                    ),
                    "visual_panel_png": str(Path("long_ttt_branch_visual_panels") / panel_name) if panel_name else "",
                }
            )


def _write_visual_review(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame",
        "chunk",
        "read_world_overlap_mass",
        "proposed_static_carrier_mass",
        "proposed_static_carrier_given_read_world",
        "risky_semantic_given_read_world",
        "action_top_given_proposed_static_carrier",
        "negative_given_risky_read_world",
        "read_world_labels",
        "proposed_static_carrier_labels",
        "risky_read_world_labels",
        "review_note",
    ]
    with (out_dir / "visual_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row["frame"],
                    "chunk": row["primary_chunk_id"],
                    "read_world_overlap_mass": row.get("read_world_overlap_mass"),
                    "proposed_static_carrier_mass": row.get("proposed_static_carrier_mass"),
                    "proposed_static_carrier_given_read_world": row.get("proposed_static_carrier_given_read_world"),
                    "risky_semantic_given_read_world": row.get("risky_semantic_given_read_world"),
                    "action_top_given_proposed_static_carrier": row.get("action_top_given_proposed_static_carrier"),
                    "negative_given_risky_read_world": row.get("negative_given_risky_read_world"),
                    "read_world_labels": _label_text(row, "read_world_labels"),
                    "proposed_static_carrier_labels": _label_text(row, "proposed_static_carrier_labels"),
                    "risky_read_world_labels": _label_text(row, "risky_read_world_labels"),
                    "review_note": "partial Phase9 TTT/trajectory visual seed; missing panel groups are reported in visual_integrity_audit.json",
                }
            )


def _image_checks(paths: list[Path]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for path in paths:
        image = Image.open(path)
        bbox = image.getbbox()
        checks.append(
            {
                "path": str(path),
                "size": list(image.size),
                "nonblank_bbox": list(bbox) if bbox else None,
                "bytes": path.stat().st_size,
            }
        )
    return checks


def _write_text_reports(
    out_dir: Path,
    bridge: Path,
    data: dict[str, Any],
    swa_ttt: dict[str, Any] | None,
    short_qk: dict[str, Any] | None,
) -> None:
    agg = data["aggregate"]
    rows = data["rows"]
    frames = ", ".join(str(row["frame"]) for row in rows)
    first_panel = ""
    first_plot = ""
    if rows:
        panel = rows[0].get("visual_panel_png")
        if panel:
            first_panel = f"`long_ttt_branch_visual_panels/{Path(str(panel)).name}`"
    for path in data.get("plot_paths", {}).values():
        first_plot = f"`merge_boundary_visual_panels/{Path(str(path)).name}`"
        break
    baseline = data.get("baseline", "baseline")
    candidate = data.get("candidate", "candidate")
    control = data.get("control")
    control_line = (
        f"- {candidate} vs {control} mean delta: `{agg.get('delta_error_vs_control_m_mean')}` m.\n"
        if control
        else "- No same-run control delta was provided for this seed.\n"
    )
    swa_lines = ""
    if swa_ttt:
        swa_agg = swa_ttt.get("aggregate", {})
        swa_lines = f"""
SWA-to-TTT seed evidence:

- Selected bad-frame SWA cache-V top to TTT action overlap mean: `{swa_agg.get('selected_bad_action_top_given_swa_cache_v_top_mean')}`.
- Selected bad-frame SWA cache-V top to world-delta top overlap mean: `{swa_agg.get('selected_bad_world_top_given_swa_cache_v_top_mean')}`.
- Selected bad-frame SWA cache-V top to READ q90 overlap mean: `{swa_agg.get('selected_bad_read_q90_given_swa_cache_v_top_mean')}`.
- Selected bad-frame TTT positive role within SWA cache-V top mean: `{swa_agg.get('selected_bad_positive_given_swa_cache_v_top_mean')}`.
"""
    qk_lines = ""
    if short_qk:
        qk_agg = short_qk.get("aggregate", {})
        qk_lines = f"""
Short Q/K seed evidence:

- Selected bad-frame frame-Q top to frame-K top proxy-overlap mean: `{qk_agg.get('selected_bad_frame_q_top_given_frame_k_top_mean')}`.
- Selected bad-frame global-Q top to global-K top proxy-overlap mean: `{qk_agg.get('selected_bad_global_q_top_given_global_k_top_mean')}`.
- Selected bad-frame frame-Q top to READ q90 overlap mean: `{qk_agg.get('selected_bad_read_q90_given_frame_q_top_mean')}`.
- Selected bad-frame frame-Q top to world-delta top overlap mean: `{qk_agg.get('selected_bad_world_top_given_frame_q_top_mean')}`.
- Selected bad-frame global-Q top to TTT action top overlap mean: `{qk_agg.get('selected_bad_action_top_given_global_q_top_mean')}`.

Note: Q/K overlap here is PCA-energy top-mask overlap, not a true attention dot-product.
"""
    insight = f"""# Phase9 Rediscovery Seed: Geometry/TTT/Semantic Bridge

Status: rediscovery visual seed; visual gate status is recorded in `visual_integrity_audit.json`.

This is not a method-success claim.

Source bridge: `{bridge}`

Key evidence:

- Selected frames: {frames}.
- {candidate} vs {baseline} mean delta: `{agg['delta_error_vs_baseline_m_mean']}` m.
{control_line}- Example TTT/semantic panel: {first_panel or "`missing`"}.
- Example trajectory plot: {first_plot or "`missing`"}.
- READ q90 and world-delta top overlap mass mean: `{agg['read_world_overlap_mass_mean']}`.
- Proposed stable semantic carrier mass mean: `{agg['proposed_static_carrier_mass_mean']}`.
- Proposed stable carrier among READ-world overlap mean: `{agg['proposed_static_carrier_given_read_world_mean']}`.
- Risky semantics among READ-world overlap mean: `{agg['risky_semantic_given_read_world_mean']}`.
- TTT action-top coverage of proposed static carrier mean: `{agg['action_top_given_proposed_static_carrier_mean']}`.
- Negative role on risky READ-world mean: `{agg['negative_given_risky_read_world_mean']}`.

Interpretation:

The selected geometry frames do have READ/world overlap, but the overlap is dominated by risky or non-static semantic regions in the selected rows. The stable semantic carrier is very small and TTT action top does not cover it in this diagnostic. This supports a carrier-break interpretation: semantic READ/SWA evidence can describe part of the scene, but the current TTT action write surface is spatially disjoint from that evidence.
{swa_lines}
{qk_lines}

Decision for current branch:

Do not continue long role-strength or global-write sweeps from this evidence alone. This seed supports returning to merge/gauge or Q/K/SWA-to-TTT alignment, while preserving this TTT evidence as a failed-carrier visual seed.
"""
    (out_dir / "visual_insight.md").write_text(insight, encoding="utf-8")

    hypothesis = f"""# New Semantic Memory Hypothesis Bank Seed

Status: partial, generated from current geometry/TTT/semantic bridge evidence only.

## HYP-01: READ-world risky overlap is a no-persistent signal

- memory body: TTT long memory
- case type: selected frame/chunk bridge rows, frames {frames}
- visual evidence file: {first_panel or "`long_ttt_branch_visual_panels/`"}
- semantic role: READ/world overlap dominated by sky/tree risky regions; risky overlap is mostly negative, not positive
- proposed action point: no-persistent or one-hop TTL for risky READ-world overlap; do not promote it to positive persistent write
- expected metric: reduce downstream future/boundary drift only if this overlap was being persistently written
- controls: same-mass random role, geometry-only TTL, label shuffle
- stop rule: if action-top remains disjoint from READ/world carrier, stop TTT write sweep and move to merge/gauge

## HYP-02: Stable semantic carrier is too sparse for current TTT write

- memory body: TTT long memory
- case type: same selected bridge frames
- visual evidence file: `visual_review.csv`
- semantic role: proposed stable carrier = READ q90 AND world-delta top AND static/structure labels
- proposed action point: only test a tiny smoke if runtime can align TTT action to this carrier without GT; otherwise do not run
- expected metric: stable carrier mass must be large enough and action-top overlap must become nonzero before trajectory promotion
- controls: same-write-mass random and label/conf shuffle
- stop rule: if proposed static carrier mass stays below 1% or action-top coverage stays 0, mark carrier-not-current and return to SWA/merge

## HYP-03: Current geometry failure is more likely merge/gauge than semantic TTT-write carrier

- memory body: SWA/merge boundary and long TTT interaction
- case type: selected bridge frames whose TTT action carrier is disjoint from READ/SWA evidence
- visual evidence file: {first_plot or "`merge_boundary_visual_panels/`"}
- semantic role: semantic can explain error-region labels but not the TTT action carrier
- proposed action point: run boundary-local merge/gauge diagnostic before any new TTT role sweep
- expected metric: boundary_step_error/global_future_from_boundary should improve or expose a non-semantic gauge bottleneck
- controls: native/no-gate, same-mass random, best TTT-only/single-path control
- stop rule: if boundary/gauge diagnostic also matches random/no-op, do not claim semantic success; keep as rediscovery evidence

## HYP-04: Current SWA cache carrier is not enough to rescue current TTT

- memory body: SWA-to-TTT handoff
- case type: selected bridge frames with available SWA-to-TTT alignment panels
- visual evidence file: `mid_swa_qkv_visual_panels/` if populated
- semantic role: SWA cache-V/K/Q top regions should overlap READ q90, world-delta top, and TTT action if SWA is the missing carrier
- proposed action point: do not promote SWA cache energy directly to TTT persistent write unless overlap with READ/world/action becomes nontrivial and beats random
- expected metric: action/world/read overlap should rise before any trajectory smoke promotion
- controls: same-mass random role and same-mass SWA/source masks
- stop rule: if SWA cache-V top has low READ/world/action overlap and zero TTT positive role in selected bad frames, do not launch a long SWA-to-TTT sweep

## HYP-05: Short Q/K carrier must be fixed before semantic write promotion

- memory body: short READ/global-frame attention
- case type: selected bridge frames; Q/K dump availability must be audited
- visual evidence file: `short_qk_pair_visual_panels/` if populated
- semantic role: Q/K carrier should explain where READ q90 and geometry world-delta concentrate before SWA/TTT can preserve or write it
- proposed action point: test query-key pair read bias only if Q/K top masks align with READ/world/action better than current failed carrier evidence
- expected metric: READ q90 and world-delta overlap should rise before any downstream SWA or TTT promotion
- controls: same-mass random Q/K masks, geometry-only Q/K control, semantic label/conf shuffle
- stop rule: if Q/K proxy overlap is low or unrelated to READ/world/action on selected bad frames, do not promote semantic TTT writes from this branch
"""
    (out_dir / "new_semantic_memory_hypothesis_bank.md").write_text(hypothesis, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    data = json.loads(args.bridge_summary.read_text(encoding="utf-8"))
    swa_ttt = (
        json.loads(args.swa_ttt_summary.read_text(encoding="utf-8"))
        if args.swa_ttt_summary is not None and args.swa_ttt_summary.exists()
        else None
    )
    short_qk = (
        json.loads(args.short_qk_summary.read_text(encoding="utf-8"))
        if args.short_qk_summary is not None and args.short_qk_summary.exists()
        else None
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "long_ttt_branch_visual_panels",
        "merge_boundary_visual_panels",
        "short_qk_pair_visual_panels",
        "mid_swa_qkv_visual_panels",
    ]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    copied_panels = [
        copied
        for row in data["rows"]
        if (copied := _copy_if_exists(row.get("visual_panel_png"), args.out_dir / "long_ttt_branch_visual_panels")) is not None
    ]
    copied_swa_panels: list[Path] = []
    if swa_ttt:
        swa_root = Path(str(swa_ttt.get("outputs", {}).get("visual_panels_dir", "")))
        if swa_root.exists():
            for path in sorted(swa_root.glob("*.png")):
                copied = _copy_if_exists(str(path), args.out_dir / "mid_swa_qkv_visual_panels")
                if copied is not None:
                    copied_swa_panels.append(copied)
    copied_qk_panels: list[Path] = []
    if short_qk:
        qk_root = Path(str(short_qk.get("outputs", {}).get("visual_panels_dir", "")))
        if qk_root.exists():
            for path in sorted(qk_root.glob("*.png")):
                copied = _copy_if_exists(str(path), args.out_dir / "short_qk_pair_visual_panels")
                if copied is not None:
                    copied_qk_panels.append(copied)
    copied_plots = [
        copied
        for plot_path in data.get("plot_paths", {}).values()
        if (copied := _copy_if_exists(plot_path, args.out_dir / "merge_boundary_visual_panels")) is not None
    ]
    _write_failed_questions(args.out_dir, data["rows"])
    _write_visual_review(args.out_dir, data["rows"])
    _write_text_reports(args.out_dir, args.bridge_summary, data, swa_ttt, short_qk)
    if not copied_qk_panels:
        (args.out_dir / "short_qk_pair_visual_panels" / "README.md").write_text(
            "Not generated in this partial v80 bridge seed. Required for a full Phase9 visual audit pass.\n",
            encoding="utf-8",
        )
    else:
        readme = args.out_dir / "short_qk_pair_visual_panels" / "README.md"
        if readme.exists():
            readme.unlink()
    if not copied_swa_panels:
        (args.out_dir / "mid_swa_qkv_visual_panels" / "README.md").write_text(
            "Not generated in this partial v80 bridge seed. Required for a full Phase9 visual audit pass.\n",
            encoding="utf-8",
        )
    else:
        readme = args.out_dir / "mid_swa_qkv_visual_panels" / "README.md"
        if readme.exists():
            readme.unlink()

    checks = _image_checks(copied_panels + copied_swa_panels + copied_qk_panels + copied_plots)
    required_panel_sets_present = bool(copied_panels and copied_plots and copied_swa_panels and copied_qk_panels)
    all_images_nonblank = all(check["nonblank_bbox"] is not None and int(check["bytes"]) > 0 for check in checks)
    visual_gate_pass = bool(required_panel_sets_present and all_images_nonblank)
    audit = {
        "schema": "acl2_v80_phase9_rediscovery_seed_visual_integrity_v1",
        "status": "visual_audit_ready" if visual_gate_pass else "partial",
        "gate_pass": visual_gate_pass,
        "method_gate_claimed": False,
        "reason": (
            "All required Phase9 visual panel groups are present and nonblank; this is a visual audit pass, "
            "not a method-success claim."
            if visual_gate_pass
            else "Required Phase9 panel groups are incomplete or an image check failed; not a final Phase9 visual audit pass."
        ),
        "source_bridge_summary": str(args.bridge_summary),
        "source_swa_ttt_summary": str(args.swa_ttt_summary) if args.swa_ttt_summary is not None else None,
        "source_short_qk_summary": str(args.short_qk_summary) if args.short_qk_summary is not None else None,
        "ttt_visual_panel_count": len(copied_panels),
        "trajectory_plot_count": len(copied_plots),
        "short_qk_pair_visual_panel_count": len(copied_qk_panels),
        "mid_swa_qkv_visual_panel_count": len(copied_swa_panels),
        "required_panel_sets_present": required_panel_sets_present,
        "all_copied_images_nonblank": all_images_nonblank,
        "image_checks": checks,
        "aggregate": data["aggregate"],
        "swa_ttt_aggregate": swa_ttt.get("aggregate") if swa_ttt else None,
        "short_qk_aggregate": short_qk.get("aggregate") if short_qk else None,
    }
    (args.out_dir / "visual_integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [str(path.relative_to(args.out_dir)) for path in sorted(args.out_dir.rglob("*")) if path.is_file()]
    print(json.dumps({"phase9_seed_dir": str(args.out_dir), "files": files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
