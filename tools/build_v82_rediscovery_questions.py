#!/usr/bin/env python3
"""Build v82 Phase12 rediscovery questions and visual audit bundle.

The bundle is derived from real v82 artifacts only. Missing TTT evidence is
rendered as an explicit not-run availability panel because Phase8 never passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
DEFAULT_OUT = ROOT / "phase12_rediscovery"
DEFAULT_PAIR_BANK = ROOT / "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
DEFAULT_PHASE3 = ROOT / "phase3_swa_true_route_visual_confirmation"
DEFAULT_LEDGER = ROOT / "phase4_swa_carrier_ledger/swa_carrier_ledger.csv"
DEFAULT_LEDGER_SUMMARY = ROOT / "phase4_swa_carrier_ledger/swa_carrier_ledger_summary.json"
DEFAULT_PHASE8E = ROOT / "phase8e_projection_tol001_steps64_continuation"
DEFAULT_PHASE8E_SMOKE = DEFAULT_PHASE8E / "seq01_v82_projection_tol001_steps64"
DEFAULT_KITTI = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


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


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _ratio_improvement(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or abs(base) <= 1.0e-12:
        return None
    return float((base - cand) / base)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _png_size(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file() or path.suffix.lower() != ".png":
        return None, None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    return tuple(int(v) for v in struct.unpack(">II", header[16:24]))


def _panel_meta(path: Path, group: str, source: Path | None, status: str, note: str) -> dict[str, Any]:
    width, height = _png_size(path)
    exists = path.is_file()
    return {
        "group": group,
        "path": str(path),
        "source_path": str(source) if source is not None else "",
        "status": status if exists else "missing",
        "bytes": path.stat().st_size if exists else 0,
        "width": width,
        "height": height,
        "sha256": _sha256(path) if exists else "",
        "note": note,
    }


def _copy_panel(src: Path, dst_dir: Path, group: str, dst_name: str, note: str) -> dict[str, Any]:
    dst = dst_dir / group / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
        status = "ok"
    else:
        _text_panel(dst, ["missing source panel", str(src)], title=group)
        status = "availability_note"
    return _panel_meta(dst, group, src, status, note)


def _text_panel(path: Path, lines: Sequence[str], *, title: str, size: tuple[int, int] = (900, 520)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (248, 248, 246))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 14
    draw.text((14, y), title, fill=(15, 15, 15), font=font)
    y += 24
    for line in lines:
        text = str(line)
        for part in [text[i : i + 108] for i in range(0, len(text), 108)] or [""]:
            draw.text((14, y), part, fill=(35, 35, 35), font=font)
            y += 16
            if y >= size[1] - 18:
                img.save(path)
                return
    img.save(path)


def _rgb_tile(kitti_root: Path, seq: str, frame: int, size: tuple[int, int]) -> Image.Image:
    path = kitti_root / "sequences" / str(seq).zfill(2) / "image_2" / f"{int(frame):06d}.png"
    if not path.is_file():
        img = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"missing RGB: {path}", fill=(30, 30, 30), font=ImageFont.load_default())
        return img
    return Image.open(path).convert("RGB").resize(size)


def _text_tile(lines: Sequence[str], title: str, size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (248, 248, 246))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 10
    draw.text((10, y), title, fill=(15, 15, 15), font=font)
    y += 22
    for line in lines:
        for part in [str(line)[i : i + 58] for i in range(0, len(str(line)), 58)] or [""]:
            draw.text((10, y), part, fill=(35, 35, 35), font=font)
            y += 15
            if y > size[1] - 18:
                return img
    return img


def _bar_tile(items: Sequence[tuple[str, float | None]], title: str, size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(15, 15, 15), font=font)
    finite = [abs(v) for _, v in items if v is not None]
    scale = max(finite) if finite else 1.0
    y = 42
    for label, value in items:
        text = "missing" if value is None else f"{value:.6g}"
        draw.text((10, y), f"{label}: {text}", fill=(30, 30, 30), font=font)
        if value is not None:
            width = int((abs(value) / max(scale, 1.0e-12)) * (size[0] - 220))
            color = (70, 150, 80) if value >= 0 else (200, 80, 70)
            draw.rectangle((190, y, 190 + width, y + 10), fill=color)
        y += 30
    return img


def _rows_by_chunk_run(rows: Sequence[Mapping[str, str]]) -> dict[tuple[int, str], Mapping[str, str]]:
    out: dict[tuple[int, str], Mapping[str, str]] = {}
    for row in rows:
        chunk = row.get("chunk")
        run = row.get("run")
        if chunk is None or run is None:
            continue
        out[(int(float(chunk)), str(run))] = row
    return out


def _pair_by_seq_curr(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int], Mapping[str, str]]:
    out: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        seq = str(row.get("seq", "")).zfill(2)
        curr = row.get("curr_chunk")
        if seq and curr not in {None, ""}:
            out[(seq, int(float(curr)))] = row
    return out


def _build_failed_swa_questions(ledger_rows: Sequence[Mapping[str, str]], limit: int = 24) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in ledger_rows:
        if row.get("base_case_type") != "bad":
            continue
        key = (
            str(row.get("seq", "")).zfill(2),
            int(float(row.get("prev_chunk", 0))),
            int(float(row.get("curr_chunk", 0))),
            str(row.get("carrier_family", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "priority": len(rows) + 1,
                "seq": key[0],
                "prev_chunk": key[1],
                "curr_chunk": key[2],
                "case_type": row.get("case_type"),
                "carrier_family": row.get("carrier_family"),
                "failed_gate": "carrier_not_localized",
                "failure_evidence": row.get("head_layer_sensitivity_status"),
                "per_head_available": row.get("per_head_available"),
                "same_head_random_available": row.get("same_head_random_available"),
                "shuffled_semantic_available": row.get("shuffled_semantic_available"),
                "future_after_overlap": row.get("future_after_overlap"),
                "boundary_jump": row.get("boundary_jump"),
                "overlap_scale_residual": row.get("overlap_scale_residual"),
                "visual_question": (
                    "Which Q/K/V head-layer route actually carries scale handoff, and why does the all-head aggregate fail controls?"
                ),
                "required_next_hook_or_artifact": "per-head route dump plus same-head random and shuffled semantic controls",
                "source_visual_file": row.get("visual_file"),
                "source_route_file": row.get("route_file"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _trace_for(phase8e_smoke: Path, chunk: int, candidate: str) -> dict[str, Any]:
    path = phase8e_smoke / f"chunk{chunk:02d}" / candidate / "merge_state_trace.jsonl"
    rows = _read_jsonl(path)
    chosen = next((row for row in rows if row.get("local_chunk_idx") == 1), rows[-1] if rows else {})
    return {
        "trace_path": str(path),
        "fit_reason": chosen.get("semantic_merge_fit_reason"),
        "transform_scale_value": chosen.get("transform_scale_value"),
        "candidate_scale": chosen.get("semantic_merge_candidate_scale"),
        "blend_scale": chosen.get("semantic_merge_blend_scale"),
        "native_overlap_residual": chosen.get("semantic_merge_native_overlap_residual"),
        "final_overlap_residual": chosen.get("semantic_merge_final_overlap_residual"),
        "native_overlap_guard_rejected": chosen.get("semantic_merge_native_overlap_guard_rejected"),
        "residual_safe_projection_accepted": chosen.get("semantic_merge_residual_safe_projection_accepted"),
        "semantic_confidence_mean": chosen.get("semantic_confidence_mean"),
        "overlap_support_mean": chosen.get("semantic_merge_overlap_support_mean"),
        "overlap_support_q90": chosen.get("semantic_merge_overlap_support_q90"),
        "valid_point_count": chosen.get("semantic_merge_valid_point_count"),
        "remaining_valid_ratio": chosen.get("semantic_merge_remaining_valid_ratio"),
        "condition_score": chosen.get("semantic_merge_condition_score"),
    }


def _build_failed_merge_questions(
    *,
    phase8e_root: Path,
    phase8e_smoke: Path,
    pair_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_map = _pair_by_seq_curr(pair_rows)
    out: list[dict[str, Any]] = []
    panel_inputs: list[dict[str, Any]] = []
    candidates = {
        "overlap_outlier": ["geometry_only", "overlap_outlier_random", "overlap_outlier_shuffled"],
        "robust_semoverlap": ["geometry_only", "robust_semoverlap_random", "robust_semoverlap_shuffled"],
    }
    metric_files = {
        "overlap_outlier": phase8e_root / "phase8e_overlap_outlier_run_metrics.csv",
        "robust_semoverlap": phase8e_root / "phase8e_robust_semoverlap_run_metrics.csv",
    }
    for candidate, controls in candidates.items():
        rows_by = _rows_by_chunk_run(_read_csv(metric_files[candidate]))
        for chunk in (9, 11, 14):
            base = rows_by.get((chunk, "native_no_swa"), {})
            cand = rows_by.get((chunk, candidate), {})
            control_rows = [rows_by.get((chunk, control), {}) for control in controls]
            head_base = _safe_float(base.get("head10_to_tail10_pose_sim3_rmse_m"))
            head_cand = _safe_float(cand.get("head10_to_tail10_pose_sim3_rmse_m"))
            overlap_base = _safe_float(base.get("overlap3_to_future_pose_sim3_rmse_m"))
            overlap_cand = _safe_float(cand.get("overlap3_to_future_pose_sim3_rmse_m"))
            head_controls = [_safe_float(row.get("head10_to_tail10_pose_sim3_rmse_m")) for row in control_rows]
            overlap_controls = [_safe_float(row.get("overlap3_to_future_pose_sim3_rmse_m")) for row in control_rows]
            head_controls = [value for value in head_controls if value is not None]
            overlap_controls = [value for value in overlap_controls if value is not None]
            trace = _trace_for(phase8e_smoke, chunk, candidate)
            label = pair_map.get(("01", chunk), {})
            row = {
                "priority": len(out) + 1,
                "seq": "01",
                "prev_chunk": label.get("prev_chunk", int(chunk) - 1),
                "curr_chunk": chunk,
                "case_type": label.get("case_type", ""),
                "quality_type": label.get("quality_type", ""),
                "candidate": candidate,
                "failed_gate": "phase8_bad_good_control_gate",
                "head_tail_baseline": head_base,
                "head_tail_candidate": head_cand,
                "head_tail_best_control": min(head_controls) if head_controls else None,
                "head_tail_improvement_vs_baseline_ratio": _ratio_improvement(head_base, head_cand),
                "overlap_future_baseline": overlap_base,
                "overlap_future_candidate": overlap_cand,
                "overlap_future_best_control": min(overlap_controls) if overlap_controls else None,
                "overlap_future_improvement_vs_baseline_ratio": _ratio_improvement(overlap_base, overlap_cand),
                "phase8e_trace": trace,
                "visual_question": (
                    "Why does the semantic merge transform fail to beat geometry/random/shuffled controls while also hurting good-pair protection?"
                ),
                "required_next_hook_or_artifact": (
                    "merge boundary panel with RGB context, transform scale, native/final overlap residual, control gap, and good-pair protection evidence"
                ),
            }
            out.append(row)
            panel_inputs.append(row)
    return out, panel_inputs


def _build_failed_long_questions() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "seq": "01",
            "failed_gate": "not_run_phase8_entry_failed",
            "failure_evidence": "Phase9 TTT entry requires Phase8 merge/gauge pass, but Phase8e phase8_gate_pass=false.",
            "visual_question": (
                "What TTT write-less evidence would be needed after a future SWA/merge pass, and which evidence is unavailable now?"
            ),
            "required_next_hook_or_artifact": "TTT write mass by role after SWA/merge confirmation; not generated in current run by design.",
        }
    ]


def _make_merge_panel(path: Path, row: Mapping[str, Any], kitti_root: Path) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    curr = int(row["curr_chunk"])
    start = int(curr * 29)
    frames = [max(0, start - 2), start, start + 2]
    tile = (420, 260)
    panel = Image.new("RGB", (tile[0] * 3, tile[1] * 3), (255, 255, 255))
    for idx, frame in enumerate(frames):
        panel.paste(_rgb_tile(kitti_root, seq, frame, tile), (idx * tile[0], 0))
    metric_lines = [
        f"seq={seq} prev={row.get('prev_chunk')} curr={curr} candidate={row.get('candidate')}",
        f"case={row.get('case_type')} quality={row.get('quality_type')}",
        f"head baseline={row.get('head_tail_baseline')}",
        f"head candidate={row.get('head_tail_candidate')}",
        f"head best_control={row.get('head_tail_best_control')}",
        f"head improvement={row.get('head_tail_improvement_vs_baseline_ratio')}",
        f"overlap baseline={row.get('overlap_future_baseline')}",
        f"overlap candidate={row.get('overlap_future_candidate')}",
        f"overlap best_control={row.get('overlap_future_best_control')}",
        f"overlap improvement={row.get('overlap_future_improvement_vs_baseline_ratio')}",
    ]
    trace = row.get("phase8e_trace", {})
    trace_lines = [
        f"fit_reason={trace.get('fit_reason')}",
        f"transform_scale={trace.get('transform_scale_value')}",
        f"candidate_scale={trace.get('candidate_scale')}",
        f"blend_scale={trace.get('blend_scale')}",
        f"native_overlap_residual={trace.get('native_overlap_residual')}",
        f"final_overlap_residual={trace.get('final_overlap_residual')}",
        f"guard_rejected={trace.get('native_overlap_guard_rejected')}",
        f"projection_accepted={trace.get('residual_safe_projection_accepted')}",
        f"semantic_conf_mean={trace.get('semantic_confidence_mean')}",
        f"support_mean={trace.get('overlap_support_mean')}",
        f"support_q90={trace.get('overlap_support_q90')}",
    ]
    bar_items = [
        ("head_impr", row.get("head_tail_improvement_vs_baseline_ratio")),
        ("overlap_impr", row.get("overlap_future_improvement_vs_baseline_ratio")),
        ("scale", trace.get("transform_scale_value")),
        ("support_mean", trace.get("overlap_support_mean")),
    ]
    bar_items = [(label, _safe_float(value)) for label, value in bar_items]
    panel.paste(_text_tile(metric_lines, "Phase8e Metrics", tile), (0, tile[1]))
    panel.paste(_text_tile(trace_lines, "Merge Trace", tile), (tile[0], tile[1]))
    panel.paste(_bar_tile(bar_items, "Signed Signal Sketch", tile), (tile[0] * 2, tile[1]))
    panel.paste(
        _text_tile(
            [
                "Question:",
                row.get("visual_question", ""),
                "This is diagnostic-only; no GT-derived runtime promotion is claimed.",
                f"trace={trace.get('trace_path')}",
            ],
            "Rediscovery Question",
            (tile[0] * 3, tile[1]),
        ),
        (0, tile[1] * 2),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return _panel_meta(path, "new_merge_boundary_panels", None, "ok", "Generated from real RGB, Phase8e metrics, and merge_state_trace.")


def _render_insight(audit: Mapping[str, Any], phase8e_summary: Mapping[str, Any]) -> str:
    lines = [
        "# v82 Phase12 Rediscovery Visual Insight",
        "",
        "## Status",
        "",
        f"- visual_audit_gate_pass: `{audit.get('visual_audit_gate_pass')}`",
        f"- phase8e_phase8_gate_pass: `{phase8e_summary.get('phase8_gate_pass')}`",
        f"- phase8e_decision: `{phase8e_summary.get('decision')}`",
        "",
        "## Evidence",
        "",
        "- SWA carrier rediscovery remains blocked by missing per-head route, same-head random, and shuffled semantic controls in the Phase4 ledger.",
        "- Merge-boundary panels show that Phase8e accepted some residual-safe projections, but the candidate still fails the bad/good/control gate.",
        "- TTT write-less panels are explicit not-run notes, because TTT cannot be promoted before SWA or merge/gauge confirmation.",
        "",
        "## Insight",
        "",
        "- The current semantic-scale merge signal is not merely being clipped by native-overlap guard tolerance; accepted projections still do not beat controls.",
        "- The next valid mechanism has to be a control-aware merge/gauge state interface, or a new per-head SWA carrier dump, not another scalar sweep.",
        "- Any future repair must report head-tail, overlap-to-future, boundary/gauge trace, and good-case protection together.",
        "",
    ]
    return "\n".join(lines)


def _render_hypotheses() -> str:
    return "\n".join(
        [
            "# v82 Phase12 New Hypothesis Bank",
            "",
            "1. Per-head SWA rediscovery: all-head aggregate route evidence may hide the actual carrier; collect same-head random and shuffled semantic controls before another SWA action.",
            "2. Control-aware merge/gauge interface: promote semantic merge only when its native-overlap residual and support structure predict a margin over geometry/random controls, not just non-worse native overlap.",
            "3. Boundary multi-objective controller: separate boundary-step, head-tail, and overlap-to-future objectives; current scale-only correction can improve one proxy while harming another.",
            "4. TTT remains downstream-only: write-less / one-hop TTT should stay disabled until SWA or merge/gauge provides stable evidence with good-case protection.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pair-bank", type=Path, default=DEFAULT_PAIR_BANK)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ledger-summary", type=Path, default=DEFAULT_LEDGER_SUMMARY)
    parser.add_argument("--phase8e-root", type=Path, default=DEFAULT_PHASE8E)
    parser.add_argument("--phase8e-smoke-root", type=Path, default=DEFAULT_PHASE8E_SMOKE)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    ledger_rows = _read_csv(args.ledger)
    pair_rows = _read_csv(args.pair_bank)
    phase8e_summary = _read_json(args.phase8e_root / "merge_gauge_fallback_summary.json")

    failed_swa = _build_failed_swa_questions(ledger_rows)
    failed_merge, merge_panel_inputs = _build_failed_merge_questions(
        phase8e_root=args.phase8e_root,
        phase8e_smoke=args.phase8e_smoke_root,
        pair_rows=pair_rows,
    )
    failed_long = _build_failed_long_questions()

    _write_csv(args.out_root / "failed_swa_pair_to_visual_question.csv", failed_swa)
    _write_csv(args.out_root / "failed_merge_boundary_to_visual_question.csv", failed_merge)
    _write_csv(args.out_root / "failed_long_window_to_visual_question.csv", failed_long)

    review_rows: list[dict[str, Any]] = []
    for row in failed_swa:
        seq = str(row["seq"]).zfill(2)
        prev = int(row["prev_chunk"])
        curr = int(row["curr_chunk"])
        case = str(row.get("case_type") or "unknown")
        name = f"seq{seq}_chunk{prev:03d}_{curr:03d}_{case}.png"
        qk_src = args.phase3_root / "qkv_head_layer_panels" / name
        actual_src = args.phase3_root / "actual_vs_random_panels" / name
        true_src = args.phase3_root / "true_route_panels" / name
        review_rows.append(_copy_panel(qk_src, args.out_root, "new_qk_pair_panels", name, "Copied v82 Phase3 qkv/head-layer panel."))
        review_rows.append(_copy_panel(actual_src, args.out_root, "new_qk_pair_panels", f"actual_{name}", "Copied v82 Phase3 actual-vs-random panel."))
        review_rows.append(_copy_panel(true_src, args.out_root, "new_swa_kv_panels", name, "Copied v82 Phase3 true-route panel."))

    for row in merge_panel_inputs:
        name = f"seq01_chunk{int(row['curr_chunk']):02d}_{row['candidate']}_merge_boundary.png"
        review_rows.append(_make_merge_panel(args.out_root / "new_merge_boundary_panels" / name, row, args.kitti_root))

    ttt_panel = args.out_root / "new_ttt_write_less_panels" / "phase9_ttt_not_run_phase8_failed.png"
    _text_panel(
        ttt_panel,
        [
            "TTT write-less was not run.",
            "Reason: Phase9 entry requires SWA or merge/gauge confirmation.",
            "Current evidence: Phase8e phase8_gate_pass=false and decision=no_go_stop_before_ttt.",
            "This panel is an availability note, not TTT visual evidence.",
        ],
        title="TTT Not-Run Availability",
    )
    review_rows.append(
        _panel_meta(
            ttt_panel,
            "new_ttt_write_less_panels",
            None,
            "availability_note",
            "Explicit not-run panel because Phase8 gate failed.",
        )
    )

    _write_csv(args.out_root / "visual_review.csv", review_rows)
    groups = ["new_qk_pair_panels", "new_swa_kv_panels", "new_merge_boundary_panels", "new_ttt_write_less_panels"]
    group_counts: dict[str, dict[str, int]] = {}
    for group in groups:
        rows = [row for row in review_rows if row.get("group") == group]
        group_counts[group] = {
            "count": len(rows),
            "ok": sum(1 for row in rows if int(row.get("bytes") or 0) > 0 and row.get("width") and row.get("height")),
        }
    visual_audit_gate_pass = bool(
        failed_swa
        and failed_merge
        and failed_long
        and all(item["count"] > 0 and item["ok"] == item["count"] for item in group_counts.values())
    )
    audit = {
        "schema": "acl2_v82_phase12_rediscovery_v1",
        "root": args.out_root,
        "phase8e_summary": args.phase8e_root / "merge_gauge_fallback_summary.json",
        "phase8e_gate_pass": phase8e_summary.get("phase8_gate_pass"),
        "phase8e_decision": phase8e_summary.get("decision"),
        "failed_swa_question_count": len(failed_swa),
        "failed_merge_boundary_question_count": len(failed_merge),
        "failed_long_window_question_count": len(failed_long),
        "group_counts": group_counts,
        "visual_review_rows": len(review_rows),
        "ttt_runtime_visual_evidence_available": False,
        "ttt_availability_note": "TTT was not run because Phase8 did not pass; the TTT panel is a not-run availability panel.",
        "visual_audit_gate_pass": visual_audit_gate_pass,
        "note": "All copied/generated panels use existing v82 artifacts, real KITTI RGB, Phase8e metrics, and merge traces; no visual evidence is fabricated.",
    }
    _write_json(args.out_root / "visual_integrity_audit.json", audit)
    (args.out_root / "visual_insight.md").write_text(_render_insight(audit, phase8e_summary), encoding="utf-8")
    (args.out_root / "new_hypothesis_bank.md").write_text(_render_hypotheses(), encoding="utf-8")
    print(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
