#!/usr/bin/env python3
"""Generate v94 Phase9 RGB + metric visual audit panels.

The panels use real KITTI RGB frames and tabular Phase1/4/5 evidence. They do
not fabricate semantic masks, object topology overlays, or raw overlap points
when those artifacts are not present.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
KITTI_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_OUT = ROOT / "phase9_visual_audit_or_blocked/rgb_metric_visual_audit"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}g}"


def seq_text(value: Any) -> str:
    try:
        return f"{int(float(value)):02d}"
    except (TypeError, ValueError):
        return str(value).zfill(2)


def frame_path(seq: str, frame: int) -> Path:
    return KITTI_ROOT / seq / "image_2" / f"{int(frame):06d}.png"


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        img = Image.new("RGB", size, (40, 40, 40))
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), f"missing\n{path}", fill=(255, 120, 120))
        return img
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    x, y = xy
    fnt = font(22)
    bbox = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=fnt)


def panel_for_row(row: dict[str, str], sem: dict[str, str], out_path: Path) -> dict[str, Any]:
    seq = seq_text(row.get("seq"))
    pair_id = str(row.get("pair_id"))
    frame_start = int(float(sem.get("frame_start") or 0))
    frame_end = int(float(sem.get("frame_end") or frame_start))
    prev_path = frame_path(seq, frame_start)
    curr_path = frame_path(seq, frame_end)

    canvas = Image.new("RGB", (1500, 900), (245, 245, 240))
    draw = ImageDraw.Draw(canvas)
    title_font = font(26)
    body_font = font(20)
    small_font = font(17)

    draw.text((24, 18), f"{pair_id} seq{seq} frames {frame_start}->{frame_end}", fill=(20, 20, 20), font=title_font)
    label = str(row.get("case_label_offline_only"))
    label_color = (190, 40, 40) if label == "bad" else (30, 120, 70) if label == "good" else (80, 80, 80)
    draw_label(draw, (1160, 20), f"{label}", label_color)

    image_size = (720, 218)
    prev = load_rgb(prev_path, image_size)
    curr = load_rgb(curr_path, image_size)
    canvas.paste(prev, (24, 66))
    canvas.paste(curr, (756, 66))
    draw.text((24, 292), f"previous RGB: {prev_path}", fill=(40, 40, 40), font=small_font)
    draw.text((756, 292), f"current RGB: {curr_path}", fill=(40, 40, 40), font=small_font)

    role = str(row.get("semantic_evidence_type"))
    best_role_positive = role == "SEM_INVALID_BOUNDARY"
    broad_policy_positive = role in {"SEM_INVALID_BOUNDARY", "SEM_WEAK_CONTEXT"}
    sections = [
        ("Semantic evidence", [
            f"type={role}; source_level={row.get('semantic_source_level')}",
            f"S_invalid={fmt(row.get('S_invalid'))}; S_context={fmt(row.get('S_context'))}; S_multi={fmt(row.get('S_multi'))}; S_lowobs={fmt(row.get('S_lowobs'))}",
            f"semantic_shuffle={row.get('semantic_shuffle_category')}; component_shuffle={row.get('component_shuffle_category')}; regime_shuffle={row.get('regime_shuffle_category')}",
            f"best_role_positive(SEM_INVALID_BOUNDARY)={best_role_positive}; broad_invalid_or_weak={broad_policy_positive}",
        ]),
        ("Carrier / runtime proxy", [
            f"boundary_update_norm={fmt(row.get('carrier_error_boundary_update_norm'))}; merge_residual={fmt(row.get('carrier_error_merge_residual_after_abs'))}; abs_log_scale_jump={fmt(row.get('carrier_error_abs_log_scale_jump_runtime'))}",
            f"I_J_runtime_proxy={fmt(row.get('I_J_runtime_proxy'))}; native J={fmt(row.get('J_handoff_runtime_proxy_native'))}; probe J={fmt(row.get('J_handoff_runtime_proxy_probe'))}",
            f"variant={row.get('variant')}; probe_root={Path(str(row.get('probe_root'))).name}",
        ]),
        ("Phase1 / local signals", [
            f"failure={sem.get('failure_type_primary')}; secondary={sem.get('failure_type_secondary')}; assignment={sem.get('assignment_reason')}",
            f"raw_overlap_residual={fmt(sem.get('raw_overlap_residual'))}; raw_overlap_inlier_count={sem.get('raw_overlap_inlier_count')}; observability_score={fmt(sem.get('observability_score'))}",
            f"local_shape_mode_entropy={fmt(sem.get('local_shape_mode_entropy'))}; component_boundary_ratio={fmt(sem.get('component_boundary_ratio'))}; cross_component_ratio={fmt(sem.get('cross_component_ratio'))}",
        ]),
        ("Visual artifact honesty", [
            "real RGB frames: yes",
            "semantic/component/object overlay image: unavailable in v94 artifacts; tabular fields shown instead",
            "raw overlap point overlay: unavailable in v94 artifacts; residual/count fields shown instead",
            "counterfactual/runtime/TTT visual: blocked by Phase5; not fabricated",
        ]),
    ]

    y = 340
    for heading, lines in sections:
        draw.text((28, y), heading, fill=(15, 55, 95), font=body_font)
        y += 30
        for line in lines:
            for wrapped in textwrap.wrap(line, width=118):
                draw.text((46, y), wrapped, fill=(30, 30, 30), font=small_font)
                y += 24
        y += 12

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "pair_id": pair_id,
        "seq": seq,
        "case_label_offline_only": label,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "previous_rgb": str(prev_path),
        "current_rgb": str(curr_path),
        "panel_path": str(out_path),
        "panel_exists": out_path.exists(),
        "panel_non_empty": out_path.exists() and out_path.stat().st_size > 0,
        "semantic_evidence_type": role,
        "best_role_positive": best_role_positive,
        "broad_invalid_or_weak_positive": broad_policy_positive,
        "real_rgb_available": prev_path.exists() and curr_path.exists(),
        "semantic_component_overlay_available": False,
        "raw_overlap_point_overlay_available": False,
        "raw_overlap_summary_available": bool(sem.get("raw_overlap_residual") or sem.get("raw_overlap_inlier_count")),
        "no_fake_runtime_panel": True,
    }


def build_contact_sheet(manifest: list[dict[str, Any]], out_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for row in manifest:
        path = Path(str(row["panel_path"]))
        if path.exists():
            thumbs.append(Image.open(path).convert("RGB").resize((375, 225), Image.Resampling.BILINEAR))
    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 375, rows * 225), (245, 245, 240))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 375
        y = (idx // cols) * 225
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-unlabelled", action="store_true")
    args = parser.parse_args()

    phase5_rows = read_csv_rows(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    phase4_rows = read_csv_rows(args.root / "phase4_semantic_evidence_taxonomy/semantic_evidence_rows.csv")
    sem_by_pair = {str(row["pair_id"]): row for row in phase4_rows}
    rows = [
        row for row in phase5_rows
        if args.include_unlabelled or row.get("case_label_offline_only") in {"bad", "good"}
    ]
    rows.sort(key=lambda row: (row.get("case_label_offline_only") != "bad", row.get("seq", ""), row.get("pair_id", "")))

    manifest: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        out_path = args.out_dir / "panels" / f"{pair_id}_{row.get('case_label_offline_only')}.png"
        manifest.append(panel_for_row(row, sem_by_pair.get(pair_id, {}), out_path))

    contact_sheet = args.out_dir / "rgb_metric_contact_sheet.png"
    build_contact_sheet(manifest, contact_sheet)
    write_csv(args.out_dir / "rgb_metric_visual_manifest.csv", manifest)

    labelled_count = len([row for row in phase5_rows if row.get("case_label_offline_only") in {"bad", "good"}])
    summary = {
        "phase": "Phase9_rgb_metric_visual_audit",
        "visual_audit_produced": True,
        "visual_gate_pass": False,
        "visual_gate_blocker": "semantic_component_object_overlay_images_unavailable;raw_overlap_point_overlay_unavailable",
        "reviewed_rows": len(manifest),
        "labelled_carrier_rows": labelled_count,
        "review_coverage": len(manifest) / labelled_count if labelled_count else 0.0,
        "all_panels_exist": all(row["panel_exists"] for row in manifest),
        "all_panels_non_empty": all(row["panel_non_empty"] for row in manifest),
        "all_rgb_available": all(row["real_rgb_available"] for row in manifest),
        "semantic_component_overlay_available": False,
        "raw_overlap_point_overlay_available": False,
        "raw_overlap_summary_available_count": sum(bool(row["raw_overlap_summary_available"]) for row in manifest),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_exists": contact_sheet.exists(),
        "contact_sheet_non_empty": contact_sheet.exists() and contact_sheet.stat().st_size > 0,
        "no_fake_runtime_panels": all(row["no_fake_runtime_panel"] for row in manifest),
        "note": "Panels use real KITTI RGB frames plus tabular Phase1/4/5 evidence; unavailable overlays are marked unavailable.",
    }
    write_json(args.out_dir / "rgb_metric_visual_audit_summary.json", summary)
    print(f"visual_audit_produced={summary['visual_audit_produced']}")
    print(f"visual_gate_pass={summary['visual_gate_pass']}")
    print(f"reviewed_rows={summary['reviewed_rows']}")
    print(f"review_coverage={summary['review_coverage']}")
    print(f"all_panels_non_empty={summary['all_panels_non_empty']}")
    print(f"all_rgb_available={summary['all_rgb_available']}")
    print(f"visual_gate_blocker={summary['visual_gate_blocker']}")


if __name__ == "__main__":
    main()
