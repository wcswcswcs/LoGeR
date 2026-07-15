#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_frame_ids(value: str, start: int, stride: int, count: int) -> list[int]:
    if value.strip():
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [start + i * stride for i in range(count)]


def _read_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = hw
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _read_label(path: Path, hw: tuple[int, int]) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    h, w = hw
    if label.shape[:2] != (h, w):
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.int32, copy=False)


def _read_mask(path: Path, hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    h, w = hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _put_label(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _contact_sheet(images: list[np.ndarray], cols: int = 4, pad: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("empty sheet")
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 20
    for idx, image in enumerate(images):
        y = (idx // cols) * (h + pad)
        x = (idx % cols) * (w + pad)
        canvas[y : y + h, x : x + w] = image
    return canvas


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _ratio(num: int | float, den: int | float) -> float:
    if den == 0:
        return 1.0
    return float(num) / float(den)


def _frame_overlay_tiles(
    rgb: np.ndarray,
    idx: int,
    frame_id: int,
    x0_fg: np.ndarray,
    core: np.ndarray,
    envelope: np.ndarray,
    proxy_definite: np.ndarray,
    uncertain: np.ndarray,
    exact_gap: np.ndarray,
) -> dict[str, np.ndarray]:
    exact_tracked = _put_label(_overlay(rgb, x0_fg, (0, 220, 60), 0.55), f"{idx:02d} id={frame_id} X0 tracked")

    core_envelope = rgb.copy()
    core_envelope = _overlay(core_envelope, envelope, (255, 196, 0), 0.40)
    core_envelope = _overlay(core_envelope, core, (0, 220, 60), 0.55)
    core_envelope = _put_label(core_envelope, f"{idx:02d} core/envelope")

    proxy_def = _put_label(_overlay(rgb, proxy_definite, (255, 0, 0), 0.70), f"{idx:02d} proxy definite")
    proxy_unc = _put_label(_overlay(rgb, uncertain, (255, 196, 0), 0.55), f"{idx:02d} proxy uncertain")

    hit = exact_gap & proxy_definite
    missed = exact_gap & (~proxy_definite)
    false_proxy = proxy_definite & (~exact_gap)
    compare = rgb.copy()
    compare = _overlay(compare, false_proxy, (255, 0, 0), 0.55)
    compare = _overlay(compare, missed, (0, 255, 255), 0.70)
    compare = _overlay(compare, hit, (255, 0, 255), 0.75)
    compare = _put_label(compare, f"{idx:02d} magenta=hit cyan=miss red=false")
    return {
        "exact_tracked": exact_tracked,
        "core_envelope": core_envelope,
        "proxy_definite": proxy_def,
        "proxy_uncertain": proxy_unc,
        "exact_gap_vs_proxy": compare,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rgb_root = Path(args.rgb_root)
    if not rgb_root.is_absolute():
        rgb_root = (REPO_ROOT / rgb_root).resolve()
    x0_dir = Path(args.x0_label_dir)
    x1_dir = Path(args.x1_label_dir)
    alltracker_dir = Path(args.alltracker_dir)
    if not x0_dir.is_absolute():
        x0_dir = (REPO_ROOT / x0_dir).resolve()
    if not x1_dir.is_absolute():
        x1_dir = (REPO_ROOT / x1_dir).resolve()
    if not alltracker_dir.is_absolute():
        alltracker_dir = (REPO_ROOT / alltracker_dir).resolve()

    frame_ids = _parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)
    hw = (args.canonical_height, args.canonical_width)

    records: list[dict[str, Any]] = []
    group_tiles: dict[str, list[np.ndarray]] = {
        "exact_tracked": [],
        "core_envelope": [],
        "proxy_definite": [],
        "proxy_uncertain": [],
        "exact_gap_vs_proxy": [],
    }
    group_records: list[dict[str, Any]] = []
    group_paths: list[dict[str, Any]] = []

    for idx, frame_id in enumerate(frame_ids):
        rgb = _read_rgb(rgb_root / args.scene_id / "color" / f"{frame_id}.jpg", hw)
        x0_label = _read_label(x0_dir / f"frame_{frame_id:06d}.png", hw)
        x1_label = _read_label(x1_dir / f"frame_{frame_id:06d}.png", hw)
        core = _read_mask(alltracker_dir / "coverage_masks" / f"frame_{frame_id:06d}_core.png", hw)
        envelope = _read_mask(alltracker_dir / "coverage_masks" / f"frame_{frame_id:06d}_envelope.png", hw)

        x0_fg = x0_label > 0
        x1_fg = x1_label > 0
        exact_gap = x1_fg & (~x0_fg)
        proxy_definite = x1_fg & (~envelope)
        uncertain = x1_fg & envelope & (~core)
        proxy_definite_or_uncertain = proxy_definite | uncertain
        hit = exact_gap & proxy_definite
        hit_du = exact_gap & proxy_definite_or_uncertain
        miss = exact_gap & (~proxy_definite)
        false_proxy = proxy_definite & (~exact_gap)

        tiles = _frame_overlay_tiles(
            rgb,
            idx,
            frame_id,
            x0_fg,
            core,
            envelope,
            proxy_definite,
            uncertain,
            exact_gap,
        )
        for key, tile in tiles.items():
            group_tiles[key].append(tile)

        exact_area = int(np.count_nonzero(exact_gap))
        proxy_area = int(np.count_nonzero(proxy_definite))
        x1_area = int(np.count_nonzero(x1_fg))
        records.append(
            {
                "chunk_frame_index": idx,
                "frame_id": frame_id,
                "x0_tracked_area": int(np.count_nonzero(x0_fg)),
                "x1_foreground_area": x1_area,
                "exact_gap_area": exact_area,
                "alltracker_core_area": int(np.count_nonzero(core & x1_fg)),
                "alltracker_envelope_area": int(np.count_nonzero(envelope & x1_fg)),
                "proxy_definite_gap_area": proxy_area,
                "proxy_uncertain_band_area": int(np.count_nonzero(uncertain)),
                "exact_gap_proxy_definite_hit_area": int(np.count_nonzero(hit)),
                "exact_gap_proxy_definite_or_uncertain_hit_area": int(np.count_nonzero(hit_du)),
                "exact_gap_miss_area": int(np.count_nonzero(miss)),
                "false_proxy_definite_area": int(np.count_nonzero(false_proxy)),
                "exact_gap_proxy_definite_recall": _ratio(int(np.count_nonzero(hit)), exact_area),
                "exact_gap_proxy_definite_or_uncertain_recall": _ratio(int(np.count_nonzero(hit_du)), exact_area),
                "false_proxy_definite_area_ratio_over_x1_fg": _ratio(int(np.count_nonzero(false_proxy)), x1_area),
                "proxy_definite_area_ratio_over_x1_fg": _ratio(proxy_area, x1_area),
            }
        )

        if (idx + 1) % args.group_size == 0 or idx == len(frame_ids) - 1:
            group_start = idx - (len(group_tiles["exact_tracked"]) - 1)
            group_end = idx
            group_name = f"frames_{group_start:02d}_{group_end:02d}"
            paths: dict[str, Any] = {
                "group": group_name,
                "start_chunk_frame_index": group_start,
                "end_chunk_frame_index": group_end,
            }
            for key, tiles_for_key in group_tiles.items():
                sheet = _contact_sheet(tiles_for_key, cols=args.sheet_cols)
                path = output_root / "sheets" / f"{args.scene_id}_{group_name}_{key}.jpg"
                _write_rgb(path, sheet)
                paths[f"{key}_sheet"] = _rel(path)
            group_paths.append(paths)

            rows = records[group_start : group_end + 1]
            exact_total = int(sum(row["exact_gap_area"] for row in rows))
            hit_total = int(sum(row["exact_gap_proxy_definite_hit_area"] for row in rows))
            hit_du_total = int(sum(row["exact_gap_proxy_definite_or_uncertain_hit_area"] for row in rows))
            false_total = int(sum(row["false_proxy_definite_area"] for row in rows))
            x1_total = int(sum(row["x1_foreground_area"] for row in rows))
            group_records.append(
                {
                    "group": group_name,
                    "start_chunk_frame_index": group_start,
                    "end_chunk_frame_index": group_end,
                    "frame_count": len(rows),
                    "exact_gap_area_total": exact_total,
                    "proxy_definite_hit_recall_total": _ratio(hit_total, exact_total),
                    "proxy_definite_or_uncertain_hit_recall_total": _ratio(hit_du_total, exact_total),
                    "false_proxy_definite_area_ratio_over_x1_fg_total": _ratio(false_total, x1_total),
                    "min_frame_proxy_definite_recall": min(row["exact_gap_proxy_definite_recall"] for row in rows),
                    "min_frame_proxy_definite_or_uncertain_recall": min(
                        row["exact_gap_proxy_definite_or_uncertain_recall"] for row in rows
                    ),
                }
            )
            group_tiles = {key: [] for key in group_tiles}

    records_path = output_root / "phase4_proxy_gap_frame_records.json"
    groups_path = output_root / "phase4_proxy_gap_group_records.json"
    sheets_path = output_root / "phase4_proxy_gap_sheet_records.json"
    _write_json(records_path, {"schema_version": "stream4d_v105_phase4_proxy_gap_frame_records_v1", "row_count": len(records), "rows": records})
    _write_json(groups_path, {"schema_version": "stream4d_v105_phase4_proxy_gap_group_records_v1", "row_count": len(group_records), "rows": group_records})
    _write_json(sheets_path, {"schema_version": "stream4d_v105_phase4_proxy_gap_sheet_records_v1", "row_count": len(group_paths), "rows": group_paths})

    exact_total = int(sum(row["exact_gap_area"] for row in records))
    hit_total = int(sum(row["exact_gap_proxy_definite_hit_area"] for row in records))
    hit_du_total = int(sum(row["exact_gap_proxy_definite_or_uncertain_hit_area"] for row in records))
    false_total = int(sum(row["false_proxy_definite_area"] for row in records))
    x1_total = int(sum(row["x1_foreground_area"] for row in records))
    summary = {
        "schema_version": "stream4d_v105_phase4_proxy_gap_summary_v1",
        "scene_id": args.scene_id,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "diagnostic_only": True,
        "prediction_modified": False,
        "exact_gap_definition": "x1_foreground_minus_x0_foreground_at_canonical_240x320",
        "proxy_definite_definition": "x1_foreground_outside_alltracker_envelope",
        "proxy_uncertain_definition": "x1_foreground_inside_envelope_outside_core",
        "x0_label_dir": _rel(x0_dir),
        "x1_label_dir": _rel(x1_dir),
        "alltracker_dir": _rel(alltracker_dir),
        "alltracker_summary_sha256": _sha256_file(alltracker_dir / "alltracker_contract_summary.json"),
        "records_json": _rel(records_path),
        "groups_json": _rel(groups_path),
        "sheets_json": _rel(sheets_path),
        "sheet_group_count": len(group_paths),
        "exact_gap_area_total": exact_total,
        "proxy_definite_hit_recall_total": _ratio(hit_total, exact_total),
        "proxy_definite_or_uncertain_hit_recall_total": _ratio(hit_du_total, exact_total),
        "false_proxy_definite_area_ratio_over_x1_fg_total": _ratio(false_total, x1_total),
        "min_group_proxy_definite_recall": min(row["proxy_definite_hit_recall_total"] for row in group_records),
        "min_group_proxy_definite_or_uncertain_recall": min(
            row["proxy_definite_or_uncertain_hit_recall_total"] for row in group_records
        ),
        "visual_review_required": True,
        "contract_artifacts_complete": bool(len(group_paths) and len(group_records) and len(records) == len(frame_ids)),
        "notes": [
            "This is a post-hoc diagnostic against frozen X0/X1 labels and does not alter predictions.",
            "False proxy definite area is expected to be large for new-view floor/wall regions; visual review decides whether obvious exact gaps are recalled.",
            "No threshold tuning is performed from these exact-gap diagnostics.",
        ],
    }
    _write_json(output_root / "phase4_proxy_gap_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--x0-label-dir", required=True)
    parser.add_argument("--x1-label-dir", required=True)
    parser.add_argument("--alltracker-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--canonical-height", type=int, default=240)
    parser.add_argument("--canonical-width", type=int, default=320)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sheet-cols", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "scene_id": summary["scene_id"],
                "contract_artifacts_complete": summary["contract_artifacts_complete"],
                "exact_gap_area_total": summary["exact_gap_area_total"],
                "proxy_definite_hit_recall_total": summary["proxy_definite_hit_recall_total"],
                "proxy_definite_or_uncertain_hit_recall_total": summary[
                    "proxy_definite_or_uncertain_hit_recall_total"
                ],
                "false_proxy_definite_area_ratio_over_x1_fg_total": summary[
                    "false_proxy_definite_area_ratio_over_x1_fg_total"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
