from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png, _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _float_or_none, _load_csv_rows, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import _discover_pipeline_root, _mask_dir_from_pipeline  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in row:
            value = _float_or_none(row.get(name))
            if value is not None:
                return value
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value in np.unique(labels):
        value = int(value)
        if value <= 0:
            continue
        colors[labels == value] = np.asarray(
            [
                (value * 37 + 59) % 255,
                (value * 67 + 101) % 255,
                (value * 97 + 149) % 255,
            ],
            dtype=np.uint8,
        )
    return colors


def _overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.copy()
    colors = _label_colors(labels)
    mask = labels > 0
    out[mask] = ((1.0 - alpha) * out[mask].astype(np.float32) + alpha * colors[mask].astype(np.float32)).astype(np.uint8)
    return out


def _panel_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, title[:88], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _resize_rgb(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape_hw:
        return image
    return cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def _load_mask(mask_dir: Path, frame_id: int, shape_hw: tuple[int, int]) -> np.ndarray:
    path = mask_dir / f"{int(frame_id)}.png"
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.int64)
    return _read_label_png(path, shape_hw)


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str, int, int], tuple[str, str, int, int]] = {}

    def add(self, item: tuple[str, str, int, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[str, str, int, int]) -> tuple[str, str, int, int]:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: tuple[str, str, int, int], right: tuple[str, str, int, int]) -> None:
        l_root = self.find(left)
        r_root = self.find(right)
        if l_root != r_root:
            self.parent[r_root] = l_root


def _build_method_mapping(
    *,
    capsule_edge_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    variant: str,
) -> dict[tuple[str, str, int, int], int]:
    dsu = _DSU()
    for row in candidate_rows:
        if _bool_text(row.get("uses_gt_for_prediction")) or _bool_text(row.get("forbidden_for_method_table")):
            continue
        scene = str(row.get("scene_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        frame_id = _int_or_none(row.get("frame_id"))
        mask_id = _int_or_none(row.get("mask_id"))
        if not scene or not chunk_id or frame_id is None or mask_id is None:
            continue
        dsu.add((scene, chunk_id, frame_id, mask_id))
    for row in capsule_edge_rows:
        row_variant = str(row.get("capsule_variant") or row.get("variant") or "")
        if row_variant != variant:
            continue
        if _bool_text(row.get("uses_gt_for_prediction")) or _bool_text(row.get("forbidden_for_method_table")):
            continue
        scene = str(row.get("scene_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        lf = _int_or_none(row.get("left_frame"))
        lm = _int_or_none(row.get("left_mask"))
        rf = _int_or_none(row.get("right_frame"))
        rm = _int_or_none(row.get("right_mask"))
        if not scene or not chunk_id or None in {lf, lm, rf, rm}:
            continue
        dsu.union((scene, chunk_id, int(lf), int(lm)), (scene, chunk_id, int(rf), int(rm)))

    roots_by_chunk: dict[tuple[str, str], dict[tuple[str, str, int, int], int]] = defaultdict(dict)
    mapping: dict[tuple[str, str, int, int], int] = {}
    for item in sorted(dsu.parent):
        scene, chunk_id, frame_id, mask_id = item
        root = dsu.find(item)
        chunk_roots = roots_by_chunk[(scene, chunk_id)]
        if root not in chunk_roots:
            chunk_roots[root] = len(chunk_roots) + 1
        mapping[(scene, chunk_id, frame_id, mask_id)] = chunk_roots[root]
    return mapping


def _candidate_ids_by_frame(candidate_rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], set[int]]:
    out: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for row in candidate_rows:
        if _bool_text(row.get("uses_gt_for_prediction")) or _bool_text(row.get("forbidden_for_method_table")):
            continue
        scene = str(row.get("scene_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        frame_id = _int_or_none(row.get("frame_id"))
        mask_id = _int_or_none(row.get("mask_id"))
        if scene and chunk_id and frame_id is not None and mask_id is not None:
            out[(scene, chunk_id, frame_id)].add(mask_id)
    return out


def _resolve_pipeline_root(scene: str, artifact_source: Any) -> Path | None:
    text = str(artifact_source or "")
    if text:
        candidate = _rooted(text)
        if (candidate / "pipeline_summary.json").exists():
            return candidate
    return _discover_pipeline_root(scene)


def _labels_from_mapping(mask: np.ndarray, mapping: dict[tuple[str, str, int, int], int], scene: str, chunk_id: str, frame_id: int) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int64)
    for mask_id in np.unique(mask):
        mask_id = int(mask_id)
        if mask_id <= 0:
            continue
        object_id = int(mapping.get((scene, chunk_id, frame_id, mask_id), 0))
        if object_id > 0:
            labels[mask == mask_id] = object_id
    return labels


def _labels_from_candidates(mask: np.ndarray, candidate_ids: set[int]) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int64)
    if not candidate_ids:
        candidate_ids = {int(mask_id) for mask_id in np.unique(mask) if int(mask_id) > 0}
    for idx, mask_id in enumerate(sorted(candidate_ids), start=1):
        labels[mask == int(mask_id)] = idx
    return labels


def _make_case_image(
    *,
    case: dict[str, Any],
    method_mapping: dict[tuple[str, str, int, int], int],
    candidate_by_frame: dict[tuple[str, str, int], set[int]],
    output_path: Path,
) -> dict[str, Any]:
    scene = str(case.get("scene_id") or "")
    chunk_id = str(case.get("chunk_id") or "")
    frame_id = _int_or_none(case.get("frame_id")) or _int_or_none(case.get("frame_min")) or 0
    pipeline_root = _resolve_pipeline_root(scene, case.get("artifact_source"))
    if not scene or not chunk_id or chunk_id == "summary" or pipeline_root is None:
        return {"rendered": False, "reason": "missing_scene_chunk_or_pipeline_root"}
    try:
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        shape_hw = tuple(int(value) for value in stream.load_depth(frame_id).shape)
        rgb = _resize_rgb(stream.load_rgb(frame_id), shape_hw)
        raw_mask = _load_mask(mask_dir, frame_id, shape_hw)
    except Exception as exc:
        return {"rendered": False, "reason": f"input_load_failed:{type(exc).__name__}:{exc}"}

    method_labels = _labels_from_mapping(raw_mask, method_mapping, scene, chunk_id, frame_id)
    candidate_labels = _labels_from_candidates(raw_mask, candidate_by_frame.get((scene, chunk_id, frame_id), set()))
    try:
        gt = _load_gt_2d(scene, frame_id, shape_hw)
        gt_available = bool(np.any(gt > 0))
    except Exception:
        gt = np.zeros(shape_hw, dtype=np.int64)
        gt_available = False

    metric_title = f"{case.get('phase')} {case.get('failure_type')} SF50={case.get('local_SF50')}"
    panels = [
        _panel_title(rgb, f"RGB {scene} frame {frame_id}"),
        _panel_title(_overlay(rgb, method_labels), "v70 method objects, GT-free"),
        _panel_title(_overlay(rgb, candidate_labels), "candidate mask layer, GT-free"),
        _panel_title(_overlay(rgb, gt), "GT instance diagnostic only"),
    ]
    thumb_h = 240
    resized = []
    for panel in panels:
        scale = thumb_h / panel.shape[0]
        resized.append(cv2.resize(panel, (int(panel.shape[1] * scale), thumb_h), interpolation=cv2.INTER_AREA))
    canvas = np.concatenate(resized, axis=1)
    footer_h = 36
    footer = np.zeros((footer_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, metric_title[:150], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.concatenate([canvas, footer], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return {
        "rendered": bool(ok),
        "reason": "" if ok else "cv2_imwrite_failed",
        "frame_id": int(frame_id),
        "pipeline_root": _rel(pipeline_root),
        "mask_dir": _rel(mask_dir),
        "method_pixel_count": int(np.count_nonzero(method_labels)),
        "candidate_pixel_count": int(np.count_nonzero(candidate_labels)),
        "gt_pixel_count": int(np.count_nonzero(gt)),
        "gt_available": gt_available,
    }


def _write_html_casebook(output_root: Path, case_rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    html_path = output_root / "v70_casebook.html"
    cards = []
    for row in case_rows:
        screenshot = str(row.get("screenshot_path") or "")
        if not screenshot:
            continue
        src = Path(screenshot)
        if src.parts and src.parts[0] == "outputs":
            src = ROOT / src
        try:
            image_src = str(src.relative_to(output_root))
        except ValueError:
            image_src = screenshot
        title = html.escape(str(row.get("case_id") or "case"))
        meta = html.escape(
            f"{row.get('scene_id')} {row.get('chunk_id')} {row.get('phase')} {row.get('failure_type')} "
            f"SF50={row.get('local_SF50')} AP50={row.get('local_AP50')}"
        )
        cards.append(
            "<article>"
            f"<h2>{title}</h2>"
            f"<p>{meta}</p>"
            f"<img src=\"{html.escape(image_src)}\" alt=\"{title}\" loading=\"lazy\">"
            "</article>"
        )
    body = "\n".join(cards)
    payload = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Stream4D v70 Casebook</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; background: #f7f7f4; }}
    header {{ max-width: 1080px; margin-bottom: 24px; }}
    article {{ max-width: 1280px; margin: 0 0 22px; padding: 12px 0 18px; border-bottom: 1px solid #d8d8d0; }}
    h1 {{ font-size: 28px; margin: 0 0 10px; }}
    h2 {{ font-size: 16px; margin: 0 0 6px; }}
    p {{ margin: 0 0 10px; font-size: 13px; }}
    img {{ width: 100%; max-width: 1280px; height: auto; display: block; border: 1px solid #c9cbc5; background: white; }}
    code {{ background: #ecece6; padding: 1px 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Stream4D v70 visual casebook</h1>
    <p>Generated from method artifacts without GT for prediction. GT appears only in the diagnostic panel. Decision: <code>{html.escape(str(summary.get("decision")))}</code>.</p>
  </header>
  {body}
</body>
</html>
"""
    html_path.write_text(payload, encoding="utf-8")
    return html_path


def _add_chunk_cases(
    *,
    rows: list[dict[str, Any]],
    variant_field: str,
    variant: str,
    phase: str,
    failure_type: str,
    limit: int,
    case_rows: list[dict[str, Any]],
) -> None:
    subset = [row for row in rows if str(row.get("variant") or row.get(variant_field)) == variant]
    subset.sort(
        key=lambda row: (
            _metric(row, "local_score_free_match50_recall", "local_SF50", "tracklet_SF50") or 0.0,
            _metric(row, "local_AP50") or 0.0,
        )
    )
    for row in subset[:limit]:
        scene = str(row.get("scene_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        case_rows.append(
            {
                "case_id": f"{phase}:{failure_type}:{len(case_rows):04d}",
                "scene_id": scene,
                "chunk_id": chunk_id,
                "frame_id": _int_or_none(row.get("frame_min")),
                "frame_min": _int_or_none(row.get("frame_min")),
                "frame_max": _int_or_none(row.get("frame_max")),
                "phase": phase,
                "variant": variant,
                "failure_type": failure_type,
                "local_SF50": _metric(row, "local_score_free_match50_recall", "local_SF50", "tracklet_SF50"),
                "local_AP50": _metric(row, "local_AP50"),
                "GT_best_IoU": _metric(row, "local_GT_best_IoU_mean", "GT_best_IoU_mean"),
                "single_frame_rate": _metric(row, "single_frame_object_rate", "tracklet_single_frame_rate"),
                "same_frame_violation_count": row.get("same_frame_cannot_link_violation_count") or row.get("same_frame_violation_count"),
                "diagnostic_only": False,
                "uses_gt_for_prediction": False,
                "artifact_source": row.get("pipeline_root") or "",
            }
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    viewer_root = _rooted(args.viewer_root)
    output_root.mkdir(parents=True, exist_ok=True)
    viewer_root.mkdir(parents=True, exist_ok=True)

    closure_summary = _load_json(_rooted(args.closure_summary))
    tracklet_summary = _load_json(_rooted(args.tracklet_summary))
    capsule_summary = _load_json(_rooted(args.capsule_summary))
    capsule_oracle_summary = _load_json(_rooted(args.capsule_oracle_summary))

    case_rows: list[dict[str, Any]] = []
    closure_rows = _load_csv_rows(_rooted((closure_summary.get("rows") or {}).get("closure_chunk_rows_csv", "")))
    tracklet_rows = _load_csv_rows(_rooted((tracklet_summary.get("rows") or {}).get("tracklet_chunk_rows_csv", "")))
    capsule_rows = _load_csv_rows(_rooted((capsule_summary.get("rows") or {}).get("capsule_chunk_rows_csv", "")))
    closure_best = (closure_summary.get("best_closure_variant") or {}).get("closure_variant") or "TC1_carrier_inside"
    tracklet_best = (tracklet_summary.get("best_tracklet_variant") or {}).get("tracklet_variant") or "TR5_greedy_adjacent_flow"
    capsule_best = (capsule_summary.get("best_capsule_variant") or {}).get("capsule_variant") or "OC3_appearance_cc_t060"
    _add_chunk_cases(
        rows=closure_rows,
        variant_field="closure_variant",
        variant=str(closure_best),
        phase="phase2_true_material_closure",
        failure_type="low_temporal_specific_closure",
        limit=int(args.per_phase_limit),
        case_rows=case_rows,
    )
    _add_chunk_cases(
        rows=tracklet_rows,
        variant_field="tracklet_variant",
        variant=str(tracklet_best),
        phase="phase3_masklet_tracklets",
        failure_type="short_single_frame_tracklets",
        limit=int(args.per_phase_limit),
        case_rows=case_rows,
    )
    _add_chunk_cases(
        rows=capsule_rows,
        variant_field="capsule_variant",
        variant=str(capsule_best),
        phase="phase4_object_capsule",
        failure_type="object_capsule_overfragment_underseg_dependency",
        limit=int(args.per_phase_limit),
        case_rows=case_rows,
    )

    oracle_metric_path = _rooted((capsule_oracle_summary.get("rows") or {}).get("capsule_metric_rows_csv", ""))
    if oracle_metric_path.exists():
        for row in _load_csv_rows(oracle_metric_path):
            variant = str(row.get("capsule_variant") or row.get("variant") or "")
            if variant.startswith("OC9") or variant.startswith("OC10"):
                case_rows.append(
                    {
                        "case_id": f"phase4_oracle:{variant}:{len(case_rows):04d}",
                        "scene_id": "scene0011_00",
                        "chunk_id": "summary",
                        "frame_id": 0,
                        "frame_min": "",
                        "frame_max": "",
                        "phase": "phase4_oracle_diagnostic",
                        "variant": variant,
                        "failure_type": "oracle_underseg_dependency" if variant.startswith("OC9") else "nonshared_oracle_low_ceiling",
                        "local_SF50": row.get("local_SF50"),
                        "local_AP50": row.get("local_AP50"),
                        "GT_best_IoU": row.get("GT_best_IoU_mean"),
                        "single_frame_rate": row.get("single_frame_object_rate"),
                        "same_frame_violation_count": row.get("same_frame_violation_count"),
                        "diagnostic_only": True,
                        "uses_gt_for_prediction": True,
                        "artifact_source": _rel(oracle_metric_path),
                    }
                )

    failure_counts = Counter(str(row.get("failure_type") or "") for row in case_rows)
    scenes = {
        str(row.get("scene_id"))
        for row in case_rows
        if row.get("scene_id") and row.get("scene_id") != "summary"
    }
    for summary in [closure_summary, tracklet_summary, capsule_summary]:
        for scene in summary.get("scenes") or []:
            scenes.add(str(scene))
    scenes = sorted(scenes)
    bookmarks = [
        {
            "bookmark_id": f"v70_bookmark_{idx:03d}",
            "case_id": row["case_id"],
            "scene_id": row["scene_id"],
            "phase": row["phase"],
            "failure_type": row["failure_type"],
            "artifact_source": row["artifact_source"],
        }
        for idx, row in enumerate(case_rows[: int(args.bookmark_limit)])
    ]
    candidate_rows = _load_csv_rows(_rooted(args.candidate_rows))
    capsule_edge_rows = _load_csv_rows(_rooted(args.capsule_edge_rows))
    method_mapping = _build_method_mapping(
        capsule_edge_rows=capsule_edge_rows,
        candidate_rows=candidate_rows,
        variant=str(args.capsule_edge_variant or capsule_best),
    )
    candidate_by_frame = _candidate_ids_by_frame(candidate_rows)
    screenshot_count = 0
    render_error_counts: Counter[str] = Counter()
    for row in case_rows:
        row["screenshot_path"] = ""
        row["method_layer_available_without_GT"] = False
        row["GT_layer_diagnostic_flag"] = True
        if screenshot_count >= int(args.max_screenshots):
            continue
        scene = str(row.get("scene_id") or "unknown_scene")
        case_id = str(row.get("case_id") or f"case_{screenshot_count:04d}").replace(":", "_").replace("/", "_")
        image_path = output_root / "images" / scene / f"{screenshot_count:04d}_{case_id}.png"
        diag = _make_case_image(
            case=row,
            method_mapping=method_mapping,
            candidate_by_frame=candidate_by_frame,
            output_path=image_path,
        )
        if not diag.get("rendered"):
            render_error_counts[str(diag.get("reason") or "unknown")] += 1
            row["render_error"] = diag.get("reason") or "unknown"
            continue
        screenshot_count += 1
        row["screenshot_path"] = _rel(image_path)
        row["render_error"] = ""
        row["frame_id"] = diag.get("frame_id")
        row["method_layer_available_without_GT"] = int(diag.get("method_pixel_count") or 0) > 0
        row["method_pixel_count"] = diag.get("method_pixel_count")
        row["candidate_pixel_count"] = diag.get("candidate_pixel_count")
        row["gt_pixel_count"] = diag.get("gt_pixel_count")
        row["gt_available"] = diag.get("gt_available")
        row["visual_pipeline_root"] = diag.get("pipeline_root")
        row["visual_mask_dir"] = diag.get("mask_dir")

    rendered_scene_counts = Counter(str(row.get("scene_id")) for row in case_rows if row.get("screenshot_path"))
    method_layer_loaded = bool(screenshot_count) and any(_bool_text(row.get("method_layer_available_without_GT")) for row in case_rows)
    viewer_scene_index = [
        {
            "scene_id": scene,
            "method_layers_available_without_GT": method_layer_loaded and rendered_scene_counts.get(scene, 0) > 0,
            "fallback_case_rows": sum(1 for row in case_rows if row.get("scene_id") == scene),
            "fallback_screenshot_count": int(rendered_scene_counts.get(scene, 0)),
            "fallback_html": _rel(output_root / "v70_casebook.html"),
            "viser_live_load_verified": False,
            "notes": "Fallback 2D PNG/HTML casebook; Viser live load was not verified in this run.",
        }
        for scene in scenes
    ]
    major_failure_types = [
        "low_temporal_specific_closure",
        "short_single_frame_tracklets",
        "object_capsule_overfragment_underseg_dependency",
    ]
    gate = {
        "viewer_scene_count_ge_5": len(scenes) >= 5,
        "method_layers_load_without_GT": method_layer_loaded,
        "bookmark_count_ge_30": len(bookmarks) >= 30,
        "case_count_ge_80": len(case_rows) >= 80,
        "screenshot_count_ge_30": screenshot_count >= 30,
        "major_failure_types_have_examples": all(failure_counts.get(name, 0) >= 5 for name in major_failure_types),
        "fallback_html_exported": True,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    summary = {
        "phase": "v70_casebook_fallback_2d_visual",
        "decision": "VISUALIZATION_BLOCKER" if not gate["pass"] else "PASS_FALLBACK_2D_CASEBOOK_VISUAL_EVIDENCE",
        "gate": gate,
        "viewer_scene_count": int(len(scenes)),
        "method_layers_load_without_GT": method_layer_loaded,
        "GT_layer_diagnostic_flag": True,
        "bookmark_count": int(len(bookmarks)),
        "screenshot_count": int(screenshot_count),
        "case_count": int(len(case_rows)),
        "failure_type_counts": dict(failure_counts),
        "render_error_counts": dict(render_error_counts),
        "viser_live_load_verified": False,
        "rows": {
            "casebook_rows_csv": _rel(output_root / "casebook_rows.csv"),
            "viewer_scene_index_json": _rel(viewer_root / "viewer_scene_index.json"),
            "bookmarks_json": _rel(viewer_root / "bookmarks.json"),
            "casebook_html": _rel(output_root / "v70_casebook.html"),
        },
        "notes": [
            "Fallback 2D PNG/HTML casebook generated from method artifacts; Viser live loading was not verified.",
            "No GT is used by method layers; GT appears only in the diagnostic panel.",
            "Oracle rows remain diagnostic-only and are marked uses_gt_for_prediction=True.",
        ],
    }
    html_path = _write_html_casebook(output_root, case_rows, summary)
    summary["rows"]["casebook_html"] = _rel(html_path)
    _write_csv(output_root / "casebook_rows.csv", case_rows)
    _write_json(viewer_root / "viewer_scene_index.json", {"scenes": viewer_scene_index})
    _write_json(viewer_root / "bookmarks.json", {"bookmarks": bookmarks})
    _write_json(output_root / "casebook_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "casebook_summary.json",
        output_root / "casebook_rows.csv",
        output_root / "v70_casebook.html",
        viewer_root / "viewer_scene_index.json",
        viewer_root / "bookmarks.json",
        *sorted((output_root / "images").glob("*/*.png"))[:20],
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stream4D v70 fallback visual failure casebook.")
    parser.add_argument("--output-root", default="outputs/audit/v70_casebook")
    parser.add_argument("--viewer-root", default="outputs/audit/v70_viser")
    parser.add_argument("--closure-summary", default="outputs/audit/v70_true_material_closure/closure_summary.json")
    parser.add_argument("--tracklet-summary", default="outputs/audit/v70_masklet_tracklets_smoke_scene0011/tracklet_summary.json")
    parser.add_argument("--capsule-summary", default="outputs/audit/v70_object_capsules_repair2_probe5/capsule_summary.json")
    parser.add_argument("--capsule-oracle-summary", default="outputs/audit/v70_object_capsules_oracle_cannotlink_smoke_scene0011/capsule_summary.json")
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--capsule-edge-rows", default="outputs/audit/v70_object_capsules_repair2_probe5/capsule_rows.csv")
    parser.add_argument("--capsule-edge-variant", default="OC3_appearance_cc_t060")
    parser.add_argument("--per-phase-limit", type=int, default=40)
    parser.add_argument("--bookmark-limit", type=int, default=40)
    parser.add_argument("--max-screenshots", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
