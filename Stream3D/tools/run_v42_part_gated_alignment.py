from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.part_gated_alignment import (
    TubeRoleAlignmentInfo,
    build_alignment_edge_evidence,
    evaluate_part_gated_alignment,
)
from stream4d_native.semantic_material_part_graph import build_token_material_support
from tools.diagnose_v42_material_query_reason import (
    _build_fast_material_measurements,
    _label_maps_from_masks,
    _load_edges,
    _load_tokens,
    _source_masks,
)
from tools.run_v42_semantic_part_audit import _load_d4rt_records
from tools.run_v42_tube_role_real import _audit_material_cache_stride


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_frame_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float(value: str, default: float = 0.0) -> float:
    text = str(value).strip()
    return float(text) if text else float(default)


def _parse_int(value: str, default: int = 0) -> int:
    text = str(value).strip()
    return int(text) if text else int(default)


def _load_role_rows(role_root: Path, *, scene: str, variant: str, source: str) -> dict[int, TubeRoleAlignmentInfo]:
    rows = _read_csv(role_root / "tube_role_rows.csv")
    out: dict[int, TubeRoleAlignmentInfo] = {}
    for row in rows:
        if str(row.get("scene", "")) != str(scene):
            continue
        if str(row.get("variant", "")) != str(variant):
            continue
        if str(row.get("source", "")) != str(source):
            continue
        tube_id = _parse_int(row.get("tube_id", ""))
        out[tube_id] = TubeRoleAlignmentInfo(
            tube_id=tube_id,
            role=str(row.get("role", "unknown")),
            residual_proxy=_parse_float(row.get("self_stitch_residual_proxy", ""), default=0.0),
            object_masklet_consistency=_parse_float(row.get("object_masklet_consistency", ""), default=0.0),
            scene_role_weight=_parse_float(row.get("scene_role_weight", ""), default=0.0),
            object_role_weight=_parse_float(row.get("object_role_weight", ""), default=0.0),
            part_role_weight=_parse_float(row.get("part_role_weight", ""), default=0.0),
            unknown_role_weight=_parse_float(row.get("unknown_role_weight", ""), default=0.0),
        )
    return out


def _edge_rows(
    *,
    scene: str,
    variant: str,
    source: str,
    evidences: list[Any],
    selected_by_variant: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    selected_pairs = {
        variant_name: {(int(edge.token_i), int(edge.token_j)) for edge in selected}
        for variant_name, selected in selected_by_variant.items()
    }
    rows: list[dict[str, Any]] = []
    for edge in evidences:
        row = {
            "scene": scene,
            "variant": variant,
            "source": source,
            "token_i": int(edge.token_i),
            "token_j": int(edge.token_j),
            "object_affinity": float(edge.object_affinity),
            "semantic_affinity": float(edge.semantic_affinity),
            "diagnostic_same_gt": edge.diagnostic_same_gt,
            "same_frame_cannot_link": bool(edge.same_frame_cannot_link),
            "shared_tube_count": int(edge.shared_tube_count),
            "object_part_tube_count": int(edge.object_part_tube_count),
            "trusted_material_tube_count": int(edge.trusted_material_tube_count),
            "scene_tube_count": int(len(edge.scene_tube_ids)),
            "object_tube_count": int(len(edge.object_tube_ids)),
            "part_tube_count": int(len(edge.part_tube_ids)),
            "unknown_tube_count": int(len(edge.unknown_tube_ids)),
            "material_union_count": int(edge.material_union_count),
            "visible_outside_conflict_ratio": float(edge.visible_outside_conflict_ratio),
            "residual_proxy": float(edge.residual_proxy),
            "role_conflict": bool(edge.role_conflict),
            "shared_tube_ids": list(edge.shared_tube_ids),
            "scene_tube_ids": list(edge.scene_tube_ids),
            "object_tube_ids": list(edge.object_tube_ids),
            "part_tube_ids": list(edge.part_tube_ids),
            "unknown_tube_ids": list(edge.unknown_tube_ids),
        }
        pair = (int(edge.token_i), int(edge.token_j))
        for variant_name in selected_pairs:
            row[f"selected_{variant_name}"] = bool(pair in selected_pairs[variant_name])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v42 semantic-part-gated alignment diagnostic.")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--sources", default="dinov2_maskcut")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--part-graph-root", required=True)
    parser.add_argument("--material-cache-root", required=True)
    parser.add_argument("--role-root", required=True)
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external_stride1_smoke")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--material-max-tubes-per-window", type=int, default=1024)
    parser.add_argument("--material-image-width", type=int, default=1296)
    parser.add_argument("--material-image-height", type=int, default=968)
    parser.add_argument("--material-min-visibility", type=float, default=0.5)
    parser.add_argument("--material-min-confidence", type=float, default=0.5)
    parser.add_argument("--backfill-overlap-iou", type=float, default=0.10)
    parser.add_argument("--backfill-max-masks-per-frame", type=int, default=8)
    parser.add_argument("--material-backfill-min-tubes", type=int, default=1)
    parser.add_argument("--material-backfill-max-candidate-area-fraction", type=float, default=1.0)
    parser.add_argument("--require-material-frame-stride", type=int, default=1)
    parser.add_argument("--semantic-threshold", type=float, default=0.50)
    parser.add_argument("--part-gate-semantic-threshold", type=float, default=0.15)
    parser.add_argument("--min-shared-tubes", type=int, default=1)
    parser.add_argument("--min-part-role-tubes", type=int, default=1)
    parser.add_argument("--max-visible-outside-conflict", type=float, default=0.35)
    parser.add_argument("--residual-inlier-threshold", type=float, default=0.10)
    parser.add_argument("--robust-trim-quantile", type=float, default=0.80)
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    frame_ids = _parse_frame_ids(str(args.frame_ids))
    variant = str(args.variant)
    part_graph_root = ROOT / str(args.part_graph_root)
    material_cache_root = ROOT / str(args.material_cache_root)
    role_root = ROOT / str(args.role_root)
    external_root = ROOT / str(args.external_source_root)
    output_root = ROOT / str(args.output_root)

    all_edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for scene in scenes:
        stride_diag = _audit_material_cache_stride(
            cache_root=material_cache_root / variant,
            scene=scene,
            required_stride=int(args.require_material_frame_stride),
        )
        records, d4rt_diag = _load_d4rt_records(
            cache_root=material_cache_root / variant,
            scene=scene,
            max_tubes_per_window=int(args.material_max_tubes_per_window),
            image_width=int(args.material_image_width),
            image_height=int(args.material_image_height),
        )
        d4rt_diag["material_cache_stride_audit"] = stride_diag
        stream = ScanNetStream(seq_name=scene)
        for source in sources:
            tokens = _load_tokens(part_graph_root, variant, scene, source)
            semantic_edges = _load_edges(part_graph_root, variant, scene, source)
            role_by_tube = _load_role_rows(role_root, scene=scene, variant=variant, source=source)
            masks_by_frame, mask_diag = _source_masks(
                source=source,
                stream=stream,
                scene=scene,
                frame_ids=frame_ids,
                min_area=int(args.min_area),
                sample_frames=int(args.sample_frames),
                external_root=external_root,
                d4rt_records=records,
                backfill_overlap_iou=float(args.backfill_overlap_iou),
                backfill_max_masks_per_frame=int(args.backfill_max_masks_per_frame),
                material_backfill_min_tubes=int(args.material_backfill_min_tubes),
                material_backfill_max_candidate_area_fraction=float(
                    args.material_backfill_max_candidate_area_fraction
                ),
                material_min_visibility=float(args.material_min_visibility),
                material_min_confidence=float(args.material_min_confidence),
            )
            label_maps = _label_maps_from_masks(masks_by_frame)
            measurements, measurement_diag = _build_fast_material_measurements(
                records,
                masks_by_frame=label_maps,
                min_visibility=float(args.material_min_visibility),
                min_confidence=float(args.material_min_confidence),
            )
            support_by_token = build_token_material_support(tokens, measurements)
            evidences = build_alignment_edge_evidence(semantic_edges, support_by_token, role_by_tube)
            result = evaluate_part_gated_alignment(
                evidences,
                semantic_threshold=float(args.semantic_threshold),
                part_gate_semantic_threshold=float(args.part_gate_semantic_threshold),
                min_shared_tubes=int(args.min_shared_tubes),
                min_part_role_tubes=int(args.min_part_role_tubes),
                max_visible_outside_conflict=float(args.max_visible_outside_conflict),
                residual_inlier_threshold=float(args.residual_inlier_threshold),
                robust_trim_quantile=float(args.robust_trim_quantile),
            )
            for row in result["variant_rows"]:
                summary_rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "source": source,
                        "token_count": int(len(tokens)),
                        "semantic_edge_count": int(len(semantic_edges)),
                        "d4rt_record_count": int(len(records)),
                        "role_row_count": int(len(role_by_tube)),
                        "measurement_count": int(len(measurements)),
                        **row,
                    }
                )
            all_edge_rows.extend(
                _edge_rows(
                    scene=scene,
                    variant=variant,
                    source=source,
                    evidences=evidences,
                    selected_by_variant=result["selected_by_variant"],
                )
            )
            manifests.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "source": source,
                    "d4rt_diag": d4rt_diag,
                    "mask_diag": mask_diag,
                    "measurement_diag": measurement_diag,
                    "token_count": int(len(tokens)),
                    "semantic_edge_count": int(len(semantic_edges)),
                    "role_row_count": int(len(role_by_tube)),
                    "phase5_proxy_gate_pass": bool(result["phase5_proxy_gate_pass"]),
                    "phase5_gate_pass": bool(result["phase5_gate_pass"]),
                    "phase5_gate_blocker": result["phase5_gate_blocker"],
                }
            )

    _write_csv(output_root / "alignment_rows.csv", all_edge_rows)
    _write_csv(output_root / "alignment_summary_rows.csv", summary_rows)
    _write_json(
        output_root / "alignment_summary.json",
        {
            "phase": "v42_part_gated_alignment_diagnostic",
            "scenes": scenes,
            "variant": variant,
            "sources": sources,
            "frame_ids": frame_ids,
            "part_graph_root": str(part_graph_root),
            "material_cache_root": str(material_cache_root),
            "role_root": str(role_root),
            "external_source_root": str(external_root),
            "alignment_rows_csv": str(output_root / "alignment_rows.csv"),
            "alignment_summary_rows_csv": str(output_root / "alignment_summary_rows.csv"),
            "summary_rows": summary_rows,
            "manifests": manifests,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "phase5_gate_pass": False,
            "note": "Diagnostic correspondence-factor audit. It uses role/material/part constraints and GT only for mismatch metrics; it does not optimize 3D object transforms.",
        },
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "alignment_rows": len(all_edge_rows),
                    "summary_rows": len(summary_rows),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
