from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v65_soma_pipeline_visualization import _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _chunk_num(chunk_id: Any) -> int | None:
    text = str(chunk_id or "")
    if "chunk" in text:
        tail = text.rsplit("chunk", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _chunk_key(scene: str, chunk_id: Any) -> str:
    num = _chunk_num(chunk_id)
    if num is None:
        return str(chunk_id)
    return f"{scene}:chunk{num:03d}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_anchor_rows(path: Path, variant: str, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        if str(row.get("anchor_variant")) != variant:
            continue
        scene = str(row.get("scene_id") or row.get("scene") or "")
        if scene not in scenes:
            continue
        row["_chunk_key"] = _chunk_key(scene, row.get("chunk_id"))
        row["_frame_id"] = _safe_int(row.get("frame_id"))
        row["_mask_id"] = _safe_int(row.get("mask_id"))
        out[scene].append(row)
    return out


def _load_candidate_rows(path: Path, scenes: set[str]) -> dict[str, dict[str, dict[int, list[dict[str, Any]]]]]:
    out: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in _read_csv(path):
        scene = str(row.get("scene_id") or row.get("scene") or "")
        if scene not in scenes:
            continue
        if "representative_available" in row and not _parse_bool(row.get("representative_available")):
            continue
        row["_chunk_key"] = _chunk_key(scene, row.get("chunk_id"))
        row["_frame_id"] = _safe_int(row.get("frame_id"))
        row["_mask_id"] = _safe_int(row.get("mask_id"))
        out[scene][str(row["_chunk_key"])][int(row["_frame_id"])].append(row)
    return out


def _observation_paths(pipeline_root: Path) -> tuple[Path, Path, Path]:
    obs_root = pipeline_root / "observation_tables"
    return (
        obs_root / "carrier_observation_table.csv",
        obs_root / "mask_observation_table.csv",
        obs_root / "observation_table_summary.json",
    )


def _load_observation_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame_observations(
    path: Path,
    *,
    min_visibility: float,
    min_confidence: float,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    frame_obs: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_rows = 0
    accepted_rows = 0
    invalid_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            if not _parse_bool(row.get("valid")) or not _parse_bool(row.get("valid_uv")) or not _parse_bool(row.get("visible")):
                invalid_rows += 1
                continue
            visibility = _safe_float(row.get("visibility_prob"))
            confidence = _safe_float(row.get("confidence"))
            if visibility < float(min_visibility) or confidence < float(min_confidence):
                invalid_rows += 1
                continue
            frame_id = _safe_int(row.get("frame_id"))
            carrier_id = str(row.get("carrier_global_id") or row.get("carrier_id"))
            observed = _safe_int(row.get("observed_mask_id"), 0)
            current = frame_obs[frame_id].get(carrier_id)
            score = visibility * confidence
            if current is not None and float(current.get("_score", 0.0)) >= score:
                continue
            frame_obs[frame_id][carrier_id] = {
                "observed_mask_id": observed,
                "visibility_prob": visibility,
                "confidence": confidence,
                "uv_x": _safe_float(row.get("uv_x")),
                "uv_y": _safe_float(row.get("uv_y")),
                "_score": score,
            }
            accepted_rows += 1
    stats = {
        "raw_carrier_observation_rows": int(raw_rows),
        "accepted_carrier_observation_rows": int(accepted_rows),
        "invalid_or_below_threshold_rows": int(invalid_rows),
        "frame_count_with_observations": int(len(frame_obs)),
    }
    return frame_obs, stats


def _mask_carriers_for_frame(frame_obs: dict[str, dict[str, Any]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for carrier_id, obs in frame_obs.items():
        mask_id = int(obs.get("observed_mask_id") or 0)
        if mask_id > 0:
            out[mask_id].append(carrier_id)
    return out


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, int(round(0.90 * (len(values_sorted) - 1))))
    return float(values_sorted[idx])


def _candidate_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_underseg_risk": _parse_bool(row.get("underseg_risk")),
        "candidate_area_ratio": _safe_float(row.get("area_ratio")),
        "candidate_DINO_mode_id": row.get("semantic_mode_id") or row.get("DINO_mode_id") or "",
        "candidate_repeated_signature_id": row.get("repeated_signature_id") or "",
        "candidate_large_mask_risk": _parse_bool(row.get("large_mask_risk")),
        "candidate_small_mask_risk": _parse_bool(row.get("small_mask_risk")),
        "candidate_same_frame_overlap_count": _safe_int(row.get("same_frame_overlap_count")),
    }


def _build_scene_witness(
    *,
    scene: str,
    pipeline_root: Path,
    anchors: list[dict[str, Any]],
    candidates: dict[str, dict[int, list[dict[str, Any]]]],
    min_visibility: float,
    min_confidence: float,
    min_visible_candidate_carriers: int,
    max_rows_per_anchor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    carrier_path, mask_obs_path, obs_summary_path = _observation_paths(pipeline_root)
    missing_rows: list[dict[str, Any]] = []
    if not carrier_path.exists():
        missing_rows.append({"scene_id": scene, "missing": "carrier_observation_table", "path": _rel(carrier_path)})
        return [], [], missing_rows, {}
    if not mask_obs_path.exists():
        missing_rows.append({"scene_id": scene, "missing": "mask_observation_table", "path": _rel(mask_obs_path)})
    obs_summary = _load_observation_summary(obs_summary_path)
    frame_obs, obs_stats = _load_frame_observations(carrier_path, min_visibility=min_visibility, min_confidence=min_confidence)
    mask_carriers_by_frame = {frame_id: _mask_carriers_for_frame(obs) for frame_id, obs in frame_obs.items()}
    witness_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    anchor_counts: list[int] = []
    candidate_counts: list[int] = []
    visible_counts: list[int] = []
    inside_ratios: list[float] = []
    outside_ratios: list[float] = []
    frame_delta_hist: Counter[int] = Counter()
    anchors_with_carrier_set = 0
    anchors_with_candidate_witness = 0
    anchors_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        anchors_by_chunk[str(row["_chunk_key"])].append(row)

    for chunk_key, chunk_anchors in sorted(anchors_by_chunk.items()):
        chunk_anchor_count = len(chunk_anchors)
        chunk_anchor_with_carrier = 0
        chunk_anchor_with_candidate = 0
        chunk_witness_rows_start = len(witness_rows)
        for anchor in chunk_anchors:
            anchor_frame = int(anchor["_frame_id"])
            anchor_mask = int(anchor["_mask_id"])
            anchor_carriers = mask_carriers_by_frame.get(anchor_frame, {}).get(anchor_mask, [])
            anchor_carrier_count = len(anchor_carriers)
            anchor_counts.append(float(anchor_carrier_count))
            if anchor_carrier_count > 0:
                anchors_with_carrier_set += 1
                chunk_anchor_with_carrier += 1
            ranked_rows: list[dict[str, Any]] = []
            for candidate_frame, candidate_list in sorted(candidates.get(chunk_key, {}).items()):
                if int(candidate_frame) == anchor_frame:
                    continue
                obs_for_frame = frame_obs.get(int(candidate_frame), {})
                visible_obs = [obs_for_frame[cid] for cid in anchor_carriers if cid in obs_for_frame]
                visible_count = len(visible_obs)
                if visible_count < int(min_visible_candidate_carriers):
                    continue
                mask_counts: Counter[int] = Counter(int(obs.get("observed_mask_id") or 0) for obs in visible_obs if int(obs.get("observed_mask_id") or 0) > 0)
                if not mask_counts:
                    continue
                vis_mean = _mean([float(obs["visibility_prob"]) for obs in visible_obs])
                conf_mean = _mean([float(obs["confidence"]) for obs in visible_obs])
                by_mask = {int(row["_mask_id"]): row for row in candidate_list}
                for mask_id, inside_count in mask_counts.items():
                    candidate = by_mask.get(int(mask_id))
                    if candidate is None:
                        continue
                    outside_count = int(visible_count - inside_count)
                    inside_ratio = float(inside_count / max(visible_count, 1))
                    outside_ratio = float(outside_count / max(visible_count, 1))
                    frame_delta = int(abs(int(candidate_frame) - anchor_frame))
                    item = {
                        "scene_id": scene,
                        "chunk_id": chunk_key,
                        "anchor_frame": anchor_frame,
                        "anchor_mask": anchor_mask,
                        "candidate_frame": int(candidate_frame),
                        "candidate_mask": int(mask_id),
                        "anchor_carrier_count": int(anchor_carrier_count),
                        "visible_carrier_count": int(visible_count),
                        "inside_candidate_count": int(inside_count),
                        "outside_candidate_count": int(outside_count),
                        "inside_ratio": inside_ratio,
                        "outside_ratio": outside_ratio,
                        "confidence_mean": conf_mean,
                        "visibility_mean": vis_mean,
                        "frame_delta": frame_delta,
                        "same_frame_conflict": False,
                        "anchor_variant": str(anchor.get("anchor_variant")),
                        "anchor_DINO_mode_id": str(anchor.get("DINO_mode_id") or ""),
                        "anchor_repeated_signature_id": str(anchor.get("repeated_signature_id") or ""),
                        "anchor_underseg_risk": _parse_bool(anchor.get("underseg_risk")),
                        "uses_gt_for_prediction": False,
                        "diagnostic_only": False,
                        "forbidden_for_method_table": False,
                        "witness_source": "carrier_observation_table.observed_mask_id",
                    }
                    item.update(_candidate_meta(candidate))
                    ranked_rows.append(item)
            ranked_rows.sort(
                key=lambda row: (
                    float(row["inside_ratio"]),
                    int(row["inside_candidate_count"]),
                    int(row["visible_carrier_count"]),
                    -int(row["candidate_underseg_risk"]),
                    -int(row["frame_delta"]),
                ),
                reverse=True,
            )
            if max_rows_per_anchor > 0:
                ranked_rows = ranked_rows[: int(max_rows_per_anchor)]
            if ranked_rows:
                anchors_with_candidate_witness += 1
                chunk_anchor_with_candidate += 1
            candidate_counts.append(float(len(ranked_rows)))
            for row in ranked_rows:
                visible_counts.append(float(row["visible_carrier_count"]))
                inside_ratios.append(float(row["inside_ratio"]))
                outside_ratios.append(float(row["outside_ratio"]))
                frame_delta_hist[int(row["frame_delta"])] += 1
            witness_rows.extend(ranked_rows)
        chunk_witness_count = len(witness_rows) - chunk_witness_rows_start
        chunk_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk_key,
                "anchor_count": int(chunk_anchor_count),
                "anchor_with_carrier_set_count": int(chunk_anchor_with_carrier),
                "anchor_with_candidate_witness_count": int(chunk_anchor_with_candidate),
                "witness_row_count": int(chunk_witness_count),
                "candidate_masks_per_anchor_mean": float(chunk_witness_count / max(chunk_anchor_count, 1)),
                "uses_gt_for_prediction": False,
            }
        )

    stats = {
        "scene_id": scene,
        "pipeline_root": _rel(pipeline_root),
        "anchor_count": int(len(anchors)),
        "anchor_with_carrier_set_count": int(anchors_with_carrier_set),
        "anchor_with_candidate_witness_count": int(anchors_with_candidate_witness),
        "witness_row_count": int(len(witness_rows)),
        "mean_anchor_carrier_count": _mean(anchor_counts),
        "median_anchor_carrier_count": _median(anchor_counts),
        "candidate_masks_per_anchor_mean": _mean(candidate_counts),
        "candidate_masks_per_anchor_p90": _p90(candidate_counts),
        "visible_carrier_count_mean": _mean(visible_counts),
        "inside_ratio_mean": _mean(inside_ratios),
        "outside_ratio_mean": _mean(outside_ratios),
        "frame_delta_distribution": {str(k): int(v) for k, v in sorted(frame_delta_hist.items())},
        "carrier_observation_table": _rel(carrier_path),
        "mask_observation_table": _rel(mask_obs_path),
        "mask_root_arg": obs_summary.get("mask_root_arg"),
        "resolved_mask_dirs": obs_summary.get("resolved_mask_dirs"),
    }
    stats.update(obs_stats)
    return witness_rows, chunk_rows, missing_rows, stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes) or list(DEFAULT_SCENES)
    scene_set = set(scenes)
    anchors_by_scene = _load_anchor_rows(_rooted(args.anchor_rows), str(args.anchor_variant), scene_set)
    candidates_by_scene = _load_candidate_rows(_rooted(args.candidate_rows), scene_set)
    all_witness: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v70-carrier-witness] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        witness, chunks, missing, stats = _build_scene_witness(
            scene=scene,
            pipeline_root=pipeline_root,
            anchors=anchors_by_scene.get(scene, []),
            candidates=candidates_by_scene.get(scene, {}),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            min_visible_candidate_carriers=int(args.min_visible_candidate_carriers),
            max_rows_per_anchor=int(args.max_rows_per_anchor),
        )
        all_witness.extend(witness)
        all_chunks.extend(chunks)
        missing_rows.extend(missing)
        if stats:
            scene_rows.append(stats)

    anchor_count = sum(int(row.get("anchor_count") or 0) for row in scene_rows)
    carrier_set_count = sum(int(row.get("anchor_with_carrier_set_count") or 0) for row in scene_rows)
    candidate_witness_count = sum(int(row.get("anchor_with_candidate_witness_count") or 0) for row in scene_rows)
    witness_row_count = len(all_witness)
    frame_delta_total: Counter[str] = Counter()
    for row in scene_rows:
        for key, value in (row.get("frame_delta_distribution") or {}).items():
            frame_delta_total[str(key)] += int(value)
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "carrier_observation_tables_available": not any(row.get("missing") == "carrier_observation_table" for row in missing_rows),
        "mask_observation_tables_available": not any(row.get("missing") == "mask_observation_table" for row in missing_rows),
        "anchor_with_carrier_witness_rate_ge_0p50": anchor_count > 0 and float(candidate_witness_count / max(anchor_count, 1)) >= 0.50,
        "mean_anchor_carrier_count_ge_16": _mean([float(row.get("mean_anchor_carrier_count") or 0.0) for row in scene_rows]) is not None
        and float(_mean([float(row.get("mean_anchor_carrier_count") or 0.0) for row in scene_rows]) or 0.0) >= 16.0,
        "candidate_masks_per_anchor_mean_ge_2": anchor_count > 0 and float(witness_row_count / max(anchor_count, 1)) >= 2.0,
        "missing_carrier_cache_count_eq_0": not any(row.get("missing") == "carrier_observation_table" for row in missing_rows),
        "missing_mask_png_count_eq_0": not any(row.get("missing") == "mask_observation_table" for row in missing_rows),
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    summary = {
        "phase": "v70_carrier_witness",
        "decision": "PASS_CARRIER_WITNESS_TABLE" if gate["pass"] else "NO_GO_CARRIER_WITNESS_TABLE",
        "gate": gate,
        "anchor_variant": str(args.anchor_variant),
        "anchor_rows": _rel(_rooted(args.anchor_rows)),
        "candidate_rows": _rel(_rooted(args.candidate_rows)),
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "min_visible_candidate_carriers": int(args.min_visible_candidate_carriers),
        "max_rows_per_anchor": int(args.max_rows_per_anchor),
        "anchor_count": int(anchor_count),
        "anchor_with_carrier_set_count": int(carrier_set_count),
        "anchor_with_carrier_set_rate": float(carrier_set_count / max(anchor_count, 1)),
        "anchor_with_carrier_witness_count": int(candidate_witness_count),
        "anchor_with_carrier_witness_rate": float(candidate_witness_count / max(anchor_count, 1)),
        "mean_anchor_carrier_count": _mean([float(row.get("mean_anchor_carrier_count") or 0.0) for row in scene_rows]),
        "median_anchor_carrier_count": _median([float(row.get("median_anchor_carrier_count") or 0.0) for row in scene_rows]),
        "candidate_witness_rows": int(witness_row_count),
        "candidate_masks_per_anchor_mean": float(witness_row_count / max(anchor_count, 1)),
        "candidate_masks_per_anchor_p90": _p90([float(row.get("candidate_masks_per_anchor_p90") or 0.0) for row in scene_rows]),
        "visible_carrier_count_mean": _mean([float(row.get("visible_carrier_count_mean") or 0.0) for row in scene_rows]),
        "inside_ratio_mean": _mean([float(row.get("inside_ratio_mean") or 0.0) for row in scene_rows]),
        "outside_ratio_mean": _mean([float(row.get("outside_ratio_mean") or 0.0) for row in scene_rows]),
        "frame_delta_distribution": {str(k): int(v) for k, v in sorted(frame_delta_total.items(), key=lambda item: int(item[0]))},
        "missing_carrier_cache_count": sum(1 for row in missing_rows if row.get("missing") == "carrier_observation_table"),
        "missing_mask_png_count": sum(1 for row in missing_rows if row.get("missing") == "mask_observation_table"),
        "candidate_membership_source": "carrier_observation_table.observed_mask_id",
        "direct_mask_png_read": False,
        "uses_gt_for_prediction": False,
        "rows": {
            "witness_rows_csv": _rel(output_root / "witness_rows.csv"),
            "witness_chunk_rows_csv": _rel(output_root / "witness_chunk_rows.csv"),
            "witness_scene_rows_csv": _rel(output_root / "witness_scene_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Carrier witness rows are built from carrier_observation_table uv/visibility/confidence and observed_mask_id; no GT labels are used for prediction.",
            "missing_mask_png_count refers to missing mask observation table for this observation-table path, not direct visual PNG reads.",
        ],
    }
    _write_csv(output_root / "witness_rows.csv", all_witness)
    _write_csv(output_root / "witness_chunk_rows.csv", all_chunks)
    _write_csv(output_root / "witness_scene_rows.csv", scene_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_json(output_root / "witness_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "witness_summary.json",
        output_root / "witness_rows.csv",
        output_root / "witness_chunk_rows.csv",
        output_root / "witness_scene_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 true carrier witness builder.")
    parser.add_argument("--output-root", default="outputs/audit/v70_carrier_witness")
    parser.add_argument("--anchor-rows", default="outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_rows.csv")
    parser.add_argument("--anchor-variant", default="A9_clean_recall_support_floor_u15")
    parser.add_argument("--candidate-rows", default="outputs/audit/v68_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--min-visible-candidate-carriers", type=int, default=1)
    parser.add_argument("--max-rows-per-anchor", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
