from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_mask_if_available(stream: ScanNetStream, frame_id: int) -> np.ndarray | None:
    path = stream.mask_dir / f"{int(frame_id)}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


def _sample_mask_ids(mask: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    x = np.rint(uv_norm[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    y = np.rint(uv_norm[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out = np.zeros((uv_norm.shape[0],), dtype=np.int64)
    if np.any(in_bounds):
        out[in_bounds] = mask[y[in_bounds], x[in_bounds]]
    return out, in_bounds


def _load_window_carriers(scene_dir: Path) -> list[Path]:
    return sorted(scene_dir.glob("carriers_window*.npz"))


def _history_key(observations: list[tuple[int, int]], max_observations: int) -> tuple[tuple[int, int], ...]:
    if max_observations > 0 and len(observations) > max_observations:
        observations = observations[:max_observations]
    return tuple(sorted((int(frame_id), int(mask_id)) for frame_id, mask_id in observations))


def _cluster_window_by_mask_history(
    *,
    stream: ScanNetStream,
    carrier_path: Path,
    min_visibility: float,
    min_confidence: float,
    min_observations: int,
    max_observations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(carrier_path) as data:
        carrier_id = np.asarray(data["carrier_id"], dtype=np.int64)
        uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
        visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
        confidence = np.asarray(data["confidence_prob"], dtype=np.float32)

    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame_ids = [int(v) for v in summary.get("frame_ids", [])]
    else:
        scene_summary = carrier_path.parent / "summary.json"
        summary = json.loads(scene_summary.read_text(encoding="utf-8")) if scene_summary.exists() else {}
        frame_ids = [int(v) for v in summary.get("frame_ids", [])]
    if len(frame_ids) != uv_pred.shape[0]:
        frame_ids = list(range(uv_pred.shape[0]))

    available_masks: dict[int, np.ndarray] = {}
    for local_idx, frame_id in enumerate(frame_ids):
        mask = _load_mask_if_available(stream, frame_id)
        if mask is not None:
            available_masks[int(local_idx)] = mask

    obs_by_carrier: list[list[tuple[int, int]]] = [[] for _ in range(carrier_id.shape[0])]
    total_valid_samples = 0
    positive_samples = 0
    in_bounds_samples = 0
    for local_idx, mask in available_masks.items():
        valid = (visibility[local_idx] >= min_visibility) & (confidence[local_idx] >= min_confidence)
        if not np.any(valid):
            continue
        mask_ids, in_bounds = _sample_mask_ids(mask, uv_pred[local_idx])
        valid &= in_bounds
        total_valid_samples += int(np.count_nonzero(valid))
        in_bounds_samples += int(np.count_nonzero(in_bounds))
        positive = valid & (mask_ids > 0)
        positive_samples += int(np.count_nonzero(positive))
        frame_id = int(frame_ids[local_idx])
        for idx in np.flatnonzero(positive):
            obs_by_carrier[int(idx)].append((frame_id, int(mask_ids[int(idx)])))

    cluster_members: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    rejected_unobserved = 0
    rejected_low_observation = 0
    for idx, observations in enumerate(obs_by_carrier):
        if not observations:
            rejected_unobserved += 1
            continue
        unique_observations = sorted(set(observations))
        if len(unique_observations) < int(min_observations):
            rejected_low_observation += 1
            continue
        key = _history_key(unique_observations, max_observations=max_observations)
        cluster_members[key].append(int(idx))

    clusters: list[dict[str, Any]] = []
    for key, members in sorted(cluster_members.items(), key=lambda item: (-len(item[1]), item[0])):
        if not key:
            continue
        obs_counter = Counter()
        member_set = set(members)
        for idx in members:
            for obs in set(obs_by_carrier[idx]):
                obs_counter[obs] += 1
        clusters.append(
            {
                "history": key,
                "members": np.asarray(sorted(member_set), dtype=np.int64),
                "carrier_ids": carrier_id[np.asarray(sorted(member_set), dtype=np.int64)],
                "obs_counts": {obs: int(count) for obs, count in obs_counter.items()},
            }
        )

    diagnostics = {
        "carrier_path": str(carrier_path),
        "num_carriers": int(carrier_id.shape[0]),
        "num_frames": int(uv_pred.shape[0]),
        "num_available_mask_frames": int(len(available_masks)),
        "available_mask_frames": [int(frame_ids[idx]) for idx in sorted(available_masks)],
        "valid_visible_samples_on_mask_frames": int(total_valid_samples),
        "in_bounds_samples_on_mask_frames": int(in_bounds_samples),
        "positive_mask_samples": int(positive_samples),
        "positive_mask_sample_rate": float(positive_samples / max(total_valid_samples, 1)),
        "rejected_unobserved_carriers": int(rejected_unobserved),
        "rejected_low_observation_carriers": int(rejected_low_observation),
        "raw_clusters": int(len(clusters)),
    }
    return clusters, diagnostics


def _apply_mask_ownership(clusters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claims: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for cluster_idx, cluster in enumerate(clusters):
        size = int(cluster["members"].shape[0])
        for obs, count in cluster["obs_counts"].items():
            claims[(int(obs[0]), int(obs[1]))].append((int(cluster_idx), int(count), size))

    winners: dict[tuple[int, int], int] = {}
    competing = 0
    total_claims = 0
    dropped_claims = 0
    for obs, obs_claims in claims.items():
        total_claims += len(obs_claims)
        if len(obs_claims) > 1:
            competing += 1
        winner, _, _ = max(obs_claims, key=lambda item: (item[1], item[2], -item[0]))
        winners[obs] = winner
        dropped_claims += max(0, len(obs_claims) - 1)

    owned_clusters: list[dict[str, Any]] = []
    for cluster_idx, cluster in enumerate(clusters):
        owned_mask_list: list[tuple[int, int, float]] = []
        size = int(cluster["members"].shape[0])
        for obs, count in sorted(cluster["obs_counts"].items(), key=lambda item: (-item[1], item[0])):
            obs_key = (int(obs[0]), int(obs[1]))
            if winners.get(obs_key) != cluster_idx:
                continue
            coverage = float(count) / float(max(size, 1))
            owned_mask_list.append((int(obs[0]), int(obs[1]), coverage))
        out = dict(cluster)
        out["mask_list"] = owned_mask_list
        owned_clusters.append(out)

    diagnostics = {
        "ownership_total_claims": int(total_claims),
        "ownership_competing_masks": int(competing),
        "ownership_dropped_claims": int(dropped_claims),
    }
    return owned_clusters, diagnostics


def _clusters_to_object_dict(
    clusters: list[dict[str, Any]],
    min_carriers: int,
    min_owned_masks: int,
    max_masks_per_object: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    object_dict: dict[int, dict[str, Any]] = {}
    dropped_small = 0
    dropped_no_masks = 0
    for cluster in clusters:
        members = np.asarray(cluster["members"], dtype=np.int64)
        if members.shape[0] < int(min_carriers):
            dropped_small += 1
            continue
        mask_list = list(cluster.get("mask_list", []))
        if max_masks_per_object > 0:
            mask_list = sorted(mask_list, key=lambda item: float(item[2]), reverse=True)[: int(max_masks_per_object)]
        if len(mask_list) < int(min_owned_masks):
            dropped_no_masks += 1
            continue
        object_id = len(object_dict)
        object_dict[object_id] = {
            "mask_list": mask_list,
            "carrier_ids": np.asarray(cluster["carrier_ids"], dtype=np.int64),
            "history": list(cluster["history"]),
            "surfel_member_count": int(members.shape[0]),
        }
    diagnostics = {
        "dropped_small_clusters": int(dropped_small),
        "dropped_no_owned_mask_clusters": int(dropped_no_masks),
        "object_dict_size": int(len(object_dict)),
    }
    return object_dict, diagnostics


def _export_scene(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode=args.export_score_mode,
    )

    scene_dir = Path(args.debug_root) / seq_name
    carrier_paths = _load_window_carriers(scene_dir)
    if not carrier_paths:
        raise FileNotFoundError(f"No carrier windows under {scene_dir}")

    all_clusters: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    totals = Counter()
    for window_idx, carrier_path in enumerate(carrier_paths):
        clusters, diag = _cluster_window_by_mask_history(
            stream=stream,
            carrier_path=carrier_path,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            min_observations=int(args.min_observations),
            max_observations=int(args.max_observations),
        )
        for cluster in clusters:
            cluster = dict(cluster)
            cluster["window_index"] = int(window_idx)
            all_clusters.append(cluster)
        for key, value in diag.items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
        window_rows.append(diag)
    if totals.get("valid_visible_samples_on_mask_frames", 0.0) > 0.0:
        totals["positive_mask_sample_rate"] = (
            float(totals.get("positive_mask_samples", 0.0))
            / float(max(totals.get("valid_visible_samples_on_mask_frames", 0.0), 1.0))
        )

    owned_clusters, ownership_diag = _apply_mask_ownership(all_clusters)
    object_dict, object_diag = _clusters_to_object_dict(
        owned_clusters,
        min_carriers=int(args.min_carriers),
        min_owned_masks=int(args.min_owned_masks),
        max_masks_per_object=int(args.max_masks_per_object),
    )
    export_diag = exporter.export_object_dict_mask_backproject(object_dict)

    summary = {
        "args": vars(args),
        "algorithm": "v8_surfel_object_field_partition",
        "direction": args.prototype_direction,
        "uses_gt": False,
        "is_method_result": True,
        "seq_name": seq_name,
        "num_windows": int(len(carrier_paths)),
        "num_raw_clusters_total": int(len(all_clusters)),
        **{key: float(value) for key, value in totals.items()},
        **ownership_diag,
        **object_diag,
        **export_diag,
        "windows": window_rows,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.output_config}_{seq_name}_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_aggregate(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "args": vars(args),
        "algorithm": "v8_surfel_object_field_partition",
        "direction": args.prototype_direction,
        "uses_gt": False,
        "is_method_result": True,
        "num_scenes": len(summaries),
        "scenes": summaries,
    }
    numeric_keys = sorted(
        {
            key
            for row in summaries
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
        }
    )
    aggregate["numeric_mean"] = {
        key: float(np.mean([float(row[key]) for row in summaries if key in row]))
        for key in numeric_keys
        if any(key in row for row in summaries)
    }
    aggregate_path = out_dir / f"{args.output_config}_summary.json"
    aggregate_path.write_text(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / f"{args.output_config}_summary.csv"
    fieldnames = ["seq_name"] + numeric_keys
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})

    md_path = out_dir / f"{args.output_config}_summary.md"
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | objects | points | raw clusters | positive samples | competing masks | dropped claims | conflict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {seq} | {obj:.0f} | {pts:.0f} | {clusters} | {pos:.0f} | {comp} | {drop} | {conf:.6f} |".format(
                seq=row.get("seq_name"),
                obj=float(row.get("num_exported_objects", 0.0)),
                pts=float(row.get("num_exported_points", 0.0)),
                clusters=int(row.get("num_raw_clusters_total", 0)),
                pos=float(row.get("positive_mask_samples", 0.0)),
                comp=int(row.get("ownership_competing_masks", 0)),
                drop=int(row.get("ownership_dropped_claims", 0)),
                conf=float(row.get("export_conflict_rate", 0.0)),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root)],
        pre_points_policy="recompute",
        support_policy=f"v8_surfel_object_field:{args.prototype_direction}:mask_history_mask_backproject",
        notes=(
            f"{args.prototype_direction} lightweight surfel object field prototype. D4RT grid surfels are grouped "
            "by non-GT 2D mask observation history on available mask frames; same-frame different masks are kept "
            "separate; each 2D mask is owned by at most one cluster before full-mask backprojection. No GT is read."
        ),
        extra={
            "algorithm": "v8_surfel_object_field_partition",
            "direction": args.prototype_direction,
            "eval_policy": args.eval_policy,
            "summary_path": str(aggregate_path),
            "seq_list": str(args.seq_list),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v8 Lane3A surfel object field from D4RT grid surfel tracks.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--prototype-direction",
        choices=[
            "A_signed_history",
            "B_surfacelet_singlemask",
            "C_core_fringe_reject",
        ],
        default="A_signed_history",
    )
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v8_surfel_object_field")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--max-observations", type=int, default=0)
    parser.add_argument("--min-carriers", type=int, default=16)
    parser.add_argument("--min-owned-masks", type=int, default=1)
    parser.add_argument("--max-masks-per-object", type=int, default=2)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--eval-policy", default="own_recompute_v8_surfel_object_field")
    parser.add_argument(
        "--export-score-mode",
        choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"],
        default="reliability",
    )
    args = parser.parse_args()

    seq_names = _read_seq_list(Path(args.seq_list))
    summaries = [_export_scene(args, seq_name) for seq_name in seq_names]
    _write_aggregate(args, summaries)
    print(json.dumps(_json_safe({"output_config": args.output_config, "scenes": summaries}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
