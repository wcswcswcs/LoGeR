#!/usr/bin/env python3
"""Build v79 current bad-target ledgers with semantic diagnosis.

Diagnostic-only. This tool reads existing trajectories and semantic caches,
then writes the three Phase1 ledgers required by the v79 plan. It does not run
LoGeR, tune thresholds, or claim method success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v78_geometry_regime_contrast import (  # noqa: E402
    _aggregate_frame_features,
    _frame_features,
)
from tools.build_v78_bad_window_tables import _evaluate_run  # noqa: E402


DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
DEFAULT_RGB_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase1_current_bad_target_mining_with_semantic_diagnosis"
)
DEFAULT_H35_FULL = Path(
    "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075/01.txt"
)
DEFAULT_H35_704 = Path(
    "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_refine_screen/rollouts/"
    "V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F/01.txt"
)
DEFAULT_CURRENT_SCAN_SUMMARY = Path(
    "results/kitti_acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase4_ttt_current_baseline_window_scan_v1/current_baseline_window_scan_summary.json"
)
DEFAULT_CURRENT_TARGET_DIR = Path(
    "results/kitti_acl2_v78tf_pca_grounded_memory_control/report_final/"
    "current_ttt_bad_window_selection/v1_current_baseline_targets"
)


FAMILY_SPECS = {
    "single": {
        "id": "chunk_id",
        "metric": "local_sim3_rmse_m",
        "start": "chunk_start_frame",
        "end": "chunk_end_frame",
    },
    "adjacent": {
        "id": "chunk_pair",
        "metric": "tail3_to_future_from_boundary_sim3_rmse_m",
        "start": "pair_start_frame",
        "end": "pair_end_frame",
        "boundary": "boundary_frame",
    },
    "five": {
        "id": "window_chunks",
        "metric": "window5_joint_sim3_rmse_m",
        "start": "window_start_frame",
        "end": "window_end_frame",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--h35-full", type=Path, default=DEFAULT_H35_FULL)
    parser.add_argument("--h35-704", type=Path, default=DEFAULT_H35_704)
    parser.add_argument("--current-scan-summary", type=Path, default=DEFAULT_CURRENT_SCAN_SUMMARY)
    parser.add_argument("--current-target-dir", type=Path, default=DEFAULT_CURRENT_TARGET_DIR)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--top-k-primary", type=int, default=12)
    parser.add_argument("--top-k-supplemental", type=int, default=12)
    parser.add_argument(
        "--no-supplemental-current-v78",
        action="store_true",
        help="Do not add v78 current replay scan trajectories as supplemental rows.",
    )
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb_path(rgb_root: Path, seq: str, frame: int) -> Path | None:
    base = rgb_root / str(seq).zfill(2) / "image_2"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = base / f"{int(frame):06d}{suffix}"
        if path.exists():
            return path
    return None


class SemanticAssetResolver:
    def __init__(self, preprocess_root: Path, rgb_root: Path) -> None:
        self.preprocess_root = preprocess_root
        self.rgb_root = rgb_root
        self.index_cache: dict[str, list[dict[str, Any]]] = {}

    def _index(self, seq: str) -> list[dict[str, Any]]:
        seq = str(seq).zfill(2)
        if seq in self.index_cache:
            return self.index_cache[seq]
        path = self.preprocess_root / seq / "stage_c_cache_semantic_chunks" / "cache_index.jsonl"
        rows: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
        self.index_cache[seq] = rows
        return rows

    def asset(self, seq: str, frame: int) -> dict[str, Any] | None:
        seq = str(seq).zfill(2)
        rgb = _rgb_path(self.rgb_root, seq, int(frame))
        if rgb is None:
            return None
        for row in self._index(seq):
            start = int(row.get("start_frame", -1))
            end = int(row.get("end_frame", -1))
            if start <= int(frame) < end:
                sem = self.preprocess_root / seq / "stage_c_cache_semantic_chunks" / row["chunk"] / "masklet.pt"
                if not sem.is_file():
                    return None
                return {
                    "frame": int(frame),
                    "rgb_path": str(rgb),
                    "path": str(sem),
                    "local_frame": int(frame) - start,
                }
        return None


def _role_from_agg(agg: dict[str, Any]) -> dict[str, float | None]:
    road = _finite(agg.get("road_frac_mean"))
    static = _finite(agg.get("static_frac_mean"))
    dynamic = _finite(agg.get("dynamic_frac_mean"))
    low_conf = _finite(agg.get("low_conf_frac_mean"))
    dark = _finite(agg.get("dark_frac_mean"))
    vegetation = _finite(agg.get("vegetation_frac_mean"))
    semantic_edge = _finite(agg.get("semantic_boundary_density_mean"))
    bright = _finite(agg.get("bright_frac_mean"))

    def clipped_sum(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return float(max(0.0, min(1.0, sum(vals))))

    return {
        "stable": clipped_sum([road, static]),
        "harm": clipped_sum([dynamic, low_conf, dark]),
        "context": clipped_sum([vegetation, semantic_edge, bright]),
    }


def _role_series(frame_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in frame_rows:
        if not row.get("valid"):
            continue
        agg = {f"{key}_mean": value for key, value in row.items() if key not in {"frame", "valid"}}
        roles = _role_from_agg(agg)
        if all(roles.get(key) is not None for key in ("stable", "harm", "context")):
            out.append({key: float(roles[key]) for key in ("stable", "harm", "context")})  # type: ignore[arg-type]
    return out


def _temporal_range(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(max(vals) - min(vals))


def _semantic_features(
    *,
    resolver: SemanticAssetResolver,
    seq: str,
    frames: list[int],
    chunk_cache: dict[Path, Any],
) -> dict[str, Any]:
    assets = [resolver.asset(seq, frame) for frame in frames]
    assets = [asset for asset in assets if asset is not None]
    frame_rows: list[dict[str, Any]] = []
    for asset in assets:
        try:
            row = _frame_features(asset, chunk_cache)
        except Exception as exc:  # diagnostic rows must survive partial cache gaps
            row = {"frame": asset.get("frame"), "valid": False, "error": type(exc).__name__}
        frame_rows.append(row)
    agg = _aggregate_frame_features(frame_rows)
    roles = _role_from_agg(agg)
    role_rows = _role_series(frame_rows)
    role_ranges = {
        "stable_role_temporal_range": _temporal_range([row["stable"] for row in role_rows]),
        "harm_role_temporal_range": _temporal_range([row["harm"] for row in role_rows]),
        "context_role_temporal_range": _temporal_range([row["context"] for row in role_rows]),
    }
    return {
        "frames": frames,
        "semantic_asset_count": len(assets),
        "valid_frame_count": int(_finite(agg.get("valid_frame_count")) or 0),
        "agg": agg,
        "roles": roles,
        "role_ranges": role_ranges,
    }


def _sample_frames(row: dict[str, Any], family: str) -> list[int]:
    spec = FAMILY_SPECS[family]
    start = int(float(row[spec["start"]]))
    end = int(float(row[spec["end"]]))
    if family == "single":
        frames = [start, (start + end - 1) // 2, end - 1]
    elif family == "adjacent":
        boundary = int(float(row.get("boundary_frame") or ((start + end) // 2)))
        frames = [start, max(start, boundary - 1), min(end - 1, boundary), end - 1]
    else:
        total = max(1, end - start - 1)
        frames = [start + round(total * frac) for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    out: list[int] = []
    for frame in frames:
        frame = int(frame)
        if start <= frame < end and frame not in out:
            out.append(frame)
    return out


def _overlap_frames(row: dict[str, Any]) -> list[int]:
    start = int(float(row["pair_start_frame"]))
    end = int(float(row["pair_end_frame"]))
    boundary = int(float(row.get("boundary_frame") or ((start + end) // 2)))
    frames = [boundary - 3, boundary - 2, boundary - 1, boundary, boundary + 1, boundary + 2]
    return [frame for frame in frames if start <= frame < end]


def _split_overlap_agreement(
    *,
    resolver: SemanticAssetResolver,
    seq: str,
    row: dict[str, Any],
    chunk_cache: dict[Path, Any],
) -> float | None:
    start = int(float(row["pair_start_frame"]))
    end = int(float(row["pair_end_frame"]))
    boundary = int(float(row.get("boundary_frame") or ((start + end) // 2)))
    left = [frame for frame in [boundary - 3, boundary - 2, boundary - 1] if start <= frame < end]
    right = [frame for frame in [boundary, boundary + 1, boundary + 2] if start <= frame < end]
    if not left or not right:
        return None
    lf = _semantic_features(resolver=resolver, seq=seq, frames=left, chunk_cache=chunk_cache)["roles"]
    rf = _semantic_features(resolver=resolver, seq=seq, frames=right, chunk_cache=chunk_cache)["roles"]
    vals = []
    for key in ("stable", "harm", "context"):
        lval = _finite(lf.get(key))
        rval = _finite(rf.get(key))
        if lval is not None and rval is not None:
            vals.append(abs(lval - rval))
    if not vals:
        return None
    return float(max(0.0, 1.0 - sum(vals) / max(1, len(vals))))


def _current_scan_trajectories(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    specs: list[tuple[str, str, Path, str]] = []
    summary = _load_json(args.current_scan_summary)
    for row in summary.get("rows_ranked_by_ttt_window5_joint_sim3_rmse_m", []):
        seq = str(row.get("seq", "")).zfill(2)
        chunk_range = str(row.get("chunk_range", ""))
        root = Path(str(row.get("output_root", "")))
        path = root / "LW1_TTT_SEMANTIC_BASE" / f"{seq}.txt"
        if seq and chunk_range and path.is_file():
            specs.append(
                (
                    f"supp_current_v78_seq{seq}_chunks{chunk_range}",
                    seq,
                    path,
                    "supplemental_current_v78_replay",
                )
            )

    for row in _read_csv(args.current_target_dir / "bad_5chunk_window_table.csv"):
        seq = str(row.get("sequence", "")).zfill(2)
        case_id = str(row.get("window_chunks", ""))
        path = Path(str(row.get("trajectory", "")))
        if seq and case_id and path.is_file():
            specs.append(
                (
                    f"supp_current_v78_seq{seq}_chunks{case_id}",
                    seq,
                    path,
                    "supplemental_current_v78_replay",
                )
            )
    return specs


def _input_trajectories(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    specs: list[tuple[str, str, Path, str]] = []
    if args.h35_full.is_file():
        specs.append(("primary_kitti01_v53_h35_full", "01", args.h35_full, "primary_kitti01_v53_h35"))
    if args.h35_704.is_file():
        specs.append(("primary_kitti01_v53_h35_704", "01", args.h35_704, "primary_kitti01_v53_h35"))
    if not args.no_supplemental_current_v78:
        specs.extend(_current_scan_trajectories(args))

    seen: set[tuple[str, str, str, str]] = set()
    unique: list[tuple[str, str, Path, str]] = []
    for name, seq, path, scope in specs:
        key = (name, seq, str(path), scope)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, seq, path, scope))
    return unique


def _collect_bad_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rows = {"single": [], "adjacent": [], "five": []}
    errors: list[dict[str, str]] = []
    for name, seq, path, scope in _input_trajectories(args):
        try:
            single, adjacent, five, _summary = _evaluate_run(
                name=name,
                seq=seq,
                path=path,
                gt_root=args.gt_root,
                chunk_size=int(args.chunk_size),
                overlap=int(args.chunk_overlap),
                min_coverage=float(args.min_coverage),
            )
        except Exception as exc:
            errors.append({"name": name, "seq": seq, "path": str(path), "error": type(exc).__name__})
            continue
        for family, family_rows in (("single", single), ("adjacent", adjacent), ("five", five)):
            for row in family_rows:
                row["target_scope"] = scope
                row["source_name"] = name
                row["source_path"] = str(path)
            rows[family].extend(family_rows)
    rows["_errors"] = errors  # type: ignore[assignment]
    return rows


def _dedupe_and_select(
    rows: list[dict[str, Any]],
    *,
    family: str,
    top_k_primary: int,
    top_k_supplemental: int,
) -> list[dict[str, Any]]:
    spec = FAMILY_SPECS[family]
    metric = spec["metric"]
    by_scope_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        scope = str(row.get("target_scope", ""))
        seq = str(row.get("sequence", "")).zfill(2)
        cid = str(row.get(spec["id"], ""))
        key = (scope, seq, cid)
        old = by_scope_key.get(key)
        old_v = _finite(old.get(metric)) if old else None
        new_v = _finite(row.get(metric))
        if old is None or (new_v is not None and (old_v is None or new_v > old_v)):
            new_row = dict(row)
            contexts = []
            if old and old.get("source_contexts"):
                try:
                    contexts = json.loads(str(old["source_contexts"]))
                except json.JSONDecodeError:
                    contexts = []
            contexts.append({"source_name": row.get("source_name"), "source_path": row.get("source_path")})
            new_row["source_contexts"] = json.dumps(contexts, ensure_ascii=False)
            new_row["source_context_count"] = len(contexts)
            by_scope_key[key] = new_row
        elif old is not None:
            contexts = []
            if old.get("source_contexts"):
                try:
                    contexts = json.loads(str(old["source_contexts"]))
                except json.JSONDecodeError:
                    contexts = []
            contexts.append({"source_name": row.get("source_name"), "source_path": row.get("source_path")})
            old["source_contexts"] = json.dumps(contexts, ensure_ascii=False)
            old["source_context_count"] = len(contexts)

    selected: list[dict[str, Any]] = []
    for scope, limit in (
        ("primary_kitti01_v53_h35", top_k_primary),
        ("supplemental_current_v78_replay", top_k_supplemental),
    ):
        scoped = [row for row in by_scope_key.values() if row.get("target_scope") == scope]
        scoped.sort(key=lambda row: _finite(row.get(metric)) if _finite(row.get(metric)) is not None else -math.inf, reverse=True)
        selected.extend(scoped[: max(0, int(limit))])
    return selected


def _common_target_fields(row: dict[str, Any], family: str, rank: int) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    return {
        "target_rank": rank,
        "target_scope": row.get("target_scope"),
        "source_name": row.get("source_name"),
        "source_path": row.get("source_path"),
        "source_context_count": row.get("source_context_count", 1),
        "source_contexts": row.get("source_contexts", ""),
        "sequence": str(row.get("sequence", "")).zfill(2),
        "run": row.get("run"),
        "metric_used_for_ranking": spec["metric"],
        "metric_value": _finite(row.get(spec["metric"])),
        "semantic_source": "dense_label_confidence + RGB_luminance + LoGeR_trajectory_geometry",
        "semantic_role_proxy_note": (
            "stable=road/ground+static labels; harm=dynamic+low-confidence+darkness; "
            "context=vegetation+semantic-boundary+bright regions"
        ),
        "unavailable_fields_note": (
            "RADIO/RADSeg object topology and TTT post_zp/update fields are not present "
            "in these cached diagnostic artifacts; unavailable values are explicit."
        ),
    }


def _build_short_row(
    row: dict[str, Any],
    rank: int,
    resolver: SemanticAssetResolver,
    chunk_cache: dict[Path, Any],
) -> dict[str, Any]:
    seq = str(row.get("sequence", "")).zfill(2)
    features = _semantic_features(
        resolver=resolver,
        seq=seq,
        frames=_sample_frames(row, "single"),
        chunk_cache=chunk_cache,
    )
    agg = features["agg"]
    roles = features["roles"]
    out = _common_target_fields(row, "single", rank)
    out.update(
        {
            "chunk_id": row.get("chunk_id"),
            "frame_start": row.get("chunk_start_frame"),
            "frame_end": row.get("chunk_end_frame"),
            "sample_frames": features["frames"],
            "valid_semantic_frames": features["valid_frame_count"],
            "local_sim3_ate": _finite(row.get("local_sim3_rmse_m")),
            "head_to_tail": "unavailable_existing_single_chunk_table",
            "scale_cv": "unavailable_existing_single_chunk_table",
            "intra_scale_variance": "unavailable_existing_single_chunk_table",
            "D_geo_tail": "unavailable_existing_single_chunk_table",
            "stable_read_mass": roles.get("stable"),
            "harm_read_mass": roles.get("harm"),
            "context_read_mass": roles.get("context"),
            "L07_layout_strength": "not_run_phase2_read_candidate_yet",
            "L13_value_action_strength": "not_run_phase2_read_candidate_yet",
            "semantic_confidence_mean": _finite(agg.get("confidence_mean_mean")),
            "thing_moving_ratio": _finite(agg.get("dynamic_frac_mean")),
            "thing_static_ratio": _finite(agg.get("static_frac_mean")),
            "lowtrust_stuff_ratio": _finite(agg.get("low_conf_frac_mean")),
            "RADIO_boundary_ratio": "unavailable_no_radio_topology_cache_in_phase1",
            "semantic_boundary_density_mean": _finite(agg.get("semantic_boundary_density_mean")),
            "road_frac_mean": _finite(agg.get("road_frac_mean")),
            "dark_frac_mean": _finite(agg.get("dark_frac_mean")),
            "target_reason": (
                "high single-chunk local_sim3_rmse with semantic role diagnosis for READ/global-attention targeting"
            ),
        }
    )
    return out


def _build_mid_row(
    row: dict[str, Any],
    rank: int,
    resolver: SemanticAssetResolver,
    chunk_cache: dict[Path, Any],
) -> dict[str, Any]:
    seq = str(row.get("sequence", "")).zfill(2)
    features = _semantic_features(
        resolver=resolver,
        seq=seq,
        frames=_overlap_frames(row),
        chunk_cache=chunk_cache,
    )
    agg = features["agg"]
    roles = features["roles"]
    agreement = _split_overlap_agreement(resolver=resolver, seq=seq, row=row, chunk_cache=chunk_cache)
    out = _common_target_fields(row, "adjacent", rank)
    out.update(
        {
            "prev_chunk": row.get("start_chunk_id"),
            "curr_chunk": row.get("end_chunk_id"),
            "chunk_pair": row.get("chunk_pair"),
            "frame_start": row.get("pair_start_frame"),
            "boundary_frame": row.get("boundary_frame"),
            "frame_end": row.get("pair_end_frame"),
            "sample_frames": features["frames"],
            "valid_semantic_frames": features["valid_frame_count"],
            "future_after_overlap": _finite(row.get("tail3_to_future_from_boundary_sim3_rmse_m")),
            "boundary_jump": _finite(row.get("boundary_step_error_global_sim3_m")),
            "raw_overlap_residual": _finite(row.get("tail3_to_head3_sim3_rmse_m")),
            "stable_overlap_mass": roles.get("stable"),
            "harm_overlap_mass": roles.get("harm"),
            "context_overlap_mass": roles.get("context"),
            "V_L26_selected_minus_random": "not_run_phase3_swa_candidate_yet",
            "K_L26_selected_minus_random": "not_run_phase3_swa_candidate_yet",
            "selected_source_quality": _finite(agg.get("confidence_mean_mean")),
            "semantic_agreement": agreement,
            "RADIO_same_object_ratio": "unavailable_no_radio_topology_cache_in_phase1",
            "road_edge_confidence_mean": _finite(agg.get("road_edge_confidence_mean_mean")),
            "semantic_boundary_density_mean": _finite(agg.get("semantic_boundary_density_mean")),
            "target_reason": (
                "high adjacent overlap-to-future error with overlap semantic role diagnosis for SWA handoff targeting"
            ),
        }
    )
    return out


def _mean_finite(values: list[Any]) -> float | None:
    vals = [_finite(value) for value in values]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _build_long_row(
    row: dict[str, Any],
    rank: int,
    resolver: SemanticAssetResolver,
    chunk_cache: dict[Path, Any],
) -> dict[str, Any]:
    seq = str(row.get("sequence", "")).zfill(2)
    features = _semantic_features(
        resolver=resolver,
        seq=seq,
        frames=_sample_frames(row, "five"),
        chunk_cache=chunk_cache,
    )
    agg = features["agg"]
    roles = features["roles"]
    ranges = features["role_ranges"]
    shadow_exposure = _mean_finite(
        [
            _finite(agg.get("dark_frac_temporal_range")),
            _finite(agg.get("luminance_mean_temporal_range")),
            _finite(agg.get("bright_frac_temporal_range")),
        ]
    )
    harm_shift = _mean_finite(
        [
            _finite(agg.get("dark_frac_temporal_range")),
            _finite(agg.get("low_conf_frac_temporal_range")),
            _finite(agg.get("road_center_range_temporal_range")),
            _finite(agg.get("semantic_boundary_density_temporal_range")),
            _finite(agg.get("road_edge_confidence_mean_temporal_range")),
            ranges.get("harm_role_temporal_range"),
        ]
    )
    low_observability = _mean_finite(
        [
            _finite(agg.get("dark_frac_mean")),
            _finite(agg.get("low_conf_frac_mean")),
            1.0 - _finite(agg.get("confidence_mean_mean"))
            if _finite(agg.get("confidence_mean_mean")) is not None
            else None,
        ]
    )
    road_edge_continuity = None
    road_edge_conf = _finite(agg.get("road_edge_confidence_mean_mean"))
    road_edge_var = _finite(agg.get("road_edge_confidence_mean_temporal_range"))
    if road_edge_conf is not None and road_edge_var is not None:
        road_edge_continuity = float(road_edge_conf - road_edge_var)
    stable_range = ranges.get("stable_role_temporal_range")
    stable_continuity = None if stable_range is None else float(max(0.0, 1.0 - stable_range))
    out = _common_target_fields(row, "five", rank)
    out.update(
        {
            "window_start_chunk": row.get("start_chunk_id"),
            "window_end_chunk": row.get("end_chunk_id"),
            "window_chunks": row.get("window_chunks"),
            "frame_start": row.get("window_start_frame"),
            "frame_end": row.get("window_end_frame"),
            "sample_frames": features["frames"],
            "valid_semantic_frames": features["valid_frame_count"],
            "window5_joint_sim3_rmse": _finite(row.get("window5_joint_sim3_rmse_m")),
            "subchunk_scale_cv": _finite(row.get("window5_subchunk_scale_cv")),
            "downstream_future": "unavailable_existing_five_chunk_table",
            "stable_regime_continuity": stable_continuity,
            "harm_regime_shift_score": harm_shift,
            "low_observability_score": low_observability,
            "road_edge_continuity": road_edge_continuity,
            "shadow_exposure_change": shadow_exposure,
            "TTT_update_conflict": "not_run_phase4_ttt_candidate_yet",
            "post_zp_delta": "not_run_phase4_ttt_candidate_yet",
            "stable_role_mass": roles.get("stable"),
            "harm_role_mass": roles.get("harm"),
            "context_role_mass": roles.get("context"),
            "road_center_range_temporal_range": _finite(agg.get("road_center_range_temporal_range")),
            "semantic_boundary_density_temporal_range": _finite(agg.get("semantic_boundary_density_temporal_range")),
            "target_reason": (
                "high five-window joint Sim3/scale drift with long-window shadow/exposure/corridor/road-edge diagnosis"
            ),
        }
    )
    return out


SHORT_FIELDS = [
    "target_rank",
    "target_scope",
    "sequence",
    "run",
    "source_name",
    "chunk_id",
    "frame_start",
    "frame_end",
    "local_sim3_ate",
    "head_to_tail",
    "scale_cv",
    "intra_scale_variance",
    "D_geo_tail",
    "stable_read_mass",
    "harm_read_mass",
    "context_read_mass",
    "L07_layout_strength",
    "L13_value_action_strength",
    "semantic_confidence_mean",
    "thing_moving_ratio",
    "thing_static_ratio",
    "lowtrust_stuff_ratio",
    "RADIO_boundary_ratio",
    "semantic_boundary_density_mean",
    "road_frac_mean",
    "dark_frac_mean",
    "valid_semantic_frames",
    "sample_frames",
    "target_reason",
    "semantic_source",
    "semantic_role_proxy_note",
    "unavailable_fields_note",
    "source_path",
    "source_context_count",
    "source_contexts",
]

MID_FIELDS = [
    "target_rank",
    "target_scope",
    "sequence",
    "run",
    "source_name",
    "prev_chunk",
    "curr_chunk",
    "chunk_pair",
    "frame_start",
    "boundary_frame",
    "frame_end",
    "future_after_overlap",
    "boundary_jump",
    "raw_overlap_residual",
    "stable_overlap_mass",
    "harm_overlap_mass",
    "context_overlap_mass",
    "V_L26_selected_minus_random",
    "K_L26_selected_minus_random",
    "selected_source_quality",
    "semantic_agreement",
    "RADIO_same_object_ratio",
    "road_edge_confidence_mean",
    "semantic_boundary_density_mean",
    "valid_semantic_frames",
    "sample_frames",
    "target_reason",
    "semantic_source",
    "semantic_role_proxy_note",
    "unavailable_fields_note",
    "source_path",
    "source_context_count",
    "source_contexts",
]

LONG_FIELDS = [
    "target_rank",
    "target_scope",
    "sequence",
    "run",
    "source_name",
    "window_start_chunk",
    "window_end_chunk",
    "window_chunks",
    "frame_start",
    "frame_end",
    "window5_joint_sim3_rmse",
    "subchunk_scale_cv",
    "downstream_future",
    "stable_regime_continuity",
    "harm_regime_shift_score",
    "low_observability_score",
    "road_edge_continuity",
    "shadow_exposure_change",
    "TTT_update_conflict",
    "post_zp_delta",
    "stable_role_mass",
    "harm_role_mass",
    "context_role_mass",
    "road_center_range_temporal_range",
    "semantic_boundary_density_temporal_range",
    "valid_semantic_frames",
    "sample_frames",
    "target_reason",
    "semantic_source",
    "semantic_role_proxy_note",
    "unavailable_fields_note",
    "source_path",
    "source_context_count",
    "source_contexts",
]


def _semantic_gate(rows: list[dict[str, Any]], required_fields: list[str]) -> bool:
    if len(rows) < 5:
        return False
    for row in rows:
        if int(row.get("valid_semantic_frames") or 0) < 1:
            return False
        for field in required_fields:
            if row.get(field) in (None, "", "unavailable_existing_single_chunk_table"):
                return False
    return True


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolver = SemanticAssetResolver(args.preprocess_root, args.rgb_root)
    chunk_cache: dict[Path, Any] = {}

    raw = _collect_bad_rows(args)
    selected = {
        family: _dedupe_and_select(
            raw[family],
            family=family,
            top_k_primary=int(args.top_k_primary),
            top_k_supplemental=int(args.top_k_supplemental),
        )
        for family in ("single", "adjacent", "five")
    }

    short_rows = [
        _build_short_row(row, idx, resolver, chunk_cache)
        for idx, row in enumerate(selected["single"], start=1)
    ]
    mid_rows = [
        _build_mid_row(row, idx, resolver, chunk_cache)
        for idx, row in enumerate(selected["adjacent"], start=1)
    ]
    long_rows = [
        _build_long_row(row, idx, resolver, chunk_cache)
        for idx, row in enumerate(selected["five"], start=1)
    ]

    _write_csv(args.out_dir / "single_chunk_semantic_read_targets.csv", short_rows, SHORT_FIELDS)
    _write_csv(args.out_dir / "adjacent_semantic_handoff_targets.csv", mid_rows, MID_FIELDS)
    _write_csv(args.out_dir / "five_window_semantic_write_targets.csv", long_rows, LONG_FIELDS)

    def scope_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            scope = str(row.get("target_scope", ""))
            out[scope] = out.get(scope, 0) + 1
        return out

    gate_checks = {
        "short_targets_ge_5": len(short_rows) >= 5,
        "mid_targets_ge_5": len(mid_rows) >= 5,
        "long_targets_ge_5": len(long_rows) >= 5,
        "short_primary_kitti01_targets_ge_5": scope_counts(short_rows).get("primary_kitti01_v53_h35", 0) >= 5,
        "mid_primary_kitti01_targets_ge_5": scope_counts(mid_rows).get("primary_kitti01_v53_h35", 0) >= 5,
        "long_primary_kitti01_targets_ge_5": scope_counts(long_rows).get("primary_kitti01_v53_h35", 0) >= 5,
        "short_semantic_diagnosis_present": _semantic_gate(
            short_rows, ["stable_read_mass", "harm_read_mass", "context_read_mass", "semantic_confidence_mean"]
        ),
        "mid_semantic_diagnosis_present": _semantic_gate(
            mid_rows, ["stable_overlap_mass", "harm_overlap_mass", "context_overlap_mass", "semantic_agreement"]
        ),
        "long_semantic_diagnosis_present": _semantic_gate(
            long_rows, ["stable_regime_continuity", "harm_regime_shift_score", "shadow_exposure_change"]
        ),
    }
    summary = {
        "schema": "acl2_v79_phase1_semantic_current_bad_target_ledger_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "phase1_gate_pass": all(gate_checks.values()),
        "out_dir": str(args.out_dir),
        "input_trajectories": [
            {"name": name, "sequence": seq, "path": str(path), "target_scope": scope}
            for name, seq, path, scope in _input_trajectories(args)
        ],
        "trajectory_eval_errors": raw.get("_errors", []),
        "counts": {
            "short": len(short_rows),
            "mid": len(mid_rows),
            "long": len(long_rows),
            "short_by_scope": scope_counts(short_rows),
            "mid_by_scope": scope_counts(mid_rows),
            "long_by_scope": scope_counts(long_rows),
        },
        "gate_checks": gate_checks,
        "outputs": {
            "single_chunk_semantic_read_targets": str(args.out_dir / "single_chunk_semantic_read_targets.csv"),
            "adjacent_semantic_handoff_targets": str(args.out_dir / "adjacent_semantic_handoff_targets.csv"),
            "five_window_semantic_write_targets": str(args.out_dir / "five_window_semantic_write_targets.csv"),
        },
        "claim_boundary": [
            "Phase1 only mines targets and semantic diagnoses.",
            "It does not test READ/SWA/TTT actions or controls.",
            "Supplemental current-v78 replay rows are not KITTI01 success evidence.",
            "RADIO/RADSeg topology fields remain unavailable unless a later phase supplies those caches.",
        ],
    }
    _write_json(args.out_dir / "phase1_semantic_target_ledger_summary.json", summary)
    notes = [
        "# v79 Phase1 Semantic Feature Source Notes",
        "",
        "- `primary_kitti01_v53_h35` rows come from v53/H35 KITTI01 full and 704F trajectories when present.",
        "- `supplemental_current_v78_replay` rows come from v78 current replay scan/target trajectories and are kept separate.",
        "- Role masses are deterministic proxies from cached dense semantic labels/confidence and RGB luminance, not learned parameters.",
        "- Unavailable RADIO/RADSeg/TTT update fields are written explicitly as unavailable/not-run.",
        "- This artifact is diagnostic-only and cannot support a v79 method-success claim by itself.",
        "",
    ]
    (args.out_dir / "semantic_feature_source_notes.md").write_text("\n".join(notes), encoding="utf-8")

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
