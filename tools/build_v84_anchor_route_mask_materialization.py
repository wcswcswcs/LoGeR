#!/usr/bin/env python3
"""Materialize v84 candidate anchor route masks from overlap artifacts.

This builds query/source patch-position masks needed for a later v84
anchor-specific SWA route dump. It is a feasibility artifact only: the model
does not yet consume these CSV masks, and full-token indices require adding the
runtime model's patch_start_idx offset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v84_geometry_leverage_refinement import (  # noqa: E402
    safe_float,
    token_features,
    variant_match,
    weak_eligible,
    quantile,
)


DEFAULT_TOKENS = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates/ruler_candidate_tokens.csv"
)
DEFAULT_PAIR_AUDIT = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_audit/support_expansion_audit_by_pair.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase15_anchor_route_mask_materialization")

PATCH_GRID = (19, 66)
PATCH_TOKEN_COUNT = PATCH_GRID[0] * PATCH_GRID[1]
MODEL_PATCH_START_IDX_NOTE = "LoGeR PI3 patch_start_idx is 6 in current code; runtime hook must add the live model offset."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--pair-audit", type=Path, default=DEFAULT_PAIR_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant to materialize. Defaults to current, best good-FPR, weak, and topq90 controls.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")).zfill(2), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def patch_indices(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = coords[:, 0].float()
    x = coords[:, 1].float()
    py = torch.clamp((y / (376.0 / PATCH_GRID[0])).floor().long(), 0, PATCH_GRID[0] - 1)
    px = torch.clamp((x / (1408.0 / PATCH_GRID[1])).floor().long(), 0, PATCH_GRID[1] - 1)
    flat = py * PATCH_GRID[1] + px
    return py, px, flat


def build_thresholds(tokens: list[dict[str, str]], variants: list[str]) -> dict[str, float]:
    weak_scores: list[float] = []
    geom_scores: list[float] = []
    for row in tokens:
        feat = token_features(row)
        if weak_eligible(feat):
            weak_scores.append(feat["structural_edge_score"])
        if (
            feat["parallax"] >= 0.05
            and feat["nondeg"] >= 0.05
            and feat["overlap"] >= 0.10
            and feat["risk"] < 0.70
            and feat["far"] <= 0.90
            and feat["memory_factor"] > 0.0
        ):
            geom_scores.append(feat["geometry_memory_score"])
    thresholds: dict[str, float] = {}
    for variant in variants:
        if variant.startswith("structural_edge_topq"):
            q = float(variant.split("topq", 1)[1]) / 100.0
            thresholds[variant] = quantile(weak_scores, q) or math.inf
        elif variant.startswith("geometry_memory_topq"):
            tail = variant.split("topq", 1)[1].split("_", 1)[0]
            q = float(tail) / 100.0
            thresholds[variant] = quantile(geom_scores, q) or math.inf
    return thresholds


def selected_for_variant(
    tokens: list[dict[str, str]],
    *,
    variant: str,
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in tokens:
        feat = token_features(row)
        if variant_match(variant, row, feat, thresholds):
            out.append({**row, **{f"feat_{key}": value for key, value in feat.items()}})
    return out


def positions_for_pair(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_path_patch: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        source_path = str(row.get("source_path") or "").strip()
        py = int(float(row.get("patch_y") or 0))
        px = int(float(row.get("patch_x") or 0))
        by_path_patch[(source_path, py, px)].append(row)

    position_keys: set[tuple[str, int, int, int]] = set()
    position_rows: list[dict[str, Any]] = []
    source_positions_by_token: list[int] = []
    query_positions_by_token: list[int] = []
    missing_paths = 0
    load_errors = 0
    cached: dict[str, Any] = {}

    for (source_path, py, px), rows in by_path_patch.items():
        if not source_path:
            missing_paths += len(rows)
            continue
        if source_path not in cached:
            try:
                cached[source_path] = torch_load(Path(source_path))
            except Exception as exc:  # noqa: BLE001
                cached[source_path] = exc
        payload = cached[source_path]
        if isinstance(payload, Exception):
            load_errors += len(rows)
            continue
        curr_coords = payload.get("curr_pixel_coords")
        prev_coords = payload.get("prev_pixel_coords")
        curr_frame_ids = payload.get("curr_frame_ids")
        prev_frame_ids = payload.get("prev_frame_ids")
        curr_start = int(payload.get("curr_start_frame", 0))
        prev_start = int(payload.get("prev_start_frame", 0))
        if not all(torch.is_tensor(v) for v in [curr_coords, prev_coords, curr_frame_ids, prev_frame_ids]):
            load_errors += len(rows)
            continue
        curr_py, curr_px, _ = patch_indices(curr_coords)
        prev_py, prev_px, prev_flat = patch_indices(prev_coords)
        match = (curr_py == int(py)) & (curr_px == int(px))
        idx = torch.nonzero(match, as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            source_positions_by_token.extend([0] * len(rows))
            query_positions_by_token.extend([0] * len(rows))
            continue
        q_local = (curr_frame_ids[idx].long() - curr_start).clamp_min(0)
        s_local = (prev_frame_ids[idx].long() - prev_start).clamp_min(0)
        q_flat = curr_py[idx] * PATCH_GRID[1] + curr_px[idx]
        s_flat = prev_flat[idx]
        q_positions = sorted({(int(f), int(p)) for f, p in zip(q_local.tolist(), q_flat.tolist())})
        s_positions = sorted({(int(f), int(p)) for f, p in zip(s_local.tolist(), s_flat.tolist())})
        source_positions_by_token.extend([len(s_positions)] * len(rows))
        query_positions_by_token.extend([len(q_positions)] * len(rows))
        for local_frame, patch_flat in q_positions:
            key = ("query", local_frame, int(patch_flat), int(patch_flat))
            if key not in position_keys:
                position_keys.add(key)
                position_rows.append(
                    {
                        "side": "query",
                        "local_frame": local_frame,
                        "patch_token_index": int(patch_flat),
                        "patch_y": int(patch_flat) // PATCH_GRID[1],
                        "patch_x": int(patch_flat) % PATCH_GRID[1],
                        "model_token_index_requires_patch_start_offset": True,
                        "model_patch_start_idx_note": MODEL_PATCH_START_IDX_NOTE,
                    }
                )
        for local_frame, patch_flat in s_positions:
            key = ("source", local_frame, int(patch_flat), int(patch_flat))
            if key not in position_keys:
                position_keys.add(key)
                position_rows.append(
                    {
                        "side": "source",
                        "local_frame": local_frame,
                        "patch_token_index": int(patch_flat),
                        "patch_y": int(patch_flat) // PATCH_GRID[1],
                        "patch_x": int(patch_flat) % PATCH_GRID[1],
                        "model_token_index_requires_patch_start_offset": True,
                        "model_patch_start_idx_note": MODEL_PATCH_START_IDX_NOTE,
                    }
                )

    query_unique = sum(1 for row in position_rows if row["side"] == "query")
    source_unique = sum(1 for row in position_rows if row["side"] == "source")
    source_frames = sorted({int(row["local_frame"]) for row in position_rows if row["side"] == "source"})
    query_frames = sorted({int(row["local_frame"]) for row in position_rows if row["side"] == "query"})
    summary = {
        "selected_token_rows": len(selected),
        "query_unique_patch_positions": query_unique,
        "source_unique_patch_positions": source_unique,
        "query_local_frames": query_frames,
        "source_local_frames": source_frames,
        "query_positions_per_selected_token_mean": (
            sum(query_positions_by_token) / len(query_positions_by_token) if query_positions_by_token else None
        ),
        "source_positions_per_selected_token_mean": (
            sum(source_positions_by_token) / len(source_positions_by_token) if source_positions_by_token else None
        ),
        "missing_source_path_rows": missing_paths,
        "load_error_rows": load_errors,
    }
    return position_rows, summary


def build_report(summary: Mapping[str, Any], variant_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase15 Anchor Route Mask Materialization",
        "",
        f"- Materialization feasible: `{summary['materialization_feasible']}`",
        f"- Runtime external mask hook available: `{summary['runtime_external_mask_hook_available']}`",
        f"- Runtime action allowed: `{summary['runtime_action_allowed']}`",
        "",
        "| variant | selected tokens | positive pairs | query positions | source positions | source density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            "| {variant} | {tokens} | {pairs} | {query} | {source} | {density:.6f} |".format(
                variant=row["variant"],
                tokens=row["selected_token_rows"],
                pairs=row["positive_pair_rows"],
                query=row["query_unique_patch_positions"],
                source=row["source_unique_patch_positions"],
                density=row["source_patch_density_vs_3frame_overlap"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The CSVs materialize patch-grid source/query positions only. A runtime route dump would still need an explicit external-mask hook and live model token offset handling.",
            "Because Phase14 labelled separation failed, these masks are not sufficient to justify a runtime action by themselves.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tokens = read_csv(args.tokens)
    pair_rows = read_csv(args.pair_audit)
    variants = args.variant or [
        "current_role_anchor",
        "structural_edge_overlap_strict_fixed",
        "weak_medium_conf_nonzero_fixed",
        "structural_edge_topq90",
        "geometry_memory_topq90_no_semantic_control",
    ]
    thresholds = build_thresholds(tokens, variants)

    pairs_by_key = {pair_key(row): row for row in pair_rows}
    by_variant_rows: list[dict[str, Any]] = []
    all_position_rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    selected_case_counter: Counter[str] = Counter()

    for variant in variants:
        selected = selected_for_variant(tokens, variant=variant, thresholds=thresholds)
        by_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_pair[pair_key(row)].append(row)
        variant_position_count = 0
        variant_query_unique = 0
        variant_source_unique = 0
        positive_pairs = 0
        total_selected = 0
        rows_missing_path = 0
        rows_load_error = 0
        for key, group in sorted(by_pair.items()):
            pair_meta = pairs_by_key.get(key, {})
            positions, position_summary = positions_for_pair(group)
            positive_pairs += int(bool(group))
            total_selected += len(group)
            rows_missing_path += int(position_summary.get("missing_source_path_rows") or 0)
            rows_load_error += int(position_summary.get("load_error_rows") or 0)
            variant_query_unique += int(position_summary.get("query_unique_patch_positions") or 0)
            variant_source_unique += int(position_summary.get("source_unique_patch_positions") or 0)
            selected_case_counter[str(pair_meta.get("case_type") or group[0].get("case_type") or "")] += len(group)
            by_variant_rows.append(
                {
                    "variant": variant,
                    "seq": key[0],
                    "prev_chunk": key[1],
                    "curr_chunk": key[2],
                    "case_type": pair_meta.get("case_type") or group[0].get("case_type"),
                    "base_case_type": pair_meta.get("base_case_type") or group[0].get("base_case_type"),
                    "support_expansion_label_scope": pair_meta.get("support_expansion_label_scope"),
                    "selected_token_rows": len(group),
                    **position_summary,
                }
            )
            for pos in positions:
                all_position_rows.append(
                    {
                        "variant": variant,
                        "seq": key[0],
                        "prev_chunk": key[1],
                        "curr_chunk": key[2],
                        **pos,
                    }
                )
            variant_position_count += len(positions)
        overlap_denominator = max(1, 3 * PATCH_TOKEN_COUNT)
        variant_summaries.append(
            {
                "variant": variant,
                "threshold": thresholds.get(variant),
                "selected_token_rows": total_selected,
                "positive_pair_rows": positive_pairs,
                "query_unique_patch_positions": variant_query_unique,
                "source_unique_patch_positions": variant_source_unique,
                "position_rows": variant_position_count,
                "source_patch_density_vs_3frame_overlap": variant_source_unique / overlap_denominator,
                "missing_source_path_rows": rows_missing_path,
                "load_error_rows": rows_load_error,
                "materialized": bool(variant_source_unique > 0 and rows_missing_path == 0 and rows_load_error == 0),
            }
        )

    materialization_feasible = bool(variant_summaries) and all(row["materialized"] for row in variant_summaries)
    summary = {
        "schema": "acl2_v84_anchor_route_mask_materialization_v1",
        "materialization_feasible": materialization_feasible,
        "runtime_external_mask_hook_available": False,
        "runtime_action_allowed": False,
        "runtime_action_blocker": "Phase14 labelled separation failed and no v84 external anchor-mask runtime hook exists yet",
        "variants": variants,
        "thresholds": thresholds,
        "patch_grid": list(PATCH_GRID),
        "patch_token_count": PATCH_TOKEN_COUNT,
        "model_patch_start_idx_note": MODEL_PATCH_START_IDX_NOTE,
        "variant_summaries": variant_summaries,
        "selected_case_type_counts": {k: int(v) for k, v in sorted(selected_case_counter.items())},
        "limitations": [
            "Positions are patch-grid indices only; runtime must add patch_start_idx and match live token layout.",
            "Source masks are inferred from raw overlap correspondences, not from a model-run attention dump.",
            "This artifact enables a future v84 anchor-specific route dump but does not itself prove route usage.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "anchor_route_mask_materialization_by_pair.csv", by_variant_rows)
    write_csv(args.out_dir / "anchor_route_mask_positions.csv", all_position_rows)
    write_csv(args.out_dir / "anchor_route_mask_variant_summary.csv", variant_summaries)
    write_json(args.out_dir / "anchor_route_mask_materialization_summary.json", summary)
    (args.out_dir / "anchor_route_mask_materialization_report.md").write_text(
        build_report(summary, variant_summaries), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "materialization_feasible": materialization_feasible,
                "runtime_external_mask_hook_available": False,
                "runtime_action_allowed": False,
                "variants": variants,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
