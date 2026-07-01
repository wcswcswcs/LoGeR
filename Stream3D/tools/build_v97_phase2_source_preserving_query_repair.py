#!/usr/bin/env python3
"""Build v97 Phase2 source-preserving adaptive query repair views."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase2_source_preserving_query_repair"
RUN_ID = "v97_phase2_source_preserving_query_repair"
DEFAULT_OUT_BASE = ROOT / "outputs/audit"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row.get("scene_id", ""), row.get("window_id", ""), row.get("chunk_id", ""), row.get("frame_id", ""))


def _repair_row(row: dict[str, str], *, repair_variant: str, source_variant: str, ordinal: int, policy: str) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["variant_id"] = repair_variant
    out["query_id"] = f"{repair_variant}:repair{ordinal:05d}:{row.get('query_id', '')}"
    out["query_repair_source_variant_id"] = source_variant
    out["query_repair_policy"] = policy
    out["query_selection_uses_gt"] = False
    out["uses_future"] = False
    return out


def build_one_root(input_root: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    input_query = input_root / "query_plan_rows.csv"
    if not input_query.exists():
        raise FileNotFoundError(input_query)
    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with input_query.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            variant = row.get("variant_id", "")
            if variant in {args.base_variant, args.edge_variant}:
                grouped[_key(row)][variant].append(row)
    repair_fields = list(fieldnames)
    for field in ["query_repair_source_variant_id", "query_repair_policy"]:
        if field not in repair_fields:
            repair_fields.append(field)
    output_root.mkdir(parents=True, exist_ok=True)
    out_query = output_root / "query_plan_rows.csv"
    total_rows = 0
    frame_count = 0
    source_variant_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    missing_base_count = 0
    missing_edge_count = 0
    with out_query.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=repair_fields)
        writer.writeheader()
        for key in sorted(grouped):
            variants = grouped[key]
            base_rows = list(variants.get(args.base_variant, []))
            edge_rows = list(variants.get(args.edge_variant, []))
            if not base_rows:
                missing_base_count += 1
            if not edge_rows:
                missing_edge_count += 1
            frame_count += 1
            selected: list[tuple[str, dict[str, str]]] = []
            selected.extend((args.base_variant, row) for row in base_rows[: int(args.base_limit_per_frame)])
            edge_priority = {"boundary": 0, "conflict": 1, "semantic_gradient": 2, "interior": 3, "uniform": 4}
            edge_sorted = sorted(
                edge_rows,
                key=lambda row: (
                    edge_priority.get(row.get("query_stratum", ""), 9),
                    -float(row.get("importance_weight", "0") or 0.0),
                    row.get("query_id", ""),
                ),
            )
            selected.extend((args.edge_variant, row) for row in edge_sorted[: int(args.edge_limit_per_frame)])
            for ordinal, (source_variant, row) in enumerate(selected):
                out = _repair_row(
                    row,
                    repair_variant=args.repair_variant,
                    source_variant=source_variant,
                    ordinal=total_rows + ordinal,
                    policy=args.policy_name,
                )
                writer.writerow({field: out.get(field, "") for field in repair_fields})
                source_variant_counts[source_variant] += 1
                stratum_counts[str(out.get("query_stratum", ""))] += 1
            total_rows += len(selected)
    summary = {
        "schema": "stream4d_v97_phase2_source_preserving_query_repair_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_root": _rel(input_root),
        "output_root": _rel(output_root),
        "input_query_plan_rows": _rel(input_query),
        "query_plan_rows": _rel(out_query),
        "repair_variant": args.repair_variant,
        "base_variant": args.base_variant,
        "edge_variant": args.edge_variant,
        "base_limit_per_frame": int(args.base_limit_per_frame),
        "edge_limit_per_frame": int(args.edge_limit_per_frame),
        "target_query_count_per_frame": int(args.base_limit_per_frame) + int(args.edge_limit_per_frame),
        "frame_key_count": frame_count,
        "query_row_count": total_rows,
        "source_variant_counts": dict(sorted(source_variant_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "missing_base_frame_count": missing_base_count,
        "missing_edge_frame_count": missing_edge_count,
        "policy_name": args.policy_name,
        "metric_scope": "query_repair_view",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_base = _project(args.output_base)
    input_roots = [_project(part.strip()) for part in args.input_roots.split(",") if part.strip()]
    segment_rows = []
    for input_root in input_roots:
        suffix = input_root.name.replace("v97_phase1_query_planner_", "")
        output_root = out_base / f"v97_phase1_query_planner_{suffix}_{args.repair_variant}"
        segment_rows.append(build_one_root(input_root, output_root, args))
    manifest_root = out_base / f"v97_phase1_query_planner_{args.repair_variant}_manifest"
    manifest = {
        "schema": "stream4d_v97_phase2_source_preserving_query_repair_manifest_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repair_variant": args.repair_variant,
        "segments": segment_rows,
        "segment_count": len(segment_rows),
        "query_row_count": sum(int(row["query_row_count"]) for row in segment_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(manifest_root / "summary.json", manifest)
    print(json.dumps({"manifest_root": _rel(manifest_root), "segment_count": len(segment_rows), "query_row_count": manifest["query_row_count"]}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", required=True)
    parser.add_argument("--output-base", default=str(DEFAULT_OUT_BASE))
    parser.add_argument("--repair-variant", default="Q3_source_preserve2048")
    parser.add_argument("--base-variant", default="Q1_uniform1024")
    parser.add_argument("--edge-variant", default="Q4_boundary_conflict1024")
    parser.add_argument("--base-limit-per-frame", type=int, default=1024)
    parser.add_argument("--edge-limit-per-frame", type=int, default=1024)
    parser.add_argument("--policy-name", default="preserve_Q1_uniform1024_plus_Q4_boundary_conflict1024")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
