from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v67_local_baselines import _row_from_eval, _summarize_variant_all  # noqa: E402
from stream4d_native.v71_representative_setcover import _load_json, _load_pipeline_roots, _mean, _rel  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


@dataclass(frozen=True)
class CarrierTrackConfig:
    name: str
    min_area: float
    max_area: float
    allow_broad: bool
    max_underseg: float
    same_proto_required: bool
    allow_signature_bridge: bool
    max_frame_gap: int
    min_edge_support: int
    singleton_fill: bool
    max_per_component_members: int


CONFIGS = [
    CarrierTrackConfig("DCT0_clean_same_proto_support3", 0.006, 0.24, False, 0.75, True, False, 5, 3, False, 64),
    CarrierTrackConfig("DCT1_clean_same_proto_support1", 0.006, 0.24, False, 0.75, True, False, 5, 1, False, 64),
    CarrierTrackConfig("DCT2_clean_proto_or_signature_support2", 0.006, 0.24, False, 0.78, False, True, 10, 2, True, 64),
    CarrierTrackConfig("DCT3_clean_proto_or_signature_support1", 0.006, 0.26, False, 0.80, False, True, 10, 1, True, 64),
    CarrierTrackConfig("DCT4_risky_carrier_temporal_support5", 0.004, 0.35, True, 0.92, False, True, 10, 5, True, 96),
]


class DSU:
    def __init__(self, nodes: list[tuple[int, int]]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in ("", None):
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key)
        if value in ("", None):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _b(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key)).strip().lower() in {"1", "true", "yes", "y"}


def _chunk_key(scene: str, value: Any) -> str:
    text = str(value or "")
    if text.startswith(scene + ":chunk"):
        return text
    return f"{scene}:chunk{_i({'x': value}, 'x', -1):03d}"


def _read_candidates(path: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if scenes and row.get("scene_id") not in scenes:
                continue
            out[str(row.get("chunk_id") or "")].append(row)
    return out


def _source_rows(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            table = row.get("carrier_observation_table")
            if scene and table:
                out[scene] = _rooted(str(table))
    return out


def _load_carrier_observations(
    table: Path,
    scene: str,
    wanted_chunks: set[str],
    candidate_pairs_by_chunk: dict[str, set[tuple[int, int]]],
) -> dict[str, dict[str, list[tuple[int, int, float]]]]:
    out: dict[str, dict[str, list[tuple[int, int, float]]]] = defaultdict(lambda: defaultdict(list))
    if not table.exists():
        return out
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene:
                continue
            if str(row.get("valid")).strip().lower() not in {"true", "1", "yes", "y"}:
                continue
            if str(row.get("visible")).strip().lower() not in {"true", "1", "yes", "y"}:
                continue
            if str(row.get("mask_label_available")).strip().lower() not in {"true", "1", "yes", "y"}:
                continue
            chunk = _chunk_key(scene, row.get("chunk_id"))
            if chunk not in wanted_chunks:
                continue
            frame_id = int(float(row.get("frame_id") or -1))
            mask_id = int(float(row.get("observed_mask_id") or 0))
            if mask_id <= 0 or (frame_id, mask_id) not in candidate_pairs_by_chunk.get(chunk, set()):
                continue
            carrier = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
            if not carrier:
                continue
            conf = float(row.get("confidence") or 0.0)
            out[chunk][carrier].append((frame_id, mask_id, conf))
    for carrier_rows in out.values():
        for carrier, values in list(carrier_rows.items()):
            dedup: dict[tuple[int, int], float] = {}
            for frame_id, mask_id, conf in values:
                key = (frame_id, mask_id)
                dedup[key] = max(dedup.get(key, 0.0), float(conf))
            carrier_rows[carrier] = [(frame, mask, conf) for (frame, mask), conf in sorted(dedup.items())]
    return out


def _is_usable(row: dict[str, Any], cfg: CarrierTrackConfig) -> bool:
    area = _f(row, "area_ratio")
    if area < cfg.min_area or area > cfg.max_area:
        return False
    broad = _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or area >= 0.30
    if broad and not cfg.allow_broad:
        return False
    if _f(row, "underseg_proxy_score") >= cfg.max_underseg:
        return False
    return True


def _semantic_compatible(a: dict[str, Any], b: dict[str, Any], cfg: CarrierTrackConfig) -> bool:
    proto_a = str(a.get("semantic_prototype_id") or "")
    proto_b = str(b.get("semantic_prototype_id") or "")
    sig_a = str(a.get("repeated_signature_id") or "")
    sig_b = str(b.get("repeated_signature_id") or "")
    same_proto = bool(proto_a and proto_a == proto_b)
    if cfg.same_proto_required:
        return same_proto
    return same_proto or bool(cfg.allow_signature_bridge and sig_a and sig_a == sig_b)


def _objectness_score(row: dict[str, Any], cfg: CarrierTrackConfig) -> float:
    area = _f(row, "area_ratio")
    margin = _f(row, "semantic_prototype_margin")
    entropy = _f(row, "semantic_entropy", 1.0)
    d4rt = math.log1p(max(0.0, _f(row, "D4RT_carrier_count", 0.0)))
    broad = _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or area >= 0.30
    under = _f(row, "underseg_proxy_score") >= cfg.max_underseg
    mid = 1.0 if 0.010 <= area <= 0.18 else 0.0
    score = mid + 2.0 * margin + 0.25 * entropy + 0.10 * d4rt
    score -= 1.2 if broad else 0.0
    score -= 1.2 if under else 0.0
    return float(score)


def _select_carrier_components(
    rows: list[dict[str, Any]],
    carrier_obs: dict[str, list[tuple[int, int, float]]],
    cfg: CarrierTrackConfig,
    max_components: int,
) -> tuple[dict[tuple[int, int], int], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_pair = {(_i(row, "frame_id"), _i(row, "mask_id")): row for row in rows if _is_usable(row, cfg)}
    nodes = sorted(by_pair)
    dsu = DSU(nodes)
    edge_support: dict[tuple[tuple[int, int], tuple[int, int]], float] = defaultdict(float)
    edge_carriers: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = defaultdict(set)
    carrier_count_seen = 0
    for carrier, values in carrier_obs.items():
        filtered = [(frame, mask, conf) for frame, mask, conf in values if (frame, mask) in by_pair]
        if len(filtered) < 2:
            continue
        carrier_count_seen += 1
        filtered.sort()
        for idx, (fa, ma, ca) in enumerate(filtered):
            for fb, mb, cb in filtered[idx + 1 :]:
                if fb <= fa:
                    continue
                if fb - fa > cfg.max_frame_gap:
                    break
                a = (fa, ma)
                b = (fb, mb)
                row_a = by_pair[a]
                row_b = by_pair[b]
                if not _semantic_compatible(row_a, row_b, cfg):
                    continue
                key = (a, b) if a <= b else (b, a)
                edge_support[key] += 0.5 * (float(ca) + float(cb))
                edge_carriers[key].add(carrier)
    edge_rows = []
    for (a, b), support in edge_support.items():
        carrier_support = len(edge_carriers[(a, b)])
        if carrier_support < cfg.min_edge_support:
            continue
        dsu.union(a, b)
        edge_rows.append(
            {
                "variant": cfg.name,
                "frame_a": a[0],
                "mask_a": a[1],
                "frame_b": b[0],
                "mask_b": b[1],
                "shared_carrier_count": carrier_support,
                "shared_carrier_confidence_sum": float(support),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )

    comps: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node in nodes:
        comps[dsu.find(node)].append(node)
    scored = []
    for members in comps.values():
        if len({node[0] for node in members}) <= 1 and not cfg.singleton_fill:
            continue
        rows_member = [by_pair[node] for node in members]
        rows_member = sorted(rows_member, key=lambda row: _objectness_score(row, cfg), reverse=True)[: cfg.max_per_component_members]
        kept = [(_i(row, "frame_id"), _i(row, "mask_id")) for row in rows_member]
        frames = {node[0] for node in kept}
        edge_bonus = sum(
            len(edge_carriers[key])
            for key in edge_carriers
            if key[0] in set(kept) and key[1] in set(kept) and len(edge_carriers[key]) >= cfg.min_edge_support
        )
        score = sum(_objectness_score(row, cfg) for row in rows_member) + 0.35 * edge_bonus + 0.60 * max(0, len(frames) - 1)
        scored.append((float(score), kept))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected = scored[:max_components]

    mapping: dict[tuple[int, int], int] = {}
    object_rows: list[dict[str, Any]] = []
    for object_id, (score, members) in enumerate(selected, start=1):
        rows_member = [by_pair[node] for node in members]
        for node in members:
            mapping[node] = object_id
        object_rows.append(
            {
                "variant": cfg.name,
                "local_object_id": object_id,
                "component_score": float(score),
                "member_mask_count": len(members),
                "member_frame_count": len({node[0] for node in members}),
                "frame_min": min(node[0] for node in members),
                "frame_max": max(node[0] for node in members),
                "mean_member_mask_area": _mean([_f(row, "area_ratio") for row in rows_member]),
                "semantic_entropy_mean": _mean([_f(row, "semantic_entropy") for row in rows_member]),
                "semantic_prototype_margin_mean": _mean([_f(row, "semantic_prototype_margin") for row in rows_member]),
                "semantic_prototype_count": len({str(row.get("semantic_prototype_id") or "") for row in rows_member}),
                "repeated_signature_count": len({str(row.get("repeated_signature_id") or "") for row in rows_member}),
                "broad_large_member_rate": sum(
                    1
                    for row in rows_member
                    if _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30
                )
                / max(1, len(rows_member)),
                "underseg_proxy_member_rate": sum(1 for row in rows_member if _f(row, "underseg_proxy_score") >= cfg.max_underseg)
                / max(1, len(rows_member)),
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )

    selected_rows = [by_pair[node] for _score, members in selected for node in members]
    diag = {
        "candidate_usable_count": len(nodes),
        "carrier_count_seen": carrier_count_seen,
        "carrier_edge_count": len(edge_rows),
        "raw_component_count": len(comps),
        "selected_component_count": len(selected),
        "support_pair_count": len(mapping),
        "selected_mask_count": len(mapping),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "selected_component_member_count_mean": _mean([float(row["member_mask_count"]) for row in object_rows]),
        "selected_component_frame_count_mean": _mean([float(row["member_frame_count"]) for row in object_rows]),
        "single_frame_component_rate": sum(1 for row in object_rows if int(row["member_frame_count"]) <= 1) / max(1, len(object_rows)),
        "broad_large_member_rate": sum(
            1
            for row in selected_rows
            if _b(row, "broad_background_risk") or _b(row, "large_mask_risk") or _f(row, "area_ratio") >= 0.30
        )
        / max(1, len(selected_rows)),
        "underseg_proxy_member_rate": sum(1 for row in selected_rows if _f(row, "underseg_proxy_score") >= cfg.max_underseg)
        / max(1, len(selected_rows)),
        "mean_shared_carrier_count": _mean([float(row["shared_carrier_count"]) for row in edge_rows]),
    }
    return mapping, object_rows, edge_rows, diag


def _summarize_with_diag(metric_rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    extra_keys = [
        "candidate_usable_count",
        "carrier_count_seen",
        "carrier_edge_count",
        "raw_component_count",
        "selected_component_count",
        "selected_component_member_count_mean",
        "selected_component_frame_count_mean",
        "single_frame_component_rate",
        "broad_large_member_rate",
        "underseg_proxy_member_rate",
        "mean_shared_carrier_count",
    ]
    out = []
    for variant in variants:
        row = _summarize_variant_all(metric_rows, variant)
        subset = [item for item in metric_rows if item.get("variant") == variant]
        for key in extra_keys:
            row[f"{key}_mean"] = _mean([float(item[key]) for item in subset if item.get(key) not in ("", None)])
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes)
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_by_chunk = _read_candidates(_rooted(args.candidate_rows), set(scenes))
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), scenes)
    carrier_tables = _source_rows(_rooted(args.atom_root) / "source_rows.csv")
    atom_summary = _load_json(_rooted(args.atom_root) / "atom_summary.json")
    atom_metrics = atom_summary.get("key_metrics") if isinstance(atom_summary.get("key_metrics"), dict) else atom_summary
    diagnostic_gt_mean = float(atom_metrics.get("diagnostic_GT_count_per_chunk_mean") or 21.515923566878982)
    max_components = int(args.max_components_per_chunk or max(1, math.floor(3.0 * diagnostic_gt_mean)))
    variant_names = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    configs = [cfg for cfg in CONFIGS if cfg.name in set(variant_names)]

    object_rows_all: list[dict[str, Any]] = []
    edge_rows_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    processed = 0
    missing_rows: list[dict[str, Any]] = []
    for scene in scenes:
        pipeline_root = pipeline_roots.get(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "pipeline_root"})
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        scene_chunks = _chunk_rows(pipeline_root, scene)
        if int(args.max_chunks) > 0:
            remaining = max(0, int(args.max_chunks) - processed)
            scene_chunks = scene_chunks[:remaining]
        wanted_chunks = {str(chunk.get("chunk_id")) for chunk in scene_chunks}
        candidate_pairs_by_chunk = {
            chunk: {(_i(row, "frame_id"), _i(row, "mask_id")) for row in candidates_by_chunk.get(chunk, [])}
            for chunk in wanted_chunks
        }
        table = carrier_tables.get(scene)
        if table is None:
            missing_rows.append({"scene_id": scene, "missing": "carrier_observation_table"})
            continue
        carrier_by_chunk = _load_carrier_observations(table, scene, wanted_chunks, candidate_pairs_by_chunk)
        for chunk in scene_chunks:
            if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
                break
            chunk_id = str(chunk.get("chunk_id"))
            rows = candidates_by_chunk.get(chunk_id, [])
            if not rows:
                continue
            processed += 1
            print(f"[v71-d4rt-carrier-track] chunk {processed}: {chunk_id}", file=sys.stderr, flush=True)
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            for cfg in configs:
                mapping, object_rows, edge_rows, diag = _select_carrier_components(
                    rows,
                    carrier_by_chunk.get(chunk_id, {}),
                    cfg,
                    max_components=max_components,
                )
                for row in object_rows:
                    row.update({"scene_id": scene, "chunk_id": chunk_id})
                for row in edge_rows:
                    row.update({"scene_id": scene, "chunk_id": chunk_id})
                object_rows_all.extend(object_rows)
                edge_rows_all.extend(edge_rows)
                metric = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=cfg.name,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    raw_per_frame_masks=False,
                    diag=diag,
                    uses_gt_for_prediction=False,
                    forbidden_for_method_table=False,
                    pipeline_root=pipeline_root,
                )
                metric.update(diag)
                metric_rows.append(metric)
        if int(args.max_chunks) > 0 and processed >= int(args.max_chunks):
            break

    summary_rows = _summarize_with_diag(metric_rows, [cfg.name for cfg in configs])
    best = max(summary_rows, key=lambda row: float(row.get("local_SF50_mean") or row.get("local_score_free_match50_recall_mean") or 0.0), default={})
    summary = {
        "decision": "D4RT_CARRIER_TRACK_REPAIR_DIAGNOSTIC_DONE",
        "processed_chunk_count": processed,
        "max_components_per_chunk": max_components,
        "variants": [cfg.name for cfg in configs],
        "best_variant": best.get("variant"),
        "best_variant_local_SF50": best.get("local_SF50_mean") or best.get("local_score_free_match50_recall_mean"),
        "best_variant_GT_best_IoU_mean": best.get("local_GT_best_IoU_mean_mean"),
        "best_variant_single_frame_component_rate": best.get("single_frame_component_rate_mean"),
        "best_variant_broad_large_member_rate": best.get("broad_large_member_rate_mean"),
        "best_variant_underseg_proxy_member_rate": best.get("underseg_proxy_member_rate_mean"),
        "missing_rows": missing_rows,
        "summary_rows": summary_rows,
    }
    _write_csv(output_root / "d4rt_carrier_track_object_rows.csv", object_rows_all)
    _write_csv(output_root / "d4rt_carrier_track_edge_rows.csv", edge_rows_all)
    _write_csv(output_root / "d4rt_carrier_track_metric_rows.csv", metric_rows)
    _write_csv(output_root / "d4rt_carrier_track_variant_summary_rows.csv", summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    (output_root / "d4rt_carrier_track_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sha_rows = [
        {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.glob("*"))
        if path.is_file()
    ]
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--atom-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--output-root", default="outputs/audit/v71_d4rt_carrier_track_repair")
    parser.add_argument("--variants", default=",".join(cfg.name for cfg in CONFIGS))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-components-per-chunk", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
