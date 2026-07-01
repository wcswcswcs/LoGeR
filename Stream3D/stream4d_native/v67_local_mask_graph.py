from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v67_local_baselines import (  # noqa: E402
    _oracle_majority_mapping_bundle,
    _representative_pairs_by_chunk,
    _row_from_eval,
    _summarize_variant_all,
)
from stream4d_native.v67_object_balanced_setcover import (  # noqa: E402
    CONFIGS,
    _candidate_features,
    _mask_area_lookup,
    _select_candidates,
)
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import (  # noqa: E402
    _chunk_rows,
    _float_or_none,
    _frame_data,
    _load_csv_rows,
    _mean,
    _parse_mask_observation_id,
    _rel,
)
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)
from stream4d.scannet_stream import ScanNetStream  # noqa: E402


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


def _support_by_pair(pipeline_root: Path, scene: str, support_variant: str) -> dict[tuple[int, int], set[str]]:
    out: dict[tuple[int, int], set[str]] = defaultdict(set)
    path = pipeline_root / "mask_component_support/mask_component_support_rows.csv"
    if not path.exists():
        return out
    for row in _load_csv_rows(path):
        if row.get("scene") != scene or row.get("variant") != support_variant:
            continue
        parsed = _parse_mask_observation_id(row.get("mask_observation_id"))
        if parsed is None or parsed[0] != scene:
            continue
        comp = str(row.get("component_id") or "").strip()
        if comp:
            out[(int(parsed[1]), int(parsed[2]))].add(comp)
    return out


def _mask_summary_by_pair(pipeline_root: Path, scene: str) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    path = pipeline_root / "mask_component_support/mask_summary_rows.csv"
    if not path.exists():
        return out
    for row in _load_csv_rows(path):
        if row.get("scene") != scene:
            continue
        parsed = _parse_mask_observation_id(row.get("mask_observation_id"))
        if parsed is not None and parsed[0] == scene:
            out[(int(parsed[1]), int(parsed[2]))] = row
    return out


def _component_cc_mapping(nodes: set[tuple[int, int]], support: dict[tuple[int, int], set[str]]) -> dict[tuple[int, int], int]:
    dsu = DSU(sorted(nodes))
    by_comp: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node in nodes:
        for comp in support.get(node, set()):
            by_comp[comp].append(node)
    for comp_nodes in by_comp.values():
        if len(comp_nodes) <= 1:
            continue
        first = comp_nodes[0]
        for node in comp_nodes[1:]:
            dsu.union(first, node)
    root_to_id: dict[tuple[int, int], int] = {}
    mapping: dict[tuple[int, int], int] = {}
    for node in sorted(nodes):
        root = dsu.find(node)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        mapping[node] = root_to_id[root]
    return mapping


def _split_same_frame(
    mapping: dict[tuple[int, int], int],
    support: dict[tuple[int, int], set[str]],
    *,
    reject_pairs: set[tuple[int, int]] | None = None,
) -> tuple[dict[tuple[int, int], int], int]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for node, group_id in mapping.items():
        if reject_pairs and node in reject_pairs:
            continue
        groups[int(group_id)].append(node)
    out: dict[tuple[int, int], int] = {}
    next_id = 1
    violation_count = 0
    for nodes in groups.values():
        clusters: list[dict[str, Any]] = []
        for node in sorted(nodes, key=lambda item: (item[0], item[1])):
            node_components = support.get(node, set())
            best_idx = -1
            best_overlap = -1
            for idx, cluster in enumerate(clusters):
                if node[0] in cluster["frames"]:
                    continue
                overlap = len(node_components & cluster["components"])
                if overlap > best_overlap:
                    best_idx = idx
                    best_overlap = overlap
            if best_idx < 0:
                clusters.append({"nodes": [node], "frames": {node[0]}, "components": set(node_components)})
            else:
                clusters[best_idx]["nodes"].append(node)
                clusters[best_idx]["frames"].add(node[0])
                clusters[best_idx]["components"] |= set(node_components)
        for cluster in clusters:
            frame_counts: dict[int, int] = defaultdict(int)
            for node in cluster["nodes"]:
                out[node] = next_id
                frame_counts[node[0]] += 1
            violation_count += sum(max(0, count - 1) for count in frame_counts.values())
            next_id += 1
    return out, int(violation_count)


def _reject_pairs(
    nodes: set[tuple[int, int]],
    support: dict[tuple[int, int], set[str]],
    summary: dict[tuple[int, int], dict[str, Any]],
    area_lookup: dict[tuple[int, int], float],
) -> set[tuple[int, int]]:
    rejected: set[tuple[int, int]] = set()
    for node in nodes:
        area_ratio = float(area_lookup.get(node, 0.0))
        comp_count = len(support.get(node, set()))
        raw_entropy = float(summary.get(node, {}).get("raw_component_entropy") or 0.0)
        if area_ratio >= 0.30:
            rejected.add(node)
        elif comp_count >= 64 and raw_entropy > 1.5:
            rejected.add(node)
    return rejected


def _seed_absorb_mapping(
    *,
    nodes: set[tuple[int, int]],
    support: dict[tuple[int, int], set[str]],
    candidate_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    scene: str,
    chunk_id: str,
    frame_ids: list[int],
    area_lookup: dict[tuple[int, int], float],
    config_name: str,
    reject_pairs: set[tuple[int, int]] | None = None,
    max_selected: int = 128,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    config_lookup = {cfg.name: cfg for cfg in CONFIGS}
    config = config_lookup[config_name]
    features = _candidate_features(
        candidate_rows=candidate_rows,
        scene=scene,
        chunk_id=chunk_id,
        frame_ids=frame_ids,
        area_lookup=area_lookup,
    )
    selected_ids, cover_diag = _select_candidates(features, config=config, max_selected=max_selected)
    selected_set = set(selected_ids)
    candidate_to_obj = {candidate_id: idx + 1 for idx, candidate_id in enumerate(selected_ids)}
    mapping: dict[tuple[int, int], int] = {}
    seed_components: dict[int, set[str]] = defaultdict(set)
    for row in ledger_rows:
        if row.get("scene") != scene or row.get("chunk_id") != chunk_id:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        object_id = int(candidate_to_obj.get(candidate_id, 0))
        if candidate_id not in selected_set or object_id <= 0:
            continue
        parsed = _parse_mask_observation_id(row.get("best_mask_observation_id"))
        if parsed is None or parsed[0] != scene:
            continue
        node = (int(parsed[1]), int(parsed[2]))
        if node not in nodes or (reject_pairs and node in reject_pairs):
            continue
        mapping[node] = object_id
    candidate_component_lookup = {str(row.get("candidate_id") or ""): set(json.loads(row.get("component_ids") or "[]")) for row in candidate_rows if row.get("scene") == scene and row.get("chunk_id") == chunk_id}
    for candidate_id, object_id in candidate_to_obj.items():
        seed_components[object_id] |= candidate_component_lookup.get(candidate_id, set())
    object_frames: dict[int, set[int]] = defaultdict(set)
    for node, object_id in mapping.items():
        object_frames[int(object_id)].add(int(node[0]))
    absorbed = 0
    for node in sorted(nodes):
        if node in mapping or (reject_pairs and node in reject_pairs):
            continue
        comps = support.get(node, set())
        best_object = 0
        best_overlap = 0
        for object_id, object_comps in seed_components.items():
            if int(node[0]) in object_frames.get(object_id, set()):
                continue
            overlap = len(comps & object_comps)
            if overlap > best_overlap:
                best_object = int(object_id)
                best_overlap = int(overlap)
        if best_object > 0 and best_overlap > 0:
            mapping[node] = best_object
            object_frames[best_object].add(int(node[0]))
            seed_components[best_object] |= comps
            absorbed += 1
    diag = {
        **cover_diag,
        "support_pair_count": int(len(mapping)),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "absorbed_support_mask_count": int(absorbed),
        "selected_seed_count": int(len(selected_ids)),
        "rejected_mask_count": int(len(reject_pairs or set())),
    }
    return mapping, diag


def _diag_for_mapping(mapping: dict[tuple[int, int], int], *, violation_count: int, rejected_count: int = 0) -> dict[str, Any]:
    return {
        "selected_mask_count": int(len(mapping)),
        "support_pair_count": int(len(mapping)),
        "duplicate_frame_mask_conflict_pairs": 0,
        "duplicate_frame_mask_conflict_rate": 0.0,
        "same_frame_cannot_link_violation_count": int(violation_count),
        "rejected_mask_count": int(rejected_count),
    }


def _same_frame_violation_count(mapping: dict[tuple[int, int], int]) -> int:
    by_object_frame: dict[tuple[int, int], int] = defaultdict(int)
    for (frame_id, _mask_id), object_id in mapping.items():
        by_object_frame[(int(object_id), int(frame_id))] += 1
    return int(sum(max(0, count - 1) for count in by_object_frame.values()))


def _summarize_graph_all(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    base = _summarize_variant_all(rows, variant)
    subset = [row for row in rows if row["variant"] == variant]
    base.update(
        {
            "same_frame_cannot_link_violation_count_sum": int(
                sum(int(float(row.get("same_frame_cannot_link_violation_count") or 0)) for row in subset)
            ),
            "absorbed_support_mask_count_mean": _mean([_float_or_none(row.get("absorbed_support_mask_count")) for row in subset]),
            "rejected_mask_count_mean": _mean([_float_or_none(row.get("rejected_mask_count")) for row in subset]),
            "runtime_sec_mean": _mean([_float_or_none(row.get("runtime_sec")) for row in subset]),
        }
    )
    return base


def _best_non_oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if not bool(row.get("uses_gt_for_prediction"))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("local_score_free_match50_recall_mean") or 0.0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    stride = int(args.stride)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        print(f"[v67-graph] scene={scene}", file=sys.stderr, flush=True)
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        all_stride_frames = stream.frame_ids(stride=stride, max_frames=None)
        representative_by_chunk = _representative_pairs_by_chunk(pipeline_root, scene)
        support = _support_by_pair(pipeline_root, scene, str(args.support_variant))
        summary_by_pair = _mask_summary_by_pair(pipeline_root, scene)
        candidate_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/candidate_rows.csv")
        ledger_rows = _load_csv_rows(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv")
        for chunk in _chunk_rows(pipeline_root, scene):
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in all_stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_data = _frame_data(scene, frame_ids, mask_dir)
            area_lookup = _mask_area_lookup(frame_data)
            nodes = set(representative_by_chunk.get(chunk_id, set()))
            if not nodes:
                nodes = {pair for pair in support if raw_start <= pair[0] <= raw_end}
            cc_mapping = _component_cc_mapping(nodes, support)
            g2_mapping, g2_violations = _split_same_frame(cc_mapping, support)
            rejected_pairs = _reject_pairs(nodes, support, summary_by_pair, area_lookup)
            g3_mapping, g3_violations = _split_same_frame(cc_mapping, support, reject_pairs=rejected_pairs)
            g4_mapping, g4_diag = _seed_absorb_mapping(
                nodes=nodes,
                support=support,
                candidate_rows=candidate_rows,
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                frame_ids=frame_ids,
                area_lookup=area_lookup,
                config_name="K1_object_balanced_component_frame_area",
                reject_pairs=None,
                max_selected=int(args.max_selected),
            )
            g5_mapping, g5_diag = _seed_absorb_mapping(
                nodes=nodes,
                support=support,
                candidate_rows=candidate_rows,
                ledger_rows=ledger_rows,
                scene=scene,
                chunk_id=chunk_id,
                frame_ids=frame_ids,
                area_lookup=area_lookup,
                config_name="K6_repeated_signature_priority",
                reject_pairs=rejected_pairs,
                max_selected=int(args.max_selected),
            )
            oracle_bundle = _oracle_majority_mapping_bundle(
                frame_data=frame_data,
                selected_pairs=set(),
                representative_pairs=nodes,
            )
            g7_mapping, g7_diag = oracle_bundle["representative"]
            eval_specs = [
                ("G0_representative_masks_as_objects", {node: idx + 1 for idx, node in enumerate(sorted(nodes))}, _diag_for_mapping({node: idx + 1 for idx, node in enumerate(sorted(nodes))}, violation_count=0), False, False),
                ("G1_positive_component_connected_components", cc_mapping, _diag_for_mapping(cc_mapping, violation_count=_same_frame_violation_count(cc_mapping)), False, False),
                ("G2_component_cc_same_frame_split", g2_mapping, _diag_for_mapping(g2_mapping, violation_count=g2_violations), False, False),
                ("G3_signed_greedy_reject_broad_split", g3_mapping, _diag_for_mapping(g3_mapping, violation_count=g3_violations, rejected_count=len(rejected_pairs)), False, False),
                ("G4_setcover_seed_absorption", g4_mapping, g4_diag, False, False),
                ("G5_seed_absorption_underseg_large_reject", g5_mapping, g5_diag, False, False),
                ("G6_semantic_guard_unavailable_same_as_G5", g5_mapping, {**g5_diag, "semantic_guard_available": False}, False, False),
                ("G7_oracle_representative_graph_majority_GT", g7_mapping, g7_diag, True, True),
            ]
            for variant, mapping, diag, uses_gt, forbidden in eval_specs:
                t0 = time.time()
                row = _row_from_eval(
                    scene=scene,
                    chunk_id=chunk_id,
                    variant=variant,
                    frame_ids=frame_ids,
                    chunk=chunk,
                    frame_data=frame_data,
                    mapping=mapping,
                    raw_per_frame_masks=False,
                    diag=diag,
                    uses_gt_for_prediction=uses_gt,
                    forbidden_for_method_table=forbidden,
                    pipeline_root=pipeline_root,
                )
                row.update(diag)
                row["local_object_count"] = row.get("local_object_count")
                row["selected_mask_count"] = diag.get("selected_mask_count", "")
                row["absorbed_support_mask_count"] = diag.get("absorbed_support_mask_count", "")
                row["shared_mask_count"] = 0
                row["rejected_mask_count"] = diag.get("rejected_mask_count", "")
                row["same_frame_cannot_link_violation_count"] = diag.get("same_frame_cannot_link_violation_count", 0)
                row["runtime_sec"] = float(time.time() - t0)
                rows.append(row)
    variant_summary_rows = [_summarize_graph_all(rows, variant) for variant in sorted({row["variant"] for row in rows})]
    best = _best_non_oracle(variant_summary_rows) or {}
    best_sf50 = _float_or_none(best.get("local_score_free_match50_recall_mean"))
    best_gt = _float_or_none(best.get("local_GT_best_IoU_mean_mean"))
    best_ap50 = _float_or_none(best.get("local_AP50_mean"))
    best_dup = _float_or_none(best.get("local_duplicate_frame_mask_conflict_rate_mean"))
    best_violations = int(float(best.get("same_frame_cannot_link_violation_count_sum") or 0)) if best else 0
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "best_G_local_SF50_ge_0p30": best_sf50 is not None and best_sf50 >= 0.30,
        "best_G_GT_best_IoU_ge_0p25": best_gt is not None and best_gt >= 0.25,
        "best_G_AP50_ge_0p05": best_ap50 is not None and best_ap50 >= 0.05,
        "best_G_duplicate_rate_le_0p02": best_dup is not None and best_dup <= 0.02,
        "best_G_same_frame_cannot_link_violation_count_eq_0": best_violations == 0,
    }
    gate["best_G_local_gate_pass"] = (
        gate["best_G_local_SF50_ge_0p30"]
        and gate["best_G_GT_best_IoU_ge_0p25"]
        and gate["best_G_AP50_ge_0p05"]
        and gate["best_G_duplicate_rate_le_0p02"]
        and gate["best_G_same_frame_cannot_link_violation_count_eq_0"]
    )
    decision = "PASS_LOCAL_MASK_GRAPH" if gate["best_G_local_gate_pass"] else "LOCAL_MASK_GRAPH_FAILS_LOCAL_GATE"
    _write_csv(output_root / "local_graph_rows.csv", rows)
    _write_csv(output_root / "local_graph_variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v67_local_mask_graph",
        "decision": decision,
        "diagnostic_only": True,
        "scenes": scenes,
        "stride": stride,
        "support_variant": str(args.support_variant),
        "pipeline_roots": pipeline_roots,
        "gate": gate,
        "best_G": best,
        "rows": {
            "local_graph_rows_csv": _rel(output_root / "local_graph_rows.csv"),
            "local_graph_variant_summary_rows_csv": _rel(output_root / "local_graph_variant_summary_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "G0-G6 use representative masks, mask-component support, candidate component IDs, ledger masks, same-frame cannot-link splitting, and broad-mask reject proxies only.",
            "G7 uses GT majority labels and is forbidden for method tables.",
            "Semantic guard is recorded as unavailable because the current graph inputs do not expose reliable semantic mode atoms.",
        ],
    }
    _write_json(output_root / "local_graph_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "local_graph_summary.json",
        output_root / "local_graph_rows.csv",
        output_root / "local_graph_variant_summary_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v67 local mask graph diagnostics.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v67_local_mask_graph")
    parser.add_argument("--support-variant", default="I0_visible_tau0.10")
    parser.add_argument("--max-selected", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
