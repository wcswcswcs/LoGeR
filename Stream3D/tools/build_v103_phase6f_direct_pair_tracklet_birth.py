#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _evaluate_variant,
    _load_baseline,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6f_direct_pair_tracklet_birth"
DEFAULT_PHASE9N_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase9n_da3_bridge_pair_fused_phase4_r3_no_broad_or_rel070"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_positive_core_da3_bridge_pair_phase9n_r3_no_broad_or_rel070"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6f_direct_pair_tracklet_birth_r1"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_id": "T0_all_phase9n_pairs_specific_veto_min2",
        "min_pair_reliability": 0.0,
        "max_carrier_broad_risk": 1.01,
        "topk_per_mask": 0,
        "broad_support_node_veto": False,
        "min_object_frames": 2,
    },
    {
        "variant_id": "T1_rel070_pairs_specific_veto_top2_min2",
        "min_pair_reliability": 0.70,
        "max_carrier_broad_risk": 1.01,
        "topk_per_mask": 2,
        "broad_support_node_veto": False,
        "min_object_frames": 2,
    },
    {
        "variant_id": "T2_no_broad_pairs_specific_veto_top2_min2",
        "min_pair_reliability": 0.0,
        "max_carrier_broad_risk": 0.50,
        "topk_per_mask": 2,
        "broad_support_node_veto": False,
        "min_object_frames": 2,
    },
    {
        "variant_id": "T3_rel070_pairs_broad_support_node_veto_top2_min2",
        "min_pair_reliability": 0.70,
        "max_carrier_broad_risk": 1.01,
        "topk_per_mask": 2,
        "broad_support_node_veto": True,
        "broad_support_min_support_count": 1000,
        "min_object_frames": 2,
    },
]


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        self.parent[rb] = ra
        return ra


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_phase5_scene(phase5_root: Path, scene: str) -> dict[str, np.ndarray]:
    payload = torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
    return {
        "mask_frame": payload["mask_frame"].cpu().numpy().astype(np.int64),
        "mask_label": payload["mask_label"].cpu().numpy().astype(np.int64),
        "mask_is_broad": payload["mask_is_broad"].cpu().numpy().astype(bool),
        "mask_is_object_like": payload["mask_is_object_like"].cpu().numpy().astype(bool),
        "support_count": payload["support_count"].cpu().numpy().astype(np.int64),
    }


def _load_pair_rows(phase9n_root: Path, scene: str, variant: dict[str, Any], payload: dict[str, np.ndarray]) -> pd.DataFrame:
    path = phase9n_root / scene / "da3_bridge_pair_primitive_rows.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep = (
        (df["B_ia"].astype(float) >= float(variant["min_pair_reliability"]))
        & (df["carrier_broad_risk"].astype(float) <= float(variant["max_carrier_broad_risk"]))
    )
    df = df[keep].copy()
    if bool(variant.get("broad_support_node_veto", False)) and not df.empty:
        min_support = int(variant.get("broad_support_min_support_count", 1000))
        a = df["mask_a_observation_index"].astype(int).to_numpy()
        b = df["mask_b_observation_index"].astype(int).to_numpy()
        node_ok = ~(
            (payload["support_count"] >= min_support)
            & payload["mask_is_broad"]
            & (~payload["mask_is_object_like"])
        )
        df = df[node_ok[a] & node_ok[b]].copy()
    if int(variant.get("topk_per_mask", 0)) > 0 and not df.empty:
        selected = np.zeros((len(df),), dtype=bool)
        rows = df.reset_index(drop=True)
        by_mask: dict[int, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows.to_dict("records")):
            by_mask[int(row["mask_a_observation_index"])].append(idx)
            by_mask[int(row["mask_b_observation_index"])].append(idx)
        for indices in by_mask.values():
            order = sorted(indices, key=lambda idx: float(rows.iloc[idx]["B_ia"]), reverse=True)[: int(variant["topk_per_mask"])]
            selected[np.asarray(order, dtype=np.int64)] = True
        df = rows[selected].copy()
    return df.sort_values("B_ia", ascending=False).reset_index(drop=True)


def _would_violate(comp_members: dict[int, set[int]], cannot: set[tuple[int, int]], ra: int, rb: int) -> bool:
    a_members = comp_members.get(int(ra), {int(ra)})
    b_members = comp_members.get(int(rb), {int(rb)})
    if len(a_members) > len(b_members):
        a_members, b_members = b_members, a_members
    for a in a_members:
        for b in b_members:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key in cannot:
                return True
    return False


def _cannot_links(nodes: list[int], payload: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        by_frame[int(payload["mask_frame"][int(node)])].append(int(node))
    out: set[tuple[int, int]] = set()
    for masks in by_frame.values():
        for i, a in enumerate(masks[:-1]):
            for b in masks[i + 1 :]:
                a_specific = bool(payload["mask_is_object_like"][a]) and not bool(payload["mask_is_broad"][a])
                b_specific = bool(payload["mask_is_object_like"][b]) and not bool(payload["mask_is_broad"][b])
                if a_specific and b_specific:
                    out.add((min(a, b), max(a, b)))
    return out


def _cluster_scene(
    *,
    scene: str,
    pairs: pd.DataFrame,
    payload: dict[str, np.ndarray],
    phase2_summary: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = sorted(
        set(pairs["mask_a_observation_index"].astype(int).tolist())
        | set(pairs["mask_b_observation_index"].astype(int).tolist())
    )
    uf = UnionFind(nodes)
    comp_members: dict[int, set[int]] = {node: {node} for node in nodes}
    cannot = _cannot_links(nodes, payload)
    edge_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(pairs.to_dict("records")):
        a = int(row["mask_a_observation_index"])
        b = int(row["mask_b_observation_index"])
        ra = uf.find(a)
        rb = uf.find(b)
        accepted = False
        reason = ""
        if ra != rb:
            if _would_violate(comp_members, cannot, ra, rb):
                reason = "specific_same_frame_cannot_link"
            else:
                new_root = uf.union(ra, rb)
                old_a = comp_members.pop(ra, {ra})
                old_b = comp_members.pop(rb, {rb})
                comp_members[new_root] = set(old_a) | set(old_b)
                accepted = True
        edge_rows.append(
            {
                "schema_version": "stream4d_v103_phase6f_tracklet_edge_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "scene_id": scene,
                "edge_rank": int(rank),
                "mask_a": a,
                "mask_b": b,
                "frame_a": int(payload["mask_frame"][a]),
                "frame_b": int(payload["mask_frame"][b]),
                "affinity": float(row["B_ia"]),
                "accepted_union": bool(accepted),
                "reject_reason": reason,
                "carrier_broad_risk": float(row.get("carrier_broad_risk", 0.0)),
                "uses_gt_for_prediction": False,
            }
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        groups[uf.find(node)].append(node)
    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    cluster_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    object_idx = 0
    for _root, masks in sorted(groups.items(), key=lambda item: (-len({int(payload["mask_frame"][m]) for m in item[1]}), item[0])):
        by_frame: dict[int, list[int]] = defaultdict(list)
        for mask_idx in masks:
            by_frame[int(payload["mask_frame"][mask_idx])].append(int(mask_idx))
        if len(by_frame) < int(variant["min_object_frames"]):
            continue
        selected: dict[int, int] = {}
        for frame, candidates in by_frame.items():
            best = max(
                candidates,
                key=lambda m: (
                    int(payload["mask_is_object_like"][m]),
                    -int(payload["mask_is_broad"][m]),
                    int(payload["support_count"][m]),
                    -int(payload["mask_label"][m]),
                ),
            )
            selected[int(frame)] = int(best)
        selected_masks = list(selected.values())
        selected_broad = float(np.mean(payload["mask_is_broad"][selected_masks])) if selected_masks else 0.0
        selected_object = float(np.mean(payload["mask_is_object_like"][selected_masks])) if selected_masks else 0.0
        score = float(len(selected) / 32.0) * max(0.05, 1.0 - 0.50 * selected_broad)
        object_id = f"{variant['variant_id']}:{scene}:c0000:tracklet_{object_idx:05d}"
        object_idx += 1
        cluster_rows.append(
            {
                "schema_version": "stream4d_v103_phase6f_tracklet_cluster_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant["variant_id"]),
                "scene_id": scene,
                "window_id": "c0000",
                "mv_object_id": object_id,
                "mask_count": int(len(masks)),
                "frame_count": int(len(selected)),
                "object_score": score,
                "selected_broad_mask_ratio": selected_broad,
                "selected_object_like_mask_ratio": selected_object,
                "mean_support_count": float(np.mean(payload["support_count"][masks])) if masks else 0.0,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for frame, mask_idx in selected.items():
            frame_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6f_tracklet_frame_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "mv_object_id": object_id,
                    "object_id": object_id,
                    "scene_id": scene,
                    "chunk_id": "c0000",
                    "window_id": "c0000",
                    "frame_local_index": int(frame),
                    "frame_id": int(frame_ids[int(frame)]),
                    "selected_mask_id": int(payload["mask_label"][mask_idx]),
                    "mask_id_or_generated_id": int(payload["mask_label"][mask_idx]),
                    "object_score": score,
                    "score": score,
                    "support_count": int(payload["support_count"][mask_idx]),
                    "node_policy": "phase9n_pair_endpoint",
                    "emit_policy": "tracklet_wta_by_specificity_support",
                    "readout_mode": "phase6f_direct_pair_tracklet_birth",
                    "selected_mask_is_broad": bool(payload["mask_is_broad"][mask_idx]),
                    "selected_mask_is_object_like": bool(payload["mask_is_object_like"][mask_idx]),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return cluster_rows, frame_rows, edge_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6f: direct object birth from Phase9n induced pair primitives.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase9n-root", default=str(DEFAULT_PHASE9N_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase9n_root = _project(args.phase9n_root)
    phase5_root = _project(args.phase5_root)
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_roots}
    baseline = _load_baseline(_project(args.baseline_rows))

    all_metric_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        scene_rows: dict[str, list[dict[str, Any]]] = {}
        for scene, payload in phase5_payloads.items():
            pairs = _load_pair_rows(phase9n_root, scene, variant, payload)
            clusters, frames, edges = _cluster_scene(
                scene=scene,
                pairs=pairs,
                payload=payload,
                phase2_summary=phase2_summaries[scene],
                variant=variant,
            )
            scene_rows[scene] = frames
            all_cluster_rows.extend(clusters)
            all_frame_rows.extend(frames)
            all_edge_rows.extend(edges)
        window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, _frame_count = _evaluate_variant(
            variant_id=str(variant["variant_id"]),
            scene_rows=scene_rows,
            phase2_summaries=phase2_summaries,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            use_cupy_iou=not bool(args.disable_cupy_iou),
            cupy_device_id=int(args.cupy_device_id),
        )
        for row in window_rows:
            row["phase_id"] = PHASE_ID
        aggregate.update(
            {
                "phase_id": PHASE_ID,
                "min_pair_reliability": float(variant["min_pair_reliability"]),
                "max_carrier_broad_risk": float(variant["max_carrier_broad_risk"]),
                "topk_per_mask": int(variant["topk_per_mask"]),
                "broad_support_node_veto": bool(variant.get("broad_support_node_veto", False)),
                "pixel_collision_count": int(pixel_collision_count),
                "missing_mask_raster_count": int(missing_count),
                "metric_scope": "first32_dev_subset_window_mean; direct Phase9n pair tracklets",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        all_metric_rows.append(aggregate)
        all_window_rows.extend(window_rows)
    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    best = max(all_metric_rows, key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))), default={})
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in all_metric_rows:
        checks = [
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
            ("MV_AP_window_ge_baseline_minus_0p003", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
            ("MV_AP50_window_ge_baseline_minus_0p006", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
        ]
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6f_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": row["variant_id"],
                    "gate_name": name,
                    "pass": bool(ok),
                    "observed": observed,
                    "required": required,
                }
            )
            if row.get("variant_id") == best.get("variant_id") and not bool(ok):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase6f_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "Direct Phase9n pair primitives are insufficient for raw object birth; build stronger GT-free proto-object seeds or keep this as diagnostic evidence only.",
                    }
                )

    _write_csv(out / "tracklet_cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "tracklet_frame_rows.csv", all_frame_rows)
    _write_csv(out / "tracklet_edge_rows.csv", all_edge_rows)
    _write_csv(out / "tracklet_metric_rows.csv", all_metric_rows)
    _write_csv(out / "tracklet_window_rows.csv", all_window_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase6f_direct_pair_tracklet_birth_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE6F_DIRECT_PAIR_TRACKLET_BIRTH" if not failure_rows else "NO_GO_PHASE6F_DIRECT_PAIR_TRACKLET_BIRTH",
        "phase6f_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_MV_AP_window": best.get("MV_AP_window", ""),
        "best_MV_AP50_window": best.get("MV_AP50_window", ""),
        "baseline_contract": baseline,
        "phase9n_root": _rel(phase9n_root),
        "phase5_root": _rel(phase5_root),
        "variant_count": len(VARIANTS),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "Phase6f forms raw tracklets directly from GT-free Phase9n induced pair primitives. GT is used only by the canonical evaluator.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "tracklet_cluster_rows": _rel(out / "tracklet_cluster_rows.csv"),
            "tracklet_frame_rows": _rel(out / "tracklet_frame_rows.csv"),
            "tracklet_edge_rows": _rel(out / "tracklet_edge_rows.csv"),
            "tracklet_metric_rows": _rel(out / "tracklet_metric_rows.csv"),
            "tracklet_window_rows": _rel(out / "tracklet_window_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
