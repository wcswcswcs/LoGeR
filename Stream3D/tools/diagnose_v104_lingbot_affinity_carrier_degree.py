#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import _jsonable, _read_json  # noqa: E402
from tools.build_v103_phase6d_f2_skeleton_affinity_merge import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _adapt_f2_rows,
    _load_phase5_scene,
    _rel,
)
from tools.build_v103_r7_phase4_skeleton_confirmed_support import (  # noqa: E402
    _MaskGtCache,
    _object_gt_stats,
)


PHASE_ID = "v104_lingbot_affinity_carrier_degree_diagnostic"
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
DEFAULT_F2_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v104_lingbot_map_only_phase11f_v103_affinity_field_adapter_c0001_framecenter050_lambda035"
DEFAULT_PHASE6D_ROOT = AUDIT_ROOT / "v104_lingbot_map_only_phase12g_v103_phase6d_d9_c0001_framecenter050_lambda035"
DEFAULT_OUT = AUDIT_ROOT / "v104_lingbot_map_only_phase13_affinity_carrier_degree_diagnostic"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _load_incidence(phase5_root: Path, scene: str) -> dict[str, np.ndarray]:
    payload = torch.load(phase5_root / scene / "primitive_incidence_sparse.pt", map_location="cpu")
    return {
        "carrier": payload["carrier_local_index"].cpu().numpy().astype(np.int64),
        "mask": payload["mask_observation_index"].cpu().numpy().astype(np.int64),
        "b": payload["B_ia"].cpu().numpy().astype(np.float64),
        "mask_frame": payload["mask_frame"].cpu().numpy().astype(np.int64),
        "mask_label": payload["mask_label"].cpu().numpy().astype(np.int64),
        "mask_is_object_like": payload["mask_is_object_like"].cpu().numpy().astype(bool),
        "mask_is_broad": payload["mask_is_broad"].cpu().numpy().astype(bool),
        "is_A_anchor": payload["is_A_anchor"].cpu().numpy().astype(bool),
        "is_V_veto": payload["is_V_veto"].cpu().numpy().astype(bool),
    }


def _object_carrier_maps(scene_base: pd.DataFrame, inc: dict[str, np.ndarray]) -> dict[str, dict[int, float]]:
    by_mask: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for carrier, mask, value in zip(inc["carrier"].tolist(), inc["mask"].tolist(), inc["b"].tolist()):
        by_mask[int(mask)].append((int(carrier), float(value)))
    out: dict[str, dict[int, float]] = {}
    for oid, group in scene_base.groupby("mv_object_id", sort=True):
        acc: defaultdict[int, float] = defaultdict(float)
        for idx in sorted({int(v) for v in group["phase5_mask_index"].tolist() if int(v) >= 0}):
            for carrier, value in by_mask.get(idx, []):
                acc[int(carrier)] += float(value)
        out[str(oid)] = dict(acc)
    return out


def _object_features(scene_base: pd.DataFrame, feature: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    dim = int(feature.shape[1])
    for oid, group in scene_base.groupby("mv_object_id", sort=True):
        idxs = np.asarray(sorted({int(v) for v in group["phase5_mask_index"].tolist() if int(v) >= 0}), dtype=np.int64)
        if idxs.size == 0:
            out[str(oid)] = np.zeros((dim,), dtype=np.float32)
            continue
        vec = feature[idxs].mean(axis=0)
        norm = float(np.linalg.norm(vec))
        out[str(oid)] = (vec / max(norm, 1e-12)).astype(np.float32, copy=False)
    return out


def _object_gt_rows(scene: str, scene_base: pd.DataFrame, cache: _MaskGtCache) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for oid, group in scene_base.groupby("mv_object_id", sort=True):
        item = _object_gt_stats(scene, group, cache)
        stats[str(oid)] = item
        rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_object_gt_diagnostic_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mv_object_id": str(oid),
                "frame_count": int(group["frame_id"].nunique()),
                "row_count": int(len(group)),
                "primary_gt_id_diagnostic": int(item["primary_gt_id"]),
                "primary_gt_iou_diagnostic": float(item["primary_gt_iou"]),
                "missing_mask_count": int(item["missing_mask_count"]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )
    return stats, rows


def _pair_key(a: str, b: str) -> str:
    return "||".join(sorted((str(a), str(b))))


def _load_edge_annotations(phase6d_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = phase6d_root / "merge_edge_rows.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict("records")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_pair_key(str(row["object_a"]), str(row["object_b"]))].append(row)
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase5_root = _project(args.phase5_root)
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    phase5_payloads = {scene: _load_phase5_scene(phase5_root, scene) for scene in phase2_summaries}
    base = _adapt_f2_rows(
        f2_root=_project(args.f2_root),
        phase2_summaries=phase2_summaries,
        phase5_payloads=phase5_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    cache = _MaskGtCache(phase2_summaries)
    edge_annotations = _load_edge_annotations(_project(args.phase6d_root))

    carrier_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_gt_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        inc = _load_incidence(phase5_root, scene)
        obj_carriers = _object_carrier_maps(scene_base, inc)
        obj_features = _object_features(scene_base, phase5_payloads[scene]["feature"])
        obj_gt, gt_rows = _object_gt_rows(scene, scene_base, cache)
        object_gt_rows.extend(gt_rows)

        carrier_to_objects: dict[int, set[str]] = defaultdict(set)
        carrier_to_mass: dict[int, float] = defaultdict(float)
        for oid, cmap in obj_carriers.items():
            for carrier, mass in cmap.items():
                if mass <= 0.0:
                    continue
                carrier_to_objects[int(carrier)].add(str(oid))
                carrier_to_mass[int(carrier)] += float(mass)
        object_degree = {carrier: len(objects) for carrier, objects in carrier_to_objects.items()}
        obs_degree = np.bincount(inc["carrier"], minlength=int(max(inc["carrier"].max(initial=0) + 1, len(inc["is_A_anchor"]))))
        broad_inc = np.bincount(inc["carrier"], weights=inc["mask_is_broad"][inc["mask"]].astype(np.float64), minlength=len(obs_degree))
        object_inc = np.bincount(inc["carrier"], weights=inc["mask_is_object_like"][inc["mask"]].astype(np.float64), minlength=len(obs_degree))

        for carrier in range(len(obs_degree)):
            carrier_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_carrier_degree_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "carrier_local_index": int(carrier),
                    "observation_degree": int(obs_degree[carrier]),
                    "f2_object_degree": int(object_degree.get(carrier, 0)),
                    "object_like_incidence_count": int(object_inc[carrier]),
                    "broad_incidence_count": int(broad_inc[carrier]),
                    "object_mass_sum": float(carrier_to_mass.get(carrier, 0.0)),
                    "is_A_anchor": bool(inc["is_A_anchor"][carrier]) if carrier < len(inc["is_A_anchor"]) else False,
                    "is_V_veto": bool(inc["is_V_veto"][carrier]) if carrier < len(inc["is_V_veto"]) else False,
                    "uses_gt_for_prediction": False,
                }
            )

        for oid, cmap in obj_carriers.items():
            masses = list(cmap.values())
            total_mass = float(sum(masses))
            high_degree_mass = float(sum(m for c, m in cmap.items() if int(object_degree.get(c, 0)) >= int(args.high_degree_threshold)))
            veto_mass = float(sum(m for c, m in cmap.items() if c < len(inc["is_V_veto"]) and bool(inc["is_V_veto"][c])))
            object_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_object_carrier_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "mv_object_id": oid,
                    "carrier_count": int(len(cmap)),
                    "carrier_mass_sum": total_mass,
                    "high_degree_carrier_mass_frac": high_degree_mass / max(total_mass, 1e-12),
                    "veto_carrier_mass_frac": veto_mass / max(total_mass, 1e-12),
                    "primary_gt_id_diagnostic": int(obj_gt[oid]["primary_gt_id"]),
                    "primary_gt_iou_diagnostic": float(obj_gt[oid]["primary_gt_iou"]),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                }
            )

        object_ids = sorted(obj_carriers)
        high_affinity_same = 0
        high_affinity_diff = 0
        high_affinity_unknown = 0
        high_affinity_rows: list[dict[str, Any]] = []
        diff_high_overlap_values: list[float] = []
        same_high_overlap_values: list[float] = []
        for idx, oid_a in enumerate(object_ids[:-1]):
            carr_a = obj_carriers[oid_a]
            set_a = set(carr_a)
            mass_a = float(sum(carr_a.values()))
            gt_a = int(obj_gt[oid_a]["primary_gt_id"])
            for oid_b in object_ids[idx + 1 :]:
                carr_b = obj_carriers[oid_b]
                set_b = set(carr_b)
                mass_b = float(sum(carr_b.values()))
                shared = set_a & set_b
                union = set_a | set_b
                weighted_overlap = float(sum(min(carr_a[c], carr_b[c]) for c in shared))
                overlap_over_min = weighted_overlap / max(min(mass_a, mass_b), 1e-12)
                high_degree_shared_mass = float(
                    sum(min(carr_a[c], carr_b[c]) for c in shared if int(object_degree.get(c, 0)) >= int(args.high_degree_threshold))
                )
                veto_shared_mass = float(
                    sum(min(carr_a[c], carr_b[c]) for c in shared if c < len(inc["is_V_veto"]) and bool(inc["is_V_veto"][c]))
                )
                affinity = float(np.dot(obj_features[oid_a], obj_features[oid_b]))
                gt_b = int(obj_gt[oid_b]["primary_gt_id"])
                same_gt = bool(gt_a > 0 and gt_a == gt_b)
                diff_gt = bool(gt_a > 0 and gt_b > 0 and gt_a != gt_b)
                annotations = edge_annotations.get(_pair_key(oid_a, oid_b), [])
                variants = sorted({str(row.get("variant_id", "")) for row in annotations if str(row.get("variant_id", ""))})
                accepted_variants = sorted({str(row.get("variant_id", "")) for row in annotations if bool(row.get("accepted_union", False))})
                row = {
                    "schema_version": "stream4d_v104_lingbot_pair_carrier_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "affinity": affinity,
                    "shared_carrier_count": int(len(shared)),
                    "union_carrier_count": int(len(union)),
                    "carrier_jaccard": float(len(shared) / max(len(union), 1)),
                    "weighted_overlap_over_min": overlap_over_min,
                    "high_degree_shared_mass_frac": high_degree_shared_mass / max(weighted_overlap, 1e-12),
                    "veto_shared_mass_frac": veto_shared_mass / max(weighted_overlap, 1e-12),
                    "object_a_primary_gt_id_diagnostic": gt_a,
                    "object_b_primary_gt_id_diagnostic": gt_b,
                    "same_GT_diagnostic": same_gt,
                    "diff_GT_diagnostic": diff_gt,
                    "phase6d_edge_variant_ids": ";".join(variants),
                    "phase6d_accepted_variant_ids": ";".join(accepted_variants),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                }
                pair_rows.append(row)
                if affinity >= float(args.high_affinity_threshold):
                    high_affinity_rows.append(row)
                    if same_gt:
                        high_affinity_same += 1
                        same_high_overlap_values.append(overlap_over_min)
                    elif diff_gt:
                        high_affinity_diff += 1
                        diff_high_overlap_values.append(overlap_over_min)
                    else:
                        high_affinity_unknown += 1
        scene_pair_rows = [row for row in pair_rows if row["scene_id"] == scene]
        scene_summaries.append(
            {
                "schema_version": "stream4d_v104_lingbot_carrier_degree_scene_summary_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "object_count": int(len(object_ids)),
                "pair_count": int(len(scene_pair_rows)),
                "carrier_count": int(len(obs_degree)),
                "carrier_object_degree_ge_threshold_count": int(sum(1 for v in object_degree.values() if int(v) >= int(args.high_degree_threshold))),
                "high_affinity_threshold": float(args.high_affinity_threshold),
                "high_affinity_pair_count": int(len(high_affinity_rows)),
                "high_affinity_same_GT_count_diagnostic": int(high_affinity_same),
                "high_affinity_diff_GT_count_diagnostic": int(high_affinity_diff),
                "high_affinity_unknown_GT_count_diagnostic": int(high_affinity_unknown),
                "high_affinity_diff_GT_rate_diagnostic": float(high_affinity_diff / max(high_affinity_same + high_affinity_diff, 1)),
                "diff_GT_high_affinity_overlap_over_min_p50": _quantile(diff_high_overlap_values, 0.50),
                "diff_GT_high_affinity_overlap_over_min_p90": _quantile(diff_high_overlap_values, 0.90),
                "same_GT_high_affinity_overlap_over_min_p50": _quantile(same_high_overlap_values, 0.50),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )

    pair_rows.sort(
        key=lambda row: (
            -float(row["affinity"]),
            -float(row["weighted_overlap_over_min"]),
            str(row["scene_id"]),
            str(row["object_a"]),
            str(row["object_b"]),
        )
    )
    high_pairs = [row for row in pair_rows if float(row["affinity"]) >= float(args.high_affinity_threshold)]
    known_high = [row for row in high_pairs if bool(row["same_GT_diagnostic"]) or bool(row["diff_GT_diagnostic"])]
    false_high = [row for row in known_high if bool(row["diff_GT_diagnostic"])]
    summary = {
        "schema_version": "stream4d_v104_lingbot_carrier_degree_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "phase5_root": _rel(phase5_root),
        "phase6d_root": _rel(_project(args.phase6d_root)),
        "dataset_split": str(args.dataset_split),
        "chunk_id": str(args.chunk_id),
        "high_affinity_threshold": float(args.high_affinity_threshold),
        "high_degree_threshold": int(args.high_degree_threshold),
        "scene_summaries": scene_summaries,
        "high_affinity_known_pair_count": int(len(known_high)),
        "high_affinity_false_pair_count": int(len(false_high)),
        "high_affinity_false_pair_rate_diagnostic": float(len(false_high) / max(len(known_high), 1)),
        "high_affinity_false_overlap_over_min_p50": _quantile([float(row["weighted_overlap_over_min"]) for row in false_high], 0.50),
        "high_affinity_false_high_degree_shared_mass_frac_p50": _quantile(
            [float(row["high_degree_shared_mass_frac"]) for row in false_high], 0.50
        ),
        "decision": "DIAGNOSTIC_ONLY_FALSE_HIGH_AFFINITY_TAIL" if false_high else "DIAGNOSTIC_ONLY_NO_FALSE_HIGH_AFFINITY_TAIL",
        "truthfulness_note": "GT is read only to classify diagnostic same/different object pairs; all method features and Phase6d edges remain GT-free.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "pair_carrier_rows": _rel(out / "pair_carrier_rows.csv"),
            "carrier_degree_rows": _rel(out / "carrier_degree_rows.csv"),
            "object_carrier_rows": _rel(out / "object_carrier_rows.csv"),
            "object_gt_rows": _rel(out / "object_gt_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_csv(out / "pair_carrier_rows.csv", pair_rows[: int(args.max_pair_rows)])
    _write_csv(out / "carrier_degree_rows.csv", carrier_rows)
    _write_csv(out / "object_carrier_rows.csv", object_rows)
    _write_csv(out / "object_gt_rows.csv", object_gt_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose LingBot v103 affinity high-score pairs by shared carrier degree.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--phase6d-root", default=str(DEFAULT_PHASE6D_ROOT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--high-affinity-threshold", type=float, default=0.65)
    parser.add_argument("--high-degree-threshold", type=int, default=3)
    parser.add_argument("--max-pair-rows", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    build(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
