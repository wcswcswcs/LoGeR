#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

from tools import build_v103_phase6_mask_clustering_local_object_birth as p6  # noqa: E402


PHASE_ID = "v103_phase6c_f2_scaffold_affinity_repair"
DEFAULT_SCAFFOLD_ROWS = STREAM3D_ROOT / "outputs/audit/v100_phase2c_overlap3_local_repair/mv_object_frame_mask_rows.parquet"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase5_mask_level_pooling_q5c_phase4r7_r4_control_gate_strict_l2o"
DEFAULT_PHASE2_SCENE0011 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
DEFAULT_BASELINE_ROWS = STREAM3D_ROOT / "outputs/audit/v103_phase0_contract/baseline_metric_rows.csv"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6c_f2_scaffold_affinity_repair"


VARIANTS = [
    {
        "variant_id": "S0_scaffold_noop_original_score_control",
        "family": "control",
        "mode": "noop",
        "threshold": 1.01,
        "score_policy": "preserve_v100_original_score",
    },
    {
        "variant_id": "S1_paf_merge_tau090_no_same_frame_maxscore",
        "family": "paf_merge",
        "mode": "merge",
        "threshold": 0.90,
        "score_policy": "max_member_v100_score",
    },
    {
        "variant_id": "S2_paf_merge_tau085_no_same_frame_maxscore",
        "family": "paf_merge",
        "mode": "merge",
        "threshold": 0.85,
        "score_policy": "max_member_v100_score",
    },
    {
        "variant_id": "S3_paf_merge_tau080_no_same_frame_maxscore",
        "family": "paf_merge",
        "mode": "merge",
        "threshold": 0.80,
        "score_policy": "max_member_v100_score",
    },
    {
        "variant_id": "S4_paf_split_lowcoh015_rowtau040",
        "family": "paf_split",
        "mode": "split",
        "low_coherence_threshold": 0.15,
        "row_threshold": 0.40,
        "score_policy": "component_frame_count_plus_original_tiebreak",
    },
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return p6._rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = df[(df.get("baseline_role", "") == "current_strong_local_baseline") & (df.get("dataset_split", "") == "dev")]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "MV_AP_window": float(row.get("MV_AP_window", 0.0)),
        "MV_AP50_window": float(row.get("MV_AP50_window", 0.0)),
        "MV_AP25_window": float(row.get("MV_AP25_window", 0.0)),
        "ScoreFreeMatch50_window": float(row.get("ScoreFreeMatch50_window", 0.0)),
    }


def _normalize_rows(arr: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(arr.to(torch.float32), p=2, dim=1, eps=1e-12)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _scene_cache_path(out: Path, scene: str) -> Path:
    return out / scene / "scaffold_feature_cache.pt"


def _load_or_build_scene_cache(
    *,
    scene: str,
    rows_df: pd.DataFrame,
    phase5_root: Path,
    phase2_summary: dict[str, Any],
    out: Path,
    reuse_feature_cache: bool,
) -> dict[str, Any]:
    cache_path = _scene_cache_path(out, scene)
    if reuse_feature_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        payload["cache_hit"] = True
        return payload

    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    frame_to_local = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    feature_payload = torch.load(phase5_root / scene / "mask_level_feature.pt", map_location="cpu")
    features = _normalize_rows(feature_payload["feature"])
    mask_frame = feature_payload["mask_frame"].to(torch.long).cpu().numpy().astype(np.int64)
    mask_label = feature_payload["mask_label"].to(torch.long).cpu().numpy().astype(np.int64)
    support_count = feature_payload["support_count"].to(torch.long).cpu().numpy().astype(np.int64)
    key_to_obs = {(int(mask_frame[idx]), int(mask_label[idx])): idx for idx in range(mask_frame.shape[0])}

    scene_rows: list[dict[str, Any]] = []
    row_features: list[torch.Tensor] = []
    mapped_count = 0
    for row in rows_df.to_dict("records"):
        frame_id = int(row["frame_id"])
        if frame_id not in frame_to_local:
            continue
        local = int(frame_to_local[frame_id])
        mask_id = int(row["selected_mask_id"])
        obs_idx = int(key_to_obs.get((local, mask_id), -1))
        mapped = obs_idx >= 0
        if mapped:
            mapped_count += 1
            row_features.append(features[obs_idx].detach().cpu())
        else:
            row_features.append(torch.zeros((features.shape[1],), dtype=torch.float32))
        new = dict(row)
        new["frame_local_index"] = local
        new["selected_mask_observation_index"] = obs_idx
        new["phase5_mask_feature_mapped"] = bool(mapped)
        new["phase5_mask_support_count"] = int(support_count[obs_idx]) if mapped else 0
        scene_rows.append(new)

    if row_features:
        row_feature = torch.stack(row_features, dim=0).to(torch.float32)
    else:
        row_feature = torch.zeros((0, int(features.shape[1])), dtype=torch.float32)
    object_ids = sorted({str(row["mv_object_id"]) for row in scene_rows})
    object_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(scene_rows):
        object_to_rows[str(row["mv_object_id"])].append(idx)

    signatures: list[torch.Tensor] = []
    object_rows: list[dict[str, Any]] = []
    for obj_idx, oid in enumerate(object_ids):
        idxs = object_to_rows[oid]
        mapped = [idx for idx in idxs if int(scene_rows[idx]["selected_mask_observation_index"]) >= 0]
        if mapped:
            feat = row_feature[torch.as_tensor(mapped, dtype=torch.long)]
            weights_np = np.asarray([math.log1p(float(scene_rows[idx].get("phase5_mask_support_count", 0))) for idx in mapped], dtype=np.float32)
            if float(np.max(weights_np)) <= 0.0:
                weights_np = np.ones_like(weights_np)
            weights = torch.as_tensor(weights_np, dtype=torch.float32).reshape(-1, 1)
            sig = torch.sum(feat * weights, dim=0) / torch.clamp(torch.sum(weights), min=1e-12)
            sig = torch.nn.functional.normalize(sig.reshape(1, -1), p=2, dim=1, eps=1e-12)[0]
            if len(mapped) >= 2:
                local_feat = _normalize_rows(feat)
                sim = (local_feat @ local_feat.T).cpu().numpy()
                tri = sim[np.triu_indices(sim.shape[0], k=1)]
                internal = float(np.mean(tri)) if tri.size else 0.0
            else:
                internal = 1.0
        else:
            sig = torch.zeros((features.shape[1],), dtype=torch.float32)
            internal = 0.0
        signatures.append(sig.detach().cpu())
        frames = sorted({int(scene_rows[idx]["frame_id"]) for idx in idxs})
        object_rows.append(
            {
                "schema_version": "stream4d_v103_phase6c_scaffold_object_feature_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "source_mv_object_id": oid,
                "object_index": obj_idx,
                "frame_count": len(frames),
                "row_count": len(idxs),
                "mapped_row_count": len(mapped),
                "mapped_row_rate": float(len(mapped) / max(1, len(idxs))),
                "paf_internal_coherence_mean": internal,
                "source_score_max": max(_safe_float(scene_rows[idx].get("score")) for idx in idxs),
                "uses_gt_for_prediction": False,
            }
        )
    signature = torch.stack(signatures, dim=0).to(torch.float32) if signatures else torch.zeros((0, features.shape[1]), dtype=torch.float32)

    payload = {
        "schema_version": "stream4d_v103_phase6c_scaffold_feature_cache_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "rows": scene_rows,
        "row_feature": row_feature,
        "object_ids": object_ids,
        "object_signature": signature,
        "object_rows": object_rows,
        "frame_ids": frame_ids,
        "phase5_variant_id": feature_payload.get("variant_id", ""),
        "phase5_pair_affinity_mode": feature_payload.get("pair_affinity_mode", ""),
        "row_count": len(scene_rows),
        "mapped_row_count": mapped_count,
        "mapped_row_rate": float(mapped_count / max(1, len(scene_rows))),
        "cache_hit": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def _object_pair_rows(scene_cache: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    object_ids = list(scene_cache["object_ids"])
    sig = scene_cache["object_signature"].to(torch.float32)
    if sig.numel() == 0:
        return []
    with torch.no_grad():
        sig = _normalize_rows(sig).to(device)
        sim = (sig @ sig.T).detach().cpu().numpy()
    rows_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_cache["rows"]:
        rows_by_object[str(row["mv_object_id"])].append(row)
    pair_rows: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        frames_a = {int(row["frame_id"]) for row in rows_by_object[oid_a]}
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            frames_b = {int(row["frame_id"]) for row in rows_by_object[oid_b]}
            shared_frames = sorted(frames_a & frames_b)
            pair_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6c_scaffold_object_pair_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_cache["scene_id"],
                    "source_mv_object_id_a": oid_a,
                    "source_mv_object_id_b": oid_b,
                    "object_index_a": i,
                    "object_index_b": j,
                    "paf_object_affinity": float(sim[i, j]),
                    "same_frame_conflict": bool(shared_frames),
                    "shared_frame_count": len(shared_frames),
                    "uses_gt_for_prediction": False,
                }
            )
    return pair_rows


def _dedupe_object_frame_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["mv_object_id"]), int(row["frame_local_index"]))
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        new_score = (_safe_float(row.get("score")), _safe_float(row.get("phase5_mask_support_count")), -int(row.get("selected_mask_id", 0)))
        old_score = (_safe_float(old.get("score")), _safe_float(old.get("phase5_mask_support_count")), -int(old.get("selected_mask_id", 0)))
        if new_score > old_score:
            best[key] = row
    return [best[key] for key in sorted(best)]


def _rows_from_mapping(
    *,
    scene_cache: dict[str, Any],
    variant: dict[str, Any],
    object_mapping: dict[str, str],
    component_score: dict[str, float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in scene_cache["rows"]:
        old_oid = str(row["mv_object_id"])
        new_oid = object_mapping.get(old_oid, old_oid)
        new = dict(row)
        new["schema_version"] = "stream4d_v103_phase6c_local_object_frame_mask_row_v1"
        new["phase_id"] = PHASE_ID
        new["variant_id"] = variant["variant_id"]
        new["scaffold_source_phase_id"] = row.get("phase_id", "")
        new["source_mv_object_id"] = old_oid
        new["mv_object_id"] = new_oid
        new["object_id"] = new_oid
        new["object_score"] = float(component_score.get(new_oid, _safe_float(row.get("score"))))
        new["score"] = new["object_score"]
        new["score_policy"] = variant.get("score_policy", "")
        new["readout_mode"] = "v100_f2_scaffold_with_v103_paf_affinity"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return _dedupe_object_frame_rows(out)


def _noop_rows(scene_cache: dict[str, Any], variant: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}
    for oid in scene_cache["object_ids"]:
        new_oid = f"{variant['variant_id']}:{oid}"
        mapping[str(oid)] = new_oid
    for row in scene_cache["rows"]:
        scores[mapping[str(row["mv_object_id"])]] = max(
            scores.get(mapping[str(row["mv_object_id"])], 0.0),
            _safe_float(row.get("score")),
        )
    rows = _rows_from_mapping(scene_cache=scene_cache, variant=variant, object_mapping=mapping, component_score=scores)
    return rows, []


def _merge_rows(scene_cache: dict[str, Any], variant: dict[str, Any], pair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_ids = list(scene_cache["object_ids"])
    obj_index = {oid: idx for idx, oid in enumerate(object_ids)}
    rows_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_cache["rows"]:
        rows_by_object[str(row["mv_object_id"])].append(row)
    uf = UnionFind(len(object_ids))
    comp_frames = {idx: {int(row["frame_id"]) for row in rows_by_object[oid]} for oid, idx in obj_index.items()}
    accepted_rows: list[dict[str, Any]] = []
    threshold = float(variant["threshold"])
    candidates = sorted(
        [row for row in pair_rows if float(row["paf_object_affinity"]) >= threshold],
        key=lambda row: float(row["paf_object_affinity"]),
        reverse=True,
    )
    for rank, row in enumerate(candidates):
        a = int(row["object_index_a"])
        b = int(row["object_index_b"])
        ra = uf.find(a)
        rb = uf.find(b)
        accepted = False
        reject_reason = ""
        if ra == rb:
            reject_reason = "already_same_component"
        elif comp_frames[ra] & comp_frames[rb]:
            reject_reason = "same_frame_component_conflict"
        else:
            new_root = uf.union(ra, rb)
            old_a = comp_frames.pop(ra, set())
            old_b = comp_frames.pop(rb, set())
            comp_frames[new_root] = old_a | old_b
            accepted = True
        new = dict(row)
        new["variant_id"] = variant["variant_id"]
        new["candidate_rank"] = rank
        new["accepted_merge"] = bool(accepted)
        new["reject_reason"] = reject_reason
        accepted_rows.append(new)
    components: dict[int, list[str]] = defaultdict(list)
    for oid, idx in obj_index.items():
        components[uf.find(idx)].append(oid)
    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}
    for comp_idx, (_root, members) in enumerate(sorted(components.items(), key=lambda item: item[0])):
        new_oid = f"{variant['variant_id']}:{scene_cache['scene_id']}:c0000:obj_{comp_idx:05d}"
        for oid in members:
            mapping[oid] = new_oid
            for row in rows_by_object[oid]:
                scores[new_oid] = max(scores.get(new_oid, 0.0), _safe_float(row.get("score")))
    rows = _rows_from_mapping(scene_cache=scene_cache, variant=variant, object_mapping=mapping, component_score=scores)
    return rows, accepted_rows


def _split_rows(scene_cache: dict[str, Any], variant: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(scene_cache["rows"])
    row_feature = _normalize_rows(scene_cache["row_feature"].to(torch.float32)).to(device)
    by_object: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_object[str(row["mv_object_id"])].append(idx)
    mapping_by_row: dict[int, str] = {}
    split_audit_rows: list[dict[str, Any]] = []
    object_counter = 0
    for oid in scene_cache["object_ids"]:
        idxs = by_object[str(oid)]
        mapped = [idx for idx in idxs if int(rows[idx]["selected_mask_observation_index"]) >= 0]
        split_this = False
        if len(mapped) >= 4:
            feat = row_feature[torch.as_tensor(mapped, dtype=torch.long, device=device)]
            sim = (feat @ feat.T).detach().cpu().numpy()
            tri = sim[np.triu_indices(sim.shape[0], k=1)]
            internal = float(np.mean(tri)) if tri.size else 1.0
            split_this = internal < float(variant["low_coherence_threshold"])
        else:
            sim = np.eye(len(mapped), dtype=np.float32)
            internal = 1.0
        if not split_this:
            new_oid = f"{variant['variant_id']}:{scene_cache['scene_id']}:c0000:obj_{object_counter:05d}"
            object_counter += 1
            for idx in idxs:
                mapping_by_row[idx] = new_oid
            split_audit_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6c_split_audit_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant["variant_id"],
                    "scene_id": scene_cache["scene_id"],
                    "source_mv_object_id": oid,
                    "split_applied": False,
                    "source_row_count": len(idxs),
                    "mapped_row_count": len(mapped),
                    "internal_coherence": internal,
                    "component_count": 1,
                    "uses_gt_for_prediction": False,
                }
            )
            continue
        local_pos = {idx: pos for pos, idx in enumerate(mapped)}
        uf = UnionFind(len(mapped))
        comp_frames = {pos: {int(rows[idx]["frame_id"])} for idx, pos in local_pos.items()}
        pair_candidates: list[tuple[float, int, int]] = []
        for i in range(len(mapped) - 1):
            for j in range(i + 1, len(mapped)):
                pair_candidates.append((float(sim[i, j]), i, j))
        for score, i, j in sorted(pair_candidates, reverse=True):
            if score < float(variant["row_threshold"]):
                continue
            ri = uf.find(i)
            rj = uf.find(j)
            if ri == rj or (comp_frames[ri] & comp_frames[rj]):
                continue
            new_root = uf.union(ri, rj)
            old_i = comp_frames.pop(ri, set())
            old_j = comp_frames.pop(rj, set())
            comp_frames[new_root] = old_i | old_j
        comps: dict[int, list[int]] = defaultdict(list)
        for idx in idxs:
            if idx in local_pos:
                comps[uf.find(local_pos[idx])].append(idx)
            else:
                comps[-idx - 1].append(idx)
        for _root, members in sorted(comps.items(), key=lambda item: min(item[1])):
            new_oid = f"{variant['variant_id']}:{scene_cache['scene_id']}:c0000:obj_{object_counter:05d}"
            object_counter += 1
            for idx in members:
                mapping_by_row[idx] = new_oid
        split_audit_rows.append(
            {
                "schema_version": "stream4d_v103_phase6c_split_audit_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant["variant_id"],
                "scene_id": scene_cache["scene_id"],
                "source_mv_object_id": oid,
                "split_applied": True,
                "source_row_count": len(idxs),
                "mapped_row_count": len(mapped),
                "internal_coherence": internal,
                "component_count": len(comps),
                "uses_gt_for_prediction": False,
            }
        )
    mapping = {str(rows[idx]["mv_object_id"]): str(rows[idx]["mv_object_id"]) for idx in range(len(rows))}
    component_score: dict[str, float] = {}
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        new_oid = mapping_by_row[idx]
        component_score[new_oid] = max(component_score.get(new_oid, 0.0), len({int(r["frame_id"]) for r in rows if mapping_by_row.get(rows.index(r), "") == new_oid}) / 32.0)
        new = dict(row)
        new["schema_version"] = "stream4d_v103_phase6c_local_object_frame_mask_row_v1"
        new["phase_id"] = PHASE_ID
        new["variant_id"] = variant["variant_id"]
        new["scaffold_source_phase_id"] = row.get("phase_id", "")
        new["source_mv_object_id"] = str(row["mv_object_id"])
        new["mv_object_id"] = new_oid
        new["object_id"] = new_oid
        new["readout_mode"] = "v100_f2_scaffold_with_v103_paf_split"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    frame_counts: dict[str, int] = defaultdict(int)
    for oid in {str(row["mv_object_id"]) for row in out}:
        frame_counts[oid] = len({int(row["frame_id"]) for row in out if str(row["mv_object_id"]) == oid})
    for row in out:
        score = float(frame_counts[str(row["mv_object_id"])] / 32.0 + 1e-4 * _safe_float(row.get("score")))
        row["object_score"] = score
        row["score"] = score
        row["score_policy"] = variant.get("score_policy", "")
    return _dedupe_object_frame_rows(out), split_audit_rows


def _build_variant_scene_rows(
    *,
    scene_cache: dict[str, Any],
    variant: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mode = str(variant["mode"])
    if mode == "noop":
        rows, audit = _noop_rows(scene_cache, variant)
        return rows, audit, []
    if mode == "merge":
        rows, audit = _merge_rows(scene_cache, variant, pair_rows)
        return rows, audit, []
    if mode == "split":
        rows, split_audit = _split_rows(scene_cache, variant, device)
        return rows, [], split_audit
    raise ValueError(f"unsupported variant mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase6c F2 scaffold affinity repair using v103 mask-level primitive affinity features.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--scaffold-rows", default=str(DEFAULT_SCAFFOLD_ROWS))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(DEFAULT_BASELINE_ROWS))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0000")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--no-reuse-feature-cache", action="store_true")
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = p6._project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase5_root = p6._project(args.phase5_root)
    phase5_summary = _read_json(phase5_root / "summary.json")
    if not bool(phase5_summary.get("phase5_pass")):
        raise RuntimeError(f"Phase6c requires Phase5 pass: {phase5_root}")
    phase2_roots = {
        "scene0011_00": p6._project(args.scene0011_phase2_root),
        "scene0050_00": p6._project(args.scene0050_phase2_root),
    }
    phase2_summaries = {scene: _read_json(root / "summary.json") for scene, root in phase2_roots.items()}
    baseline = _load_baseline(p6._project(args.baseline_rows))

    scaffold_path = p6._project(args.scaffold_rows)
    df = pd.read_parquet(scaffold_path)
    df = df[
        (df["dataset_split"].astype(str) == str(args.dataset_split))
        & (df["scene_id"].astype(str).isin(sorted(phase2_roots)))
        & (df["chunk_id"].astype(str) == str(args.chunk_id))
    ].copy()
    if df.empty:
        raise RuntimeError(f"no scaffold rows after filter split={args.dataset_split} chunk={args.chunk_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scene_caches: dict[str, dict[str, Any]] = {}
    scaffold_feature_rows: list[dict[str, Any]] = []
    object_pair_rows: list[dict[str, Any]] = []
    for scene in sorted(phase2_roots):
        scene_df = df[df["scene_id"].astype(str) == scene].copy()
        cache = _load_or_build_scene_cache(
            scene=scene,
            rows_df=scene_df,
            phase5_root=phase5_root,
            phase2_summary=phase2_summaries[scene],
            out=out,
            reuse_feature_cache=not bool(args.no_reuse_feature_cache),
        )
        scene_caches[scene] = cache
        scaffold_feature_rows.extend(cache["object_rows"])
        object_pair_rows.extend(_object_pair_rows(cache, device))

    all_frame_rows: list[dict[str, Any]] = []
    all_merge_rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    control_metric: dict[str, Any] | None = None
    for variant in VARIANTS:
        scene_rows: dict[str, list[dict[str, Any]]] = {}
        for scene, cache in scene_caches.items():
            pair_rows_scene = [row for row in object_pair_rows if row["scene_id"] == scene]
            frame_rows, merge_rows, split_rows = _build_variant_scene_rows(
                scene_cache=cache,
                variant=variant,
                pair_rows=pair_rows_scene,
                device=device,
            )
            scene_rows[scene] = frame_rows
            all_frame_rows.extend(frame_rows)
            all_merge_rows.extend(merge_rows)
            all_split_rows.extend(split_rows)
        window_rows, metric_row, selected_rows, _pixel_collisions, _missing, _frames = p6._evaluate_variant(
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
            row["schema_version"] = "stream4d_v103_phase6c_window_metric_row_v1"
        metric_row["phase_id"] = PHASE_ID
        metric_row["schema_version"] = "stream4d_v103_phase6c_metric_row_v1"
        metric_row["variant_family"] = variant["family"]
        metric_row["repair_mode"] = variant["mode"]
        metric_row["threshold"] = variant.get("threshold", "")
        metric_row["score_policy"] = variant.get("score_policy", "")
        metric_row["accepted_merge_count"] = sum(1 for row in all_merge_rows if row.get("variant_id") == variant["variant_id"] and bool(row.get("accepted_merge")))
        metric_row["split_applied_count"] = sum(1 for row in all_split_rows if row.get("variant_id") == variant["variant_id"] and bool(row.get("split_applied")))
        if variant["family"] == "control" and control_metric is None:
            control_metric = dict(metric_row)
        all_window_rows.extend(window_rows)
        all_metric_rows.append(metric_row)
        all_selected_rows.extend(selected_rows)

    if control_metric is None:
        raise RuntimeError("missing scaffold control metric")
    control_ap = float(control_metric.get("MV_AP_window", 0.0))
    control_ap50 = float(control_metric.get("MV_AP50_window", 0.0))
    baseline_ap = float(baseline.get("MV_AP_window", 0.0))
    baseline_ap50 = float(baseline.get("MV_AP50_window", 0.0))
    best_noncontrol = max(
        [row for row in all_metric_rows if row.get("variant_family") != "control"],
        key=lambda row: (float(row.get("MV_AP_window", 0.0)), float(row.get("MV_AP50_window", 0.0))),
        default={},
    )
    for row in all_metric_rows:
        is_control = row.get("variant_family") == "control"
        checks = [
            ("same_frame_collision_count_eq_0", int(row["same_frame_collision_count"]) == 0, row["same_frame_collision_count"], 0),
            ("pixel_collision_rate_le_0p02", float(row["pixel_collision_rate"]) <= 0.02, row["pixel_collision_rate"], 0.02),
            ("missing_mask_raster_count_eq_0", int(row["missing_mask_raster_count"]) == 0, row["missing_mask_raster_count"], 0),
            ("uses_gt_for_prediction_false", not bool(row["uses_gt_for_prediction"]), row["uses_gt_for_prediction"], False),
            ("uses_future_false", not bool(row["uses_future"]), row["uses_future"], False),
        ]
        if not is_control:
            checks.extend(
                [
                    ("MV_AP_window_ge_scaffold_control_minus_0p003", float(row["MV_AP_window"]) >= control_ap - 0.003, row["MV_AP_window"], control_ap - 0.003),
                    ("MV_AP50_window_ge_scaffold_control_minus_0p006", float(row["MV_AP50_window"]) >= control_ap50 - 0.006, row["MV_AP50_window"], control_ap50 - 0.006),
                    ("MV_AP_window_gain_ge_0p002_vs_scaffold_control", float(row["MV_AP_window"]) >= control_ap + 0.002, row["MV_AP_window"], control_ap + 0.002),
                ]
            )
        else:
            checks.extend(
                [
                    ("control_MV_AP_window_ge_global_baseline_minus_0p003_context_only", float(row["MV_AP_window"]) >= baseline_ap - 0.003, row["MV_AP_window"], baseline_ap - 0.003),
                    ("control_MV_AP50_window_ge_global_baseline_minus_0p006_context_only", float(row["MV_AP50_window"]) >= baseline_ap50 - 0.006, row["MV_AP50_window"], baseline_ap50 - 0.006),
                ]
            )
        for name, ok, observed, required in checks:
            gate_rows.append(
                {
                    "schema_version": "stream4d_v103_phase6c_gate_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": row["variant_id"],
                    "gate_name": name,
                    "pass": bool(ok),
                    "observed": observed,
                    "required": required,
                }
            )
            if row.get("variant_id") == best_noncontrol.get("variant_id") and not bool(ok):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase6c_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": row["variant_id"],
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": "D4RT primitive-affinity on the F2 scaffold did not produce a safe non-control gain; inspect candidate merge coverage, object-signature coherence, and Phase2/3 support before DA3.",
                    }
                )

    _write_csv(out / "scaffold_feature_rows.csv", scaffold_feature_rows)
    _write_csv(out / "scaffold_object_pair_rows.csv", object_pair_rows)
    _write_csv(out / "scaffold_affinity_frame_mask_rows.csv", all_frame_rows)
    _write_csv(out / "scaffold_affinity_selected_rows.csv", all_selected_rows)
    _write_csv(out / "scaffold_affinity_merge_candidate_rows.csv", all_merge_rows)
    _write_csv(out / "scaffold_affinity_split_rows.csv", all_split_rows)
    _write_csv(out / "scaffold_affinity_metric_rows.csv", all_metric_rows)
    _write_csv(out / "scaffold_affinity_window_rows.csv", all_window_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    if all_selected_rows:
        pd.DataFrame(all_selected_rows).to_parquet(out / "scaffold_affinity_selected_rows.parquet", index=False)

    noncontrol_gain = bool(
        best_noncontrol
        and float(best_noncontrol.get("MV_AP_window", 0.0)) >= control_ap + 0.002
        and float(best_noncontrol.get("MV_AP_window", 0.0)) >= control_ap - 0.003
        and float(best_noncontrol.get("MV_AP50_window", 0.0)) >= control_ap50 - 0.006
        and int(best_noncontrol.get("same_frame_collision_count", 1)) == 0
        and float(best_noncontrol.get("pixel_collision_rate", 1.0)) <= 0.02
    )
    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    summary = {
        "schema_version": "stream4d_v103_phase6c_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_SCAFFOLD_AFFINITY_NONCONTROL_GAIN_RECHECK_PHASE6" if noncontrol_gain else "NO_GO_PHASE6C_PAF_ON_F2_SCAFFOLD_NO_SAFE_GAIN",
        "phase6c_pass": noncontrol_gain,
        "failure_count": len(failure_rows),
        "control_variant_id": control_metric.get("variant_id"),
        "control_MV_AP_window": control_metric.get("MV_AP_window"),
        "control_MV_AP50_window": control_metric.get("MV_AP50_window"),
        "best_noncontrol_variant_id": best_noncontrol.get("variant_id", ""),
        "best_noncontrol_MV_AP_window": best_noncontrol.get("MV_AP_window", ""),
        "best_noncontrol_MV_AP50_window": best_noncontrol.get("MV_AP50_window", ""),
        "best_noncontrol_accepted_merge_count": best_noncontrol.get("accepted_merge_count", ""),
        "best_noncontrol_split_applied_count": best_noncontrol.get("split_applied_count", ""),
        "global_baseline_contract": baseline,
        "metric_scope": "first32_dev_subset_c0000_scene0011_scene0050; scaffold control is context, not a v103 method success",
        "scene_cache_rows": [
            {
                "scene_id": scene,
                "row_count": cache["row_count"],
                "mapped_row_count": cache["mapped_row_count"],
                "mapped_row_rate": cache["mapped_row_rate"],
                "feature_cache": p6._rel(_scene_cache_path(out, scene)),
                "feature_cache_hit": bool(cache.get("cache_hit", False)),
            }
            for scene, cache in sorted(scene_caches.items())
        ],
        "variant_count": len(VARIANTS),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "iou_backend_requested": "cupy" if not bool(args.disable_cupy_iou) else "cpu",
        "gpu_device": str(device),
        "gpu_memory_peak_MB": peak_mb,
        "truthfulness_note": "This runner uses v100 F2 rows only as a mask-view scaffold/control and applies v103 D4RT primitive-affinity features as GT-free merge/split evidence. It does not claim raw v103 object birth success from the scaffold control.",
        "outputs": {
            "summary": p6._rel(out / "summary.json"),
            "scaffold_feature_rows": p6._rel(out / "scaffold_feature_rows.csv"),
            "scaffold_object_pair_rows": p6._rel(out / "scaffold_object_pair_rows.csv"),
            "scaffold_affinity_metric_rows": p6._rel(out / "scaffold_affinity_metric_rows.csv"),
            "scaffold_affinity_window_rows": p6._rel(out / "scaffold_affinity_window_rows.csv"),
            "scaffold_affinity_merge_candidate_rows": p6._rel(out / "scaffold_affinity_merge_candidate_rows.csv"),
            "gate_rows": p6._rel(out / "gate_rows.csv"),
            "failure_rows": p6._rel(out / "failure_rows.csv"),
            "last_command": p6._rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if noncontrol_gain else 2


if __name__ == "__main__":
    raise SystemExit(main())
