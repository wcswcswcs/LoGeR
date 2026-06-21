#!/usr/bin/env python3
"""Phase 3 semantic explanation of geometry-memory cues for ACL2 v73."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    auc_binary,
    best_auc,
    find_chunk_dir,
    load_json,
    parse_chunks,
    safe_float,
    spearman,
    torch_load,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_OUT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase3_semantic_explanation")
DEFAULT_REPORT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final")
DEFAULT_STAGE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_RADIO = Path("results/kitti_preprocess/01/radseg_sidecar_chunks_slide336_stride224")


STABLE_WORDS = ("wall", "fence", "pole", "traffic sign", "bridge", "building", "house", "construction")
DYNAMIC_WORDS = ("person", "car", "truck", "bus", "bicycle", "motorcycle", "cyclist")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "void")
ROAD_WORDS = ("road", "ground", "crosswalk")
SKY_WORDS = ("sky",)


def _label_ids(names: list[str], words: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    lowered = [name.lower() for name in names]
    for idx, name in enumerate(lowered):
        if any(word in name for word in words):
            out.append(idx)
    return out


def _semantic_features(stage_dir: Path, chunk_id: int) -> dict[str, Any]:
    chunk_dir = find_chunk_dir(stage_dir, chunk_id)
    out: dict[str, Any] = {
        "chunk_id": chunk_id,
        "semantic_available": False,
        "semantic_confidence_mean": None,
        "semantic_nonvoid_ratio": None,
        "stable_structure_ratio": None,
        "dynamic_thing_ratio": None,
        "lowtrust_stuff_ratio": None,
        "road_context_ratio": None,
        "sky_context_ratio": None,
        "thing_source_ratio": None,
        "stuff_source_ratio": None,
    }
    if chunk_dir is None:
        return out
    path = chunk_dir / "masklet.pt"
    if not path.exists():
        return out
    try:
        payload = torch_load(path)
        sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    except Exception as exc:
        out["semantic_error"] = type(exc).__name__
        return out
    if not isinstance(sem, dict) or not hasattr(sem.get("label_maps"), "shape"):
        return out
    labels = sem["label_maps"].detach().cpu().numpy()
    conf = sem.get("confidence_maps")
    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.ones_like(labels, dtype=np.float32)
    names = [str(x) for x in sem.get("label_names", [])]
    nonvoid = labels != 0
    denom = max(int(labels.size), 1)
    out["semantic_available"] = True
    out["semantic_confidence_mean"] = float(np.mean(np.clip(conf_np.astype(np.float32), 0.0, 1.0)))
    out["semantic_nonvoid_ratio"] = float(np.mean(nonvoid))
    for key, words in (
        ("stable_structure_ratio", STABLE_WORDS),
        ("dynamic_thing_ratio", DYNAMIC_WORDS),
        ("lowtrust_stuff_ratio", LOWTRUST_WORDS),
        ("road_context_ratio", ROAD_WORDS),
        ("sky_context_ratio", SKY_WORDS),
    ):
        ids = _label_ids(names, words)
        out[key] = float(np.isin(labels, np.asarray(ids, dtype=labels.dtype)).sum() / denom) if ids else 0.0
    source = payload.get("source_type") if isinstance(payload, dict) else []
    if isinstance(source, list) and source:
        lowered = [str(x).lower() for x in source]
        out["thing_source_ratio"] = float(sum("thing" in x for x in lowered) / len(lowered))
        out["stuff_source_ratio"] = float(sum("stuff" in x or "structure" in x for x in lowered) / len(lowered))
    return out


def _radio_features(radio_dir: Path, chunk_id: int) -> dict[str, Any]:
    chunk_dir = find_chunk_dir(radio_dir, chunk_id)
    out: dict[str, Any] = {
        "chunk_id": chunk_id,
        "radio_available": False,
        "radio_static_mean": None,
        "radio_dynamic_mean": None,
        "radio_lowtrust_mean": None,
        "radio_sky_mean": None,
        "radio_boundary_mean": None,
        "radio_interior_mean": None,
        "radio_temporal_stability_mean": None,
        "radio_component_count_mean": None,
    }
    if chunk_dir is None:
        return out
    path = chunk_dir / "radio_sidecar.pt"
    if not path.exists():
        return out
    try:
        payload = torch_load(path)
    except Exception as exc:
        out["radio_error"] = type(exc).__name__
        return out
    out["radio_available"] = True
    for key, dest in (
        ("radio_static_score", "radio_static_mean"),
        ("radio_dynamic_score", "radio_dynamic_mean"),
        ("radio_lowtrust_score", "radio_lowtrust_mean"),
        ("radio_sky_context_score", "radio_sky_mean"),
        ("object_boundary_score", "radio_boundary_mean"),
        ("object_interior_score", "radio_interior_mean"),
        ("temporal_stability", "radio_temporal_stability_mean"),
    ):
        value = payload.get(key) if isinstance(payload, dict) else None
        out[dest] = float(value.float().mean().item()) if hasattr(value, "float") else None
    comp = payload.get("object_component_id") if isinstance(payload, dict) else None
    if hasattr(comp, "shape"):
        counts = []
        arr = comp.detach().cpu().numpy()
        for frame in arr:
            counts.append(len(set(int(x) for x in np.unique(frame) if int(x) >= 0)))
        out["radio_component_count_mean"] = float(np.mean(counts)) if counts else None
    return out


def _score(values: list[Any], labels: list[Any]) -> dict[str, Any]:
    auc = best_auc(values, labels)
    return {
        "spearman": spearman(values, labels),
        "auc": auc.get("auc"),
        "best_auc": auc.get("best_auc"),
        "direction": auc.get("direction"),
    }


def _best_feature(rows: list[dict[str, Any]], features: list[str], label: str) -> dict[str, Any]:
    best: dict[str, Any] = {"feature": None, "best_auc": None}
    for feature in features:
        metrics = _score([row.get(feature) for row in rows], [row.get(label) for row in rows])
        score = safe_float(metrics.get("best_auc")) or 0.0
        if best["best_auc"] is None or score > float(best["best_auc"]):
            best = {"feature": feature, **metrics}
    return best


def _conditioned_scores(rows: list[dict[str, Any]], geom: str, stable: str, harm: str) -> list[float | None]:
    geom_vals = [safe_float(row.get(geom)) for row in rows]
    stable_vals = [safe_float(row.get(stable)) for row in rows]
    harm_vals = [safe_float(row.get(harm)) for row in rows]
    def norm(vals: list[float | None]) -> list[float | None]:
        finite = [v for v in vals if v is not None]
        if not finite:
            return [None for _ in vals]
        mean = float(np.mean(finite))
        std = float(np.std(finite)) or 1.0
        return [None if v is None else float((v - mean) / std) for v in vals]
    gz, sz, hz = norm(geom_vals), norm(stable_vals), norm(harm_vals)
    out: list[float | None] = []
    for g, s, h in zip(gz, sz, hz):
        if g is None or s is None or h is None:
            out.append(None)
        else:
            out.append(float(g + h - s))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stage-c-cache", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--radio-sidecar-dir", type=Path, default=DEFAULT_RADIO)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cue_payload = load_json(args.report_root / "phase2_geometry_memory_cue_ledger" / "geometry_cue_by_chunk.json") or {}
    cue_rows = cue_payload.get("rows", []) if isinstance(cue_payload, dict) else []
    cue_by_chunk = {int(row["chunk_id"]): row for row in cue_rows if row.get("chunk_id") is not None}
    rows: list[dict[str, Any]] = []
    for chunk_id in parse_chunks(args.target_chunks):
        row = dict(cue_by_chunk.get(chunk_id, {"chunk_id": chunk_id}))
        row.update(_semantic_features(args.stage_c_cache, chunk_id))
        row.update(_radio_features(args.radio_sidecar_dir, chunk_id))
        rows.append(row)

    geometry_features = [
        "D_geo_mean_patch",
        "D_geo_q90_patch",
        "global_k_layer5_gram_motion",
        "global_k_layer7_gram_motion",
        "raw_overlap_residual_rmse",
        "merge_transform_abs_log_scale",
        "ttt_write_score_mean",
    ]
    semantic_features = [
        "semantic_confidence_mean",
        "semantic_nonvoid_ratio",
        "stable_structure_ratio",
        "dynamic_thing_ratio",
        "lowtrust_stuff_ratio",
        "road_context_ratio",
        "sky_context_ratio",
        "radio_static_mean",
        "radio_dynamic_mean",
        "radio_lowtrust_mean",
        "radio_boundary_mean",
        "radio_interior_mean",
        "radio_temporal_stability_mean",
    ]
    harm_features = ["dynamic_thing_ratio", "lowtrust_stuff_ratio", "sky_context_ratio", "radio_dynamic_mean", "radio_lowtrust_mean", "radio_boundary_mean"]
    stable_features = ["stable_structure_ratio", "road_context_ratio", "radio_static_mean", "radio_interior_mean", "radio_temporal_stability_mean"]
    eval_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(73)
    for label in ("Y_short", "Y_mid", "Y_scale_drift"):
        geom_best = _best_feature(rows, geometry_features, label)
        sem_best = _best_feature(rows, semantic_features, label)
        best_conditioned: dict[str, Any] = {"best_auc": None}
        labels = [row.get(label) for row in rows]
        for geom in geometry_features:
            for stable in stable_features:
                for harm in harm_features:
                    scores = _conditioned_scores(rows, geom, stable, harm)
                    metrics = _score(scores, labels)
                    score = safe_float(metrics.get("best_auc")) or 0.0
                    if best_conditioned.get("best_auc") is None or score > float(best_conditioned["best_auc"]):
                        best_conditioned = {"geom_feature": geom, "stable_feature": stable, "harm_feature": harm, **metrics}
        shuffle_aucs: list[float] = []
        for _ in range(20):
            shuffled_rows = [dict(row) for row in rows]
            for feature in semantic_features:
                vals = [row.get(feature) for row in shuffled_rows]
                rng.shuffle(vals)
                for row, val in zip(shuffled_rows, vals):
                    row[feature] = val
            scores = _conditioned_scores(
                shuffled_rows,
                best_conditioned.get("geom_feature"),
                best_conditioned.get("stable_feature"),
                best_conditioned.get("harm_feature"),
            )
            auc = best_auc(scores, labels).get("best_auc")
            if auc is not None:
                shuffle_aucs.append(float(auc))
        geom_auc = safe_float(geom_best.get("best_auc"))
        cond_auc = safe_float(best_conditioned.get("best_auc"))
        shuffle_max = float(np.max(shuffle_aucs)) if shuffle_aucs else None
        gate = bool(
            cond_auc is not None
            and geom_auc is not None
            and (cond_auc - geom_auc) >= 0.05
            and (shuffle_max is None or cond_auc > shuffle_max)
        )
        eval_rows.append(
            {
                "label": label,
                "geometry_best_feature": geom_best.get("feature"),
                "geometry_best_auc": geom_best.get("best_auc"),
                "semantic_best_feature": sem_best.get("feature"),
                "semantic_best_auc": sem_best.get("best_auc"),
                "conditioned_geom_feature": best_conditioned.get("geom_feature"),
                "conditioned_stable_feature": best_conditioned.get("stable_feature"),
                "conditioned_harm_feature": best_conditioned.get("harm_feature"),
                "conditioned_best_auc": best_conditioned.get("best_auc"),
                "conditioned_spearman": best_conditioned.get("spearman"),
                "shuffle_best_auc_max": shuffle_max,
                "shuffle_best_auc_median": float(np.median(shuffle_aucs)) if shuffle_aucs else None,
                "phase3_gate_pass": gate,
                "gate_rule": "conditioned AUC >= geometry AUC + 0.05 and greater than shuffled max",
            }
        )

    summary = {
        "schema": "acl2_v73_phase3_semantic_explains_geometry_v1",
        "created_at": utc_now(),
        "rows": len(rows),
        "target_chunks": parse_chunks(args.target_chunks),
        "phase3_any_gate_pass": any(bool(row.get("phase3_gate_pass")) for row in eval_rows),
        "diagnostic_scope": "Ledger/cue diagnostic only; no online memory action is claimed.",
        "gate_rows": eval_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "semantic_geometry_features_by_chunk.csv", rows)
    write_json(args.out_dir / "semantic_geometry_features_by_chunk.json", {"summary": summary, "rows": rows})
    write_csv(args.out_dir / "semantic_explanation_gate_rows.csv", eval_rows)
    write_json(args.out_dir / "semantic_explanation_summary.json", summary)
    print({"out_dir": str(args.out_dir), "phase3_any_gate_pass": summary["phase3_any_gate_pass"]})


if __name__ == "__main__":
    main()
