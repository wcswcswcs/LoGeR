#!/usr/bin/env python3
"""GT-only diagnostic for v100 local2history link precision/coverage.

This script does not produce a method artifact and must not be used as
prediction evidence. It labels already-produced local2history links with
scene-level GT instance signatures to decide whether the next repair should
focus on false-merge prevention or missing-link coverage.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v100_phase4h_overlap3_exact_history_memory as p4h  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase4n_gt_link_precision_diagnostic"
PHASE4H_DIR = AUDIT_ROOT / "v100_phase4h_overlap3_exact_history_memory"
PHASE4K_DIR = AUDIT_ROOT / "v100_phase4k_phase2c_semantic_scene_repair"
PHASE4M_DIR = AUDIT_ROOT / "v100_phase4m_temporal_maskview_history_memory"

TOP_GT_MIN_PIXELS = 100
TOP_GT_MIN_FRACTION = 0.25

_FRAME_CACHE: OrderedDict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_FRAME_CACHE_LIMIT = 8


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _jsonable(value: Any) -> Any:
    return p4h._jsonable(value)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _sha256(path: Path) -> str:
    return p4h._sha256(path)


def _load_frame(scope: dict[str, Any], *, split: str, scene: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
    key = (split, scene, int(frame))
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        _FRAME_CACHE.move_to_end(key)
        return cached
    mask_path = scope["mask_path_by_frame"].get((scene, int(frame)))
    if mask_path is None or not Path(mask_path).exists():
        label = np.zeros((968, 1296), dtype=np.int64)
    else:
        label = p1._read_label(Path(mask_path))
    shape_hw = tuple(int(v) for v in label.shape[:2])
    gt = p1._load_gt_2d(scene, int(frame), shape_hw)
    _FRAME_CACHE[key] = (label, gt)
    _FRAME_CACHE.move_to_end(key)
    while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
        _FRAME_CACHE.popitem(last=False)
    return label, gt


def _object_signatures(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    scopes = {split: p4h._scope_for_split(split) for split in ["dev", "holdout"]}
    signatures: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        counts_by_object: dict[str, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))
        positive_by_object: dict[str, int] = defaultdict(int)
        frame_count_by_object: dict[str, set[int]] = defaultdict(set)
        scene_by_object: dict[str, str] = {}
        chunk_by_object: dict[str, str] = {}
        for row in rows:
            oid = str(row["mv_object_id"])
            scene = str(row["scene_id"])
            frame = int(_num(row["frame_id"], -1))
            mask_id = int(_num(row["selected_mask_id"], -1))
            if frame < 0 or mask_id < 0:
                continue
            label, gt = _load_frame(scopes[split], split=split, scene=scene, frame=frame)
            mask = label == mask_id
            if not np.any(mask):
                continue
            vals, counts = np.unique(gt[mask], return_counts=True)
            pos = 0
            for raw, count in zip(vals, counts):
                raw_i = int(raw)
                count_i = int(count)
                if raw_i <= 0:
                    continue
                counts_by_object[oid][(scene, raw_i)] += count_i
                pos += count_i
            positive_by_object[oid] += pos
            frame_count_by_object[oid].add(frame)
            scene_by_object[oid] = scene
            chunk_by_object[oid] = str(row["chunk_id"])
        for oid, counts in counts_by_object.items():
            positive = int(positive_by_object.get(oid, 0))
            if counts:
                top_key, top_pixels = max(counts.items(), key=lambda item: (item[1], item[0]))
            else:
                top_key, top_pixels = ("", -1), 0
            top_fraction = float(top_pixels / max(1, positive))
            valid = positive >= TOP_GT_MIN_PIXELS and top_fraction >= TOP_GT_MIN_FRACTION and bool(counts)
            signatures[oid] = {
                "mv_object_id": oid,
                "dataset_split": split,
                "scene_id": scene_by_object.get(oid, ""),
                "chunk_id": chunk_by_object.get(oid, ""),
                "top_gt_key": f"{top_key[0]}:{top_key[1]}" if valid else "",
                "top_gt_scene": top_key[0] if valid else "",
                "top_gt_instance_id": top_key[1] if valid else "",
                "top_gt_pixels": int(top_pixels),
                "positive_gt_pixels": positive,
                "top_gt_fraction": top_fraction,
                "valid_gt_signature": valid,
                "gt_instance_count": len(counts),
                "support_frame_count": len(frame_count_by_object.get(oid, set())),
            }
        for row in rows:
            oid = str(row["mv_object_id"])
            signatures.setdefault(
                oid,
                {
                    "mv_object_id": oid,
                    "dataset_split": split,
                    "scene_id": str(row.get("scene_id", "")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "top_gt_key": "",
                    "top_gt_scene": "",
                    "top_gt_instance_id": "",
                    "top_gt_pixels": 0,
                    "positive_gt_pixels": 0,
                    "top_gt_fraction": 0.0,
                    "valid_gt_signature": False,
                    "gt_instance_count": 0,
                    "support_frame_count": 0,
                },
            )
    return signatures


def _load_internal_rows_by_split() -> dict[str, list[dict[str, Any]]]:
    df = pd.read_parquet(PHASE4H_DIR / "internal_overlap_mv_object_frame_mask_rows.parquet")
    return {
        split: [dict(row) for row in sub.to_dict(orient="records")]
        for split, sub in df.groupby("dataset_split")
    }


def _link_label(source_id: str, variant_id: str, split: str, a: str, b: str, signatures: dict[str, dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    sig_a = signatures.get(a, {})
    sig_b = signatures.get(b, {})
    valid_a = bool(sig_a.get("valid_gt_signature"))
    valid_b = bool(sig_b.get("valid_gt_signature"))
    same = bool(valid_a and valid_b and sig_a.get("top_gt_key") == sig_b.get("top_gt_key"))
    different = bool(valid_a and valid_b and sig_a.get("top_gt_key") != sig_b.get("top_gt_key"))
    row = {
        "schema_version": "stream4d_v100_phase4n_link_gt_diagnostic_row_v1",
        "phase_id": "v100_phase4n_gt_link_precision_diagnostic",
        "source_id": source_id,
        "variant_id": variant_id,
        "dataset_split": split,
        "mv_object_id_a": a,
        "mv_object_id_b": b,
        "scene_id": sig_a.get("scene_id") or sig_b.get("scene_id") or "",
        "chunk_id_a": sig_a.get("chunk_id", ""),
        "chunk_id_b": sig_b.get("chunk_id", ""),
        "valid_gt_a": valid_a,
        "valid_gt_b": valid_b,
        "top_gt_key_a": sig_a.get("top_gt_key", ""),
        "top_gt_key_b": sig_b.get("top_gt_key", ""),
        "top_gt_fraction_a": sig_a.get("top_gt_fraction", 0.0),
        "top_gt_fraction_b": sig_b.get("top_gt_fraction", 0.0),
        "positive_gt_pixels_a": sig_a.get("positive_gt_pixels", 0),
        "positive_gt_pixels_b": sig_b.get("positive_gt_pixels", 0),
        "same_top_gt": same,
        "different_top_gt": different,
        "weak_or_missing_gt": not (valid_a and valid_b),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
    }
    if extra:
        row.update(extra)
    return row


def _accepted_merge_link_rows(path: Path, source_id: str, variant_id: str, signatures: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    df = pd.read_csv(path)
    df = df[df["variant_id"] == variant_id]
    ids = sorted(signatures)
    dsu = p4h.DSU(ids)
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        split = str(item["dataset_split"])
        a = str(item["mv_object_id_a"])
        b = str(item["mv_object_id_b"])
        if a in dsu.parent and b in dsu.parent:
            dsu.union(a, b)
        rows.append(
            _link_label(
                source_id,
                variant_id,
                split,
                a,
                b,
                signatures,
                {
                    "link_source_type": "accepted_pair_merge",
                    "candidate_family": item.get("candidate_family", ""),
                    "link_score": item.get("affinity", ""),
                    "link_margin": "",
                },
            )
        )
    return rows, {oid: dsu.find(oid) for oid in ids}


def _accepted_history_link_rows(path: Path, source_id: str, variant_id: str, signatures: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    df = pd.read_parquet(path)
    df = df[df["variant_id"] == variant_id].copy()
    df = df.sort_values(["dataset_split", "scene_id", "chunk_index", "chunk_id", "chunk_object_id"])
    ids = sorted(signatures)
    dsu = p4h.DSU(ids)
    state: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        split = str(item["dataset_split"])
        scene = str(item["scene_id"])
        hist = str(item["history_id"])
        current = str(item["chunk_object_id"])
        key = (split, scene, hist)
        action = str(item["action"])
        previous = list(state.get(key, []))
        if action == "accept_link":
            if current in dsu.parent:
                for prev in previous:
                    if prev in dsu.parent:
                        dsu.union(current, prev)
            if previous:
                same_count = 0
                diff_count = 0
                weak_count = 0
                valid_count = 0
                partner_rows: list[dict[str, Any]] = []
                for prev in previous:
                    label = _link_label(
                        source_id,
                        variant_id,
                        split,
                        prev,
                        current,
                        signatures,
                        {
                            "link_source_type": "history_accept_to_prior_member",
                            "candidate_family": "semantic_history",
                            "link_score": item.get("link_score", ""),
                            "link_margin": item.get("link_margin", ""),
                            "history_id": hist,
                            "history_previous_member_count": len(previous),
                        },
                    )
                    partner_rows.append(label)
                    if label["same_top_gt"]:
                        same_count += 1
                    elif label["different_top_gt"]:
                        diff_count += 1
                    else:
                        weak_count += 1
                    if label["valid_gt_a"] and label["valid_gt_b"]:
                        valid_count += 1
                best = sorted(
                    partner_rows,
                    key=lambda row: (
                        bool(row["same_top_gt"]),
                        _num(row.get("top_gt_fraction_a")) + _num(row.get("top_gt_fraction_b")),
                        _num(row.get("positive_gt_pixels_a")) + _num(row.get("positive_gt_pixels_b")),
                    ),
                    reverse=True,
                )[0]
                best["same_gt_partner_count"] = same_count
                best["different_gt_partner_count"] = diff_count
                best["weak_gt_partner_count"] = weak_count
                best["valid_gt_partner_count"] = valid_count
                rows.append(best)
        if action in {"birth_new_history", "accept_link"}:
            if current not in state[key]:
                state[key].append(current)
    return rows, {oid: dsu.find(oid) for oid in ids}


def _chunk_index(chunk_id: str) -> int:
    return p4h._chunk_index(chunk_id)


def _true_adjacent_pair_rows(
    source_id: str,
    variant_id: str,
    infos_by_split: dict[str, dict[str, dict[str, Any]]],
    signatures: dict[str, dict[str, Any]],
    component: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, infos in infos_by_split.items():
        by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
        chunks_by_scene: dict[str, set[str]] = defaultdict(set)
        for oid, info in infos.items():
            by_scene_chunk[(str(info["scene_id"]), str(info["chunk_id"]))].append(oid)
            chunks_by_scene[str(info["scene_id"])].add(str(info["chunk_id"]))
        for scene, chunk_set in sorted(chunks_by_scene.items()):
            chunks = sorted(chunk_set, key=_chunk_index)
            for left, right in zip(chunks[:-1], chunks[1:]):
                for a in sorted(by_scene_chunk[(scene, left)]):
                    sig_a = signatures.get(a, {})
                    if not sig_a.get("valid_gt_signature"):
                        continue
                    for b in sorted(by_scene_chunk[(scene, right)]):
                        sig_b = signatures.get(b, {})
                        if not sig_b.get("valid_gt_signature"):
                            continue
                        if sig_a.get("top_gt_key") != sig_b.get("top_gt_key"):
                            continue
                        rows.append(
                            {
                                "schema_version": "stream4d_v100_phase4n_true_adjacent_pair_row_v1",
                                "phase_id": "v100_phase4n_gt_link_precision_diagnostic",
                                "source_id": source_id,
                                "variant_id": variant_id,
                                "dataset_split": split,
                                "scene_id": scene,
                                "left_chunk_id": left,
                                "right_chunk_id": right,
                                "mv_object_id_a": a,
                                "mv_object_id_b": b,
                                "top_gt_key": sig_a.get("top_gt_key"),
                                "accepted_connected": bool(component.get(a) == component.get(b)),
                                "uses_gt_for_prediction": False,
                                "uses_gt_for_diagnostic": True,
                            }
                        )
    return rows


def _summaries(link_rows: list[dict[str, Any]], true_pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["source_id"], row["variant_id"], row["dataset_split"]) for row in link_rows + true_pair_rows})
    for source_id, variant_id, split in keys:
        links = [row for row in link_rows if row["source_id"] == source_id and row["variant_id"] == variant_id and row["dataset_split"] == split]
        pairs = [row for row in true_pair_rows if row["source_id"] == source_id and row["variant_id"] == variant_id and row["dataset_split"] == split]
        valid_links = [row for row in links if row["valid_gt_a"] and row["valid_gt_b"]]
        correct = [row for row in valid_links if row["same_top_gt"]]
        false = [row for row in valid_links if row["different_top_gt"]]
        weak = [row for row in links if row["weak_or_missing_gt"]]
        connected_true = [row for row in pairs if row["accepted_connected"]]
        out.append(
            {
                "schema_version": "stream4d_v100_phase4n_summary_row_v1",
                "phase_id": "v100_phase4n_gt_link_precision_diagnostic",
                "source_id": source_id,
                "variant_id": variant_id,
                "dataset_split": split,
                "accepted_link_count": len(links),
                "valid_gt_link_count": len(valid_links),
                "same_top_gt_link_count": len(correct),
                "different_top_gt_link_count": len(false),
                "weak_or_missing_gt_link_count": len(weak),
                "accepted_link_gt_precision": float(len(correct) / max(1, len(valid_links))),
                "false_link_rate_among_valid": float(len(false) / max(1, len(valid_links))),
                "true_adjacent_same_gt_pair_count": len(pairs),
                "accepted_connected_true_adjacent_pair_count": len(connected_true),
                "true_adjacent_pair_recall_proxy": float(len(connected_true) / max(1, len(pairs))),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
            }
        )
    return out


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_split = _load_internal_rows_by_split()
    signatures = _object_signatures(rows_by_split)
    features = {split: p4h._features_for_split(split) for split in ["dev", "holdout"]}
    infos_by_split = {split: p4h._object_infos(rows_by_split[split], features[split]) for split in ["dev", "holdout"]}

    phase4h_summary = json.loads((PHASE4H_DIR / "summary.json").read_text(encoding="utf-8"))
    phase4k_summary = json.loads((PHASE4K_DIR / "summary.json").read_text(encoding="utf-8"))
    phase4m_summary = json.loads((PHASE4M_DIR / "summary.json").read_text(encoding="utf-8"))
    specs = [
        ("phase4h_exact_adjacent", "HMO1_exact_overlap_frame_mask_adjacent", "merge_csv", PHASE4H_DIR / "local2history_merge_rows.csv"),
        ("phase4h_best", str(phase4h_summary["best_variant_id"]), "merge_csv", PHASE4H_DIR / "local2history_merge_rows.csv"),
        ("phase4k_best_semantic", str(phase4k_summary["best_variant_id"]), "history_parquet", PHASE4K_DIR / "chunk_object_history_link_rows.parquet"),
        ("phase4m_best_temporal_maskview", str(phase4m_summary["best_variant_id"]), "merge_csv", PHASE4M_DIR / "local2history_merge_rows.csv"),
    ]

    link_rows: list[dict[str, Any]] = []
    true_pair_rows: list[dict[str, Any]] = []
    for source_id, variant_id, kind, path in specs:
        if kind == "merge_csv":
            rows, component = _accepted_merge_link_rows(path, source_id, variant_id, signatures)
        else:
            rows, component = _accepted_history_link_rows(path, source_id, variant_id, signatures)
        link_rows.extend(rows)
        true_pair_rows.extend(_true_adjacent_pair_rows(source_id, variant_id, infos_by_split, signatures, component))

    summary_rows = _summaries(link_rows, true_pair_rows)
    signature_csv = OUT_DIR / "object_gt_signature_rows.csv"
    link_csv = OUT_DIR / "accepted_link_gt_diagnostic_rows.csv"
    true_pair_csv = OUT_DIR / "true_adjacent_pair_recall_rows.csv"
    summary_csv = OUT_DIR / "summary_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_csv(signature_csv, list(signatures.values()))
    _write_csv(link_csv, link_rows)
    _write_csv(true_pair_csv, true_pair_rows)
    _write_csv(summary_csv, summary_rows)
    artifacts = [
        (signature_csv, "csv", "GT-only top instance signatures for v100 chunk objects"),
        (link_csv, "csv", "Accepted local2history links labeled by GT signature"),
        (true_pair_csv, "csv", "GT-same adjacent chunk pairs and whether each source connected them"),
        (summary_csv, "csv", "Precision and recall proxy summary rows"),
    ]
    _write_csv(
        artifact_csv,
        [
            {
                "schema_version": "stream4d_v100_phase4n_artifact_manifest_row_v1",
                "phase_id": "v100_phase4n_gt_link_precision_diagnostic",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
            for path, kind, note in artifacts
        ],
    )
    best_rows = {
        f"{row['source_id']}|{row['dataset_split']}": row
        for row in summary_rows
    }
    summary = {
        "schema_version": "stream4d_v100_phase4n_gt_link_precision_diagnostic_summary_v1",
        "phase_id": "v100_phase4n_gt_link_precision_diagnostic",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "DIAGNOSTIC_ONLY_NOT_METHOD",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "top_gt_min_pixels": TOP_GT_MIN_PIXELS,
        "top_gt_min_fraction": TOP_GT_MIN_FRACTION,
        "object_signature_count": len(signatures),
        "valid_object_signature_count": sum(1 for row in signatures.values() if row["valid_gt_signature"]),
        "accepted_link_diagnostic_count": len(link_rows),
        "true_adjacent_pair_count": len(true_pair_rows),
        "summary_by_source_split": best_rows,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "object_gt_signature_rows": _rel(signature_csv),
            "accepted_link_gt_diagnostic_rows": _rel(link_csv),
            "true_adjacent_pair_recall_rows": _rel(true_pair_csv),
            "summary_rows": _rel(summary_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
