from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_carrier_supported_carving as phase3  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE0_WINDOWS = ROOT / "outputs/audit/v90_phase0_mv_ap_contract/window_support_rows.csv"
PHASE1_ROOT = ROOT / "outputs/audit/v90_phase1_variant_resurrection"
DEFAULT_NATIVE_SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
SEMANTIC_FEATURE_ROWS = [
    ROOT / "outputs/audit/v71_semantic_features/mask_feature_rows.csv",
    ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv",
]

METHOD_SOURCE_VARIANT = "R10_v82_local_B0_object_slot_config"
CONTROL_SOURCE_VARIANT = "C0_semantic_only_control"
ORIGINAL_MASK_VARIANTS = {
    "W0_no_witness_cover",
    "W1_carrier_FPS_witnesses",
    "W2_semantic_diverse_witnesses",
    "W3_geo_semantic_weighted_witnesses",
    "W5_witness_cover_multi_masklet",
    "W6_geo_semantic_risk_controlled_witnesses",
    "W8a_risk_balanced_p135_witnesses",
    "W8b_risk_balanced_p165_witnesses",
    "W8c_risk_balanced_p195_witnesses",
    "C0_W0_semantic_control",
}
GENERATED_MASK_VARIANTS = {
    "W4_witness_cover_plus_carving",
    "W7_risk_controlled_witness_cover_plus_carving",
    "W9a_risk_balanced_p135_plus_carving",
    "W9b_risk_balanced_p165_plus_carving",
    "W9c_risk_balanced_p195_plus_carving",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _mask_dir_by_scene() -> dict[str, Path]:
    rows = _read_csv(PHASE0_WINDOWS)
    out: dict[str, Path] = {}
    for row in rows:
        scene = row.get("scene_id", "")
        mask_source = row.get("mask_source", "")
        if scene and mask_source:
            out[scene] = ROOT / mask_source
    return out


def _window_maps() -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], str]]:
    frame_to_window_index: dict[tuple[str, int], int] = {}
    frame_to_window_id: dict[tuple[str, int], str] = {}
    for row in _read_csv(PHASE0_WINDOWS):
        scene = row.get("scene_id", "")
        window_index = _int(row.get("window_index"), -1)
        window_id = row.get("window_id") or f"w{window_index:04d}"
        start = _int(row.get("frame_id_start"), -1)
        end = _int(row.get("frame_id_end"), -1)
        if not scene or window_index < 0 or start < 0 or end < 0:
            continue
        for frame_id in range(start, end + 1, 5):
            frame_to_window_index[(scene, frame_id)] = window_index
            frame_to_window_id[(scene, frame_id)] = window_id
    return frame_to_window_index, frame_to_window_id


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _load_source_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], str], dict[tuple[str, str], str], dict[tuple[str, str], float]]:
    rows: list[dict[str, Any]] = []
    slot_to_obj: dict[tuple[str, str], str] = {}
    proto_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    area_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in _read_csv(PHASE1_ROOT / "adapter_input_frame_mask_rows.csv"):
        variant = row.get("variant", "")
        if variant not in {METHOD_SOURCE_VARIANT, CONTROL_SOURCE_VARIANT}:
            continue
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        slot = phase3._local_slot_from_row(row)
        if not slot:
            continue
        item = {**row, "local_slot_id": slot}
        rows.append(item)
        if variant == METHOD_SOURCE_VARIANT:
            key = (row.get("scene_id", ""), slot)
            slot_to_obj.setdefault(key, row.get("mv_object_id", "") or f"{row.get('scene_id')}:{slot}")
            proto = row.get("semantic_proto_id", "")
            if proto:
                proto_counter[key][proto] += 1
            area = _num(row.get("mask_area"), 0.0)
            if area > 0:
                area_acc[key].append(area)
    slot_to_proto = {key: counter.most_common(1)[0][0] for key, counter in proto_counter.items() if counter}
    slot_to_area = {key: _mean(vals) for key, vals in area_acc.items()}
    return rows, slot_to_obj, slot_to_proto, slot_to_area


def _load_semantic_features() -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in SEMANTIC_FEATURE_ROWS:
        for row in _read_csv(path):
            if _bool(row.get("uses_gt_for_prediction")):
                continue
            scene = row.get("scene_id", "")
            frame_id = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), -1)
            if not scene or frame_id < 0 or mask_id <= 0:
                continue
            out[(scene, frame_id, mask_id)] = {
                "semantic_prototype_id": row.get("semantic_prototype_id", ""),
                "semantic_prototype_margin": _num(row.get("semantic_prototype_margin"), 0.0),
                "semantic_entropy": _num(row.get("semantic_entropy"), 1.0),
                "semantic_background_score_proxy": _bool(row.get("semantic_background_score_proxy")),
                "broad_background_risk": _bool(row.get("broad_background_risk")),
                "feature_source": _rel(path),
            }
    return out


def _source_risk_mean(source_rows: list[dict[str, Any]], semantic_features: dict[tuple[str, int, int], dict[str, Any]], variant: str) -> float:
    vals: list[float] = []
    for row in source_rows:
        if row.get("variant") != variant:
            continue
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        feat = semantic_features.get(key, {})
        risk = bool(feat.get("broad_background_risk", False)) or bool(feat.get("semantic_background_score_proxy", False))
        vals.append(1.0 if risk else 0.0)
    return _mean(vals)


def _load_support_candidates(
    support_rows_path: Path,
    source_slots: set[tuple[str, str]],
    semantic_features: dict[tuple[str, int, int], dict[str, Any]],
    mask_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(support_rows_path):
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        if not _bool(row.get("native_support_allowed", "True")):
            continue
        scene = row.get("scene_id", "")
        slot = row.get("local_slot_id", "")
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        if (scene, slot) not in source_slots or frame_id < 0 or mask_id <= 0:
            continue
        grouped[(scene, slot, frame_id, mask_id)].append(
            {
                "x": _num(row.get("carrier_uv_x"), 0.0),
                "y": _num(row.get("carrier_uv_y"), 0.0),
                "confidence": _num(row.get("confidence"), 1.0),
                "visibility_prob": _num(row.get("visibility_prob"), 1.0),
                "density": _num(row.get("observed_mask_support_density"), 0.0),
                "native_carrier_global_id": row.get("native_carrier_global_id", ""),
            }
        )

    label_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}

    def mask_area(scene: str, frame_id: int, mask_id: int) -> tuple[int, int]:
        key = (scene, frame_id)
        if key not in label_cache:
            label = _read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
            counts = np.bincount(label.reshape(-1).astype(np.int64))
            label_cache[key] = (label, counts, int(label.size))
        _label, counts, image_area = label_cache[key]
        area = int(counts[mask_id]) if 0 <= mask_id < counts.shape[0] else 0
        return area, image_area

    candidates: list[dict[str, Any]] = []
    for (scene, slot, frame_id, mask_id), points in sorted(grouped.items()):
        area, image_area = mask_area(scene, frame_id, mask_id)
        if area <= 0:
            continue
        xs = [float(p["x"]) for p in points]
        ys = [float(p["y"]) for p in points]
        conf = [float(p["confidence"]) for p in points]
        vis = [float(p["visibility_prob"]) for p in points]
        density = [float(p["density"]) for p in points]
        feat = semantic_features.get((scene, frame_id, mask_id), {})
        area_ratio = float(area / max(1, image_area))
        broad = bool(feat.get("broad_background_risk", False)) or bool(feat.get("semantic_background_score_proxy", False)) or area_ratio >= 0.35
        candidates.append(
            {
                "scene_id": scene,
                "local_slot_id": slot,
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
                "support_count": int(len(points)),
                "carrier_count_unique": int(len({p.get("native_carrier_global_id", "") for p in points})),
                "confidence_mean": _mean(conf),
                "visibility_mean": _mean(vis),
                "observed_density_mean": _mean(density),
                "uv_x_mean": _mean(xs),
                "uv_y_mean": _mean(ys),
                "uv_x_std": float(np.std(xs)) if xs else 0.0,
                "uv_y_std": float(np.std(ys)) if ys else 0.0,
                "mask_area": int(area),
                "image_area": int(image_area),
                "area_ratio": area_ratio,
                "semantic_prototype_id": feat.get("semantic_prototype_id", ""),
                "semantic_prototype_margin": _num(feat.get("semantic_prototype_margin"), 0.0),
                "semantic_entropy": _num(feat.get("semantic_entropy"), 1.0),
                "semantic_background_score_proxy": bool(feat.get("semantic_background_score_proxy", False)),
                "broad_background_risk": bool(broad),
                "feature_source": feat.get("feature_source", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return candidates, grouped


def _score_candidate(row: dict[str, Any], variant: str, slot_proto: str, slot_area_mean: float) -> float:
    support_count = _num(row.get("support_count"), 0.0)
    conf = _num(row.get("confidence_mean"), 1.0)
    vis = _num(row.get("visibility_mean"), 1.0)
    area = max(1.0, _num(row.get("mask_area"), 1.0))
    area_ratio = _num(row.get("area_ratio"), 0.0)
    margin = _num(row.get("semantic_prototype_margin"), 0.0)
    entropy = _num(row.get("semantic_entropy"), 1.0)
    proto = str(row.get("semantic_prototype_id", ""))
    proto_match = 1.0 if slot_proto and proto == slot_proto else 0.0
    broad = 1.0 if _bool(row.get("broad_background_risk")) else 0.0
    area_ref = max(1.0, float(slot_area_mean or area))
    area_change = abs(math.log(max(1.0, area) / area_ref))
    geo_score = math.log1p(support_count) * conf * vis
    density_score = support_count / math.sqrt(area)
    semantic_score = 0.55 * proto_match + 1.25 * margin - 0.20 * entropy
    risk_penalty = 0.85 * broad + 0.35 * max(0.0, area_ratio - 0.25) + 0.15 * area_change
    if variant == "W1_carrier_FPS_witnesses":
        return float(geo_score + 0.35 * density_score - risk_penalty)
    if variant == "W2_semantic_diverse_witnesses":
        return float(semantic_score + 0.20 * math.log1p(support_count) - risk_penalty)
    if variant in {"W3_geo_semantic_weighted_witnesses", "W4_witness_cover_plus_carving", "W5_witness_cover_multi_masklet"}:
        return float(0.75 * geo_score + 0.45 * semantic_score + 0.30 * density_score - risk_penalty)
    if variant in {"W6_geo_semantic_risk_controlled_witnesses", "W7_risk_controlled_witness_cover_plus_carving"}:
        return float(0.75 * geo_score + 0.55 * semantic_score + 0.30 * density_score - 2.35 * risk_penalty - 0.75 * broad)
    if variant in {"W8a_risk_balanced_p135_witnesses", "W9a_risk_balanced_p135_plus_carving"}:
        return float(0.75 * geo_score + 0.52 * semantic_score + 0.30 * density_score - 1.35 * risk_penalty - 0.25 * broad)
    if variant in {"W8b_risk_balanced_p165_witnesses", "W9b_risk_balanced_p165_plus_carving"}:
        return float(0.75 * geo_score + 0.53 * semantic_score + 0.30 * density_score - 1.65 * risk_penalty - 0.45 * broad)
    if variant in {"W8c_risk_balanced_p195_witnesses", "W9c_risk_balanced_p195_plus_carving"}:
        return float(0.75 * geo_score + 0.54 * semantic_score + 0.30 * density_score - 1.95 * risk_penalty - 0.60 * broad)
    return float(geo_score - risk_penalty)


def _build_witness_rows(
    candidates: list[dict[str, Any]],
    slot_to_proto: dict[tuple[str, str], str],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        scene = row["scene_id"]
        slot = row["local_slot_id"]
        frame_id = _int(row.get("frame_id"), -1)
        window_index = frame_to_window_index.get((scene, frame_id), -1)
        if window_index < 0:
            continue
        grouped[(scene, slot, window_index)].append(row)
    rows: list[dict[str, Any]] = []
    for (scene, slot, window_index), items in sorted(grouped.items()):
        frames = sorted({_int(row.get("frame_id"), -1) for row in items})
        carriers = sum(_int(row.get("carrier_count_unique"), _int(row.get("support_count"), 0)) for row in items)
        proto = slot_to_proto.get((scene, slot), "")
        proto_hits = sum(1 for row in items if proto and row.get("semantic_prototype_id") == proto)
        non_bg = [0.0 if _bool(row.get("broad_background_risk")) else 1.0 for row in items]
        geo = min(1.0, math.log1p(carriers) / math.log(256.0))
        sem = float(proto_hits / max(1, len(items))) if proto else 0.0
        vis = min(1.0, len(frames) / 8.0)
        nonbg = _mean(non_bg)
        reliability = float(geo * max(0.15, sem if proto else 0.5) * vis * max(0.05, nonbg))
        rows.append(
            {
                "witness_id": f"{scene}:{slot}:w{window_index:04d}",
                "scene_id": scene,
                "local_slot_id": slot,
                "window_id": frame_to_window_id.get((scene, frames[0]), f"w{window_index:04d}") if frames else f"w{window_index:04d}",
                "window_index": int(window_index),
                "visible_frame_count": int(len(frames)),
                "frame_first": int(frames[0]) if frames else -1,
                "frame_last": int(frames[-1]) if frames else -1,
                "carrier_observation_count": int(sum(_int(row.get("support_count")) for row in items)),
                "carrier_count_proxy": int(carriers),
                "semantic_prototype_id": proto,
                "semantic_compactness": sem,
                "visibility_reliability": vis,
                "geo_reliability": geo,
                "nonbg_reliability": nonbg,
                "witness_reliability": reliability,
                "uv_x_mean": _mean([_num(row.get("uv_x_mean")) for row in items]),
                "uv_y_mean": _mean([_num(row.get("uv_y_mean")) for row in items]),
                "background_risk_mean": 1.0 - nonbg,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _fill_slot_priors_from_candidates(
    candidates: list[dict[str, Any]],
    slot_to_proto: dict[tuple[str, str], str],
    slot_to_area: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], float]]:
    proto_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    area_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in candidates:
        key = (str(row.get("scene_id", "")), str(row.get("local_slot_id", "")))
        proto = str(row.get("semantic_prototype_id", ""))
        if proto and not _bool(row.get("broad_background_risk")):
            proto_counter[key][proto] += max(1, _int(row.get("support_count"), 1))
        area = _num(row.get("mask_area"), 0.0)
        if area > 0 and not _bool(row.get("broad_background_risk")):
            area_acc[key].append(area)
    out_proto = dict(slot_to_proto)
    out_area = dict(slot_to_area)
    for key, counter in proto_counter.items():
        if key not in out_proto and counter:
            out_proto[key] = counter.most_common(1)[0][0]
    for key, vals in area_acc.items():
        if key not in out_area and vals:
            out_area[key] = _mean(vals)
    return out_proto, out_area


def _build_distance_rows(witness_rows: list[dict[str, Any]], max_per_window: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in witness_rows:
        grouped[(row["scene_id"], _int(row.get("window_index"), -1))].append(row)
    out: list[dict[str, Any]] = []
    for (scene, window_index), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: _num(r.get("witness_reliability")), reverse=True)[:max_per_window]
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                dx = _num(a.get("uv_x_mean")) - _num(b.get("uv_x_mean"))
                dy = _num(a.get("uv_y_mean")) - _num(b.get("uv_y_mean"))
                d_geo = math.sqrt(dx * dx + dy * dy)
                d_sem = 0.0 if a.get("semantic_prototype_id") and a.get("semantic_prototype_id") == b.get("semantic_prototype_id") else 1.0
                d_time = abs(_int(a.get("frame_first")) - _int(b.get("frame_first"))) / 100.0
                distance = 0.55 * d_geo + 0.30 * d_sem + 0.15 * min(1.0, d_time)
                out.append(
                    {
                        "scene_id": scene,
                        "window_index": int(window_index),
                        "witness_id_a": a.get("witness_id", ""),
                        "witness_id_b": b.get("witness_id", ""),
                        "d_geo": d_geo,
                        "d_sem": d_sem,
                        "d_time": min(1.0, d_time),
                        "witness_distance": distance,
                    }
                )
    return out


def _select_original_masklets(
    candidates: list[dict[str, Any]],
    slot_to_obj: dict[tuple[str, str], str],
    slot_to_proto: dict[tuple[str, str], str],
    slot_to_area: dict[tuple[str, str], float],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_slot_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slot_frame[(row["scene_id"], row["local_slot_id"], _int(row.get("frame_id"), -1))].append(row)
    selection_pre: list[dict[str, Any]] = []
    for variant in [
        "W1_carrier_FPS_witnesses",
        "W2_semantic_diverse_witnesses",
        "W3_geo_semantic_weighted_witnesses",
        "W5_witness_cover_multi_masklet",
        "W6_geo_semantic_risk_controlled_witnesses",
        "W8a_risk_balanced_p135_witnesses",
        "W8b_risk_balanced_p165_witnesses",
        "W8c_risk_balanced_p195_witnesses",
    ]:
        max_masks = 2 if variant == "W5_witness_cover_multi_masklet" else 1
        for (scene, slot, frame_id), items in sorted(by_slot_frame.items()):
            slot_key = (scene, slot)
            if slot_key not in slot_to_obj:
                continue
            scored = []
            for item in items:
                score = _score_candidate(item, variant, slot_to_proto.get(slot_key, ""), slot_to_area.get(slot_key, 0.0))
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored:
                continue
            selected = scored[:1]
            if max_masks > 1:
                top = scored[0][0]
                for score, item in scored[1:]:
                    if len(selected) >= max_masks:
                        break
                    if _bool(item.get("broad_background_risk")):
                        continue
                    if score >= top - 0.35:
                        selected.append((score, item))
            for rank, (score, item) in enumerate(selected, start=1):
                window_index = frame_to_window_index.get((scene, frame_id), -1)
                selection_pre.append(
                    {
                        **item,
                        "variant_id": variant,
                        "mv_object_id": f"{variant}:{slot_to_obj[slot_key]}",
                        "window_index": int(window_index),
                        "window_id": frame_to_window_id.get((scene, frame_id), ""),
                        "selection_score": float(score),
                        "selection_rank": int(rank),
                        "selection_stage": "pre_conflict_wta",
                        "selection_reason": f"{variant}_gt_free_support_semantic_cover",
                        "risk_penalty": 1.0 if _bool(item.get("broad_background_risk")) else 0.0,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )

    kept: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for row in sorted(selection_pre, key=lambda r: _num(r.get("selection_score")), reverse=True):
        key = (row["variant_id"], row["scene_id"], _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        old = kept.get(key)
        if old is None:
            kept[key] = {**row, "selection_stage": "post_conflict_wta", "conflict_dropped": False}
        else:
            dropped.append({**row, "selection_stage": "dropped_by_same_frame_mask_wta", "conflict_dropped": True, "kept_mv_object_id": old.get("mv_object_id", "")})
    final_rows = sorted(kept.values(), key=lambda r: (r["variant_id"], r["scene_id"], r["local_slot_id"], _int(r.get("frame_id")), -_num(r.get("selection_score"))))
    return final_rows, dropped


def _w0_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in source_rows:
        if row.get("variant") != METHOD_SOURCE_VARIANT:
            continue
        out.append(
            {
                "split": "dev",
                "scene_id": row.get("scene_id", ""),
                "source_variant": "W0_no_witness_cover",
                "variant": "W0_no_witness_cover",
                "mv_object_id": f"W0_no_witness_cover:{row.get('mv_object_id', '')}",
                "frame_id": _int(row.get("frame_id"), -1),
                "mask_id": _int(row.get("mask_id"), -1),
                "frame_mask_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                "object_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "selection_reason": "phase4_W0_original_R10_no_witness_cover",
            }
        )
    return out


def _control_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in source_rows:
        if row.get("variant") != CONTROL_SOURCE_VARIANT:
            continue
        out.append(
            {
                "split": "dev",
                "scene_id": row.get("scene_id", ""),
                "source_variant": "C0_W0_semantic_control",
                "variant": "C0_W0_semantic_control",
                "mv_object_id": f"C0_W0_semantic_control:{row.get('mv_object_id', '')}",
                "frame_id": _int(row.get("frame_id"), -1),
                "mask_id": _int(row.get("mask_id"), -1),
                "frame_mask_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                "object_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "selection_reason": "phase4_semantic_control_no_witness_cover",
            }
        )
    return out


def _selection_to_eval_rows(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in selection_rows:
        variant = row.get("variant_id", "")
        if variant not in ORIGINAL_MASK_VARIANTS:
            continue
        out.append(
            {
                "split": "dev",
                "scene_id": row.get("scene_id", ""),
                "source_variant": variant,
                "variant": variant,
                "mv_object_id": row.get("mv_object_id", ""),
                "frame_id": _int(row.get("frame_id"), -1),
                "mask_id": _int(row.get("mask_id"), -1),
                "frame_mask_score": _num(row.get("selection_score"), 1.0),
                "object_score": _num(row.get("selection_score"), 1.0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "selection_reason": row.get("selection_reason", ""),
            }
        )
    return out


def _generate_carved_masks(
    w3_selection_rows: list[dict[str, Any]],
    support_points: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    mask_dirs: dict[str, Path],
    radius: int,
    support_point_radius: int,
    variant: str,
    source_variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generated_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in w3_selection_rows:
        by_frame[(row["scene_id"], _int(row.get("frame_id"), -1))].append(row)
    for (scene, frame_id), rows in sorted(by_frame.items()):
        label = _read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
        shape = label.shape
        frame_items: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda r: _num(r.get("selection_score")), reverse=True):
            slot = row["local_slot_id"]
            mask_id = _int(row.get("mask_id"), -1)
            points = support_points.get((scene, slot, frame_id, mask_id), [])
            if not points:
                continue
            source_mask = label == int(mask_id)
            if not np.any(source_mask):
                continue
            _heat, support = phase3._paint_support(points, shape, max(1, support_point_radius))
            dilated = phase3._dilate(support, radius)
            carved = phase3._connected_component_around_support(source_mask, dilated)
            if not np.any(carved):
                continue
            frame_items.append({**row, "generated_mask": carved, "source_mask_area": int(np.count_nonzero(source_mask)), "support_area": int(np.count_nonzero(dilated))})
        if not frame_items:
            continue
        label_out = np.zeros(shape, dtype=np.uint16)
        for item in frame_items:
            new_mask_id = int(np.max(label_out)) + 1
            write_mask = item["generated_mask"] & (label_out == 0)
            if np.any(write_mask):
                label_out[write_mask] = new_mask_id
                item["new_mask_id"] = new_mask_id
        out_dir = OUT / "generated_masks" / variant / scene / "mask"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(frame_id)}.png"
        if not cv2.imwrite(str(out_path), label_out):
            raise RuntimeError(f"failed to write {out_path}")
        for item in frame_items:
            new_mask_id = _int(item.get("new_mask_id"), -1)
            if new_mask_id <= 0:
                continue
            final_area = int(np.count_nonzero(label_out == new_mask_id))
            if final_area <= 0:
                continue
            generated_rows.append(
                {
                    "variant_id": variant,
                    "source_variant": source_variant,
                    "scene_id": scene,
                    "window_id": item.get("window_id", ""),
                    "window_index": item.get("window_index", ""),
                    "mv_object_id": item.get("mv_object_id", "").replace(f"{source_variant}:", f"{variant}:"),
                    "local_slot_id": item.get("local_slot_id", ""),
                    "frame_id": int(frame_id),
                    "source_mask_id": _int(item.get("mask_id"), -1),
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": _rel(out_path),
                    "carving_mode": f"phase4_witness_connected_component_r{radius:02d}",
                    "carrier_support_count": _int(item.get("support_count"), 0),
                    "support_area": _int(item.get("support_area"), 0),
                    "source_mask_area": _int(item.get("source_mask_area"), 0),
                    "generated_mask_area": int(final_area),
                    "object_score": _num(item.get("selection_score"), 1.0),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                }
            )
            audit_item = {k: v for k, v in item.items() if k != "generated_mask"}
            selection_rows.append(
                {
                    **audit_item,
                    "variant_id": variant,
                    "mv_object_id": generated_rows[-1]["mv_object_id"],
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": _rel(out_path),
                    "generated_mask_area": int(final_area),
                    "selection_stage": "post_w3_plus_carving",
                }
            )
            eval_rows.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "source_variant": variant,
                    "variant": variant,
                    "mv_object_id": generated_rows[-1]["mv_object_id"],
                    "frame_id": int(frame_id),
                    "mask_id": int(new_mask_id),
                    "frame_mask_score": _num(item.get("selection_score"), 1.0),
                    "object_score": _num(item.get("selection_score"), 1.0),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"phase4_{variant}_{source_variant}_selection_plus_carrier_component_carving",
                }
            )
    return generated_rows, selection_rows, eval_rows


def _all_iou_rows(iou: np.ndarray, pred_ids: list[int], gt_ids: list[int], top_k: int = 100, **_kwargs: Any) -> list[dict[str, Any]]:
    rows = []
    if iou.size == 0:
        return rows
    for pidx, pred_id in enumerate(pred_ids):
        for gidx, gt_id in enumerate(gt_ids):
            rows.append({"pred_id": int(pred_id), "gt_id": int(gt_id), "iou": float(iou[pidx, gidx])})
    return rows


def _evaluate(eval_rows: list[dict[str, Any]], mask_dirs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    original_top = recalc._top_iou_rows
    original_mask_dir = recalc._mask_dir
    recalc._top_iou_rows = _all_iou_rows
    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    local_export_root = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
    variants = sorted({str(row.get("variant", "")) for row in eval_rows if row.get("variant")})
    try:
        for variant in variants:
            if variant in GENERATED_MASK_VARIANTS:
                recalc._mask_dir = lambda scene, _variant=variant: OUT / "generated_masks" / _variant / scene / "mask"
            else:
                recalc._mask_dir = lambda scene: mask_dirs[scene]
            for scene in ["scene0011_00", "scene0050_00"]:
                rows = [row for row in eval_rows if row.get("variant") == variant and row.get("scene_id") == scene]
                if not rows:
                    continue
                frame_ids = scope.get(("dev", scene))
                metric, cases, tops, _window_rows = recalc._evaluate_frame_mask_variant_local_window(
                    scene=scene,
                    split="dev",
                    variant=variant,
                    frame_ids=frame_ids,
                    rows=rows,
                    score_mode="input",
                    local_export_root=local_export_root,
                    window_source_step="S3D_L1_local_merged_masks",
                )
                sf50_f1 = phase1._f1(metric.get("SF50_precision"), metric.get("SF50_recall"))
                metric = {
                    **metric,
                    "variant_id": variant,
                    "MV_AP_window": metric.get("MV_AP"),
                    "MV_AP50_window": metric.get("MV_AP50"),
                    "MV_AP25_window": metric.get("MV_AP25"),
                    "score_free_Match50_window": sf50_f1,
                    "score_free_Match50_precision_window": metric.get("SF50_precision"),
                    "score_free_Match50_recall_window": metric.get("SF50_recall"),
                    "same_frame_collision_count": 0,
                    "metric_scope": "local_window_gt_projection",
                }
                metric_rows.append(metric)
                case_rows.extend(cases)
                for row in tops:
                    iou_rows.append(
                        {
                            **row,
                            "variant_id": variant,
                            "mv_iou": row.get("iou", ""),
                            "matrix_scope": "phase4_full_pred_gt_iou_matrix_local_window_support",
                            "full_zero_pairs_omitted": False,
                        }
                    )
    finally:
        recalc._top_iou_rows = original_top
        recalc._mask_dir = original_mask_dir
    return metric_rows, iou_rows, case_rows


def _aggregate(
    metric_rows: list[dict[str, Any]],
    witness_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped_metric[str(row.get("variant_id", ""))].append(row)
    selected_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selection_rows:
        selected_by_variant[str(row.get("variant_id", ""))].append(row)
    dropped_by_variant = Counter(str(row.get("variant_id", "")) for row in dropped_rows)
    witness_keys = {(row["scene_id"], row["local_slot_id"], _int(row.get("window_index"), -1)): _num(row.get("witness_reliability"), 0.0) for row in witness_rows}
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped_metric.items()):
        selected = selected_by_variant.get(variant, [])
        covered = {(row["scene_id"], row["local_slot_id"], _int(row.get("window_index"), -1)) for row in selected if _int(row.get("window_index"), -1) >= 0}
        total_weight = sum(witness_keys.values())
        covered_weight = sum(weight for key, weight in witness_keys.items() if key in covered)
        selected_by_object = Counter(str(row.get("mv_object_id", "")) for row in selected)
        risk_vals = [1.0 if _bool(row.get("broad_background_risk")) else 0.0 for row in selected]
        gen = [row for row in generated_rows if row.get("variant_id") == variant]
        source_area = [_num(row.get("source_mask_area")) for row in gen]
        generated_area = [_num(row.get("generated_mask_area")) for row in gen]
        out.append(
            {
                "variant_id": variant,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([_num(row.get("MV_AP_window")) for row in rows]),
                "mean_MV_AP50_window": _mean([_num(row.get("MV_AP50_window")) for row in rows]),
                "mean_MV_AP25_window": _mean([_num(row.get("MV_AP25_window")) for row in rows]),
                "mean_score_free_Match50_window": _mean([_num(row.get("score_free_Match50_window")) for row in rows]),
                "mean_GT_best_IoU_window": _mean([_num(row.get("gt_best_iou_mean")) for row in rows]),
                "witness_count": len(witness_keys),
                "witness_reliability_mean": _mean(list(witness_keys.values())),
                "witness_visibility_span_mean": _mean([_num(row.get("visible_frame_count")) for row in witness_rows]),
                "witness_background_risk_mean": _mean([_num(row.get("background_risk_mean")) for row in witness_rows]),
                "masklet_count_per_object": _mean([float(v) for v in selected_by_object.values()]),
                "witness_coverage_rate": float(len(covered) / max(1, len(witness_keys))),
                "weighted_witness_coverage": float(covered_weight / max(1e-12, total_weight)),
                "risk_penalty_mean": _mean(risk_vals),
                "selection_row_count": len(selected),
                "conflict_dropped_count": int(dropped_by_variant.get(variant, 0)),
                "generated_mask_rows": len(gen),
                "area_shrink_ratio": float(_mean(generated_area) / max(1.0, _mean(source_area))) if gen else 1.0,
                "fragmentation_rate_proxy": float(max(0.0, len(selected_by_object) - _mean([_num(row.get("gt_object_count")) for row in rows])) / max(1.0, _mean([_num(row.get("gt_object_count")) for row in rows]))),
                "overmerge_rate_proxy": float(_mean([_num(row.get("duplicate_frame_mask_conflict_count")) for row in rows]) / max(1.0, _mean([_num(row.get("frame_mask_row_count")) for row in rows]))),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in selected),
                "uses_future": any(_bool(row.get("uses_future")) for row in selected),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mask_dirs = _mask_dir_by_scene()
    frame_to_window_index, frame_to_window_id = _window_maps()
    source_rows, slot_to_obj, slot_to_proto, slot_to_area = _load_source_rows()
    semantic_features = _load_semantic_features()
    b0_risk_proxy = _source_risk_mean(source_rows, semantic_features, METHOD_SOURCE_VARIANT)
    support_rows_path = Path(args.native_support_rows)
    if not support_rows_path.is_absolute():
        support_rows_path = ROOT / support_rows_path
    candidates, support_points = _load_support_candidates(support_rows_path, set(slot_to_obj), semantic_features, mask_dirs)
    slot_to_proto, slot_to_area = _fill_slot_priors_from_candidates(candidates, slot_to_proto, slot_to_area)
    witness_rows = _build_witness_rows(candidates, slot_to_proto, frame_to_window_index, frame_to_window_id)
    witness_distance_rows = _build_distance_rows(witness_rows, max_per_window=int(args.distance_max_per_window))
    selection_rows, dropped_rows = _select_original_masklets(candidates, slot_to_obj, slot_to_proto, slot_to_area, frame_to_window_index, frame_to_window_id)
    w3_rows = [row for row in selection_rows if row.get("variant_id") == "W3_geo_semantic_weighted_witnesses"]
    generated_rows, w4_selection_rows, w4_eval_rows = _generate_carved_masks(
        w3_rows,
        support_points,
        mask_dirs,
        radius=int(args.carving_radius),
        support_point_radius=int(args.support_point_radius),
        variant="W4_witness_cover_plus_carving",
        source_variant="W3_geo_semantic_weighted_witnesses",
    )
    w6_rows = [row for row in selection_rows if row.get("variant_id") == "W6_geo_semantic_risk_controlled_witnesses"]
    generated_rows_w7, w7_selection_rows, w7_eval_rows = _generate_carved_masks(
        w6_rows,
        support_points,
        mask_dirs,
        radius=int(args.carving_radius),
        support_point_radius=int(args.support_point_radius),
        variant="W7_risk_controlled_witness_cover_plus_carving",
        source_variant="W6_geo_semantic_risk_controlled_witnesses",
    )
    generated_rows.extend(generated_rows_w7)
    balanced_eval_rows: list[dict[str, Any]] = []
    for source_variant, carved_variant in [
        ("W8a_risk_balanced_p135_witnesses", "W9a_risk_balanced_p135_plus_carving"),
        ("W8b_risk_balanced_p165_witnesses", "W9b_risk_balanced_p165_plus_carving"),
        ("W8c_risk_balanced_p195_witnesses", "W9c_risk_balanced_p195_plus_carving"),
    ]:
        source_rows_for_carving = [row for row in selection_rows if row.get("variant_id") == source_variant]
        generated_rows_tmp, selection_rows_tmp, eval_rows_tmp = _generate_carved_masks(
            source_rows_for_carving,
            support_points,
            mask_dirs,
            radius=int(args.carving_radius),
            support_point_radius=int(args.support_point_radius),
            variant=carved_variant,
            source_variant=source_variant,
        )
        generated_rows.extend(generated_rows_tmp)
        selection_rows.extend(selection_rows_tmp)
        balanced_eval_rows.extend(eval_rows_tmp)
    selection_rows.extend(w4_selection_rows)
    selection_rows.extend(w7_selection_rows)

    eval_rows = []
    eval_rows.extend(_w0_rows(source_rows))
    eval_rows.extend(_control_rows(source_rows))
    eval_rows.extend(_selection_to_eval_rows(selection_rows))
    eval_rows.extend(w4_eval_rows)
    eval_rows.extend(w7_eval_rows)
    eval_rows.extend(balanced_eval_rows)

    metric_rows, iou_rows, case_rows = _evaluate(eval_rows, mask_dirs)
    aggregate_rows = _aggregate(metric_rows, witness_rows, selection_rows, generated_rows, dropped_rows)
    for row in aggregate_rows:
        if row.get("variant_id") == "W0_no_witness_cover":
            row["risk_penalty_mean"] = b0_risk_proxy

    _write_csv(OUT / "witness_rows.csv", witness_rows)
    _write_csv(OUT / "witness_distance_rows.csv", witness_distance_rows)
    _write_csv(OUT / "masklet_candidate_rows.csv", candidates)
    _write_csv(OUT / "witness_cover_selection_rows.csv", selection_rows + dropped_rows)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", eval_rows)
    _write_csv(OUT / "generated_mask_rows.csv", generated_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_iou_matrix_rows.csv", iou_rows)
    _write_csv(OUT / "witness_cover_casebook_rows.csv", case_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)

    b0 = next((row for row in _read_csv(PHASE1_ROOT / "mv_metric_aggregate_rows.csv") if row.get("variant_id") == "B0_local_only"), {})
    control = next((row for row in aggregate_rows if row.get("variant_id") == "C0_W0_semantic_control"), {})
    w3 = next((row for row in aggregate_rows if row.get("variant_id") == "W3_geo_semantic_weighted_witnesses"), {})
    w4 = next((row for row in aggregate_rows if row.get("variant_id") == "W4_witness_cover_plus_carving"), {})
    w6 = next((row for row in aggregate_rows if row.get("variant_id") == "W6_geo_semantic_risk_controlled_witnesses"), {})
    w7 = next((row for row in aggregate_rows if row.get("variant_id") == "W7_risk_controlled_witness_cover_plus_carving"), {})
    balanced_gate_rows = [
        row
        for row in aggregate_rows
        if str(row.get("variant_id", "")).startswith(("W8", "W9"))
    ]
    method_candidates = [row for row in aggregate_rows if str(row.get("variant_id", "")).startswith("W")]
    best = max(method_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    gate_candidates = [row for row in [w3, w4, w6, w7] if row] + balanced_gate_rows
    best_gate_candidate = max(gate_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    risk_safe_candidates = [
        row
        for row in gate_candidates
        if _num(row.get("risk_penalty_mean")) <= b0_risk_proxy + 1e-12
    ]
    best_risk_safe_candidate = max(risk_safe_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    progress_gate = bool(best_risk_safe_candidate) and (
        _num(best_risk_safe_candidate.get("mean_MV_AP_window")) >= _num(b0.get("mean_MV_AP_window")) + 0.01
        and _num(best_risk_safe_candidate.get("mean_MV_AP50_window")) >= _num(b0.get("mean_MV_AP50_window")) + 0.02
        and _num(best_risk_safe_candidate.get("mean_MV_AP_window")) >= _num(control.get("mean_MV_AP_window")) + 0.005
        and _num(best_risk_safe_candidate.get("witness_coverage_rate")) >= 0.70
        and not _bool(best_risk_safe_candidate.get("uses_gt_for_prediction"))
        and not _bool(best_risk_safe_candidate.get("uses_future"))
    )
    summary = {
        "phase": "v90_phase4_geo_semantic_witness_cover",
        "schema": "stream4d_v90_phase4_geo_semantic_witness_cover_v1",
        "phase4_pass": bool(metric_rows),
        "progress_gate": progress_gate,
        "runtime_sec": time.time() - t0,
        "gpu_usage_note": "No model forward was run; Phase4 uses precomputed D4RT UV support, precomputed semantic features, CPU mask selection/carving, and MV_AP_window evaluation.",
        "inputs": {
            "native_carrier_support_rows": _rel(support_rows_path),
            "phase1_adapter_rows": _rel(PHASE1_ROOT / "adapter_input_frame_mask_rows.csv"),
            "semantic_feature_rows": [_rel(path) for path in SEMANTIC_FEATURE_ROWS if path.exists()],
            "window_support_rows": _rel(PHASE0_WINDOWS),
        },
        "row_counts": {
            "witness_rows": len(witness_rows),
            "witness_distance_rows": len(witness_distance_rows),
            "masklet_candidate_rows": len(candidates),
            "witness_cover_selection_rows": len(selection_rows) + len(dropped_rows),
            "mv_object_frame_mask_rows": len(eval_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_iou_matrix_rows": len(iou_rows),
            "witness_cover_casebook_rows": len(case_rows),
        },
        "B0_local_only_phase1": b0,
        "B0_risk_penalty_mean_proxy": b0_risk_proxy,
        "best_method_variant": best.get("variant_id", ""),
        "best_method_metrics": best,
        "best_gate_candidate_variant": best_gate_candidate.get("variant_id", ""),
        "best_gate_candidate_metrics": best_gate_candidate,
        "best_risk_safe_gate_candidate_variant": best_risk_safe_candidate.get("variant_id", ""),
        "best_risk_safe_gate_candidate_metrics": best_risk_safe_candidate,
        "W3_metrics": w3,
        "W4_metrics": w4,
        "W6_metrics": w6,
        "W7_metrics": w7,
        "balanced_risk_sweep_metrics": balanced_gate_rows,
        "best_control_variant": control.get("variant_id", ""),
        "best_control_metrics": control,
        "progress_gate_criteria": {
            "W3_or_W4_MV_AP_window": ">= B0 + 0.01",
            "W3_or_W4_MV_AP50_window": ">= B0 + 0.02",
            "W3_or_W4_MV_AP_window_vs_control": ">= control + 0.005",
            "witness_coverage_rate": ">= 0.70",
            "broad_background_risk": "not increased vs B0 proxy",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        "outputs": {
            "witness_rows": _rel(OUT / "witness_rows.csv"),
            "witness_distance_rows": _rel(OUT / "witness_distance_rows.csv"),
            "masklet_candidate_rows": _rel(OUT / "masklet_candidate_rows.csv"),
            "witness_cover_selection_rows": _rel(OUT / "witness_cover_selection_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT / "mv_object_frame_mask_rows.csv"),
            "generated_mask_rows": _rel(OUT / "generated_mask_rows.csv"),
            "mv_metric_rows": _rel(OUT / "mv_metric_rows.csv"),
            "mv_iou_matrix_rows": _rel(OUT / "mv_iou_matrix_rows.csv"),
            "witness_cover_casebook_rows": _rel(OUT / "witness_cover_casebook_rows.csv"),
            "mv_metric_aggregate_rows": _rel(OUT / "mv_metric_aggregate_rows.csv"),
        },
    }
    _write_json(OUT / "summary.json", summary)
    sha_paths = [
        OUT / "witness_rows.csv",
        OUT / "witness_distance_rows.csv",
        OUT / "masklet_candidate_rows.csv",
        OUT / "witness_cover_selection_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_iou_matrix_rows.csv",
        OUT / "witness_cover_casebook_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v90 Phase4 D4RT geo-semantic witness cover and MV_AP_window eval.")
    parser.add_argument("--native-support-rows", default=str(DEFAULT_NATIVE_SUPPORT_ROWS.relative_to(ROOT)))
    parser.add_argument("--support-point-radius", type=int, default=3)
    parser.add_argument("--carving-radius", type=int, default=16)
    parser.add_argument("--distance-max-per-window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
