from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v93_phase4_edge_only_materialization"
RUN_ID = "v93_phase4_edge_only_materialization"
OUT = ROOT / "outputs/audit/v93_phase4_edge_only_materialization"

V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract"
BASE_SOURCE_VARIANT = "V91_AD4_sr2_adapt_sig8_b05_j075_r12_source"

MATERIALIZED_VARIANTS = [
    "E1_outer_edge_only",
    "E2_nested_overlap_edge",
    "E3_competing_edge",
    "R0_random_edge_control",
    "R1_shuffled_edge_control",
]
ALL_EDGE_VARIANTS = [*MATERIALIZED_VARIANTS, "E4_repeated_multiview_edge"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    return image.astype(np.int32, copy=False)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _adaptive_barrier_mask(
    *,
    source_mask: np.ndarray,
    edge_label_mask: np.ndarray,
    initial_radius: int,
    min_keep_ratio: float,
    min_keep_pixels: int,
) -> tuple[np.ndarray, int, float]:
    source_area = int(np.count_nonzero(source_mask))
    if source_area <= 0 or not np.any(edge_label_mask):
        return np.zeros_like(source_mask, dtype=bool), 0, 1.0
    last_barrier = np.zeros_like(source_mask, dtype=bool)
    last_radius = 0
    last_keep_ratio = 1.0
    for radius in [int(initial_radius), 2, 1]:
        radius = max(1, radius)
        barrier = source_mask & _dilate(edge_label_mask, radius)
        keep_ratio = float(np.count_nonzero(source_mask & ~barrier) / max(1, source_area))
        last_barrier = barrier
        last_radius = radius
        last_keep_ratio = keep_ratio
        if keep_ratio >= float(min_keep_ratio) and np.count_nonzero(source_mask & ~barrier) >= int(min_keep_pixels):
            break
    return last_barrier, last_radius, last_keep_ratio


def _random_trim(source_mask: np.ndarray, remove_count: int, seed_text: str, min_keep_ratio: float) -> np.ndarray:
    source_idx = np.flatnonzero(source_mask.reshape(-1))
    source_area = int(source_idx.size)
    if source_area <= 0 or remove_count <= 0:
        return source_mask.copy()
    min_keep = max(1, int(math.ceil(float(min_keep_ratio) * source_area)))
    remove_count = min(int(remove_count), max(0, source_area - min_keep))
    if remove_count <= 0:
        return source_mask.copy()
    rng = np.random.default_rng(_stable_seed(seed_text))
    remove = rng.choice(source_idx, size=remove_count, replace=False)
    out = source_mask.reshape(-1).copy()
    out[remove] = False
    return out.reshape(source_mask.shape)


def _translate_mask_to_bbox(mask: np.ndarray, source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    sx0, sy0, sx1, sy1 = _bbox(source_mask)
    tx0, ty0, tx1, ty1 = _bbox(target_mask)
    if sx1 <= sx0 or sy1 <= sy0 or tx1 <= tx0 or ty1 <= ty0 or not np.any(mask):
        return np.zeros_like(target_mask, dtype=bool)
    sx = (sx0 + sx1) // 2
    sy = (sy0 + sy1) // 2
    tx = (tx0 + tx1) // 2
    ty = (ty0 + ty1) // 2
    ys, xs = np.nonzero(mask)
    yy = ys + (ty - sy)
    xx = xs + (tx - sx)
    valid = (yy >= 0) & (yy < target_mask.shape[0]) & (xx >= 0) & (xx < target_mask.shape[1])
    out = np.zeros_like(target_mask, dtype=bool)
    out[yy[valid], xx[valid]] = True
    return out & target_mask


class FrameWriter:
    def __init__(self, out: Path, variant_ids: list[str]) -> None:
        self.out = out
        self.variant_ids = variant_ids
        self.current: tuple[str, int] | None = None
        self.labels: dict[str, np.ndarray] = {}
        self.next_ids: dict[str, int] = {}

    def flush(self) -> None:
        if self.current is None:
            return
        scene, frame_id = self.current
        for variant_id, label in self.labels.items():
            out_dir = self.out / "generated_masks" / variant_id / scene / "mask"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{int(frame_id)}.png"
            if not cv2.imwrite(str(out_path), label.astype(np.uint16)):
                raise RuntimeError(f"failed to write {out_path}")
        self.current = None
        self.labels = {}
        self.next_ids = {}

    def ensure_frame(self, scene: str, frame_id: int, shape: tuple[int, int]) -> None:
        key = (scene, int(frame_id))
        if self.current != key:
            self.flush()
            self.current = key
            self.labels = {variant_id: np.zeros(shape, dtype=np.uint16) for variant_id in self.variant_ids}
            self.next_ids = {variant_id: 1 for variant_id in self.variant_ids}

    def add_mask(self, variant_id: str, mask: np.ndarray) -> int:
        label = self.labels[variant_id]
        write = np.asarray(mask, dtype=bool) & (label == 0)
        if not np.any(write):
            return -1
        new_id = self.next_ids[variant_id]
        label[write] = int(new_id)
        self.next_ids[variant_id] = int(new_id) + 1
        return int(new_id)


def _key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("variant_id", "")),
        str(row.get("scene_id", "")),
        _int(row.get("frame_id"), -1),
        _int(row.get("source_mask_id"), -1),
    )


def _label_union(label: np.ndarray, labels: list[int]) -> np.ndarray:
    if not labels:
        return np.zeros(label.shape, dtype=bool)
    return np.isin(label, np.asarray([int(v) for v in labels], dtype=label.dtype))


def _load_inputs(base_source_variant: str) -> tuple[
    dict[tuple[str, str, int, int], dict[str, str]],
    list[dict[str, str]],
    dict[tuple[str, str, int, int], dict[str, list[int]]],
]:
    source_rows = {
        _key(row): row
        for row in _read_csv(V93_PHASE1 / "source_container_rows.csv")
        if row.get("variant_id") == base_source_variant
    }
    link_rows = [
        row
        for row in _read_csv(V93_PHASE1 / "object_container_link_rows.csv")
        if row.get("variant_id") == base_source_variant
    ]
    edge_groups: dict[tuple[str, str, int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_csv(V93_PHASE1 / "mask_edge_hypothesis_rows.csv"):
        if row.get("variant_id") != base_source_variant:
            continue
        edge_type = str(row.get("edge_type", ""))
        other = _int(row.get("edge_mask_id_b"), 0)
        if other > 0 and edge_type in {"nested_overlap", "competing", "repeated_multiview"}:
            edge_groups[_key(row)][edge_type].append(other)
    return source_rows, link_rows, edge_groups


def _compute_source_masks(
    *,
    key: tuple[str, str, int, int],
    source_row: dict[str, str],
    edge_groups: dict[tuple[str, str, int, int], dict[str, list[int]]],
    label_cache: dict[Path, np.ndarray],
    source_mask_cache: dict[tuple[str, str, int, int], dict[str, Any]],
    donor_key: tuple[str, str, int, int] | None,
    min_keep_ratio: float,
    barrier_radius: int,
) -> dict[str, Any]:
    if key in source_mask_cache:
        return source_mask_cache[key]
    mask_path = _resolve(source_row.get("mask_path", ""))
    if mask_path not in label_cache:
        label_cache[mask_path] = _load_label(mask_path)
    label = label_cache[mask_path]
    source_id = key[3]
    source_mask = label == int(source_id)
    source_area = int(np.count_nonzero(source_mask))
    nested_labels = sorted(set(edge_groups.get(key, {}).get("nested_overlap", [])))
    competing_labels = sorted(set(edge_groups.get(key, {}).get("competing", [])))
    repeated_labels = sorted(set(edge_groups.get(key, {}).get("repeated_multiview", [])))

    nested_barrier, nested_radius, nested_keep = _adaptive_barrier_mask(
        source_mask=source_mask,
        edge_label_mask=_label_union(label, nested_labels),
        initial_radius=barrier_radius,
        min_keep_ratio=min_keep_ratio,
        min_keep_pixels=8,
    )
    competing_barrier, competing_radius, competing_keep = _adaptive_barrier_mask(
        source_mask=source_mask,
        edge_label_mask=_label_union(label, competing_labels),
        initial_radius=barrier_radius,
        min_keep_ratio=min_keep_ratio,
        min_keep_pixels=8,
    )
    e1 = source_mask
    e2 = source_mask & ~nested_barrier
    e3 = source_mask & ~competing_barrier
    competing_remove = int(np.count_nonzero(source_mask & ~e3))
    r0 = _random_trim(source_mask, competing_remove, f"R0:{key}", min_keep_ratio)

    shuffled_barrier = np.zeros_like(source_mask, dtype=bool)
    shuffled_note = ""
    if donor_key is not None and donor_key in source_mask_cache:
        donor = source_mask_cache[donor_key]
        shuffled_barrier = _translate_mask_to_bbox(donor["competing_barrier"], donor["source_mask"], source_mask)
    if int(np.count_nonzero(shuffled_barrier)) < max(1, int(0.25 * max(0, competing_remove))):
        donor_remove = int(source_area * source_mask_cache.get(donor_key, {}).get("competing_removed_ratio", 0.0)) if donor_key else competing_remove
        r1 = _random_trim(source_mask, donor_remove, f"R1_shuffle_fallback:{donor_key}:{key}", min_keep_ratio)
        shuffled_note = "translated_donor_barrier_low_intersection_used_density_fallback"
    else:
        r1 = source_mask & ~shuffled_barrier
        shuffled_note = "translated_donor_competing_barrier"

    payload = {
        "mask_path": mask_path,
        "label": label,
        "source_mask": source_mask,
        "source_area": source_area,
        "nested_labels": nested_labels,
        "competing_labels": competing_labels,
        "repeated_labels": repeated_labels,
        "nested_barrier": nested_barrier,
        "competing_barrier": competing_barrier,
        "competing_removed_ratio": float(competing_remove / max(1, source_area)),
        "variant_masks": {
            "E1_outer_edge_only": e1,
            "E2_nested_overlap_edge": e2,
            "E3_competing_edge": e3,
            "R0_random_edge_control": r0,
            "R1_shuffled_edge_control": r1,
        },
        "variant_notes": {
            "E1_outer_edge_only": "outer_edge_containment_only_equals_source_mask",
            "E2_nested_overlap_edge": f"nested_labels={len(nested_labels)}; adaptive_radius={nested_radius}; keep_ratio={nested_keep:.6f}",
            "E3_competing_edge": f"competing_labels={len(competing_labels)}; adaptive_radius={competing_radius}; keep_ratio={competing_keep:.6f}",
            "R0_random_edge_control": "stable_random_pixels_same_remove_count_as_E3_competing_barrier",
            "R1_shuffled_edge_control": shuffled_note,
        },
    }
    source_mask_cache[key] = payload
    return payload


def _variant_config_rows(base_source_variant: str, created_at: str, barrier_radius: int, min_keep_ratio: float) -> list[dict[str, Any]]:
    specs = {
        "E1_outer_edge_only": ("outer_edge", "real_edge", "source outer edge containment only; no D4RT/RADIO/GT"),
        "E2_nested_overlap_edge": ("nested_overlap_edge", "real_edge", "source minus dilated nested/contained mask barrier"),
        "E3_competing_edge": ("competing_edge", "real_edge", "source minus dilated competing same-frame mask barriers"),
        "R0_random_edge_control": ("random_edge", "control", "stable random source pixels removed at E3 density"),
        "R1_shuffled_edge_control": ("shuffled_edge", "control", "competing barrier shuffled across source containers with density fallback"),
        "E4_repeated_multiview_edge": ("repeated_multiview_edge", "unsupported_real_edge", "not materialized because Phase1 has no repeated_multiview edge rows"),
    }
    return [
        {
            "schema_version": "stream4d_v93_phase4_edge_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": variant_id,
            "variant_kind": kind,
            "family": family,
            "base_source_variant": base_source_variant,
            "barrier_radius_initial_px": int(barrier_radius),
            "min_keep_ratio": float(min_keep_ratio),
            "uses_d4rt_witness": False,
            "uses_radio_region_feature": False,
            "uses_mask_edge_hypothesis": variant_id != "R0_random_edge_control",
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "description": desc,
            "created_at": created_at,
        }
        for variant_id, (kind, family, desc) in specs.items()
    ]


def _metric_rows_with_common(rows: list[dict[str, Any]], created_at: str) -> list[dict[str, Any]]:
    source = OUT / "control_metric_rows.csv"
    area_by_variant = _area_summary()
    r0 = next((row for row in rows if row.get("variant_id") == "R0_random_edge_control"), {})
    whole = _read_json(ROOT / "outputs/audit/v93_phase4_cue_isolation/summary.json").get("whole_source_MV_AP_window", "")
    if whole == "":
        whole = _read_json(ROOT / "outputs/audit/v92_phase6_attribution/summary.json").get("whole_source_MV_AP_window", "")
    out = []
    for row in rows:
        variant_id = str(row.get("variant_id", ""))
        area = area_by_variant.get(variant_id, {})
        mv = _num(row.get("mean_MV_AP_window"))
        out.append(
            {
                "schema_version": "stream4d_v93_phase4_edge_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "source_artifact": _rel(source),
                "source_artifact_sha256": _sha256(source) if source.exists() else "",
                "created_at": created_at,
                **row,
                "mean_generated_area_ratio": area.get("mean_generated_area_ratio", ""),
                "undercoverage_proxy": area.get("mean_source_removed_ratio", ""),
                "overcoverage_proxy": area.get("mean_generated_area_ratio", ""),
                "proxy_scope": "source_area_only_not_GT",
                "control_gap_vs_random_edge": mv - _num(r0.get("mean_MV_AP_window")) if r0 else "",
                "control_gap_vs_whole_source": mv - _num(whole) if whole != "" else "",
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    return out


def _area_summary() -> dict[str, dict[str, Any]]:
    rows = _read_csv(OUT / "generated_mask_rows.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("variant_id", "")].append(row)
    out: dict[str, dict[str, Any]] = {}
    for variant_id, group in grouped.items():
        ratios = [_num(row.get("generated_mask_area_ratio")) for row in group]
        removed = [1.0 - _num(row.get("generated_mask_area_ratio")) for row in group]
        out[variant_id] = {
            "mean_generated_area_ratio": _mean(ratios),
            "mean_source_removed_ratio": _mean(removed),
            "generated_row_count": len(group),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve(args.output_root)
    global OUT
    OUT = out
    if out.exists() and bool(args.clean):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    source_rows, link_rows, edge_groups = _load_inputs(str(args.base_source_variant))
    link_rows.sort(key=lambda row: (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1), row.get("object_hypothesis_id", "")))
    source_keys = sorted(source_rows)
    donor_by_key = {key: source_keys[(idx + 137) % len(source_keys)] for idx, key in enumerate(source_keys)} if source_keys else {}

    label_cache: dict[Path, np.ndarray] = {}
    source_mask_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    frame_writer = FrameWriter(out, MATERIALIZED_VARIANTS)
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    object_rows_seen: set[tuple[str, str]] = set()
    object_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for link in link_rows:
        key = _key(link)
        source = source_rows.get(key)
        if source is None:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v93_phase4_edge_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": "ALL",
                    "failure_type": "SOURCE_CONTAINER_MISSING_FOR_LINK",
                    "repair_direction": "repair Phase1 source/link key join before edge-only materialization",
                    "scene_id": link.get("scene_id", ""),
                    "frame_id": link.get("frame_id", ""),
                    "source_mask_id": link.get("source_mask_id", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": created_at,
                }
            )
            continue
        payload = _compute_source_masks(
            key=key,
            source_row=source,
            edge_groups=edge_groups,
            label_cache=label_cache,
            source_mask_cache=source_mask_cache,
            donor_key=donor_by_key.get(key),
            min_keep_ratio=float(args.min_keep_ratio),
            barrier_radius=int(args.barrier_radius),
        )
        source_mask = payload["source_mask"]
        if not np.any(source_mask):
            continue
        scene = key[1]
        frame_id = int(key[2])
        source_mask_id = int(key[3])
        frame_writer.ensure_frame(scene, frame_id, source_mask.shape)
        object_id = str(link.get("object_hypothesis_id", ""))
        base_score = _num(link.get("mask_selected_score"), _num(link.get("adapter_score_raw"), 1.0))
        for variant_id, mask in payload["variant_masks"].items():
            new_id = frame_writer.add_mask(variant_id, mask)
            if new_id <= 0:
                continue
            selected_area = int(np.count_nonzero(mask))
            source_area = int(payload["source_area"])
            mv_object_id = f"{variant_id}:{object_id}"
            gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"
            generated_rows.append(
                {
                    "schema_version": "stream4d_v93_phase4_edge_generated_mask_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "split": link.get("split", "dev"),
                    "window_id": link.get("window_id", source.get("window_id", "")),
                    "frame_id": frame_id,
                    "source_mask_id": source_mask_id,
                    "new_mask_id": int(new_id),
                    "object_hypothesis_id": object_id,
                    "generated_mask_path": _rel(gen_path),
                    "source_mask_area": source_area,
                    "generated_mask_area": selected_area,
                    "generated_mask_area_ratio": float(selected_area / max(1, source_area)),
                    "nested_edge_label_count": len(payload["nested_labels"]),
                    "competing_edge_label_count": len(payload["competing_labels"]),
                    "repeated_edge_label_count": len(payload["repeated_labels"]),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "notes": payload["variant_notes"].get(variant_id, ""),
                    "created_at": created_at,
                }
            )
            mv_rows.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "source_variant": variant_id,
                    "variant": variant_id,
                    "mv_object_id": mv_object_id,
                    "frame_id": frame_id,
                    "mask_id": int(new_id),
                    "frame_mask_score": float(base_score),
                    "object_score": float(base_score),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"v93_phase4_{variant_id}_from_phase1_mask_edges",
                }
            )
            if (variant_id, mv_object_id) not in object_rows_seen:
                object_rows_seen.add((variant_id, mv_object_id))
                object_rows.append(
                    {
                        "variant_id": variant_id,
                        "mv_object_id": mv_object_id,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
            quality_rows.append(
                {
                    "schema_version": "stream4d_v93_phase4_edge_quality_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "split": "dev",
                    "window_id": link.get("window_id", ""),
                    "frame_id": frame_id,
                    "source_mask_id": source_mask_id,
                    "object_hypothesis_id": object_id,
                    "source_mask_area": int(source_area),
                    "generated_mask_area": int(selected_area),
                    "generated_mask_area_ratio": float(selected_area / max(1, source_area)),
                    "nested_edge_label_count": len(payload["nested_labels"]),
                    "competing_edge_label_count": len(payload["competing_labels"]),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": created_at,
                }
            )

    frame_writer.flush()

    repeated_support_rows = sum(1 for values in edge_groups.values() if values.get("repeated_multiview"))
    if repeated_support_rows == 0:
        failure_rows.append(
            {
                "schema_version": "stream4d_v93_phase4_edge_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": "E4_repeated_multiview_edge",
                "failure_type": "NO_REPEATED_MULTIVIEW_EDGE_SUPPORT",
                "repair_direction": "do not fabricate E4; implement repeated_multiview edge extraction across views/windows before materializing this variant",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )

    config_rows = _variant_config_rows(str(args.base_source_variant), created_at, int(args.barrier_radius), float(args.min_keep_ratio))
    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "edge_materialization_quality_rows.csv", quality_rows)

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    radius_sweep.OUT = out
    for variant_id in MATERIALIZED_VARIANTS:
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, v91repair._phase8_baselines())

    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "control_metric_rows.csv", control_rows)
    variant_metric_rows = _metric_rows_with_common(control_rows, created_at)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)

    gate_rows = []
    for row in variant_metric_rows:
        gate_rows.append(
            {
                "schema_version": "stream4d_v93_phase4_edge_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("variant_id", ""),
                "scene_id": "ALL_DEV",
                "split": "dev",
                "gate_same_frame_collision_count_eq_0": row.get("gate_same_frame_collision_count_eq_0", ""),
                "gate_missing_mask_raster_count_eq_0": row.get("gate_missing_mask_raster_count_eq_0", ""),
                "gate_uses_gt_for_prediction_false": row.get("gate_uses_gt_for_prediction_false", ""),
                "gate_uses_future_false": row.get("gate_uses_future_false", ""),
                "v91_phase8_progress_gate_pass": row.get("v91_phase8_progress_gate_pass", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows)

    best = max(variant_metric_rows, key=lambda row: _num(row.get("mean_MV_AP_window"), -999.0), default={})
    summary = {
        "schema": "stream4d_v93_phase4_edge_only_materialization_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE4_EDGE_ONLY_MATERIALIZATION_WITH_E4_UNSUPPORTED"
        if len(variant_metric_rows) == len(MATERIALIZED_VARIANTS)
        else "BLOCK_V93_PHASE4_EDGE_ONLY_MATERIALIZATION",
        "base_source_variant": str(args.base_source_variant),
        "materialized_variant_count": len(variant_metric_rows),
        "materialized_variants": [row.get("variant_id", "") for row in variant_metric_rows],
        "unsupported_variant_count": 1 if repeated_support_rows == 0 else 0,
        "unsupported_variants": ["E4_repeated_multiview_edge"] if repeated_support_rows == 0 else [],
        "best_edge_variant_id": best.get("variant_id", ""),
        "best_edge_MV_AP_window": best.get("mean_MV_AP_window", ""),
        "best_edge_MV_AP50_window": best.get("mean_MV_AP50_window", ""),
        "best_edge_control_gap_vs_whole_source": best.get("control_gap_vs_whole_source", ""),
        "best_edge_control_gap_vs_random_edge": best.get("control_gap_vs_random_edge", ""),
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(object_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "mv_metric_rows": len(metric_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_failure_rows": len(failure_rows),
        },
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "notes": "E1/E2/E3/R0/R1 are materialized from Phase1 mask-edge registry and mask rasters; E4 is not fabricated because repeated_multiview edge rows are absent.",
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V93_PHASE1 / "source_container_rows.csv",
                V93_PHASE1 / "object_container_link_rows.csv",
                V93_PHASE1 / "mask_edge_hypothesis_rows.csv",
                V93_PHASE0 / "summary.json",
            ]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "edge_materialization_quality_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "control_metric_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize v93 Phase4 edge-only diagnostic readouts and evaluate MV_AP.")
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--base-source-variant", default=BASE_SOURCE_VARIANT)
    parser.add_argument("--barrier-radius", type=int, default=4)
    parser.add_argument("--min-keep-ratio", type=float, default=0.10)
    parser.add_argument("--clean", action="store_true", help="Remove output root before rebuilding.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
