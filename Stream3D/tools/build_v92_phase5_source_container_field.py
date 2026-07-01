from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


DEFAULT_OUT = ROOT / "outputs/audit/v92_phase5_source_container_field"
DEFAULT_REGION_ROOT = ROOT / "outputs/audit/v92_phase4_semantic_region_affinity"
DEFAULT_LINK_ROWS = ROOT / "outputs/audit/v92_phase1_source_container_registry/object_container_link_rows.csv"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v92_phase1_source_container_registry/source_container_rows.csv"
DEFAULT_LOWRES_SUPPORT = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
DEFAULT_HR2_SUPPORT = ROOT / "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16/highres_native_carrier_support_rows.csv"
BASE_SOURCE_VARIANT = "V91_AD4_sr2_adapt_sig8_b05_j075_r12_source"

MEMBERSHIP_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "source_mask_id",
    "object_hypothesis_id",
    "variant_id",
    "region_id",
    "p_object",
    "unary_d4rt",
    "unary_semantic",
    "negative_witness_penalty",
    "pairwise_smoothness_score",
    "selected_as_object",
    "uses_region_edge_graph",
    "uses_gt_for_prediction",
    "uses_future",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


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
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _variant_specs(profile: str = "default") -> list[dict[str, Any]]:
    base = [
        {"variant_id": "V92_F0_whole_source_mask", "kind": "whole", "family": "real"},
        {"variant_id": "V92_F1_d4rt_seed_only", "kind": "d4rt_low", "family": "real"},
        {"variant_id": "V92_F2_radio_region_only", "kind": "semantic", "family": "real"},
        {"variant_id": "V92_F3_d4rt_plus_radio_unary", "kind": "d4rt_semantic_unary", "family": "real"},
        {"variant_id": "V92_F4_d4rt_radio_graph", "kind": "d4rt_semantic_graph", "family": "real"},
        {"variant_id": "V92_F5_d4rt_radio_negative", "kind": "negative", "family": "real"},
        {"variant_id": "V92_F6_hr2_d4rt_radio_graph", "kind": "hr2_graph", "family": "real"},
        {"variant_id": "V92_C5_random_region_seed_control", "kind": "random", "family": "control"},
        {"variant_id": "V92_C7_d4rt_only_control", "kind": "d4rt_low", "family": "control"},
        {"variant_id": "V92_C8_radio_only_control", "kind": "semantic", "family": "control"},
    ]
    if profile == "default":
        return base
    if profile != "phase5c_tight_field":
        raise ValueError(f"unknown variant profile: {profile}")
    return [
        {"variant_id": "V92_F0_whole_source_mask", "kind": "whole", "family": "baseline"},
        {
            "variant_id": "V92_F7_tight_radio_edge085_cap055",
            "kind": "tight_graph",
            "family": "real",
            "edge_min_cos": 0.85,
            "edge_steps": 1,
            "sem_seed_quantile": 0.75,
            "sem_keep_quantile": 0.65,
            "area_cap": 0.55,
            "uses_highres_d4rt": False,
        },
        {
            "variant_id": "V92_F8_support_sem_intersection_cap050",
            "kind": "support_sem_intersection",
            "family": "real",
            "edge_min_cos": 0.88,
            "edge_steps": 1,
            "sem_seed_quantile": 0.70,
            "sem_keep_quantile": 0.60,
            "area_cap": 0.50,
            "uses_highres_d4rt": False,
        },
        {
            "variant_id": "V92_F9_boundary_trim_radio_edge080_cap065",
            "kind": "boundary_trim_graph",
            "family": "real",
            "edge_min_cos": 0.80,
            "edge_steps": 2,
            "sem_seed_quantile": 0.75,
            "sem_keep_quantile": 0.60,
            "area_cap": 0.65,
            "uses_highres_d4rt": False,
        },
        {
            "variant_id": "V92_F10_hr2_tight_radio_edge085_cap055",
            "kind": "tight_graph",
            "family": "real",
            "edge_min_cos": 0.85,
            "edge_steps": 1,
            "sem_seed_quantile": 0.75,
            "sem_keep_quantile": 0.65,
            "area_cap": 0.55,
            "uses_highres_d4rt": True,
        },
        {
            "variant_id": "V92_F11_core_seed_labelprop_cap060",
            "kind": "core_seed_labelprop",
            "family": "real",
            "edge_min_cos": 0.82,
            "edge_steps": 2,
            "sem_seed_quantile": 0.80,
            "sem_keep_quantile": 0.62,
            "area_cap": 0.60,
            "uses_highres_d4rt": True,
        },
        {
            "variant_id": "V92_C9_random_same_mass_tight_control",
            "kind": "random_same_mass_tight",
            "family": "control",
            "area_cap": 0.55,
            "uses_highres_d4rt": False,
        },
    ]


def _load_source_meta(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        scene = str(row.get("scene_id", ""))
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id", row.get("mask_id", -1)), -1)
        if not scene or frame_id < 0 or mask_id <= 0:
            continue
        key = (scene, int(frame_id), int(mask_id))
        if key not in out or str(row.get("variant_id")) == "B0_local_only":
            out[key] = row
    return out


def _load_links(path: Path, source_variant: str) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        if row.get("variant_id") != source_variant:
            continue
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        scene = str(row.get("scene_id", ""))
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("source_mask_id"), -1)
        if not scene or frame_id < 0 or mask_id <= 0:
            continue
        by_key[(scene, int(frame_id), int(mask_id))].append(row)
    return by_key


def _load_support_points(path: Path, selected_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], list[tuple[float, float]]]:
    out: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    for row in _read_csv(path):
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        if not _bool(row.get("native_support_allowed", "True")):
            continue
        scene = str(row.get("scene_id", ""))
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        key = (scene, int(frame_id), int(mask_id))
        if key not in selected_keys:
            continue
        out[key].append((_num(row.get("carrier_uv_x")), _num(row.get("carrier_uv_y"))))
    return out


def _region_index(region_id: str) -> int:
    tail = str(region_id).split(":")[-1]
    if tail.startswith("r") and tail[1:].isdigit():
        return int(tail[1:])
    return -1


def _load_edge_adjacency(path: Path, selected_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[int, list[tuple[int, float]]]]:
    adjacency: dict[tuple[str, int, int], dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("scene_id", "")),
                _int(row.get("frame_id"), -1),
                _int(row.get("source_mask_id"), -1),
            )
            if key not in selected_keys:
                continue
            a = _region_index(str(row.get("region_id_a", "")))
            b = _region_index(str(row.get("region_id_b", "")))
            if a < 0 or b < 0:
                continue
            cos = _num(row.get("radio_cosine"), 0.0)
            adjacency[key][a].append((b, cos))
            adjacency[key][b].append((a, cos))
    return {key: dict(value) for key, value in adjacency.items()}


def _stable_random_indices(n: int, k: int, seed_text: str) -> set[int]:
    if n <= 0 or k <= 0:
        return set()
    k = min(n, k)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(np.arange(n), size=k, replace=False).tolist())


def _expand(coords: list[tuple[int, int]], selected: set[int], steps: int = 1) -> set[int]:
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}
    out = set(selected)
    for _ in range(int(steps)):
        cur = set(out)
        for idx in cur:
            y, x = coords[idx]
            for nb in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if nb in coord_to_idx:
                    out.add(coord_to_idx[nb])
    return out


def _edge_expand(edge_adj: dict[int, list[tuple[int, float]]], selected: set[int], *, min_cos: float, steps: int) -> set[int]:
    out = set(selected)
    for _ in range(int(steps)):
        cur = set(out)
        for idx in cur:
            for nb, cos in edge_adj.get(idx, []):
                if float(cos) >= float(min_cos):
                    out.add(int(nb))
    return out


def _quantile_seed(values: list[float], quantile: float) -> set[int]:
    if not values:
        return set()
    threshold = float(np.quantile(np.asarray(values, dtype=np.float32), float(quantile)))
    return {i for i, value in enumerate(values) if float(value) >= threshold}


def _cap_selected_by_area(
    selected: set[int],
    *,
    nodes: list[dict[str, Any]],
    max_fraction: float,
    scores: list[float],
    must_keep: set[int] | None = None,
) -> set[int]:
    if not selected or max_fraction <= 0.0 or max_fraction >= 1.0:
        return set(selected)
    total_area = sum(max(1, _int(node.get("pixel_count"), 1)) for node in nodes)
    target_area = max(1.0, float(max_fraction) * float(total_area))
    must_keep = set(must_keep or set()) & set(selected)
    ordered = sorted(
        selected,
        key=lambda idx: (
            idx in must_keep,
            float(scores[idx]) if idx < len(scores) else 0.0,
            max(1, _int(nodes[idx].get("pixel_count"), 1)),
        ),
        reverse=True,
    )
    out: set[int] = set()
    area = 0
    for idx in ordered:
        node_area = max(1, _int(nodes[idx].get("pixel_count"), 1))
        if out and area + node_area > target_area and idx not in must_keep:
            continue
        out.add(int(idx))
        area += int(node_area)
        if area >= target_area and must_keep.issubset(out):
            break
    return out or set(selected)


def _support_counts(nodes: list[dict[str, Any]], points: list[tuple[float, float]], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    counts = [0 for _ in nodes]
    if not points:
        return counts
    pixel_points = [(float(x) * (w - 1), float(y) * (h - 1)) for x, y in points]
    for i, node in enumerate(nodes):
        x0 = _int(node.get("bbox_x0"), 0)
        x1 = _int(node.get("bbox_x1"), 0)
        y0 = _int(node.get("bbox_y0"), 0)
        y1 = _int(node.get("bbox_y1"), 0)
        counts[i] = sum(1 for x, y in pixel_points if x0 <= x <= x1 and y0 <= y <= y1)
    return counts


def _select_nodes(
    *,
    spec: dict[str, Any],
    nodes: list[dict[str, Any]],
    low_counts: list[int],
    hr_counts: list[int],
    source_key: tuple[str, int, int],
    broad: bool,
    edge_adj: dict[int, list[tuple[int, float]]] | None = None,
    use_edge_graph: bool = False,
) -> tuple[set[int], list[float], list[float], list[float], list[float]]:
    n = len(nodes)
    sem = [_num(row.get("source_mean_cosine"), 0.0) for row in nodes]
    coords = [(_int(row.get("feature_y"), 0), _int(row.get("feature_x"), 0)) for row in nodes]
    q50 = float(np.quantile(np.asarray(sem, dtype=np.float32), 0.50)) if sem else 1.0
    q65 = float(np.quantile(np.asarray(sem, dtype=np.float32), 0.65)) if sem else 1.0
    q75 = float(np.quantile(np.asarray(sem, dtype=np.float32), 0.75)) if sem else 1.0
    low_seed = {i for i, count in enumerate(low_counts) if count > 0}
    hr_seed = {i for i, count in enumerate(hr_counts) if count > 0}
    active_seed = hr_seed if _bool(spec.get("uses_highres_d4rt")) else low_seed
    sem_seed = {i for i, score in enumerate(sem) if score >= q50}
    strong_sem = {i for i, score in enumerate(sem) if score >= q75}
    kind = str(spec["kind"])
    if kind == "whole":
        selected = set(range(n))
    elif kind == "d4rt_low":
        selected = _expand(coords, low_seed, steps=1)
    elif kind == "semantic":
        selected = set(sem_seed)
    elif kind == "d4rt_semantic_unary":
        selected = _expand(coords, low_seed, steps=1) | strong_sem
    elif kind == "d4rt_semantic_graph":
        seed = _expand(coords, low_seed, steps=1) | strong_sem
        if use_edge_graph and edge_adj:
            selected = _edge_expand(edge_adj, seed, min_cos=0.78, steps=2)
        else:
            selected = _expand(coords, seed, steps=1)
            selected |= {i for i, score in enumerate(sem) if score >= q65 and any(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]) == 1 for j in selected)}
    elif kind == "negative":
        seed = _expand(coords, low_seed, steps=1) | strong_sem
        base = _edge_expand(edge_adj or {}, seed, min_cos=0.80, steps=2) if use_edge_graph and edge_adj else _expand(coords, seed, steps=1)
        selected = {
            i
            for i in base
            if not (_bool(nodes[i].get("boundary_token")) and low_counts[i] == 0 and sem[i] < (q65 if broad else q50))
        }
    elif kind == "hr2_graph":
        seed = _expand(coords, hr_seed, steps=1) | strong_sem
        if use_edge_graph and edge_adj:
            selected = _edge_expand(edge_adj, seed, min_cos=0.78, steps=2)
        else:
            selected = _expand(coords, seed, steps=1)
            selected |= {i for i, score in enumerate(sem) if score >= q65 and any(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]) == 1 for j in selected)}
    elif kind == "random":
        ref_k = max(1, len(_expand(coords, low_seed, steps=1) | strong_sem))
        selected = _stable_random_indices(n, ref_k, f"{source_key}:{spec['variant_id']}")
    elif kind == "tight_graph":
        sem_core = _quantile_seed(sem, _num(spec.get("sem_seed_quantile"), 0.75))
        sem_keep = _quantile_seed(sem, _num(spec.get("sem_keep_quantile"), 0.65))
        seed = active_seed | (sem_core & _expand(coords, active_seed, steps=1) if active_seed else sem_core)
        if use_edge_graph and edge_adj:
            selected = _edge_expand(
                edge_adj,
                seed,
                min_cos=_num(spec.get("edge_min_cos"), 0.85),
                steps=_int(spec.get("edge_steps"), 1),
            )
            selected &= sem_keep | seed
        else:
            selected = _expand(coords, seed, steps=1) & (sem_keep | seed)
    elif kind == "support_sem_intersection":
        sem_core = _quantile_seed(sem, _num(spec.get("sem_seed_quantile"), 0.70))
        sem_keep = _quantile_seed(sem, _num(spec.get("sem_keep_quantile"), 0.60))
        seed = active_seed & sem_keep
        if not seed:
            seed = active_seed | sem_core
        if use_edge_graph and edge_adj:
            selected = _edge_expand(
                edge_adj,
                seed,
                min_cos=_num(spec.get("edge_min_cos"), 0.88),
                steps=_int(spec.get("edge_steps"), 1),
            )
            selected &= sem_keep | seed
        else:
            selected = _expand(coords, seed, steps=1) & (sem_keep | seed)
    elif kind == "boundary_trim_graph":
        sem_core = _quantile_seed(sem, _num(spec.get("sem_seed_quantile"), 0.75))
        sem_keep = _quantile_seed(sem, _num(spec.get("sem_keep_quantile"), 0.60))
        seed = _expand(coords, active_seed, steps=1) | sem_core
        base = (
            _edge_expand(edge_adj or {}, seed, min_cos=_num(spec.get("edge_min_cos"), 0.80), steps=_int(spec.get("edge_steps"), 2))
            if use_edge_graph and edge_adj
            else _expand(coords, seed, steps=1)
        )
        selected = {
            i
            for i in base
            if i in sem_keep or active_seed and i in _expand(coords, active_seed, steps=1)
        }
        selected = {
            i
            for i in selected
            if not (_bool(nodes[i].get("boundary_token")) and low_counts[i] == 0 and hr_counts[i] == 0 and sem[i] < q75)
        }
    elif kind == "core_seed_labelprop":
        sem_core = _quantile_seed(sem, _num(spec.get("sem_seed_quantile"), 0.80))
        sem_keep = _quantile_seed(sem, _num(spec.get("sem_keep_quantile"), 0.62))
        seed = active_seed | sem_core
        selected = _edge_expand(
            edge_adj or {},
            seed,
            min_cos=_num(spec.get("edge_min_cos"), 0.82),
            steps=_int(spec.get("edge_steps"), 2),
        ) if use_edge_graph and edge_adj else _expand(coords, seed, steps=2)
        selected &= sem_keep | seed
    elif kind == "random_same_mass_tight":
        sem_core = _quantile_seed(sem, 0.75)
        ref = _cap_selected_by_area(
            sem_core | _expand(coords, active_seed, steps=1),
            nodes=nodes,
            max_fraction=_num(spec.get("area_cap"), 0.55),
            scores=sem,
            must_keep=active_seed,
        )
        selected = _stable_random_indices(n, len(ref), f"{source_key}:{spec['variant_id']}")
    else:
        selected = set()
    if not selected and n:
        selected = {int(np.argmax(np.asarray(sem, dtype=np.float32)))}
    unary_d4rt = [min(1.0, math.log1p(float(low_counts[i] + hr_counts[i])) / math.log(8.0)) for i in range(n)]
    unary_sem = [float(max(0.0, min(1.0, (sem[i] - min(sem)) / max(1e-6, max(sem) - min(sem))))) if sem else 0.0 for i in range(n)]
    neg = [1.0 if _bool(nodes[i].get("boundary_token")) and low_counts[i] == 0 and hr_counts[i] == 0 else 0.0 for i in range(n)]
    smooth = [1.0 - abs(float(sem[i]) - q50) for i in range(n)]
    if "area_cap" in spec and kind != "random_same_mass_tight":
        rank_scores = [
            0.45 * unary_sem[i]
            + 0.35 * unary_d4rt[i]
            + 0.20 * smooth[i]
            - 0.30 * neg[i]
            for i in range(n)
        ]
        selected = _cap_selected_by_area(
            selected,
            nodes=nodes,
            max_fraction=_num(spec.get("area_cap"), 1.0),
            scores=rank_scores,
            must_keep=active_seed,
        )
    return selected, unary_d4rt, unary_sem, neg, smooth


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


def _node_mask(nodes: list[dict[str, Any]], selected: set[int], source_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(source_mask, dtype=bool)
    for idx in selected:
        node = nodes[idx]
        x0 = _int(node.get("bbox_x0"), 0)
        x1 = _int(node.get("bbox_x1"), 0)
        y0 = _int(node.get("bbox_y0"), 0)
        y1 = _int(node.get("bbox_y1"), 0)
        out[y0 : y1 + 1, x0 : x1 + 1] |= source_mask[y0 : y1 + 1, x0 : x1 + 1]
    return out & source_mask


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    region_root = _resolve(args.region_root)
    source_meta = _load_source_meta(_resolve(args.source_container_rows))
    links = _load_links(_resolve(args.object_container_link_rows), str(args.base_source_variant))
    selected_keys = set(links)
    low_support = _load_support_points(_resolve(args.lowres_support_rows), selected_keys)
    hr_support = _load_support_points(_resolve(args.highres_support_rows), selected_keys)
    edge_adjacency = _load_edge_adjacency(_resolve(args.region_edge_rows), selected_keys) if bool(args.use_edge_graph) else {}
    specs = _variant_specs(str(args.variant_profile))
    variant_ids = [spec["variant_id"] for spec in specs]
    radius_sweep.OUT = out
    frame_writer = FrameWriter(out, variant_ids)
    membership_path = out / "field_region_membership_rows.csv"
    membership_handle = membership_path.open("w", newline="", encoding="utf-8")
    membership_writer = csv.DictWriter(membership_handle, fieldnames=MEMBERSHIP_FIELDS)
    membership_writer.writeheader()

    config_rows = [
        {
            "variant_id": spec["variant_id"],
            "variant_kind": spec["kind"],
            "family": spec["family"],
            "variant_profile": str(args.variant_profile),
            "region_source": _rel(region_root / "region_node_rows.csv"),
            "base_source_variant": str(args.base_source_variant),
            "uses_radio_region_feature": spec["kind"] not in {"whole", "d4rt_low"},
            "uses_d4rt_witness": spec["kind"]
            in {
                "d4rt_low",
                "d4rt_semantic_unary",
                "d4rt_semantic_graph",
                "negative",
                "hr2_graph",
                "tight_graph",
                "support_sem_intersection",
                "boundary_trim_graph",
                "core_seed_labelprop",
            },
            "uses_highres_d4rt": bool(spec.get("uses_highres_d4rt", spec["kind"] == "hr2_graph")),
            "edge_min_cos": spec.get("edge_min_cos", ""),
            "edge_steps": spec.get("edge_steps", ""),
            "sem_seed_quantile": spec.get("sem_seed_quantile", ""),
            "sem_keep_quantile": spec.get("sem_keep_quantile", ""),
            "area_cap": spec.get("area_cap", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    support_quality_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    nodes_by_key: list[dict[str, Any]] = []
    current_key: tuple[str, int, int] | None = None
    label_cache: dict[tuple[str, int], np.ndarray] = {}

    def process_group(key: tuple[str, int, int] | None, nodes: list[dict[str, Any]]) -> None:
        if key is None or not nodes or key not in links:
            return
        scene, frame_id, mask_id = key
        meta = source_meta.get(key, {})
        mask_path = _resolve(meta.get("mask_path", ""))
        frame_key = (scene, int(frame_id))
        if frame_key not in label_cache:
            label_cache[frame_key] = _read_label(mask_path)
        label = label_cache[frame_key]
        source_mask = label == int(mask_id)
        if not np.any(source_mask):
            return
        frame_writer.ensure_frame(scene, int(frame_id), label.shape)
        low_counts = _support_counts(nodes, low_support.get(key, []), label.shape)
        hr_counts = _support_counts(nodes, hr_support.get(key, []), label.shape)
        broad = _bool(meta.get("broad_mask_flag")) or _num(meta.get("mask_area_ratio"), 0.0) >= 0.35
        for link in links[key]:
            object_id = str(link.get("object_hypothesis_id", ""))
            base_score = _num(link.get("mask_selected_score"), _num(link.get("adapter_score_raw"), 1.0))
            for spec in specs:
                variant_id = spec["variant_id"]
                selected, unary_d4rt, unary_sem, neg, smooth = _select_nodes(
                    spec=spec,
                    nodes=nodes,
                    low_counts=low_counts,
                    hr_counts=hr_counts,
                    source_key=key,
                    broad=broad,
                    edge_adj=edge_adjacency.get(key, {}),
                    use_edge_graph=bool(args.use_edge_graph),
                )
                for i, node in enumerate(nodes):
                    p_object = 0.55 * unary_sem[i] + 0.35 * unary_d4rt[i] + 0.10 * smooth[i] - 0.25 * neg[i]
                    membership_writer.writerow(
                        {
                            "schema_version": "stream4d_v92_phase5_membership_v1",
                            "phase_id": "v92_phase5_source_container_field",
                            "run_id": args.run_id,
                            "scene_id": scene,
                            "split": link.get("split", "dev"),
                            "window_id": link.get("window_id", node.get("window_id", "")),
                            "frame_id": int(frame_id),
                            "source_mask_id": int(mask_id),
                            "object_hypothesis_id": object_id,
                            "variant_id": variant_id,
                            "region_id": node.get("region_id", ""),
                            "p_object": float(p_object),
                            "unary_d4rt": float(unary_d4rt[i]),
                            "unary_semantic": float(unary_sem[i]),
                            "negative_witness_penalty": float(neg[i]),
                            "pairwise_smoothness_score": float(smooth[i]),
                            "selected_as_object": bool(i in selected),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "uses_region_edge_graph": bool(args.use_edge_graph),
        }
                    )
                mask = _node_mask(nodes, selected, source_mask)
                new_id = frame_writer.add_mask(variant_id, mask)
                if new_id <= 0:
                    continue
                selected_area = int(np.count_nonzero(mask))
                source_area = int(np.count_nonzero(source_mask))
                score = float(base_score * (0.75 + 0.25 * min(1.0, len(selected) / max(1, len(nodes)))))
                gen_path = out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"
                generated_rows.append(
                    {
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "window_id": link.get("window_id", ""),
                        "frame_id": int(frame_id),
                        "source_mask_id": int(mask_id),
                        "new_mask_id": int(new_id),
                        "object_hypothesis_id": object_id,
                        "generated_mask_path": _rel(gen_path),
                        "source_mask_area": int(source_area),
                        "generated_mask_area": int(selected_area),
                        "generated_mask_area_ratio": float(selected_area / max(1, source_area)),
                        "selected_region_count": int(len(selected)),
                        "total_region_count": int(len(nodes)),
                        "lowres_support_count": int(sum(low_counts)),
                        "hr2_support_count": int(sum(hr_counts)),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
                mv_rows.append(
                    {
                        "split": "dev",
                        "scene_id": scene,
                        "source_variant": variant_id,
                        "variant": variant_id,
                        "mv_object_id": f"{variant_id}:{object_id}",
                        "frame_id": int(frame_id),
                        "mask_id": int(new_id),
                        "frame_mask_score": score,
                        "object_score": score,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "materializable": True,
                        "selection_reason": f"v92_phase5_{spec['kind']}_source_container_region_field",
                    }
                )
                support_quality_rows.append(
                    {
                        "variant_id": variant_id,
                        "scene_id": scene,
                        "window_id": link.get("window_id", ""),
                        "mv_object_id": f"{variant_id}:{object_id}",
                        "frame_id": int(frame_id),
                        "source_mask_id": int(mask_id),
                        "generated_mask_area": int(selected_area),
                        "source_mask_area": int(source_area),
                        "generated_mask_area_ratio": float(selected_area / max(1, source_area)),
                        "lowres_support_count": int(sum(low_counts)),
                        "hr2_support_count": int(sum(hr_counts)),
                        "selected_region_count": int(len(selected)),
                        "total_region_count": int(len(nodes)),
                        "broad_risk": bool(broad),
                    }
                )

    region_rows_path = region_root / "region_node_rows.csv"
    with region_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))
            if key not in selected_keys:
                continue
            if current_key is not None and key != current_key:
                process_group(current_key, nodes_by_key)
                nodes_by_key = []
            current_key = key
            nodes_by_key.append(row)
    process_group(current_key, nodes_by_key)
    frame_writer.flush()
    membership_handle.close()

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for spec in specs:
        variant_id = spec["variant_id"]
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, v91repair._phase8_baselines())
    best = max(
        control_rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )
    passing = [row for row in control_rows if _bool(row.get("v91_phase8_progress_gate_pass"))]
    if not passing:
        for row in control_rows:
            failure_rows.append(
                {
                    "variant_id": row.get("variant_id", ""),
                    "failure_type": "phase5_dev_gate_fail",
                    "repair_direction": "if semantic-only beats fused, inspect D4RT witness quality/fusion; otherwise tune region graph family up to plan limit",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                    "best_control_MV_AP_window": row.get("best_control_MV_AP_window", ""),
                    "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    summary = {
        "phase_id": "v92_phase5_source_container_field",
        "schema": "stream4d_v92_phase5_source_container_field_summary_v1",
        "run_id": args.run_id,
        "variant_profile": str(args.variant_profile),
        "base_source_variant": str(args.base_source_variant),
        "variant_count": len(specs),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_phase5_dev_gate_pass": bool(passing),
        "decision": "PASS_V92_PHASE5_DEV_GATE" if passing else "NO_GO_V92_PHASE5_FIELD_READOUT_NO_AP_GAIN",
        "row_counts": {
            "field_variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "field_failure_rows": len(failure_rows),
            "support_quality_rows": len(support_quality_rows),
        },
        "field_region_membership_rows_sha256": _sha256(membership_path),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_csv(out / "field_variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", [{"variant_id": row["variant"], "mv_object_id": row["mv_object_id"], "uses_gt_for_prediction": False, "uses_future": False} for row in mv_rows])
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "control_metric_rows.csv", control_rows)
    _write_csv(out / "support_quality_rows.csv", support_quality_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_csv(out / "field_failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "field_variant_config_rows.csv",
        out / "field_region_membership_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "control_metric_rows.csv",
        out / "support_quality_rows.csv",
        out / "casebook_rows.csv",
        out / "field_failure_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v92 Phase5 source-container object membership field readout.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--region-root", default=str(DEFAULT_REGION_ROOT))
    parser.add_argument("--object-container-link-rows", default=str(DEFAULT_LINK_ROWS))
    parser.add_argument("--source-container-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--lowres-support-rows", default=str(DEFAULT_LOWRES_SUPPORT))
    parser.add_argument("--highres-support-rows", default=str(DEFAULT_HR2_SUPPORT))
    parser.add_argument("--region-edge-rows", default=str(DEFAULT_REGION_ROOT / "region_edge_rows.csv"))
    parser.add_argument("--base-source-variant", default=BASE_SOURCE_VARIANT)
    parser.add_argument("--run-id", default="v92_phase5_source_container_field")
    parser.add_argument("--use-edge-graph", action="store_true")
    parser.add_argument("--variant-profile", choices=["default", "phase5c_tight_field"], default="default")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
