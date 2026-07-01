#!/usr/bin/env python3
"""Build ACL2 v85 Phase1 anchor-pair universe.

This is a data-materialization step only. Existing v84 QK compatibility values
are kept as proxies and are not promoted to true SWA Q/K features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_TOKENS = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates/ruler_candidate_tokens.csv"
)
DEFAULT_PAIR_SUMMARY = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates/ruler_candidate_pair_summary.csv"
)
DEFAULT_POSITIONS = Path(
    "results/acl2_v84tf_memory_ruler_audit/phase15_anchor_route_mask_materialization/anchor_route_mask_positions.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe")

PATCH_GRID = (19, 66)
IMAGE_HW = (376.0, 1408.0)

REQUIRED_ROW_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "case_label",
    "quality_label",
    "prev_frame_id",
    "curr_frame_id",
    "prev_patch_id",
    "curr_patch_id",
    "prev_pixel_x",
    "prev_pixel_y",
    "curr_pixel_x",
    "curr_pixel_y",
    "prev_sem_label",
    "curr_sem_label",
    "prev_sem_conf",
    "curr_sem_conf",
    "same_label",
    "same_role",
    "same_object_available",
    "same_object_flag",
    "cross_boundary_flag",
    "dynamic_risk_flag",
    "zero_conf_flag",
    "raw_overlap_residual",
    "confidence_weighted_residual",
    "local_shape_residual",
    "pairwise_distance_ratio_residual",
    "parallax_score",
    "local_3d_spread_prev",
    "local_3d_spread_curr",
    "geometry_leverage_score",
    "read_usage_current",
    "swa_qk_proxy",
    "true_route_available",
    "true_route_mass",
    "anchor_weight",
    "anchor_maturity",
    "anchor_support_class",
    "risk_reason",
    "feature_q_available",
    "feature_k_available",
    "feature_v_available_diagnostic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--pair-summary", type=Path, default=DEFAULT_PAIR_SUMMARY)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--patch-neighborhood-radius",
        type=int,
        default=0,
        help="Match raw overlap pixels within +/- radius patch cells when exact patch mapping is sparse.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], *, fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def parse_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mode_int(values: list[int]) -> int | None:
    vals = [int(v) for v in values]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def seq_norm(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(2) if text else ""


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return seq_norm(row.get("seq")), str(row.get("prev_chunk", "")).strip(), str(row.get("curr_chunk", "")).strip()


def patch_id(py: int | None, px: int | None) -> int | None:
    if py is None or px is None:
        return None
    if py < 0 or px < 0:
        return None
    return py * PATCH_GRID[1] + px


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_artifact_paths(value: Any) -> list[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    paths: list[str] = []
    if isinstance(parsed, list):
        paths = [str(item) for item in parsed]
    elif isinstance(parsed, str):
        paths = [parsed]
    else:
        paths = re.findall(r"results/[^\"'\]\s,]+", text)
    return [Path(path) for path in paths if path]


class QKFeatureLookup:
    required_keys = ("tap::pca_swa_current_q_layers", "tap::pca_swa_cache_k_layers")

    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}
        self.stats: Counter[str] = Counter()

    def lookup(self, row: Mapping[str, Any]) -> dict[str, Any]:
        candidates = [
            path
            for path in extract_artifact_paths(row.get("source_artifact_paths"))
            if "pca_features" in str(path) or path.name.endswith(".pt")
        ]
        for path in candidates:
            result = self._check_path(path)
            if result.get("feature_q_available") and result.get("feature_k_available"):
                return result
        self.stats["qk_feature_missing"] += 1
        return {
            "feature_q_available": False,
            "feature_k_available": False,
            "feature_source_path": "",
            "feature_schema": "",
            "feature_authority_source": "missing_direct_swa_qk_pca_dump",
        }

    def _check_path(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key not in self.cache:
            if not path.exists():
                self.cache[key] = {"ok": False, "reason": "path_missing"}
                self.stats["qk_feature_path_missing"] += 1
            else:
                try:
                    payload = torch_load(path)
                except Exception as exc:  # noqa: BLE001
                    self.cache[key] = {"ok": False, "reason": f"load_error:{type(exc).__name__}"}
                    self.stats["qk_feature_load_error"] += 1
                else:
                    ok = isinstance(payload, dict) and all(req in payload for req in self.required_keys)
                    self.cache[key] = {
                        "ok": bool(ok),
                        "schema": payload.get("schema", "") if isinstance(payload, dict) else "",
                        "reason": "direct_swa_qk_pca_dump" if ok else "required_qk_keys_missing",
                    }
                    self.stats["qk_feature_direct_dump_ok" if ok else "qk_feature_schema_missing"] += 1
        cached = self.cache[key]
        ok = bool(cached.get("ok"))
        return {
            "feature_q_available": ok,
            "feature_k_available": ok,
            "feature_source_path": str(path) if ok else "",
            "feature_schema": cached.get("schema", ""),
            "feature_authority_source": cached.get("reason", ""),
        }


def patch_indices(coords: Any) -> tuple[Any, Any, Any]:
    import torch

    y = coords[:, 0].float()
    x = coords[:, 1].float()
    py = torch.clamp((y / (IMAGE_HW[0] / PATCH_GRID[0])).floor().long(), 0, PATCH_GRID[0] - 1)
    px = torch.clamp((x / (IMAGE_HW[1] / PATCH_GRID[1])).floor().long(), 0, PATCH_GRID[1] - 1)
    flat = py * PATCH_GRID[1] + px
    return py, px, flat


def vector_spread(points: Any) -> float | None:
    import torch

    if points is None or not torch.is_tensor(points) or points.numel() == 0:
        return None
    p = points.detach().cpu().float()
    if p.ndim != 2 or p.shape[0] == 0:
        return None
    center = p.mean(dim=0, keepdim=True)
    return float(torch.linalg.norm(p - center, dim=1).mean().item())


class RawOverlapLookup:
    def __init__(self, *, patch_neighborhood_radius: int = 0) -> None:
        self.cache: dict[str, Any] = {}
        self.stats: Counter[str] = Counter()
        self.patch_neighborhood_radius = max(0, int(patch_neighborhood_radius))

    def load(self, source_path: str) -> Any:
        if not source_path:
            self.stats["missing_source_path"] += 1
            return None
        if source_path not in self.cache:
            try:
                self.cache[source_path] = torch_load(Path(source_path))
                self.stats["raw_overlap_load_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                self.cache[source_path] = exc
                self.stats["raw_overlap_load_error"] += 1
        payload = self.cache[source_path]
        return None if isinstance(payload, Exception) else payload

    def lookup(self, row: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        py = safe_int(row.get("patch_y"))
        px = safe_int(row.get("patch_x"))
        source_path = str(row.get("source_path") or "")
        payload = self.load(source_path)
        out: dict[str, Any] = {
            "raw_coord_available": False,
            "raw_lookup_status": "missing_or_unloaded_raw_overlap",
        }
        if payload is None or py is None or px is None:
            return out
        required = [
            "prev_pixel_coords",
            "curr_pixel_coords",
            "prev_frame_ids",
            "curr_frame_ids",
            "prev_semantic_labels",
            "curr_semantic_labels",
            "prev_semantic_conf",
            "curr_semantic_conf",
        ]
        if not isinstance(payload, dict) or any(key not in payload for key in required):
            self.stats["raw_overlap_schema_missing"] += 1
            out["raw_lookup_status"] = "raw_overlap_schema_missing"
            return out

        curr_coords = payload["curr_pixel_coords"]
        prev_coords = payload["prev_pixel_coords"]
        if not torch.is_tensor(curr_coords) or not torch.is_tensor(prev_coords):
            self.stats["raw_overlap_tensor_missing"] += 1
            out["raw_lookup_status"] = "raw_overlap_tensor_missing"
            return out

        curr_py, curr_px, _ = patch_indices(curr_coords)
        radius = self.patch_neighborhood_radius
        if radius > 0:
            match = (curr_py - int(py)).abs().le(radius) & (curr_px - int(px)).abs().le(radius)
        else:
            match = (curr_py == int(py)) & (curr_px == int(px))
        idx = torch.nonzero(match, as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            self.stats["raw_overlap_patch_no_match"] += 1
            out["raw_lookup_status"] = "raw_overlap_patch_no_match"
            return out

        self.stats["raw_overlap_patch_match"] += 1
        prev_sel = prev_coords[idx].detach().cpu()
        curr_sel = curr_coords[idx].detach().cpu()
        prev_points = payload.get("prev_overlap_local_points")
        if prev_points is None:
            prev_points = payload.get("prev_overlap_points")
        curr_points = payload.get("curr_overlap_local_points")
        if curr_points is None:
            curr_points = payload.get("curr_overlap_points")
        if torch.is_tensor(prev_points):
            prev_points = prev_points[idx]
        if torch.is_tensor(curr_points):
            curr_points = curr_points[idx]

        prev_sem_labels = payload["prev_semantic_labels"][idx].detach().cpu().long().tolist()
        curr_sem_labels = payload["curr_semantic_labels"][idx].detach().cpu().long().tolist()
        prev_sem_conf = payload["prev_semantic_conf"][idx].detach().cpu().float().tolist()
        curr_sem_conf = payload["curr_semantic_conf"][idx].detach().cpu().float().tolist()
        prev_frames = payload["prev_frame_ids"][idx].detach().cpu().long().tolist()
        curr_frames = payload["curr_frame_ids"][idx].detach().cpu().long().tolist()

        prev_xyz = payload.get("prev_overlap_points")
        curr_xyz = payload.get("curr_overlap_points")
        local_shape_residual = None
        if torch.is_tensor(prev_xyz) and torch.is_tensor(curr_xyz):
            delta = prev_xyz[idx].detach().cpu().float() - curr_xyz[idx].detach().cpu().float()
            local_shape_residual = float(torch.linalg.norm(delta, dim=1).median().item())

        return {
            "raw_coord_available": True,
            "raw_lookup_status": f"raw_overlap_patch_match_radius{radius}",
            "prev_frame_id": mode_int(prev_frames),
            "curr_frame_id": mode_int(curr_frames),
            "prev_pixel_x": median(prev_sel[:, 1].float().tolist()),
            "prev_pixel_y": median(prev_sel[:, 0].float().tolist()),
            "curr_pixel_x": median(curr_sel[:, 1].float().tolist()),
            "curr_pixel_y": median(curr_sel[:, 0].float().tolist()),
            "prev_sem_label": mode_int(prev_sem_labels),
            "curr_sem_label": mode_int(curr_sem_labels),
            "prev_sem_conf": mean(prev_sem_conf),
            "curr_sem_conf": mean(curr_sem_conf),
            "same_label": mode_int(prev_sem_labels) == mode_int(curr_sem_labels),
            "local_shape_residual": local_shape_residual,
            "local_3d_spread_prev": vector_spread(prev_points),
            "local_3d_spread_curr": vector_spread(curr_points),
        }


def classify_anchor(row: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[str, str, str]:
    seq = seq_norm(row.get("seq"))
    quality = str(row.get("quality_type") or "")
    quality_source = str(row.get("quality_source") or "")
    case_type = str(row.get("case_type") or "")
    role = str(row.get("ruler_role") or "")
    risk_score = clamp01(safe_float(row.get("risk_score")))
    sem_conf = clamp01(safe_float(row.get("semantic_confidence")))
    purity = clamp01(safe_float(row.get("patch_purity")))
    geom = safe_float(row.get("geometry_leverage")) or 0.0
    residual = safe_float(row.get("overlap_residual"))
    cross_boundary = clamp01(safe_float(row.get("cross_boundary_proxy"))) >= 0.5
    zero_conf = sem_conf <= 0.0
    reasons: list[str] = []

    if seq == "01" and (quality == "low_conf_stress" or quality_source == "minconf0" or "lowconf" in case_type):
        reasons.append("seq01_minconf0_or_low_conf_stress_not_positive_support")
        return "bootstrap_unverified", "A_STRESS_SEQ01", ";".join(reasons)
    if quality == "low_conf_stress" or quality_source == "minconf0" or "lowconf" in case_type:
        reasons.append("low_conf_stress_not_positive_support")
        return "bootstrap_unverified", "A_RISK", ";".join(reasons)
    if zero_conf:
        reasons.append("zero_confidence")
    if cross_boundary:
        reasons.append("cross_boundary_proxy")
    if risk_score >= 0.65:
        reasons.append("high_v84_risk_score")
    if residual is not None and residual >= 0.10:
        reasons.append("high_overlap_residual")
    if role == "RULER_RISK":
        reasons.append("v84_ruler_role_risk")
    if reasons:
        return "bootstrap_unverified", "A_RISK", ";".join(reasons)

    if role == "RULER_ANCHOR" and sem_conf >= 0.45 and purity >= 0.50 and geom > 0.0:
        return "bootstrap_unverified", "A_STRONG_BOOTSTRAP", "strong_by_v84_anchor_role_no_mature_history_evidence"
    if role in {"RULER_CONTEXT", "RULER_DEGENERATE"}:
        return "diagnostic_only", "A_CONTEXT_DEGENERATE", f"v84_{role.lower()}"
    if geom > 0.0:
        return "diagnostic_only", "A_WEAK_GEOMETRY", "geometry_leverage_present_but_not_strong_anchor"
    return "diagnostic_only", "A_RISK", "no_positive_anchor_evidence"


def build_rows(tokens: list[dict[str, str]], *, patch_neighborhood_radius: int = 0) -> tuple[list[dict[str, Any]], Counter[str]]:
    raw_lookup = RawOverlapLookup(patch_neighborhood_radius=patch_neighborhood_radius)
    qk_lookup = QKFeatureLookup()
    rows: list[dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        seq = seq_norm(token.get("seq"))
        prev_chunk = safe_int(token.get("prev_chunk"))
        curr_chunk = safe_int(token.get("curr_chunk"))
        py = safe_int(token.get("patch_y"))
        px = safe_int(token.get("patch_x"))
        pid = patch_id(py, px)
        raw = raw_lookup.lookup(token)
        qk_feature = qk_lookup.lookup(token)
        maturity, support_class, class_reason = classify_anchor(token, raw)
        sem_conf = safe_float(token.get("semantic_confidence"))
        row: dict[str, Any] = {
            "pair_id": f"{seq}_{prev_chunk}_{curr_chunk}_tok{idx:05d}_p{pid if pid is not None else 'na'}",
            "seq": seq,
            "prev_chunk": prev_chunk,
            "curr_chunk": curr_chunk,
            "case_label": token.get("base_case_type"),
            "quality_label": token.get("quality_type"),
            "prev_frame_id": raw.get("prev_frame_id"),
            "curr_frame_id": raw.get("curr_frame_id", safe_int(token.get("frame_id"))),
            "prev_patch_id": pid,
            "curr_patch_id": pid,
            "prev_pixel_x": raw.get("prev_pixel_x"),
            "prev_pixel_y": raw.get("prev_pixel_y"),
            "curr_pixel_x": raw.get("curr_pixel_x"),
            "curr_pixel_y": raw.get("curr_pixel_y"),
            "prev_sem_label": raw.get("prev_sem_label", token.get("semantic_label")),
            "curr_sem_label": raw.get("curr_sem_label", token.get("semantic_label")),
            "prev_sem_conf": raw.get("prev_sem_conf", sem_conf),
            "curr_sem_conf": raw.get("curr_sem_conf", sem_conf),
            "same_label": raw.get("same_label"),
            "same_role": "",
            "same_object_available": False,
            "same_object_flag": "",
            "cross_boundary_flag": (safe_float(token.get("cross_boundary_proxy")) or 0.0) >= 0.5,
            "dynamic_risk_flag": "",
            "zero_conf_flag": (sem_conf or 0.0) <= 0.0,
            "raw_overlap_residual": safe_float(token.get("overlap_residual")),
            "confidence_weighted_residual": safe_float(token.get("confidence_weighted_residual")),
            "local_shape_residual": raw.get("local_shape_residual"),
            "pairwise_distance_ratio_residual": "",
            "parallax_score": safe_float(token.get("parallax_proxy")),
            "local_3d_spread_prev": raw.get("local_3d_spread_prev"),
            "local_3d_spread_curr": raw.get("local_3d_spread_curr", safe_float(token.get("geometry_spread"))),
            "geometry_leverage_score": safe_float(token.get("geometry_leverage")),
            "read_usage_current": safe_float(token.get("READ_usage")),
            "swa_qk_proxy": safe_float(token.get("QK_compatibility")),
            "true_route_available": False,
            "true_route_mass": "",
            "anchor_weight": safe_float(token.get("ruler_anchor_score")),
            "anchor_maturity": maturity,
            "anchor_support_class": support_class,
            "risk_reason": ";".join(
                part
                for part in [
                    class_reason,
                    f"raw_lookup={raw.get('raw_lookup_status')}",
                    "qk_feature_direct_pca_available" if qk_feature.get("feature_q_available") else "qk_feature_missing",
                    "route_mass_still_unavailable",
                ]
                if part
            ),
            "feature_q_available": qk_feature.get("feature_q_available", False),
            "feature_k_available": qk_feature.get("feature_k_available", False),
            "feature_v_available_diagnostic": safe_float(token.get("cache_V_compatibility")) is not None,
            "feature_source_path": qk_feature.get("feature_source_path"),
            "feature_schema": qk_feature.get("feature_schema"),
            "feature_authority_source": qk_feature.get("feature_authority_source"),
            "source_path": token.get("source_path"),
            "source_artifact_paths": token.get("source_artifact_paths"),
            "swa_usage_source": token.get("SWA_usage_source"),
            "v84_ruler_role": token.get("ruler_role"),
            "v84_case_type": token.get("case_type"),
            "patch_y": py,
            "patch_x": px,
            "raw_coord_available": raw.get("raw_coord_available", False),
            "raw_lookup_status": raw.get("raw_lookup_status"),
        }
        rows.append(row)
    stats = Counter()
    stats.update(raw_lookup.stats)
    stats.update(qk_lookup.stats)
    return rows, stats


def summarize_by_pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seq"], row["prev_chunk"], row["curr_chunk"])].append(row)
    out: list[dict[str, Any]] = []
    for (seq, prev_chunk, curr_chunk), group in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), int(item[0][2]))):
        case_counts = Counter(str(r.get("case_label") or "") for r in group)
        quality_counts = Counter(str(r.get("quality_label") or "") for r in group)
        class_counts = Counter(str(r.get("anchor_support_class") or "") for r in group)
        q_count = sum(1 for r in group if parse_bool(r.get("feature_q_available")) is True or r.get("feature_q_available") is True)
        k_count = sum(1 for r in group if parse_bool(r.get("feature_k_available")) is True or r.get("feature_k_available") is True)
        raw_count = sum(1 for r in group if parse_bool(r.get("raw_coord_available")) is True or r.get("raw_coord_available") is True)
        positive_count = sum(1 for r in group if r.get("anchor_support_class") in {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP"})
        zero_positive = sum(
            1
            for r in group
            if r.get("anchor_support_class") in {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP"}
            and (parse_bool(r.get("zero_conf_flag")) is True or r.get("zero_conf_flag") is True)
        )
        out.append(
            {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "case_label": case_counts.most_common(1)[0][0],
                "quality_label": quality_counts.most_common(1)[0][0],
                "anchor_pair_count": len(group),
                "positive_anchor_count": positive_count,
                "strong_mature_count": class_counts.get("A_STRONG_MATURE", 0),
                "strong_bootstrap_count": class_counts.get("A_STRONG_BOOTSTRAP", 0),
                "weak_geometry_count": class_counts.get("A_WEAK_GEOMETRY", 0),
                "context_degenerate_count": class_counts.get("A_CONTEXT_DEGENERATE", 0),
                "risk_count": class_counts.get("A_RISK", 0),
                "stress_seq01_count": class_counts.get("A_STRESS_SEQ01", 0),
                "zero_conf_positive_count": zero_positive,
                "feature_q_available_ratio": q_count / len(group) if group else 0.0,
                "feature_k_available_ratio": k_count / len(group) if group else 0.0,
                "raw_coord_available_ratio": raw_count / len(group) if group else 0.0,
            }
        )
    return out


def summarize_feature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = summarize_by_pair(rows)
    all_rows = {
        "scope": "all_rows",
        "seq": "*",
        "prev_chunk": "*",
        "curr_chunk": "*",
        "row_count": len(rows),
        "feature_q_available_count": sum(1 for r in rows if r.get("feature_q_available") is True),
        "feature_k_available_count": sum(1 for r in rows if r.get("feature_k_available") is True),
        "feature_v_available_diagnostic_count": sum(1 for r in rows if r.get("feature_v_available_diagnostic") is True),
        "true_route_available_count": sum(1 for r in rows if r.get("true_route_available") is True),
        "authority_note": "v84 qk compatibility retained as proxy; no true Q/K authority promoted in Phase1",
    }
    rows_out = [all_rows]
    for row in by_pair:
        rows_out.append(
            {
                "scope": "seq_chunk",
                "seq": row["seq"],
                "prev_chunk": row["prev_chunk"],
                "curr_chunk": row["curr_chunk"],
                "row_count": row["anchor_pair_count"],
                "feature_q_available_count": int(round(row["feature_q_available_ratio"] * row["anchor_pair_count"])),
                "feature_k_available_count": int(round(row["feature_k_available_ratio"] * row["anchor_pair_count"])),
                "feature_v_available_diagnostic_count": "",
                "true_route_available_count": 0,
                "authority_note": "proxy_only" if row["feature_q_available_ratio"] == 0 else "true_qk_detected",
            }
        )
    return rows_out


def write_missing_report(path: Path, rows: list[dict[str, Any]], raw_stats: Counter[str], inputs: Mapping[str, Path]) -> None:
    q_true = sum(1 for row in rows if row.get("feature_q_available") is True)
    k_true = sum(1 for row in rows if row.get("feature_k_available") is True)
    raw_ok = sum(1 for row in rows if row.get("raw_coord_available") is True)
    lines = [
        "# Phase1 Missing Artifact Report",
        "",
        "## Inputs",
        "",
    ]
    for name, p in inputs.items():
        lines.append(f"- {name}: `{p}` exists={p.exists()}")
    lines.extend(
        [
            "",
            "## Availability",
            "",
            f"- anchor rows: {len(rows)}",
            f"- raw overlap coordinate rows: {raw_ok}/{len(rows)}",
            f"- true Q feature rows: {q_true}/{len(rows)}",
            f"- true K feature rows: {k_true}/{len(rows)}",
            "- v84 `QK_compatibility` values are recorded as `swa_qk_proxy`; they are not counted as true Q/K features.",
            "- Direct PCA SWA Q/cache-K tensors are counted only when the referenced `.pt` contains `tap::pca_swa_current_q_layers` and `tap::pca_swa_cache_k_layers`.",
            "- same-object/track identity is unavailable in the current row universe and remains blank/false.",
            "- true route mass is unavailable at row level and remains false/blank.",
            "",
            "## Raw Overlap Lookup Stats",
            "",
        ]
    )
    for key, value in sorted(raw_stats.items()):
        lines.append(f"- {key}: {int(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tokens = read_csv(args.tokens)
    if args.max_rows > 0:
        tokens = tokens[: args.max_rows]
    rows, raw_stats = build_rows(tokens, patch_neighborhood_radius=args.patch_neighborhood_radius)
    by_pair = summarize_by_pair(rows)
    feature_rows = summarize_feature(rows)
    low_support = [row for row in by_pair if int(row["anchor_pair_count"]) < 24 and row["quality_label"] != "low_conf_stress"]
    seq01_rows = [row for row in rows if row["anchor_support_class"] == "A_STRESS_SEQ01"]

    out_dir = args.out_dir
    all_fields = REQUIRED_ROW_FIELDS + [
        "pair_id",
        "source_path",
        "source_artifact_paths",
        "swa_usage_source",
        "v84_ruler_role",
        "v84_case_type",
        "patch_y",
        "patch_x",
        "raw_coord_available",
        "raw_lookup_status",
        "feature_source_path",
        "feature_schema",
        "feature_authority_source",
    ]
    write_csv(out_dir / "anchor_pair_rows.csv", rows, fields=all_fields)
    write_csv(out_dir / "anchor_pair_feature_availability.csv", feature_rows)
    write_csv(out_dir / "anchor_pair_by_seq_chunk.csv", by_pair)
    write_csv(out_dir / "low_support_pairs.csv", low_support)
    write_csv(out_dir / "seq01_stress_rows.csv", seq01_rows, fields=all_fields)
    write_missing_report(
        out_dir / "missing_artifact_report.md",
        rows,
        raw_stats,
        {"tokens": args.tokens, "pair_summary": args.pair_summary, "positions": args.positions},
    )
    write_json(
        out_dir / "build_summary.json",
        {
            "row_count": len(rows),
            "seq_count": len({row["seq"] for row in rows}),
            "pair_count": len(by_pair),
            "low_support_pair_count": len(low_support),
            "seq01_stress_row_count": len(seq01_rows),
            "raw_lookup_stats": dict(raw_stats),
            "patch_neighborhood_radius": args.patch_neighborhood_radius,
            "true_q_feature_rows": sum(1 for row in rows if row.get("feature_q_available") is True),
            "true_k_feature_rows": sum(1 for row in rows if row.get("feature_k_available") is True),
            "note": (
                "Phase1 universe built from v84 candidates and raw overlap rows; "
                "QK compatibility remains proxy, but direct PCA SWA Q/cache-K tensors count as feature availability."
            ),
        },
    )
    print(f"anchor_pair_rows={len(rows)}")
    print(f"anchor_pair_by_seq_chunk={len(by_pair)}")
    print(f"low_support_pairs={len(low_support)}")
    print(f"seq01_stress_rows={len(seq01_rows)}")
    print(f"true_q_feature_rows={sum(1 for row in rows if row.get('feature_q_available') is True)}")
    print(f"raw_overlap_patch_match={raw_stats.get('raw_overlap_patch_match', 0)}")


if __name__ == "__main__":
    main()
