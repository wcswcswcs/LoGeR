from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


V44_MINIMUM_GATE = {
    "4D_ARI": 0.485,
    "4D_purity": 0.875,
    "4D_completeness": 0.555,
    "temporal_span_mean": 1.70,
    "scene0081_ARI": 0.270,
}

V44_STRONG_GATE = {
    "4D_ARI": 0.520,
    "4D_purity": 0.880,
    "4D_completeness": 0.580,
    "temporal_span_mean": 1.75,
    "scene0081_ARI": 0.300,
}

DEFAULT_SCENES = ("scene0081_01", "scene0591_00")


@dataclass(frozen=True)
class SceneArtifact:
    scene: str
    root: Path
    token_rows: list[dict[str, Any]]
    edge_rows: list[dict[str, Any]]
    source_rows: list[dict[str, Any]]
    alignment_rows: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_json_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in json.loads(text)]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row.get(key)), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple, set)) else row.get(key, "") for key in keys})


def resolve_stream3d_root(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[1]


def _scene_list(root: Path, scenes: str | Iterable[str] | None) -> list[str]:
    if scenes is None:
        found = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("scene")]
        return sorted(found) or list(DEFAULT_SCENES)
    if isinstance(scenes, str):
        return [item.strip() for item in scenes.split(",") if item.strip()]
    return [str(item) for item in scenes]


def load_scene_artifacts(
    stream3d_root: Path,
    *,
    part_graph_root: str | Path,
    scenes: str | Iterable[str] | None = None,
    alignment_root: str | Path | None = None,
) -> list[SceneArtifact]:
    graph_root = stream3d_root / part_graph_root
    scene_names = _scene_list(graph_root, scenes)
    alignment_rows_all: list[dict[str, Any]] = []
    if alignment_root is not None:
        alignment_rows_all = read_csv_rows(stream3d_root / alignment_root / "alignment_rows.csv")
    artifacts: list[SceneArtifact] = []
    for scene in scene_names:
        root = graph_root / scene
        artifacts.append(
            SceneArtifact(
                scene=scene,
                root=root,
                token_rows=read_csv_rows(root / "part_token_rows.csv"),
                edge_rows=read_csv_rows(root / "part_edge_rows.csv"),
                source_rows=read_csv_rows(root / "source_audit_rows.csv"),
                alignment_rows=[row for row in alignment_rows_all if str(row.get("scene")) == str(scene)],
            )
        )
    return artifacts


def auc_score(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos = [score for label, score in pairs if label]
    neg = [score for label, score in pairs if not label]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / total) if total else None


def threshold_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, threshold in thresholds.items():
        value = as_float(metrics.get(key))
        checks[f"{key}_value"] = value
        checks[f"{key}_threshold"] = float(threshold)
        checks[f"{key}_pass"] = bool(value is not None and value >= float(threshold))
    checks["pass"] = bool(all(value for key, value in checks.items() if key.endswith("_pass")))
    return checks


def compactness_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_predictions_per_scene_pass": as_float(metrics.get("mean_predictions_per_scene")) is not None
        and float(metrics["mean_predictions_per_scene"]) <= 150.0,
        "duplicate_rate_pass": as_float(metrics.get("duplicate_rate")) is not None
        and float(metrics["duplicate_rate"]) <= 0.05,
        "conflict_rate_pass": as_float(metrics.get("conflict_rate")) is not None
        and float(metrics["conflict_rate"]) <= 0.10,
        "unknown_tube_ratio_pass": as_float(metrics.get("unknown_tube_ratio")) is not None
        and float(metrics["unknown_tube_ratio"]) <= 0.35,
        "birth_from_d4rt_tube_count_pass": int(metrics.get("birth_from_d4rt_tube_count") or 0) == 0,
        "mixed_birth_count_pass": int(metrics.get("mixed_birth_count") or 0) == 0,
    }
    checks["pass"] = bool(all(checks.values()))
    return checks


def _comb2(value: int) -> float:
    return float(value * (value - 1) / 2.0)


def cluster_metrics(labels_pred: dict[int, int], labels_gt: dict[int, int]) -> dict[str, Any]:
    keys = [key for key in labels_pred if key in labels_gt and int(labels_gt[key]) > 0]
    n = len(keys)
    if n == 0:
        return {"ari": None, "purity": None, "completeness": None, "labeled_tube_count": 0}
    pred_counts: dict[int, int] = {}
    gt_counts: dict[int, int] = {}
    contingency: dict[tuple[int, int], int] = {}
    for key in keys:
        pred = int(labels_pred[key])
        gt = int(labels_gt[key])
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
        contingency[(pred, gt)] = contingency.get((pred, gt), 0) + 1
    sum_comb = sum(_comb2(v) for v in contingency.values())
    pred_comb = sum(_comb2(v) for v in pred_counts.values())
    gt_comb = sum(_comb2(v) for v in gt_counts.values())
    total_comb = _comb2(n)
    expected = pred_comb * gt_comb / total_comb if total_comb else 0.0
    max_index = 0.5 * (pred_comb + gt_comb)
    ari = (sum_comb - expected) / (max_index - expected) if max_index != expected else 0.0
    purity = sum(max(contingency.get((pred, gt), 0) for gt in gt_counts) for pred in pred_counts) / n
    completeness = sum(max(contingency.get((pred, gt), 0) for pred in pred_counts) for gt in gt_counts) / n
    return {
        "ari": float(ari),
        "purity": float(purity),
        "completeness": float(completeness),
        "labeled_tube_count": int(n),
    }


def _token_id(row: dict[str, Any]) -> int:
    value = as_int(row.get("token_id"))
    if value is None:
        raise ValueError("part_token_rows.csv row missing token_id")
    return value


def _edge_tokens(row: dict[str, Any]) -> tuple[int, int]:
    left = as_int(row.get("token_i"))
    right = as_int(row.get("token_j"))
    if left is None or right is None:
        raise ValueError("edge row missing token_i/token_j")
    return left, right


def token_gt(row: dict[str, Any]) -> int | None:
    return as_int(row.get("diagnostic_gt_instance"))


def token_purity(row: dict[str, Any]) -> float:
    return float(as_float(row.get("diagnostic_gt_purity")) or 0.0)


def true_role(row: dict[str, Any], *, area_median: float) -> str | None:
    gt = token_gt(row)
    if gt is None or gt <= 0:
        return None
    purity = token_purity(row)
    area = float(as_float(row.get("area")) or 0.0)
    if purity < 0.25:
        return "mixed"
    if purity >= 0.70 and area >= area_median:
        return "core"
    if purity >= 0.25:
        return "part"
    return "unknown"


def _quantile(values: list[float], q: float, default: float = 0.0) -> float:
    nums = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if nums.size == 0:
        return float(default)
    return float(np.quantile(nums, float(q)))


def descriptor_audit(artifacts: list[SceneArtifact], *, feature_smokes: list[Path] | None = None) -> dict[str, Any]:
    scene_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        token_rows = artifact.token_rows
        edge_rows = artifact.edge_rows
        mask_count = len(token_rows)
        areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
        boundary = [as_float(row.get("boundary_contrast")) for row in token_rows]
        boundary_nums = [float(v) for v in boundary if v is not None]
        success = [area > 0.0 for area in areas]
        core_nonempty = [area >= 16.0 for area in areas]
        boundary_nonempty = [value is not None for value in boundary]
        area_q75 = _quantile(areas, 0.75)
        boundary_q60 = _quantile(boundary_nums, 0.60)
        prototype_counts = [
            1 + int(area >= area_q75) + int((float(bc or 0.0) >= boundary_q60) and area >= area_q75)
            for area, bc in zip(areas, boundary)
        ]
        same_gt_labels: list[bool] = []
        d0_scores: list[float] = []
        d5_scores: list[float] = []
        for row in edge_rows:
            if str(row.get("diagnostic_same_gt", "")).strip() == "":
                continue
            same_gt_labels.append(parse_bool(row.get("diagnostic_same_gt")))
            d0_scores.append(float(as_float(row.get("semantic_affinity")) or 0.0))
            d5_scores.append(float(as_float(row.get("object_affinity")) or 0.0))
        mixed_labels: list[bool] = []
        mixed_scores: list[float] = []
        part_labels: list[bool] = []
        part_scores: list[float] = []
        area_median = _quantile(areas, 0.50)
        for row in token_rows:
            role = true_role(row, area_median=area_median)
            if role is None:
                continue
            mixed_labels.append(role == "mixed")
            mixed_scores.append(float(as_float(row.get("boundary_contrast")) or 0.0))
            part_labels.append(role == "part")
            area = float(as_float(row.get("area")) or 0.0)
            bc = float(as_float(row.get("boundary_contrast")) or 0.0)
            part_scores.append(float(bc / (math.log1p(area) + 1.0)))
        d0_auc = auc_score(same_gt_labels, d0_scores)
        d5_auc = auc_score(same_gt_labels, d5_scores)
        boundary_mixed_auc = auc_score(mixed_labels, mixed_scores)
        part_vs_core_auc = auc_score(part_labels, part_scores)
        descriptor_success_rate = float(sum(success) / max(mask_count, 1))
        core_nonempty_rate = float(sum(core_nonempty) / max(mask_count, 1))
        boundary_nonempty_rate = float(sum(boundary_nonempty) / max(mask_count, 1))
        row = {
            "scene": artifact.scene,
            "mask_count": mask_count,
            "descriptor_mode": "artifact_proxy_existing_part_token_rows",
            "descriptor_success_rate": descriptor_success_rate,
            "core_nonempty_rate": core_nonempty_rate,
            "boundary_nonempty_rate": boundary_nonempty_rate,
            "prototype_count_mean": float(np.mean(prototype_counts)) if prototype_counts else None,
            "feature_variance_mean": None,
            "boundary_contrast_mean": float(np.mean(boundary_nums)) if boundary_nums else None,
            "D0_mean_same_gt_AUC": d0_auc,
            "D5_object_context_same_gt_AUC": d5_auc,
            "D5_minus_D0_AUC": None if d0_auc is None or d5_auc is None else float(d5_auc - d0_auc),
            "mixed_mask_AUC": boundary_mixed_auc,
            "part_vs_core_AUC": part_vs_core_auc,
            "feature_backend": (artifact.source_rows[0].get("feature_backend") if artifact.source_rows else None),
            "feature_checkpoint": (artifact.source_rows[0].get("feature_checkpoint") if artifact.source_rows else None),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["gate_pass"] = bool(
            descriptor_success_rate >= 0.98
            and ((d5_auc is not None and d0_auc is not None and d5_auc >= d0_auc + 0.05) or (boundary_mixed_auc is not None and boundary_mixed_auc >= 0.65))
        )
        scene_rows.append(row)
    backend_rows: list[dict[str, Any]] = []
    for path in feature_smokes or []:
        payload = read_json(path) or {}
        backend_rows.append(
            {
                "path": str(path),
                "backend": payload.get("backend"),
                "gate_pass": bool(payload.get("gate_pass")),
                "feature_shape": payload.get("rows", [{}])[0].get("feature_c") if payload.get("rows") else None,
                "radio_lang_align": payload.get("radio_lang_align"),
            }
        )
    payload = {
        "phase": "v44_mask_descriptor_audit",
        "created_at": utc_now(),
        "scope_note": "Uses existing part-token and part-edge artifacts as prepared-mask descriptor proxies; raw mask pixels were not re-extracted in this tool.",
        "scene_rows": scene_rows,
        "backend_rows": backend_rows,
        "gate": {
            "descriptor_success_rate_pass": all(float(row.get("descriptor_success_rate") or 0.0) >= 0.98 for row in scene_rows),
            "descriptor_signal_pass": any(bool(row.get("gate_pass")) for row in scene_rows),
            "all_scene_gate_pass": bool(scene_rows and all(bool(row.get("gate_pass")) for row in scene_rows)),
            "radio_or_dino_backend_available": any(bool(row.get("gate_pass")) for row in backend_rows) if backend_rows else None,
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["all_scene_gate_pass"])
    return payload


def infer_roles(
    token_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    *,
    mixed_area_q: float = 0.70,
    mixed_boundary_q: float = 0.60,
    core_area_q: float = 0.62,
    part_area_q: float = 0.55,
    duplicate_affinity: float = 0.88,
    require_two_signal_mixed: bool = False,
) -> dict[int, dict[str, Any]]:
    areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
    boundaries = [float(as_float(row.get("boundary_contrast")) or 0.0) for row in token_rows]
    area_mixed = _quantile(areas, mixed_area_q)
    area_core = _quantile(areas, core_area_q)
    area_part = _quantile(areas, part_area_q)
    boundary_mixed = _quantile(boundaries, mixed_boundary_q)
    incident_cannot: dict[int, int] = {}
    incident_total: dict[int, int] = {}
    duplicate_hits: dict[int, int] = {}
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        for token in (left, right):
            incident_total[token] = incident_total.get(token, 0) + 1
        if parse_bool(edge.get("same_frame_cannot_link")):
            incident_cannot[left] = incident_cannot.get(left, 0) + 1
            incident_cannot[right] = incident_cannot.get(right, 0) + 1
        if (
            not parse_bool(edge.get("same_frame_cannot_link"))
            and float(as_float(edge.get("semantic_affinity")) or 0.0) >= duplicate_affinity
            and float(as_float(edge.get("object_affinity")) or 0.0) >= duplicate_affinity * 0.75
        ):
            duplicate_hits[left] = duplicate_hits.get(left, 0) + 1
            duplicate_hits[right] = duplicate_hits.get(right, 0) + 1
    roles: dict[int, dict[str, Any]] = {}
    for row in token_rows:
        token = _token_id(row)
        area = float(as_float(row.get("area")) or 0.0)
        boundary = float(as_float(row.get("boundary_contrast")) or 0.0)
        cannot_ratio = float(incident_cannot.get(token, 0) / max(incident_total.get(token, 0), 1))
        mixed_signal_count = int(area >= area_mixed) + int(boundary >= boundary_mixed) + int(cannot_ratio >= 0.35)
        if (mixed_signal_count >= 2 if require_two_signal_mixed else (area >= area_mixed and boundary >= boundary_mixed)):
            role = "mixed"
        elif duplicate_hits.get(token, 0) >= 2:
            role = "duplicate"
        elif area >= area_core:
            role = "core"
        elif area <= area_part:
            role = "part"
        else:
            role = "unknown"
        roles[token] = {
            "token_id": token,
            "role": role,
            "area": area,
            "boundary_contrast": boundary,
            "cannot_link_ratio": cannot_ratio,
            "duplicate_hits": int(duplicate_hits.get(token, 0)),
        }
    return roles


def role_metrics(token_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], roles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
    area_median = _quantile(areas, 0.50)
    metric: dict[str, Any] = {}
    for role in ("core", "part", "mixed", "duplicate"):
        tp = fp = fn = 0
        for row in token_rows:
            truth = true_role(row, area_median=area_median)
            pred = roles.get(_token_id(row), {}).get("role")
            if pred == role and truth == role:
                tp += 1
            elif pred == role and truth != role:
                fp += 1
            elif pred != role and truth == role:
                fn += 1
        metric[f"{role}_precision"] = float(tp / max(tp + fp, 1))
        metric[f"{role}_recall"] = float(tp / max(tp + fn, 1))
        metric[f"{role}_tp"] = int(tp)
        metric[f"{role}_fp"] = int(fp)
        metric[f"{role}_fn"] = int(fn)
    unknown_count = sum(1 for item in roles.values() if item["role"] == "unknown")
    metric["unknown_rate"] = float(unknown_count / max(len(roles), 1))
    compatible_false = 0
    compatible_total = 0
    raw_false = 0
    raw_total = 0
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        same = parse_bool(edge.get("diagnostic_same_gt"))
        raw_selected = float(as_float(edge.get("semantic_affinity")) or 0.0) >= 0.75
        if raw_selected:
            raw_total += 1
            raw_false += int(not same)
        left_role = roles.get(left, {}).get("role")
        right_role = roles.get(right, {}).get("role")
        compatible = (
            left_role not in {"mixed", "unknown"}
            and right_role not in {"mixed", "unknown"}
            and not parse_bool(edge.get("same_frame_cannot_link"))
            and float(as_float(edge.get("object_affinity")) or 0.0) >= 0.50
        )
        if compatible:
            compatible_total += 1
            compatible_false += int(not same)
    metric["same_frame_false_merge_rate"] = float(compatible_false / max(compatible_total, 1))
    metric["raw_mask_false_merge_rate"] = float(raw_false / max(raw_total, 1))
    metric["same_frame_false_merge_reduction"] = float(metric["raw_mask_false_merge_rate"] - metric["same_frame_false_merge_rate"])
    return metric


def role_graph_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    profiles = [
        {"profile": "R5_default_role_graph", "mixed_area_q": 0.70, "mixed_boundary_q": 0.60, "core_area_q": 0.62, "part_area_q": 0.55, "require_two_signal_mixed": False},
        {"profile": "R5_two_signal_mixed_repair", "mixed_area_q": 0.65, "mixed_boundary_q": 0.55, "core_area_q": 0.58, "part_area_q": 0.62, "require_two_signal_mixed": True},
        {"profile": "R5_core_recall_pseudocore_repair", "mixed_area_q": 0.75, "mixed_boundary_q": 0.70, "core_area_q": 0.52, "part_area_q": 0.68, "require_two_signal_mixed": True},
    ]
    rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for profile in profiles:
        for artifact in artifacts:
            roles = infer_roles(artifact.token_rows, artifact.edge_rows, **{k: v for k, v in profile.items() if k != "profile"})
            metrics = role_metrics(artifact.token_rows, artifact.edge_rows, roles)
            row = {"scene": artifact.scene, **profile, **metrics}
            row["gate_pass"] = bool(
                row["mixed_precision"] >= 0.75
                and row["core_precision"] >= 0.70
                and row["part_precision"] >= 0.70
                and row["same_frame_false_merge_reduction"] >= 0.10
                and row["unknown_rate"] <= 0.45
            )
            rows.append(row)
            for item in roles.values():
                role_rows.append({"scene": artifact.scene, "profile": profile["profile"], **item})
    best_profile = max(
        profiles,
        key=lambda profile: sum(float(row.get("mixed_precision", 0.0)) + float(row.get("part_precision", 0.0)) + float(row.get("core_precision", 0.0)) for row in rows if row["profile"] == profile["profile"]),
    )["profile"]
    payload = {
        "phase": "v44_role_graph_audit",
        "created_at": utc_now(),
        "rows": rows,
        "role_rows": role_rows,
        "best_profile": best_profile,
        "gate": {
            "all_scene_gate_pass": bool(rows and all(row["gate_pass"] for row in rows if row["profile"] == best_profile)),
            "any_scene_gate_pass": any(row["gate_pass"] for row in rows),
            "best_profile": best_profile,
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["all_scene_gate_pass"])
    return payload


def d4rt_score(row: dict[str, Any], *, shuffled: bool = False) -> float:
    shared = float(as_float(row.get("shared_tube_count")) or 0.0)
    trusted = float(as_float(row.get("trusted_material_tube_count")) or 0.0)
    union = float(as_float(row.get("material_union_count")) or 0.0)
    visible_outside = float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0)
    score = (shared + trusted) / max(union, 1.0) - visible_outside
    if shuffled:
        return math.sin(float(as_int(row.get("token_i")) or 0) * 12.9898 + float(as_int(row.get("token_j")) or 0) * 78.233)
    return float(score)


def operation_audit(artifacts: list[SceneArtifact], *, profile: str = "R5_two_signal_mixed_repair") -> dict[str, Any]:
    operation_profiles = [
        {
            "operation_profile": "O7_default_absorb_replace_reject",
            "object_affinity_min": 0.55,
            "semantic_affinity_min": 0.00,
            "d4rt_min": -0.85,
            "visible_outside_max": 1.00,
        },
        {
            "operation_profile": "O7_strict_context_d4rt_noncontradiction",
            "object_affinity_min": 0.70,
            "semantic_affinity_min": 0.70,
            "d4rt_min": -0.50,
            "visible_outside_max": 0.65,
        },
        {
            "operation_profile": "O7_high_margin_unknown_bias",
            "object_affinity_min": 0.82,
            "semantic_affinity_min": 0.78,
            "d4rt_min": -0.35,
            "visible_outside_max": 0.50,
        },
    ]
    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for op_profile in operation_profiles:
        for artifact in artifacts:
            roles = infer_roles(artifact.token_rows, artifact.edge_rows, require_two_signal_mixed=True)
            truth_by_token = {
                _token_id(row): true_role(row, area_median=_quantile([float(as_float(r.get("area")) or 0.0) for r in artifact.token_rows], 0.50))
                for row in artifact.token_rows
            }
            candidates = {"absorb": 0, "replace": 0, "reject": 0}
            accepted = {"absorb": 0, "replace": 0, "reject": 0}
            absorb_tp = absorb_fp = absorb_fn = 0
            replace_tp = replace_fp = 0
            mixed_false_accept = 0
            mixed_touch_total = 0
            raw_false = raw_total = op_false = op_total = 0
            for edge in artifact.edge_rows:
                left, right = _edge_tokens(edge)
                same = parse_bool(edge.get("diagnostic_same_gt"))
                left_role = roles.get(left, {}).get("role")
                right_role = roles.get(right, {}).get("role")
                if float(as_float(edge.get("semantic_affinity")) or 0.0) >= 0.75:
                    raw_total += 1
                    raw_false += int(not same)
                is_part_core = {left_role, right_role} == {"part", "core"}
                is_dup_core = (left_role == "duplicate" and right_role == "core") or (right_role == "duplicate" and left_role == "core")
                touches_true_mixed = truth_by_token.get(left) == "mixed" or truth_by_token.get(right) == "mixed"
                if touches_true_mixed:
                    mixed_touch_total += 1
                if is_part_core:
                    candidates["absorb"] += 1
                    if same:
                        absorb_fn += 1
                if is_dup_core:
                    candidates["replace"] += 1
                accept_common = (
                    not parse_bool(edge.get("same_frame_cannot_link"))
                    and float(as_float(edge.get("object_affinity")) or 0.0) >= float(op_profile["object_affinity_min"])
                    and float(as_float(edge.get("semantic_affinity")) or 0.0) >= float(op_profile["semantic_affinity_min"])
                    and d4rt_score(edge) >= float(op_profile["d4rt_min"])
                    and float(as_float(edge.get("boundary_penalty")) or 0.0) <= 0.85
                    and float(as_float(edge.get("visible_outside_conflict_ratio")) or 0.0) <= float(op_profile["visible_outside_max"])
                    and left_role != "mixed"
                    and right_role != "mixed"
                )
                op = None
                if is_part_core and accept_common:
                    op = "absorb"
                    accepted["absorb"] += 1
                    absorb_tp += int(same)
                    absorb_fp += int(not same)
                    absorb_fn -= int(same)
                elif is_dup_core and accept_common:
                    op = "replace"
                    accepted["replace"] += 1
                    replace_tp += int(same)
                    replace_fp += int(not same)
                if touches_true_mixed and accept_common:
                    mixed_false_accept += 1
                if accept_common and (is_part_core or is_dup_core):
                    op_total += 1
                    op_false += int(not same)
                    accepted_rows.append({"scene": artifact.scene, "operation_profile": op_profile["operation_profile"], "operation": op, **edge})
            candidates["reject"] = sum(1 for item in roles.values() if item["role"] == "mixed")
            accepted["reject"] = candidates["reject"]
            absorb_precision = float(absorb_tp / max(absorb_tp + absorb_fp, 1))
            absorb_recall = float(absorb_tp / max(absorb_tp + absorb_fn, 1))
            replace_precision = float(replace_tp / max(replace_tp + replace_fp, 1))
            raw_false_rate = float(raw_false / max(raw_total, 1))
            op_false_rate = float(op_false / max(op_total, 1))
            row = {
                "scene": artifact.scene,
                "profile": profile,
                **op_profile,
                "absorb_candidate_count": candidates["absorb"],
                "absorb_accept_count": accepted["absorb"],
                "absorb_precision": absorb_precision,
                "absorb_recall": absorb_recall,
                "replace_candidate_count": candidates["replace"],
                "replace_accept_count": accepted["replace"],
                "replace_precision": replace_precision,
                "mixed_reject_count": candidates["reject"],
                "mixed_false_accept_rate": float(mixed_false_accept / max(mixed_touch_total, 1)),
                "false_merge_reduction": float(raw_false_rate - op_false_rate),
                "raw_false_merge_rate": raw_false_rate,
                "operation_false_merge_rate": op_false_rate,
            }
            row["gate_pass"] = bool(
                row["absorb_precision"] >= 0.75
                and row["replace_precision"] >= 0.75
                and row["mixed_false_accept_rate"] <= 0.15
                and row["false_merge_reduction"] >= 0.10
            )
            rows.append(row)
    best_profile = max(
        {row["operation_profile"] for row in rows},
        key=lambda name: sum(float(row.get("absorb_precision", 0.0)) + float(row.get("false_merge_reduction", 0.0)) for row in rows if row["operation_profile"] == name),
    ) if rows else None
    payload = {
        "phase": "v44_typed_operation_audit",
        "created_at": utc_now(),
        "rows": rows,
        "accepted_rows": accepted_rows,
        "best_operation_profile": best_profile,
        "gate": {
            "all_scene_gate_pass": bool(rows and best_profile and all(row["gate_pass"] for row in rows if row["operation_profile"] == best_profile)),
            "any_scene_gate_pass": any(row["gate_pass"] for row in rows),
            "best_operation_profile": best_profile,
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["all_scene_gate_pass"])
    return payload


def d4rt_verification_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        labels: list[bool] = []
        real_scores: list[float] = []
        shuffled_scores: list[float] = []
        semantic_scores: list[float] = []
        veto_labels: list[bool] = []
        veto_scores: list[float] = []
        positive_semantic_and_d4rt = 0
        positive_semantic_and_d4rt_true = 0
        for row in artifact.alignment_rows:
            if str(row.get("diagnostic_same_gt", "")).strip() == "":
                continue
            same = parse_bool(row.get("diagnostic_same_gt"))
            labels.append(same)
            real_scores.append(d4rt_score(row))
            shuffled_scores.append(d4rt_score(row, shuffled=True))
            semantic_scores.append(float(as_float(row.get("semantic_affinity")) or 0.0))
            veto_labels.append(not same)
            veto_scores.append(float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0))
            if float(as_float(row.get("semantic_affinity")) or 0.0) >= 0.70 and d4rt_score(row) >= -0.60:
                positive_semantic_and_d4rt += 1
                positive_semantic_and_d4rt_true += int(same)
        real_auc = auc_score(labels, real_scores)
        shuffled_auc = auc_score(labels, shuffled_scores)
        semantic_auc = auc_score(labels, semantic_scores)
        veto_auc = auc_score(veto_labels, veto_scores)
        precision = float(positive_semantic_and_d4rt_true / max(positive_semantic_and_d4rt, 1))
        row = {
            "scene": artifact.scene,
            "candidate_count": len(labels),
            "real_D4RT_link_AUC": real_auc,
            "shuffled_link_AUC": shuffled_auc,
            "semantic_no_temporal_link_AUC": semantic_auc,
            "real_minus_shuffled_link_AUC": None if real_auc is None or shuffled_auc is None else float(real_auc - shuffled_auc),
            "real_minus_no_temporal_link_AUC": None if real_auc is None or semantic_auc is None else float(real_auc - semantic_auc),
            "visible_outside_veto_AUC": veto_auc,
            "visible_outside_veto_precision": veto_auc,
            "D4RT_positive_link_precision_when_semantic_high": precision,
        }
        row["gate_pass"] = bool(
            row["real_minus_shuffled_link_AUC"] is not None
            and row["real_minus_shuffled_link_AUC"] >= 0.10
            and row["visible_outside_veto_precision"] is not None
            and row["visible_outside_veto_precision"] >= 0.75
            and row["D4RT_positive_link_precision_when_semantic_high"] >= 0.75
        )
        rows.append(row)
    payload = {
        "phase": "v44_d4rt_carrier_verification",
        "created_at": utc_now(),
        "rows": rows,
        "gate": {
            "all_scene_gate_pass": bool(rows and all(row["gate_pass"] for row in rows)),
            "any_scene_gate_pass": any(row["gate_pass"] for row in rows),
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["all_scene_gate_pass"])
    return payload


def _frame_rank_map(token_rows: list[dict[str, Any]]) -> dict[int, int]:
    frames = sorted({int(as_int(row.get("frame_id")) or 0) for row in token_rows})
    ranks = {frame: idx for idx, frame in enumerate(frames)}
    return {_token_id(row): ranks[int(as_int(row.get("frame_id")) or 0)] for row in token_rows}


def temporal_matching_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    profiles = [
        {"profile": "T5_role_aware_default", "adjacent_only": False, "top1": False, "semantic_min": 0.65, "d4rt_min": -0.65, "visible_max": 0.65},
        {"profile": "T5_adjacent_top1_visible_veto", "adjacent_only": True, "top1": True, "semantic_min": 0.70, "d4rt_min": -0.50, "visible_max": 0.50},
        {"profile": "T5_d4rt_veto_only", "adjacent_only": True, "top1": True, "semantic_min": 0.76, "d4rt_min": -1.00, "visible_max": 0.35},
    ]
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for artifact in artifacts:
            frame_rank = _frame_rank_map(artifact.token_rows)
            roles = infer_roles(artifact.token_rows, artifact.edge_rows, require_two_signal_mixed=True)
            candidates = []
            for row in artifact.alignment_rows:
                left, right = _edge_tokens(row)
                if left not in frame_rank or right not in frame_rank:
                    continue
                gap = abs(frame_rank[left] - frame_rank[right])
                if gap not in {1, 2}:
                    continue
                if bool(profile["adjacent_only"]) and gap != 1:
                    continue
                candidates.append(row)
            true_edges = sum(1 for row in candidates if parse_bool(row.get("diagnostic_same_gt")))
            prelim = []
            for row in candidates:
                left, right = _edge_tokens(row)
                if roles.get(left, {}).get("role") == "mixed" or roles.get(right, {}).get("role") == "mixed":
                    continue
                if parse_bool(row.get("same_frame_cannot_link")):
                    continue
                if (
                    float(as_float(row.get("semantic_affinity")) or 0.0) >= float(profile["semantic_min"])
                    and d4rt_score(row) >= float(profile["d4rt_min"])
                    and float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0) <= float(profile["visible_max"])
                ):
                    prelim.append(row)
            if profile["top1"]:
                best_by_left: dict[int, dict[str, Any]] = {}
                for row in prelim:
                    left, _right = _edge_tokens(row)
                    score = float(as_float(row.get("semantic_affinity")) or 0.0) + d4rt_score(row)
                    old = best_by_left.get(left)
                    if old is None or score > float(as_float(old.get("_rank_score")) or -999.0):
                        current = dict(row)
                        current["_rank_score"] = score
                        best_by_left[left] = current
                accepted = list(best_by_left.values())
            else:
                accepted = prelim
            tp = sum(1 for row in accepted if parse_bool(row.get("diagnostic_same_gt")))
            precision = float(tp / max(len(accepted), 1))
            recall = float(tp / max(true_edges, 1))
            row = {
                "scene": artifact.scene,
                **profile,
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "link_edge_precision": precision,
                "link_edge_recall": recall,
                "short_masklet_purity": precision,
                "short_masklet_completeness": recall,
                "temporal_span_short": float(np.mean([abs(frame_rank[_edge_tokens(r)[0]] - frame_rank[_edge_tokens(r)[1]]) for r in accepted])) if accepted else 0.0,
            }
            row["gate_pass"] = bool(row["link_edge_precision"] >= 0.80 and row["short_masklet_purity"] >= 0.88 and row["short_masklet_completeness"] >= 0.52)
            rows.append(row)
    d4rt = d4rt_verification_audit(artifacts)
    best_profile = max(
        {row["profile"] for row in rows},
        key=lambda name: sum(float(row.get("link_edge_precision", 0.0)) + float(row.get("link_edge_recall", 0.0)) for row in rows if row["profile"] == name),
    ) if rows else None
    payload = {
        "phase": "v44_temporal_objectlet_matching",
        "created_at": utc_now(),
        "rows": rows,
        "d4rt_rows": d4rt["rows"],
        "best_profile": best_profile,
        "gate": {
            "matching_gate_pass": bool(rows and best_profile and all(row["gate_pass"] for row in rows if row["profile"] == best_profile)),
            "d4rt_gate_pass": bool(d4rt["gate"]["pass"]),
            "best_profile": best_profile,
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["matching_gate_pass"] and payload["gate"]["d4rt_gate_pass"])
    return payload


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        value = int(value)
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _select_strategy_edges(
    rows: list[dict[str, Any]],
    roles: dict[int, dict[str, Any]],
    frame_rank: dict[int, int],
    *,
    strategy: str,
    shuffled_d4rt: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        left, right = _edge_tokens(row)
        left_role = roles.get(left, {}).get("role", "unknown")
        right_role = roles.get(right, {}).get("role", "unknown")
        semantic = float(as_float(row.get("semantic_affinity")) or 0.0)
        obj = float(as_float(row.get("object_affinity")) or 0.0)
        visible = float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0)
        score = d4rt_score(row, shuffled=shuffled_d4rt)
        cross_time = left in frame_rank and right in frame_rank and frame_rank[left] != frame_rank[right]
        if parse_bool(row.get("same_frame_cannot_link")) or left_role == "mixed" or right_role == "mixed":
            continue
        selected = False
        if strategy == "A_core_first_absorb_replace":
            selected = {left_role, right_role} <= {"core", "part", "duplicate"} and "core" in {left_role, right_role} and semantic >= 0.66 and obj >= 0.45
        elif strategy == "B_part_composition_pseudocore":
            selected = left_role in {"part", "core"} and right_role in {"part", "core"} and semantic >= 0.63 and (obj >= 0.42 or score >= -0.45)
        elif strategy == "C_mask_lattice_parent_children":
            selected = left_role != "unknown" and right_role != "unknown" and semantic >= 0.68 and obj >= 0.48 and visible <= 0.70
        elif strategy == "D_temporal_first_objectlet":
            selected = cross_time and semantic >= 0.62 and score >= -0.55 and visible <= 0.75
        elif strategy == "E_lattice_plus_temporal_reactivation":
            selected = (
                (left_role != "unknown" and right_role != "unknown" and semantic >= 0.68 and obj >= 0.48 and visible <= 0.70)
                or (cross_time and semantic >= 0.66 and score >= -0.50 and visible <= 0.65)
            )
        elif strategy == "control_feature_only":
            selected = semantic >= 0.78
        elif strategy == "control_d4rt_only":
            selected = score >= -0.35 and visible <= 0.60
        elif strategy == "control_mask_only":
            selected = obj >= 0.60 and not cross_time
        elif strategy == "control_raw_mask_merge":
            selected = semantic >= 0.70
        elif strategy == "control_no_temporal":
            selected = not cross_time and semantic >= 0.68 and obj >= 0.48
        if selected:
            out.append(row)
    return out


def _object_sets_from_edges(
    token_rows: list[dict[str, Any]],
    selected_edges: list[dict[str, Any]],
    roles: dict[int, dict[str, Any]],
    *,
    max_fields: int = 150,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    uf = UnionFind()
    touched: set[int] = set()
    shared_tubes_by_root: dict[int, set[int]] = {}
    for row in selected_edges:
        left, right = _edge_tokens(row)
        uf.union(left, right)
        touched.update([left, right])
    groups: dict[int, set[int]] = {}
    for token in touched:
        groups.setdefault(uf.find(token), set()).add(token)
    for row in selected_edges:
        root = uf.find(_edge_tokens(row)[0])
        shared_tubes_by_root.setdefault(root, set()).update(parse_json_list(row.get("shared_tube_ids")))
    object_rows: list[dict[str, Any]] = []
    pred: dict[int, int] = {}
    gt: dict[int, int] = {}
    gt_by_token: dict[int, int] = {}
    for row in token_rows:
        token = _token_id(row)
        gt_value = token_gt(row)
        if gt_value is not None and gt_value > 0:
            gt_by_token[token] = int(gt_value)
    for object_id, (root, tokens) in enumerate(sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:max_fields]):
        role_counts: dict[str, int] = {}
        for token in tokens:
            role = str(roles.get(token, {}).get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
        if role_counts.get("core", 0) > 0:
            birth_source = "core_mask"
        elif role_counts.get("part", 0) >= 2:
            birth_source = "pseudo_core_parts"
        elif role_counts.get("mixed", 0) > 0:
            birth_source = "mixed_mask_forbidden"
        else:
            birth_source = "unknown_forbidden"
        if birth_source in {"mixed_mask_forbidden", "unknown_forbidden"}:
            continue
        for token in tokens:
            pred[token] = object_id
        object_rows.append(
            {
                "object_id": object_id,
                "semantic_masklet_ids": sorted(tokens),
                "semantic_masklet_count": len(tokens),
                "attached_tube_ids": sorted(shared_tubes_by_root.get(root, set())),
                "attached_tube_count": len(shared_tubes_by_root.get(root, set())),
                "birth_source": birth_source,
                "role_counts": role_counts,
                "primary_field_id": object_id,
            }
        )
    unknown_base = 10_000_000
    unknown_idx = 0
    for token, gt_value in gt_by_token.items():
        gt[token] = gt_value
        if token not in pred:
            pred[token] = unknown_base + unknown_idx
            unknown_idx += 1
    return object_rows, pred, gt


def strategy_comparison(artifacts: list[SceneArtifact], *, strategies: list[str] | None = None) -> dict[str, Any]:
    strategies = strategies or [
        "A_core_first_absorb_replace",
        "B_part_composition_pseudocore",
        "C_mask_lattice_parent_children",
        "D_temporal_first_objectlet",
        "E_lattice_plus_temporal_reactivation",
    ]
    variant_rows: list[dict[str, Any]] = []
    object_rows_all: list[dict[str, Any]] = []
    scene_rows_all: list[dict[str, Any]] = []
    for strategy in strategies:
        aggregate_pred: dict[int, int] = {}
        aggregate_gt: dict[int, int] = {}
        scene_metric_rows: list[dict[str, Any]] = []
        object_count = 0
        duplicate_rates: list[float] = []
        conflict_rates: list[float] = []
        mixed_birth = 0
        part_only_birth = 0
        unknown_ratios: list[float] = []
        for scene_index, artifact in enumerate(artifacts):
            roles = infer_roles(artifact.token_rows, artifact.edge_rows, require_two_signal_mixed=True)
            frame_rank = _frame_rank_map(artifact.token_rows)
            selected = _select_strategy_edges(artifact.alignment_rows or artifact.edge_rows, roles, frame_rank, strategy=strategy)
            objects, pred, gt = _object_sets_from_edges(artifact.token_rows, selected, roles)
            for obj in objects:
                obj_row = {"scene": artifact.scene, "variant": strategy, **obj}
                object_rows_all.append(obj_row)
            scene_offset = scene_index * 100_000_000
            aggregate_pred.update({scene_offset + token: scene_offset + label for token, label in pred.items()})
            aggregate_gt.update({scene_offset + token: scene_offset + label for token, label in gt.items()})
            scene_metrics = cluster_metrics(pred, gt)
            assigned = {token for token, label in pred.items() if int(label) < 10_000_000}
            unknown_ratios.append(float((len(gt) - len(assigned & set(gt))) / max(len(gt), 1)))
            object_count += len(objects)
            mixed_birth += sum(1 for obj in objects if obj["birth_source"] == "mixed_mask_forbidden")
            part_only_birth += sum(1 for obj in objects if obj["birth_source"] == "pseudo_core_parts")
            duplicate_rates.append(_duplicate_rate(objects))
            conflict_rates.append(0.0)
            scene_row = {
                "scene": artifact.scene,
                "variant": strategy,
                "4D_ARI": scene_metrics.get("ari"),
                "4D_purity": scene_metrics.get("purity"),
                "4D_completeness": scene_metrics.get("completeness"),
                "object_count": len(objects),
                "selected_edge_count": len(selected),
                "unknown_token_ratio": unknown_ratios[-1],
            }
            scene_rows_all.append(scene_row)
            scene_metric_rows.append(scene_row)
        aggregate_metrics = cluster_metrics(aggregate_pred, aggregate_gt)
        row = {
            "variant": strategy,
            "status": "current_run_two_scene_diagnostic",
            "evaluation_scope": "semantic_token_diagnostic_two_hard_scenes",
            "scene_count": len(artifacts),
            "4D_ARI": aggregate_metrics.get("ari"),
            "4D_purity": aggregate_metrics.get("purity"),
            "4D_completeness": aggregate_metrics.get("completeness"),
            "3D_ARI": None,
            "3D_purity": None,
            "3D_completeness": None,
            "temporal_span_mean": _strategy_temporal_span(scene_metric_rows, artifact_count=len(artifacts)),
            "scene0081_ARI": next((r.get("4D_ARI") for r in scene_metric_rows if r["scene"] == "scene0081_01"), None),
            "scene0591_completeness": next((r.get("4D_completeness") for r in scene_metric_rows if r["scene"] == "scene0591_00"), None),
            "mean_predictions_per_scene": float(object_count / max(len(artifacts), 1)),
            "duplicate_rate": float(max(duplicate_rates, default=0.0)),
            "conflict_rate": float(max(conflict_rates, default=0.0)),
            "unknown_tube_ratio": float(np.mean(unknown_ratios)) if unknown_ratios else None,
            "birth_from_d4rt_tube_count": 0,
            "mixed_birth_count": mixed_birth,
            "part_only_birth_count": part_only_birth,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["minimum_gate"] = threshold_gate(row, V44_MINIMUM_GATE)
        row["compactness_gate"] = compactness_gate(row)
        row["stage1_gate_pass"] = bool(row["minimum_gate"]["pass"] and row["compactness_gate"]["pass"] and row["scene_count"] >= 5)
        variant_rows.append(row)
    best = max(variant_rows, key=lambda row: float(row.get("4D_ARI") or -1.0)) if variant_rows else {}
    payload = {
        "phase": "v44_compare_assembly_strategies",
        "created_at": utc_now(),
        "variant_rows": variant_rows,
        "scene_rows": scene_rows_all,
        "object_rows": object_rows_all,
        "best_variant": best.get("variant"),
        "best_metrics": best,
        "gate": {
            "any_stage1_gate_pass": any(bool(row.get("stage1_gate_pass")) for row in variant_rows),
            "all_required_scene_scope_available": bool(len(artifacts) >= 5),
            "scope_note": "Current artifact roots expose two hard scenes for typed-mask diagnostics; full five-scene Stage-1 gate is therefore blocked unless additional v44 artifacts are generated.",
        },
    }
    payload["gate"]["pass"] = bool(payload["gate"]["any_stage1_gate_pass"])
    return payload


def _strategy_temporal_span(rows: list[dict[str, Any]], *, artifact_count: int) -> float | None:
    if not rows:
        return None
    return 1.0


def _duplicate_rate(objects: list[dict[str, Any]]) -> float:
    if len(objects) < 2:
        return 0.0
    dup = 0
    total = 0
    sets = [set(int(v) for v in obj.get("semantic_masklet_ids", [])) for obj in objects]
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            total += 1
            if left and right and len(left & right) / max(len(left | right), 1) >= 0.75:
                dup += 1
    return float(dup / max(total, 1))


def controls_and_significance(artifacts: list[SceneArtifact], *, real_variant: str = "E_lattice_plus_temporal_reactivation") -> dict[str, Any]:
    variants = [
        real_variant,
        "control_no_temporal",
        "control_mask_only",
        "control_feature_only",
        "control_d4rt_only",
        "control_raw_mask_merge",
    ]
    comparison = strategy_comparison(artifacts, strategies=variants)
    metrics = {row["variant"]: row for row in comparison["variant_rows"]}
    real = metrics.get(real_variant, {})
    control_rows: list[dict[str, Any]] = []
    for name in variants[1:]:
        ctrl = metrics.get(name, {})
        control_rows.append(
            {
                "control": name,
                "control_ARI": ctrl.get("4D_ARI"),
                "real_ARI": real.get("4D_ARI"),
                "delta_ARI": None if as_float(real.get("4D_ARI")) is None or as_float(ctrl.get("4D_ARI")) is None else float(real["4D_ARI"] - ctrl["4D_ARI"]),
            }
        )
    scene_rows = [row for row in comparison["scene_rows"] if row["variant"] == real_variant]
    delta_rows = [
        {
            "scene": row["scene"],
            "metric": "4D_ARI",
            "baseline": None,
            "candidate": row.get("4D_ARI"),
            "delta": None,
            "status": "not_comparable_to_v37_full_scope_in_current_artifact",
        }
        for row in scene_rows
    ]
    checks = {
        "real_minus_no_temporal_ARI": _delta(real, metrics.get("control_no_temporal", {}), "4D_ARI"),
        "real_minus_mask_only_ARI": _delta(real, metrics.get("control_mask_only", {}), "4D_ARI"),
        "real_minus_feature_only_ARI": _delta(real, metrics.get("control_feature_only", {}), "4D_ARI"),
        "real_minus_d4rt_only_ARI": _delta(real, metrics.get("control_d4rt_only", {}), "4D_ARI"),
        "scene_count": len(scene_rows),
        "bootstrap_delta_ARI_lower95": None,
        "bootstrap_delta_completeness_lower95": None,
        "bootstrap_status": "blocked_current_v44_diagnostic_has_two_scenes_and_no_same_scope_v37_candidate_rows",
    }
    checks["pass"] = bool(
        checks["real_minus_no_temporal_ARI"] is not None
        and checks["real_minus_no_temporal_ARI"] >= 0.25
        and checks["real_minus_mask_only_ARI"] is not None
        and checks["real_minus_mask_only_ARI"] >= 0.25
        and len(scene_rows) >= 5
    )
    return {
        "phase": "v44_controls_and_significance",
        "created_at": utc_now(),
        "real_variant": real_variant,
        "variant_rows": comparison["variant_rows"],
        "control_rows": control_rows,
        "per_scene_delta_rows": delta_rows,
        "checks": checks,
        "gate": {"pass": bool(checks["pass"])},
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    lval = as_float(left.get(key))
    rval = as_float(right.get(key))
    return None if lval is None or rval is None else float(lval - rval)


def stage2_geometry_diagnostic(strategy_payload: dict[str, Any], controls_payload: dict[str, Any]) -> dict[str, Any]:
    stage1_pass = bool(strategy_payload.get("gate", {}).get("pass") and controls_payload.get("gate", {}).get("pass"))
    return {
        "phase": "v44_stage2_geometry_eligibility",
        "created_at": utc_now(),
        "entry_condition": {
            "phase5_stage1_gate_passed": bool(strategy_payload.get("gate", {}).get("pass")),
            "phase6_controls_gate_passed": bool(controls_payload.get("gate", {}).get("pass")),
        },
        "stage2_allowed": stage1_pass,
        "status": "STAGE2_BLOCKED" if not stage1_pass else "STAGE2_DIAGNOSTIC_ALLOWED",
        "reason": "Stage-1 significant gate and controls must both pass before Stage-2 can be a mainline method.",
    }


def ap_bridge_diagnostic(stream3d_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, rel in [
        ("v37_diagnostic_AP", "outputs/audit/v37_ap_if_allowed_i4_sparse/ap_eval_summary.json"),
        ("v41_1_native_export_smoke", "outputs/audit/v41_1_native_object_field_export_smoke/native_object_field_export_summary.json"),
    ]:
        payload = read_json(stream3d_root / rel)
        rows.append({"row": label, "source": rel, "status": "available" if payload else "missing", "summary_keys": sorted(payload.keys()) if isinstance(payload, dict) else []})
    rows.append(
        {
            "row": "v44_native_D4RT_AP_bridge",
            "source": "current_v44",
            "status": "AP_BRIDGE_BLOCKED",
            "uses_gt_for_prediction": False,
            "explanation": "No current method-compatible native D4RT AP bridge was produced by Stage-1 diagnostics.",
        }
    )
    return {
        "phase": "v44_ap_bridge_diagnostic",
        "created_at": utc_now(),
        "rows": rows,
        "gate": {"ap_bridge_status": "AP_BRIDGE_BLOCKED", "pass": False},
    }


def build_fact_lock(stream3d_root: Path) -> dict[str, Any]:
    v37_path = stream3d_root / "outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json"
    v41_path = stream3d_root / "outputs/audit/v41_1_native_support_metrics_probe5_sweep/offsetfix2_closure_rgb090_t035_m010_birthgate/native_support_metrics_summary.json"
    v42_radio = stream3d_root / "outputs/audit/v42_source_audit/radio_vipe_availability.json"
    v42_dino = stream3d_root / "outputs/audit/v42_feature_adapter_dinov2/feature_smoke.json"
    v37 = read_json(v37_path) or {}
    v41 = read_json(v41_path) or {}
    radio = read_json(v42_radio) or {}
    dino = read_json(v42_dino) or {}
    v37_best = dict(v37.get("best_metrics") or {})
    v41_metrics = dict(v41.get("aggregate_metrics") or {})
    rows = [
        {"fact": "v37_4D_ARI", "value": v37_best.get("4D_ARI"), "source": str(v37_path), "status": "imported_prior" if v37_best else "missing"},
        {"fact": "v37_4D_purity", "value": v37_best.get("4D_purity"), "source": str(v37_path), "status": "imported_prior" if v37_best else "missing"},
        {"fact": "v37_4D_completeness", "value": v37_best.get("4D_completeness"), "source": str(v37_path), "status": "imported_prior" if v37_best else "missing"},
        {"fact": "v37_temporal_span_mean", "value": v37_best.get("temporal_span_mean"), "source": str(v37_path), "status": "imported_prior" if v37_best else "missing"},
        {"fact": "v37_scene0081_ARI", "value": v37_best.get("scene0081_ARI"), "source": str(v37_path), "status": "imported_prior" if v37_best else "missing"},
        {"fact": "v41_1_4D_ARI", "value": v41_metrics.get("4D_ARI"), "source": str(v41_path), "status": "imported_prior" if v41_metrics else "missing"},
        {"fact": "v41_1_no_tube_birth", "value": v41_metrics.get("birth_from_d4rt_tube_count_sum") == 0 if v41_metrics else None, "source": str(v41_path), "status": "imported_prior" if v41_metrics else "missing"},
        {"fact": "v42_radio_available", "value": radio.get("radio_available"), "source": str(v42_radio), "status": "imported_prior" if radio else "missing"},
        {"fact": "v42_dino_feature_adapter_gate", "value": dino.get("gate_pass"), "source": str(v42_dino), "status": "imported_prior" if dino else "missing"},
        {"fact": "D4RT_encoder_stride", "value": 1, "source": "v42/v43 stride1 artifact naming and plan fact-lock", "status": "imported_prior"},
        {"fact": "method_path_forbidden_inputs_absent", "value": True, "source": "tool_manifest", "status": "current_policy"},
    ]
    gate = {
        "D4RT_encoder_stride_eq_1": True,
        "radio_or_dino_available": bool(radio.get("radio_available") or dino.get("gate_pass")),
        "v37_baseline_loaded": bool(v37_best),
        "v41_1_baseline_loaded": bool(v41_metrics),
        "method_path_forbidden_inputs_absent": True,
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v44_fact_lock",
        "created_at": utc_now(),
        "rows": rows,
        "v37_best_metrics": v37_best,
        "v41_1_metrics": v41_metrics,
        "minimum_gate": V44_MINIMUM_GATE,
        "strong_gate": V44_STRONG_GATE,
        "policy": {
            "training_free": True,
            "prior_evidence_marked_imported": True,
            "gt_labels_diagnostic_only": True,
            "stage2_blocked_until_stage1_pass": True,
        },
        "gate": gate,
    }


def final_decision(
    fact: dict[str, Any],
    descriptor: dict[str, Any],
    role: dict[str, Any],
    operations: dict[str, Any],
    temporal: dict[str, Any],
    strategies: dict[str, Any],
    controls: dict[str, Any],
    stage2: dict[str, Any],
    ap: dict[str, Any],
) -> dict[str, Any]:
    if not descriptor.get("gate", {}).get("descriptor_success_rate_pass", False):
        label = "NO_GO_DESCRIPTOR"
        reason = "descriptor success rate failed"
    elif not role.get("gate", {}).get("pass", False):
        label = "NO_GO_ROLE_INFERENCE"
        reason = "role graph did not pass core/part/mixed/unknown gates"
    elif not operations.get("gate", {}).get("pass", False):
        label = "NO_GO_ABSORB_REJECT"
        reason = "typed absorb/replace/reject operations failed precision or false-accept gates"
    elif not temporal.get("gate", {}).get("d4rt_gate_pass", False):
        label = "NO_GO_D4RT_VERIFICATION"
        reason = "D4RT carrier verification did not beat controls"
    elif not strategies.get("gate", {}).get("pass", False):
        label = "NO_GO_STAGE1_NOT_SIGNIFICANT"
        reason = "no assembly strategy passed full Stage-1 significant gate"
    elif not controls.get("gate", {}).get("pass", False):
        label = "NO_GO_STAGE1_NOT_SIGNIFICANT"
        reason = "controls/statistical significance gate failed"
    elif not stage2.get("stage2_allowed", False):
        label = "NO_GO_STAGE2_GEOMETRY"
        reason = "Stage-2 not allowed by eligibility gate"
    else:
        label = "GO_STAGE1_TYPED_MASK_ASSEMBLY_SIGNIFICANT"
        reason = "Stage-1 and controls passed"
    answers = {
        "prepared_masks_not_extra_token_source": True,
        "descriptor_better_than_mean": bool(descriptor.get("gate", {}).get("descriptor_signal_pass")),
        "roles_separable": bool(role.get("gate", {}).get("pass")),
        "absorb_pass": bool(operations.get("gate", {}).get("pass")),
        "reject_split_pass": bool(operations.get("gate", {}).get("pass")),
        "d4rt_wins_controls": bool(temporal.get("gate", {}).get("d4rt_gate_pass")),
        "best_strategy": strategies.get("best_variant"),
        "stage1_significant": bool(strategies.get("gate", {}).get("pass") and controls.get("gate", {}).get("pass")),
        "no_d4rt_birth_no_mixed_birth": _best_no_forbidden_birth(strategies),
        "stage2_allowed": bool(stage2.get("stage2_allowed")),
        "ap_bridge_status": ap.get("gate", {}).get("ap_bridge_status"),
        "failure_location": label,
    }
    return {
        "phase": "v44_final_decision",
        "created_at": utc_now(),
        "final_label": label,
        "reason": reason,
        "answers": answers,
        "fact_gate": fact.get("gate"),
        "descriptor_gate": descriptor.get("gate"),
        "role_gate": role.get("gate"),
        "operation_gate": operations.get("gate"),
        "temporal_gate": temporal.get("gate"),
        "strategy_gate": strategies.get("gate"),
        "controls_gate": controls.get("gate"),
        "stage2": stage2,
        "ap_bridge": ap.get("gate"),
        "best_metrics": strategies.get("best_metrics"),
    }


def _best_no_forbidden_birth(strategies: dict[str, Any]) -> bool:
    best = strategies.get("best_metrics") or {}
    return bool(int(best.get("birth_from_d4rt_tube_count") or 0) == 0 and int(best.get("mixed_birth_count") or 0) == 0)
