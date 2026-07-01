from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, read_csv, read_json, safe_mean, utc_now, write_csv, write_json


AP_SCOPES = {
    "FULLMESH",
    "USED_FRAME_VISIBLE_SUPPORT",
    "PREDICTION_UNION_ISLAND",
    "SAME_SUPPORT_STREAM3D_PARITY",
    "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC",
    "NATIVE_CARRIER_SUPPORT_NO_SCANNET_AP_MASK",
    "UNKNOWN_SUPPORT",
}

PROBE5_SCENES = ["scene0050_00", "scene0081_01", "scene0591_00", "scene0011_00", "scene0030_00"]


def project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def rel(path: str | Path) -> str:
    path_obj = project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def load_dict(path: str | Path) -> dict[str, Any]:
    path_obj = project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def load_rows(path: str | Path) -> list[dict[str, str]]:
    path_obj = project(path)
    return read_csv(path_obj) if path_obj.exists() else []


def sha256_file(path: str | Path) -> str:
    path_obj = project(path)
    if not path_obj.exists() or not path_obj.is_file():
        return ""
    digest = hashlib.sha256()
    with path_obj.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    return parse_bool(value)


def parse_eval_metric_file(path: str | Path) -> dict[str, float | None]:
    path_obj = project(path)
    if not path_obj.exists():
        return {"AP": None, "AP50": None, "AP25": None}
    lines = [line.strip() for line in path_obj.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return {"AP": None, "AP50": None, "AP25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"AP": None, "AP50": None, "AP25": None}
    return {"AP": float_or_none(parts[0]), "AP50": float_or_none(parts[1]), "AP25": float_or_none(parts[2])}


def first_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get(key) or "") == value:
            return row
    return {}


def mean_by_key(rows: list[dict[str, Any]], key: str) -> float | None:
    return safe_mean(float_or_none(row.get(key)) for row in rows)


def support_scope_from_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    if "used_frame" in text or "used-support" in text or "used_support" in text or "visible_mask_support" in text:
        return "USED_FRAME_VISIBLE_SUPPORT"
    if "prediction_union" in text or "ap_eval_pre_points" in text or "bridge_wta" in text:
        return "PREDICTION_UNION_ISLAND"
    if "fullmesh" in text or "full_mesh" in text or "full_scene" in text:
        return "FULLMESH"
    if "same_support" in text:
        return "SAME_SUPPORT_STREAM3D_PARITY"
    if "cross_prepoints" in text:
        return "SAME_SUPPORT_STREAM3D_PARITY"
    return "UNKNOWN_SUPPORT"


def support_stats_from_summary(path: str | Path) -> dict[str, Any]:
    payload = load_dict(path)
    numeric = payload.get("numeric_mean") if isinstance(payload.get("numeric_mean"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    counts = [float_or_none(row.get("support_count") or row.get("pre_points_count")) for row in rows]
    counts = [count for count in counts if count is not None]
    gt_counts = [float_or_none(row.get("gt_instance_count")) for row in rows]
    gt_counts = [count for count in gt_counts if count is not None]
    return {
        "pre_points_count_mean": numeric.get("support_count") or numeric.get("support_unique_points") or mean_by_key(rows, "pre_points_count"),
        "pre_points_count_min": min(counts) if counts else None,
        "pre_points_count_max": max(counts) if counts else None,
        "gt_instance_count_mean": mean_by_key(rows, "gt_instance_count") if gt_counts else None,
        "prediction_union_ratio": numeric.get("prediction_union_ratio") or mean_by_key(rows, "prediction_union_ratio"),
        "prediction_union_inside_support_ratio": numeric.get("prediction_union_inside_support_ratio")
        or mean_by_key(rows, "prediction_union_inside_support_ratio"),
    }


def write_standard_outputs(output_root: str | Path, files: dict[str, Any]) -> None:
    root = project(output_root)
    for name, payload in files.items():
        path = root / name
        if name.endswith(".json"):
            write_json(path, payload)
        elif name.endswith(".csv"):
            write_csv(path, payload if isinstance(payload, list) else [])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(payload), encoding="utf-8")


def tiny_png(path: str | Path, title: str, rows: list[tuple[str, float | None]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path_obj = project(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _value in rows]
    values = [0.0 if value is None else float(value) for _label, value in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=120)
    ax.barh(np.arange(len(labels)), values, color=["#4776b4", "#d67553", "#62a87c", "#8a6fb0"][: len(labels)])
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_obj)
    plt.close(fig)


def now_payload(phase: str) -> dict[str, Any]:
    return {"phase": phase, "created_at": utc_now()}
