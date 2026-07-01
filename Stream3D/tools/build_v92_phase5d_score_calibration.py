from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


DEFAULT_IN = ROOT / "outputs/audit/v92_phase5b_source_container_edge_field"
DEFAULT_OUT = ROOT / "outputs/audit/v92_phase5d_score_calibration"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _score_specs() -> list[dict[str, str]]:
    return [
        {"variant_id": "V92_S0_F4_input_score", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "input_copy", "family": "score_baseline"},
        {"variant_id": "V92_S1_F4_area_penalty_score", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "area_penalty", "family": "score_repair"},
        {"variant_id": "V92_S2_F4_support_density_score", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "support_density", "family": "score_repair"},
        {"variant_id": "V92_S3_F4_balanced_support_area_score", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "balanced_support_area", "family": "score_repair"},
        {"variant_id": "V92_S4_F0_area_penalty_score", "base_variant": "V92_F0_whole_source_mask", "score_mode": "area_penalty", "family": "score_repair"},
        {"variant_id": "V92_S5_F4_compact_support_score", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "compact_support", "family": "score_repair"},
        {"variant_id": "V92_C10_F4_random_score_control", "base_variant": "V92_F4_d4rt_radio_graph", "score_mode": "random_score", "family": "control"},
    ]


def _stable_random_score(row: dict[str, Any], variant_id: str) -> float:
    text = f"{variant_id}:{row.get('scene_id')}:{row.get('frame_id')}:{row.get('mask_id')}:{row.get('mv_object_id')}"
    raw = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return 0.001 + (raw % 1000000) / 1000000.0


def _new_object_id(old: str, new_variant: str) -> str:
    parts = str(old).split(":", 1)
    suffix = parts[1] if len(parts) == 2 else str(old)
    return f"{new_variant}:{suffix}"


def _support_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("variant_id", row.get("variant", ""))),
        str(row.get("scene_id", "")),
        int(_num(row.get("frame_id"), -1)),
        str(row.get("mv_object_id", "")),
    )


def _score(row: dict[str, Any], support: dict[str, Any], spec: dict[str, str]) -> float:
    base = max(1e-6, _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)))
    area = max(1e-4, min(1.5, _num(support.get("generated_mask_area_ratio"), 1.0)))
    low = max(0.0, _num(support.get("lowres_support_count"), 0.0))
    hr = max(0.0, _num(support.get("hr2_support_count"), 0.0))
    support_mass = low + hr
    selected_regions = max(1.0, _num(support.get("selected_region_count"), 1.0))
    total_regions = max(selected_regions, _num(support.get("total_region_count"), selected_regions))
    compact = max(0.05, min(1.0, selected_regions / total_regions))
    broad_penalty = 0.72 if _bool(support.get("broad_risk")) else 1.0
    mode = str(spec["score_mode"])
    if mode == "input_copy":
        return base
    if mode == "area_penalty":
        return base * broad_penalty * max(0.05, 1.05 - 0.85 * area)
    if mode == "support_density":
        return broad_penalty * math.log1p(support_mass) / (0.12 + area)
    if mode == "balanced_support_area":
        support_term = math.log1p(support_mass)
        area_term = max(0.05, 1.0 - 0.65 * area)
        return base * broad_penalty * (0.35 + 0.35 * support_term + 0.30 * area_term)
    if mode == "compact_support":
        return base * broad_penalty * (0.25 + 0.45 * math.log1p(support_mass)) * math.sqrt(compact) * max(0.05, 1.1 - area)
    if mode == "random_score":
        return _stable_random_score(row, spec["variant_id"])
    raise ValueError(f"unknown score mode: {mode}")


def _link_masks(input_root: Path, out: Path, base_variant: str, new_variant: str) -> None:
    for scene in ["scene0011_00", "scene0050_00"]:
        src = input_root / "generated_masks" / base_variant / scene / "mask"
        dst = out / "generated_masks" / new_variant / scene / "mask"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel_src = os.path.relpath(src, dst.parent)
        try:
            dst.symlink_to(rel_src, target_is_directory=True)
        except FileExistsError:
            pass
        except OSError:
            shutil.copytree(src, dst)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    input_root = _resolve(args.input_root)
    out = _resolve(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    radius_sweep.OUT = out

    mv_rows_in = _read_csv(input_root / "mv_object_frame_mask_rows.csv")
    support_rows = _read_csv(input_root / "support_quality_rows.csv")
    support_by_key = {_support_key(row): row for row in support_rows}
    specs = _score_specs()
    selected_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for spec in specs:
        variant_id = spec["variant_id"]
        base_variant = spec["base_variant"]
        _link_masks(input_root, out, base_variant, variant_id)
        config_rows.append(
            {
                **spec,
                "source_mask_root": _rel(input_root / "generated_masks" / base_variant),
                "mask_link_root": _rel(out / "generated_masks" / variant_id),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for row in mv_rows_in:
            if row.get("variant") != base_variant:
                continue
            support = support_by_key.get(_support_key(row), {})
            score = _score(row, support, spec)
            new_row = dict(row)
            new_row["source_variant"] = variant_id
            new_row["variant"] = variant_id
            new_row["mv_object_id"] = _new_object_id(str(row.get("mv_object_id", "")), variant_id)
            new_row["frame_mask_score"] = score
            new_row["object_score"] = score
            new_row["selection_reason"] = f"v92_phase5d_{spec['score_mode']}_score_calibration_from_{base_variant}"
            new_row["uses_gt_for_prediction"] = False
            new_row["uses_future"] = False
            selected_rows.append(new_row)
            calibration_rows.append(
                {
                    "variant_id": variant_id,
                    "base_variant": base_variant,
                    "score_mode": spec["score_mode"],
                    "scene_id": row.get("scene_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "mv_object_id": new_row["mv_object_id"],
                    "old_score": row.get("object_score", ""),
                    "new_score": score,
                    "generated_mask_area_ratio": support.get("generated_mask_area_ratio", ""),
                    "lowres_support_count": support.get("lowres_support_count", ""),
                    "hr2_support_count": support.get("hr2_support_count", ""),
                    "selected_region_count": support.get("selected_region_count", ""),
                    "total_region_count": support.get("total_region_count", ""),
                    "broad_risk": support.get("broad_risk", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for spec in specs:
        variant_id = spec["variant_id"]
        rows = [row for row in selected_rows if row.get("variant") == variant_id]
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
    failure_rows = []
    if not passing:
        for row in control_rows:
            failure_rows.append(
                {
                    "variant_id": row.get("variant_id", ""),
                    "failure_type": "phase5d_score_calibration_gate_fail",
                    "repair_direction": "score-free/AP gap was tested with GT-free score calibration; if all fail, ranking alone is not sufficient and object extent/source quality remains blocker.",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                    "score_free_Match50_window": row.get("mean_score_free_Match50_window", ""),
                    "best_control_MV_AP_window": row.get("best_control_MV_AP_window", ""),
                    "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    summary = {
        "phase_id": "v92_phase5d_score_calibration",
        "schema": "stream4d_v92_phase5d_score_calibration_summary_v1",
        "run_id": str(args.run_id),
        "input_root": _rel(input_root),
        "variant_count": len(specs),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_phase5d_dev_gate_pass": bool(passing),
        "decision": "PASS_V92_PHASE5D_SCORE_CALIBRATION" if passing else "NO_GO_V92_PHASE5D_SCORE_CALIBRATION_NO_AP_GAIN",
        "row_counts": {
            "score_variant_config_rows": len(config_rows),
            "score_calibration_rows": len(calibration_rows),
            "mv_object_frame_mask_rows": len(selected_rows),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "failure_rows": len(failure_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }

    _write_csv(out / "score_variant_config_rows.csv", config_rows)
    _write_csv(out / "score_calibration_rows.csv", calibration_rows)
    _write_csv(out / "mv_object_rows.csv", [{"variant_id": row["variant"], "mv_object_id": row["mv_object_id"], "uses_gt_for_prediction": False, "uses_future": False} for row in selected_rows])
    _write_csv(out / "mv_object_frame_mask_rows.csv", selected_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "control_metric_rows.csv", control_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_csv(out / "score_failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "score_variant_config_rows.csv",
        out / "score_calibration_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "control_metric_rows.csv",
        out / "casebook_rows.csv",
        out / "score_failure_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v92 Phase5D GT-free score calibration over Phase5B masks.")
    parser.add_argument("--input-root", default=str(DEFAULT_IN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--run-id", default="v92_phase5d_score_calibration")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
