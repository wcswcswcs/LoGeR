from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v91_phase4_adaptive_uncertainty_materialization as v91_adapt  # noqa: E402


DEFAULT_SUPPORT_ROWS = ROOT / "outputs/audit/v92_phase3_d4rt_highres_hr2_grid16/highres_native_carrier_support_rows.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/audit/v92_phase3c_hr2_uncertainty_readout"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
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


def _resolve_workspace_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _variant_specs() -> list[dict[str, Any]]:
    base = {
        "high_risk_max_masks": 4,
        "high_risk_extra_score_delta": 0.65,
        "high_risk_allow_broad_extra": True,
        "low_risk_max_masks": 2,
        "low_risk_extra_score_delta": 0.35,
        "low_risk_allow_broad_extra": False,
        "broad_rate_threshold": 0.65,
        "drop_per_selected_threshold": 1.0,
    }
    return [
        {
            **base,
            "variant_id": "V92_U0_HR2_fixed_r16_sp3",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 0.0,
            "beta": 0.0,
            "lambda_jitter": 0.0,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 16,
            "max_support_point_radius": 3,
        },
        {
            **base,
            "variant_id": "V92_U1_HR2_conf_sigma8_b075_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.75,
            "lambda_jitter": 0.0,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 28,
            "max_support_point_radius": 7,
        },
        {
            **base,
            "variant_id": "V92_U2_HR2_jitter_sigma8_j075_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.0,
            "lambda_jitter": 0.75,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 28,
            "max_support_point_radius": 7,
        },
        {
            **base,
            "variant_id": "V92_U3_HR2_conf_jitter_sigma8_b075_j075_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.75,
            "lambda_jitter": 0.75,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 28,
            "max_support_point_radius": 7,
        },
        {
            **base,
            "variant_id": "V92_U4_HR2_robust_cap_sigma8_b05_j05_r12",
            "base_radius": 12,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.5,
            "lambda_jitter": 0.5,
            "radius_scale": 1.0,
            "support_point_scale": 0.20,
            "max_radius": 24,
            "max_support_point_radius": 6,
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    support_rows = _resolve_workspace_path(args.support_rows)
    output_root = _resolve_workspace_path(args.output_root)
    if not support_rows.exists():
        raise FileNotFoundError(support_rows)
    output_root.mkdir(parents=True, exist_ok=True)

    original_out = v91_adapt.OUT
    original_support = v91_adapt.SUPPORT_ROWS
    original_specs = v91_adapt._variant_specs
    try:
        v91_adapt.OUT = output_root
        v91_adapt.SUPPORT_ROWS = support_rows
        v91_adapt._variant_specs = _variant_specs
        wrapped_summary = v91_adapt.run(argparse.Namespace())
    finally:
        v91_adapt.OUT = original_out
        v91_adapt.SUPPORT_ROWS = original_support
        v91_adapt._variant_specs = original_specs

    best = {}
    best_path = output_root / "best_variant_summary.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
    wrapper_summary = {
        "phase_id": "v92_phase3c_d4rt_uncertainty_readout",
        "schema": "stream4d_v92_phase3c_uncertainty_readout_wrapper_v1",
        "run_id": args.run_id,
        "support_rows": _rel(support_rows),
        "support_rows_sha256": _sha256(support_rows),
        "output_root": _rel(output_root),
        "variant_ids": [spec["variant_id"] for spec in _variant_specs()],
        "wrapped_script": "tools/run_v91_phase4_adaptive_uncertainty_materialization.py",
        "wrapped_protocol": "same local-window MV_AP evaluator and materializer; only U0-U4 uncertainty specs, SUPPORT_ROWS, and OUT are rebound",
        "wrapped_decision": wrapped_summary.get("decision", ""),
        "best_variant_id": best.get("variant_id", wrapped_summary.get("best_variant_id", "")),
        "best_MV_AP_window": best.get("mean_MV_AP_window", ""),
        "best_MV_AP50_window": best.get("mean_MV_AP50_window", ""),
        "best_delta_vs_phase8_best_MV_AP_window": wrapped_summary.get("best_delta_vs_phase8_best_MV_AP_window", ""),
        "best_delta_vs_phase8_best_MV_AP50_window": wrapped_summary.get("best_delta_vs_phase8_best_MV_AP50_window", ""),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "runtime_sec": time.time() - started,
    }
    wrapper_summary["decision"] = (
        "PASS_V92_PHASE3C_UNCERTAINTY_READOUT"
        if wrapped_summary.get("any_v91_phase8_progress_gate_pass")
        else "NO_GO_V92_PHASE3C_UNCERTAINTY_READOUT_NO_AP_GAIN"
    )
    wrapper_path = output_root / "v92_phase3c_wrapper_summary.json"
    wrapper_path.write_text(json.dumps(_jsonable(wrapper_summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sha_path = output_root / "SHA256SUMS.json"
    sha_payload = {}
    if sha_path.exists():
        sha_payload = json.loads(sha_path.read_text(encoding="utf-8"))
    sha_payload[_rel(wrapper_path)] = _sha256(wrapper_path)
    sha_payload[_rel(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    sha_path.write_text(json.dumps(sha_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(_jsonable(wrapper_summary), indent=2, sort_keys=True), flush=True)
    return wrapper_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v92 Phase3C D4RT uncertainty-aware readout on dev local windows.")
    parser.add_argument("--support-rows", default=str(DEFAULT_SUPPORT_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="v92_phase3c_hr2_uncertainty_readout")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
