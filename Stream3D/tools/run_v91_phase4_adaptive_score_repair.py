from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v91_phase4_multimask_score_repair as base_score  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_adaptive_score_repair"
MATERIALIZATION_ROOT = ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization"
PARENT_VARIANT = "V91_AD4_sr2_adapt_sig8_b05_j075_r12"


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
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


def _variant_specs() -> list[dict[str, str]]:
    return [
        {"variant_id": "V91_AS1_AD4_drop2_sceneorig", "risk_mode": "drop_broad_low_h9_2", "score_mode": "broad_scene_orig_ge065"},
        {"variant_id": "V91_AS2_AD4_all_sceneorig", "risk_mode": "all", "score_mode": "broad_scene_orig_ge065"},
        {"variant_id": "V91_AS3_AD4_drop5_hardneg_residual_scene", "risk_mode": "drop_broad_low_h9_5", "score_mode": "hard_negative_residual_scene"},
        {"variant_id": "V91_AS4_AD4_drop5_support_consistency", "risk_mode": "drop_broad_low_h9_5", "score_mode": "support_consistency"},
        {"variant_id": "V91_AS5_AD4_drop5_support_residual_scene", "risk_mode": "drop_broad_low_h9_5", "score_mode": "support_consistency_residual_scene"},
    ]


def _rewrite_audit_metadata() -> dict[str, Any]:
    config_path = OUT / "variant_config_rows.csv"
    config_rows = _read_csv(config_path)
    for row in config_rows:
        row["changed_module"] = "phase4_adaptive_extent_d4rt_residual_score_repair"
        row["reason_for_change"] = (
            "v91 Phase8 AD4 adaptive extent still misses AP50 control-margin gate; "
            "try D4RT residual/support score variants on the current best adaptive materialization"
        )
        row["expected_blocker"] = "CONTROL_BIAS_BLOCKER"
    _write_csv(config_path, config_rows)

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["phase"] = "v91_phase4_adaptive_score_repair"
    summary["schema"] = "stream4d_v91_phase4_adaptive_score_repair_v1"
    summary["parent_variant_id"] = PARENT_VARIANT
    summary["repair_scope"] = "dev_only_score_repair_on_current_best_adaptive_extent"
    summary["metadata_rewrite_note"] = (
        "This runner reuses the v91 multimask score-repair evaluator helpers, but the parent extent "
        "and generated-mask root are AD4 adaptive uncertainty materialization."
    )
    _write_json(summary_path, summary)

    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_score.OUT = OUT
    base_score.MATERIALIZATION_ROOT = MATERIALIZATION_ROOT
    base_score.PARENT_VARIANT = PARENT_VARIANT
    base_score._variant_specs = _variant_specs
    base_score.run(args)
    summary = _rewrite_audit_metadata()
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 score repair on current best adaptive uncertainty materialization.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
