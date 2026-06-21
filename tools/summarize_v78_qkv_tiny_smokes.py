#!/usr/bin/env python3
"""Summarize v78 compact Q/K/V tiny-smoke cross-overlap signals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE9_ROOT_KITTI01 = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover"
)
PHASE9_ROOT_KITTI02 = Path(
    "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover"
)
DEFAULT_OUT_DIR = PHASE9_ROOT_KITTI01 / "qkv_tiny_smoke_crossseq_summary_v1"


SMOKES = [
    {
        "name": "KITTI01_chunk06_P9_34",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_34",
        "action_label": "weak_positive_boundary",
        "scene_geometry_score": -0.2793498072553317,
        "policy_side": "low_scene_score_select_all_head",
        "cross_csv": PHASE9_ROOT_KITTI01
        / "qkv_tiny_smoke_chunk06_p9_34_v1/compact_qkv_consistency_v1/compact_qkv_cross_overlap_rows.csv",
        "summary_json": PHASE9_ROOT_KITTI01
        / "qkv_tiny_smoke_chunk06_p9_34_v1/compact_qkv_consistency_v1/compact_qkv_consistency_summary.json",
    },
    {
        "name": "KITTI02_chunk14_P9_36",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_36",
        "action_label": "weak_positive_default",
        "scene_geometry_score": 0.7286276546427322,
        "policy_side": "high_scene_score_select_head6",
        "cross_csv": PHASE9_ROOT_KITTI02
        / "qkv_tiny_smoke_chunk14_p9_36_v1/compact_qkv_consistency_v1/compact_qkv_cross_overlap_rows.csv",
        "summary_json": PHASE9_ROOT_KITTI02
        / "qkv_tiny_smoke_chunk14_p9_36_v1/compact_qkv_consistency_v1/compact_qkv_consistency_summary.json",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _extract_smoke_rows(smoke: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(Path(smoke["cross_csv"])):
        if row.get("source_pair") != "prev_current__cur_current":
            continue
        rows.append(
            {
                "name": smoke["name"],
                "sequence": smoke["sequence"],
                "chunk": smoke["chunk"],
                "action": smoke["action"],
                "action_label": smoke["action_label"],
                "scene_geometry_score": smoke["scene_geometry_score"],
                "policy_side": smoke["policy_side"],
                "kind": row.get("kind"),
                "layer_id": int(row["layer_id"]),
                "cosine_mean": _float(row.get("cosine_mean")),
                "cosine_p05": _float(row.get("cosine_p05")),
                "cosine_p50": _float(row.get("cosine_p50")),
                "cosine_p95": _float(row.get("cosine_p95")),
                "mean_abs_diff": _float(row.get("mean_abs_diff")),
                "rmse_diff": _float(row.get("rmse_diff")),
                "cross_csv": str(smoke["cross_csv"]),
                "summary_json": str(smoke["summary_json"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for smoke in SMOKES:
        rows.extend(_extract_smoke_rows(smoke))

    rows_csv = args.out_dir / "qkv_tiny_smoke_crossseq_rows.csv"
    summary_json = args.out_dir / "qkv_tiny_smoke_crossseq_summary.json"
    _write_csv(rows_csv, rows)

    by_name_kind_layer = {
        f"{row['name']}:{row['kind']}:L{row['layer_id']}": row["cosine_mean"]
        for row in rows
    }
    deltas: dict[str, float] = {}
    for kind in ("k", "v"):
        for layer_id in (18, 26):
            a = by_name_kind_layer.get(f"KITTI01_chunk06_P9_34:{kind}:L{layer_id}")
            b = by_name_kind_layer.get(f"KITTI02_chunk14_P9_36:{kind}:L{layer_id}")
            if a is not None and b is not None:
                deltas[f"KITTI02_minus_KITTI01:{kind}:L{layer_id}"] = float(b) - float(a)

    _write_json(
        summary_json,
        {
            "schema": "acl2_v78_qkv_tiny_smoke_crossseq_summary_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "rows_csv": str(rows_csv),
            "num_rows": len(rows),
            "cosine_mean_by_name_kind_layer": by_name_kind_layer,
            "kitti02_minus_kitti01_cosine_mean_delta": deltas,
            "interpretation": [
                "Both weak-positive selector sides show very high L18 K/V overlap cosine.",
                "L26 is consistently weaker than L18, especially K.",
                "The high-scene-score P9_36 window has higher L26 K overlap cosine than the low-scene-score P9_34 window in this tiny sample.",
                "This is a diagnostic signal, not a method gate success.",
            ],
            "limitations": [
                "Only two tiny-smoke candidate runs are included.",
                "No baseline/control tiny-smoke Q/K/V dump is included.",
                "Selected-mask-conditioned Q/K/V stats are not computed because score_overlap and compact PCA grids have different token counts.",
            ],
        },
    )
    print(json.dumps({"rows": str(rows_csv), "summary": str(summary_json)}, indent=2))


if __name__ == "__main__":
    main()
