from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe


def _parse_metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    try:
        return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}
    except ValueError:
        return {"ap": None, "ap50": None, "ap25": None}


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row(root: Path, key: str, config: str, summary_path: Path | None) -> dict[str, Any]:
    metrics = _parse_metric(root / "data" / "evaluation" / "scannet" / f"{config}_class_agnostic.txt")
    s0 = _parse_metric(root / "data" / "evaluation" / "scannet" / f"{key}_on_s0_probe5_class_agnostic.txt")
    s1 = _parse_metric(root / "data" / "evaluation" / "scannet" / f"{key}_on_s1_probe5_class_agnostic.txt")
    summary = _load_summary(summary_path) if summary_path is not None else {}
    means = summary.get("numeric_mean", {})
    return {
        "key": key,
        "config": config,
        **metrics,
        "on_s0_ap": s0["ap"],
        "on_s0_ap50": s0["ap50"],
        "on_s0_ap25": s0["ap25"],
        "on_s1_ap": s1["ap"],
        "on_s1_ap50": s1["ap50"],
        "on_s1_ap25": s1["ap25"],
        "median_residual": means.get("median_residual"),
        "p90_residual": means.get("p90_residual"),
        "p95_residual": means.get("p95_residual"),
        "point_spacing_q50": means.get("point_spacing_q50"),
        "num_exported_points": means.get("num_exported_points"),
        "export_conflict_rate": means.get("export_conflict_rate"),
        "is_diagnostic_only": True,
        "is_method_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-prefix", default="outputs/audit/v13_geometry_diagnostic/geometry_diagnostic_probe5")
    parser.add_argument("--sim3-diagnostic-json", default="outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = [
        {
            "key": "G0",
            "config": "scannet",
            **_parse_metric(root / "data" / "evaluation" / "scannet" / "scannet_class_agnostic.txt"),
            "on_s0_ap": None,
            "on_s0_ap50": None,
            "on_s0_ap25": None,
            "on_s1_ap": None,
            "on_s1_ap50": None,
            "on_s1_ap25": None,
            "median_residual": None,
            "p90_residual": None,
            "p95_residual": None,
            "point_spacing_q50": None,
            "num_exported_points": None,
            "export_conflict_rate": None,
            "is_diagnostic_only": False,
            "is_method_result": True,
        },
        _row(root, "stream4d_v10_g1", "stream4d_v10_g1_d4rt_raw_probe5", root / "outputs" / "v10_d4rt_geometry" / "stream4d_v10_g1_d4rt_raw_probe5_summary.json"),
        _row(root, "stream4d_v10_g2", "stream4d_v10_g2_d4rt_scene_sim3_probe5", root / "outputs" / "v10_d4rt_geometry" / "stream4d_v10_g2_d4rt_scene_sim3_probe5_summary.json"),
        _row(root, "stream4d_v10_g3", "stream4d_v10_g3_d4rt_window_sim3_probe5", root / "outputs" / "v10_d4rt_geometry" / "stream4d_v10_g3_d4rt_window_sim3_probe5_summary.json"),
        _row(root, "stream4d_v10_g4", "stream4d_v10_g4_d4rt_scene_sim3_density_probe5", root / "outputs" / "v10_d4rt_geometry" / "stream4d_v10_g4_d4rt_scene_sim3_density_probe5_summary.json"),
        _row(root, "stream4d_v10_g5", "stream4d_v10_g5_d4rt_window_sim3_density_probe5", root / "outputs" / "v10_d4rt_geometry" / "stream4d_v10_g5_d4rt_window_sim3_density_probe5_summary.json"),
    ]
    sim3_payload = _load_summary(root / args.sim3_diagnostic_json)
    payload = {
        "algorithm": "v13_geometry_diagnostic_summary",
        "rows": rows,
        "sim3_residual_diagnostic": sim3_payload.get("summary", {}),
        "notes": (
            "G1-G5 are diagnostic D4RT geometry replacement/materialization rows from v10 artifacts. "
            "They are not reportable method results."
        ),
    }
    prefix = root / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Stream4D v13 Geometry Diagnostic",
        "",
        "| row | AP | AP50 | AP25 | on S0 | on S1 | residual med/p90 | spacing q50 | exported pts | conflict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(v: Any, scale: float = 1.0) -> str:
        if v is None:
            return "NA"
        try:
            if not np.isfinite(float(v)):
                return "NA"
            return f"{float(v) * scale:.4f}"
        except Exception:
            return "NA"
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["key"],
                    fmt(row.get("ap"), 100.0),
                    fmt(row.get("ap50"), 100.0),
                    fmt(row.get("ap25"), 100.0),
                    f"{fmt(row.get('on_s0_ap'), 100.0)}/{fmt(row.get('on_s0_ap50'), 100.0)}/{fmt(row.get('on_s0_ap25'), 100.0)}",
                    f"{fmt(row.get('on_s1_ap'), 100.0)}/{fmt(row.get('on_s1_ap50'), 100.0)}/{fmt(row.get('on_s1_ap25'), 100.0)}",
                    f"{fmt(row.get('median_residual'))}/{fmt(row.get('p90_residual'))}",
                    fmt(row.get("point_spacing_q50")),
                    fmt(row.get("num_exported_points")),
                    fmt(row.get("export_conflict_rate"), 100.0),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Sim3 Residual Diagnostic", ""])
    for key, value in payload["sim3_residual_diagnostic"].items():
        lines.append(f"- {key}: `{value}`")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(payload["sim3_residual_diagnostic"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
