#!/usr/bin/env python3
"""Summarize ACL2 v68 layer-wise PCA review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-dir", required=True, type=Path)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return 0.0


def _top_rows(rows: Sequence[Mapping[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: _float(r, "auto_path_score"), reverse=True)
    out: List[Dict[str, Any]] = []
    for row in ranked[:limit]:
        out.append(
            {
                "tap": row.get("tap", ""),
                "layer": int(float(row.get("layer") or 0)),
                "auto_path_score": _float(row, "auto_path_score"),
                "support_risk_z_dist": _float(row, "support_risk_z_dist"),
                "dynamic_static_z_dist": _float(row, "dynamic_static_z_dist"),
                "explained_top3": _float(row, "explained_top3"),
                "semantic_trust_mean": _float(row, "semantic_trust_mean"),
            }
        )
    return out


def _selection_files(pca_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in ("read", "swa", "ttt", "merge"):
        out[name] = _read_json(pca_dir / f"{name}_layers_selected.json")
    return out


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v68 Layer-wise PCA Review Summary",
        "",
        f"- PCA dir: `{summary.get('pca_dir')}`",
        f"- PCA units: `{summary.get('num_pca_units')}`",
        f"- Semantic available: `{str(summary.get('semantic_available')).lower()}`",
        f"- Control group available: `{str(summary.get('control_group_available')).lower()}`",
        f"- Overall gate pass: `{str(summary.get('overall_gate_pass')).lower()}`",
        "",
        "## Availability",
        "",
        f"- Available taps: `{summary.get('available_taps')}`",
        f"- Unavailable taps: `{summary.get('unavailable_taps')}`",
        "",
        "## Top Auto Candidates",
        "",
    ]
    for row in summary.get("top_auto_candidates", []):
        lines.append(
            "- `{tap}` layer `{layer}` score `{score:.6f}` support_risk `{support:.6f}` "
            "dynamic_static `{dynamic:.6f}` explained_top3 `{explained:.6f}`".format(
                tap=row.get("tap", ""),
                layer=row.get("layer", ""),
                score=float(row.get("auto_path_score") or 0.0),
                support=float(row.get("support_risk_z_dist") or 0.0),
                dynamic=float(row.get("dynamic_static_z_dist") or 0.0),
                explained=float(row.get("explained_top3") or 0.0),
            )
        )
    lines.extend(["", "## Selection Status", ""])
    for name, payload in (summary.get("selection_status") or {}).items():
        lines.append(
            f"- `{name}`: gate_pass=`{str(payload.get('gate_pass')).lower()}`, "
            f"status=`{payload.get('selection_status')}`, selected_layers=`{payload.get('selected_layers')}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary.get("interpretation", "")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pca_dir = args.pca_dir
    pca_summary = _read_json(pca_dir / "pca_summary.json")
    rows = _read_csv(pca_dir / "auto_layer_metrics.csv")
    selections = _selection_files(pca_dir)
    gate_values = [bool(v.get("gate_pass")) for v in selections.values() if v]
    overall_gate = bool(gate_values) and all(gate_values)
    interpretation = (
        "Layer-wise PCA artifacts were generated per tap/layer. A true gate_pass only means the "
        "selected layers are ready for downstream cue construction; it is still not cue/action or "
        "trajectory-gain evidence. Empty selected_layers means downstream cue construction must use "
        "all-layer or geometry-controlled baselines instead of PCA-selected layers."
    )
    summary = {
        "schema": "acl2_v68_layerwise_pca_review_summary_v1",
        "pca_dir": str(pca_dir),
        "num_pca_units": int(len(rows)),
        "semantic_available": bool(pca_summary.get("semantic_available")),
        "available_taps": pca_summary.get("available_taps", {}),
        "unavailable_taps": pca_summary.get("unavailable_taps", {}),
        "control_group_available": bool(pca_summary.get("control_group_available")),
        "overall_gate_pass": bool(overall_gate),
        "top_auto_candidates": _top_rows(rows),
        "selection_status": selections,
        "interpretation": interpretation,
    }
    out_json = Path(args.out_json) if str(args.out_json).strip() else (pca_dir / "phaseB_pca_review_summary.json")
    out_md = Path(args.out_md) if str(args.out_md).strip() else (pca_dir / "phaseB_pca_review_summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
