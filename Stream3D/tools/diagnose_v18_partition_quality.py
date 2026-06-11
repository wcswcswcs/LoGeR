from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe


def _metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--metric-root", default="data/evaluation/scannet")
    args = parser.parse_args()
    payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    aggregate = dict(payload.get("aggregate", {}))
    output_config = str(aggregate.get("output_config") or "")
    metrics = _metric(Path(args.metric_root) / f"{output_config}_class_agnostic.txt") if output_config else {}
    rows = payload.get("rows", [])
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    diag: dict[str, Any] = {
        "source_summary_json": args.summary_json,
        "output_config": output_config,
        **metrics,
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }
    out = Path(args.output_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(json_safe(diag), indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# v18 partition quality: {output_config}", ""]
    for key in ("ap", "ap50", "ap25"):
        lines.append(f"- {key}: `{diag.get(key)}`")
    for key in ("num_kept_components", "num_exported_objects", "num_exported_points", "largest_component_ratio", "export_conflict_rate"):
        lines.append(f"- {key}_mean: `{diag['numeric_mean'].get(key)}`")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(diag), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
