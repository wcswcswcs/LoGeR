from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import json_safe


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-root", default="outputs/v12_object_explanation")
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output-prefix", default="outputs/audit/v12_object_explanation/object_explanation_internal_probe5")
    args = parser.parse_args()
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    rows = []
    for config in configs:
        path = Path(args.summary_root) / f"{config}_summary.json"
        payload = _load_summary(path)
        means = payload.get("numeric_mean", {})
        row = {"config": config, **means}
        rows.append(row)
    out = {
        "algorithm": "v12_object_explanation_internal_diagnostic",
        "uses_gt": False,
        "is_method_result": False,
        "configs": configs,
        "rows": rows,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(out), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Stream4D v12 Object Explanation Internal Diagnostic",
        "",
        "This file summarizes internal slot metrics only. AP is reported in the unified matrix.",
        "",
        "| config | slots | rejected | assigned | core | reject | explained | negative | cannot-link |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config"]),
                    f"{float(row.get('num_active_slots', 0.0)):.2f}",
                    f"{float(row.get('num_rejected_slots', 0.0)):.2f}",
                    f"{float(row.get('assigned_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('core_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('reject_surfel_ratio', 0.0)):.4f}",
                    f"{float(row.get('measurement_explained_ratio', 0.0)):.4f}",
                    f"{float(row.get('visible_outside_negative_ratio', 0.0)):.4f}",
                    f"{float(row.get('same_frame_cannot_link_violations', 0.0)):.2f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
