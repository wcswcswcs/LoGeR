from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.prediction_manifest import json_safe, manifest_paths


def _bool_or_none(value: str) -> bool | None:
    if value == "":
        return None
    lower = value.lower()
    if lower in {"1", "true", "yes"}:
        return True
    if lower in {"0", "false", "no"}:
        return False
    raise ValueError(f"Expected boolean string, got {value!r}")


def _update_path(path: Path, updates: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({key: value for key, value in updates.items() if value is not None})
    audit = list(payload.get("metadata_update_audit", []))
    audit.append(
        {
            "tool": "tools.update_config_manifest_fields",
            "reason": updates.get("metadata_update_reason", "v10 manifest protocol completion"),
            "updated_fields": sorted(key for key, value in updates.items() if value is not None),
        }
    )
    payload["metadata_update_audit"] = audit
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch manifest metadata fields for pre-existing configs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--eval-policy", default="")
    parser.add_argument("--support-source", default="")
    parser.add_argument("--geometry-source", default="")
    parser.add_argument("--prediction-config", default="")
    parser.add_argument("--pre-points-config", default="")
    parser.add_argument("--algorithm-name", default="")
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--uses-gt-for-prediction", default="")
    parser.add_argument("--uses-gt-for-diagnostic", default="")
    parser.add_argument("--is-method-result", default="")
    parser.add_argument("--is-diagnostic-only", default="")
    parser.add_argument("--forbidden-for-method-table", default="")
    parser.add_argument("--gt-selected-output", default="")
    parser.add_argument("--reason", default="v10 manifest protocol completion for pre-existing artifact")
    args = parser.parse_args()

    updates: dict[str, Any] = {
        "eval_policy": args.eval_policy or None,
        "support_source": args.support_source or None,
        "geometry_source": args.geometry_source or None,
        "prediction_config": args.prediction_config or args.config,
        "pre_points_config": args.pre_points_config or args.config,
        "algorithm_name": args.algorithm_name or None,
        "algorithm": args.algorithm or None,
        "uses_gt_for_prediction": _bool_or_none(args.uses_gt_for_prediction),
        "uses_gt_for_diagnostic": _bool_or_none(args.uses_gt_for_diagnostic),
        "is_method_result": _bool_or_none(args.is_method_result),
        "is_diagnostic_only": _bool_or_none(args.is_diagnostic_only),
        "forbidden_for_method_table": _bool_or_none(args.forbidden_for_method_table),
        "gt_selected_output": _bool_or_none(args.gt_selected_output),
        "metadata_update_reason": args.reason,
    }
    written = []
    for path in manifest_paths(args.root, args.config, args.pred_suffix):
        if _update_path(path, updates):
            written.append(str(path))
    if not written:
        raise FileNotFoundError(f"No manifest found for {args.config}")
    print(json.dumps({"config": args.config, "updated": written}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
