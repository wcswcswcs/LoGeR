from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.prediction_manifest import (
    load_prediction_manifest,
    prediction_dir,
    tmp_dir,
    write_retroactive_manifest,
)


def _split_configs(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_manifest(manifest: dict[str, Any] | None, key: str, default: bool = False) -> bool:
    if manifest is None:
        return default
    return bool(manifest.get(key, default))


def scan_configs(
    *,
    root: str | Path,
    configs: list[str],
    pred_suffix: str = "class_agnostic",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for config in configs:
        manifest, manifest_path = load_prediction_manifest(root_path, config, pred_suffix)
        pred_path = prediction_dir(root_path, config, pred_suffix)
        tmp_path = tmp_dir(root_path, config)
        name_has_oracle = "oracle" in config.lower()
        parse_error = bool(manifest and manifest.get("manifest_parse_error"))
        uses_gt = _bool_manifest(manifest, "uses_gt", default=name_has_oracle)
        uses_gt_for_prediction = _bool_manifest(manifest, "uses_gt_for_prediction", default=uses_gt)
        uses_gt_for_diagnostic = _bool_manifest(manifest, "uses_gt_for_diagnostic", default=uses_gt)
        gt_selected_output = _bool_manifest(manifest, "gt_selected_output", default=False)
        forbidden_for_method_table = _bool_manifest(manifest, "forbidden_for_method_table", default=uses_gt)
        alignment_used_for_prediction = _bool_manifest(manifest, "alignment_used_for_prediction", default=False)
        is_method_result = _bool_manifest(manifest, "is_method_result", default=not name_has_oracle)
        is_diagnostic_only = _bool_manifest(manifest, "is_diagnostic_only", default=name_has_oracle)
        suspicious_reasons: list[str] = []
        if manifest is None:
            suspicious_reasons.append("missing_manifest")
        if parse_error:
            suspicious_reasons.append("manifest_parse_error")
        if uses_gt and is_method_result:
            suspicious_reasons.append("uses_gt_and_method_result")
        if uses_gt_for_prediction:
            suspicious_reasons.append("uses_gt_for_prediction")
        if uses_gt_for_prediction and is_method_result:
            suspicious_reasons.append("uses_gt_for_prediction_and_method_result")
        if name_has_oracle and is_method_result:
            suspicious_reasons.append("oracle_name_marked_method")
        if uses_gt and not is_diagnostic_only:
            suspicious_reasons.append("uses_gt_not_diagnostic_only")
        if uses_gt_for_diagnostic and is_method_result:
            suspicious_reasons.append("uses_gt_for_diagnostic_and_method_result")
        if gt_selected_output and is_method_result:
            suspicious_reasons.append("gt_selected_output_and_method_result")
        if forbidden_for_method_table and is_method_result:
            suspicious_reasons.append("forbidden_for_method_table_and_method_result")
        if alignment_used_for_prediction:
            suspicious_reasons.append("alignment_used_for_prediction")
        if is_method_result and is_diagnostic_only:
            suspicious_reasons.append("method_marked_diagnostic_only")
        rows.append(
            {
                "config": config,
                "prediction_dir_exists": pred_path.exists(),
                "tmp_dir_exists": tmp_path.exists(),
                "manifest_exists": manifest is not None and not parse_error,
                "manifest_path": str(manifest_path) if manifest_path else "",
                "uses_gt": uses_gt,
                "uses_gt_for_prediction": uses_gt_for_prediction,
                "uses_gt_for_diagnostic": uses_gt_for_diagnostic,
                "gt_selected_output": gt_selected_output,
                "forbidden_for_method_table": forbidden_for_method_table,
                "alignment_source": "" if manifest is None else str(manifest.get("alignment_source", "")),
                "alignment_used_for_prediction": alignment_used_for_prediction,
                "alignment_used_for_diagnostic": _bool_manifest(manifest, "alignment_used_for_diagnostic", default=False),
                "is_method_result": is_method_result,
                "is_diagnostic_only": is_diagnostic_only,
                "name_has_oracle": name_has_oracle,
                "pre_points_policy": "" if manifest is None else str(manifest.get("pre_points_policy", "")),
                "eval_policy": "" if manifest is None else str(manifest.get("eval_policy", "")),
                "support_source": "" if manifest is None else str(manifest.get("support_source", "")),
                "geometry_source": "" if manifest is None else str(manifest.get("geometry_source", "")),
                "support_policy": "" if manifest is None else str(manifest.get("support_policy", "")),
                "source_configs": "" if manifest is None else ",".join(str(v) for v in manifest.get("source_configs", [])),
                "suspicious": bool(suspicious_reasons),
                "suspicious_reasons": ",".join(suspicious_reasons),
            }
        )
    summary = {
        "num_configs": len(rows),
        "num_configs_missing_manifest": int(sum(1 for row in rows if not row["manifest_exists"])),
        "num_oracle_configs": int(sum(1 for row in rows if row["name_has_oracle"] or row["uses_gt"])),
        "num_reportable_method_configs": int(
            sum(1 for row in rows if row["manifest_exists"] and row["is_method_result"] and not row["uses_gt"])
        ),
        "num_diagnostic_only_configs": int(sum(1 for row in rows if row["is_diagnostic_only"])),
        "num_suspicious_configs": int(sum(1 for row in rows if row["suspicious"])),
        "num_uses_gt_and_method_result": int(
            sum(1 for row in rows if row["uses_gt"] and row["is_method_result"])
        ),
        "num_uses_gt_for_prediction": int(sum(1 for row in rows if row["uses_gt_for_prediction"])),
        "num_uses_gt_for_prediction_and_method_result": int(
            sum(1 for row in rows if row["uses_gt_for_prediction"] and row["is_method_result"])
        ),
        "num_uses_gt_for_diagnostic_and_method_result": int(
            sum(1 for row in rows if row["uses_gt_for_diagnostic"] and row["is_method_result"])
        ),
        "num_gt_selected_output_and_method_result": int(
            sum(1 for row in rows if row["gt_selected_output"] and row["is_method_result"])
        ),
        "num_forbidden_for_method_table_and_method_result": int(
            sum(1 for row in rows if row["forbidden_for_method_table"] and row["is_method_result"])
        ),
        "num_alignment_used_for_prediction": int(sum(1 for row in rows if row["alignment_used_for_prediction"])),
        "num_configs_missing_eval_policy": int(
            sum(1 for row in rows if row["manifest_exists"] and not row["eval_policy"])
        ),
    }
    return {"summary": summary, "rows": rows}


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = ["# Stream4D v5 Reportable Config Scan", "", "## Summary", ""]
    for key in (
        "num_configs",
        "num_configs_missing_manifest",
        "num_oracle_configs",
        "num_reportable_method_configs",
        "num_diagnostic_only_configs",
        "num_suspicious_configs",
        "num_uses_gt_and_method_result",
        "num_uses_gt_for_prediction",
        "num_uses_gt_for_prediction_and_method_result",
        "num_uses_gt_for_diagnostic_and_method_result",
        "num_gt_selected_output_and_method_result",
        "num_forbidden_for_method_table_and_method_result",
        "num_alignment_used_for_prediction",
        "num_configs_missing_eval_policy",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.extend(
        [
            "",
            "## Configs",
            "",
            "| Config | pred dir | TMP dir | manifest | eval policy | uses GT | GT selected | align pred | method | diagnostic | suspicious | reasons |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["config"]),
                    str(row["prediction_dir_exists"]),
                    str(row["tmp_dir_exists"]),
                    str(row["manifest_exists"]),
                    str(row["eval_policy"]),
                    f"{row['uses_gt']}/{row['uses_gt_for_prediction']}/{row['uses_gt_for_diagnostic']}",
                    str(row["gt_selected_output"]),
                    str(row["alignment_used_for_prediction"]),
                    str(row["is_method_result"]),
                    str(row["is_diagnostic_only"]),
                    str(row["suspicious"]),
                    str(row["suspicious_reasons"]),
                ]
            )
            + " |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--configs", required=True, help="comma-separated config names")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--output", default="outputs/audit/v5_reportable_config_scan.md")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--require-eval-policy", action="store_true")
    parser.add_argument(
        "--retroactive-method-manifest",
        action="store_true",
        help="write non-GT method manifests for missing non-oracle configs before scanning",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    configs = _split_configs(args.configs)
    if args.retroactive_method_manifest:
        for config in configs:
            if "oracle" in config.lower():
                continue
            manifest, _ = load_prediction_manifest(root, config, args.pred_suffix)
            if manifest is None:
                write_retroactive_manifest(
                    root=root,
                    output_config=config,
                    pred_suffix=args.pred_suffix,
                    uses_gt=False,
                    is_method_result=True,
                    is_diagnostic_only=False,
                    notes=(
                        "Retroactive v5 audit manifest for a pre-existing non-oracle artifact. "
                        "This does not prove metric safety by itself; pair with metric integrity scan."
                    ),
                )
    payload = scan_configs(root=root, configs=configs, pred_suffix=args.pred_suffix)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["rows"][0].keys()) if payload["rows"] else ["config"])
        writer.writeheader()
        writer.writerows(payload["rows"])
    _write_markdown(output, payload)
    print(f"[scan-reportable-configs] wrote {output}")
    print(f"[scan-reportable-configs] wrote {output.with_suffix('.json')}")
    print(f"[scan-reportable-configs] wrote {output.with_suffix('.csv')}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if payload["summary"]["num_uses_gt_and_method_result"] > 0:
        raise SystemExit(2)
    if payload["summary"]["num_uses_gt_for_prediction"] > 0:
        raise SystemExit(5)
    if payload["summary"]["num_uses_gt_for_diagnostic_and_method_result"] > 0:
        raise SystemExit(6)
    if payload["summary"]["num_gt_selected_output_and_method_result"] > 0:
        raise SystemExit(7)
    if payload["summary"]["num_forbidden_for_method_table_and_method_result"] > 0:
        raise SystemExit(8)
    if payload["summary"]["num_alignment_used_for_prediction"] > 0:
        raise SystemExit(9)
    if args.require_manifest and payload["summary"]["num_configs_missing_manifest"] > 0:
        raise SystemExit(3)
    if args.require_eval_policy and payload["summary"]["num_configs_missing_eval_policy"] > 0:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
