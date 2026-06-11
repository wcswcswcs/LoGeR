from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fmt(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * scale:.6f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    p_labels = payload["prediction_labels"]
    s_labels = payload["support_labels"]
    lines = ["# Stream4D v7 Gap Matrix", ""]
    lines.append("## Prediction Rows")
    lines.append("")
    for idx, item in enumerate(payload["prediction_configs"]):
        lines.append(f"- P{idx} `{p_labels[idx]}`: `{item}`")
    lines.append("")
    lines.append("## Support Columns")
    lines.append("")
    for idx, item in enumerate(payload["support_configs"]):
        lines.append(f"- S{idx} `{s_labels[idx]}`: `{item}`")
    for metric in ("ap", "ap50", "ap25"):
        lines.extend(["", f"## {metric.upper()} Matrix", ""])
        header = "| Prediction | " + " | ".join(s_labels) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---:"] * len(s_labels)) + "|")
        for p_idx, p_label in enumerate(p_labels):
            row = [p_label]
            for s_idx in range(len(s_labels)):
                cell = payload["matrix"].get(f"p{p_idx}_s{s_idx}", {})
                row.append(_fmt(cell.get(metric)))
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Cells", ""])
    lines.append(
        "| Cell | Pred | Support | AP | AP50 | AP25 | target pre % | union % | union in target target % | #pred |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cell"],
                    row["prediction_label"],
                    row["support_label"],
                    _fmt(row.get("ap")),
                    _fmt(row.get("ap50")),
                    _fmt(row.get("ap25")),
                    _fmt(row.get("mean_target_pre_points_ratio"), 100.0),
                    _fmt(row.get("mean_prediction_union_ratio"), 100.0),
                    _fmt(row.get("mean_prediction_union_in_target_ratio_of_target"), 100.0),
                    _fmt(row.get("mean_num_pred_instances"), 1.0),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_heatmap(path: Path, payload: dict[str, Any], metric: str = "ap") -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    p_labels = payload["prediction_labels"]
    s_labels = payload["support_labels"]
    data = np.full((len(p_labels), len(s_labels)), np.nan, dtype=np.float32)
    for p_idx in range(len(p_labels)):
        for s_idx in range(len(s_labels)):
            value = payload["matrix"].get(f"p{p_idx}_s{s_idx}", {}).get(metric)
            if value is not None:
                data[p_idx, s_idx] = float(value)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="viridis", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(np.arange(len(s_labels)), s_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(p_labels)), p_labels)
    for p_idx in range(len(p_labels)):
        for s_idx in range(len(s_labels)):
            if np.isfinite(data[p_idx, s_idx]):
                ax.text(s_idx, p_idx, f"{data[p_idx, s_idx]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(f"v7 gap matrix {metric}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/audit/v7_gap_matrix/cross_prepoints_audit.json")
    parser.add_argument("--output-prefix", default="outputs/audit/v7_gap_matrix")
    parser.add_argument("--prediction-configs", required=True)
    parser.add_argument("--support-configs", required=True)
    parser.add_argument("--prediction-labels", default="P0,P1,P2,P3,P4,P5,P6")
    parser.add_argument("--support-labels", default="S0,S1,S2,S3,S4,S5,S6")
    args = parser.parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    prediction_configs = _split(args.prediction_configs)
    support_configs = _split(args.support_configs)
    prediction_labels = _split(args.prediction_labels)
    support_labels = _split(args.support_labels)
    pattern = re.compile(r"stream4d_v7_gap_p(\d+)_on_s(\d+)$")
    rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, Any]] = {}
    for item in payload.get("aggregates", []):
        output_config = str(item.get("output_config", ""))
        match = pattern.match(output_config)
        if not match:
            continue
        p_idx = int(match.group(1))
        s_idx = int(match.group(2))
        row = {
            "cell": f"p{p_idx}_s{s_idx}",
            "output_config": output_config,
            "prediction_label": prediction_labels[p_idx],
            "support_label": support_labels[s_idx],
            "prediction_config": prediction_configs[p_idx],
            "support_config": support_configs[s_idx],
            **item,
        }
        rows.append(row)
        matrix[f"p{p_idx}_s{s_idx}"] = row
    rows.sort(key=lambda row: (row["cell"].split("_")[0], row["cell"].split("_")[1]))
    out_payload = {
        "source": str(input_path),
        "prediction_configs": prediction_configs,
        "support_configs": support_configs,
        "prediction_labels": prediction_labels,
        "support_labels": support_labels,
        "rows": rows,
        "matrix": matrix,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    png_path = prefix.with_suffix(".png")
    json_path.write_text(json.dumps(_json_safe(out_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list))})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    _write_markdown(md_path, out_payload)
    heatmap = _write_heatmap(png_path, out_payload, metric="ap")
    print(f"[v7-gap-summary] wrote {json_path}")
    print(f"[v7-gap-summary] wrote {csv_path}")
    print(f"[v7-gap-summary] wrote {md_path}")
    if heatmap:
        print(f"[v7-gap-summary] wrote {heatmap}")


if __name__ == "__main__":
    main()
