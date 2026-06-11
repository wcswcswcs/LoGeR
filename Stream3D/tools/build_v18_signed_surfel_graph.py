from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.signed_surfel_graph import build_signed_surfel_graph, summarize_signed_surfel_graph


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    numeric_mean = {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in numeric_keys
        if any(row.get(key) is not None for row in rows)
    }
    gate = {
        "num_nodes_mean_ge_10k": bool(numeric_mean.get("num_nodes", 0.0) >= 10000.0),
        "visible_track_length_mean_ge_10": bool(numeric_mean.get("track_length_visible_mean", 0.0) >= 10.0),
        "uv_in01_rate_ge_0p95": bool(numeric_mean.get("uv_in01_rate", 0.0) >= 0.95),
        "cycle_uv_error_p90_le_5px": bool(numeric_mean.get("cycle_uv_error_p90", 999.0) <= 5.0),
        "largest_component_ratio_between_0p3_0p95": bool(
            0.3 <= numeric_mean.get("largest_graph_component_ratio", 0.0) <= 0.95
        ),
        "unobserved_surfel_ratio_le_0p05": bool(numeric_mean.get("unobserved_surfel_ratio", 1.0) <= 0.05),
    }
    gate["phase1_pass_without_edge_count"] = bool(all(gate.values()))
    return {
        "algorithm": "v18_signed_surfel_graph",
        "num_scenes": int(len(rows)),
        "num_ok_scenes": int(sum(1 for row in rows if row.get("status") == "ok")),
        "numeric_mean": numeric_mean,
        "gate": gate,
    }


def _write_bundle(prefix: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), dict)})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v18 Signed Surfel Graph Summary",
        "",
        "## Aggregate Gate",
        "",
    ]
    for key, value in aggregate["gate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Aggregate Means",
            "",
        ]
    )
    for key in (
        "num_nodes",
        "num_edges",
        "num_visible_surfels_per_frame_mean",
        "track_length_visible_mean",
        "uv_in01_rate",
        "cycle_uv_error_p90",
        "unobserved_surfel_ratio",
        "largest_graph_component_ratio",
    ):
        lines.append(f"- {key}: `{aggregate['numeric_mean'].get(key)}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | nodes | edges | visible/frame | track mean | uv in01 | cycle p90 | unobserved | largest comp | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_nodes")),
                    str(row.get("num_edges")),
                    f"{float(row.get('num_visible_surfels_per_frame_mean') or 0.0):.2f}",
                    f"{float(row.get('track_length_visible_mean') or 0.0):.4f}",
                    f"{float(row.get('uv_in01_rate') or 0.0):.6f}",
                    str(row.get("cycle_uv_error_p90")),
                    f"{float(row.get('unobserved_surfel_ratio') or 0.0):.6f}",
                    f"{float(row.get('largest_graph_component_ratio') or 0.0):.6f}",
                    str(row.get("phase1_pass")),
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_basic_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [str(row["scene"]) for row in rows]
    written: list[str] = []
    for key, name, ylabel in (
        ("num_edges", "edge_count_bar.png", "edges"),
        ("largest_graph_component_ratio", "largest_component_ratio_bar.png", "largest component ratio"),
        ("unobserved_surfel_ratio", "unobserved_surfel_ratio_bar.png", "unobserved surfel ratio"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(scenes, [float(row.get(key) or 0.0) for row in rows])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                json_safe({"phase": "v18_phase1", "figure": str(path), "metric": key, "source_rows": scenes}),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v18_phase1")
    parser.add_argument("--output-prefix", default="outputs/audit/v18_phase1/signed_surfel_graph_probe5")
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-max-frames", type=int, default=16)
    parser.add_argument("--knn-max-nodes-per-frame", type=int, default=0)
    parser.add_argument("--cross-frame-neighbors", type=int, default=4)
    parser.add_argument("--cross-frame-max-edges", type=int, default=0)
    parser.add_argument("--precut-mask-disagreement-ratio", type=float, default=0.25)
    parser.add_argument("--precut-source-rgb-discontinuity", type=float, default=0.45)
    parser.add_argument("--precut-uv-discontinuity", type=float, default=0.06)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
        graph = build_signed_surfel_graph(
            bank,
            knn_k=int(args.knn_k),
            knn_max_frames=int(args.knn_max_frames),
            knn_max_nodes_per_frame=int(args.knn_max_nodes_per_frame),
            cross_frame_neighbors=int(args.cross_frame_neighbors),
            cross_frame_max_edges=int(args.cross_frame_max_edges),
            precut_mask_disagreement_ratio=float(args.precut_mask_disagreement_ratio),
            precut_source_rgb_discontinuity=float(args.precut_source_rgb_discontinuity),
            precut_uv_discontinuity=float(args.precut_uv_discontinuity),
        )
        scene_dir = Path(args.output_root) / scene
        graph_path = scene_dir / "signed_surfel_graph.npz"
        graph.save(graph_path)
        row = summarize_signed_surfel_graph(graph, bank)
        row["bank_path"] = str(Path(args.bank_root) / scene / "measurement_bank.npz")
        row["graph_path"] = str(graph_path)
        row["edge_counts_json"] = json.dumps(row.get("edge_counts", {}), sort_keys=True)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "signed_surfel_graph_summary.json").write_text(
            json.dumps(json_safe(row), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows.append(row)
    aggregate = _aggregate(rows)
    _write_basic_plots(Path(args.output_root) / "figures", rows)
    _write_bundle(Path(args.output_prefix), rows, aggregate, args)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
