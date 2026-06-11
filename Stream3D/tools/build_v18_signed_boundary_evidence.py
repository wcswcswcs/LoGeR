from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.signed_boundary_evidence import build_signed_boundary_evidence, summarize_signed_boundary_evidence
from stream4d.signed_surfel_graph import SignedSurfelGraph


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "algorithm": "v18_signed_boundary_evidence",
        "variant": variant,
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }


def _write_bundle(prefix: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        f"# Stream4D v18 Signed Boundary Evidence: {aggregate['variant']}",
        "",
        "## Aggregate Means",
        "",
    ]
    for key, value in aggregate["numeric_mean"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | edges | merge mean | cut mean | score mean | score p90 | frames used |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_edges")),
                    f"{float(row.get('merge_weight_mean') or 0.0):.6f}",
                    f"{float(row.get('cut_weight_mean') or 0.0):.6f}",
                    f"{float(row.get('cut_score_mean') or 0.0):.6f}",
                    f"{float(row.get('cut_score_p90') or 0.0):.6f}",
                    f"{float(row.get('frames_used_mean') or 0.0):.4f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_histogram(prefix: Path, rows: list[dict[str, Any]], scene_paths: list[Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    scores = []
    for path in scene_paths:
        with np.load(path, allow_pickle=False) as data:
            scores.append(np.asarray(data["cut_score"], dtype=np.float32))
    if not scores:
        return
    all_scores = np.concatenate(scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_scores, bins=40)
    ax.set_xlabel("cut score")
    ax.set_ylabel("edge count")
    fig.tight_layout()
    out = prefix.with_name(prefix.name + "_cut_score_hist.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    out.with_suffix(".json").write_text(
        json.dumps(json_safe({"phase": "v18_phase3", "figure": str(out), "num_scores": int(all_scores.shape[0])}), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--graph-root", default="outputs/audit/v18_phase1")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v18_phase3")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--cut-lambda", type=float, default=1.0)
    parser.add_argument("--merge-lambda", type=float, default=0.65)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
        graph = SignedSurfelGraph.load(Path(args.graph_root) / scene / "signed_surfel_graph.npz")
        evidence = build_signed_boundary_evidence(
            bank,
            graph,
            variant=args.variant,
            boundary_safe_px=float(args.boundary_safe_px),
            cut_lambda=float(args.cut_lambda),
            merge_lambda=float(args.merge_lambda),
            bias=float(args.bias),
            seed=int(args.seed),
        )
        scene_dir = Path(args.output_root) / args.variant / scene
        evidence_path = scene_dir / "signed_boundary_evidence.npz"
        evidence.save(evidence_path)
        row = summarize_signed_boundary_evidence(evidence)
        row["bank_path"] = str(Path(args.bank_root) / scene / "measurement_bank.npz")
        row["graph_path"] = str(Path(args.graph_root) / scene / "signed_surfel_graph.npz")
        row["evidence_path"] = str(evidence_path)
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "signed_boundary_evidence_summary.json").write_text(
            json.dumps(json_safe(row), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows.append(row)
        evidence_paths.append(evidence_path)
    aggregate = _aggregate(rows, args.variant)
    prefix = Path(args.output_prefix) if args.output_prefix else Path(args.output_root) / args.variant / "signed_boundary_evidence_probe5"
    _write_bundle(prefix, rows, aggregate, args)
    _write_histogram(prefix, rows, evidence_paths)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
