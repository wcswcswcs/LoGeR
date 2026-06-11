from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.oracle_candidate_upper_bound import build_parser as build_oracle_parser
from tools.oracle_candidate_upper_bound import run as run_oracle


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_metric_file(path: Path) -> dict[str, float | None]:
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


def _evaluate(output_config: str) -> None:
    output_file = Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(Path("data/prediction") / f"{output_config}_class_agnostic"),
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(output_file),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    subprocess.run(cmd, check=True)


def _load_atom_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("numeric_mean", {})


def _find_atom_summary(root: Path, atom_summary_root: str, variant: str, candidate_config: str) -> dict[str, Any]:
    direct = root / atom_summary_root / variant / f"{candidate_config}_summary.json"
    if direct.exists():
        return _load_atom_summary(direct)
    matches = sorted((root / atom_summary_root).glob(f"*/{candidate_config}_summary.json"))
    if matches:
        return _load_atom_summary(matches[0])
    return {}


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pre_points_support_ratio(root: Path, candidate_config: str, scenes: list[str]) -> float | None:
    ratios: list[float] = []
    for scene in scenes:
        pre_path = root / "data" / "TMP" / candidate_config / f"{scene}_pre_points.npy"
        gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
        if not pre_path.exists() or not gt_path.exists():
            continue
        pre_points = np.load(pre_path)
        gt_count = sum(1 for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip())
        ratios.append(float(pre_points.shape[0] / max(gt_count, 1)))
    return float(np.mean(ratios)) if ratios else None


def _write_markdown(prefix: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v14 Atom Oracle Diagnostic",
        "",
        "GT is used only to select oracle candidates and compute upper-bound metrics. These rows are diagnostic-only.",
        "",
        "| variant | candidate AP/AP50/AP25 | oracle AP/AP50/AP25 | pre % | atom known % | atoms/scene | surfels/atom | entropy p90 | traj var p90 | best IoU | high cand/GT | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    f"{_fmt(row.get('candidate_ap'), 100.0)}/{_fmt(row.get('candidate_ap50'), 100.0)}/{_fmt(row.get('candidate_ap25'), 100.0)}",
                    f"{_fmt(row.get('oracle_ap'), 100.0)}/{_fmt(row.get('oracle_ap50'), 100.0)}/{_fmt(row.get('oracle_ap25'), 100.0)}",
                    _fmt(row.get("support_pre_ratio"), 100.0),
                    _fmt(row.get("atom_known_support_ratio"), 100.0),
                    _fmt(row.get("num_atoms"), 1.0, 2),
                    _fmt(row.get("mean_surfels_per_atom"), 1.0, 2),
                    _fmt(row.get("mask_entropy_p90")),
                    _fmt(row.get("trajectory_variance_p90"), 1.0, 6),
                    _fmt(row.get("mean_best_iou_per_gt")),
                    _fmt(row.get("mean_gt_best_iou_ge_0p50")),
                    str(row.get("phase2_gate_pass")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- Phase 2 success requires broad-support oracle AP50 >= 0.60, AP25 >= 0.78, support pre% >= 25, and best-IoU improvement over C_hybrid oracle.",
            f"- best_variant: `{payload['summary'].get('best_variant')}`",
            f"- any_phase2_gate_pass: `{payload['summary'].get('any_phase2_gate_pass')}`",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value) * scale:.{digits}f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--variant", action="append", required=True, help="variant:candidate_config")
    parser.add_argument("--summary-root", default="outputs/audit/v14_atom_oracle")
    parser.add_argument("--atom-summary-root", default="outputs/v14_surfel_atom_bank")
    parser.add_argument("--min-select-iou", type=float, default=0.25)
    parser.add_argument("--c-hybrid-mean-best-iou", type=float, default=0.2756112117281759)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scenes = _read_seq_list(root / args.seq_list)
    rows: list[dict[str, Any]] = []
    for spec in args.variant:
        variant, candidate_config = spec.split(":", 1)
        oracle_config = f"stream4d_v14_oracle_{variant.lower()}_atom_candidate_probe5"
        oracle_args = build_oracle_parser().parse_args(
            [
                "--root",
                str(root),
                "--seq-list",
                str(root / args.seq_list),
                "--pred-config",
                candidate_config,
                "--pre-points-config",
                candidate_config,
                "--output-config",
                oracle_config,
                "--pred-suffix",
                "class_agnostic",
                "--min-select-iou",
                str(float(args.min_select_iou)),
                "--summary-root",
                args.summary_root,
            ]
        )
        run_oracle(oracle_args)
        _evaluate(oracle_config)
        candidate_metrics = _parse_metric_file(root / "data" / "evaluation" / "scannet" / f"{candidate_config}_class_agnostic.txt")
        oracle_metrics = _parse_metric_file(root / "data" / "evaluation" / "scannet" / f"{oracle_config}_class_agnostic.txt")
        oracle_summary_path = root / args.summary_root / f"{oracle_config}_oracle_upper_bound_summary.json"
        oracle_summary = json.loads(oracle_summary_path.read_text(encoding="utf-8"))
        atom_summary = _find_atom_summary(root, args.atom_summary_root, variant, candidate_config)
        atom_known_support = atom_summary.get("known_atom_support_ratio", atom_summary.get("atom_support_pre_ratio"))
        support = _pre_points_support_ratio(root, candidate_config, scenes)
        mean_best = float(oracle_summary["aggregate"].get("mean_best_iou_per_gt", 0.0))
        gate = bool(
            oracle_metrics["ap50"] is not None
            and oracle_metrics["ap25"] is not None
            and support is not None
            and float(oracle_metrics["ap50"]) >= 0.60
            and float(oracle_metrics["ap25"]) >= 0.78
            and float(support) >= 0.25
            and mean_best >= float(args.c_hybrid_mean_best_iou) + 0.08
        )
        rows.append(
            {
                "variant": variant,
                "candidate_config": candidate_config,
                "oracle_config": oracle_config,
                "candidate_ap": candidate_metrics["ap"],
                "candidate_ap50": candidate_metrics["ap50"],
                "candidate_ap25": candidate_metrics["ap25"],
                "oracle_ap": oracle_metrics["ap"],
                "oracle_ap50": oracle_metrics["ap50"],
                "oracle_ap25": oracle_metrics["ap25"],
                "support_pre_ratio": support,
                "atom_known_support_ratio": atom_known_support,
                "num_atoms": atom_summary.get("num_atoms"),
                "mean_surfels_per_atom": atom_summary.get("mean_surfels_per_atom"),
                "mask_entropy_mean": atom_summary.get("mask_entropy_mean"),
                "mask_entropy_p90": atom_summary.get("mask_entropy_p90"),
                "trajectory_variance_mean": atom_summary.get("trajectory_variance_mean"),
                "trajectory_variance_p90": atom_summary.get("trajectory_variance_p90"),
                "boundary_safe_ratio_mean": atom_summary.get("boundary_safe_ratio_mean"),
                "negative_visible_outside_ratio_mean": atom_summary.get("negative_visible_outside_ratio_mean"),
                "mean_best_iou_per_gt": mean_best,
                "mean_gt_best_iou_ge_0p25": oracle_summary["aggregate"].get("mean_gt_best_iou_ge_0p25"),
                "mean_gt_best_iou_ge_0p50": oracle_summary["aggregate"].get(
                    "mean_gt_best_iou_ge_0p50",
                    oracle_summary["aggregate"].get("mean_gt_best_iou_ge_0p5"),
                ),
                "phase2_gate_pass": gate,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "is_method_result": False,
                "is_diagnostic_only": True,
                "oracle_summary_path": str(oracle_summary_path),
            }
        )

    best = max(rows, key=lambda row: float(row.get("oracle_ap50") or 0.0)) if rows else {}
    payload = {
        "summary": {
            "num_variants": int(len(rows)),
            "best_variant": best.get("variant", ""),
            "best_oracle_ap50": best.get("oracle_ap50"),
            "any_phase2_gate_pass": bool(any(row.get("phase2_gate_pass") for row in rows)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
        },
        "rows": rows,
    }
    out_root = root / args.summary_root
    out_root.mkdir(parents=True, exist_ok=True)
    prefix = out_root / "atom_oracle_matrix_probe5"
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(_json_safe(rows))
    _write_markdown(prefix, payload)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))
    print(f"[v14-atom-oracle] wrote {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
