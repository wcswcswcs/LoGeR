from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_p10": 0.0, f"{prefix}_median": 0.0, f"{prefix}_p90": 0.0}
    return {
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def summarize_bank_v17(bank: MeasurementBank, *, boundary_safe_px: float = 3.0) -> dict[str, Any]:
    visible_counts = bank.visible_ok.sum(axis=0).astype(np.float64)
    positive_counts = bank.positive_observation.sum(axis=0).astype(np.float64)
    source_counts = bank.source_positive_propagated.sum(axis=0).astype(np.float64)
    negative_counts = bank.negative_observation.sum(axis=0).astype(np.float64)
    source_positive = bank.src_mask_id > 0
    uv_in01 = (
        bank.valid
        & np.isfinite(bank.uv_pred).all(axis=2)
        & (bank.uv_pred[:, :, 0] >= 0.0)
        & (bank.uv_pred[:, :, 0] <= 1.0)
        & (bank.uv_pred[:, :, 1] >= 0.0)
        & (bank.uv_pred[:, :, 1] <= 1.0)
    )
    boundary_safe = bank.positive_observation & (bank.boundary_distance >= float(boundary_safe_px))
    ambiguous = source_positive & (positive_counts > 0) & (negative_counts > 0)
    target_mask_counts: dict[tuple[int, int], int] = {}
    for frame_idx, frame_id in enumerate(bank.frame_ids.tolist()):
        ids, counts = np.unique(
            bank.target_mask_id[frame_idx][bank.positive_observation[frame_idx]], return_counts=True
        )
        for mask_id, count in zip(ids.tolist(), counts.tolist()):
            if int(mask_id) > 0:
                key = (int(frame_id), int(mask_id))
                target_mask_counts[key] = target_mask_counts.get(key, 0) + int(count)
    mask_count_values = np.asarray(list(target_mask_counts.values()), dtype=np.float64)
    row: dict[str, Any] = {
        "scene": bank.scene,
        "status": "ok",
        "num_frames": int(bank.frame_ids.shape[0]),
        "num_surfels": int(bank.num_surfels),
        "num_valid_tracks": int(np.count_nonzero(visible_counts > 0)),
        "num_mask_frames_available": int(np.count_nonzero(bank.mask_frame_available)),
        "num_mask_frames_missing": int(bank.mask_frame_available.shape[0] - np.count_nonzero(bank.mask_frame_available)),
        "uv_in01_rate": float(np.count_nonzero(uv_in01) / max(int(uv_in01.size), 1)),
        "visible_ok_rate": float(np.count_nonzero(bank.visible_ok) / max(int(bank.visible_ok.size), 1)),
        "track_length_visible_mean": float(np.mean(visible_counts)) if visible_counts.size else 0.0,
        "self_uv_error_p90": bank.meta.get("self_uv_error_p90_mean"),
        "cycle_uv_error_p90": bank.meta.get("cycle_uv_error_p90_mean"),
        "positive_observation_count_per_surfel_mean": float(np.mean(positive_counts)) if positive_counts.size else 0.0,
        "positive_observation_count_per_surfel_median": float(np.median(positive_counts)) if positive_counts.size else 0.0,
        "source_propagated_count_per_surfel_mean": float(np.mean(source_counts)) if source_counts.size else 0.0,
        "source_propagated_count_per_surfel_median": float(np.median(source_counts)) if source_counts.size else 0.0,
        "target_positive_samples_total": int(np.count_nonzero(bank.positive_observation)),
        "source_positive_samples_total": int(np.count_nonzero(bank.source_positive_propagated)),
        "negative_samples_total": int(np.count_nonzero(bank.negative_observation)),
        "surfel_positive_observation_rate": float(np.count_nonzero(positive_counts > 0) / max(bank.num_surfels, 1)),
        "surfel_source_propagated_rate": float(np.count_nonzero(source_counts > 0) / max(bank.num_surfels, 1)),
        "surfel_negative_observation_rate": float(np.count_nonzero(negative_counts > 0) / max(bank.num_surfels, 1)),
        "unobserved_surfel_ratio": float(np.count_nonzero(positive_counts == 0) / max(bank.num_surfels, 1)),
        "ambiguous_surfel_ratio": float(np.count_nonzero(ambiguous) / max(np.count_nonzero(source_positive), 1)),
        "boundary_safe_surfel_ratio": float(np.count_nonzero(boundary_safe.any(axis=0)) / max(bank.num_surfels, 1)),
        "mask_to_surfel_count_mean": float(np.mean(mask_count_values)) if mask_count_values.size else 0.0,
        "mask_to_surfel_count_p10": float(np.percentile(mask_count_values, 10)) if mask_count_values.size else 0.0,
        "mask_to_surfel_count_p50": float(np.percentile(mask_count_values, 50)) if mask_count_values.size else 0.0,
        "mask_to_surfel_count_p90": float(np.percentile(mask_count_values, 90)) if mask_count_values.size else 0.0,
        "v17_stat_fix": True,
        "legacy_v12_mean_positive_was_source_propagated": True,
    }
    row.update(_quantiles(positive_counts, "positive_observation_count_per_surfel"))
    row.update(_quantiles(source_counts, "source_propagated_count_per_surfel"))
    return row


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in keys
        if any(row.get(key) is not None for row in rows)
    }


def _write_bundle(prefix: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    numeric = _numeric_mean(rows)
    gate = {
        "surfel_positive_observation_rate_ge_0p90": bool(numeric.get("surfel_positive_observation_rate", 0.0) >= 0.90),
        "positive_observation_count_mean_ge_3p0": bool(
            numeric.get("positive_observation_count_per_surfel_mean", 0.0) >= 3.0
        ),
        "unobserved_surfel_ratio_le_0p05": bool(numeric.get("unobserved_surfel_ratio", 1.0) <= 0.05),
    }
    gate["measurement_density_not_main_bottleneck"] = bool(all(gate.values()))
    sparse_gate = {
        "surfel_positive_observation_rate_lt_0p70": bool(numeric.get("surfel_positive_observation_rate", 1.0) < 0.70),
        "positive_observation_count_mean_lt_1p5": bool(
            numeric.get("positive_observation_count_per_surfel_mean", 99.0) < 1.5
        ),
    }
    payload = {
        "args": vars(args),
        "aggregate": {
            "diagnostic_only": True,
            "uses_gt": False,
            "is_method_result": False,
            "num_scenes": int(len(rows)),
            "num_ok_scenes": int(sum(1 for row in rows if row.get("status") == "ok")),
            "numeric_mean": numeric,
            "gate": gate,
            "sparse_measurement_warning": sparse_gate,
        },
        "scenes": rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v17 Measurement Bank Fixed Statistics",
        "",
        "This diagnostic does not read GT. It separates target `positive_observation` from source `source_positive_propagated`.",
        "",
        "## Gate",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
        ]
    )
    for key in (
        "num_mask_frames_available",
        "num_mask_frames_missing",
        "uv_in01_rate",
        "visible_ok_rate",
        "track_length_visible_mean",
        "self_uv_error_p90",
        "cycle_uv_error_p90",
        "positive_observation_count_per_surfel_mean",
        "positive_observation_count_per_surfel_median",
        "source_propagated_count_per_surfel_mean",
        "source_propagated_count_per_surfel_median",
        "surfel_positive_observation_rate",
        "surfel_source_propagated_rate",
        "unobserved_surfel_ratio",
        "ambiguous_surfel_ratio",
        "boundary_safe_surfel_ratio",
    ):
        lines.append(f"- {key}: `{numeric.get(key)}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | frames | masks | uv in01 | pos obs mean | source prop mean | pos rate | source rate | unobserved | ambiguous | boundary safe |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_frames")),
                    f"{row.get('num_mask_frames_available')}/{row.get('num_mask_frames_missing')}",
                    f"{float(row.get('uv_in01_rate') or 0.0):.6f}",
                    f"{float(row.get('positive_observation_count_per_surfel_mean') or 0.0):.4f}",
                    f"{float(row.get('source_propagated_count_per_surfel_mean') or 0.0):.4f}",
                    f"{float(row.get('surfel_positive_observation_rate') or 0.0):.4f}",
                    f"{float(row.get('surfel_source_propagated_rate') or 0.0):.4f}",
                    f"{float(row.get('unobserved_surfel_ratio') or 0.0):.4f}",
                    f"{float(row.get('ambiguous_surfel_ratio') or 0.0):.4f}",
                    f"{float(row.get('boundary_safe_surfel_ratio') or 0.0):.4f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-prefix", default="outputs/audit/v17_phase1/measurement_bank_fixed_probe5")
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank_path = Path(args.bank_root) / scene / "measurement_bank.npz"
        bank = MeasurementBank.load(bank_path)
        row = summarize_bank_v17(bank, boundary_safe_px=float(args.boundary_safe_px))
        row["bank_path"] = str(bank_path)
        rows.append(row)
    payload = _write_bundle(Path(args.output_prefix), rows, args)
    print(json.dumps(json_safe(payload["aggregate"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
