from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_safe(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in keys
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_str(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    pos = [idx for idx, label in enumerate(labels) if label]
    neg = [idx for idx, label in enumerate(labels) if not label]
    if not pos or not neg:
        return None
    wins = 0.0
    total = float(len(pos) * len(neg))
    for pidx in pos:
        pscore = scores[pidx]
        for nidx in neg:
            nscore = scores[nidx]
            if pscore > nscore:
                wins += 1.0
            elif pscore == nscore:
                wins += 0.5
    return wins / total


def _stable_key_to_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return raw / float((1 << 64) - 1)


def _load_run_arrays(smoke_roots: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for root in smoke_roots:
        run_name = root.name
        carrier = np.load(root / "carrier_batch_smoke.npz")
        out[run_name] = {
            "valid": np.asarray(carrier["valid"], dtype=bool),
            "uv_pred": np.asarray(carrier["uv_pred"], dtype=np.float32),
            "visibility_prob": np.asarray(carrier["visibility_prob"], dtype=np.float32),
            "confidence_prob": np.asarray(carrier["confidence_prob"], dtype=np.float32),
        }
    return out


def _load_diagnostic_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row.get("material_node_id", ""): row for row in _read_csv(path)}


def _diagnostic_label(row: dict[str, Any], novelty: dict[str, str] | None) -> bool | None:
    if novelty is None:
        return None
    can_confirm = _bool_str(novelty.get("can_enter_confirmed_core")) and _bool_str(novelty.get("diagnostic_exact_match"))
    can_quarantine = _bool_str(novelty.get("can_enter_quarantine")) and (
        row.get("candidate_source") == "shared_shortcut_boundary" or novelty.get("state") in {"shared", "quarantine"}
    )
    return bool(can_confirm or can_quarantine)


def _score_rows(
    rows: list[dict[str, str]],
    arrays_by_run: dict[str, dict[str, np.ndarray]],
    novelty_by_material: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    real_scores_by_run: dict[str, list[float]] = {}
    for row in rows:
        run_name = row["run_name"]
        arrays = arrays_by_run[run_name]
        query_index = _int(row.get("d4rt_query_index"))
        valid = arrays["valid"][:, query_index]
        uv = arrays["uv_pred"][:, query_index]
        visibility = arrays["visibility_prob"][:, query_index]
        confidence = arrays["confidence_prob"][:, query_index]
        in_bounds = valid & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        visible_conf = np.asarray(visibility * confidence, dtype=np.float32)
        temporal_score = float(np.mean(visible_conf[in_bounds])) if np.count_nonzero(in_bounds) else 0.0
        accepted_ratio = _float(row.get("accepted_track_ratio"))
        in_bounds_ratio = _float(row.get("in_bounds_valid_ratio"))
        real_score = float(0.55 * accepted_ratio + 0.35 * temporal_score + 0.10 * in_bounds_ratio)
        source_frame = max(0, min(_int(row.get("support_frame_local")), int(valid.shape[0]) - 1))
        source_in_bounds = bool(in_bounds[source_frame])
        source_score = float(visible_conf[source_frame]) if source_in_bounds else 0.0
        no_temporal_score = float(0.70 * source_score + 0.30 * (1.0 if source_in_bounds else 0.0))
        candidate_prior_score = float(
            0.35 * (1.0 if _bool_str(row.get("has_K_mat")) else 0.0)
            + 0.20 * (1.0 if row.get("candidate_source") in {"bridge_low_support", "update_new_low_support"} else 0.0)
            + 0.20 * (1.0 if row.get("candidate_source") == "shared_shortcut_boundary" else 0.0)
            + 0.15 * min(_float(row.get("support_observation_count")) / 3.0, 1.0)
            + 0.10 * _stable_key_to_unit(row.get("query_candidate_id", ""))
        )
        diagnostic_label = _diagnostic_label(row, novelty_by_material.get(row.get("material_node_id", "")))
        scored_row = {
            **row,
            "control_success_label": bool(_bool_str(row.get("query_to_confirm")) or _bool_str(row.get("query_to_quarantine"))),
            "diagnostic_success_label": diagnostic_label,
            "real_evidence_score": real_score,
            "no_temporal_evidence_score": no_temporal_score,
            "candidate_prior_score": candidate_prior_score,
            "temporal_visibility_confidence_score": temporal_score,
            "source_frame_score": source_score,
            "control_label_source": "material_outcome_label_from_real_d4rt_track",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
        }
        scored.append(scored_row)
        real_scores_by_run.setdefault(run_name, []).append(real_score)

    shuffled_offsets = {
        run_name: max(1, len(scores) // 3) if len(scores) > 1 else 0
        for run_name, scores in real_scores_by_run.items()
    }
    run_indices: dict[str, int] = {}
    for row in scored:
        run_name = row["run_name"]
        idx = run_indices.get(run_name, 0)
        run_indices[run_name] = idx + 1
        scores = real_scores_by_run[run_name]
        offset = shuffled_offsets[run_name]
        row["shuffled_evidence_score"] = float(scores[(idx + offset) % len(scores)]) if scores else 0.0
        row["shuffle_policy"] = "deterministic_within_run_cyclic_shift"
    return scored


def _summarize(rows: list[dict[str, Any]], *, best_fixed: float) -> dict[str, Any]:
    labels = [bool(row["control_success_label"]) for row in rows]
    real_scores = [_float(row["real_evidence_score"]) for row in rows]
    shuffled_scores = [_float(row["shuffled_evidence_score"]) for row in rows]
    no_temporal_scores = [_float(row["no_temporal_evidence_score"]) for row in rows]
    prior_scores = [_float(row["candidate_prior_score"]) for row in rows]
    real_auc = _rank_auc(labels, real_scores)
    shuffled_auc = _rank_auc(labels, shuffled_scores)
    no_temporal_auc = _rank_auc(labels, no_temporal_scores)
    prior_auc = _rank_auc(labels, prior_scores)
    diagnostic_labels_raw = [row.get("diagnostic_success_label") for row in rows]
    diagnostic_rows = [idx for idx, label in enumerate(diagnostic_labels_raw) if label is not None]
    diagnostic_labels = [bool(diagnostic_labels_raw[idx]) for idx in diagnostic_rows]
    diagnostic_real_auc = _rank_auc(diagnostic_labels, [real_scores[idx] for idx in diagnostic_rows])
    diagnostic_shuffled_auc = _rank_auc(diagnostic_labels, [shuffled_scores[idx] for idx in diagnostic_rows])
    diagnostic_no_temporal_auc = _rank_auc(diagnostic_labels, [no_temporal_scores[idx] for idx in diagnostic_rows])
    diagnostic_prior_auc = _rank_auc(diagnostic_labels, [prior_scores[idx] for idx in diagnostic_rows])
    diagnostic_real_minus_shuffled = (
        None if diagnostic_real_auc is None or diagnostic_shuffled_auc is None else float(diagnostic_real_auc - diagnostic_shuffled_auc)
    )
    diagnostic_real_minus_no_temporal = (
        None if diagnostic_real_auc is None or diagnostic_no_temporal_auc is None else float(diagnostic_real_auc - diagnostic_no_temporal_auc)
    )
    success_count = sum(1 for label in labels if label)
    query_count = len(rows)
    confirm_or_quarantine = success_count / float(query_count) if query_count else None
    real_minus_shuffled = None if real_auc is None or shuffled_auc is None else float(real_auc - shuffled_auc)
    real_minus_no_temporal = None if real_auc is None or no_temporal_auc is None else float(real_auc - no_temporal_auc)
    return {
        "phase": "v62_active_query_control_auc_diagnostic",
        "query_count": query_count,
        "positive_count": int(success_count),
        "negative_count": int(query_count - success_count),
        "query_to_confirm_or_quarantine_rate": confirm_or_quarantine,
        "best_fixed_query_to_confirm_or_quarantine_rate": float(best_fixed),
        "best_fixed_plus_0_15": float(best_fixed + 0.15),
        "real_query_AUC": real_auc,
        "shuffled_query_AUC": shuffled_auc,
        "no_temporal_query_AUC": no_temporal_auc,
        "candidate_prior_AUC": prior_auc,
        "real_minus_shuffled_query_AUC": real_minus_shuffled,
        "real_minus_no_temporal_query_AUC": real_minus_no_temporal,
        "control_label_source": "material_outcome_label_from_real_d4rt_track",
        "diagnostic_label_source": "v62_increment_attribution_diagnostic_exact_match_or_can_enter_quarantine",
        "diagnostic_label_count": len(diagnostic_labels),
        "diagnostic_positive_count": int(sum(1 for label in diagnostic_labels if label)),
        "diagnostic_negative_count": int(sum(1 for label in diagnostic_labels if not label)),
        "diagnostic_real_query_AUC": diagnostic_real_auc,
        "diagnostic_shuffled_query_AUC": diagnostic_shuffled_auc,
        "diagnostic_no_temporal_query_AUC": diagnostic_no_temporal_auc,
        "diagnostic_candidate_prior_AUC": diagnostic_prior_auc,
        "diagnostic_real_minus_shuffled_query_AUC": diagnostic_real_minus_shuffled,
        "diagnostic_real_minus_no_temporal_query_AUC": diagnostic_real_minus_no_temporal,
        "control_status": "diagnostic_outcome_separability_not_independent_gt_auc",
        "method_note": (
            "AUC labels are the real D4RT material outcomes from Phase 5C. "
            "This measures whether full temporal carrier evidence separates resolved vs unresolved queries better than "
            "within-run shuffled scores and source-frame-only no-temporal scores. It is not an independent GT AUC."
        ),
        "gate": {
            "valid_query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15": (confirm_or_quarantine or 0.0) >= best_fixed + 0.15,
            "real_minus_shuffled_query_AUC_ge_0_15": (real_minus_shuffled or 0.0) >= 0.15,
            "real_minus_no_temporal_query_AUC_ge_0_10": (real_minus_no_temporal or 0.0) >= 0.10,
            "independent_gt_or_external_outcome_labels": False,
            "diagnostic_real_minus_shuffled_query_AUC_ge_0_15": (diagnostic_real_minus_shuffled or 0.0) >= 0.15,
            "diagnostic_real_minus_no_temporal_query_AUC_ge_0_10": (diagnostic_real_minus_no_temporal or 0.0) >= 0.10,
            "pass": False,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": bool(diagnostic_rows),
        "ap_status": "not_run",
    }


def _best_fixed_rate(path: Path) -> float:
    if not path.exists():
        return 0.296875
    data = _read_json(path)
    fixed = [row for row in data.get("baseline_rows", []) if row.get("baseline_id") != "Q7"]
    if not fixed:
        return 0.296875
    return max(float(row.get("query_to_confirm_or_quarantine_rate") or 0.0) for row in fixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v62 active-query shuffled/no-temporal control AUC diagnostics.")
    parser.add_argument("--material-outcome-root", required=True)
    parser.add_argument("--smoke-root", action="append", required=True)
    parser.add_argument("--novelty-csv", default="Stream3D/outputs/audit/v62_increment_attribution/novelty_material_rows.csv")
    parser.add_argument("--v61-query-summary", default="Stream3D/outputs/audit/v61_manifold_query/query_summary.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    material_root = Path(args.material_outcome_root)
    rows = _read_csv(material_root / "material_outcome_rows.csv")
    arrays_by_run = _load_run_arrays([Path(root) for root in args.smoke_root])
    scored = _score_rows(rows, arrays_by_run, _load_diagnostic_labels(Path(args.novelty_csv)))
    summary = _summarize(scored, best_fixed=_best_fixed_rate(Path(args.v61_query_summary)))
    summary["input_material_outcome_rows"] = str(material_root / "material_outcome_rows.csv")
    summary["input_smoke_roots"] = [str(Path(root)) for root in args.smoke_root]
    output_root = Path(args.output_root)
    _write_csv(output_root / "query_control_auc_rows.csv", scored)
    _write_json(output_root / "query_control_auc_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
