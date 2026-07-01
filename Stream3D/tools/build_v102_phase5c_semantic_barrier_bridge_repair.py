from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_phase5c_semantic_barrier_bridge_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE1_SUMMARY = AUDIT_ROOT / "v102_phase1_fragment_casebook" / "summary.json"
PHASE5B_DIR = AUDIT_ROOT / "v102_phase5b_chunk32_short_range_bridge_repair"
BRIDGE_ROWS = PHASE5B_DIR / "mask_pair_primitive_bridge_rows.parquet"
FEATURE_STORE = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_features.npz"
FEATURE_MANIFEST = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "feature_store_manifest.json"


VARIANTS = [
    {
        "variant_id": "semantic_tau0p4_gap4_missing_allow",
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap4_missing_allow",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p6_gap4_missing_allow",
        "semantic_cosine_min": 0.60,
        "missing_feature_policy": "allow",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap4_missing_block",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "block",
        "max_gap": 4,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
    {
        "variant_id": "semantic_tau0p5_gap2_missing_allow",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "max_gap": 2,
        "min_shared": 1,
        "ratio_min": 0.001,
        "broad_limit": 0.20,
    },
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | str:
    labels = labels.astype(bool)
    pos = int(np.sum(labels))
    neg = int(np.sum(~labels))
    if pos == 0 or neg == 0:
        return ""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for group_id in range(len(unique_scores)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) > 1:
            ranks[idx] = float(np.mean(ranks[idx]))
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _load_feature_map() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    store = np.load(FEATURE_STORE)
    features = store["features"].astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    ids = [str(x) for x in store["mask_observation_id"]]
    return {mask_id: features[i] for i, mask_id in enumerate(ids)}, _read_json(FEATURE_MANIFEST)


def _add_semantic_scores(df: pd.DataFrame, feature_map: dict[str, np.ndarray]) -> pd.DataFrame:
    cosines: list[float] = []
    available: list[bool] = []
    for row in df[["mask_a_observation_id", "mask_b_observation_id"]].itertuples(index=False):
        fa = feature_map.get(str(row.mask_a_observation_id))
        fb = feature_map.get(str(row.mask_b_observation_id))
        if fa is None or fb is None:
            cosines.append(np.nan)
            available.append(False)
        else:
            cosines.append(float(np.dot(fa, fb)))
            available.append(True)
    out = df.copy()
    out["semantic_residual_cosine"] = cosines
    out["semantic_residual_available"] = available
    return out


def _variant_metrics(df: pd.DataFrame, phase1: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_mask = (df["diagnostic_same_gt"] | df["diagnostic_different_gt"]).to_numpy(dtype=bool)
    labels = df.loc[label_mask, "diagnostic_same_gt"].to_numpy(dtype=bool)
    scores = df.loc[label_mask, "final_bridge_score"].to_numpy(dtype=np.float64)
    auc = _auc(scores, labels)
    positive_total = int(np.sum(df["diagnostic_same_gt"]))
    negative_total = int(np.sum(df["diagnostic_different_gt"]))
    same_semantic_negative_total = int(np.sum(df["diagnostic_same_semantic_different_gt"]))
    semantic_available_pair_count = int(np.sum(df["semantic_residual_available"]))
    semantic_available_rate = float(np.mean(df["semantic_residual_available"])) if len(df) else 0.0

    rows: list[dict[str, Any]] = []
    for spec in VARIANTS:
        broad_ok = (
            np.ones(len(df), dtype=bool)
            if spec["broad_limit"] is None
            else df["broad_contamination_score"].to_numpy(dtype=np.float64) <= float(spec["broad_limit"])
        )
        semantic_ok = df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"])
        if spec["missing_feature_policy"] == "allow":
            semantic_ok = semantic_ok | (~df["semantic_residual_available"].to_numpy(dtype=bool))
        accepted = (
            (df["frame_gap_index"].to_numpy(dtype=np.int64) <= int(spec["max_gap"]))
            & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= int(spec["min_shared"]))
            & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= float(spec["ratio_min"]))
            & broad_ok
            & semantic_ok
        )
        accepted_labeled = accepted & label_mask
        tp = int(np.sum(accepted & df["diagnostic_same_gt"].to_numpy(dtype=bool)))
        fp = int(np.sum(accepted & df["diagnostic_different_gt"].to_numpy(dtype=bool)))
        fp_same_sem = int(np.sum(accepted & df["diagnostic_same_semantic_different_gt"].to_numpy(dtype=bool)))
        accepted_count = int(np.sum(accepted))
        accepted_labeled_count = int(np.sum(accepted_labeled))
        recall = float(tp / max(positive_total, 1)) if positive_total else ""
        diff_false_among_accepted = float(fp / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        same_sem_false_among_accepted = (
            float(fp_same_sem / max(accepted_labeled_count, 1)) if accepted_labeled_count else ""
        )
        same_sem_hn_false_accept = (
            float(fp_same_sem / max(same_semantic_negative_total, 1)) if same_semantic_negative_total else ""
        )
        hard_negative_false_accept = float(fp / max(negative_total, 1)) if negative_total else ""
        phase5_formal = bool(
            recall != ""
            and diff_false_among_accepted != ""
            and same_sem_false_among_accepted != ""
            and hard_negative_false_accept != ""
            and auc != ""
            and recall >= 0.35
            and diff_false_among_accepted <= 0.20
            and same_sem_false_among_accepted <= 0.20
            and hard_negative_false_accept <= 0.20
            and auc >= 0.65
        )
        phase6_allowed = bool(phase5_formal and int(phase1.get("repair_candidate_pair_count", 0)) >= 30)
        rows.append(
            {
                "schema_version": "stream4d_v102_phase5c_semantic_barrier_variant_row_v1",
                "phase_id": "v102_phase5c_semantic_barrier_bridge_repair",
                "variant_id": spec["variant_id"],
                "repair_family": "false_bridge_high_semantic_residual_barrier",
                "max_gap": spec["max_gap"],
                "min_shared_gaussian_count": spec["min_shared"],
                "ratio_min_threshold": spec["ratio_min"],
                "broad_veto_area_ratio": spec["broad_limit"],
                "semantic_cosine_min": spec["semantic_cosine_min"],
                "missing_feature_policy": spec["missing_feature_policy"],
                "accepted_count": accepted_count,
                "accepted_labeled_count": accepted_labeled_count,
                "true_positive_same_gt_count": tp,
                "false_positive_different_gt_count": fp,
                "false_positive_same_semantic_different_gt_count": fp_same_sem,
                "diagnostic_positive_pair_count": positive_total,
                "diagnostic_negative_pair_count": negative_total,
                "same_semantic_different_gt_hard_negative_count": same_semantic_negative_total,
                "same_object_bridge_recall": recall,
                "different_gt_false_bridge_among_accepted": diff_false_among_accepted,
                "same_semantic_different_gt_false_bridge_among_accepted": same_sem_false_among_accepted,
                "same_semantic_hard_negative_false_accept_rate": same_sem_hn_false_accept,
                "hard_negative_false_accept_rate": hard_negative_false_accept,
                "bridge_auc": auc,
                "semantic_available_pair_count": semantic_available_pair_count,
                "semantic_available_rate": semantic_available_rate,
                "phase5_formal_bridge_gate_pass": phase5_formal,
                "phase6_ap_repair_allowed": phase6_allowed,
                "phase6_blocker": ""
                if phase6_allowed
                else "Phase1 repair_candidate_pair_count remains <30, or this variant does not pass Phase5.",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    best_passing = [r for r in rows if r["phase5_formal_bridge_gate_pass"]]
    if best_passing:
        best = max(best_passing, key=lambda r: float(r["same_object_bridge_recall"]))
    else:
        best = max(rows, key=lambda r: float(r["same_object_bridge_recall"]) if r["same_object_bridge_recall"] != "" else -1)
    bits = {
        "bridge_auc": auc,
        "diagnostic_positive_pair_count": positive_total,
        "diagnostic_negative_pair_count": negative_total,
        "same_semantic_different_gt_hard_negative_count": same_semantic_negative_total,
        "semantic_available_pair_count": semantic_available_pair_count,
        "semantic_available_rate": semantic_available_rate,
        "best_variant_id": best["variant_id"],
        "best_variant_same_object_bridge_recall": best["same_object_bridge_recall"],
        "best_variant_different_gt_false_bridge_among_accepted": best[
            "different_gt_false_bridge_among_accepted"
        ],
        "best_variant_same_semantic_different_gt_false_bridge_among_accepted": best[
            "same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "any_phase5_formal_bridge_gate_pass": any(bool(r["phase5_formal_bridge_gate_pass"]) for r in rows),
        "any_phase6_ap_repair_allowed": any(bool(r["phase6_ap_repair_allowed"]) for r in rows),
    }
    return rows, bits


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1_SUMMARY)
    feature_map, feature_manifest = _load_feature_map()
    bridge_df = pd.read_parquet(BRIDGE_ROWS)
    bridge_df = _add_semantic_scores(bridge_df, feature_map)
    variant_rows, bits = _variant_metrics(bridge_df, phase1)

    scored_bridge_path = OUT_DIR / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    variant_path = OUT_DIR / "semantic_barrier_variant_rows.csv"
    provider_path = OUT_DIR / "provider_bridge_summary_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    bridge_df.to_parquet(scored_bridge_path, index=False)
    _write_csv(variant_path, variant_rows)

    best = next(row for row in variant_rows if row["variant_id"] == bits["best_variant_id"])
    provider_rows = [
        {
            "schema_version": "stream4d_v102_phase5c_provider_bridge_summary_row_v1",
            "phase_id": "v102_phase5c_semantic_barrier_bridge_repair",
            "provider_id": "P6_DA3_GIANT_1_1_3DGS_official_plus_v91_RADIO_semantic_residual",
            "source_bridge_rows": _rel(BRIDGE_ROWS),
            "feature_store": _rel(FEATURE_STORE),
            "feature_store_sha256": feature_manifest.get("store_sha256"),
            "semantic_backend": feature_manifest.get("backend"),
            "candidate_pair_count": int(len(bridge_df)),
            "semantic_available_pair_count": bits["semantic_available_pair_count"],
            "semantic_available_rate": bits["semantic_available_rate"],
            "bridge_auc": bits["bridge_auc"],
            "best_variant_id": bits["best_variant_id"],
            "best_variant_same_object_bridge_recall": bits["best_variant_same_object_bridge_recall"],
            "best_variant_different_gt_false_bridge_among_accepted": bits[
                "best_variant_different_gt_false_bridge_among_accepted"
            ],
            "best_variant_same_semantic_different_gt_false_bridge_among_accepted": bits[
                "best_variant_same_semantic_different_gt_false_bridge_among_accepted"
            ],
            "phase5_formal_bridge_gate_pass": bits["any_phase5_formal_bridge_gate_pass"],
            "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
            "phase6_ap_repair_allowed": bits["any_phase6_ap_repair_allowed"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
    ]
    _write_csv(provider_path, provider_rows)

    gate_rows = [
        {
            "gate_id": "same_object_bridge_recall_best_variant",
            "pass": best["same_object_bridge_recall"] != "" and float(best["same_object_bridge_recall"]) >= 0.35,
            "expected": ">=0.35",
            "observed": best["same_object_bridge_recall"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "different_gt_false_bridge_best_variant",
            "pass": best["different_gt_false_bridge_among_accepted"] != ""
            and float(best["different_gt_false_bridge_among_accepted"]) <= 0.20,
            "expected": "<=0.20",
            "observed": best["different_gt_false_bridge_among_accepted"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "same_semantic_different_gt_false_bridge_best_variant",
            "pass": best["same_semantic_different_gt_false_bridge_among_accepted"] != ""
            and float(best["same_semantic_different_gt_false_bridge_among_accepted"]) <= 0.20,
            "expected": "<=0.20",
            "observed": best["same_semantic_different_gt_false_bridge_among_accepted"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "bridge_auc",
            "pass": bits["bridge_auc"] != "" and float(bits["bridge_auc"]) >= 0.65,
            "expected": ">=0.65",
            "observed": bits["bridge_auc"],
            "variant_id": "all_scores",
        },
        {
            "gate_id": "phase5_formal_bridge_gate",
            "pass": bits["any_phase5_formal_bridge_gate_pass"],
            "expected": True,
            "observed": bits["any_phase5_formal_bridge_gate_pass"],
            "variant_id": bits["best_variant_id"],
        },
        {
            "gate_id": "phase1_repair_candidate_pair_count",
            "pass": int(phase1.get("repair_candidate_pair_count", 0)) >= 30,
            "expected": ">=30 before Phase6 AP repair",
            "observed": phase1.get("repair_candidate_pair_count"),
            "variant_id": "phase1",
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
            "variant_id": "all",
        },
    ]
    _write_csv(gate_path, gate_rows)

    decision = (
        "PASS_PHASE5_BRIDGE_GATE__PHASE6_BLOCKED_BY_PHASE1"
        if bits["any_phase5_formal_bridge_gate_pass"] and not bits["any_phase6_ap_repair_allowed"]
        else "PASS_PHASE5_AND_PHASE6_ALLOWED"
        if bits["any_phase6_ap_repair_allowed"]
        else "NO_GO_SEMANTIC_BARRIER_REPAIR_GATE_STILL_FAILS"
    )
    summary = {
        "schema_version": "stream4d_v102_phase5c_semantic_barrier_bridge_repair_summary_v1",
        "phase_id": "v102_phase5c_semantic_barrier_bridge_repair",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "candidate_pair_count": int(len(bridge_df)),
        "variant_count": len(variant_rows),
        "semantic_backend": feature_manifest.get("backend"),
        "feature_store": _rel(FEATURE_STORE),
        "feature_store_sha256": feature_manifest.get("store_sha256"),
        "semantic_available_pair_count": bits["semantic_available_pair_count"],
        "semantic_available_rate": bits["semantic_available_rate"],
        "diagnostic_positive_pair_count": bits["diagnostic_positive_pair_count"],
        "diagnostic_negative_pair_count": bits["diagnostic_negative_pair_count"],
        "same_semantic_different_gt_hard_negative_count": bits["same_semantic_different_gt_hard_negative_count"],
        "bridge_auc": bits["bridge_auc"],
        "best_variant_id": bits["best_variant_id"],
        "best_variant_same_object_bridge_recall": bits["best_variant_same_object_bridge_recall"],
        "best_variant_different_gt_false_bridge_among_accepted": bits[
            "best_variant_different_gt_false_bridge_among_accepted"
        ],
        "best_variant_same_semantic_different_gt_false_bridge_among_accepted": bits[
            "best_variant_same_semantic_different_gt_false_bridge_among_accepted"
        ],
        "any_phase5_formal_bridge_gate_pass": bits["any_phase5_formal_bridge_gate_pass"],
        "phase1_repair_candidate_pair_count": phase1.get("repair_candidate_pair_count"),
        "phase1_broad_contamination_rate": phase1.get("broad_contamination_rate"),
        "phase6_ap_repair_allowed": bits["any_phase6_ap_repair_allowed"],
        "phase6_blocker": ""
        if bits["any_phase6_ap_repair_allowed"]
        else "Phase1 repair_candidate_pair_count remains below 30, so Phase6 AP repair is still not allowed.",
        "truthfulness_note": (
            "The semantic barrier uses v91 RADIO features only. Diagnostic GT labels are used only to score recall/"
            "false-bridge gates, not to accept or reject bridge pairs."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "mask_pair_primitive_bridge_rows_with_semantic": _rel(scored_bridge_path),
            "semantic_barrier_variant_rows": _rel(variant_path),
            "provider_bridge_summary_rows": _rel(provider_path),
            "variant_gate_rows": _rel(gate_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
