from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUTS = {
    "v71_phase0_fact_lock": "outputs/audit/v71_phase0_fact_lock/fact_lock_summary.json",
    "v71_candidate_bank": "outputs/audit/v71_candidate_bank/candidate_bank_summary.json",
    "v71_semantic_features": "outputs/audit/v71_semantic_features/semantic_summary.json",
    "v71_d4rt_atoms": "outputs/audit/v71_d4rt_atoms/atom_summary.json",
    "v71_key_atoms": "outputs/audit/v71_key_atoms/key_atom_summary.json",
    "v71_phase5_setcover": "outputs/audit/v71_representative_setcover/setcover_summary.json",
    "v71_objectness_proxy": "outputs/audit/v71_objectness_proxy_separability12/objectness_proxy_summary.json",
    "v71_highbudget_oracle": "outputs/audit/v71_representative_setcover_oracle_budget_sweep12_highbudget/oracle_budget_sweep_summary.json",
}


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _gate_pass(summary: dict[str, Any], expected_decision: str | None = None) -> bool:
    gate = summary.get("gate") or {}
    if expected_decision is not None and summary.get("decision") != expected_decision:
        return False
    if "pass" in gate:
        return _bool(gate.get("pass"))
    return summary.get("decision") == expected_decision if expected_decision else False


def _metric_row(metric: str, value: Any, source: str, expected: str = "", passed: bool | None = None, notes: str = "") -> dict[str, Any]:
    return {
        "metric": metric,
        "value": value,
        "source": source,
        "expected": expected,
        "pass": passed,
        "notes": notes,
    }


def _find_oracle_budget(summary: dict[str, Any], budget: int) -> dict[str, Any]:
    for row in summary.get("summaries") or []:
        if int(row.get("budget") or -1) == int(budget):
            return row
    return {}


def _collect_gt_method_violations(payload: Any, source: str, path: str = "$") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        uses_gt = _bool(payload.get("uses_gt_for_prediction"))
        diagnostic = _bool(payload.get("diagnostic_only"))
        forbidden = _bool(payload.get("forbidden_for_method_table"))
        if uses_gt and not (diagnostic and forbidden):
            rows.append(
                {
                    "source": source,
                    "json_path": path,
                    "uses_gt_for_prediction": payload.get("uses_gt_for_prediction"),
                    "diagnostic_only": payload.get("diagnostic_only"),
                    "forbidden_for_method_table": payload.get("forbidden_for_method_table"),
                    "notes": "GT use is allowed only for rows marked diagnostic_only=true and forbidden_for_method_table=true.",
                }
            )
        for key, value in payload.items():
            rows.extend(_collect_gt_method_violations(value, source, f"{path}.{key}"))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            rows.extend(_collect_gt_method_violations(value, source, f"{path}[{idx}]"))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = dict(DEFAULT_INPUTS)
    missing = []
    loaded: dict[str, dict[str, Any]] = {}
    for name, rel_path in inputs.items():
        path = _rooted(rel_path)
        if not path.exists():
            missing.append({"input_name": name, "path": rel_path, "resolved_path": str(path)})
            continue
        loaded[name] = _load_json(path)

    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        _write_csv(output_root / "fact_metric_rows.csv", [])
        summary = {
            "phase": "v72_phase0_fact_lock",
            "decision": "NO_GO_PHASE0_MISSING_INPUT",
            "can_enter_v72_phase1": False,
            "missing_input_count": len(missing),
            "missing_inputs": missing,
            "method_prediction_uses_gt_anywhere": None,
        }
        _write_json(output_root / "fact_lock_summary.json", summary)
        _write_sha_rows(output_root, inputs)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    phase0 = loaded["v71_phase0_fact_lock"]
    candidate = loaded["v71_candidate_bank"]
    semantic = loaded["v71_semantic_features"]
    d4rt = loaded["v71_d4rt_atoms"]
    key_atoms = loaded["v71_key_atoms"]
    setcover = loaded["v71_phase5_setcover"]
    objectness = loaded["v71_objectness_proxy"]
    highbudget = loaded["v71_highbudget_oracle"]

    best_method = setcover.get("best_method") or {}
    objectness_best = {
        "SF50": objectness.get("best_non_oracle_representative_oracle_SF50"),
        "broad_rate": objectness.get("best_non_oracle_broad_large_selected_rate"),
        "underseg_rate": objectness.get("best_non_oracle_underseg_proxy_selected_rate"),
        "variant": objectness.get("best_non_oracle_variant"),
        "budget": objectness.get("best_non_oracle_budget"),
    }
    oracle192 = _find_oracle_budget(highbudget, 192)
    oracle192_means = oracle192.get("means") or {}

    gt_violations: list[dict[str, Any]] = []
    for name, payload in loaded.items():
        gt_violations.extend(_collect_gt_method_violations(payload, inputs[name]))
    method_prediction_uses_gt_anywhere = len(gt_violations) > 0

    facts = {
        "v71_phase0_pass": _gate_pass(phase0, "PASS_V71_FACT_LOCK"),
        "v71_candidate_bank_pass": _gate_pass(candidate, "PASS_V71_CANDIDATE_BANK"),
        "v71_semantic_features_pass": _gate_pass(semantic, "PASS_V71_SEMANTIC_FEATURES"),
        "v71_d4rt_atoms_pass": _gate_pass(d4rt, "PASS_V71_D4RT_ATOMS"),
        "v71_key_atoms_pass": _gate_pass(key_atoms, "PASS_V71_KEY_ATOMS"),
        "v71_phase5_decision": setcover.get("decision"),
        "v71_phase5_best_method_variant": best_method.get("variant"),
        "v71_phase5_representative_oracle_SF50": best_method.get("representative_oracle_SF50"),
        "v71_phase5_GT_best_IoU_mean": best_method.get("representative_GT_best_IoU_mean"),
        "v71_phase5_covered_D4RT_atom_weight_ratio": best_method.get("covered_D4RT_atom_weight_ratio"),
        "v71_phase5_covered_semantic_atom_weight_ratio": best_method.get("covered_semantic_atom_weight_ratio"),
        "v71_phase5_selected_mask_area_ratio_mean": best_method.get("selected_mask_area_ratio_mean"),
        "v71_objectness_proxy_best_non_oracle_variant": objectness_best["variant"],
        "v71_objectness_proxy_best_non_oracle_budget": objectness_best["budget"],
        "v71_objectness_proxy_best_non_oracle_SF50": objectness_best["SF50"],
        "v71_objectness_proxy_best_non_oracle_broad_rate": objectness_best["broad_rate"],
        "v71_objectness_proxy_best_non_oracle_underseg_rate": objectness_best["underseg_rate"],
        "v71_oracle_budget192_SF50": oracle192_means.get("oracle_SF50"),
        "v71_oracle_budget192_broad_rate": oracle192_means.get("broad_large_selected_rate"),
        "v71_oracle_budget192_underseg_rate": oracle192_means.get("underseg_proxy_selected_rate"),
        "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
    }
    can_enter_v72_phase1 = all(
        [
            facts["v71_phase0_pass"],
            facts["v71_candidate_bank_pass"],
            facts["v71_semantic_features_pass"],
            facts["v71_d4rt_atoms_pass"],
            facts["v71_key_atoms_pass"],
            facts["v71_phase5_decision"] == "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER",
            not method_prediction_uses_gt_anywhere,
        ]
    )
    facts["can_enter_v72_phase1"] = can_enter_v72_phase1

    rows = [
        _metric_row("v71_phase0_pass", facts["v71_phase0_pass"], inputs["v71_phase0_fact_lock"], "true", facts["v71_phase0_pass"]),
        _metric_row("v71_candidate_bank_pass", facts["v71_candidate_bank_pass"], inputs["v71_candidate_bank"], "true", facts["v71_candidate_bank_pass"]),
        _metric_row("v71_semantic_features_pass", facts["v71_semantic_features_pass"], inputs["v71_semantic_features"], "true", facts["v71_semantic_features_pass"]),
        _metric_row("v71_d4rt_atoms_pass", facts["v71_d4rt_atoms_pass"], inputs["v71_d4rt_atoms"], "true", facts["v71_d4rt_atoms_pass"]),
        _metric_row("v71_key_atoms_pass", facts["v71_key_atoms_pass"], inputs["v71_key_atoms"], "true", facts["v71_key_atoms_pass"]),
        _metric_row("v71_phase5_decision", facts["v71_phase5_decision"], inputs["v71_phase5_setcover"], "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER", facts["v71_phase5_decision"] == "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER"),
        _metric_row("v71_phase5_best_method_variant", facts["v71_phase5_best_method_variant"], inputs["v71_phase5_setcover"]),
        _metric_row("v71_phase5_representative_oracle_SF50", facts["v71_phase5_representative_oracle_SF50"], inputs["v71_phase5_setcover"], "record only"),
        _metric_row("v71_phase5_GT_best_IoU_mean", facts["v71_phase5_GT_best_IoU_mean"], inputs["v71_phase5_setcover"], "record only"),
        _metric_row("v71_phase5_covered_D4RT_atom_weight_ratio", facts["v71_phase5_covered_D4RT_atom_weight_ratio"], inputs["v71_phase5_setcover"], "record only"),
        _metric_row("v71_phase5_covered_semantic_atom_weight_ratio", facts["v71_phase5_covered_semantic_atom_weight_ratio"], inputs["v71_phase5_setcover"], "record only"),
        _metric_row("v71_phase5_selected_mask_area_ratio_mean", facts["v71_phase5_selected_mask_area_ratio_mean"], inputs["v71_phase5_setcover"], "record only"),
        _metric_row("v71_objectness_proxy_best_non_oracle_SF50", facts["v71_objectness_proxy_best_non_oracle_SF50"], inputs["v71_objectness_proxy"], "record only"),
        _metric_row("v71_objectness_proxy_best_non_oracle_broad_rate", facts["v71_objectness_proxy_best_non_oracle_broad_rate"], inputs["v71_objectness_proxy"], "record only"),
        _metric_row("v71_objectness_proxy_best_non_oracle_underseg_rate", facts["v71_objectness_proxy_best_non_oracle_underseg_rate"], inputs["v71_objectness_proxy"], "record only"),
        _metric_row("v71_oracle_budget192_SF50", facts["v71_oracle_budget192_SF50"], inputs["v71_highbudget_oracle"], "diagnostic oracle only"),
        _metric_row("v71_oracle_budget192_broad_rate", facts["v71_oracle_budget192_broad_rate"], inputs["v71_highbudget_oracle"], "diagnostic oracle only"),
        _metric_row("v71_oracle_budget192_underseg_rate", facts["v71_oracle_budget192_underseg_rate"], inputs["v71_highbudget_oracle"], "diagnostic oracle only"),
        _metric_row("method_prediction_uses_gt_anywhere", facts["method_prediction_uses_gt_anywhere"], "recursive_summary_scan", "false", not method_prediction_uses_gt_anywhere),
        _metric_row("can_enter_v72_phase1", facts["can_enter_v72_phase1"], "phase0_gate", "true", can_enter_v72_phase1),
    ]

    _write_csv(output_root / "fact_metric_rows.csv", rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_csv(output_root / "gt_method_violation_rows.csv", gt_violations)
    summary = {
        "phase": "v72_phase0_fact_lock",
        "decision": "PASS_V72_PHASE0_FACT_LOCK" if can_enter_v72_phase1 else "NO_GO_PHASE0_FACT_MISMATCH",
        "inputs": inputs,
        "key_metrics": facts,
        "gate": {
            "all_inputs_present": True,
            "v71_phase0_pass": facts["v71_phase0_pass"],
            "v71_candidate_bank_pass": facts["v71_candidate_bank_pass"],
            "v71_semantic_features_pass": facts["v71_semantic_features_pass"],
            "v71_d4rt_atoms_pass": facts["v71_d4rt_atoms_pass"],
            "v71_key_atoms_pass": facts["v71_key_atoms_pass"],
            "v71_phase5_is_no_go": facts["v71_phase5_decision"] == "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER",
            "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
            "pass": can_enter_v72_phase1,
        },
        "can_enter_v72_phase1": can_enter_v72_phase1,
        "gt_method_violation_count": len(gt_violations),
        "notes": [
            "Phase0 reads existing v71 artifacts only; no method prediction is recomputed.",
            "GT oracle rows remain allowed only when diagnostic_only=true and forbidden_for_method_table=true.",
        ],
    }
    _write_json(output_root / "fact_lock_summary.json", summary)
    _write_sha_rows(output_root, inputs)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _write_sha_rows(output_root: Path, inputs: dict[str, str]) -> None:
    rows: list[dict[str, Any]] = []
    for name, rel_path in inputs.items():
        path = _rooted(rel_path)
        if path.exists():
            rows.append({"name": name, "path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"name": f"output:{path.name}", "path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v72 Phase0 fact lock over existing v71 artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v72_phase0_fact_lock")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
