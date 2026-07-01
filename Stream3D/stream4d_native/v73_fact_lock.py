from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUTS = {
    "v68_final_decision": "outputs/audit/v68_final_decision/final_decision.json",
    "v70_final_decision": "outputs/audit/v70_final_decision/final_decision.json",
    "v71_phase0_fact_lock": "outputs/audit/v71_phase0_fact_lock/fact_lock_summary.json",
    "v71_candidate_bank": "outputs/audit/v71_candidate_bank/candidate_bank_summary.json",
    "v71_semantic_features": "outputs/audit/v71_semantic_features/semantic_summary.json",
    "v71_d4rt_atoms": "outputs/audit/v71_d4rt_atoms/atom_summary.json",
    "v71_key_atoms": "outputs/audit/v71_key_atoms/key_atom_summary.json",
    "v71_phase5_setcover": "outputs/audit/v71_representative_setcover/setcover_summary.json",
    "v71_oracle_budget192": "outputs/audit/v71_representative_setcover_oracle_budget_sweep12_highbudget/oracle_budget_192_summary.json",
    "v72_final_decision": "outputs/audit/v72_final_decision/final_decision.json",
    "v72_phase1_signal": "outputs/audit/v72_phase1_signal_adequacy/signal_adequacy_summary.json",
    "v72_phase3_d4rt_uvmember": "outputs/audit/v72_phase3_d4rt_proposal_verification_area_bin1_uvmember/d4rt_proposal_summary.json",
    "v72_phase4_objectness": "outputs/audit/v72_phase4_objectness_ranking_area_bin1_riskcap_bgproxyfix/objectness_summary.json",
    "v72_phase5_setcover_area": "outputs/audit/v72_phase5_proposal_setcover_area_bin1_uvmember_uvcoverage/proposal_setcover_summary.json",
    "v72_phase5_setcover_no_area": "outputs/audit/v72_phase5_proposal_setcover_no_area_floor_uvmember_uvcoverage/proposal_setcover_summary.json",
}


ROW_CONTRACT_DEFAULTS = {
    "scene_id": "aggregate",
    "chunk_id": "aggregate",
    "phase": "v73_phase0_fact_lock",
    "variant": "fact_lock",
    "uses_gt_for_prediction": False,
    "uses_gt_for_evaluation": False,
    "diagnostic_only": False,
    "forbidden_for_method_table": False,
    "method_prediction_safe": True,
    "score_mode": "not_applicable",
    "support_scope": "prior_artifact_summary",
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _gate_pass(payload: dict[str, Any], expected_decision: str | None = None) -> bool:
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    if "pass" in gate:
        return _bool(gate.get("pass"))
    if expected_decision is not None:
        return payload.get("decision") == expected_decision
    return False


def _metric_row(metric: str, value: Any, source: str, expected: str = "", passed: bool | None = None, notes: str = "") -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "metric": metric,
            "value": value,
            "source_artifact": source,
            "expected": expected,
            "pass": passed,
            "notes": notes,
        }
    )
    return row


def _gt_row(source: str, json_path: str, payload: dict[str, Any], allowed: bool, notes: str) -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "phase": "v73_phase0_gt_boundary",
            "variant": "gt_boundary_scan",
            "uses_gt_for_prediction": payload.get("uses_gt_for_prediction"),
            "uses_gt_for_evaluation": payload.get("uses_gt_for_evaluation"),
            "diagnostic_only": payload.get("diagnostic_only"),
            "forbidden_for_method_table": payload.get("forbidden_for_method_table"),
            "method_prediction_safe": allowed,
            "source_artifact": source,
            "json_path": json_path,
            "gt_boundary_allowed": allowed,
            "notes": notes,
        }
    )
    return row


def _collect_gt_boundary(payload: Any, source: str, path: str = "$") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        has_boundary_fields = any(
            key in payload
            for key in ("uses_gt_for_prediction", "uses_gt_for_evaluation", "diagnostic_only", "forbidden_for_method_table")
        )
        if has_boundary_fields:
            uses_gt = _bool(payload.get("uses_gt_for_prediction"))
            diagnostic = _bool(payload.get("diagnostic_only"))
            forbidden = _bool(payload.get("forbidden_for_method_table"))
            allowed = (not uses_gt) or (diagnostic and forbidden)
            notes = "allowed diagnostic/oracle GT row" if allowed and uses_gt else "method-safe/no GT prediction"
            if not allowed:
                notes = "GT prediction row is not marked diagnostic_only=true and forbidden_for_method_table=true"
            rows.append(_gt_row(source, path, payload, allowed, notes))
        for key, value in payload.items():
            rows.extend(_collect_gt_boundary(value, source, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(_collect_gt_boundary(value, source, f"{path}[{index}]"))
    return rows


def _source_row(name: str, path: Path, present: bool) -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "metric": "input_presence",
            "input_name": name,
            "value": present,
            "source_artifact": _rel(path) if present else str(path),
            "expected": "true",
            "pass": present,
            "bytes": path.stat().st_size if present else None,
            "sha256": _sha256(path) if present else None,
        }
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for name, rel_path in DEFAULT_INPUTS.items():
        path = _rooted(rel_path)
        present = path.exists()
        source_rows.append(_source_row(name, path, present))
        if not present:
            missing.append({"input_name": name, "path": rel_path, "resolved_path": str(path)})
            continue
        loaded[name] = _load_json(path)

    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        _write_csv(output_root / "main_rows.csv", source_rows)
        _write_csv(output_root / "metric_rows.csv", [])
        _write_csv(output_root / "fact_metric_rows.csv", [])
        _write_csv(output_root / "gt_boundary_rows.csv", [])
        _write_csv(output_root / "variant_summary_rows.csv", [])
        summary = {
            "phase": "v73_phase0_fact_lock",
            "decision": "NO_GO_PHASE0_MISSING_INPUT",
            "missing_input_count": len(missing),
            "can_enter_v73_local": False,
            "method_prediction_uses_gt_anywhere": None,
        }
        _write_json(output_root / "summary.json", summary)
        _write_json(output_root / "fact_lock_summary.json", summary)
        _write_sha_rows(output_root, DEFAULT_INPUTS)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    v71_phase0 = loaded["v71_phase0_fact_lock"]
    v71_candidate = loaded["v71_candidate_bank"]
    v71_semantic = loaded["v71_semantic_features"]
    v71_d4rt = loaded["v71_d4rt_atoms"]
    v71_key = loaded["v71_key_atoms"]
    v71_setcover = loaded["v71_phase5_setcover"]
    v71_oracle192 = loaded["v71_oracle_budget192"]
    v72_final = loaded["v72_final_decision"]
    v72_phase1 = loaded["v72_phase1_signal"]
    v72_phase3 = loaded["v72_phase3_d4rt_uvmember"]
    v72_phase4 = loaded["v72_phase4_objectness"]
    v72_phase5_area = loaded["v72_phase5_setcover_area"]
    v72_phase5_no_area = loaded["v72_phase5_setcover_no_area"]

    gt_rows: list[dict[str, Any]] = []
    for name, payload in loaded.items():
        gt_rows.extend(_collect_gt_boundary(payload, DEFAULT_INPUTS[name]))
    gt_violation_count = sum(1 for row in gt_rows if not _bool(row.get("gt_boundary_allowed")))
    method_prediction_uses_gt_anywhere = gt_violation_count > 0 or _bool(v72_final.get("method_uses_gt_anywhere"))

    v71_required_pass = all(
        [
            _gate_pass(v71_phase0, "PASS_V71_FACT_LOCK"),
            _gate_pass(v71_candidate, "PASS_V71_CANDIDATE_BANK"),
            _gate_pass(v71_semantic, "PASS_V71_SEMANTIC_FEATURES"),
            _gate_pass(v71_d4rt, "PASS_V71_D4RT_ATOMS"),
            _gate_pass(v71_key, "PASS_V71_KEY_ATOMS"),
        ]
    )
    v71_phase5_no_go = v71_setcover.get("decision") == "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER"
    v72_phase5_no_go = v72_final.get("decision") == "NO_GO_PHASE5_PROPOSAL_SETCOVER"
    v73_algorithm_reset_required = v71_required_pass and v71_phase5_no_go and v72_phase5_no_go
    can_enter_v73_local = v73_algorithm_reset_required and not method_prediction_uses_gt_anywhere

    v71_best = v71_setcover.get("best_method") or {}
    oracle_means = v71_oracle192.get("means") or {}
    v72_best = v72_phase5_no_area.get("best_method") or {}
    key_metrics = {
        "v71_phase0_to_phase4_pass": v71_required_pass,
        "v71_phase5_decision": v71_setcover.get("decision"),
        "v71_best_representative_SF50": v71_best.get("representative_oracle_SF50"),
        "v71_best_GT_best_IoU": v71_best.get("representative_GT_best_IoU_mean"),
        "v71_oracle_budget192_SF50": oracle_means.get("oracle_SF50"),
        "v71_oracle_budget192_broad_underseg_rate": _mean_pair(
            oracle_means.get("broad_large_selected_rate"), oracle_means.get("underseg_proxy_selected_rate")
        ),
        "v72_phase1_semantic_signal_pass": v72_phase1.get("decision"),
        "v72_phase3_D4RT_real_minus_shuffled": v72_phase3.get("real_minus_shuffled_SF50"),
        "v72_phase4_objectness_ranking_pass": v72_phase4.get("decision"),
        "v72_phase5_decision": v72_final.get("decision"),
        "v72_PSC7_SF50": v72_best.get("representative_proposal_oracle_SF50_diagnostic"),
        "v72_PSC7_unresolved_risk": v72_best.get("unresolved_broad_underseg_rate"),
        "v72_PSC9_SF50": _nested(v72_phase5_area, "best_method", "representative_proposal_oracle_SF50_diagnostic"),
        "v72_PSC9_unresolved_risk": _nested(v72_phase5_area, "best_method", "unresolved_broad_underseg_rate"),
        "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
        "can_enter_v73_local": can_enter_v73_local,
    }

    metric_rows = [
        _metric_row("v71_phase0_to_phase4_pass", key_metrics["v71_phase0_to_phase4_pass"], "v71 phase0/candidate/semantic/d4rt/key summaries", "true", v71_required_pass),
        _metric_row("v71_phase5_decision", key_metrics["v71_phase5_decision"], DEFAULT_INPUTS["v71_phase5_setcover"], "NO_GO_PHASE5_REPRESENTATIVE_SETCOVER", v71_phase5_no_go),
        _metric_row("v71_best_representative_SF50", key_metrics["v71_best_representative_SF50"], DEFAULT_INPUTS["v71_phase5_setcover"], "record only"),
        _metric_row("v71_best_GT_best_IoU", key_metrics["v71_best_GT_best_IoU"], DEFAULT_INPUTS["v71_phase5_setcover"], "record only"),
        _metric_row("v71_oracle_budget192_SF50", key_metrics["v71_oracle_budget192_SF50"], DEFAULT_INPUTS["v71_oracle_budget192"], "diagnostic oracle only"),
        _metric_row("v71_oracle_budget192_broad_underseg_rate", key_metrics["v71_oracle_budget192_broad_underseg_rate"], DEFAULT_INPUTS["v71_oracle_budget192"], "diagnostic oracle only"),
        _metric_row("v72_phase1_semantic_signal_pass", key_metrics["v72_phase1_semantic_signal_pass"], DEFAULT_INPUTS["v72_phase1_signal"], "record only"),
        _metric_row("v72_phase3_D4RT_real_minus_shuffled", key_metrics["v72_phase3_D4RT_real_minus_shuffled"], DEFAULT_INPUTS["v72_phase3_d4rt_uvmember"], "record only"),
        _metric_row("v72_phase4_objectness_ranking_pass", key_metrics["v72_phase4_objectness_ranking_pass"], DEFAULT_INPUTS["v72_phase4_objectness"], "PASS_V72_PHASE4_OBJECTNESS_RANKING", key_metrics["v72_phase4_objectness_ranking_pass"] == "PASS_V72_PHASE4_OBJECTNESS_RANKING"),
        _metric_row("v72_phase5_decision", key_metrics["v72_phase5_decision"], DEFAULT_INPUTS["v72_final_decision"], "NO_GO_PHASE5_PROPOSAL_SETCOVER", v72_phase5_no_go),
        _metric_row("v72_PSC7_SF50", key_metrics["v72_PSC7_SF50"], DEFAULT_INPUTS["v72_phase5_setcover_no_area"], "record only"),
        _metric_row("v72_PSC7_unresolved_risk", key_metrics["v72_PSC7_unresolved_risk"], DEFAULT_INPUTS["v72_phase5_setcover_no_area"], "record only"),
        _metric_row("v72_PSC9_SF50", key_metrics["v72_PSC9_SF50"], DEFAULT_INPUTS["v72_phase5_setcover_area"], "record only"),
        _metric_row("v72_PSC9_unresolved_risk", key_metrics["v72_PSC9_unresolved_risk"], DEFAULT_INPUTS["v72_phase5_setcover_area"], "record only"),
        _metric_row("method_prediction_uses_gt_anywhere", method_prediction_uses_gt_anywhere, "recursive_summary_scan + v72 final decision", "false", not method_prediction_uses_gt_anywhere),
        _metric_row("v73_algorithm_reset_required", v73_algorithm_reset_required, "phase0_gate", "true", v73_algorithm_reset_required),
        _metric_row("can_enter_v73_local", can_enter_v73_local, "phase0_gate", "true", can_enter_v73_local),
    ]

    summary = {
        "phase": "v73_phase0_fact_lock",
        "schema": "stream4d_v73_phase0_fact_lock_v1",
        "decision": "PASS_V73_PHASE0_FACT_LOCK" if can_enter_v73_local else "NO_GO_PHASE0_FACT_MISMATCH",
        "inputs": DEFAULT_INPUTS,
        "key_metrics": key_metrics,
        "gate": {
            "all_required_summaries_present": True,
            "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
            "v71_phase5_no_go": v71_phase5_no_go,
            "v72_phase5_no_go": v72_phase5_no_go,
            "v73_algorithm_reset_required": v73_algorithm_reset_required,
            "pass": can_enter_v73_local,
        },
        "can_enter_v73_local": can_enter_v73_local,
        "gt_boundary_row_count": len(gt_rows),
        "gt_boundary_violation_count": gt_violation_count,
        "notes": [
            "Phase0 reads prior artifacts only and does not compute a new method result.",
            "v73 local can start only because prior method rows remain no-GT and v71/v72 both ended in local/proposal No-Go states.",
            "Diagnostic/oracle GT rows are allowed only when marked diagnostic_only=true and forbidden_for_method_table=true.",
        ],
    }

    variant_summary_rows = [
        _metric_row("v73_phase0_fact_lock_decision", summary["decision"], "phase0_summary", "PASS_V73_PHASE0_FACT_LOCK", can_enter_v73_local),
    ]
    _write_csv(output_root / "main_rows.csv", source_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_csv(output_root / "fact_metric_rows.csv", metric_rows)
    _write_csv(output_root / "gt_boundary_rows.csv", gt_rows)
    _write_csv(output_root / "variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "fact_lock_summary.json", summary)
    _write_sha_rows(output_root, DEFAULT_INPUTS)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _mean_pair(a: Any, b: Any) -> float | None:
    values = [value for value in (_float(a), _float(b)) if value is not None]
    return float(sum(values) / len(values)) if values else None


def _write_sha_rows(output_root: Path, inputs: dict[str, str]) -> None:
    rows: list[dict[str, Any]] = []
    for name, rel_path in inputs.items():
        path = _rooted(rel_path)
        if path.exists():
            row = dict(ROW_CONTRACT_DEFAULTS)
            row.update({"name": name, "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            row = dict(ROW_CONTRACT_DEFAULTS)
            row.update({"name": f"output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v73 Phase0 fact lock over v71/v72 Stream4D artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase0_fact_lock")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
