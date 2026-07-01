from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


REQUIRED_ARTIFACTS = {
    "final_decision": "outputs/audit/v56_final_decision/final_decision.json",
    "phase0": "outputs/audit/v56_phase0_evidence_typing/phase0_summary.json",
    "core_update": "outputs/audit/v56_core_update/core_update_summary.json",
    "tentative_support": "outputs/audit/v56_tentative_support/tentative_support_summary.json",
    "promotion": "outputs/audit/v56_promotion/promotion_summary.json",
    "quarantine": "outputs/audit/v56_quarantine/quarantine_summary.json",
    "native_field": "outputs/audit/v56_native_field/native_field_summary.json",
    "stress_proxy": "outputs/audit/v56_stress_proxy/stress_proxy_summary.json",
}


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_required(paths: dict[str, str | Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key, path in paths.items():
        path_obj = _project(path)
        if not path_obj.exists():
            missing.append(_rel(path_obj))
            continue
        payloads[key] = read_json(path_obj)
    return payloads, missing


def _safe_get(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    return payload.get(key, default)


def build_v58_fact_lock(
    artifact_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    paths = {**REQUIRED_ARTIFACTS, **(artifact_paths or {})}
    artifacts, missing = _load_required(paths)
    final_decision = artifacts.get("final_decision", {})
    core_update = artifacts.get("core_update", {})
    tentative = artifacts.get("tentative_support", {})
    promotion = artifacts.get("promotion", {})
    quarantine = artifacts.get("quarantine", {})
    native_field = artifacts.get("native_field", {})
    stress_proxy = artifacts.get("stress_proxy", {})

    fact_lock = {
        "phase": "v58_phase0_fact_lock",
        "created_at": utc_now(),
        "input_paths": {key: _rel(path) for key, path in paths.items()},
        "missing_artifacts": missing,
        "v56_final_label": _safe_get(final_decision, "final_label"),
        "v56_partial_label": _safe_get(final_decision, "partial_label"),
        "v56_goal_achieved": _safe_get(final_decision, "goal_achieved"),
        "v56_core_ARI": _safe_get(final_decision, "core_4D_ARI"),
        "v56_core_purity": _safe_get(final_decision, "core_purity"),
        "v56_core_completeness": _safe_get(final_decision, "core_completeness"),
        "v56_core_temporal_span": _safe_get(final_decision, "core_temporal_span_mean"),
        "v56_core_real_minus_shuffled_ARI": _safe_get(final_decision, "real_minus_shuffled_ARI"),
        "v56_core_real_minus_no_temporal_ARI": _safe_get(final_decision, "real_minus_no_temporal_ARI"),
        "v56_core_gate_pass": _safe_get(core_update.get("gate", {}), "pass"),
        "v56_expanded_ARI": _safe_get(final_decision, "expanded_4D_ARI"),
        "v56_expanded_purity": _safe_get(final_decision, "expanded_purity"),
        "v56_expanded_completeness": _safe_get(final_decision, "expanded_completeness"),
        "v56_expanded_temporal_span": _safe_get(final_decision, "expanded_temporal_span_mean"),
        "v56_tentative_added_component_count": _safe_get(tentative, "tentative_added_component_count"),
        "v56_tentative_gate_pass": _safe_get(tentative.get("gate", {}), "pass"),
        "v56_promotion_candidate_count": _safe_get(promotion, "promotion_candidate_count"),
        "v56_promoted_component_count": _safe_get(promotion, "promoted_component_count"),
        "v56_promotion_gate_pass": _safe_get(promotion.get("gate", {}), "pass"),
        "v56_quarantine_component_count": _safe_get(quarantine, "quarantine_component_count"),
        "v56_quarantine_gate_pass": _safe_get(quarantine.get("gate", {}), "pass"),
        "v56_native_field_available": _safe_get(final_decision, "native_field_available"),
        "v56_native_field_gate_pass": _safe_get(native_field.get("gate", {}), "pass"),
        "v56_stress_pass_count": _safe_get(stress_proxy, "stress_real_minus_mask_only_ARI_pass_count"),
        "v56_best_stress_gain": _safe_get(stress_proxy, "best_real_minus_mask_only_ARI_proxy"),
        "v56_stress_proxy_gate_pass": _safe_get(stress_proxy.get("gate", {}), "pass"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    gate = {
        "no_missing_required_artifacts": not missing,
        "v56_final_label_is_no_go_d4rt_control": fact_lock["v56_final_label"] == "NO_GO_D4RT_CONTROL",
        "v56_partial_label_is_tentative_support_signal": fact_lock["v56_partial_label"]
        == "PARTIAL_TENTATIVE_SUPPORT_SIGNAL",
        "v56_core_purity_ge_0_89": float(fact_lock["v56_core_purity"] or 0.0) >= 0.89,
        "v56_tentative_added_component_count_gt_0": int(fact_lock["v56_tentative_added_component_count"] or 0) > 0,
        "v56_promoted_component_count_eq_0": fact_lock["v56_promoted_component_count"] is not None
        and int(fact_lock["v56_promoted_component_count"]) == 0,
        "v56_native_field_available_true": bool(fact_lock["v56_native_field_available"]),
    }
    gate["pass"] = bool(all(gate.values()))
    fact_lock["gate"] = gate

    rows = [
        {
            "row": "v56_confirmed_core",
            "ARI": fact_lock["v56_core_ARI"],
            "purity": fact_lock["v56_core_purity"],
            "completeness": fact_lock["v56_core_completeness"],
            "temporal_span_mean": fact_lock["v56_core_temporal_span"],
            "real_minus_shuffled_ARI": fact_lock["v56_core_real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": fact_lock["v56_core_real_minus_no_temporal_ARI"],
            "gate_pass": fact_lock["v56_core_gate_pass"],
        },
        {
            "row": "v56_expanded_tentative",
            "ARI": fact_lock["v56_expanded_ARI"],
            "purity": fact_lock["v56_expanded_purity"],
            "completeness": fact_lock["v56_expanded_completeness"],
            "temporal_span_mean": fact_lock["v56_expanded_temporal_span"],
            "tentative_added_component_count": fact_lock["v56_tentative_added_component_count"],
            "gate_pass": fact_lock["v56_tentative_gate_pass"],
        },
        {
            "row": "v56_promotion",
            "promotion_candidate_count": fact_lock["v56_promotion_candidate_count"],
            "promoted_component_count": fact_lock["v56_promoted_component_count"],
            "gate_pass": fact_lock["v56_promotion_gate_pass"],
        },
        {
            "row": "v56_quarantine_native_stress",
            "quarantine_component_count": fact_lock["v56_quarantine_component_count"],
            "native_field_available": fact_lock["v56_native_field_available"],
            "stress_pass_count": fact_lock["v56_stress_pass_count"],
            "best_stress_gain": fact_lock["v56_best_stress_gain"],
        },
    ]
    return {"fact_lock": fact_lock, "v56_baseline_rows": rows}


def _write_dashboard(path: Path, fact_lock: dict[str, Any]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["core", "expanded"]
        ari = [fact_lock["v56_core_ARI"], fact_lock["v56_expanded_ARI"]]
        purity = [fact_lock["v56_core_purity"], fact_lock["v56_expanded_purity"]]
        completeness = [fact_lock["v56_core_completeness"], fact_lock["v56_expanded_completeness"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(labels))
        width = 0.22
        ax.bar([i - width for i in x], ari, width=width, label="ARI")
        ax.bar(list(x), purity, width=width, label="purity")
        ax.bar([i + width for i in x], completeness, width=width, label="completeness")
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("v56 facts locked for v58 reset")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return "created"
    except Exception as exc:  # pragma: no cover - optional visualization backend
        path.with_suffix(".txt").write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return "unavailable"


def write_v58_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    out = _project(output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "fact_lock.json", payload["fact_lock"])
    write_csv(out / "v56_baseline_rows.csv", payload["v56_baseline_rows"])
    dashboard_path = _project("outputs/audit/v58_visualizations/phase0/v56_to_v58_reset_dashboard.png")
    status = _write_dashboard(dashboard_path, payload["fact_lock"])
    return {
        "fact_lock": _rel(out / "fact_lock.json"),
        "v56_baseline_rows": _rel(out / "v56_baseline_rows.csv"),
        "dashboard": _rel(dashboard_path),
        "dashboard_status": status,
    }
