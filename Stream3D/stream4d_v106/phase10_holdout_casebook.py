from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .artifacts import read_json, sha256_file, write_json


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"_read_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_payload": payload}


def _read_table(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json_if_exists(path)
    if isinstance(payload.get("_payload"), list):
        return payload["_payload"]
    if isinstance(payload.get("records"), list):
        return payload["records"]
    return []


def run_phase10_holdout_casebook(
    repo_root: Path,
    config_path: Path,
    config: Any,
    phase_config: Any,
    output_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    phase9_summary_path = (
        _resolve(repo_root, phase_config.phase9_summary)
        if phase_config.phase9_summary
        else output_root / "phase9" / "phase9_full_dev_summary.json"
    )
    phase9_summary = _read_json_if_exists(phase9_summary_path)
    phase9_pass = bool(phase9_summary.get("passes", False))
    freeze_path = phase9_summary_path.parent / "freeze_decision.json"
    freeze_decision = _read_json_if_exists(freeze_path)
    frozen_variant = freeze_decision.get("frozen_variant")
    has_frozen_config = bool(phase9_pass and frozen_variant)

    holdout_run = False
    status = "NO_GO_HOLDOUT_NOT_RUN"
    reason = (
        "Phase10 requires a frozen config from a passing Phase9 full-dev run. "
        "Current Phase9 evidence did not freeze any variant, so holdout was not run."
    )
    if has_frozen_config:
        status = "READY_FOR_EXTERNAL_HOLDOUT_RUN"
        reason = (
            "Phase9 claims a frozen variant, but this harness does not execute a fresh holdout scene run. "
            "Run the frozen config once on holdout with cache_mode=write_only_verified_no_read."
        )

    phase9_metric_table = _read_table(phase9_summary_path.parent / "variant_comparison_table.json")
    phase9_latency_table = _read_table(phase9_summary_path.parent / "latency_memory_table.json")
    metric_table = {
        "schema_version": "stream4d_v106_phase10_metric_table_v1",
        "status": "holdout_not_run",
        "records": [],
        "phase9_dev_reference_records": phase9_metric_table,
        "reason": reason,
    }
    latency_memory_table = {
        "schema_version": "stream4d_v106_phase10_latency_memory_table_v1",
        "status": "holdout_not_run",
        "records": [],
        "phase9_dev_reference_records": phase9_latency_table,
        "reason": reason,
    }
    casebook = {
        "schema_version": "stream4d_v106_phase10_identity_failure_casebook_v1",
        "status": "holdout_not_run",
        "records": [],
        "phase9_missing_evidence_json": _rel(
            repo_root, phase9_summary_path.parent / "missing_evidence_records.json"
        ),
        "reason": reason,
        "note": "No visual observations are written because no frozen holdout videos were generated.",
    }

    missing_required_outputs: List[str] = []
    config_frozen_path = output_dir / "config_frozen.yaml"
    config_sha_path = output_dir / "config_sha256.txt"
    if has_frozen_config:
        shutil.copyfile(config_path, config_frozen_path)
        config_sha_path.write_text(sha256_file(config_frozen_path) + "\n", encoding="utf-8")
    else:
        missing_required_outputs.extend(["config_frozen.yaml", "config_sha256.txt", "holdout videos"])

    video_manifest = {
        "schema_version": "stream4d_v106_phase10_video_manifest_v1",
        "status": "holdout_not_run",
        "videos": [],
        "reason": reason,
    }
    write_json(videos_dir / "video_manifest.json", video_manifest)

    final_decision = {
        "schema_version": "stream4d_v106_phase10_final_decision_v1",
        "status": status,
        "method_success": False,
        "phase9_pass": phase9_pass,
        "phase9_summary_json": _rel(repo_root, phase9_summary_path),
        "freeze_decision_json": _rel(repo_root, freeze_path),
        "frozen_variant": frozen_variant,
        "holdout_run": holdout_run,
        "holdout_split": phase_config.split_name,
        "forbid_holdout_parameter_callback": bool(phase_config.forbid_holdout_parameter_callback),
        "missing_required_outputs": missing_required_outputs,
        "reason": reason,
        "no_parameter_callback_performed": True,
        "config": {
            "run": asdict(config.run),
            "phase10": asdict(phase_config),
        },
        "outputs": {
            "final_decision": _rel(repo_root, output_dir / "final_decision.json"),
            "metric_table": _rel(repo_root, output_dir / "metric_table.json"),
            "latency_memory_table": _rel(repo_root, output_dir / "latency_memory_table.json"),
            "identity_failure_casebook": _rel(repo_root, output_dir / "identity_failure_casebook.json"),
            "videos": _rel(repo_root, videos_dir),
            "config_frozen": _rel(repo_root, config_frozen_path) if config_frozen_path.exists() else None,
            "config_sha256": _rel(repo_root, config_sha_path) if config_sha_path.exists() else None,
        },
    }

    write_json(output_dir / "metric_table.json", metric_table)
    write_json(output_dir / "latency_memory_table.json", latency_memory_table)
    write_json(output_dir / "identity_failure_casebook.json", casebook)
    write_json(output_dir / "final_decision.json", final_decision)
    return final_decision
