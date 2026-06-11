from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_NAME = "config_manifest.json"
SCHEMA_VERSION = "stream4d_prediction_manifest_v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_pred_suffix(suffix: str = "class_agnostic") -> str:
    suffix = suffix.strip()
    if not suffix:
        suffix = "class_agnostic"
    return suffix[1:] if suffix.startswith("_") else suffix


def prediction_dir(root: str | Path, output_config: str, pred_suffix: str = "class_agnostic") -> Path:
    suffix = normalize_pred_suffix(pred_suffix)
    return Path(root) / "data" / "prediction" / f"{output_config}_{suffix}"


def tmp_dir(root: str | Path, output_config: str) -> Path:
    return Path(root) / "data" / "TMP" / output_config


def manifest_paths(root: str | Path, output_config: str, pred_suffix: str = "class_agnostic") -> list[Path]:
    return [
        prediction_dir(root, output_config, pred_suffix) / MANIFEST_NAME,
        tmp_dir(root, output_config) / MANIFEST_NAME,
    ]


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def repository_code_hash(root: str | Path = ".") -> str:
    root_path = Path(root).resolve()
    for name in (
        "stream4d_code_review_packet.sha256",
        "stream4d_v4_1_code_review_packet.sha256",
        "core_code_audit_pack.sha256",
    ):
        path = root_path / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip().split()
            if text:
                return text[0]
    return f"git:{_git_revision(root_path)}"


def build_prediction_manifest(
    *,
    output_config: str,
    root: str | Path = ".",
    is_method_result: bool = True,
    is_diagnostic_only: bool = False,
    uses_gt: bool = False,
    gt_usage: str = "none",
    source_configs: list[str] | None = None,
    pre_points_policy: str = "unknown",
    support_policy: str = "unknown",
    command: str | None = None,
    code_packet_sha256: str | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "output_config": output_config,
        "is_method_result": bool(is_method_result),
        "is_diagnostic_only": bool(is_diagnostic_only),
        "uses_gt": bool(uses_gt),
        "uses_rgbd_for_prediction": bool(extra.get("uses_rgbd_for_prediction", False)) if extra else False,
        "uses_pose_for_prediction": bool(extra.get("uses_pose_for_prediction", False)) if extra else False,
        "uses_scannet_mesh_for_prediction": bool(extra.get("uses_scannet_mesh_for_prediction", False)) if extra else False,
        "uses_gt_for_prediction": bool(extra.get("uses_gt_for_prediction", False)) if extra else False,
        "uses_gt_sim3_for_prediction": bool(extra.get("uses_gt_sim3_for_prediction", False)) if extra else False,
        "uses_d4rt_self_sim3": bool(extra.get("uses_d4rt_self_sim3", False)) if extra else False,
        "uses_rgbd_for_evaluation": bool(extra.get("uses_rgbd_for_evaluation", False)) if extra else False,
        "uses_gt_for_diagnostic": bool(extra.get("uses_gt_for_diagnostic", bool(uses_gt))) if extra else bool(uses_gt),
        "gt_selected_output": bool(extra.get("gt_selected_output", False)) if extra else False,
        "forbidden_for_method_table": bool(extra.get("forbidden_for_method_table", bool(uses_gt))) if extra else bool(uses_gt),
        "gt_usage": str(gt_usage),
        "alignment_source": str(extra.get("alignment_source", "none")) if extra else "none",
        "alignment_used_for_prediction": bool(extra.get("alignment_used_for_prediction", False)) if extra else False,
        "alignment_used_for_diagnostic": bool(extra.get("alignment_used_for_diagnostic", False)) if extra else False,
        "source_configs": list(source_configs or []),
        "pre_points_policy": str(pre_points_policy),
        "support_policy": str(support_policy),
        "eval_policy": str(extra.get("eval_policy", pre_points_policy)) if extra else str(pre_points_policy),
        "support_source": str(extra.get("support_source", "unknown")) if extra else "unknown",
        "geometry_source": str(extra.get("geometry_source", "rgbd_eval_bridge")) if extra else "rgbd_eval_bridge",
        "chunking_policy": str(extra.get("chunking_policy", "unknown")) if extra else "unknown",
        "opend4rt_reference_policy": str(extra.get("opend4rt_reference_policy", "unknown")) if extra else "unknown",
        "command": command if command is not None else " ".join(sys.argv),
        "code_packet_sha256": code_packet_sha256 or repository_code_hash(root_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return json_safe(payload)


def write_prediction_manifest(
    output_config: str,
    payload: dict[str, Any],
    *,
    root: str | Path = ".",
    pred_suffix: str = "class_agnostic",
) -> list[Path]:
    if not output_config:
        raise ValueError("output_config is required for a prediction manifest")
    payload = dict(payload)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("output_config", output_config)
    written: list[Path] = []
    for path in manifest_paths(root, output_config, pred_suffix):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)
    return written


def load_prediction_manifest(
    root: str | Path,
    output_config: str,
    pred_suffix: str = "class_agnostic",
) -> tuple[dict[str, Any] | None, Path | None]:
    for path in manifest_paths(root, output_config, pred_suffix):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except Exception as exc:
                return {"manifest_parse_error": f"{type(exc).__name__}: {exc}"}, path
    return None, None


def write_retroactive_manifest(
    *,
    root: str | Path,
    output_config: str,
    pred_suffix: str = "class_agnostic",
    uses_gt: bool = False,
    is_method_result: bool = True,
    is_diagnostic_only: bool = False,
    source_configs: list[str] | None = None,
    pre_points_policy: str = "unknown",
    support_policy: str = "unknown",
    notes: str = "",
) -> list[Path]:
    payload = build_prediction_manifest(
        root=root,
        output_config=output_config,
        is_method_result=is_method_result,
        is_diagnostic_only=is_diagnostic_only,
        uses_gt=uses_gt,
        gt_usage="oracle_selection" if uses_gt else "none",
        source_configs=source_configs or [],
        pre_points_policy=pre_points_policy,
        support_policy=support_policy,
        notes=notes or "Retroactive manifest for an existing artifact; verify with metric integrity before reporting.",
        extra={"retroactive_manifest": True},
    )
    return write_prediction_manifest(output_config, payload, root=root, pred_suffix=pred_suffix)


def sha256_prefix(path: str | Path, prefix_len: int = 8) -> str:
    digest = _file_sha256(Path(path))
    return digest[: int(prefix_len)]
