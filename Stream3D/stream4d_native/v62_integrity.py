from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_ZIP = "code_audit_pack/stream4d_v61_code_audit_packet_20260621_205517.zip"

REQUIRED_ARTIFACTS = [
    "sha256_payload.txt",
    "included_filelist.txt",
    "payload/docs/stream4d_v61_实验结果复盘.md",
    "payload/docs/stream4d_v61_执行日志.md",
    "payload/Stream3D/outputs/audit/v61_final_decision/final_decision.json",
    "payload/Stream3D/outputs/audit/v61_global_embedding/embedding_summary.json",
    "payload/Stream3D/outputs/audit/v61_graph_v3/graph_v3_summary.json",
    "payload/Stream3D/outputs/audit/v61_refinement/refinement_summary.json",
    "payload/Stream3D/outputs/audit/v61_manifold_query/query_summary.json",
    "payload/Stream3D/outputs/audit/v61_stress/stress_summary.json",
    "payload/Stream3D/outputs/audit/v61_native_field/native_field_summary.json",
    "payload/Stream3D/stream4d_native/v61_global_embedding.py",
    "payload/Stream3D/stream4d_native/v61_graph_v3.py",
    "payload/Stream3D/tests/test_v61_global_embedding.py",
    "payload/Stream3D/tests/test_v61_graph_v3.py",
]


LOCAL_SUMMARIES = {
    "final_decision": "outputs/audit/v61_final_decision/final_decision.json",
    "global_embedding": "outputs/audit/v61_global_embedding/embedding_summary.json",
    "graph_v3": "outputs/audit/v61_graph_v3/graph_v3_summary.json",
    "refinement": "outputs/audit/v61_refinement/refinement_summary.json",
    "query": "outputs/audit/v61_manifold_query/query_summary.json",
    "stress": "outputs/audit/v61_stress/stress_summary.json",
    "native_field": "outputs/audit/v61_native_field/native_field_summary.json",
}


@dataclass(frozen=True)
class V62IntegrityConfig:
    zip_path: str | Path = DEFAULT_ZIP
    output_root: str | Path = "outputs/audit/v62_phase0_integrity"
    visualization_root: str | Path = "outputs/audit/v62_visualizations/phase0"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _project_repo_input(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    stream3d_path = ROOT / path_obj
    if stream3d_path.exists():
        return stream3d_path
    return ROOT.parent / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_integrity(config: V62IntegrityConfig | None = None) -> dict[str, Any]:
    cfg = config or V62IntegrityConfig()
    zip_path = _project_repo_input(cfg.zip_path)
    zip_exists = zip_path.exists()
    zip_unzip_test_pass = False
    zip_first_bad_file: str | None = None
    entries: list[str] = []
    zip_sha256: str | None = None
    if zip_exists:
        zip_sha256 = _sha256(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                entries = archive.namelist()
                zip_first_bad_file = archive.testzip()
                zip_unzip_test_pass = zip_first_bad_file is None
        except zipfile.BadZipFile as exc:
            zip_first_bad_file = f"BadZipFile:{exc}"
            zip_unzip_test_pass = False

    required_rows: list[dict[str, Any]] = []
    missing_required = 0
    for suffix in REQUIRED_ARTIFACTS:
        included = any(entry.endswith(suffix) for entry in entries)
        if not included:
            missing_required += 1
        required_rows.append(
            {
                "required_suffix": suffix,
                "zip_included": included,
                "matched_entry": next((entry for entry in entries if entry.endswith(suffix)), ""),
            }
        )

    summaries = {name: _read_local_summary(path) for name, path in LOCAL_SUMMARIES.items()}
    final = summaries["final_decision"]
    embedding = summaries["global_embedding"]
    query = summaries["query"]
    refinement = summaries["refinement"]
    native_field = summaries["native_field"]
    graph = summaries["graph_v3"]
    stress = summaries["stress"]

    final_gate = final.get("go_gate") or {}
    summary = {
        "phase": "v62_phase0_integrity",
        "created_at": utc_now(),
        "zip_path": _rel(zip_path),
        "zip_exists": zip_exists,
        "zip_sha256": zip_sha256,
        "zip_unzip_test_pass": zip_unzip_test_pass,
        "zip_first_bad_file": zip_first_bad_file,
        "included_file_count": len(entries),
        "missing_required_artifact_count": missing_required,
        "final_decision_label": final.get("decision_label"),
        "go_gate_pass": bool(final_gate.get("pass")),
        "global_embedding_gate_pass": bool((embedding.get("gate") or {}).get("pass")),
        "graph_v3_candidate_gate_pass": bool((graph.get("gate") or {}).get("pass")),
        "query_gate_pass": bool((query.get("gate") or {}).get("pass")),
        "refinement_gate_pass": bool((refinement.get("gate") or {}).get("pass")),
        "stress_gate_pass": bool((stress.get("gate") or {}).get("pass")),
        "native_field_gate_pass": bool((native_field.get("gate") or {}).get("pass")),
        "ap_diagnostic_status": native_field.get("ap_diagnostic_status"),
        "blocked_claims": final.get("blocked_claims", []),
        "key_metrics": final.get("key_metrics", {}),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "gate": {},
        "input_paths": {"zip": _rel(zip_path), **LOCAL_SUMMARIES},
    }
    gate = {
        "zip_unzip_test_pass": zip_unzip_test_pass,
        "missing_required_artifact_count_eq_0": missing_required == 0,
        "final_decision_label_is_GO": final.get("decision_label") == "GO_SOMA_MANIFOLD_GLOBAL_EMBEDDING",
        "global_embedding_gate_pass": summary["global_embedding_gate_pass"],
        "query_gate_pass_false_as_expected": summary["query_gate_pass"] is False,
        "refinement_gate_pass_false_as_expected": summary["refinement_gate_pass"] is False,
        "native_field_gate_pass": summary["native_field_gate_pass"],
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {"summary": summary, "included_artifact_rows": required_rows}


def write_v62_integrity(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "integrity_summary": root / "integrity_summary.json",
        "included_artifact_rows": root / "included_artifact_rows.csv",
    }
    write_json(paths["integrity_summary"], result["summary"])
    write_csv(paths["included_artifact_rows"], result["included_artifact_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v62_integrity_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summary = result["summary"]
        labels = ["zip", "required", "final", "global", "query_false", "refine_false", "native"]
        values = [
            bool(summary["zip_unzip_test_pass"]),
            summary["missing_required_artifact_count"] == 0,
            summary["final_decision_label"] == "GO_SOMA_MANIFOLD_GLOBAL_EMBEDDING",
            summary["global_embedding_gate_pass"],
            summary["query_gate_pass"] is False,
            summary["refinement_gate_pass"] is False,
            summary["native_field_gate_pass"],
        ]
        path = root / "v61_integrity_dashboard.png"
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        ax.bar(labels, [1 if value else 0 for value in values], color=["#2A9D8F" if value else "#B56576" for value in values])
        ax.set_ylim(0, 1.2)
        ax.set_title("v62 Phase 0 v61 package integrity")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"v61_integrity_dashboard": _rel(path), "visualization_status": "created"}
    except Exception as exc:  # pragma: no cover
        error_path = root / "v62_integrity_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _read_local_summary(path: str) -> dict[str, Any]:
    full_path = ROOT / path
    return read_json(full_path) if full_path.exists() else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

