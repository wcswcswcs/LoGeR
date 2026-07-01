#!/usr/bin/env python3
"""Build v81S multi-sequence SWA overlap-pair artifacts from geometry."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair"
)
DEFAULT_GEOMETRY_ROOT = DEFAULT_ROOT / "geometry_prefix_runs"
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")


def _parse_seqs(text: str) -> list[str]:
    return [part.strip().zfill(2) for part in str(text).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow(row)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _geometry_dir_for_seq(root: Path, seq: str, end_frame: int) -> Path:
    return root / f"seq{seq}_native_prefix{int(end_frame)}" / "per_chunk_geometry"


def _semantic_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"semantic_full_pt_exists": False, "semantic_label_maps_available": False, "semantic_confidence_available": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    if not isinstance(sem, dict):
        return {"semantic_full_pt_exists": True, "semantic_label_maps_available": False, "semantic_confidence_available": False}
    labels = sem.get("label_maps")
    conf = sem.get("confidence_maps")
    return {
        "semantic_full_pt_exists": True,
        "semantic_label_maps_available": torch.is_tensor(labels),
        "semantic_confidence_available": torch.is_tensor(conf),
        "semantic_num_frames": int(labels.shape[0]) if torch.is_tensor(labels) else 0,
    }


def _first_pair_conf_projected(pair_dir: Path) -> bool:
    for path in sorted(pair_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            continue
        conf = payload.get("prev_semantic_conf")
        saved = int(payload.get("saved_pair_count", 0) or 0)
        return bool(torch.is_tensor(conf) and conf.numel() == saved and saved > 0)
    return False


def _run_builder(
    *,
    geometry_dir: Path,
    semantic_full_pt: Path,
    out_dir: Path,
    summary_json: Path,
    sample_policy: str,
    max_pairs_per_frame: int,
    min_conf: float,
    overwrite: bool,
) -> dict[str, Any]:
    if not geometry_dir.is_dir():
        return {
            "returncode": None,
            "skipped": True,
            "reason": "missing_per_chunk_geometry",
            "geometry_dir": str(geometry_dir),
        }
    if not semantic_full_pt.is_file():
        return {
            "returncode": None,
            "skipped": True,
            "reason": "missing_semantic_full_pt",
            "semantic_full_pt": str(semantic_full_pt),
        }
    cmd = [
        sys.executable,
        "tools/build_v67_raw_overlap_pairs_from_geometry.py",
        "--geometry-dir",
        str(geometry_dir),
        "--semantic-full-pt",
        str(semantic_full_pt),
        "--out-dir",
        str(out_dir),
        "--summary-json",
        str(summary_json),
        "--max-pairs-per-frame",
        str(max_pairs_per_frame),
        "--min-conf",
        str(min_conf),
        "--sample-policy",
        str(sample_policy),
    ]
    if overwrite:
        cmd.append("--overwrite")
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    log_path = summary_json.with_suffix(".build.log")
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    return {"returncode": int(proc.returncode), "cmd": " ".join(cmd), "log": str(log_path)}


def _gate_row(seq: str, summary: dict[str, Any], semantic: dict[str, Any], pair_dir: Path, build_status: dict[str, Any]) -> dict[str, Any]:
    written = int(summary.get("overlap_pair_files_written") or 0)
    ratio = float(summary.get("semantic_label_projected_pair_ratio") or 0.0)
    median_saved = summary.get("median_saved_pairs_per_overlap")
    median_residual = summary.get("median_raw_residual_rmse")
    conf_projected = bool(_first_pair_conf_projected(pair_dir)) if written else False
    allowed = bool(
        written >= 20
        and ratio >= 0.90
        and median_saved is not None
        and float(median_saved) >= 10000.0
        and median_residual is not None
        and bool(semantic.get("semantic_confidence_available"))
        and conf_projected
    )
    return {
        "seq": seq,
        "geometry_dir": summary.get("geometry_dir") or build_status.get("geometry_dir", ""),
        "semantic_full_pt": summary.get("semantic_full_pt") or "",
        "overlap_pairs_dir": str(pair_dir),
        "build_returncode": build_status.get("returncode"),
        "build_skipped": bool(build_status.get("skipped", False)),
        "build_reason": build_status.get("reason", ""),
        "overlap_pair_files_written": written,
        "semantic_label_projected_pair_ratio": ratio,
        "median_saved_pairs_per_overlap": median_saved,
        "median_raw_residual_rmse": median_residual,
        "overlap_scale_residual_computable": median_residual is not None,
        "semantic_confidence_available": bool(semantic.get("semantic_confidence_available")),
        "stage_c_semantic_confidence_projected": conf_projected,
        "radio_used": False,
        "radio_projection_finite": "",
        "swa_action_allowed": allowed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--geometry-prefix-end-frame", type=int, default=641)
    parser.add_argument(
        "--sample-policy",
        choices=("top_conf", "top_residual", "top_residual_conf_product"),
        default="top_residual_conf_product",
    )
    parser.add_argument("--max-pairs-per-frame", type=int, default=20000)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    seqs = _parse_seqs(args.seqs)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    build_status_by_seq: dict[str, Any] = {}

    for seq in seqs:
        geometry_dir = _geometry_dir_for_seq(args.geometry_root, seq, int(args.geometry_prefix_end_frame))
        semantic_full_pt = args.preprocess_root / seq / "sparse_masklets_with_semantic.pt"
        pair_dir = args.out_root / "overlap_pairs" / seq
        summary_json = args.out_root / "per_seq_build_summaries" / f"seq{seq}_overlap_pair_build_summary.json"
        semantic = _semantic_info(semantic_full_pt)
        build_status = _run_builder(
            geometry_dir=geometry_dir,
            semantic_full_pt=semantic_full_pt,
            out_dir=pair_dir,
            summary_json=summary_json,
            sample_policy=args.sample_policy,
            max_pairs_per_frame=int(args.max_pairs_per_frame),
            min_conf=float(args.min_conf),
            overwrite=bool(args.overwrite),
        )
        build_status_by_seq[seq] = build_status
        summary = _read_json(summary_json)
        summaries[seq] = summary
        rows.append(_gate_row(seq, summary, semantic, pair_dir, build_status))

    allowed = [row["seq"] for row in rows if bool(row.get("swa_action_allowed"))]
    gate = {
        "swa_action_allowed_seqs": allowed,
        "swa_action_allowed_seq_count": len(allowed),
        "includes_seq01": "01" in allowed,
        "includes_one_of_00_02_05": any(seq in allowed for seq in ("00", "02", "05")),
    }
    gate["phaseS1_gate_pass"] = bool(
        gate["swa_action_allowed_seq_count"] >= 3
        and gate["includes_seq01"]
        and gate["includes_one_of_00_02_05"]
    )
    aggregate = {
        "schema": "acl2_v81s_multiseq_swa_overlap_repair_v1",
        "sample_policy": str(args.sample_policy),
        "max_pairs_per_frame": int(args.max_pairs_per_frame),
        "min_conf": float(args.min_conf),
        "rows": rows,
        "gate": gate,
        "build_status_by_seq": build_status_by_seq,
    }
    _write_csv(args.out_root / "overlap_pairs_summary_by_seq.csv", rows)
    _write_csv(args.out_root / "semantic_overlap_projection_audit.csv", rows)
    _write_json(args.out_root / "swa_action_allowed_by_seq.json", aggregate)

    report_lines = [
        "# v81S S1 Multi-Sequence SWA/Overlap Artifact Repair",
        "",
        f"sample_policy: `{args.sample_policy}`",
        f"phaseS1_gate_pass: `{gate['phaseS1_gate_pass']}`",
        f"swa_action_allowed_seqs: `{allowed}`",
        "",
        "## Per-Sequence Gate Rows",
        "",
    ]
    for row in rows:
        report_lines.append(
            "- seq{seq}: allowed={allowed}, files={files}, ratio={ratio}, median_pairs={pairs}, residual={residual}, reason={reason}".format(
                seq=row["seq"],
                allowed=row["swa_action_allowed"],
                files=row["overlap_pair_files_written"],
                ratio=row["semantic_label_projected_pair_ratio"],
                pairs=row["median_saved_pairs_per_overlap"],
                residual=row["median_raw_residual_rmse"],
                reason=row["build_reason"],
            )
        )
    report_lines.extend(
        [
            "",
            "No RADIO projection is claimed in S1; `radio_used=false` for this artifact repair.",
            "Missing geometry or semantic inputs remain explicit blockers and are not backfilled.",
            "",
        ]
    )
    (args.out_root / "missing_artifact_repair_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(_jsonable({"gate": gate, "rows": rows}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
