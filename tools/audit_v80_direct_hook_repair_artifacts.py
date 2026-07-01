#!/usr/bin/env python3
"""Audit ACL2 v80 Phase2 direct-hook repair artifacts.

This tool is intentionally conservative: it reports direct hook evidence only
when a concrete artifact file exists and its payload can be inspected. Missing
files and unavailable taps remain missing; no visual/action readiness is
inferred from proxy panels alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TTT_REQUIRED_TAPS = [
    "pca_ttt_operator_output_layers",
    "pca_ttt_update_term_layers",
    "pca_ttt_final_output_layers",
]

READ_REQUIRED_TAPS = [
    "pca_attn_global_k_layers",
    "pca_attn_global_v_layers",
    "pca_attn_frame_v_layers",
]

SWA_REQUIRED_TAPS = [
    "pca_swa_current_q_layers",
    "pca_swa_current_k_layers",
    "pca_swa_current_v_layers",
    "pca_swa_cache_k_layers",
    "pca_swa_cache_v_layers",
]


def _load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import torch

        if torch.is_tensor(value):
            return {
                "shape": [int(x) for x in value.shape],
                "dtype": str(value.dtype),
            }
    except Exception:
        pass
    return value


def _chunk_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.search(r"chunk[_]?(\d{3})", part)
        if match:
            return int(match.group(1))
    match = re.search(r"chunk[_]?(\d{3})", path.name)
    if match:
        return int(match.group(1))
    return None


def _seq_from_path(path: Path) -> str:
    text = str(path)
    for pattern in (r"seq[_-]?(\d{2})", r"/(\d{2})/"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "unknown"


def _tap_status(payload: Dict[str, Any], required: Sequence[str]) -> Dict[str, Any]:
    taps = payload.get("taps")
    if not isinstance(taps, dict):
        return {
            "available_taps": [],
            "missing_taps": list(required),
            "tap_shapes": {},
        }
    available: List[str] = []
    missing: List[str] = []
    shapes: Dict[str, Any] = {}
    for tap in required:
        meta = taps.get(tap)
        if isinstance(meta, dict) and bool(meta.get("available")):
            available.append(tap)
            shapes[tap] = {
                "original_shape": meta.get("original_shape"),
                "saved_shape": meta.get("saved_shape"),
                "selected_layer_ids": meta.get("selected_layer_ids"),
                "reason": meta.get("reason"),
            }
        else:
            missing.append(tap)
            if isinstance(meta, dict):
                shapes[tap] = {
                    "original_shape": meta.get("original_shape"),
                    "saved_shape": meta.get("saved_shape"),
                    "selected_layer_ids": meta.get("selected_layer_ids"),
                    "reason": meta.get("reason"),
                }
    return {
        "available_taps": available,
        "missing_taps": missing,
        "tap_shapes": shapes,
    }


def _audit_pca(path: Path, required: Sequence[str], group: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "artifact_group": group,
        "artifact_type": "v68_layer_pca_feature_dump",
        "path": str(path),
        "seq": _seq_from_path(path),
        "exists": path.exists(),
        "chunk": _chunk_from_path(path),
        "schema": "",
        "status": "missing",
        "available_taps": "",
        "missing_taps": ",".join(required),
        "detail_json": "",
        "error": "",
    }
    if not path.exists():
        return row
    try:
        payload = _load_torch(path)
        if not isinstance(payload, dict):
            row["status"] = "invalid_payload"
            row["error"] = f"payload_type={type(payload).__name__}"
            return row
        row["schema"] = str(payload.get("schema", ""))
        status = _tap_status(payload, required)
        row["available_taps"] = ",".join(status["available_taps"])
        row["missing_taps"] = ",".join(status["missing_taps"])
        row["detail_json"] = json.dumps(_jsonable(status["tap_shapes"]), sort_keys=True)
        row["status"] = "complete" if not status["missing_taps"] else "partial"
        return row
    except Exception as exc:
        row["status"] = "load_failed"
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row


def _audit_read_cue(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "artifact_group": "read",
        "artifact_type": "read_cue_patch_dump",
        "path": str(path),
        "seq": _seq_from_path(path),
        "exists": path.exists(),
        "chunk": _chunk_from_path(path),
        "schema": "",
        "status": "missing",
        "available_taps": "",
        "missing_taps": "",
        "detail_json": "",
        "error": "",
    }
    if not path.exists():
        return row
    try:
        payload = _load_torch(path)
        if not isinstance(payload, dict):
            row["status"] = "invalid_payload"
            row["error"] = f"payload_type={type(payload).__name__}"
            return row
        row["schema"] = str(payload.get("schema", ""))
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        keys = sorted(k for k in payload.keys() if not str(k).startswith("_"))
        row["available_taps"] = ",".join(keys)
        row["detail_json"] = json.dumps(_jsonable({"stats": stats, "keys": keys}), sort_keys=True)
        row["status"] = "complete" if row["schema"] else "partial"
        return row
    except Exception as exc:
        row["status"] = "load_failed"
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row


def _audit_ttt_spatial(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "artifact_group": "ttt",
        "artifact_type": "ttt_spatial_post_delta_map",
        "path": str(path),
        "seq": _seq_from_path(path),
        "exists": path.exists(),
        "chunk": _chunk_from_path(path),
        "schema": "",
        "status": "missing",
        "available_taps": "",
        "missing_taps": "",
        "detail_json": "",
        "error": "",
    }
    if not path.exists():
        return row
    try:
        payload = _load_torch(path)
        if not isinstance(payload, dict):
            row["status"] = "invalid_payload"
            row["error"] = f"payload_type={type(payload).__name__}"
            return row
        row["schema"] = str(payload.get("schema", ""))
        keys = sorted(k for k in payload.keys() if not str(k).startswith("_"))
        row["available_taps"] = ",".join(keys)
        detail = {
            "keys": keys,
            "chunk_idx": payload.get("chunk_idx"),
            "global_chunk_idx": payload.get("global_chunk_idx"),
            "stats": payload.get("stats"),
        }
        row["detail_json"] = json.dumps(_jsonable(detail), sort_keys=True)
        row["status"] = "complete" if row["schema"] else "partial"
        return row
    except Exception as exc:
        row["status"] = "load_failed"
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row


def _discover_rows(repair_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pca in sorted(repair_root.glob("**/pca_features/chunk_*.pt")):
        text = str(pca).lower()
        groups: List[tuple[str, Sequence[str]]] = []
        if "ttt" in text or "t0_native_pca" in text:
            groups.append(("ttt", TTT_REQUIRED_TAPS))
        if "read" in text:
            groups.append(("read", READ_REQUIRED_TAPS))
        if "swa" in text:
            groups.append(("swa", SWA_REQUIRED_TAPS))
        if groups:
            for group, required in groups:
                rows.append(_audit_pca(pca, required, group))
        else:
            required = sorted(set(TTT_REQUIRED_TAPS + READ_REQUIRED_TAPS + SWA_REQUIRED_TAPS))
            rows.append(_audit_pca(pca, required, "unknown_pca"))
    for path in sorted(repair_root.glob("**/read_cue_patch_dumps/*.pt")):
        rows.append(_audit_read_cue(path))
    for path in sorted(repair_root.glob("**/ttt_spatial_post_delta_maps/*.pt")):
        rows.append(_audit_ttt_spatial(path))
    return rows


def _case_chunks(case_bank_dir: Optional[Path]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"seq01_short_bad": [], "seq01_mid_bad": [], "seq01_long_bad": []}
    if case_bank_dir is None:
        return out
    short = case_bank_dir / "short_single_chunk_cases.csv"
    mid = case_bank_dir / "mid_adjacent_pair_cases.csv"
    long = case_bank_dir / "long_five_chunk_cases.csv"
    if short.exists():
        with short.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("seq") == "01" and row.get("case_type") == "bad" and row.get("chunk_id"):
                    out["seq01_short_bad"].append(int(row["chunk_id"]))
    if mid.exists():
        with mid.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("seq") == "01" and row.get("case_type") == "bad":
                    for key in ("prev_chunk", "curr_chunk"):
                        if row.get(key):
                            out["seq01_mid_bad"].append(int(row[key]))
    if long.exists():
        with long.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("seq") == "01" and row.get("case_type") == "bad":
                    start = int(row["chunk_start"])
                    end = int(row["chunk_end"])
                    out["seq01_long_bad"].extend(range(start, end + 1))
    return {key: sorted(set(vals)) for key, vals in out.items()}


def _case_requirements(case_bank_dir: Optional[Path]) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = {}
    if case_bank_dir is None:
        return out

    def _ensure(seq: str) -> Dict[str, List[int]]:
        seq = str(seq).zfill(2)
        if seq not in out:
            out[seq] = {
                "short_read_pca": [],
                "short_read_cue_patch": [],
                "short_swa_pca": [],
                "mid_swa_pca": [],
                "long_ttt_pca": [],
                "long_ttt_spatial_post_delta": [],
            }
        return out[seq]

    short = case_bank_dir / "short_single_chunk_cases.csv"
    mid = case_bank_dir / "mid_adjacent_pair_cases.csv"
    long = case_bank_dir / "long_five_chunk_cases.csv"
    if short.exists():
        with short.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seq_req = _ensure(row.get("seq", ""))
                chunk = int(row["chunk_id"])
                seq_req["short_read_pca"].append(chunk)
                seq_req["short_read_cue_patch"].append(chunk)
                seq_req["short_swa_pca"].append(chunk)
    if mid.exists():
        with mid.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seq_req = _ensure(row.get("seq", ""))
                for key in ("prev_chunk", "curr_chunk"):
                    seq_req["mid_swa_pca"].append(int(row[key]))
    if long.exists():
        with long.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seq_req = _ensure(row.get("seq", ""))
                for chunk in range(int(row["chunk_start"]), int(row["chunk_end"]) + 1):
                    seq_req["long_ttt_pca"].append(chunk)
                    seq_req["long_ttt_spatial_post_delta"].append(chunk)
    return {
        seq: {name: sorted(set(chunks)) for name, chunks in reqs.items()}
        for seq, reqs in out.items()
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "artifact_group",
        "artifact_type",
        "seq",
        "chunk",
        "status",
        "schema",
        "exists",
        "available_taps",
        "missing_taps",
        "path",
        "detail_json",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _complete_chunks_for(rows: Sequence[Dict[str, Any]], *, group: str, artifact_type: str) -> List[int]:
    chunks: List[int] = []
    for row in rows:
        if row.get("artifact_group") != group:
            continue
        if row.get("artifact_type") != artifact_type:
            continue
        if row.get("status") != "complete":
            continue
        chunk = row.get("chunk")
        if chunk is not None:
            chunks.append(int(chunk))
    return sorted(set(chunks))


def _complete_chunks_for_seq(rows: Sequence[Dict[str, Any]], *, seq: str, group: str, artifact_type: str) -> List[int]:
    chunks: List[int] = []
    for row in rows:
        if row.get("seq") != seq:
            continue
        if row.get("artifact_group") != group:
            continue
        if row.get("artifact_type") != artifact_type:
            continue
        if row.get("status") != "complete":
            continue
        chunk = row.get("chunk")
        if chunk is not None:
            chunks.append(int(chunk))
    return sorted(set(chunks))


def _coverage(required: Sequence[int], available: Sequence[int]) -> Dict[str, Any]:
    required_set = set(int(x) for x in required)
    available_set = set(int(x) for x in available)
    covered = sorted(required_set & available_set)
    missing = sorted(required_set - available_set)
    return {
        "required_chunks": sorted(required_set),
        "covered_chunks": covered,
        "missing_chunks": missing,
        "coverage": (float(len(covered)) / float(len(required_set))) if required_set else None,
        "gate_pass": not missing if required_set else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--case-bank-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _discover_rows(args.repair_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "direct_hook_repair_audit.csv"
    _write_csv(csv_path, rows)

    by_type: Dict[str, Dict[str, int]] = {}
    complete_by_group: Dict[str, int] = {}
    chunks_by_group: Dict[str, List[int]] = {}
    for row in rows:
        artifact_type = str(row.get("artifact_type"))
        status = str(row.get("status"))
        group = str(row.get("artifact_group"))
        by_type.setdefault(artifact_type, {})
        by_type[artifact_type][status] = by_type[artifact_type].get(status, 0) + 1
        if status == "complete":
            complete_by_group[group] = complete_by_group.get(group, 0) + 1
            chunk = row.get("chunk")
            if chunk is not None:
                chunks_by_group.setdefault(group, []).append(int(chunk))

    case_chunks = _case_chunks(args.case_bank_dir)
    case_requirements = _case_requirements(args.case_bank_dir)
    read_pca_chunks = _complete_chunks_for(rows, group="read", artifact_type="v68_layer_pca_feature_dump")
    read_cue_chunks = _complete_chunks_for(rows, group="read", artifact_type="read_cue_patch_dump")
    swa_pca_chunks = _complete_chunks_for(rows, group="swa", artifact_type="v68_layer_pca_feature_dump")
    ttt_pca_chunks = _complete_chunks_for(rows, group="ttt", artifact_type="v68_layer_pca_feature_dump")
    ttt_spatial_chunks = _complete_chunks_for(rows, group="ttt", artifact_type="ttt_spatial_post_delta_map")
    all_case_coverage_by_seq: Dict[str, Any] = {}
    for seq, reqs in case_requirements.items():
        seq_read_pca = _complete_chunks_for_seq(rows, seq=seq, group="read", artifact_type="v68_layer_pca_feature_dump")
        seq_read_cue = _complete_chunks_for_seq(rows, seq=seq, group="read", artifact_type="read_cue_patch_dump")
        seq_swa = _complete_chunks_for_seq(rows, seq=seq, group="swa", artifact_type="v68_layer_pca_feature_dump")
        seq_ttt_pca = _complete_chunks_for_seq(rows, seq=seq, group="ttt", artifact_type="v68_layer_pca_feature_dump")
        seq_ttt_spatial = _complete_chunks_for_seq(rows, seq=seq, group="ttt", artifact_type="ttt_spatial_post_delta_map")
        all_case_coverage_by_seq[seq] = {
            "short_read_pca": _coverage(reqs.get("short_read_pca", []), seq_read_pca),
            "short_read_cue_patch": _coverage(reqs.get("short_read_cue_patch", []), seq_read_cue),
            "short_swa_pca": _coverage(reqs.get("short_swa_pca", []), seq_swa),
            "mid_swa_pca": _coverage(reqs.get("mid_swa_pca", []), seq_swa),
            "long_ttt_pca": _coverage(reqs.get("long_ttt_pca", []), seq_ttt_pca),
            "long_ttt_spatial_post_delta": _coverage(reqs.get("long_ttt_spatial_post_delta", []), seq_ttt_spatial),
        }

    summary = {
        "schema": "acl2_v80_phase2_direct_hook_repair_audit_v1",
        "repair_root": str(args.repair_root),
        "artifact_rows": len(rows),
        "status_by_artifact_type": by_type,
        "complete_by_group": complete_by_group,
        "complete_chunks_by_group": {
            key: sorted(set(vals)) for key, vals in chunks_by_group.items()
        },
        "complete_chunks_by_artifact": {
            "read_pca": read_pca_chunks,
            "read_cue_patch": read_cue_chunks,
            "swa_pca": swa_pca_chunks,
            "ttt_pca": ttt_pca_chunks,
            "ttt_spatial_post_delta": ttt_spatial_chunks,
        },
        "case_bank_seq01_bad_chunks": case_chunks,
        "case_bank_all_case_requirements": case_requirements,
        "seq01_bad_case_coverage": {
            "short_read_pca": _coverage(case_chunks.get("seq01_short_bad", []), read_pca_chunks),
            "short_read_cue_patch": _coverage(case_chunks.get("seq01_short_bad", []), read_cue_chunks),
            "short_swa_pca": _coverage(case_chunks.get("seq01_short_bad", []), swa_pca_chunks),
            "mid_swa_pca": _coverage(case_chunks.get("seq01_mid_bad", []), swa_pca_chunks),
            "long_ttt_pca": _coverage(case_chunks.get("seq01_long_bad", []), ttt_pca_chunks),
            "long_ttt_spatial_post_delta": _coverage(case_chunks.get("seq01_long_bad", []), ttt_spatial_chunks),
        },
        "all_case_coverage_by_seq": all_case_coverage_by_seq,
        "csv": str(csv_path),
    }
    summary_path = args.out_dir / "direct_hook_repair_audit_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
