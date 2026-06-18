#!/usr/bin/env python3
"""ACL2 v66B dense semantic scale / harmful-region diagnostics.

This diagnostic is deliberately conservative:

* Phase 0 is a hard cache-integrity gate for dense semantic label maps.
* Phase 1/2 are passive projection/correlation audits.
* Phase 3/4 use pointmap-region fitting only when per-chunk geometry exists.
  Missing geometry is reported as unavailable; no semantic causality is
  inferred from pose-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_SEMANTIC = ROOT / "results/kitti_preprocess/01/sparse_masklets_with_semantic.pt"
DEFAULT_STAGE_C_CACHE = ROOT / "results/kitti_preprocess/01/stage_c_cache_semantic_chunks"
DEFAULT_H35_RUN = (
    ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_OUT = ROOT / "results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final"
V62_OUT = ROOT / "results/kitti01_hmc_v2/acl2_v62_kitti01_error_source_autopsy_orig_c9_h35/report_final"
TOKEN_TYPE_PATCH = 2
EPS = 1e-12


GROUPS = {
    "dynamic": {"car", "truck", "bus", "van", "person", "people", "rider", "cyclist", "bicycle", "motorcycle", "animal"},
    "sky": {"sky", "cloud"},
    "vegetation": {"vegetation", "tree", "grass", "plant", "terrain", "mountain"},
    "vertical_static": {
        "building",
        "house",
        "wall",
        "fence",
        "railing",
        "pole",
        "traffic sign",
        "traffic light",
        "bridge",
        "billboard",
        "handrail",
    },
    "ground_static": {"road", "ground", "sidewalk", "floor", "crosswalk"},
    "void_lowtrust": {"void", "unknown", "unlabeled"},
}
GROUP_ORDER = ["dynamic", "sky", "vegetation", "vertical_static", "ground_static", "void_lowtrust"]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in fields:
                value = _clean(row.get(key))
                if value is None:
                    out[key] = ""
                elif isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    out[key] = value
            writer.writerow(out)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    vals: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            vals.append(val)
    return vals


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.median(vals)) if vals else None


def _corr(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = _safe_float(x)
        yf = _safe_float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if float(np.std(arr[:, 0])) <= EPS or float(np.std(arr[:, 1])) <= EPS:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def _normalise_label(label: str) -> str:
    out = str(label).strip().lower().replace("_", " ").replace("/", " ")
    if "fence" in out or "handrail" in out:
        return "fence"
    if "billboard" in out or "bulletin" in out:
        return "billboard"
    if out == "house":
        return "house"
    if out == "mountain":
        return "mountain"
    if out == "other construction":
        return "building"
    return out


def _label_group_ids(label_names: Sequence[str]) -> Dict[str, List[int]]:
    out = {group: [] for group in GROUP_ORDER}
    missing = {group: sorted(list(names)) for group, names in GROUPS.items()}
    for idx, raw in enumerate(label_names):
        name = _normalise_label(str(raw))
        matched = False
        for group, names in GROUPS.items():
            if name in names:
                out[group].append(int(idx))
                matched = True
                if name in missing[group]:
                    missing[group].remove(name)
        if not matched and idx == 0:
            out["void_lowtrust"].append(int(idx))
    return {**out, "missing_labels_by_group": missing}  # type: ignore[dict-item]


def _isin_label(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros_like(labels, dtype=torch.bool)
    ids_t = torch.tensor(sorted(int(x) for x in ids), dtype=labels.dtype, device=labels.device)
    return (labels[..., None] == ids_t).any(dim=-1)


def _project_label_maps(label_maps: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(label_maps[:, None].float(), size=size, mode="nearest").squeeze(1).long()


def _top_labels(label_maps: torch.Tensor, label_names: Sequence[str], limit: int = 10) -> List[Dict[str, Any]]:
    vals, counts = torch.unique(label_maps.reshape(-1), return_counts=True)
    pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda item: int(item[1]), reverse=True)
    out: List[Dict[str, Any]] = []
    total = float(label_maps.numel()) if label_maps.numel() else 1.0
    for label_id, count in pairs[:limit]:
        name = label_names[int(label_id)] if 0 <= int(label_id) < len(label_names) else f"id_{label_id}"
        out.append({"id": int(label_id), "name": str(name), "pixels": int(count), "ratio": float(count) / total})
    return out


def _group_ratios(labels: torch.Tensor, group_ids: Mapping[str, Sequence[int]]) -> Dict[str, float]:
    total = float(labels.numel()) if labels.numel() else 1.0
    return {
        f"{group}_ratio": float(_isin_label(labels, group_ids.get(group, [])).sum().item()) / total
        for group in GROUP_ORDER
    }


def _read_cache_rows(stage_c_cache: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(stage_c_cache / "cache_index.jsonl")
    rows.sort(key=lambda row: int(row.get("chunk_idx", row.get("chunk_id", 0))))
    return rows


def _parse_target_chunks(value: str, all_ids: Sequence[int]) -> List[int]:
    if value.strip().lower() == "all":
        return list(all_ids)
    out: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def phase0_cache_audit(full_semantic_pt: Path, stage_c_cache: Path, out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[int]]]:
    phase_dir = out_dir / "phase0_cache_audit"
    data = torch.load(full_semantic_pt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected dict full payload: {full_semantic_pt}")
    sem = data.get("semantic_segmentation")
    if not isinstance(sem, dict) or "label_maps" not in sem:
        raise SystemExit("Full payload missing semantic_segmentation.label_maps; v66B cannot continue")

    full_maps = sem["label_maps"].detach().cpu() if torch.is_tensor(sem["label_maps"]) else torch.as_tensor(sem["label_maps"])
    label_names = [str(x) for x in sem.get("label_names", [])]
    label_to_id = sem.get("label_to_id", {})
    group_ids = _label_group_ids(label_names)
    rows = _read_cache_rows(stage_c_cache)
    audit_rows: List[Dict[str, Any]] = []
    for fallback_idx, row in enumerate(rows):
        chunk_name = str(row.get("chunk", ""))
        chunk_idx = int(row.get("chunk_idx", fallback_idx))
        start = int(row.get("start_frame", 0))
        end = int(row.get("end_frame", start))
        masklet_pt = stage_c_cache / chunk_name / "masklet.pt"
        audit: Dict[str, Any] = {
            "chunk_id": chunk_idx,
            "chunk_name": chunk_name,
            "start_frame": start,
            "end_frame": end,
            "masklet_pt_exists": masklet_pt.is_file(),
            "has_semantic_segmentation": False,
            "semantic_shape": None,
            "global_start_frame": None,
            "global_end_frame": None,
            "slice_equal_full": False,
            "diff_pixels": None,
            "nonvoid_pixels": None,
            "nonvoid_ratio": None,
            "num_labels_present": None,
            "top10_labels": None,
        }
        if masklet_pt.is_file():
            chunk = torch.load(masklet_pt, map_location="cpu", weights_only=False)
            csem = chunk.get("semantic_segmentation", {}) if isinstance(chunk, dict) else {}
            audit["has_semantic_segmentation"] = isinstance(csem, dict) and "label_maps" in csem
            if audit["has_semantic_segmentation"]:
                cmaps = csem["label_maps"].detach().cpu() if torch.is_tensor(csem["label_maps"]) else torch.as_tensor(csem["label_maps"])
                expected = full_maps[start:end]
                diff_pixels = int((cmaps != expected).sum().item()) if tuple(cmaps.shape) == tuple(expected.shape) else None
                audit.update(
                    {
                        "semantic_shape": list(cmaps.shape),
                        "global_start_frame": int(csem.get("global_start_frame", -1)),
                        "global_end_frame": int(csem.get("global_end_frame", -1)),
                        "slice_equal_full": bool(diff_pixels == 0),
                        "diff_pixels": diff_pixels,
                        "nonvoid_pixels": int((cmaps != 0).sum().item()),
                        "nonvoid_ratio": float((cmaps != 0).float().mean().item()) if cmaps.numel() > 0 else None,
                        "num_labels_present": int(torch.unique(cmaps).numel()),
                        "top10_labels": _top_labels(cmaps, label_names, limit=10),
                    }
                )
        audit_rows.append(audit)

    nonvoid = [_safe_float(row.get("nonvoid_ratio")) for row in audit_rows]
    gate = {
        "full_payload_format": data.get("format"),
        "full_has_semantic_segmentation": isinstance(sem, dict),
        "full_semantic_format": sem.get("format"),
        "full_label_maps_shape": list(full_maps.shape),
        "full_label_maps_dtype": str(full_maps.dtype),
        "full_num_frames": int(data.get("num_frames", full_maps.shape[0])),
        "label_names_0": label_names[0] if label_names else None,
        "label_to_id_reverse_ok": all(str(label_to_id.get(name, idx)) == str(idx) or label_to_id.get(name) == idx for idx, name in enumerate(label_names)),
        "stage_c_cache": str(stage_c_cache),
        "num_chunks": int(len(audit_rows)),
        "masklet_exists_all": all(bool(row["masklet_pt_exists"]) for row in audit_rows),
        "has_semantic_all": all(bool(row["has_semantic_segmentation"]) for row in audit_rows),
        "slice_equal_full_all": all(bool(row["slice_equal_full"]) for row in audit_rows),
        "median_nonvoid_ratio": _median(nonvoid),
    }
    gate["gate_pass"] = bool(
        gate["full_payload_format"] == "sparse_masklets_v1"
        and gate["full_has_semantic_segmentation"]
        and gate["full_semantic_format"] == "semantic_label_maps_v1"
        and gate["num_chunks"] == 38
        and gate["masklet_exists_all"]
        and gate["has_semantic_all"]
        and gate["slice_equal_full_all"]
        and _safe_float(gate["median_nonvoid_ratio"]) >= 0.50
    )

    _write_csv(phase_dir / "semantic_cache_audit.csv", audit_rows)
    _write_json(phase_dir / "semantic_cache_gate.json", gate)
    lines = [
        "# Phase 0 Semantic Cache Audit",
        "",
        f"gate_pass: {gate['gate_pass']}",
        f"num_chunks: {gate['num_chunks']}",
        f"masklet_exists_all: {gate['masklet_exists_all']}",
        f"has_semantic_all: {gate['has_semantic_all']}",
        f"slice_equal_full_all: {gate['slice_equal_full_all']}",
        f"median_nonvoid_ratio: {gate['median_nonvoid_ratio']}",
        "",
        "This phase is a hard gate. If `gate_pass` is false, later semantic experiments are invalid for v66B.",
    ]
    _write_text(phase_dir / "semantic_cache_audit.md", lines)
    _write_json(phase_dir / "semantic_group_mapping.json", {k: v for k, v in group_ids.items()})
    return gate, audit_rows, group_ids  # type: ignore[return-value]


def _infer_patch_grid(token_type: Optional[torch.Tensor], T: int, H_p: int, W_p: int) -> Tuple[int, int]:
    if token_type is not None and T > 0:
        patch_count = int((token_type.detach().cpu().long() == TOKEN_TYPE_PATCH).sum().item())
        per_frame = patch_count // max(T, 1)
        if per_frame > 0:
            ratio = float(W_p) / max(float(H_p), 1.0)
            best: Optional[Tuple[float, int, int]] = None
            for h in range(1, int(math.sqrt(per_frame)) + 2):
                if per_frame % h != 0:
                    continue
                w = per_frame // h
                score = abs((float(w) / max(float(h), 1.0)) - ratio)
                if best is None or score < best[0]:
                    best = (score, h, w)
            if best is not None:
                return int(best[1]), int(best[2])
    return max(1, H_p // 14), max(1, W_p // 14)


def _load_geometry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data if isinstance(data, dict) else {}


def phase1_projection_audit(
    audit_rows: Sequence[Mapping[str, Any]],
    stage_c_cache: Path,
    per_chunk_geometry_dir: Optional[Path],
    group_ids: Mapping[str, Sequence[int]],
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    phase_dir = out_dir / "phase1_projection_audit"
    rows: List[Dict[str, Any]] = []
    label_names: List[str] = []
    for row in audit_rows:
        chunk_id = int(row["chunk_id"])
        chunk_name = str(row["chunk_name"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        chunk = torch.load(stage_c_cache / chunk_name / "masklet.pt", map_location="cpu", weights_only=False)
        sem = chunk["semantic_segmentation"]
        label_maps = sem["label_maps"].detach().cpu().long()
        if not label_names:
            label_names = [str(x) for x in sem.get("label_names", [])]
        T, H, W = int(label_maps.shape[0]), int(label_maps.shape[1]), int(label_maps.shape[2])

        geo_path = per_chunk_geometry_dir / f"chunk_{chunk_id:03d}.pt" if per_chunk_geometry_dir is not None else Path("__missing__")
        geo = _load_geometry(geo_path)
        conf = geo.get("conf")
        if conf is None:
            conf = geo.get("confidence")
        has_pointmap = torch.is_tensor(conf)
        H_p, W_p = (int(conf.shape[-2]), int(conf.shape[-1])) if has_pointmap else (H, W)
        H_tok, W_tok = _infer_patch_grid(geo.get("token_type") if torch.is_tensor(geo.get("token_type")) else None, T, H_p, W_p)
        label_point = _project_label_maps(label_maps, (H_p, W_p))
        label_patch = _project_label_maps(label_maps, (H_tok, W_tok))
        point_ratios = _group_ratios(label_point, group_ids)
        patch_ratios = _group_ratios(label_patch, group_ids)
        rec: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "start_frame": start,
            "end_frame": end,
            "H": H,
            "W": W,
            "H_p": H_p,
            "W_p": W_p,
            "H_tok": H_tok,
            "W_tok": W_tok,
            "pointmap_available": bool(has_pointmap),
            "geometry_path": str(geo_path) if has_pointmap else None,
            "semantic_source": "dense_label_maps",
            "patch_purity": "unavailable_nearest_proxy",
        }
        for group in GROUP_ORDER:
            rec[f"point_{group}_ratio"] = point_ratios[f"{group}_ratio"]
            rec[f"patch_{group}_ratio"] = patch_ratios[f"{group}_ratio"]
        if has_pointmap:
            conf_t = conf.detach().cpu().float().clamp(0.0, 1.0)
            if int(conf_t.shape[0]) != T:
                conf_t = conf_t[:T]
                label_point = label_point[: int(conf_t.shape[0])]
            d_geo = (1.0 - conf_t).clamp(0.0, 1.0)
            for group in GROUP_ORDER:
                mask = _isin_label(label_point, group_ids.get(group, []))
                if bool(mask.any().item()):
                    rec[f"conf_mean_by_{group}"] = float(conf_t[mask].mean().item())
                    rec[f"D_geo_mean_by_{group}"] = float(d_geo[mask].mean().item())
                else:
                    rec[f"conf_mean_by_{group}"] = None
                    rec[f"D_geo_mean_by_{group}"] = None
        rows.append(rec)

    nonempty = [row for row in rows if _safe_float(row.get("point_void_lowtrust_ratio")) < 0.999]
    summary = {
        "chunk_count": len(rows),
        "pointmap_available_count": sum(1 for row in rows if row.get("pointmap_available")),
        "projection_nonempty_count": len(nonempty),
        "projection_nonempty_ratio": float(len(nonempty)) / max(float(len(rows)), 1.0),
        "gate_pass": bool(len(rows) > 0 and float(len(nonempty)) / max(float(len(rows)), 1.0) >= 0.95),
        "semantic_source": "dense_label_maps",
        "missing_labels_by_group": group_ids.get("missing_labels_by_group", {}),
        "label_names": label_names,
    }
    _write_csv(phase_dir / "semantic_projection_by_chunk.csv", rows)
    group_rows = []
    for group in GROUP_ORDER:
        group_rows.append(
            {
                "group": group,
                "point_ratio_median": _median(row.get(f"point_{group}_ratio") for row in rows),
                "patch_ratio_median": _median(row.get(f"patch_{group}_ratio") for row in rows),
                "conf_mean_median": _median(row.get(f"conf_mean_by_{group}") for row in rows),
                "D_geo_mean_median": _median(row.get(f"D_geo_mean_by_{group}") for row in rows),
            }
        )
    _write_csv(phase_dir / "semantic_group_geometry_stats.csv", group_rows)
    _write_json(phase_dir / "phase1_projection_summary.json", summary)
    return summary, rows


def _by_chunk(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row["chunk_id"])] = dict(row)
        except Exception:
            continue
    return out


def phase2_passive_atlas(projection_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    phase_dir = out_dir / "phase2_passive_atlas"
    h35_intr = _by_chunk(_read_csv(V62_OUT / "phase3_intrachunk/h35_intrachunk_metrics.csv"))
    h35_scale = _by_chunk(_read_csv(V62_OUT / "phase5_intrachunk_scale/h35_intrachunk_scale_metrics.csv"))
    h35_inter = _by_chunk(_read_csv(V62_OUT / "phase4_interchunk/h35_interchunk_metrics.csv"))
    taxonomy = _by_chunk(_read_csv(V62_OUT / "phase7_taxonomy/chunk_error_taxonomy.csv"))
    table: List[Dict[str, Any]] = []
    for row in projection_rows:
        cid = int(row["chunk_id"])
        rec = dict(row)
        intr = h35_intr.get(cid, {})
        scale = h35_scale.get(cid, {})
        inter = h35_inter.get(cid, {})
        tax = taxonomy.get(cid, {})
        rec.update(
            {
                "global_chunk_ate": _safe_float(intr.get("global_chunk_ate")),
                "local_sim3_chunk_ate": _safe_float(intr.get("local_sim3_chunk_ate")),
                "local_to_global_ratio": _safe_float(intr.get("local_to_global_ate_ratio")),
                "intra_scale_variance": _safe_float(scale.get("intra_scale_variance")),
                "head_to_tail_transfer_ratio": _safe_float(scale.get("head_to_tail_transfer_ratio")),
                "future_after_overlap_error": _safe_float(inter.get("nonoverlap_future_error_after_overlap_sim3")),
                "scale_jump_vs_prev": _safe_float(inter.get("abs_scale_jump_gtlocal")),
                "rolling100_error": _safe_float(intr.get("rolling100_error") or inter.get("rolling100_error")),
                "taxonomy_type": tax.get("primary_error_type"),
                "metric_source": "v62_h35_landed_csv",
            }
        )
        table.append(rec)

    metrics = [
        "global_chunk_ate",
        "local_sim3_chunk_ate",
        "local_to_global_ratio",
        "intra_scale_variance",
        "head_to_tail_transfer_ratio",
        "future_after_overlap_error",
        "scale_jump_vs_prev",
        "rolling100_error",
    ]
    corr_rows: List[Dict[str, Any]] = []
    enrich_rows: List[Dict[str, Any]] = []
    for group in GROUP_ORDER:
        ratio_key = f"point_{group}_ratio"
        for metric in metrics:
            corr_rows.append(
                {
                    "group": group,
                    "ratio_key": ratio_key,
                    "metric": metric,
                    "correlation": _corr((row.get(ratio_key) for row in table), (row.get(metric) for row in table)),
                    "decision_hint": "ranking_only",
                }
            )
            valid = [row for row in table if math.isfinite(_safe_float(row.get(metric))) and math.isfinite(_safe_float(row.get(ratio_key)))]
            valid.sort(key=lambda row: _safe_float(row.get(metric)), reverse=True)
            if valid:
                top_n = max(1, int(math.ceil(0.2 * len(valid))))
                bot_n = max(1, int(math.ceil(0.5 * len(valid))))
                top_mean = _mean(row.get(ratio_key) for row in valid[:top_n])
                bot_mean = _mean(row.get(ratio_key) for row in valid[-bot_n:])
                enrich = (float(top_mean) / (float(bot_mean) + 1e-9)) if top_mean is not None and bot_mean is not None else None
            else:
                top_mean = bot_mean = enrich = None
            enrich_rows.append(
                {
                    "group": group,
                    "ratio_key": ratio_key,
                    "metric": metric,
                    "top20_error_mean_ratio": top_mean,
                    "bottom50_error_mean_ratio": bot_mean,
                    "enrichment": enrich,
                    "decision_hint": "ranking_only",
                }
            )

    bad = sorted(table, key=lambda row: _safe_float(row.get("global_chunk_ate")), reverse=True)[:10]
    summary = {
        "chunk_rows": len(table),
        "metric_source": "v62_h35_landed_csv",
        "correlation_abs_ge_0p30_count": sum(1 for row in corr_rows if abs(_safe_float(row.get("correlation"))) >= 0.30),
        "enrichment_ge_1p5_count": sum(1 for row in enrich_rows if _safe_float(row.get("enrichment")) >= 1.5),
    }
    _write_csv(phase_dir / "semantic_error_chunk_table.csv", table)
    _write_csv(phase_dir / "semantic_error_correlation.csv", corr_rows)
    _write_csv(phase_dir / "semantic_error_enrichment.csv", enrich_rows)
    _write_csv(phase_dir / "top_bad_chunks_semantic_composition.csv", bad)
    _write_json(phase_dir / "phase2_summary.json", summary)
    lines = [
        "# Phase 2 Passive Semantic Error Atlas",
        "",
        "This phase is correlation/ranking only; it is not causal evidence.",
        f"chunk_rows: {summary['chunk_rows']}",
        f"correlation_abs_ge_0p30_count: {summary['correlation_abs_ge_0p30_count']}",
        f"enrichment_ge_1p5_count: {summary['enrichment_ge_1p5_count']}",
    ]
    _write_text(phase_dir / "semantic_error_atlas_report.md", lines)
    return summary, table


def _weighted_umeyama(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Optional[Tuple[float, torch.Tensor, torch.Tensor, float]]:
    if x.shape[0] < 30:
        return None
    w = w.float().clamp_min(0.0)
    total = float(w.sum().item())
    if total <= EPS:
        return None
    ww = w / total
    x = x.float()
    y = y.float()
    mux = (ww[:, None] * x).sum(dim=0)
    muy = (ww[:, None] * y).sum(dim=0)
    xc = x - mux
    yc = y - muy
    varx = float((ww * (xc * xc).sum(dim=1)).sum().item())
    if varx <= EPS:
        return None
    cov = (ww[:, None] * yc).T @ xc
    try:
        u, svals, vh = torch.linalg.svd(cov)
    except RuntimeError:
        return None
    d = torch.ones(3, dtype=torch.float32)
    if float(torch.det(u @ vh).item()) < 0:
        d[-1] = -1.0
    rot = u @ torch.diag(d) @ vh
    scale = float((svals * d).sum().item() / varx)
    trans = muy - scale * (rot @ mux)
    condition = float((svals[-1] / (svals[0] + 1e-12)).item()) if svals.numel() >= 3 else None
    if not math.isfinite(scale):
        return None
    return scale, rot, trans, condition if condition is not None else 0.0


def _apply_sim3(points: torch.Tensor, fit: Tuple[float, torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    scale, rot, trans, _ = fit
    return float(scale) * (points.float() @ rot.T) + trans[None, :]


def _sample_flat(mask: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if idx.numel() <= max_points:
        return idx
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=gen)[: int(max_points)]
    return idx[perm]


def _grid_coverage(mask: torch.Tensor, grid: int = 16) -> float:
    if mask.numel() == 0:
        return 0.0
    spatial = mask.bool().any(dim=0).float()[None, None]
    pooled = F.adaptive_max_pool2d(spatial, (grid, grid)).squeeze()
    return float((pooled > 0).float().mean().item())


def _road_boundary(ground_mask: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(ground_mask, dtype=torch.bool)
    b[..., 1:, :] |= ground_mask[..., 1:, :] != ground_mask[..., :-1, :]
    b[..., :-1, :] |= ground_mask[..., 1:, :] != ground_mask[..., :-1, :]
    b[..., :, 1:] |= ground_mask[..., :, 1:] != ground_mask[..., :, :-1]
    b[..., :, :-1] |= ground_mask[..., :, 1:] != ground_mask[..., :, :-1]
    return F.max_pool2d(b.float().reshape(-1, 1, b.shape[-2], b.shape[-1]), kernel_size=5, stride=1, padding=2).reshape_as(b).bool()


def _strategy_weights(
    strategy: str,
    base_valid: torch.Tensor,
    conf: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    *,
    chunk_id: int,
    label_point: torch.Tensor,
    random_seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    d_geo = (1.0 - conf).clamp(0.0, 1.0)
    base_w = (conf * (1.0 - d_geo)).clamp(0.0, 1.0) * base_valid.float()
    all_w = conf.clamp(0.0, 1.0) * base_valid.float()
    dyn = masks["dynamic"]
    sky = masks["sky"]
    veg = masks["vegetation"]
    vert = masks["vertical_static"]
    ground = masks["ground_static"]
    void = masks["void_lowtrust"]
    boundary = masks.get("road_boundary")
    if boundary is None:
        boundary = _road_boundary(ground)

    info: Dict[str, Any] = {}
    core = strategy
    control = None
    if strategy.endswith("_RANDOM"):
        core = strategy[: -len("_RANDOM")]
        control = "random"
    elif strategy.endswith("_SHUFFLED"):
        core = strategy[: -len("_SHUFFLED")]
        control = "shuffled"

    local_masks = dict(masks)
    if control == "shuffled":
        shuffled = torch.empty_like(label_point)
        gen = torch.Generator().manual_seed(int(random_seed) + int(chunk_id) * 997 + len(strategy))
        for t in range(int(label_point.shape[0])):
            flat = label_point[t].reshape(-1)
            shuffled[t] = flat[torch.randperm(flat.numel(), generator=gen)].reshape_as(label_point[t])
        local_masks = {
            "dynamic": _isin_label(shuffled, masks["dynamic_ids"]),
            "sky": _isin_label(shuffled, masks["sky_ids"]),
            "vegetation": _isin_label(shuffled, masks["vegetation_ids"]),
            "vertical_static": _isin_label(shuffled, masks["vertical_static_ids"]),
            "ground_static": _isin_label(shuffled, masks["ground_static_ids"]),
            "void_lowtrust": _isin_label(shuffled, masks["void_lowtrust_ids"]),
        }
        local_masks["road_boundary"] = _road_boundary(local_masks["ground_static"])
        dyn = local_masks["dynamic"]
        sky = local_masks["sky"]
        veg = local_masks["vegetation"]
        vert = local_masks["vertical_static"]
        ground = local_masks["ground_static"]
        void = local_masks["void_lowtrust"]
        boundary = local_masks["road_boundary"]

    if core == "S0_ALL":
        w = all_w
    elif core == "S1_GEOMETRY_ONLY":
        w = base_w
    elif core == "S2_SUPPRESS_DYNAMIC":
        w = base_w.masked_fill(dyn, 0.0)
    elif core == "S3_SUPPRESS_SKY":
        w = base_w.masked_fill(sky, 0.0)
    elif core == "S4_SUPPRESS_VEGETATION":
        w = base_w.masked_fill(veg, 0.0)
    elif core == "S5_SUPPRESS_DYNAMIC_SKY":
        w = base_w.masked_fill(dyn | sky, 0.0)
    elif core == "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION":
        w = base_w.masked_fill(dyn | sky | veg, 0.0)
    elif core == "S7_STATIC_ANCHOR_ONLY":
        w = base_w.masked_fill(~(vert | ground), 0.0)
    elif core == "S8_VERTICAL_STATIC_ONLY":
        w = base_w.masked_fill(~vert, 0.0)
    elif core == "S9_ROAD_GROUND_ONLY":
        w = base_w.masked_fill(~ground, 0.0)
    elif core == "S10_VERTICAL_PLUS_ROAD_BOUNDARY":
        w = base_w.masked_fill(~(vert | boundary), 0.0)
    elif core == "S11_SEMANTIC_GEOMETRY_WEIGHTED":
        anchor = vert.float() * (1.0 - d_geo).clamp(0.0, 1.0) * conf
        harm = (dyn | sky | veg | void).float()
        w = (base_w * (1.0 + anchor) * torch.exp(-2.0 * harm)).clamp(0.0, 2.0) * base_valid.float()
    elif core == "V68_ROBUST_SEMOVERLAP_WEIGHT":
        conf_safe = conf.clamp(0.0, 1.0)
        semantic_role = torch.ones_like(conf_safe) * 0.5
        semantic_role = semantic_role.masked_fill(ground, 0.3)
        semantic_role = semantic_role.masked_fill(boundary, 0.7)
        semantic_role = semantic_role.masked_fill(vert, 1.0)
        semantic_role = semantic_role.masked_fill(void, 0.05)
        semantic_role = semantic_role.masked_fill(veg, 0.1)
        semantic_role = semantic_role.masked_fill(dyn, 0.1)
        semantic_role = semantic_role.masked_fill(sky, 0.0)
        low_conf = torch.where(conf_safe < 0.5, conf_safe, torch.ones_like(conf_safe))
        # Pre-fit approximation of the plan's robust residual term: downweight low-confidence
        # geometry as a Cauchy-like proxy until a two-pass residual hook is available.
        robust_proxy = torch.reciprocal(1.0 + torch.square((1.0 - conf_safe) / 0.35))
        w = (conf_safe * semantic_role * low_conf * robust_proxy).clamp(0.0, 2.0) * base_valid.float()
        info.update(
            {
                "v68_robust_semoverlap_weight": True,
                "v68_robust_semoverlap_residual_mode": "pre_fit_confidence_proxy",
                "v68_robust_semoverlap_static_high_conf_weight": 1.0,
                "v68_robust_semoverlap_road_boundary_weight": 0.7,
                "v68_robust_semoverlap_road_plane_weight": 0.3,
                "v68_robust_semoverlap_dynamic_weight": 0.1,
                "v68_robust_semoverlap_sky_weight": 0.0,
                "v68_robust_semoverlap_low_conf_mean": float(low_conf[base_valid].mean().item())
                if bool(base_valid.any().item())
                else 0.0,
                "v68_robust_semoverlap_proxy_mean": float(robust_proxy[base_valid].mean().item())
                if bool(base_valid.any().item())
                else 0.0,
            }
        )
    elif core == "V68_OVERLAP_SUPPORT_WEIGHT":
        support = masks.get("overlap_support")
        if not torch.is_tensor(support):
            raise ValueError("V68_OVERLAP_SUPPORT_WEIGHT requires overlap_support tensor")
        support = support.to(device=base_w.device, dtype=base_w.dtype).clamp(0.0, 1.0)
        floor_raw = masks.get("overlap_support_floor", 0.25)
        try:
            support_floor = float(floor_raw)
        except (TypeError, ValueError):
            support_floor = 0.25
        support_floor = min(1.0, max(0.0, support_floor))
        support_weight = (support_floor + (1.0 - support_floor) * support).clamp(0.0, 1.0)
        semantic_base = base_w.masked_fill(dyn | sky, 0.0)
        w = semantic_base * support_weight
        valid_support = support[base_valid]
        valid_weight = support_weight[base_valid]
        info.update(
            {
                "v68_overlap_support_weight": True,
                "overlap_support_weight_floor": float(support_floor),
                "overlap_support_score_mean": float(valid_support.mean().item()) if bool(valid_support.numel()) else 0.0,
                "overlap_support_score_q90": float(torch.quantile(valid_support.float(), 0.90).item()) if bool(valid_support.numel()) else 0.0,
                "overlap_support_weight_mean": float(valid_weight.mean().item()) if bool(valid_weight.numel()) else 0.0,
                "overlap_support_weight_q90": float(torch.quantile(valid_weight.float(), 0.90).item()) if bool(valid_weight.numel()) else 0.0,
                "overlap_support_weighted_mass": float(w.sum().item()),
            }
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if control == "random":
        target_count = int((w > 0).sum().item())
        idx = _sample_flat(base_valid, target_count, int(random_seed) + int(chunk_id) * 101 + len(strategy))
        rw = torch.zeros_like(base_w).reshape(-1)
        flat_base = base_w.reshape(-1)
        if idx.numel() > 0:
            rw[idx] = flat_base[idx]
        w = rw.reshape_as(base_w)

    before = float(base_w.sum().item())
    remaining = float(w.sum().item())
    boundary_mass = float((base_w * boundary.float()).sum().item())
    dynamic_mass = float((base_w * dyn.float()).sum().item())
    sky_mass = float((base_w * sky.float()).sum().item())
    vegetation_mass = float((base_w * veg.float()).sum().item())
    vertical_mass = float((base_w * vert.float()).sum().item())
    ground_mass = float((base_w * ground.float()).sum().item())
    void_mass = float((base_w * void.float()).sum().item())
    anchor_mass = (
        0.30 * vertical_mass
        + 0.20 * boundary_mass
        + 0.15 * min(vertical_mass + ground_mass, before)
        + 0.20 * remaining
    )
    weak_mass = (
        0.20 * sky_mass
        + 0.15 * dynamic_mass
        + 0.20 * vegetation_mass
        + 0.25 * ground_mass
        + 0.20 * void_mass
    )
    qscale = anchor_mass / (anchor_mass + weak_mass + 1.0e-12)
    info.update(
        {
            "removed_dynamic_mass": dynamic_mass,
            "removed_sky_mass": sky_mass,
            "removed_vegetation_mass": vegetation_mass,
            "kept_vertical_static_mass": float((w * vert.float()).sum().item()),
            "kept_ground_mass": float((w * ground.float()).sum().item()),
            "role_vertical_static_mass": vertical_mass,
            "role_ground_static_mass": ground_mass,
            "role_road_boundary_mass": boundary_mass,
            "role_void_lowtrust_mass": void_mass,
            "role_base_mass": before,
            "qscale_anchor_mass": anchor_mass,
            "qscale_weak_mass": weak_mass,
            "qscale_observability": qscale,
            "remaining_valid_mass": remaining,
            "remaining_valid_ratio": remaining / (before + 1e-12),
            "control_type": control or "none",
        }
    )
    return w, info


def _fit_window(
    local: torch.Tensor,
    world: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    frame_slice: slice,
    *,
    max_points: int,
    seed: int,
) -> Dict[str, Any]:
    mask = (weights[frame_slice] > 0) & valid[frame_slice]
    valid_count = int(mask.sum().item())
    if valid_count < 30:
        return {"fit_success": False, "fit_degenerate": True, "valid_point_count": valid_count}
    idx = _sample_flat(mask, int(max_points), seed)
    flat_local = local[frame_slice].reshape(-1, 3)
    flat_world = world[frame_slice].reshape(-1, 3)
    flat_w = weights[frame_slice].reshape(-1)
    fit = _weighted_umeyama(flat_local[idx], flat_world[idx], flat_w[idx])
    if fit is None:
        return {"fit_success": False, "fit_degenerate": True, "valid_point_count": valid_count, "fit_point_count": int(idx.numel())}
    scale, rot, trans, condition = fit
    return {
        "fit_success": True,
        "fit_degenerate": False,
        "valid_point_count": valid_count,
        "fit_point_count": int(idx.numel()),
        "scale": float(scale),
        "log_scale": float(math.log(abs(scale) + 1e-12)),
        "rot": rot,
        "trans": trans,
        "condition": float(condition),
        "fit": fit,
    }


def _eval_fit(
    fit: Tuple[float, torch.Tensor, torch.Tensor, float],
    local: torch.Tensor,
    world: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    frame_slice: slice,
    *,
    max_points: int,
    seed: int,
) -> Optional[float]:
    mask = (weights[frame_slice] > 0) & valid[frame_slice]
    if int(mask.sum().item()) < 1:
        return None
    idx = _sample_flat(mask, int(max_points), seed)
    x = local[frame_slice].reshape(-1, 3)[idx]
    y = world[frame_slice].reshape(-1, 3)[idx]
    pred = _apply_sim3(x, fit)
    err = torch.linalg.norm(pred - y.float(), dim=1)
    return float(torch.sqrt((err * err).mean()).item()) if err.numel() > 0 else None


def _load_chunk_semantic(stage_c_cache: Path, audit_row: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    chunk = torch.load(stage_c_cache / str(audit_row["chunk_name"]) / "masklet.pt", map_location="cpu", weights_only=False)
    sem = chunk["semantic_segmentation"]
    return sem["label_maps"].detach().cpu().long(), sem


def _chunk_masks(label_point: torch.Tensor, group_ids: Mapping[str, Sequence[int]]) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for group in GROUP_ORDER:
        masks[group] = _isin_label(label_point, group_ids.get(group, []))
        masks[f"{group}_ids"] = list(group_ids.get(group, []))  # type: ignore[assignment]
    masks["road_boundary"] = _road_boundary(masks["ground_static"])
    return masks


PHASE3_STRATEGIES = [
    "S0_ALL",
    "S1_GEOMETRY_ONLY",
    "S2_SUPPRESS_DYNAMIC",
    "S3_SUPPRESS_SKY",
    "S4_SUPPRESS_VEGETATION",
    "S5_SUPPRESS_DYNAMIC_SKY",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION",
    "S7_STATIC_ANCHOR_ONLY",
    "S8_VERTICAL_STATIC_ONLY",
    "S9_ROAD_GROUND_ONLY",
    "S10_VERTICAL_PLUS_ROAD_BOUNDARY",
    "S11_SEMANTIC_GEOMETRY_WEIGHTED",
    "V68_ROBUST_SEMOVERLAP_WEIGHT",
    "V68_ROBUST_SEMOVERLAP_WEIGHT_RANDOM",
    "V68_OVERLAP_SUPPORT_WEIGHT",
    "V68_OVERLAP_SUPPORT_WEIGHT_RANDOM",
    "V68_OVERLAP_SUPPORT_WEIGHT_SHUFFLED",
    "S5_SUPPRESS_DYNAMIC_SKY_RANDOM",
    "S5_SUPPRESS_DYNAMIC_SKY_SHUFFLED",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION_RANDOM",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION_SHUFFLED",
    "S7_STATIC_ANCHOR_ONLY_RANDOM",
    "S7_STATIC_ANCHOR_ONLY_SHUFFLED",
    "S8_VERTICAL_STATIC_ONLY_RANDOM",
    "S8_VERTICAL_STATIC_ONLY_SHUFFLED",
]


def phase3_intrachunk_scale(
    audit_rows: Sequence[Mapping[str, Any]],
    stage_c_cache: Path,
    per_chunk_geometry_dir: Optional[Path],
    group_ids: Mapping[str, Sequence[int]],
    out_dir: Path,
    *,
    target_chunks: Sequence[int],
    max_points_per_fit: int,
    random_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    phase_dir = out_dir / "phase3_intrachunk_scale"
    if per_chunk_geometry_dir is None:
        summary = {"mode": "unavailable", "reason": "per_chunk_geometry_dir not provided", "gate_pass": False}
        _write_json(phase_dir / "phase3_summary.json", summary)
        _write_csv(phase_dir / "strategy_metrics_by_chunk.csv", [])
        return summary, []
    target_set = set(int(x) for x in target_chunks)
    rows: List[Dict[str, Any]] = []
    for audit in audit_rows:
        cid = int(audit["chunk_id"])
        if cid not in target_set:
            continue
        geo = _load_geometry(per_chunk_geometry_dir / f"chunk_{cid:03d}.pt")
        if not geo:
            continue
        local = geo.get("local_points")
        world = geo.get("points", geo.get("world_points"))
        conf = geo.get("conf", geo.get("confidence"))
        if not (torch.is_tensor(local) and torch.is_tensor(world) and torch.is_tensor(conf)):
            continue
        local = local.detach().cpu().float()
        world = world.detach().cpu().float()
        conf = conf.detach().cpu().float().clamp(0.0, 1.0)
        T = min(int(local.shape[0]), int(world.shape[0]), int(conf.shape[0]))
        local, world, conf = local[:T], world[:T], conf[:T]
        label_maps, _ = _load_chunk_semantic(stage_c_cache, audit)
        label_point = _project_label_maps(label_maps[:T], (int(conf.shape[-2]), int(conf.shape[-1])))
        masks = _chunk_masks(label_point, group_ids)
        valid = torch.isfinite(local).all(dim=-1) & torch.isfinite(world).all(dim=-1) & (conf > 0.05)
        n0 = max(1, T // 3)
        n1 = max(n0 + 1, (2 * T) // 3)
        windows = {"head": slice(0, n0), "mid": slice(n0, n1), "tail": slice(n1, T)}
        strategy_results: Dict[str, Dict[str, Any]] = {}
        for strategy in PHASE3_STRATEGIES:
            weights, info = _strategy_weights(
                strategy,
                valid,
                conf,
                masks,
                chunk_id=cid,
                label_point=label_point,
                random_seed=random_seed,
            )
            fit_rows = {}
            for name, sl in windows.items():
                fit_rows[name] = _fit_window(
                    local,
                    world,
                    weights,
                    valid,
                    sl,
                    max_points=max_points_per_fit,
                    seed=random_seed + cid * 1000 + len(strategy) * 10 + len(name),
                )
            head_fit = fit_rows["head"].get("fit")
            mid_fit = fit_rows["mid"].get("fit")
            h2t = _eval_fit(head_fit, local, world, weights, valid, windows["tail"], max_points=max_points_per_fit, seed=random_seed + cid * 17) if head_fit else None
            m2t = _eval_fit(mid_fit, local, world, weights, valid, windows["tail"], max_points=max_points_per_fit, seed=random_seed + cid * 19) if mid_fit else None
            log_scales = [fit_rows[x].get("log_scale") for x in ("head", "mid", "tail") if fit_rows[x].get("fit_success")]
            intra_var = float(np.var([float(x) for x in log_scales])) if len(log_scales) >= 2 else None
            rec: Dict[str, Any] = {
                "chunk_id": cid,
                "strategy": strategy,
                "mode": "pointmap_region_sampled",
                "valid_point_count": int((weights > 0).sum().item()),
                "valid_point_ratio": float((weights > 0).float().mean().item()),
                "valid_weight_sum": float(weights.sum().item()),
                "grid_coverage_ratio": _grid_coverage(weights > 0),
                "sim3_condition_score": _median(fit_rows[x].get("condition") for x in ("head", "mid", "tail")),
                "fit_success": all(bool(fit_rows[x].get("fit_success")) for x in ("head", "mid", "tail")),
                "fit_degenerate": any(bool(fit_rows[x].get("fit_degenerate")) for x in ("head", "mid", "tail")),
                "log_scale_head": fit_rows["head"].get("log_scale"),
                "log_scale_mid": fit_rows["mid"].get("log_scale"),
                "log_scale_tail": fit_rows["tail"].get("log_scale"),
                "fit_point_count_head": fit_rows["head"].get("fit_point_count"),
                "fit_point_count_mid": fit_rows["mid"].get("fit_point_count"),
                "fit_point_count_tail": fit_rows["tail"].get("fit_point_count"),
                "intra_scale_variance": intra_var,
                "head_to_tail_transfer_error": h2t,
                "mid_to_tail_transfer_error": m2t,
                "future_after_overlap_error": None,
                "tail_error_after_overlap": None,
                "max_points_per_fit": int(max_points_per_fit),
                **info,
            }
            strategy_results[strategy] = rec
            rows.append(rec)
        baseline = strategy_results.get("S1_GEOMETRY_ONLY", {})
        for strategy, rec in strategy_results.items():
            h2t_base = _safe_float(baseline.get("head_to_tail_transfer_error"))
            var_base = _safe_float(baseline.get("intra_scale_variance"))
            h2t = _safe_float(rec.get("head_to_tail_transfer_error"))
            var = _safe_float(rec.get("intra_scale_variance"))
            rec["head_to_tail_improvement_vs_S1"] = (h2t_base - h2t) / h2t_base if math.isfinite(h2t_base) and h2t_base > 0 and math.isfinite(h2t) else None
            rec["intra_scale_variance_improvement_vs_S1"] = (var_base - var) / var_base if math.isfinite(var_base) and var_base > 0 and math.isfinite(var) else None
            rec["decision"] = "positive_proxy" if max(_safe_float(rec.get("head_to_tail_improvement_vs_S1")), _safe_float(rec.get("intra_scale_variance_improvement_vs_S1"))) >= 0.10 else "not_positive_proxy"
        for base in ("S5_SUPPRESS_DYNAMIC_SKY", "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION", "S7_STATIC_ANCHOR_ONLY", "S8_VERTICAL_STATIC_ONLY"):
            if base in strategy_results:
                b = strategy_results[base]
                rnd = strategy_results.get(f"{base}_RANDOM", {})
                shuf = strategy_results.get(f"{base}_SHUFFLED", {})
                b["beats_random"] = _safe_float(b.get("head_to_tail_improvement_vs_S1")) > _safe_float(rnd.get("head_to_tail_improvement_vs_S1"))
                b["beats_shuffled"] = _safe_float(b.get("head_to_tail_improvement_vs_S1")) > _safe_float(shuf.get("head_to_tail_improvement_vs_S1"))

    summary_rows: List[Dict[str, Any]] = []
    for strategy in PHASE3_STRATEGIES:
        subset = [row for row in rows if row.get("strategy") == strategy]
        summary_rows.append(
            {
                "strategy": strategy,
                "chunk_count": len(subset),
                "fit_success_count": sum(1 for row in subset if row.get("fit_success")),
                "median_head_to_tail_transfer_error": _median(row.get("head_to_tail_transfer_error") for row in subset),
                "median_intra_scale_variance": _median(row.get("intra_scale_variance") for row in subset),
                "median_head_to_tail_improvement_vs_S1": _median(row.get("head_to_tail_improvement_vs_S1") for row in subset),
                "median_intra_scale_variance_improvement_vs_S1": _median(row.get("intra_scale_variance_improvement_vs_S1") for row in subset),
                "positive_proxy_count": sum(1 for row in subset if row.get("decision") == "positive_proxy"),
            }
        )
    _write_csv(phase_dir / "strategy_metrics_by_chunk.csv", rows)
    _write_csv(phase_dir / "strategy_summary.csv", summary_rows)
    best = sorted(summary_rows, key=lambda row: _safe_float(row.get("median_head_to_tail_improvement_vs_S1")), reverse=True)
    summary = {
        "mode": "pointmap_region_sampled",
        "geometry_dir": str(per_chunk_geometry_dir),
        "target_chunk_count": len(target_set),
        "measured_rows": len(rows),
        "max_points_per_fit": int(max_points_per_fit),
        "best_by_median_head_to_tail_improvement": best[0] if best else None,
        "gate_note": "proxy gate; sampled pointmap region fitting, not full pipeline action",
    }
    _write_json(phase_dir / "phase3_summary.json", summary)
    return summary, rows


PHASE4_STRATEGIES = [
    "S1_GEOMETRY_ONLY",
    "S5_SUPPRESS_DYNAMIC_SKY",
    "S6_SUPPRESS_DYNAMIC_SKY_VEGETATION",
    "S7_STATIC_ANCHOR_ONLY",
    "S8_VERTICAL_STATIC_ONLY",
    "S10_VERTICAL_PLUS_ROAD_BOUNDARY",
    "S11_SEMANTIC_GEOMETRY_WEIGHTED",
    "V68_ROBUST_SEMOVERLAP_WEIGHT",
    "V68_ROBUST_SEMOVERLAP_WEIGHT_RANDOM",
    "V68_OVERLAP_SUPPORT_WEIGHT",
    "V68_OVERLAP_SUPPORT_WEIGHT_RANDOM",
    "V68_OVERLAP_SUPPORT_WEIGHT_SHUFFLED",
    "S5_SUPPRESS_DYNAMIC_SKY_RANDOM",
    "S5_SUPPRESS_DYNAMIC_SKY_SHUFFLED",
    "S8_VERTICAL_STATIC_ONLY_RANDOM",
    "S8_VERTICAL_STATIC_ONLY_SHUFFLED",
]


def phase4_overlap_merge_anchor(
    audit_rows: Sequence[Mapping[str, Any]],
    stage_c_cache: Path,
    per_chunk_geometry_dir: Optional[Path],
    group_ids: Mapping[str, Sequence[int]],
    out_dir: Path,
    *,
    target_chunks: Sequence[int],
    max_points_per_fit: int,
    random_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    phase_dir = out_dir / "phase4_overlap_merge_anchor"
    if per_chunk_geometry_dir is None:
        summary = {"mode": "unavailable", "reason": "per_chunk_geometry_dir not provided"}
        _write_json(phase_dir / "phase4_summary.json", summary)
        _write_csv(phase_dir / "overlap_strategy_results.csv", [])
        return summary, []
    by_id = {int(row["chunk_id"]): row for row in audit_rows}
    target_set = set(int(x) for x in target_chunks)
    rows: List[Dict[str, Any]] = []
    for cid in sorted(target_set):
        if cid <= 0 or cid not in by_id or (cid - 1) not in by_id:
            continue
        cur_geo = _load_geometry(per_chunk_geometry_dir / f"chunk_{cid:03d}.pt")
        prev_geo = _load_geometry(per_chunk_geometry_dir / f"chunk_{cid - 1:03d}.pt")
        if not cur_geo or not prev_geo:
            continue
        cur_local = cur_geo.get("local_points")
        prev_world = prev_geo.get("points", prev_geo.get("world_points"))
        cur_world = cur_geo.get("points", cur_geo.get("world_points"))
        conf = cur_geo.get("conf", cur_geo.get("confidence"))
        if not (torch.is_tensor(cur_local) and torch.is_tensor(prev_world) and torch.is_tensor(cur_world) and torch.is_tensor(conf)):
            continue
        start = int(by_id[cid]["start_frame"])
        prev_start = int(by_id[cid - 1]["start_frame"])
        prev_end = int(by_id[cid - 1]["end_frame"])
        overlap_start = max(start, prev_start)
        overlap_end = min(int(by_id[cid]["end_frame"]), prev_end)
        if overlap_end <= overlap_start:
            continue
        cur_sl = slice(overlap_start - start, overlap_end - start)
        prev_sl = slice(overlap_start - prev_start, overlap_end - prev_start)
        future_sl = slice(overlap_end - start, int(cur_local.shape[0]))
        if future_sl.start >= future_sl.stop:
            continue
        cur_local = cur_local.detach().cpu().float()
        cur_world = cur_world.detach().cpu().float()
        prev_world = prev_world.detach().cpu().float()
        conf = conf.detach().cpu().float().clamp(0.0, 1.0)
        label_maps, _ = _load_chunk_semantic(stage_c_cache, by_id[cid])
        label_point = _project_label_maps(label_maps[: int(conf.shape[0])], (int(conf.shape[-2]), int(conf.shape[-1])))
        masks = _chunk_masks(label_point, group_ids)
        valid_cur = torch.isfinite(cur_local).all(dim=-1) & torch.isfinite(cur_world).all(dim=-1) & (conf > 0.05)
        for strategy in PHASE4_STRATEGIES:
            weights, info = _strategy_weights(
                strategy,
                valid_cur,
                conf,
                masks,
                chunk_id=cid,
                label_point=label_point,
                random_seed=random_seed,
            )
            fit = _fit_window(
                cur_local,
                prev_world[prev_sl.start : prev_sl.stop],
                weights[cur_sl.start : cur_sl.stop],
                valid_cur[cur_sl.start : cur_sl.stop],
                slice(0, cur_sl.stop - cur_sl.start),
                max_points=max_points_per_fit,
                seed=random_seed + cid * 811 + len(strategy),
            )
            fit_obj = fit.get("fit")
            overlap_residual = None
            future_err = None
            tail_err = None
            if fit_obj is not None:
                overlap_residual = _eval_fit(
                    fit_obj,
                    cur_local[cur_sl.start : cur_sl.stop],
                    prev_world[prev_sl.start : prev_sl.stop],
                    weights[cur_sl.start : cur_sl.stop],
                    valid_cur[cur_sl.start : cur_sl.stop],
                    slice(0, cur_sl.stop - cur_sl.start),
                    max_points=max_points_per_fit,
                    seed=random_seed + cid * 823,
                )
                future_err = _eval_fit(
                    fit_obj,
                    cur_local,
                    cur_world,
                    weights,
                    valid_cur,
                    future_sl,
                    max_points=max_points_per_fit,
                    seed=random_seed + cid * 827,
                )
                tail_start = max(future_sl.start, int(cur_local.shape[0]) - max(1, int(cur_local.shape[0]) // 3))
                tail_err = _eval_fit(
                    fit_obj,
                    cur_local,
                    cur_world,
                    weights,
                    valid_cur,
                    slice(tail_start, int(cur_local.shape[0])),
                    max_points=max_points_per_fit,
                    seed=random_seed + cid * 829,
                )
            rows.append(
                {
                    "chunk_id": cid,
                    "prev_chunk_id": cid - 1,
                    "strategy": strategy,
                    "mode": "overlap_pointmap_region_sampled",
                    "overlap_frame_count": int(overlap_end - overlap_start),
                    "overlap_valid_point_count": int(((weights[cur_sl] > 0) & valid_cur[cur_sl]).sum().item()),
                    "overlap_valid_ratio": float(((weights[cur_sl] > 0) & valid_cur[cur_sl]).float().mean().item()),
                    "overlap_residual": overlap_residual,
                    "future_after_overlap_error": future_err,
                    "tail_error_after_overlap": tail_err,
                    "scale_jump_after_strategy": fit.get("scale"),
                    "sim3_condition_score": fit.get("condition"),
                    "fit_success": fit.get("fit_success"),
                    "max_points_per_fit": int(max_points_per_fit),
                    "decision": "pending_vs_baseline",
                    **info,
                }
            )
    by_chunk_strategy = {(int(row["chunk_id"]), str(row["strategy"])): row for row in rows}
    for row in rows:
        base = by_chunk_strategy.get((int(row["chunk_id"]), "S1_GEOMETRY_ONLY"), {})
        b_future = _safe_float(base.get("future_after_overlap_error"))
        future = _safe_float(row.get("future_after_overlap_error"))
        b_overlap = _safe_float(base.get("overlap_residual"))
        overlap = _safe_float(row.get("overlap_residual"))
        row["future_improvement_vs_S1"] = (b_future - future) / b_future if math.isfinite(b_future) and b_future > 0 and math.isfinite(future) else None
        row["overlap_residual_change_vs_S1"] = (overlap - b_overlap) / b_overlap if math.isfinite(b_overlap) and b_overlap > 0 and math.isfinite(overlap) else None
        row["decision"] = (
            "positive_proxy"
            if _safe_float(row.get("future_improvement_vs_S1")) >= 0.10 and _safe_float(row.get("overlap_residual_change_vs_S1")) <= 0.20
            else "not_positive_proxy"
        )
    summary_rows: List[Dict[str, Any]] = []
    for strategy in PHASE4_STRATEGIES:
        subset = [row for row in rows if row.get("strategy") == strategy]
        summary_rows.append(
            {
                "strategy": strategy,
                "chunk_count": len(subset),
                "fit_success_count": sum(1 for row in subset if row.get("fit_success")),
                "median_overlap_residual": _median(row.get("overlap_residual") for row in subset),
                "median_future_after_overlap_error": _median(row.get("future_after_overlap_error") for row in subset),
                "median_future_improvement_vs_S1": _median(row.get("future_improvement_vs_S1") for row in subset),
                "positive_proxy_count": sum(1 for row in subset if row.get("decision") == "positive_proxy"),
            }
        )
    _write_csv(phase_dir / "overlap_strategy_results.csv", rows)
    _write_csv(phase_dir / "overlap_strategy_summary.csv", summary_rows)
    best = sorted(summary_rows, key=lambda row: _safe_float(row.get("median_future_improvement_vs_S1")), reverse=True)
    summary = {
        "mode": "overlap_pointmap_region_sampled",
        "geometry_dir": str(per_chunk_geometry_dir),
        "measured_rows": len(rows),
        "best_by_median_future_improvement": best[0] if best else None,
        "gate_note": "proxy gate; sampled duplicate-overlap pointmap fitting",
    }
    _write_json(phase_dir / "phase4_summary.json", summary)
    return summary, rows


def write_final_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v66B Dense Semantic Scale Diagnostic Report",
        "",
        "This report is generated from real artifacts only. Missing phases are marked as not_run or unavailable.",
        "",
        "## Summary",
        "",
        f"- Phase 0 gate_pass: {summary.get('phase0', {}).get('gate_pass')}",
        f"- Phase 1 gate_pass: {summary.get('phase1', {}).get('gate_pass')}",
        f"- Phase 3 mode: {summary.get('phase3', {}).get('mode')}",
        f"- Phase 4 mode: {summary.get('phase4', {}).get('mode')}",
        "",
        "## Required Questions",
        "",
        f"1. New semantic cache used: {'yes' if summary.get('phase0', {}).get('gate_pass') else 'not fully verified'}",
        f"2. 38/38 chunk slice equality: {summary.get('phase0', {}).get('slice_equal_full_all')}",
        "3. dynamic scale pollution: see Phase 3 strategy rows; only positive if semantic strategy beats S1 and controls.",
        "4. sky scale pollution / context: see Phase 3/4 strategy rows; READ context is not tested by this offline script.",
        "5. vegetation: see S4/S6 rows; no READ conclusion from offline rows alone.",
        "6. vertical static anchor: see S8/S10 rows and control comparisons.",
        "7. road/ground degeneration: see S9 rows and sim3_condition_score/grid_coverage_ratio.",
        "8. dynamic+sky vs random: see S5 rows with beats_random/beats_shuffled.",
        "9. static anchor vs geometry-only: see S7/S8 improvements vs S1.",
        "10. semantic READ: not_run until offline gates justify Phase 5.",
        "11. semantic TTT: not_run until offline gates justify Phase 6.",
        "12. SWA/VGGT4D-style skip: not_run until offline gates justify Phase 7.",
        "13. path attribution: offline data covers scale fitting and overlap/merge proxy only.",
        "14. old conclusions: not revised by READ/TTT/SWA until action smokes run.",
        "15. next direction: choose from Phase 3/4 positive strategies if they beat random/shuffled controls.",
    ]
    _write_text(out_dir / "v66b_final_report.md", lines)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--full_semantic_pt", type=Path, default=DEFAULT_FULL_SEMANTIC)
    parser.add_argument("--stage_c_cache", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--h35_run", type=Path, default=DEFAULT_H35_RUN)
    parser.add_argument("--gt_poses", type=Path, default=DEFAULT_GT)
    parser.add_argument("--per_chunk_geometry_dir", type=Path, default=None)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--chunk_overlap", type=int, default=3)
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--target_chunks", default="all")
    parser.add_argument("--max_chunks", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--max_points_per_fit", type=int, default=12000)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    phase0, audit_rows, group_ids = phase0_cache_audit(args.full_semantic_pt, args.stage_c_cache, out_dir)
    if not phase0.get("gate_pass"):
        summary = {"status": "blocked_phase0_cache_gate", "phase0": phase0}
        _write_json(out_dir / "v66b_summary.json", summary)
        write_final_report(out_dir, summary)
        raise SystemExit("Phase 0 gate failed; stopping as required by v66B plan")

    all_ids = [int(row["chunk_id"]) for row in audit_rows]
    target_chunks = _parse_target_chunks(args.target_chunks, all_ids)
    if int(args.max_chunks) > 0:
        target_chunks = target_chunks[: int(args.max_chunks)]

    phase1, projection_rows = phase1_projection_audit(audit_rows, args.stage_c_cache, args.per_chunk_geometry_dir, group_ids, out_dir)
    phase2, atlas_rows = phase2_passive_atlas(projection_rows, out_dir)
    phase3, phase3_rows = phase3_intrachunk_scale(
        audit_rows,
        args.stage_c_cache,
        args.per_chunk_geometry_dir,
        group_ids,
        out_dir,
        target_chunks=target_chunks,
        max_points_per_fit=int(args.max_points_per_fit),
        random_seed=int(args.random_seed),
    )
    phase4, phase4_rows = phase4_overlap_merge_anchor(
        audit_rows,
        args.stage_c_cache,
        args.per_chunk_geometry_dir,
        group_ids,
        out_dir,
        target_chunks=target_chunks,
        max_points_per_fit=int(args.max_points_per_fit),
        random_seed=int(args.random_seed),
    )
    summary = {
        "status": "offline_phase0_4_complete",
        "seq": args.seq,
        "full_semantic_pt": str(args.full_semantic_pt),
        "stage_c_cache": str(args.stage_c_cache),
        "h35_run": str(args.h35_run),
        "gt_poses": str(args.gt_poses),
        "per_chunk_geometry_dir": str(args.per_chunk_geometry_dir) if args.per_chunk_geometry_dir is not None else None,
        "target_chunks": target_chunks,
        "phase0": phase0,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
        "phase5_read": {"status": "not_run", "reason": "action smoke requires explicit promotion after offline review"},
        "phase6_ttt": {"status": "not_run", "reason": "action smoke requires explicit promotion after offline review"},
        "phase7_swa": {"status": "not_run", "reason": "action smoke requires explicit promotion after offline review"},
    }
    _write_json(out_dir / "v66b_summary.json", summary)
    write_final_report(out_dir, summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
