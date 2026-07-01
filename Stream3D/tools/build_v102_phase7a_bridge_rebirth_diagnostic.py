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
OUT_DIR = AUDIT_ROOT / "v102_phase7a_bridge_rebirth_diagnostic"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE5C_ROWS = (
    AUDIT_ROOT
    / "v102_phase5c_semantic_barrier_bridge_repair"
    / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
)


VARIANTS = [
    ("tau04_missing_allow", 0.40, "allow"),
    ("tau05_missing_allow", 0.50, "allow"),
    ("tau06_missing_allow", 0.60, "allow"),
    ("tau05_missing_block", 0.50, "block"),
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


def _node_meta(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for prefix in ["a", "b"]:
        cols = {
            "a": [
                "mask_a_observation_id",
                "frame_a",
                "diagnostic_gt_a",
                "diagnostic_semantic_label_a",
            ],
            "b": [
                "mask_b_observation_id",
                "frame_b",
                "diagnostic_gt_b",
                "diagnostic_semantic_label_b",
            ],
        }[prefix]
        sub = df[cols].drop_duplicates(cols[0]).copy()
        sub.columns = ["node_id", "frame_id", "diagnostic_gt_instance", "diagnostic_semantic_label"]
        parts.append(sub)
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("node_id")
    return meta.set_index("node_id")


def _accepted_edges(df: pd.DataFrame, tau: float, missing_policy: str) -> pd.DataFrame:
    semantic_ok = df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= tau
    if missing_policy == "allow":
        semantic_ok = semantic_ok | (~df["semantic_residual_available"].to_numpy(dtype=bool))
    accept = (
        (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
        & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= 1)
        & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= 0.001)
        & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
        & semantic_ok
    )
    return df.loc[accept].copy()


def _components(edges: pd.DataFrame, meta: pd.DataFrame, variant_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = sorted(set(edges["mask_a_observation_id"]) | set(edges["mask_b_observation_id"]))
    parent = {node: node for node in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in edges[["mask_a_observation_id", "mask_b_observation_id"]].itertuples(index=False):
        union(str(row.mask_a_observation_id), str(row.mask_b_observation_id))

    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)

    rows: list[dict[str, Any]] = []
    purities: list[float] = []
    collision_count = 0
    multi_gt_count = 0
    clean_count = 0
    largest = 0
    for component_index, component_nodes in enumerate(groups.values()):
        component_nodes = sorted(component_nodes)
        m = meta.loc[component_nodes]
        frames = m["frame_id"].astype(int).tolist()
        gts = m["diagnostic_gt_instance"].astype(str).tolist()
        sems = m["diagnostic_semantic_label"].astype(str).tolist()
        gt_counts = pd.Series(gts).value_counts()
        sem_counts = pd.Series(sems).value_counts()
        purity = float(gt_counts.iloc[0] / len(component_nodes)) if len(component_nodes) else 0.0
        same_frame_collision = len(frames) != len(set(frames))
        multi_gt = int(len(gt_counts) > 1)
        clean = bool(not same_frame_collision and purity >= 0.80)
        purities.append(purity)
        collision_count += int(same_frame_collision)
        multi_gt_count += int(multi_gt)
        clean_count += int(clean)
        largest = max(largest, len(component_nodes))
        rows.append(
            {
                "schema_version": "stream4d_v102_phase7a_component_row_v1",
                "phase_id": "v102_phase7a_bridge_rebirth_diagnostic",
                "variant_id": variant_id,
                "component_id": f"{variant_id}:component_{component_index:04d}",
                "node_count": int(len(component_nodes)),
                "frame_count": int(len(set(frames))),
                "same_frame_collision": same_frame_collision,
                "diagnostic_gt_dominant": gt_counts.index[0] if len(gt_counts) else "",
                "diagnostic_gt_purity": purity,
                "diagnostic_gt_count": int(len(gt_counts)),
                "diagnostic_semantic_dominant": sem_counts.index[0] if len(sem_counts) else "",
                "diagnostic_semantic_count": int(len(sem_counts)),
                "clean_component_proxy": clean,
                "node_ids_joined": "|".join(component_nodes[:50]),
                "node_ids_truncated": len(component_nodes) > 50,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )
    stats = {
        "component_count": int(len(groups)),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "largest_component_size": int(largest),
        "same_frame_collision_component_count": int(collision_count),
        "multi_gt_component_count": int(multi_gt_count),
        "clean_component_proxy_count": int(clean_count),
        "component_purity_mean": float(np.mean(purities)) if purities else 0.0,
        "component_purity_p10": float(np.quantile(purities, 0.10)) if purities else 0.0,
    }
    return rows, stats


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge_df = pd.read_parquet(PHASE5C_ROWS)
    meta = _node_meta(bridge_df)
    component_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    best_stats: dict[str, Any] | None = None
    for variant_id, tau, missing_policy in VARIANTS:
        edges = _accepted_edges(bridge_df, tau, missing_policy)
        rows, stats = _components(edges, meta, variant_id)
        component_rows.extend(rows)
        safe_for_rebirth = bool(
            stats["same_frame_collision_component_count"] == 0
            and stats["multi_gt_component_count"] == 0
            and stats["component_purity_p10"] >= 0.80
        )
        variant_row = {
            "schema_version": "stream4d_v102_phase7a_rebirth_variant_row_v1",
            "phase_id": "v102_phase7a_bridge_rebirth_diagnostic",
            "variant_id": variant_id,
            "semantic_cosine_min": tau,
            "missing_feature_policy": missing_policy,
            **stats,
            "safe_for_primitive_rebirth": safe_for_rebirth,
            "blocker": ""
            if safe_for_rebirth
            else "Accepted bridge graph creates same-frame collisions, multi-GT components, or low-purity components.",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        variant_rows.append(variant_row)
        if best_stats is None or (
            variant_row["clean_component_proxy_count"],
            -variant_row["same_frame_collision_component_count"],
            -variant_row["largest_component_size"],
        ) > (
            best_stats["clean_component_proxy_count"],
            -best_stats["same_frame_collision_component_count"],
            -best_stats["largest_component_size"],
        ):
            best_stats = variant_row

    assert best_stats is not None
    component_path = OUT_DIR / "bridge_component_rows.csv"
    variant_path = OUT_DIR / "rebirth_variant_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(component_path, component_rows)
    _write_csv(variant_path, variant_rows)
    gate_rows = [
        {
            "gate_id": "any_variant_safe_for_primitive_rebirth",
            "pass": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
            "expected": True,
            "observed": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
        },
        {
            "gate_id": "best_variant_same_frame_collision_component_count",
            "pass": int(best_stats["same_frame_collision_component_count"]) == 0,
            "expected": 0,
            "observed": best_stats["same_frame_collision_component_count"],
            "variant_id": best_stats["variant_id"],
        },
        {
            "gate_id": "best_variant_component_purity_p10",
            "pass": float(best_stats["component_purity_p10"]) >= 0.80,
            "expected": ">=0.80",
            "observed": best_stats["component_purity_p10"],
            "variant_id": best_stats["variant_id"],
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
        },
    ]
    _write_csv(gate_path, gate_rows)

    decision = (
        "PASS_PRIMITIVE_REBIRTH_DIAGNOSTIC_READY"
        if any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows)
        else "NO_GO_PRIMITIVE_REBIRTH_COMPONENTS_UNSAFE"
    )
    summary = {
        "schema_version": "stream4d_v102_phase7a_bridge_rebirth_diagnostic_summary_v1",
        "phase_id": "v102_phase7a_bridge_rebirth_diagnostic",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "variant_count": len(variant_rows),
        "best_variant_id": best_stats["variant_id"],
        "best_variant_edge_count": best_stats["edge_count"],
        "best_variant_node_count": best_stats["node_count"],
        "best_variant_component_count": best_stats["component_count"],
        "best_variant_largest_component_size": best_stats["largest_component_size"],
        "best_variant_same_frame_collision_component_count": best_stats["same_frame_collision_component_count"],
        "best_variant_multi_gt_component_count": best_stats["multi_gt_component_count"],
        "best_variant_component_purity_mean": best_stats["component_purity_mean"],
        "best_variant_component_purity_p10": best_stats["component_purity_p10"],
        "any_variant_safe_for_primitive_rebirth": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
        "truthfulness_note": (
            "This is an optional Phase7 diagnostic only. GT labels are used to audit components, not to form them."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "bridge_component_rows": _rel(component_path),
            "rebirth_variant_rows": _rel(variant_path),
            "variant_gate_rows": _rel(gate_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
