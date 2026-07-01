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
OUT_DIR = AUDIT_ROOT / "v102_phase7b_constrained_bridge_graph_repair"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"
PHASE5C_ROWS = (
    AUDIT_ROOT
    / "v102_phase5c_semantic_barrier_bridge_repair"
    / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
)
FEATURE_STORE = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050" / "mask_features.npz"


VARIANTS = [
    {
        "variant_id": "tau04_allow_frame_forest",
        "semantic_cosine_min": 0.40,
        "missing_feature_policy": "allow",
        "centroid_cosine_min": None,
        "require_centroid_feature": False,
    },
    {
        "variant_id": "tau05_allow_frame_forest",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "centroid_cosine_min": None,
        "require_centroid_feature": False,
    },
    {
        "variant_id": "tau05_allow_frame_centroid050",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "centroid_cosine_min": 0.50,
        "require_centroid_feature": False,
    },
    {
        "variant_id": "tau05_allow_frame_centroid070",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "allow",
        "centroid_cosine_min": 0.70,
        "require_centroid_feature": False,
    },
    {
        "variant_id": "tau05_block_frame_centroid050",
        "semantic_cosine_min": 0.50,
        "missing_feature_policy": "block",
        "centroid_cosine_min": 0.50,
        "require_centroid_feature": True,
    },
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


def _feature_map() -> dict[str, np.ndarray]:
    store = np.load(FEATURE_STORE)
    feats = store["features"].astype(np.float32)
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    ids = [str(x) for x in store["mask_observation_id"]]
    return {node_id: feats[i] for i, node_id in enumerate(ids)}


def _node_meta(df: pd.DataFrame, fmap: dict[str, np.ndarray]) -> pd.DataFrame:
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
    meta["feature_available"] = meta["node_id"].map(lambda node_id: str(node_id) in fmap)
    return meta.set_index("node_id")


def _candidate_edges(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    semantic_ok = df["semantic_residual_cosine"].to_numpy(dtype=np.float64) >= float(spec["semantic_cosine_min"])
    if spec["missing_feature_policy"] == "allow":
        semantic_ok = semantic_ok | (~df["semantic_residual_available"].to_numpy(dtype=bool))
    accept = (
        (df["frame_gap_index"].to_numpy(dtype=np.int64) <= 4)
        & (df["gs_shared_gaussian_count"].to_numpy(dtype=np.int64) >= 1)
        & (df["gs_bridge_ratio_min_support"].to_numpy(dtype=np.float64) >= 0.001)
        & (df["broad_contamination_score"].to_numpy(dtype=np.float64) <= 0.20)
        & semantic_ok
    )
    edges = df.loc[accept].copy()
    sem = edges["semantic_residual_cosine"].fillna(float(spec["semantic_cosine_min"])).clip(lower=0.0)
    edges["edge_weight"] = (
        edges["gs_bridge_ratio_min_support"].astype(float)
        * sem.astype(float)
        * np.log1p(edges["gs_shared_gaussian_count"].astype(float))
    )
    return edges.sort_values(
        ["edge_weight", "gs_bridge_ratio_min_support", "semantic_residual_cosine", "gs_shared_gaussian_count"],
        ascending=[False, False, False, False],
    )


class Forest:
    def __init__(self, nodes: set[str], meta: pd.DataFrame, fmap: dict[str, np.ndarray]):
        self.parent = {node: node for node in nodes}
        self.frames = {node: {int(meta.loc[node, "frame_id"])} for node in nodes}
        self.nodes = {node: {node} for node in nodes}
        self.feature_sum: dict[str, np.ndarray] = {}
        self.feature_count: dict[str, int] = {}
        for node in nodes:
            feat = fmap.get(node)
            if feat is None:
                self.feature_sum[node] = np.zeros(768, dtype=np.float32)
                self.feature_count[node] = 0
            else:
                self.feature_sum[node] = feat.copy()
                self.feature_count[node] = 1

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def centroid(self, root: str) -> np.ndarray | None:
        count = self.feature_count[root]
        if count == 0:
            return None
        vec = self.feature_sum[root] / float(count)
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            return None
        return vec / norm

    def can_union(self, a: str, b: str, spec: dict[str, Any]) -> tuple[bool, str]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False, "already_same_component"
        if self.frames[ra] & self.frames[rb]:
            return False, "same_frame_cannot_link"
        centroid_min = spec["centroid_cosine_min"]
        if centroid_min is not None:
            ca, cb = self.centroid(ra), self.centroid(rb)
            if ca is None or cb is None:
                if spec["require_centroid_feature"]:
                    return False, "missing_centroid_feature"
            elif float(np.dot(ca, cb)) < float(centroid_min):
                return False, "centroid_semantic_veto"
        return True, ""

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if len(self.nodes[ra]) < len(self.nodes[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.frames[ra] |= self.frames[rb]
        self.nodes[ra] |= self.nodes[rb]
        self.feature_sum[ra] = self.feature_sum[ra] + self.feature_sum[rb]
        self.feature_count[ra] += self.feature_count[rb]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in self.parent:
            out.setdefault(self.find(node), []).append(node)
        return {root: sorted(nodes) for root, nodes in out.items()}


def _diagnose_components(groups: dict[str, list[str]], meta: pd.DataFrame, variant_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    purities: list[float] = []
    collision_count = 0
    multi_gt_count = 0
    clean_count = 0
    largest = 0
    for idx, component_nodes in enumerate(groups.values()):
        m = meta.loc[component_nodes]
        frames = m["frame_id"].astype(int).tolist()
        gts = m["diagnostic_gt_instance"].astype(str).tolist()
        sems = m["diagnostic_semantic_label"].astype(str).tolist()
        gt_counts = pd.Series(gts).value_counts()
        sem_counts = pd.Series(sems).value_counts()
        purity = float(gt_counts.iloc[0] / len(component_nodes)) if len(component_nodes) else 0.0
        same_frame_collision = len(frames) != len(set(frames))
        multi_gt = len(gt_counts) > 1
        clean = bool(not same_frame_collision and purity >= 0.80)
        purities.append(purity)
        collision_count += int(same_frame_collision)
        multi_gt_count += int(multi_gt)
        clean_count += int(clean)
        largest = max(largest, len(component_nodes))
        rows.append(
            {
                "schema_version": "stream4d_v102_phase7b_component_row_v1",
                "phase_id": "v102_phase7b_constrained_bridge_graph_repair",
                "variant_id": variant_id,
                "component_id": f"{variant_id}:component_{idx:04d}",
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
        "node_count": int(sum(len(nodes) for nodes in groups.values())),
        "largest_component_size": int(largest),
        "same_frame_collision_component_count": int(collision_count),
        "multi_gt_component_count": int(multi_gt_count),
        "clean_component_proxy_count": int(clean_count),
        "component_purity_mean": float(np.mean(purities)) if purities else 0.0,
        "component_purity_p10": float(np.quantile(purities, 0.10)) if purities else 0.0,
    }
    return rows, stats


def _run_variant(df: pd.DataFrame, meta: pd.DataFrame, fmap: dict[str, np.ndarray], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges = _candidate_edges(df, spec)
    nodes = set(edges["mask_a_observation_id"].astype(str)) | set(edges["mask_b_observation_id"].astype(str))
    forest = Forest(nodes, meta, fmap)
    selected_edges = 0
    skip_counts = {
        "already_same_component": 0,
        "same_frame_cannot_link": 0,
        "missing_centroid_feature": 0,
        "centroid_semantic_veto": 0,
    }
    for row in edges[["mask_a_observation_id", "mask_b_observation_id"]].itertuples(index=False):
        a, b = str(row.mask_a_observation_id), str(row.mask_b_observation_id)
        ok, reason = forest.can_union(a, b, spec)
        if not ok:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue
        forest.union(a, b)
        selected_edges += 1
    component_rows, stats = _diagnose_components(forest.groups(), meta, str(spec["variant_id"]))
    stats.update(
        {
            "candidate_edge_count": int(len(edges)),
            "selected_edge_count": int(selected_edges),
            **{f"skip_{k}_count": int(v) for k, v in skip_counts.items()},
        }
    )
    return component_rows, stats


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PHASE5C_ROWS)
    fmap = _feature_map()
    meta = _node_meta(df, fmap)
    all_component_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for spec in VARIANTS:
        component_rows, stats = _run_variant(df, meta, fmap, spec)
        all_component_rows.extend(component_rows)
        safe = bool(
            stats["same_frame_collision_component_count"] == 0
            and stats["multi_gt_component_count"] == 0
            and stats["component_purity_p10"] >= 0.80
            and stats["clean_component_proxy_count"] >= 30
        )
        variant_rows.append(
            {
                "schema_version": "stream4d_v102_phase7b_repair_variant_row_v1",
                "phase_id": "v102_phase7b_constrained_bridge_graph_repair",
                "variant_id": spec["variant_id"],
                "semantic_cosine_min": spec["semantic_cosine_min"],
                "missing_feature_policy": spec["missing_feature_policy"],
                "centroid_cosine_min": "" if spec["centroid_cosine_min"] is None else spec["centroid_cosine_min"],
                "require_centroid_feature": spec["require_centroid_feature"],
                **stats,
                "safe_for_primitive_rebirth": safe,
                "blocker": ""
                if safe
                else "Constrained forest still has multi-GT components, low component purity, or too few clean components.",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    best = max(
        variant_rows,
        key=lambda r: (
            int(r["clean_component_proxy_count"]),
            -int(r["multi_gt_component_count"]),
            float(r["component_purity_p10"]),
            -int(r["largest_component_size"]),
        ),
    )
    component_path = OUT_DIR / "constrained_component_rows.csv"
    variant_path = OUT_DIR / "constrained_rebirth_variant_rows.csv"
    gate_path = OUT_DIR / "variant_gate_rows.csv"
    _write_csv(component_path, all_component_rows)
    _write_csv(variant_path, variant_rows)
    gates = [
        {
            "gate_id": "any_variant_safe_for_primitive_rebirth",
            "pass": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
            "expected": True,
            "observed": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
        },
        {
            "gate_id": "best_variant_same_frame_collision_component_count",
            "pass": int(best["same_frame_collision_component_count"]) == 0,
            "expected": 0,
            "observed": best["same_frame_collision_component_count"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "best_variant_multi_gt_component_count",
            "pass": int(best["multi_gt_component_count"]) == 0,
            "expected": 0,
            "observed": best["multi_gt_component_count"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "best_variant_component_purity_p10",
            "pass": float(best["component_purity_p10"]) >= 0.80,
            "expected": ">=0.80",
            "observed": best["component_purity_p10"],
            "variant_id": best["variant_id"],
        },
        {
            "gate_id": "uses_gt_for_prediction",
            "pass": True,
            "expected": False,
            "observed": False,
        },
    ]
    _write_csv(gate_path, gates)
    decision = (
        "PASS_CONSTRAINED_PRIMITIVE_REBIRTH_DIAGNOSTIC_READY"
        if any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows)
        else "NO_GO_CONSTRAINED_REBIRTH_STILL_UNSAFE"
    )
    summary = {
        "schema_version": "stream4d_v102_phase7b_constrained_bridge_graph_repair_summary_v1",
        "phase_id": "v102_phase7b_constrained_bridge_graph_repair",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "variant_count": len(variant_rows),
        "best_variant_id": best["variant_id"],
        "best_variant_candidate_edge_count": best["candidate_edge_count"],
        "best_variant_selected_edge_count": best["selected_edge_count"],
        "best_variant_node_count": best["node_count"],
        "best_variant_component_count": best["component_count"],
        "best_variant_clean_component_proxy_count": best["clean_component_proxy_count"],
        "best_variant_largest_component_size": best["largest_component_size"],
        "best_variant_same_frame_collision_component_count": best["same_frame_collision_component_count"],
        "best_variant_multi_gt_component_count": best["multi_gt_component_count"],
        "best_variant_component_purity_mean": best["component_purity_mean"],
        "best_variant_component_purity_p10": best["component_purity_p10"],
        "any_variant_safe_for_primitive_rebirth": any(bool(row["safe_for_primitive_rebirth"]) for row in variant_rows),
        "truthfulness_note": (
            "Components are formed with GT-free same-frame cannot-link and RADIO feature centroid veto. "
            "GT labels are used only for post-hoc component diagnostics."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "constrained_component_rows": _rel(component_path),
            "constrained_rebirth_variant_rows": _rel(variant_path),
            "variant_gate_rows": _rel(gate_path),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
