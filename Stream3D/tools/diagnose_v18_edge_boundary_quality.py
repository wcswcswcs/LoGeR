from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list
from stream4d.scannet_stream import ScanNetStream
from stream4d.signed_boundary_evidence import SignedBoundaryEvidence
from stream4d.signed_graph_partition import _mask_votes
from stream4d.signed_surfel_graph import SignedSurfelGraph
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _load_scene_points(stream: ScanNetStream) -> np.ndarray:
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)


def _backproject_uv_to_mesh(
    stream: ScanNetStream,
    scene_points: np.ndarray,
    scene_tree: Any,
    frame_id: int,
    uv: np.ndarray,
    *,
    nn_radius: float,
) -> np.ndarray:
    out = np.full((uv.shape[0],), -1, dtype=np.int64)
    if uv.size == 0:
        return out
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    if not np.isfinite(pose).all():
        return out
    intr = stream.load_intrinsics()
    h, w = depth.shape
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return out
    z = depth[y[in_bounds], x[in_bounds]]
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return out
    x_valid = x[in_bounds][valid].astype(np.float32)
    y_valid = y[in_bounds][valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    cam = np.stack(
        [(x_valid - cx) * z_valid / fx, (y_valid - cy) * z_valid / fy, z_valid, np.ones_like(z_valid)],
        axis=1,
    )
    world = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(world).all(axis=1)
    if not np.any(finite):
        return out
    dist, idx = scene_tree.query(world[finite], k=1, distance_upper_bound=float(nn_radius))
    hit = np.isfinite(dist) & (idx < scene_points.shape[0])
    original_indices = np.flatnonzero(in_bounds)[valid][finite]
    out[original_indices[hit]] = idx[hit].astype(np.int64)
    return out


def surfel_gt_labels(
    bank: MeasurementBank,
    *,
    scannet_root: str,
    backbone: str,
    gt_root: str,
    nn_radius: float,
    max_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    stream = ScanNetStream(seq_name=bank.scene, backbone=backbone, root=scannet_root)
    scene_points = _load_scene_points(stream)
    from scipy.spatial import cKDTree

    scene_tree = cKDTree(scene_points)
    gt_ids = np.loadtxt(Path(gt_root) / f"{bank.scene}.txt", dtype=np.int64)
    if gt_ids.shape[0] != scene_points.shape[0]:
        raise RuntimeError(f"GT/mesh vertex count mismatch for {bank.scene}: {gt_ids.shape[0]} vs {scene_points.shape[0]}")

    labels_by_surfel: list[list[int]] = [[] for _ in range(bank.num_surfels)]
    visible = np.asarray(bank.visible_ok, dtype=bool)
    frame_ids = np.asarray(bank.frame_ids, dtype=np.int64)
    frames = range(min(int(max_frames), frame_ids.shape[0])) if max_frames > 0 else range(frame_ids.shape[0])
    total_queries = 0
    total_hits = 0
    for frame_idx in frames:
        surfels = np.flatnonzero(visible[frame_idx])
        if surfels.size == 0:
            continue
        mesh_ids = _backproject_uv_to_mesh(
            stream,
            scene_points,
            scene_tree,
            int(frame_ids[frame_idx]),
            np.asarray(bank.uv_pred[frame_idx, surfels], dtype=np.float32),
            nn_radius=float(nn_radius),
        )
        total_queries += int(mesh_ids.shape[0])
        hit = mesh_ids >= 0
        total_hits += int(np.count_nonzero(hit))
        for surfel_idx, mesh_idx in zip(surfels[hit].tolist(), mesh_ids[hit].tolist()):
            label = int(gt_ids[int(mesh_idx)])
            if label >= 1000:
                labels_by_surfel[int(surfel_idx)].append(label)

    labels = np.full((bank.num_surfels,), -1, dtype=np.int64)
    votes = np.zeros((bank.num_surfels,), dtype=np.int16)
    for idx, vals in enumerate(labels_by_surfel):
        if not vals:
            continue
        label, count = Counter(vals).most_common(1)[0]
        labels[idx] = int(label)
        votes[idx] = int(count)
    diag = {
        "scene": bank.scene,
        "surfel_gt_label_coverage": float(np.count_nonzero(labels >= 1000) / max(bank.num_surfels, 1)),
        "surfel_gt_vote_mean": float(np.mean(votes[votes > 0])) if np.any(votes > 0) else 0.0,
        "backproject_queries": int(total_queries),
        "backproject_hits": int(total_hits),
        "backproject_hit_rate": float(total_hits / max(total_queries, 1)),
        "nn_radius": float(nn_radius),
        "max_frames": int(max_frames),
        "uses_gt_for_diagnostic": True,
        "uses_gt_for_prediction": False,
    }
    return labels, diag


def _edge_labels(graph: SignedSurfelGraph, surfel_gt: np.ndarray) -> dict[str, Any]:
    a = surfel_gt[graph.src]
    b = surfel_gt[graph.dst]
    known = (a >= 1000) & (b >= 1000)
    same = known & (a == b)
    cut = known & (a != b)
    return {
        "known": known,
        "same": same,
        "cut": cut,
        "edge_gt_label_coverage": float(np.count_nonzero(known) / max(graph.num_edges, 1)),
        "same_gt_edges": int(np.count_nonzero(same)),
        "cut_gt_edges": int(np.count_nonzero(cut)),
        "unknown_gt_edges": int(graph.num_edges - np.count_nonzero(known)),
    }


def _auc_ap(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"edge_cut_AUC": None, "edge_cut_AP": None}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "edge_cut_AUC": float(roc_auc_score(y_true, score)),
            "edge_cut_AP": float(average_precision_score(y_true, score)),
        }
    except Exception:
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(order.shape[0])
        pos = y_true == 1
        neg = ~pos
        auc = float((np.sum(ranks[pos]) - np.count_nonzero(pos) * (np.count_nonzero(pos) - 1) / 2) / max(np.count_nonzero(pos) * np.count_nonzero(neg), 1))
        sorted_true = y_true[np.argsort(-score, kind="mergesort")]
        tp = np.cumsum(sorted_true == 1)
        prec = tp / np.maximum(np.arange(sorted_true.shape[0]) + 1, 1)
        ap = float(np.sum(prec[sorted_true == 1]) / max(np.count_nonzero(pos), 1))
        return {"edge_cut_AUC": auc, "edge_cut_AP": ap}


def _edge_quality_row(scene: str, graph: SignedSurfelGraph, evidence: SignedBoundaryEvidence, labels: dict[str, Any]) -> dict[str, Any]:
    known = labels["known"]
    cut_gt = labels["cut"]
    same_gt = labels["same"]
    y = cut_gt[known].astype(np.int32)
    score = evidence.cut_score[known]
    metrics = _auc_ap(y, score)
    row: dict[str, Any] = {
        "scene": scene,
        "variant": evidence.variant,
        "num_edges": int(graph.num_edges),
        "num_edges_evaluated": int(np.count_nonzero(known)),
        "unknown_edge_ratio": float(1.0 - np.count_nonzero(known) / max(graph.num_edges, 1)),
        "same_gt_edges": int(np.count_nonzero(same_gt)),
        "cut_gt_edges": int(np.count_nonzero(cut_gt)),
        "mean_cut_score_same_GT": float(np.mean(evidence.cut_score[same_gt])) if np.any(same_gt) else None,
        "mean_cut_score_different_GT": float(np.mean(evidence.cut_score[cut_gt])) if np.any(cut_gt) else None,
        "score_separation_margin": None,
        **metrics,
    }
    if row["mean_cut_score_same_GT"] is not None and row["mean_cut_score_different_GT"] is not None:
        row["score_separation_margin"] = float(row["mean_cut_score_different_GT"] - row["mean_cut_score_same_GT"])
    if np.any(known):
        known_scores = evidence.cut_score[known]
        known_cut = cut_gt[known]
        for pct in (1, 5, 10):
            k = max(1, int(np.ceil(known_scores.shape[0] * pct / 100.0)))
            top = np.argsort(-known_scores, kind="mergesort")[:k]
            row[f"precision_at_top_{pct}_percent_cut_edges"] = float(np.mean(known_cut[top]))
        top10 = np.argsort(-known_scores, kind="mergesort")[: max(1, int(np.ceil(known_scores.shape[0] * 0.10)))]
        row["GT_boundary_recall_at_top_10_percent"] = float(np.count_nonzero(known_cut[top10]) / max(np.count_nonzero(known_cut), 1))
    else:
        for pct in (1, 5, 10):
            row[f"precision_at_top_{pct}_percent_cut_edges"] = None
        row["GT_boundary_recall_at_top_10_percent"] = None
    if np.any(same_gt):
        row["false_cut_rate_inside_same_GT"] = float(np.mean(evidence.cut_score[same_gt] >= 0.5))
    else:
        row["false_cut_rate_inside_same_GT"] = None
    if np.any(cut_gt):
        row["false_merge_rate_across_GT"] = float(np.mean(evidence.cut_score[cut_gt] < 0.5))
    else:
        row["false_merge_rate_across_GT"] = None
    return row


def _oracle_components(graph: SignedSurfelGraph, surfel_gt: np.ndarray, min_surfels: int) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for a, b in zip(graph.src.tolist(), graph.dst.tolist()):
        if surfel_gt[int(a)] >= 1000 and surfel_gt[int(a)] == surfel_gt[int(b)]:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    seen = np.zeros((graph.num_nodes,), dtype=bool)
    comps: list[np.ndarray] = []
    for start in range(graph.num_nodes):
        if seen[start] or surfel_gt[start] < 1000:
            continue
        seen[start] = True
        q: deque[int] = deque([start])
        nodes: list[int] = []
        while q:
            node = q.popleft()
            nodes.append(int(node))
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        if len(nodes) >= int(min_surfels):
            comps.append(np.asarray(nodes, dtype=np.int64))
    comps.sort(key=lambda arr: (-arr.shape[0], int(arr[0]) if arr.size else -1))
    return comps


def _gt_coverage(bank: MeasurementBank, graph: SignedSurfelGraph, surfel_gt: np.ndarray, edge_labels: dict[str, Any]) -> dict[str, Any]:
    gt_valid = surfel_gt[surfel_gt >= 1000]
    counts = Counter(int(v) for v in gt_valid.tolist())
    return {
        "node_gt_label_coverage": float(np.count_nonzero(surfel_gt >= 1000) / max(bank.num_surfels, 1)),
        "edge_gt_label_coverage": float(edge_labels["edge_gt_label_coverage"]),
        "valid_gt_instances_ge20_surfels": int(sum(1 for v in counts.values() if int(v) >= 20)),
        "valid_gt_instances_ge100_surfels": int(sum(1 for v in counts.values() if int(v) >= 100)),
        "num_gt_instances_touched": int(len(counts)),
        "per_GT_num_surfels_mean": float(np.mean(list(counts.values()))) if counts else 0.0,
        "per_GT_num_surfels_p10": float(np.percentile(list(counts.values()), 10)) if counts else 0.0,
        "per_GT_num_surfels_p90": float(np.percentile(list(counts.values()), 90)) if counts else 0.0,
        "same_gt_edges": int(edge_labels["same_gt_edges"]),
        "cut_gt_edges": int(edge_labels["cut_gt_edges"]),
        "unknown_gt_edges": int(edge_labels["unknown_gt_edges"]),
    }


def _export_oracle(
    *,
    bank: MeasurementBank,
    components: list[np.ndarray],
    output_config: str,
    backbone: str,
    min_points_per_object: int,
    export_core_nn_radius: float,
    export_fringe_nn_radius: float,
    export_fringe_radius: float,
    export_fringe_max_ratio: float,
) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=bank.scene, backbone=backbone)
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_support_mode="posterior_support",
        export_core_nn_radius=float(export_core_nn_radius),
        export_fringe_nn_radius=float(export_fringe_nn_radius),
        export_fringe_radius=float(export_fringe_radius),
        export_fringe_max_ratio=float(export_fringe_max_ratio),
        export_min_points_per_object=int(min_points_per_object),
        export_score_mode="observations",
    )
    object_dict: dict[int, dict[str, Any]] = {}
    for comp in components:
        mask_list = _mask_votes(bank, comp, max_votes=8)
        if not mask_list:
            continue
        object_dict[len(object_dict)] = {
            "mask_list": mask_list,
            "carrier_ids": comp,
            "core_surfels": comp,
            "fringe_surfels": np.empty((0,), dtype=np.int64),
            "unknown_surfels": np.empty((0,), dtype=np.int64),
            "reject_surfels": np.empty((0,), dtype=np.int64),
        }
    return exporter.export_object_slot_posterior_support(object_dict, bank)


def _parse_metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}


def _write_rows(prefix: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any], args: argparse.Namespace) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (dict, list))})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [f"# {prefix.name}", "", "## Aggregate", ""]
    for key, value in aggregate.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Rows", ""])
    if rows:
        keys = [k for k in rows[0].keys() if k in {"scene", "variant", "ap", "ap50", "ap25", "node_gt_label_coverage", "edge_gt_label_coverage", "edge_cut_AUC", "edge_cut_AP", "precision_at_top_10_percent_cut_edges", "false_cut_rate_inside_same_GT"}]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(k)) for k in keys) + " |")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_numeric(rows: list[dict[str, Any]], *, phase: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    out: dict[str, Any] = {
        "phase": phase,
        "num_rows": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }
    if metrics:
        out.update(metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--graph-root", default="outputs/audit/v18_phase1")
    parser.add_argument("--evidence-root", default="outputs/audit/v18_phase3")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--mode", choices=["oracle", "evidence"], required=True)
    parser.add_argument("--variant", default="E5_full_signed")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--gt-root", default="data/scannet/gt")
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--max-gt-label-frames", type=int, default=16)
    parser.add_argument("--oracle-output-config", default="stream4d_v18_edge_oracle_probe5")
    parser.add_argument("--oracle-min-surfels", type=int, default=20)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--oracle-export-core-nn-radius", type=float, default=0.05)
    parser.add_argument("--oracle-export-fringe-nn-radius", type=float, default=0.05)
    parser.add_argument("--oracle-export-fringe-radius", type=float, default=0.0)
    parser.add_argument("--oracle-export-fringe-max-ratio", type=float, default=0.35)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    scene_components: list[dict[str, Any]] = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
        graph = SignedSurfelGraph.load(Path(args.graph_root) / scene / "signed_surfel_graph.npz")
        labels, label_diag = surfel_gt_labels(
            bank,
            scannet_root=args.scannet_root,
            backbone=args.backbone,
            gt_root=args.gt_root,
            nn_radius=float(args.nn_radius),
            max_frames=int(args.max_gt_label_frames),
        )
        scene_dir = Path(args.output_prefix).parent / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "surfel_gt_labels.npy", labels)
        edge_labels = _edge_labels(graph, labels)
        if args.mode == "oracle":
            comps = _oracle_components(graph, labels, min_surfels=int(args.oracle_min_surfels))
            export_diag = _export_oracle(
                bank=bank,
                components=comps,
                output_config=args.oracle_output_config,
                backbone=args.backbone,
                min_points_per_object=int(args.min_points_per_object),
                export_core_nn_radius=float(args.oracle_export_core_nn_radius),
                export_fringe_nn_radius=float(args.oracle_export_fringe_nn_radius),
                export_fringe_radius=float(args.oracle_export_fringe_radius),
                export_fringe_max_ratio=float(args.oracle_export_fringe_max_ratio),
            )
            row = {
                **label_diag,
                **_gt_coverage(bank, graph, labels, edge_labels),
                **export_diag,
                "scene": scene,
                "num_oracle_components": int(len(comps)),
                "oracle_component_size_mean": float(np.mean([c.shape[0] for c in comps])) if comps else 0.0,
                "oracle_component_size_p90": float(np.percentile([c.shape[0] for c in comps], 90)) if comps else 0.0,
                "uses_gt_for_prediction": True,
                "uses_gt_for_diagnostic": True,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
            rows.append(row)
            scene_components.append({"scene": scene, "num_components": len(comps)})
        else:
            evidence = SignedBoundaryEvidence.load(Path(args.evidence_root) / args.variant / scene / "signed_boundary_evidence.npz")
            row = {
                **label_diag,
                **_gt_coverage(bank, graph, labels, edge_labels),
                **_edge_quality_row(scene, graph, evidence, edge_labels),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "is_diagnostic_only": True,
            }
            rows.append(row)

    metrics: dict[str, Any] = {}
    if args.mode == "oracle":
        manifest = build_prediction_manifest(
            root=".",
            output_config=args.oracle_output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=True,
            gt_usage="edge oracle partition diagnostic",
            source_configs=[args.graph_root, args.bank_root],
            pre_points_policy="recompute",
            support_policy="v18_edge_oracle_gt_same_components",
            notes="GT edge oracle diagnostic. Forbidden for method tables.",
            extra={
                "algorithm": "v18_edge_partition_oracle",
                "uses_gt_for_prediction": True,
                "uses_gt_for_diagnostic": True,
                "forbidden_for_method_table": True,
                "gt_selected_output": True,
                "eval_policy": "oracle_diagnostic_own_recompute",
            },
        )
        write_prediction_manifest(args.oracle_output_config, manifest, root=".", pred_suffix="class_agnostic")
        metric_path = Path("data/evaluation/scannet") / f"{args.oracle_output_config}_class_agnostic.txt"
        if not args.skip_eval:
            cmd = [
                sys.executable,
                "-m",
                "evaluation.evaluate",
                "--pred_path",
                f"data/prediction/{args.oracle_output_config}_class_agnostic",
                "--gt_path",
                args.gt_root,
                "--dataset",
                "scannet",
                "--output_file",
                str(metric_path),
                "--tmp_root",
                "data/TMP",
                "--tmp_config",
                args.oracle_output_config,
                "--no_class",
                "--require-manifest",
                "--allow-oracle-eval",
            ]
            subprocess.run(cmd, check=True)
        metrics = _parse_metric(metric_path)
        metrics["oracle_output_config"] = args.oracle_output_config
        numeric = _aggregate_numeric(rows, phase="phase2_oracle")["numeric_mean"]
        metrics["phase2_oracle_ap_gate"] = bool(
            (metrics.get("ap") or 0.0) >= 0.25
            and (metrics.get("ap50") or 0.0) >= 0.50
            and (metrics.get("ap25") or 0.0) >= 0.70
        )
        metrics["phase2_graph_coverage_gate"] = bool(
            numeric.get("node_gt_label_coverage", 0.0) >= 0.70
            and numeric.get("edge_gt_label_coverage", 0.0) >= 0.60
        )
        metrics["phase2_min_gate"] = bool(metrics["phase2_oracle_ap_gate"] and metrics["phase2_graph_coverage_gate"])
    else:
        numeric = _aggregate_numeric(rows, phase="phase3")["numeric_mean"]
        metrics["variant"] = args.variant
        metrics["phase3_min_gate"] = bool(
            numeric.get("edge_cut_AUC", 0.0) >= 0.70
            and numeric.get("edge_cut_AP", 0.0) >= 0.35
            and numeric.get("precision_at_top_10_percent_cut_edges", 0.0) >= 0.55
            and numeric.get("false_cut_rate_inside_same_GT", 1.0) <= 0.25
        )
    aggregate = _aggregate_numeric(rows, phase=f"phase2_{args.mode}" if args.mode == "oracle" else "phase3_evidence", metrics=metrics)
    _write_rows(Path(args.output_prefix), rows, aggregate, args)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
