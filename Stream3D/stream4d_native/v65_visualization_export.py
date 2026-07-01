from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import read_csv, write_json
from .v65_common import PROBE5_SCENES, float_or_none, load_dict, project, rel, sha256_file, tiny_png, write_standard_outputs


VIS_ROOT = "outputs/audit/v65_visualization"
CASE_ROOT = "outputs/audit/v65_casebook"
D4RT_DEBUG_ROOT = "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1"

VIS_VARIANTS = {
    "A3_bridge_prediction_union": "v64r2_probe5_v53_bridge_wta",
    "A4_bridge_used_frame_support": "v64r2_probe5_v53_bridge_wta_used_support",
    "A5_d4rt_g11_prediction_union": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
    "A7_d4rt_g12_prediction_union": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12",
}


def check_viser_import() -> dict[str, Any]:
    try:
        import viser  # type: ignore

        return {
            "viser_import_ok": True,
            "viser_version": getattr(viser, "__version__", "unknown"),
            "python_executable": sys.executable,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {
            "viser_import_ok": False,
            "viser_version": "",
            "python_executable": sys.executable,
            "error": f"{type(exc).__name__}: {exc}",
        }


def export_v65_visualization(output_root: str | Path = VIS_ROOT) -> dict[str, Any]:
    root = project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    viser_status = check_viser_import()
    scene_rows: list[dict[str, Any]] = []
    screenshot_count = 0
    for scene in PROBE5_SCENES:
        scene_payload = _export_scene(root, scene)
        scene_rows.append(scene_payload)
        screenshot_count += int(scene_payload.get("screenshot_count") or 0)
    summary = {
        "phase": "v65_visualization_export",
        **viser_status,
        "scene_count": len(scene_rows),
        "fallback_export_available": all(row.get("viewer_data_exists") for row in scene_rows),
        "bookmarked_screenshot_count": screenshot_count,
        "coordinate_frame_note": "D4RT xyz_ref carrier samples are real debug-cache coordinates; AP support and GT layers are exported as evaluator index-space diagnostics when mesh vertex coordinates are unavailable.",
        "ownership_note": "SOMA material_state rows exist as tabular ownership summaries, but xyz_tracks are empty and AP prediction ids are not linked to history/material/component ids; this blocks a true 3D ownership layer.",
        "gate": {
            "at_least_5_scenes_load_in_viser": bool(viser_status["viser_import_ok"] and len(scene_rows) >= 5),
            "D4RT_geometry_layer_exported": all(row.get("d4rt_xyz_point_count", 0) > 0 for row in scene_rows),
            "SOMA_ownership_summary_layer_exported": all(row.get("ownership_summary_exists") for row in scene_rows),
            "SOMA_semantic_ownership_3d_layer_visible": False,
            "AP_support_scopes_exported": all(row.get("ap_support_scope_count", 0) >= 4 for row in scene_rows),
            "GT_diagnostic_layer_available": all(row.get("gt_label_count", 0) > 0 for row in scene_rows),
            "bookmarked_screenshot_count_ge_20": screenshot_count >= 20,
        },
        "scene_rows": scene_rows,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    summary["visualization_status"] = "GO_VISER_3D_DIAGNOSIS" if summary["gate"]["pass"] else "NO_GO_VISUALIZATION"
    write_json(root / "viser_scene_index.json", summary)
    return {"summary": summary, "scene_rows": scene_rows}


def run_v65_viser_server_status(output_root: str | Path = VIS_ROOT) -> dict[str, Any]:
    root = project(output_root)
    status = check_viser_import()
    smoke: dict[str, Any] = {
        "server_smoke_ran": False,
        "server_started": False,
        "scene_point_clouds_added": 0,
        "smoke_error": "",
    }
    if status["viser_import_ok"]:
        try:
            import viser  # type: ignore

            server = viser.ViserServer(port=0, verbose=False)
            smoke["server_smoke_ran"] = True
            smoke["server_started"] = True
            for scene_dir in sorted(path for path in root.iterdir() if path.is_dir())[:5]:
                data_path = scene_dir / "viewer_data.npz"
                if not data_path.exists():
                    continue
                with np.load(data_path) as payload:
                    points = np.asarray(payload["d4rt_xyz_ref"], dtype=np.float32)
                    window = np.asarray(payload["d4rt_window_id"], dtype=np.int64)
                if points.shape[0] == 0:
                    continue
                points = points[:2000]
                window = window[: points.shape[0]]
                colors = _window_colors(window)
                server.scene.add_point_cloud(
                    f"/v65/{scene_dir.name}/d4rt_xyz_ref",
                    points=points,
                    colors=colors,
                    point_size=0.025,
                    point_shape="circle",
                )
                smoke["scene_point_clouds_added"] = int(smoke["scene_point_clouds_added"]) + 1
            server.stop()
        except Exception as exc:  # pragma: no cover - depends on local display/network env
            smoke["smoke_error"] = f"{type(exc).__name__}: {exc}"
    status.update(
        {
            "phase": "v65_viser_server_status",
            **smoke,
            "reason": ""
            if status["viser_import_ok"]
            else "viser import failed; fallback exports are available but no interactive server was started.",
            "scene_index": rel(root / "viser_scene_index.json"),
            "gate": {
                "viser_import_ok": status["viser_import_ok"],
                "server_smoke_ran": smoke["server_smoke_ran"],
                "at_least_5_scene_point_clouds_added": int(smoke["scene_point_clouds_added"]) >= 5,
                "smoke_error_empty": smoke["smoke_error"] == "",
            },
        }
    )
    status["gate"]["pass"] = bool(all(status["gate"].values()))
    write_json(root / "viser_server_status.json", status)
    return status


def serve_v65_viser(
    output_root: str | Path = VIS_ROOT,
    *,
    host: str = "0.0.0.0",
    port: int = 8081,
    scene_id: str | None = "scene0050_00",
    pred_config: str = "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
    d4rt_debug_root: str | Path = D4RT_DEBUG_ROOT,
    d4rt_mode: str = "self_stitched_scale_normalized_eval_sim3",
    max_d4rt_points: int = 0,
) -> dict[str, Any]:
    root = project(output_root)
    status = check_viser_import()
    if not status["viser_import_ok"]:
        raise RuntimeError(f"viser import failed in {status['python_executable']}: {status['error']}")

    import viser  # type: ignore

    if scene_id and scene_id != "all":
        scene_dir = root / scene_id
        data_path = scene_dir / "viewer_data.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing v65 viewer data for scene {scene_id}: {data_path}")
        scene_dirs = [scene_dir]
    else:
        scene_dirs = [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "viewer_data.npz").exists()]
    server = viser.ViserServer(host=host, port=port, verbose=True)
    layer_handles: dict[str, list[Any]] = {
        "gt_geometry": [],
        "gt_sem": [],
        "pred_geometry": [],
        "pred_sem": [],
        "d4rt_raw": [],
    }
    gui_controls: list[str] = []

    def _add_layer(layer: str, handle: Any, *, visible: bool = True) -> None:
        handle.visible = bool(visible)
        layer_handles.setdefault(layer, []).append(handle)

    def _set_layer(layer: str, visible: bool) -> None:
        for handle in layer_handles.get(layer, []):
            handle.visible = bool(visible)

    live_status: dict[str, Any] = {
        "phase": "v65_live_viser_server",
        **status,
        "selected_scene": scene_id or "all",
        "pred_config": pred_config,
        "d4rt_debug_root": rel(d4rt_debug_root),
        "d4rt_alignment_mode": d4rt_mode,
        "max_d4rt_points": int(max_d4rt_points),
        "host": host,
        "port": port,
        "url": f"http://localhost:{port}",
        "bind_url": f"http://{host}:{port}",
        "scene_count": 0,
        "scene_point_clouds_added": 0,
        "layer_count": 0,
        "point_count_total": 0,
        "scene_rows": [],
        "required_primary_visual_layers": ["gt_geometry", "gt_sem", "pred_geometry", "pred_sem"],
        "support_scope_note": "Pred semantic AP masks are rendered from the exact pred_masks file used by the AP evaluator; GT sem is diagnostic-only.",
        "ownership_note": "SOMA ownership summary is shown as text because material_state rows have no usable xyz_tracks/AP join key for a true 3D ownership layer.",
        "visual_contract_note": "Primary layers are GT geometry, GT sem, Pred geometry, Pred sem. Pred geometry is D4RT provider Sim3-aligned geometry for the AP/SOMA config; Pred sem is the SOMA/AP pred_masks overlay.",
    }
    server.scene.add_grid(
        "/v65/grid",
        width=max(8.0, float(len(scene_dirs)) * 8.0),
        height=8.0,
        plane="xy",
        cell_size=1.0,
        section_size=4.0,
        position=(0.0, 0.0, -0.05),
    )
    server.scene.add_label(
        "/v65/title",
        f"v65 live Viser: {scene_id or 'all'} GT geometry/sem + Pred D4RT geometry/SOMA sem ({pred_config})",
        position=(0.0, -3.2, 2.8),
        font_screen_scale=1.2,
        anchor="top-left",
    )
    for scene_index, scene_dir in enumerate(scene_dirs):
        offset = np.asarray([float(scene_index) * 8.0, 0.0, 0.0], dtype=np.float32)
        scene_points, scene_colors, mesh_path = _load_scene_mesh(scene_dir.name)
        gt_labels = _load_gt(scene_dir.name)
        if gt_labels.shape[0] != scene_points.shape[0]:
            raise ValueError(
                f"GT/mesh length mismatch for {scene_dir.name}: gt={gt_labels.shape[0]} mesh={scene_points.shape[0]}"
            )
        with np.load(scene_dir / "viewer_data.npz") as payload:
            raw_points = np.asarray(payload["d4rt_xyz_ref"], dtype=np.float32)
            raw_window = np.asarray(payload["d4rt_window_id"], dtype=np.int64)
            support_counts = {
                key.removeprefix("support_"): int(np.count_nonzero(payload[key]))
                for key in payload.files
                if key.startswith("support_")
            }
        aligned_points, aligned_windows, aligned_diag = _load_aligned_d4rt_scene_points(
            scene_dir.name,
            debug_root=d4rt_debug_root,
            mode=d4rt_mode,
            max_points=max_d4rt_points,
        )
        pred_manifest = load_dict(project("data/prediction") / f"{pred_config}_class_agnostic" / "config_manifest.json")
        gt_geometry_handle = server.scene.add_point_cloud(
            f"/v65/{scene_dir.name}/gt_geometry_scannet_mesh_rgb",
            points=scene_points + offset,
            colors=scene_colors,
            point_size=0.006,
            point_shape="circle",
            precision="float32",
        )
        _add_layer("gt_geometry", gt_geometry_handle, visible=True)
        live_status["layer_count"] = int(live_status["layer_count"]) + 1
        gt_positive = gt_labels > 0
        gt_positive_points = scene_points[gt_positive]
        if gt_positive_points.shape[0] > 0:
            gt_sem_handle = server.scene.add_point_cloud(
                f"/v65/{scene_dir.name}/gt_sem_instance_labels",
                points=gt_positive_points + offset,
                colors=_id_colors(gt_labels[gt_positive]),
                point_size=0.011,
                point_shape="circle",
                precision="float32",
            )
            _add_layer("gt_sem", gt_sem_handle, visible=True)
            live_status["layer_count"] = int(live_status["layer_count"]) + 1
        pred_overlay = _load_prediction_overlay(scene_dir.name, pred_config, scene_points.shape[0])
        pred_points = np.asarray(pred_overlay.get("_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        pred_colors = np.asarray(pred_overlay.get("_colors", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        if pred_points.shape[0] > 0:
            pred_sem_handle = server.scene.add_point_cloud(
                f"/v65/{scene_dir.name}/pred_sem_soma_ap_masks_{pred_config}",
                points=pred_points + offset,
                colors=pred_colors,
                point_size=0.018,
                point_shape="circle",
                precision="float32",
            )
            _add_layer("pred_sem", pred_sem_handle, visible=True)
            live_status["layer_count"] = int(live_status["layer_count"]) + 1
        if aligned_points.shape[0] > 0:
            colors = _window_colors(aligned_windows[: aligned_points.shape[0]])
            pred_geometry_handle = server.scene.add_point_cloud(
                f"/v65/{scene_dir.name}/pred_geometry_d4rt_sim3_aligned_carriers",
                points=aligned_points + offset,
                colors=colors,
                point_size=0.018,
                point_shape="circle",
                precision="float32",
            )
            _add_layer("pred_geometry", pred_geometry_handle, visible=True)
            live_status["layer_count"] = int(live_status["layer_count"]) + 1
        if raw_points.shape[0] > 0:
            colors = _window_colors(raw_window[: raw_points.shape[0]])
            raw_handle = server.scene.add_point_cloud(
                f"/v65/{scene_dir.name}/d4rt_raw_xyz_ref_sample_audit",
                points=raw_points + offset,
                colors=colors,
                point_size=0.025,
                point_shape="circle",
                precision="float32",
            )
            _add_layer("d4rt_raw", raw_handle, visible=False)
            live_status["layer_count"] = int(live_status["layer_count"]) + 1
        centroid_handle = server.scene.add_point_cloud(
            f"/v65/{scene_dir.name}/gt_sem_instance_centroids",
            points=_gt_centroids(scene_points, gt_labels) + offset,
            colors=_id_colors(np.unique(gt_labels[gt_labels > 0])),
            point_size=0.06,
            point_shape="sparkle",
            precision="float32",
        )
        _add_layer("gt_sem", centroid_handle, visible=True)
        live_status["layer_count"] = int(live_status["layer_count"]) + 1
        server.scene.add_frame(
            f"/v65/{scene_dir.name}/origin",
            position=tuple(float(x) for x in offset),
            axes_length=0.35,
            axes_radius=0.012,
        )
        ownership_path = scene_dir / "ownership_summary.json"
        ownership = load_dict(ownership_path) if ownership_path.exists() else {}
        label_lines = [
            scene_dir.name,
            f"GT geometry vertices: {scene_points.shape[0]}",
            f"GT sem vertices/instances: {int(np.count_nonzero(gt_positive))}/{int(np.unique(gt_labels[gt_labels > 0]).shape[0])}",
            f"Pred geometry D4RT aligned points: {aligned_points.shape[0]}",
            f"Pred sem SOMA vertices/instances: {pred_overlay.get('pred_vertex_count', 0)}/{pred_overlay.get('pred_instance_count', 0)}",
            f"D4RT raw sample hidden: {raw_points.shape[0]}",
            f"ownership rows: {ownership.get('material_count', 0)}; 3D ownership blocked",
        ]
        server.scene.add_label(
            f"/v65/{scene_dir.name}/audit_label",
            "\n".join(label_lines),
            position=tuple(float(x) for x in offset + np.asarray([-2.5, 2.2, 2.0], dtype=np.float32)),
            font_screen_scale=0.85,
            anchor="top-left",
        )
        live_status["scene_rows"].append(
            {
                "scene_id": scene_dir.name,
                "viewer_data": rel(scene_dir / "viewer_data.npz"),
                "mesh_path": rel(mesh_path),
                "gt_path": rel(project("data/scannet/gt") / f"{scene_dir.name}.txt"),
                "primary_visual_layers": [
                    {
                        "layer_key": "gt_geometry",
                        "viser_path": f"/v65/{scene_dir.name}/gt_geometry_scannet_mesh_rgb",
                        "visible": True,
                        "point_count": int(scene_points.shape[0]),
                        "source": rel(mesh_path),
                        "semantics": "geometry only, ScanNet scene mesh RGB",
                    },
                    {
                        "layer_key": "gt_sem",
                        "viser_path": f"/v65/{scene_dir.name}/gt_sem_instance_labels",
                        "visible": bool(gt_positive_points.shape[0] > 0),
                        "point_count": int(np.count_nonzero(gt_positive)),
                        "source": rel(project("data/scannet/gt") / f"{scene_dir.name}.txt"),
                        "instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
                        "semantics": "GT instance/semantic diagnostic labels on scene geometry",
                    },
                    {
                        "layer_key": "pred_geometry",
                        "viser_path": f"/v65/{scene_dir.name}/pred_geometry_d4rt_sim3_aligned_carriers",
                        "visible": bool(aligned_points.shape[0] > 0),
                        "point_count": int(aligned_points.shape[0]),
                        "source": rel(d4rt_debug_root),
                        "geometry_source": pred_manifest.get("geometry_source", ""),
                        "provider_mode": d4rt_mode,
                        "semantics": "D4RT geometry used by the AP/SOMA prediction config, Sim3-aligned for visual diagnosis",
                    },
                    {
                        "layer_key": "pred_sem",
                        "viser_path": f"/v65/{scene_dir.name}/pred_sem_soma_ap_masks_{pred_config}",
                        "visible": bool(pred_points.shape[0] > 0),
                        "point_count": int(pred_points.shape[0]),
                        "source": pred_overlay.get("pred_path", ""),
                        "source_sha256": pred_overlay.get("pred_path_sha256", ""),
                        "instance_count": pred_overlay.get("pred_instance_count", 0),
                        "semantics": "SOMA/AP pred_masks overlay used by the evaluator",
                    },
                ],
                "full_scene_vertex_count": int(scene_points.shape[0]),
                "gt_positive_vertex_count": int(np.count_nonzero(gt_positive)),
                "gt_instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
                "d4rt_raw_sample_point_count": int(raw_points.shape[0]),
                "d4rt_aligned_point_count": int(aligned_points.shape[0]),
                "d4rt_alignment_applied": bool(aligned_points.shape[0] > 0),
                "d4rt_alignment_mode": d4rt_mode,
                "d4rt_alignment_source": "D4RTCarrierProjectionProvider",
                "d4rt_uses_eval_sim3_for_visual_alignment": "eval_sim3" in d4rt_mode,
                "d4rt_alignment_diag": aligned_diag,
                "pred_config": pred_config,
                "pred_manifest_path": rel(project("data/prediction") / f"{pred_config}_class_agnostic" / "config_manifest.json"),
                "pred_manifest_sha256": sha256_file(project("data/prediction") / f"{pred_config}_class_agnostic" / "config_manifest.json"),
                "pred_manifest_geometry_source": pred_manifest.get("geometry_source", ""),
                "pred_path": pred_overlay.get("pred_path", ""),
                "pred_path_sha256": pred_overlay.get("pred_path_sha256", ""),
                "pred_pre_points_path": pred_overlay.get("pre_points_path", ""),
                "pred_pre_points_path_sha256": pred_overlay.get("pre_points_path_sha256", ""),
                "pred_pre_points_count": pred_overlay.get("pre_points_count", 0),
                "pred_vertex_count": pred_overlay.get("pred_vertex_count", 0),
                "pred_instance_count": pred_overlay.get("pred_instance_count", 0),
                "pred_mask_contract": pred_overlay.get("mask_contract", ""),
                "pred_error": pred_overlay.get("error", ""),
                "point_count": int(scene_points.shape[0]),
                "support_counts": support_counts,
                "ownership_summary": rel(ownership_path) if ownership_path.exists() else "",
                "ownership_material_count": int(ownership.get("material_count", 0) or 0),
                "ownership_3d_layer_visible": False,
                "gt_geometry_layer_visible": True,
                "gt_sem_layer_visible": bool(gt_positive_points.shape[0] > 0),
                "pred_geometry_layer_visible": bool(aligned_points.shape[0] > 0),
                "pred_sem_layer_visible": bool(pred_points.shape[0] > 0),
                "full_scene_layer_visible": True,
                "gt_diagnostic_layer_visible": bool(gt_positive_points.shape[0] > 0),
                "pred_layer_visible": bool(pred_points.shape[0] > 0),
                "d4rt_sim3_aligned_layer_visible": bool(aligned_points.shape[0] > 0),
                "d4rt_raw_sample_layer_visible": False,
            }
        )
        live_status["scene_count"] = int(live_status["scene_count"]) + 1
        live_status["scene_point_clouds_added"] = int(live_status["scene_point_clouds_added"]) + 1
        live_status["point_count_total"] = (
            int(live_status["point_count_total"])
            + int(scene_points.shape[0])
            + int(gt_positive_points.shape[0])
            + int(pred_points.shape[0])
            + int(aligned_points.shape[0])
        )
    with server.gui.add_folder("v65 layer controls"):
        gt_geometry_cb = server.gui.add_checkbox("GT geometry", initial_value=True)
        gt_sem_cb = server.gui.add_checkbox("GT sem", initial_value=True)
        pred_geometry_cb = server.gui.add_checkbox("Pred geometry", initial_value=True)
        pred_sem_cb = server.gui.add_checkbox("Pred sem", initial_value=True)
        raw_cb = server.gui.add_checkbox("Audit raw D4RT sample", initial_value=False)
        gt_geometry_on = server.gui.add_button("GT geometry on")
        gt_geometry_off = server.gui.add_button("GT geometry off")
        gt_sem_on = server.gui.add_button("GT sem on")
        gt_sem_off = server.gui.add_button("GT sem off")
        pred_geometry_on = server.gui.add_button("Pred geometry on")
        pred_geometry_off = server.gui.add_button("Pred geometry off")
        pred_sem_on = server.gui.add_button("Pred sem on")
        pred_sem_off = server.gui.add_button("Pred sem off")
        show_four = server.gui.add_button("Show 4 primary")
        hide_four = server.gui.add_button("Hide 4 primary")
    gui_controls.extend(
        [
            "checkbox:GT geometry",
            "checkbox:GT sem",
            "checkbox:Pred geometry",
            "checkbox:Pred sem",
            "checkbox:Audit raw D4RT sample",
            "button:GT geometry on",
            "button:GT geometry off",
            "button:GT sem on",
            "button:GT sem off",
            "button:Pred geometry on",
            "button:Pred geometry off",
            "button:Pred sem on",
            "button:Pred sem off",
            "button:Show 4 primary",
            "button:Hide 4 primary",
        ]
    )

    @gt_geometry_cb.on_update
    def _(_event: Any) -> None:
        _set_layer("gt_geometry", bool(gt_geometry_cb.value))

    @gt_sem_cb.on_update
    def _(_event: Any) -> None:
        _set_layer("gt_sem", bool(gt_sem_cb.value))

    @pred_geometry_cb.on_update
    def _(_event: Any) -> None:
        _set_layer("pred_geometry", bool(pred_geometry_cb.value))

    @pred_sem_cb.on_update
    def _(_event: Any) -> None:
        _set_layer("pred_sem", bool(pred_sem_cb.value))

    @raw_cb.on_update
    def _(_event: Any) -> None:
        _set_layer("d4rt_raw", bool(raw_cb.value))

    @gt_geometry_on.on_click
    def _(_event: Any) -> None:
        gt_geometry_cb.value = True
        _set_layer("gt_geometry", True)

    @gt_geometry_off.on_click
    def _(_event: Any) -> None:
        gt_geometry_cb.value = False
        _set_layer("gt_geometry", False)

    @gt_sem_on.on_click
    def _(_event: Any) -> None:
        gt_sem_cb.value = True
        _set_layer("gt_sem", True)

    @gt_sem_off.on_click
    def _(_event: Any) -> None:
        gt_sem_cb.value = False
        _set_layer("gt_sem", False)

    @pred_geometry_on.on_click
    def _(_event: Any) -> None:
        pred_geometry_cb.value = True
        _set_layer("pred_geometry", True)

    @pred_geometry_off.on_click
    def _(_event: Any) -> None:
        pred_geometry_cb.value = False
        _set_layer("pred_geometry", False)

    @pred_sem_on.on_click
    def _(_event: Any) -> None:
        pred_sem_cb.value = True
        _set_layer("pred_sem", True)

    @pred_sem_off.on_click
    def _(_event: Any) -> None:
        pred_sem_cb.value = False
        _set_layer("pred_sem", False)

    @show_four.on_click
    def _(_event: Any) -> None:
        gt_geometry_cb.value = True
        gt_sem_cb.value = True
        pred_geometry_cb.value = True
        pred_sem_cb.value = True
        for layer in ["gt_geometry", "gt_sem", "pred_geometry", "pred_sem"]:
            _set_layer(layer, True)

    @hide_four.on_click
    def _(_event: Any) -> None:
        gt_geometry_cb.value = False
        gt_sem_cb.value = False
        pred_geometry_cb.value = False
        pred_sem_cb.value = False
        for layer in ["gt_geometry", "gt_sem", "pred_geometry", "pred_sem"]:
            _set_layer(layer, False)

    live_status["gui_controls"] = gui_controls
    live_status["gui_controls_available"] = True
    live_status["four_primary_layer_controls_available"] = all(
        name in gui_controls
        for name in [
            "checkbox:GT geometry",
            "checkbox:GT sem",
            "checkbox:Pred geometry",
            "checkbox:Pred sem",
        ]
    )
    live_status["gate"] = {
        "viser_import_ok": status["viser_import_ok"],
        "server_started": True,
        "requested_scene_loaded": int(live_status["scene_point_clouds_added"]) == len(scene_dirs),
        "single_scene_mode": scene_id != "all",
        "gt_geometry_layer_visible": all(bool(row.get("gt_geometry_layer_visible")) for row in live_status["scene_rows"]),
        "gt_sem_layer_visible": all(bool(row.get("gt_sem_layer_visible")) for row in live_status["scene_rows"]),
        "pred_geometry_layer_visible": all(bool(row.get("pred_geometry_layer_visible")) for row in live_status["scene_rows"]),
        "pred_sem_layer_visible": all(bool(row.get("pred_sem_layer_visible")) for row in live_status["scene_rows"]),
        "full_scene_layer_visible": all(bool(row.get("full_scene_layer_visible")) for row in live_status["scene_rows"]),
        "gt_diagnostic_layer_visible": all(bool(row.get("gt_diagnostic_layer_visible")) for row in live_status["scene_rows"]),
        "pred_layer_visible": all(bool(row.get("pred_layer_visible")) for row in live_status["scene_rows"]),
        "d4rt_sim3_aligned_layer_visible": all(
            bool(row.get("d4rt_sim3_aligned_layer_visible")) for row in live_status["scene_rows"]
        ),
        "gui_controls_available": bool(gui_controls),
        "four_primary_layer_controls_available": bool(live_status["four_primary_layer_controls_available"]),
        "SOMA_semantic_ownership_3d_layer_visible": False,
    }
    live_status["gate"]["four_primary_layers_visible"] = bool(
        live_status["gate"]["gt_geometry_layer_visible"]
        and live_status["gate"]["gt_sem_layer_visible"]
        and live_status["gate"]["pred_geometry_layer_visible"]
        and live_status["gate"]["pred_sem_layer_visible"]
    )
    live_status["gate"]["pass"] = bool(
        live_status["gate"]["viser_import_ok"]
        and live_status["gate"]["server_started"]
        and live_status["gate"]["requested_scene_loaded"]
        and live_status["gate"]["four_primary_layers_visible"]
        and live_status["gate"]["gui_controls_available"]
        and live_status["gate"]["four_primary_layer_controls_available"]
    )
    write_json(root / "live_viser_server_status.json", live_status)
    print(json.dumps(live_status, indent=2, sort_keys=True))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return live_status


def _load_aligned_d4rt_scene_points(
    scene: str,
    *,
    debug_root: str | Path,
    mode: str,
    max_points: int = 0,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit

    provider = D4RTCarrierProjectionProvider(
        debug_root=project(debug_root),
        mode=mode,
        nn_radius=0.05,
        min_visibility=min_visibility,
        min_confidence=min_confidence,
        max_anchors=8000,
        robust_trim_percentile=90.0,
        density_alpha=2.0,
    )
    old_cwd = Path.cwd()
    try:
        os.chdir(project("."))
        cache = provider._load_scene(scene)
    finally:
        os.chdir(old_cwd)
    point_parts: list[np.ndarray] = []
    window_parts: list[np.ndarray] = []
    raw_slot_count = 0
    filtered_slot_count = 0
    for window_index, window in enumerate(cache["windows"]):
        for local_idx, _frame_id in enumerate(window.frame_ids):
            xyz = _apply_fit(window.xyz[local_idx], window.transform)
            uv = window.uv[local_idx]
            ok = (
                window.valid[local_idx]
                & np.isfinite(xyz).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (window.visibility[local_idx] >= float(min_visibility))
                & (window.confidence[local_idx] >= float(min_confidence))
            )
            raw_slot_count += int(ok.shape[0])
            filtered_slot_count += int(np.count_nonzero(ok))
            if not np.any(ok):
                continue
            pts = np.asarray(xyz[ok], dtype=np.float32)
            point_parts.append(pts)
            window_parts.append(np.full((pts.shape[0],), window_index, dtype=np.int64))
    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        windows = np.concatenate(window_parts, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        windows = np.zeros((0,), dtype=np.int64)
    sampled = False
    if max_points and max_points > 0 and points.shape[0] > int(max_points):
        idx = _sample_indices(points.shape[0], int(max_points), seed=_stable_seed(f"{scene}:{mode}:aligned_d4rt"))
        points = points[idx]
        windows = windows[idx]
        sampled = True
    diag = {
        "scene": scene,
        "mode": mode,
        "debug_root": rel(debug_root),
        "window_count": int(len(cache.get("windows", []))),
        "raw_slot_count": int(raw_slot_count),
        "filtered_slot_count": int(filtered_slot_count),
        "returned_point_count": int(points.shape[0]),
        "sampled": bool(sampled),
        "max_points": int(max_points),
        "min_visibility": float(min_visibility),
        "min_confidence": float(min_confidence),
        "scene_fit": cache.get("scene_fit", {}),
        "anchor_diag": cache.get("anchor_diag", {}),
        "stitch_diag": cache.get("stitch_diag", {}),
        "uses_eval_sim3_for_visual_alignment": "eval_sim3" in mode,
        "diagnostic_only": True,
    }
    return points, windows, diag


def _load_scene_mesh(scene: str) -> tuple[np.ndarray, np.ndarray, Path]:
    mesh_path = project("data/scannet/processed") / scene / f"{scene}_vh_clean_2.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing ScanNet scene mesh for visualization: {mesh_path}")
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("open3d is required to load the full ScanNet scene for live Viser") from exc
    cloud = o3d.io.read_point_cloud(str(mesh_path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(f"Failed to load scene mesh points from {mesh_path}")
    colors = np.asarray(cloud.colors, dtype=np.float32)
    if colors.shape != points.shape:
        rgb = np.full(points.shape, 176, dtype=np.uint8)
    else:
        rgb = np.asarray(np.clip(colors, 0.0, 1.0) * 255.0, dtype=np.uint8)
    return points, rgb, mesh_path


def _id_colors(ids: np.ndarray) -> np.ndarray:
    labels = np.asarray(ids, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    values = np.abs(labels).astype(np.uint64)
    colors = np.stack(
        [
            40 + (values * 37) % 200,
            35 + (values * 67) % 205,
            45 + (values * 97) % 195,
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def _gt_centroids(scene_points: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    labels = np.unique(gt_labels[gt_labels > 0])
    if labels.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centroids = []
    for label in labels:
        pts = scene_points[gt_labels == label]
        if pts.shape[0] > 0:
            centroids.append(np.mean(pts, axis=0))
    if not centroids:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(centroids, dtype=np.float32)


def _load_prediction_overlay(scene: str, config: str, vertex_count: int) -> dict[str, Any]:
    pred_path = project("data/prediction") / f"{config}_class_agnostic" / f"{scene}.npz"
    pre_points_path = project("data/TMP") / config / f"{scene}_pre_points.npy"
    pre_points_error = ""
    pre_points_count = 0
    pre_points_array: np.ndarray | None = None
    if pre_points_path.exists():
        try:
            pre_points_array = np.asarray(np.load(pre_points_path), dtype=np.int64)
            pre_points_count = int(pre_points_array.shape[0])
        except Exception as exc:  # pragma: no cover - corrupt local artifact diagnostic
            pre_points_count = -1
            pre_points_error = f"{type(exc).__name__}: {exc}"
    if not pred_path.exists():
        return {
            "pred_path": rel(pred_path),
            "pre_points_path": rel(pre_points_path),
            "pre_points_path_sha256": sha256_file(pre_points_path),
            "pre_points_count": pre_points_count,
            "pred_vertex_count": 0,
            "pred_instance_count": 0,
            "mask_contract": "missing_prediction_npz",
            "error": f"missing prediction npz: {rel(pred_path)}",
            "_points": np.zeros((0, 3), dtype=np.float32),
            "_colors": np.zeros((0, 3), dtype=np.uint8),
        }
    with np.load(pred_path) as payload:
        masks = np.asarray(payload["pred_masks"], dtype=bool)
        if "pred_score" in payload.files:
            scores = np.asarray(payload["pred_score"], dtype=np.float32)
        elif "pred_scores" in payload.files:
            scores = np.asarray(payload["pred_scores"], dtype=np.float32)
        else:
            scores = masks.sum(axis=0).astype(np.float32) if masks.ndim == 2 else np.asarray([], dtype=np.float32)
    if masks.ndim != 2:
        return {
            "pred_path": rel(pred_path),
            "pre_points_path": rel(pre_points_path),
            "pre_points_path_sha256": sha256_file(pre_points_path),
            "pre_points_count": pre_points_count,
            "pred_vertex_count": 0,
            "pred_instance_count": 0,
            "mask_contract": f"invalid_mask_rank_{masks.ndim}",
            "error": "pred_masks is not rank-2",
            "_points": np.zeros((0, 3), dtype=np.float32),
            "_colors": np.zeros((0, 3), dtype=np.uint8),
        }
    if masks.shape[0] == vertex_count:
        point_ids = np.arange(vertex_count, dtype=np.int64)
        local_masks = masks
        mask_contract = "full_scene_vertex_mask"
    elif pre_points_array is not None:
        pre_points = pre_points_array
        if masks.shape[0] != pre_points.shape[0]:
            return {
                "pred_path": rel(pred_path),
                "pre_points_path": rel(pre_points_path),
                "pre_points_path_sha256": sha256_file(pre_points_path),
                "pre_points_count": pre_points_count,
                "pred_vertex_count": 0,
                "pred_instance_count": int(masks.shape[1]),
                "mask_contract": "pre_points_length_mismatch",
                "error": f"mask rows={masks.shape[0]} pre_points={pre_points.shape[0]}",
                "_points": np.zeros((0, 3), dtype=np.float32),
                "_colors": np.zeros((0, 3), dtype=np.uint8),
            }
        valid = (pre_points >= 0) & (pre_points < vertex_count)
        point_ids = pre_points[valid]
        local_masks = masks[valid]
        mask_contract = "pre_points_vertex_mask"
    else:
        return {
            "pred_path": rel(pred_path),
            "pre_points_path": rel(pre_points_path),
            "pre_points_path_sha256": sha256_file(pre_points_path),
            "pre_points_count": pre_points_count,
            "pred_vertex_count": 0,
            "pred_instance_count": int(masks.shape[1]),
            "mask_contract": "unsupported_mask_contract",
            "error": f"mask rows={masks.shape[0]} vertex_count={vertex_count}; no usable pre_points file; {pre_points_error}".strip(),
            "_points": np.zeros((0, 3), dtype=np.float32),
            "_colors": np.zeros((0, 3), dtype=np.uint8),
        }
    owner = _prediction_owner_ids(local_masks, scores)
    covered = owner >= 0
    covered_point_ids = point_ids[covered]
    return {
        "pred_path": rel(pred_path),
        "pred_path_sha256": sha256_file(pred_path),
        "pre_points_path": rel(pre_points_path) if pre_points_path.exists() else "",
        "pre_points_path_sha256": sha256_file(pre_points_path),
        "pre_points_count": pre_points_count,
        "pred_vertex_count": int(covered_point_ids.shape[0]),
        "pred_instance_count": int(masks.shape[1]),
        "mask_contract": mask_contract,
        "error": "",
        "_point_ids": covered_point_ids,
        "_points": _scene_points_by_id(scene, covered_point_ids),
        "_colors": _id_colors(owner[covered] + 1),
    }


def _prediction_owner_ids(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if masks.shape[0] == 0 or masks.shape[1] == 0:
        return np.full((masks.shape[0],), -1, dtype=np.int64)
    if scores.shape[0] != masks.shape[1]:
        scores = masks.sum(axis=0).astype(np.float32)
    best_score = np.full((masks.shape[0],), -np.inf, dtype=np.float32)
    owner = np.full((masks.shape[0],), -1, dtype=np.int64)
    for idx in range(masks.shape[1]):
        mask = masks[:, idx]
        if not np.any(mask):
            continue
        score = float(scores[idx])
        update = mask & (score >= best_score)
        best_score[update] = score
        owner[update] = idx
    return owner


def _scene_points_by_id(scene: str, point_ids: np.ndarray) -> np.ndarray:
    points, _colors, _mesh_path = _load_scene_mesh(scene)
    if point_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return points[np.asarray(point_ids, dtype=np.int64)]


def build_v65_visual_casebook(
    case_root: str | Path = CASE_ROOT,
    vis_root: str | Path = VIS_ROOT,
) -> dict[str, Any]:
    root = project(case_root)
    root.mkdir(parents=True, exist_ok=True)
    vis_index = load_dict(project(vis_root) / "viser_scene_index.json")
    cases: list[dict[str, Any]] = []
    cases.extend(_scope_cases())
    cases.extend(_fragment_cases())
    cases.extend(_geometry_cases())
    cases.extend(_ownership_trace_cases())
    cases.extend(_failure_category_cases("F_undercoverage", "C7", 5))
    cases.extend(_failure_category_cases("F_score", "C8", 5))
    cases.extend(_failure_category_cases("F_overmerge", "C6", 2))
    cases = cases[: max(30, len(cases))]
    for idx, case in enumerate(cases, start=1):
        scene = case.get("scene_id") or PROBE5_SCENES[(idx - 1) % len(PROBE5_SCENES)]
        screenshot = project(vis_root) / scene / "screenshots" / "failure_categories.png"
        if not screenshot.exists():
            screenshot = project(vis_root) / scene / "screenshots" / "support_counts.png"
        case["case_id"] = f"v65_case_{idx:03d}"
        case["viewer_bookmark"] = f"{scene}:failure_categories"
        case["screenshot_paths"] = rel(screenshot) if screenshot.exists() else ""
        case.setdefault("recommended_fix", "Inspect linked v65 artifact rows before changing algorithms.")
    summary = _casebook_summary(cases, vis_index)
    write_standard_outputs(
        root,
        {
            "casebook_summary.json": summary,
            "case_rows.csv": cases,
        },
    )
    _write_casebook_html(root / "casebook.html", cases, summary)
    return {"summary": summary, "case_rows": cases}


def _export_scene(root: Path, scene: str) -> dict[str, Any]:
    out = root / scene
    screenshot_dir = out / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    gt = _load_gt(scene)
    xyz, window_ids = _load_d4rt_xyz(scene)
    sample_idx = _sample_indices(gt.shape[0], 5000, seed=_stable_seed(scene))
    support_layers = {name: _load_support(scene, cfg, gt.shape[0], sample_idx) for name, cfg in VIS_VARIANTS.items()}
    pred_summaries = [_prediction_summary(scene, name, cfg) for name, cfg in VIS_VARIANTS.items()]
    ownership_summary = _ownership_state_counts(scene)
    npz_path = out / "viewer_data.npz"
    np.savez_compressed(
        npz_path,
        d4rt_xyz_ref=xyz,
        d4rt_window_id=window_ids,
        mesh_vertex_index_sample=sample_idx,
        gt_label_sample=gt[sample_idx],
        **{f"support_{name}": values for name, values in support_layers.items()},
    )
    write_json(out / "ownership_summary.json", ownership_summary)
    layers = {
        "scene_id": scene,
        "viewer_kind": "viser_npz_png_html",
        "viser_required_for_interactive_3d": True,
        "coordinate_frame_note": "d4rt_xyz_ref is real carrier geometry; support layers are evaluator vertex-index masks.",
        "layers": [
            {"name": "D4RT xyz_ref carriers", "path": rel(npz_path), "key": "d4rt_xyz_ref", "diagnostic_only": False},
            {"name": "AP support scopes", "path": rel(npz_path), "key_prefix": "support_", "diagnostic_only": True},
            {"name": "GT instance labels", "path": rel(npz_path), "key": "gt_label_sample", "diagnostic_only": True},
            {
                "name": "SOMA semantic ownership summary",
                "path": rel(out / "ownership_summary.json"),
                "available": bool(ownership_summary.get("material_count")),
                "is_3d_layer": False,
                "reason": "material_state rows have state/history ids, but xyz_tracks are empty and AP prediction ids are not linked to history/material/component ids.",
            },
        ],
        "prediction_summaries": pred_summaries,
    }
    bookmarks = [
        {"bookmark_id": "d4rt_overview", "description": "D4RT xyz_ref carrier sample overview", "screenshot": "screenshots/d4rt_xyz_overview.png"},
        {"bookmark_id": "support_counts", "description": "AP support scope point counts", "screenshot": "screenshots/support_counts.png"},
        {"bookmark_id": "prediction_counts", "description": "Prediction and tiny fragment counts", "screenshot": "screenshots/prediction_counts.png"},
        {"bookmark_id": "failure_categories", "description": "Failure category counts for scene", "screenshot": "screenshots/failure_categories.png"},
    ]
    write_json(out / "viewer_layers.json", layers)
    write_json(out / "bookmarks.json", bookmarks)
    _scatter_png(screenshot_dir / "d4rt_xyz_overview.png", scene, xyz, window_ids)
    tiny_png(
        screenshot_dir / "support_counts.png",
        f"{scene} support counts",
        [(name.replace("_", " "), float(np.count_nonzero(values))) for name, values in support_layers.items()],
    )
    tiny_png(
        screenshot_dir / "prediction_counts.png",
        f"{scene} prediction counts",
        [(row["variant"], float_or_none(row["pred_count"])) for row in pred_summaries],
    )
    failure_counts = _scene_failure_counts(scene)
    tiny_png(
        screenshot_dir / "failure_categories.png",
        f"{scene} failure categories",
        [(key, float(value)) for key, value in sorted(failure_counts.items())[:8]],
    )
    return {
        "scene_id": scene,
        "viewer_data": rel(npz_path),
        "viewer_data_exists": npz_path.exists(),
        "viewer_layers": rel(out / "viewer_layers.json"),
        "bookmarks": rel(out / "bookmarks.json"),
        "ownership_summary": rel(out / "ownership_summary.json"),
        "ownership_summary_exists": (out / "ownership_summary.json").exists(),
        "ownership_material_count": ownership_summary.get("material_count", 0),
        "screenshot_count": 4,
        "d4rt_xyz_point_count": int(xyz.shape[0]),
        "ap_support_scope_count": len(support_layers),
        "gt_label_count": int(np.unique(gt[gt > 0]).shape[0]),
        "viewer_data_sha256": sha256_file(npz_path),
    }


def _load_gt(scene: str) -> np.ndarray:
    return np.loadtxt(project("data/scannet/gt") / f"{scene}.txt", dtype=np.int64)


def _sample_indices(count: int, max_count: int, *, seed: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(count, dtype=np.int64), size=max_count, replace=False))


def _stable_seed(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) % (2**32)
    return value


def _load_support(scene: str, config: str, total_count: int, sample_idx: np.ndarray) -> np.ndarray:
    path = project("data/TMP") / config / f"{scene}_pre_points.npy"
    mask = np.zeros((total_count,), dtype=bool)
    if path.exists():
        pts = np.load(path).astype(np.int64)
        pts = pts[(pts >= 0) & (pts < total_count)]
        mask[pts] = True
    return mask[sample_idx]


def _load_d4rt_xyz(scene: str, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    xyz_parts: list[np.ndarray] = []
    window_parts: list[np.ndarray] = []
    for path in sorted((project(D4RT_DEBUG_ROOT) / scene).glob("carriers_window*.npz")):
        with np.load(path) as payload:
            xyz = np.asarray(payload["xyz_ref"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool) if "valid" in payload else np.ones(xyz.shape[:-1], dtype=bool)
        window = int(path.stem.replace("carriers_window", ""))
        if xyz.ndim == 3:
            xyz = xyz[valid]
        elif xyz.ndim == 2:
            xyz = xyz[valid.reshape(-1)] if valid.ndim > 1 else xyz[valid]
        else:
            xyz = xyz.reshape(-1, 3)
        xyz_parts.append(xyz)
        window_parts.append(np.full((xyz.shape[0],), window, dtype=np.int64))
    if not xyz_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyz = np.concatenate(xyz_parts, axis=0)
    windows = np.concatenate(window_parts, axis=0)
    idx = _sample_indices(xyz.shape[0], max_points, seed=17)
    return xyz[idx], windows[idx]


def _prediction_summary(scene: str, variant_name: str, config: str) -> dict[str, Any]:
    pred_path = project("data/prediction") / f"{config}_class_agnostic" / f"{scene}.npz"
    if not pred_path.exists():
        return {"variant": variant_name, "pred_count": "", "tiny_lt100_count": "", "pred_path": rel(pred_path)}
    with np.load(pred_path) as payload:
        masks = np.asarray(payload["pred_masks"], dtype=bool)
    areas = masks.sum(axis=0) if masks.ndim == 2 else np.asarray([], dtype=np.int64)
    return {
        "variant": variant_name,
        "pred_count": int(areas.shape[0]),
        "tiny_lt100_count": int(np.count_nonzero(areas < 100)),
        "pred_path": rel(pred_path),
    }


def _scene_failure_counts(scene: str) -> dict[str, int]:
    path = project("outputs/audit/v65_ap_failure_decomp/failure_rows.csv")
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for row in read_csv(path):
        if row.get("scene_id") != scene:
            continue
        key = str(row.get("failure_category") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ownership_state_counts(scene: str) -> dict[str, Any]:
    path = project("outputs/audit/v64r2_native_contract/material_state_rows.csv")
    counts: dict[str, int] = {}
    histories: set[str] = set()
    xyz_track_nonempty = 0
    if not path.exists():
        return {
            "scene_id": scene,
            "material_count": 0,
            "state_counts": counts,
            "history_count": 0,
            "xyz_track_nonempty_count": 0,
            "source": rel(path),
        }
    for row in read_csv(path):
        if row.get("scene_id") != scene:
            continue
        state = str(row.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
        if row.get("history_id"):
            histories.add(str(row["history_id"]))
        if row.get("xyz_tracks_if_available") not in {"", "[]", None}:
            xyz_track_nonempty += 1
    return {
        "scene_id": scene,
        "material_count": sum(counts.values()),
        "state_counts": counts,
        "history_count": len(histories),
        "xyz_track_nonempty_count": xyz_track_nonempty,
        "source": rel(path),
        "source_sha256": sha256_file(path),
    }


def _window_colors(window_ids: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [53, 95, 173],
            [211, 101, 67],
            [88, 151, 104],
            [132, 102, 176],
            [196, 153, 63],
        ],
        dtype=np.uint8,
    )
    if window_ids.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return palette[np.asarray(window_ids, dtype=np.int64) % palette.shape[0]]


def _scatter_png(path: Path, scene: str, xyz: np.ndarray, window_ids: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.4, 4.8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    if xyz.size:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=window_ids, s=1.5, cmap="viridis", alpha=0.75)
    ax.set_title(f"{scene} D4RT xyz_ref carriers")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scope_cases() -> list[dict[str, Any]]:
    rows = read_csv(project("outputs/audit/v65_ap_failure_decomp/scope_contrast_rows.csv"))
    cases: list[dict[str, Any]] = []
    for row in rows:
        for scene in PROBE5_SCENES:
            cases.append(
                {
                    "case_type": "C1 high diagnostic AP but low used-support AP",
                    "scene_id": scene,
                    "variant": f"{row.get('left_row_id')}->{row.get('right_row_id')}",
                    "support_scope": f"{row.get('left_support_scope')}->{row.get('right_support_scope')}",
                    "AP": row.get("left_AP"),
                    "AP50": "",
                    "AP25": "",
                    "geometry_level": "G5",
                    "geometry_metric_summary": f"AP delta {row.get('delta_AP')}",
                    "failure_category": "F_scope",
                    "history_id": "",
                    "gt_id": "",
                    "recommended_fix": "Never compare AP rows without identical support hash and input-frame policy.",
                }
            )
            if len(cases) >= 5:
                return cases
    return cases


def _fragment_cases() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(project("outputs/audit/v65_ap_failure_decomp/fragmentation_rows.csv"))
        if row.get("variant_row_id") in {"A5", "A7"}
    ]
    rows = sorted(rows, key=lambda r: float_or_none(r.get("tiny_fragment_ratio")) or 0.0, reverse=True)
    cases = []
    for row in rows[:5]:
        cases.append(
            {
                "case_type": "C2 D4RT G11/G12 tiny fragment AP failure",
                "scene_id": row.get("scene_id"),
                "variant": row.get("variant_row_id"),
                "support_scope": row.get("support_scope"),
                "AP": "",
                "AP50": "",
                "AP25": "",
                "geometry_level": "G4",
                "geometry_metric_summary": f"tiny_fragment_ratio={row.get('tiny_fragment_ratio')}, pred_best_iou_median={row.get('pred_best_iou_median')}",
                "failure_category": "F_tiny_fragments",
                "history_id": row.get("history_id") or "",
                "gt_id": "",
                "recommended_fix": "Add non-GT object-level aggregation trace before claiming AP repair.",
            }
        )
    return cases


def _geometry_cases() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(project("outputs/audit/v65_geometry_contract/chunk_scale_rows.csv"))
        if row.get("pass_v65_10pct_gate") == "False"
    ]
    cases = []
    for row in rows[:5]:
        cases.append(
            {
                "case_type": "C3 chunk scale drift / self-stitch failure",
                "scene_id": row.get("scene_id"),
                "variant": "D4RT chunk scale",
                "support_scope": "",
                "AP": "",
                "AP50": "",
                "AP25": "",
                "geometry_level": "G2",
                "geometry_metric_summary": f"window_pair={row.get('window_pair')}, scale_next_over_prev={row.get('scale_next_over_prev')}",
                "failure_category": "chunk_scale_drift",
                "history_id": "",
                "gt_id": "",
                "recommended_fix": "Inspect adjacent chunk self-stitch residuals before more AP tuning.",
            }
        )
    return cases


def _ownership_trace_cases() -> list[dict[str, Any]]:
    cases = []
    for scene in PROBE5_SCENES:
        cases.append(
            {
                "case_type": "C5 SOMA ownership/material trace unavailable for AP materialization",
                "scene_id": scene,
                "variant": "I2-I5 blocked",
                "support_scope": "D4RT prediction support",
                "AP": "",
                "AP50": "",
                "AP25": "",
                "geometry_level": "G4/G5",
                "geometry_metric_summary": "history/material/component trace is not linked to AP prediction ids",
                "failure_category": "ownership_trace_missing",
                "history_id": "",
                "gt_id": "",
                "recommended_fix": "Bind D4RT fragments to object-level ownership histories and rerun non-GT aggregation.",
            }
        )
    return cases


def _failure_category_cases(category: str, case_type: str, limit: int) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(project("outputs/audit/v65_ap_failure_decomp/failure_rows.csv")) if row.get("failure_category") == category]
    cases = []
    for row in rows[:limit]:
        cases.append(
            {
                "case_type": f"{case_type} {category}",
                "scene_id": row.get("scene_id"),
                "variant": row.get("variant_row_id"),
                "support_scope": row.get("support_scope"),
                "AP": "",
                "AP50": "",
                "AP25": "",
                "geometry_level": "G4/G5",
                "geometry_metric_summary": f"best_iou={row.get('best_iou')}, pred_area={row.get('pred_area')}, gt_area={row.get('gt_area')}",
                "failure_category": category,
                "history_id": row.get("history_id") or "",
                "gt_id": row.get("gt_id") or "",
                "recommended_fix": "Use failure row and visual bookmark to decide whether support, aggregation, or scoring is the next repair.",
            }
        )
    return cases


def _casebook_summary(cases: list[dict[str, Any]], vis_index: dict[str, Any]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    for case in cases:
        prefix = str(case.get("case_type") or "").split(" ", 1)[0]
        type_counts[prefix] = type_counts.get(prefix, 0) + 1
    summary = {
        "phase": "v65_casebook",
        "case_count": len(cases),
        "case_type_counts": type_counts,
        "visualization_status": vis_index.get("visualization_status") or "unknown",
        "uses_fallback_screenshots": not bool(vis_index.get("viser_import_ok")),
        "gate": {
            "case_count_ge_30": len(cases) >= 30,
            "AP_scope_cases_ge_5": type_counts.get("C1", 0) >= 5,
            "geometry_cases_ge_5": type_counts.get("C3", 0) >= 5,
            "SOMA_ownership_cases_ge_5": type_counts.get("C5", 0) >= 5,
            "D4RT_fragment_aggregation_cases_ge_5": type_counts.get("C2", 0) >= 5,
            "each_case_has_screenshot_or_bookmark": all(case.get("screenshot_paths") or case.get("viewer_bookmark") for case in cases),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return summary


def _write_casebook_html(path: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = []
    for case in cases:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(case.get(key, '')))}</td>"
                for key in ["case_id", "case_type", "scene_id", "variant", "failure_category", "geometry_metric_summary", "screenshot_paths", "recommended_fix"]
            )
            + "</tr>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>v65 visual casebook</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;font-size:12px;vertical-align:top}}th{{background:#eee}}</style>
</head><body>
<h1>Stream4D v65 Visual Casebook</h1>
<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
<table><thead><tr><th>case_id</th><th>case_type</th><th>scene</th><th>variant</th><th>failure</th><th>metric summary</th><th>screenshot</th><th>recommended fix</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
