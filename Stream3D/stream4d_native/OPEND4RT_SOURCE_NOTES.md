# OpenD4RT Source Notes

This file records the source alignment required by the Stream4D v21.3 plan.
The native implementation ports helper semantics instead of inventing a new
D4RT runner shape.

| Stream4D native module | Borrowed / aligned OpenD4RT source | Notes |
|---|---|---|
| `stream4d_native.opend4rt_long_video.make_anchor_clip_indices` | `Open-d4rt/infer_track_3d.py::_make_anchor_clip_indices` | Preserves the source+target-in-clip invariant for long sparse tracking diagnostics. |
| `stream4d_native.chunk_alignment.make_sliding_window_clip_ranges` | `Open-d4rt/vis/build_like_demo.py` sliding-window usage and `Open-d4rt/infer_track_3d.py` clip-limit semantics | Uses checkpoint clip length and half-overlap default. |
| `stream4d_native.sim3.estimate_overlap_sim3` | `Open-d4rt/src/eval/tasks.py::_umeyama_sim3`; `Open-d4rt/vis/build_like_demo.py` overlap stitching intent | Estimates D4RT-only curr-to-prev Sim3 from overlapping xyz/visibility/confidence. |
| `stream4d_native.d4rt_scene_builder.D4RTNativeSceneBuilder` | `Open-d4rt/infer_track_3d.py::_run_clip_queries_for_target_indices`; `Open-d4rt/src/eval/tasks.py::_encode_model_memory` and `_run_model_for_queries` | The concrete model call is isolated behind `infer_carriers`; query batch size is separate from temporal chunk length. |
| `stream4d_native.occupancy_dense_tracker` | D4RT paper Algorithm 1 dense occupancy idea and `model.yaml dense_tracking.algorithm=occupancy_grid_tracking_all_pixels` | Maintains spatio-temporal visited state and samples only unvisited source pixels. |

Additional OpenD4RT helpers checked by the audit:

- `Open-d4rt/src/eval/tasks.py::_model_clip_frames`
- `Open-d4rt/src/eval/tasks.py::_umeyama_sim3`
- `Open-d4rt/infer_track_3d.py::_infer_tracks`

Evaluation-only code that touches ScanNet depth, pose, mesh, GT labels, or
GT/RGB-D Sim3 must live outside `stream4d_native`.
