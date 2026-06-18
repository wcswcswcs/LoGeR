from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.object_tube_io import MergeGeometryError, TubeRecord, assert_merge_geometry_valid
from stream4d_native.sim3 import Sim3Transform, apply_sim3_to_xyz, invert_sim3


def _builder() -> D4RTNativeSceneBuilder:
    return D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 4}}}, temporal_chunk_size=4, temporal_chunk_stride=2)


def _canonical_points(frame_ids: list[int], carrier_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for frame in frame_ids:
        pts = []
        for cid in carrier_ids:
            pts.append(
                [
                    0.17 * cid + 0.011 * frame,
                    0.07 * cid * cid + 0.013 * frame,
                    0.31 + 0.021 * cid + 0.019 * frame + 0.003 * cid * frame,
                ]
            )
        out[int(frame)] = np.asarray(pts, dtype=np.float32)
    return out


def _tube_chunk(
    *,
    chunk_id: int,
    start: int,
    frame_ids: list[int],
    canonical_by_frame: dict[int, np.ndarray],
    carrier_ids: list[int],
    transform_from_local: Sim3Transform | None = None,
    order: list[int] | None = None,
    persistent: bool = True,
    carrier_offset: int = 0,
    source_pixel: bool = True,
    uv_offset: float = 0.0,
) -> dict:
    order = list(range(len(carrier_ids))) if order is None else list(order)
    inv = invert_sim3(transform_from_local) if transform_from_local is not None else None
    tubes = []
    for local_pos, base_idx in enumerate(order):
        carrier_id = int(carrier_ids[base_idx])
        xyz = np.stack([canonical_by_frame[int(frame)][base_idx] for frame in frame_ids], axis=0)
        if inv is not None:
            xyz = apply_sim3_to_xyz(xyz, transform=inv)
        uv = np.stack(
            [
                np.asarray([0.18 + 0.07 * base_idx + uv_offset, 0.22 + 0.05 * (frame % 5) + uv_offset], dtype=np.float32)
                for frame in frame_ids
            ],
            axis=0,
        )
        tubes.append(
            {
                "carrier_id": carrier_id + int(carrier_offset),
                "persistent_tube_id": carrier_id if persistent else -1,
                "uv_norm": uv,
                "xyz": xyz.astype(np.float32),
                "visibility": np.ones((len(frame_ids),), dtype=np.float32),
                "confidence": np.ones((len(frame_ids),), dtype=np.float32),
                "valid": np.ones((len(frame_ids),), dtype=bool),
                "source_frame_global": carrier_id if source_pixel else -1,
                "source_xy": (carrier_id + 1, carrier_id + 3) if source_pixel else (-1, -1),
                "source_frame_local": 0,
            }
        )
    return {
        "chunk": {"chunk_id": int(chunk_id), "start": int(start), "end": int(start) + len(frame_ids), "frame_ids": list(frame_ids)},
        "tubes": tubes,
        "diagnostics": {},
    }


def _record(
    *,
    tube_id: int,
    chunk_id: int,
    submap_id: int,
    coordinate_frame: str,
    alignment_source: str,
    allow_metric_merge: bool,
    pass_gate: bool,
) -> TubeRecord:
    return TubeRecord(
        tube_id=tube_id,
        persistent_tube_id=tube_id,
        chunk_id=chunk_id,
        submap_id=submap_id,
        source_frame_global=0,
        source_xy=(1, 2),
        source_uv=(0.1, 0.2),
        target_frames_global=np.asarray([0, 1], dtype=np.int64),
        uv=np.zeros((2, 2), dtype=np.float32),
        visibility=np.ones((2,), dtype=np.float32),
        confidence=np.ones((2,), dtype=np.float32),
        xyz_local=np.zeros((2, 3), dtype=np.float32),
        xyz_ref0=np.zeros((2, 3), dtype=np.float32),
        xyz_canonical=np.zeros((2, 3), dtype=np.float32) if coordinate_frame == "d4rt_canonical" else None,
        T_chunk_to_canonical={"scale": 1.0, "rot": np.eye(3).tolist(), "trans": [0.0, 0.0, 0.0]},
        alignment_quality={"pass_gate": bool(pass_gate)},
        coordinate_frame=coordinate_frame,
        scale_status="canonical" if coordinate_frame == "d4rt_canonical" else "chunk_local",
        allow_metric_merge=bool(allow_metric_merge),
        alignment_source=alignment_source,
        transform_id=f"t{tube_id}",
    )


class V24ScaleConsistencyTest(unittest.TestCase):
    def test_merge_rejects_xyz_local_cross_chunk(self) -> None:
        a = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="chunk_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True)
        b = _record(tube_id=2, chunk_id=1, submap_id=0, coordinate_frame="chunk_local", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True)
        with self.assertRaises(MergeGeometryError):
            assert_merge_geometry_valid(a, b, "unit")

    def test_merge_rejects_xyz_ref0_cross_chunk(self) -> None:
        a = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="ref0_local", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True)
        b = _record(tube_id=2, chunk_id=1, submap_id=0, coordinate_frame="ref0_local", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True)
        with self.assertRaises(MergeGeometryError):
            assert_merge_geometry_valid(a, b, "unit")

    def test_merge_rejects_eval_aligned_xyz(self) -> None:
        a = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="eval_gt_sim3", allow_metric_merge=False, pass_gate=True)
        b = _record(tube_id=2, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True)
        with self.assertRaises(MergeGeometryError):
            assert_merge_geometry_valid(a, b, "unit")

    def test_merge_accepts_xyz_canonical_with_valid_alignment(self) -> None:
        a = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True)
        b = _record(tube_id=2, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=True, pass_gate=True)
        event = assert_merge_geometry_valid(a, b, "unit")
        self.assertTrue(event["guard_pass"])

    def test_weak_alignment_disables_metric_merge(self) -> None:
        a = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True)
        b = _record(tube_id=2, chunk_id=1, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="d4rt_self_sim3", allow_metric_merge=False, pass_gate=False)
        with self.assertRaises(MergeGeometryError):
            assert_merge_geometry_valid(a, b, "unit")

    def test_tube_record_preserves_transform_metadata(self) -> None:
        record = _record(tube_id=1, chunk_id=0, submap_id=0, coordinate_frame="d4rt_canonical", alignment_source="same_chunk_identity", allow_metric_merge=True, pass_gate=True)
        recovered = TubeRecord.from_jsonable(record.to_jsonable())
        self.assertEqual(recovered.transform_id, record.transform_id)
        self.assertEqual(recovered.T_chunk_to_canonical["scale"], 1.0)
        self.assertEqual(recovered.coordinate_frame, "d4rt_canonical")

    def test_builder_uses_true_overlap_frames_and_recovers_known_sim3(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3, 4, 5]
        canonical = _canonical_points([0, 1, 2, 3, 4, 5], carriers)
        transform = Sim3Transform(
            scale=1.7,
            rot=np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            trans=np.asarray([0.3, -0.2, 0.4], dtype=np.float64),
        )
        first = _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1, 2, 3], canonical_by_frame=canonical, carrier_ids=carriers)
        second = _tube_chunk(
            chunk_id=1,
            start=2,
            frame_ids=[2, 3, 4, 5],
            canonical_by_frame=canonical,
            carrier_ids=carriers,
            transform_from_local=transform,
            order=[3, 0, 5, 1, 4, 2],
        )
        stitched = builder.stitch_to_canonical([first, second])
        self.assertEqual(stitched["diagnostics"]["weak_alignment_chunk_count"], 0)
        got = stitched["chunks"][1]["T_chunk_to_canonical"]
        self.assertAlmostEqual(float(got["scale"]), 1.7, places=5)
        self.assertTrue(np.allclose(got["rot"], transform.rot, atol=1e-5))
        self.assertTrue(np.allclose(got["trans"], transform.trans, atol=1e-5))
        chunk = stitched["chunks"][1]
        for tube in chunk["tubes"]:
            cid = int(tube["persistent_tube_id"])
            expected = np.stack([canonical[frame][carriers.index(cid)] for frame in [2, 3, 4, 5]], axis=0)
            self.assertTrue(np.allclose(tube["xyz_canonical"], expected, atol=1e-5))

    def test_stitch_to_canonical_applies_transform(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3]
        canonical = _canonical_points([0, 1, 2, 3], carriers)
        transform = Sim3Transform(scale=1.25, rot=np.eye(3), trans=np.asarray([0.4, 0.2, -0.1]))
        first = _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1, 2, 3], canonical_by_frame=canonical, carrier_ids=carriers)
        second = _tube_chunk(
            chunk_id=1,
            start=0,
            frame_ids=[0, 1, 2, 3],
            canonical_by_frame=canonical,
            carrier_ids=carriers,
            transform_from_local=transform,
        )
        stitched = builder.stitch_to_canonical([first, second])
        raw = stitched["chunks"][1]["tubes"][0]["xyz"]
        canonicalized = stitched["chunks"][1]["tubes"][0]["xyz_canonical"]
        self.assertFalse(np.allclose(raw, canonicalized))
        self.assertTrue(np.allclose(canonicalized, first["tubes"][0]["xyz"], atol=1e-5))

    def test_three_chunk_chain_composition_nontrivial_rotation_scale_translation(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3, 4, 5]
        canonical = _canonical_points(list(range(8)), carriers)
        rot_z = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        t1 = Sim3Transform(scale=1.4, rot=rot_z, trans=np.asarray([0.1, -0.2, 0.3], dtype=np.float64))
        t2 = Sim3Transform(scale=0.7, rot=rot_x, trans=np.asarray([-0.3, 0.4, 0.2], dtype=np.float64))
        chunks = [
            _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1, 2, 3], canonical_by_frame=canonical, carrier_ids=carriers),
            _tube_chunk(chunk_id=1, start=2, frame_ids=[2, 3, 4, 5], canonical_by_frame=canonical, carrier_ids=carriers, transform_from_local=t1),
            _tube_chunk(chunk_id=2, start=4, frame_ids=[4, 5, 6, 7], canonical_by_frame=canonical, carrier_ids=carriers, transform_from_local=t2),
        ]
        stitched = builder.stitch_to_canonical(chunks)
        self.assertEqual(stitched["diagnostics"]["weak_alignment_chunk_count"], 0)
        self.assertEqual(stitched["diagnostics"]["submap_count"], 1)
        got = stitched["chunks"][2]["T_chunk_to_canonical"]
        self.assertAlmostEqual(float(got["scale"]), float(t2.scale), places=5)
        self.assertTrue(np.allclose(got["rot"], t2.rot, atol=1e-5))
        self.assertTrue(np.allclose(got["trans"], t2.trans, atol=1e-5))

    def test_builder_recovers_known_sim3_with_source_pixel_fallback(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3, 4, 5]
        canonical = _canonical_points([0, 1, 2, 3], carriers)
        transform = Sim3Transform(scale=0.8, rot=np.eye(3), trans=np.asarray([0.2, -0.1, 0.05]))
        first = _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1, 2, 3], canonical_by_frame=canonical, carrier_ids=carriers, persistent=False, carrier_offset=0)
        second = _tube_chunk(
            chunk_id=1,
            start=0,
            frame_ids=[0, 1, 2, 3],
            canonical_by_frame=canonical,
            carrier_ids=carriers,
            transform_from_local=transform,
            persistent=False,
            carrier_offset=100,
            source_pixel=True,
        )
        stitched = builder.stitch_to_canonical([first, second])
        stats = stitched["diagnostics"]["pairwise_self_sim3"][0]["match_stats"]
        self.assertEqual(stats["used_source_pixel_match_count"], len(carriers) * len([0, 1, 2, 3]))
        self.assertEqual(stitched["diagnostics"]["weak_alignment_chunk_count"], 0)

    def test_builder_mutual_uv_fallback_requires_overlap_frame(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3]
        canonical = _canonical_points([0, 1, 2, 3], carriers)
        first = _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1], canonical_by_frame=canonical, carrier_ids=carriers, persistent=False, carrier_offset=0, source_pixel=False)
        second = _tube_chunk(chunk_id=1, start=2, frame_ids=[2, 3], canonical_by_frame=canonical, carrier_ids=carriers, persistent=False, carrier_offset=100, source_pixel=False)
        fit = builder.estimate_overlap_self_sim3(first, second)
        self.assertFalse(bool(fit["pass_gate"]))
        self.assertEqual(fit["match_stats"]["overlap_frame_count"], 0)

    def test_failed_pair_starts_new_submap_and_disables_merge(self) -> None:
        builder = _builder()
        carriers = [0, 1, 2, 3]
        canonical = _canonical_points([0, 1, 2, 3], carriers)
        first = _tube_chunk(chunk_id=0, start=0, frame_ids=[0, 1], canonical_by_frame=canonical, carrier_ids=carriers)
        second = _tube_chunk(chunk_id=1, start=2, frame_ids=[2, 3], canonical_by_frame=canonical, carrier_ids=carriers)
        stitched = builder.stitch_to_canonical([first, second])
        self.assertEqual(stitched["diagnostics"]["weak_alignment_chunk_count"], 1)
        self.assertEqual(stitched["diagnostics"]["submap_count"], 2)
        self.assertFalse(stitched["chunks"][1]["allow_metric_merge"])


if __name__ == "__main__":
    unittest.main()
