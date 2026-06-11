from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream4d.evidence_terms import (
    ExplanationParams,
    birth_groups,
    measurement_votes,
    posterior_for_group,
)
from stream4d.measurement_bank import MeasurementBank


def toy_bank(*, negative_second_object: bool = False) -> MeasurementBank:
    n = 10
    t = 3
    visible = np.ones((t, n), dtype=bool)
    target = np.zeros((t, n), dtype=np.int64)
    target[:, :5] = 7
    target[:, 5:] = 9
    negative = np.zeros((t, n), dtype=bool)
    if negative_second_object:
        negative[:, 5:] = True
    src_mask_id = np.asarray([7, 7, 7, 7, 7, 9, 9, 9, 9, 9], dtype=np.int64)
    return MeasurementBank(
        scene="toy",
        frame_ids=np.asarray([0, 1, 2], dtype=np.int64),
        carrier_id=np.arange(n, dtype=np.int64),
        uv_pred=np.full((t, n, 2), 0.5, dtype=np.float32),
        valid=visible.copy(),
        visibility=np.ones((t, n), dtype=np.float32),
        confidence=np.ones((t, n), dtype=np.float32),
        src_frame_global=np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64),
        src_mask_id=src_mask_id,
        src_xy=np.zeros((n, 2), dtype=np.int64),
        src_rgb=np.ones((n, 3), dtype=np.float32) * 0.5,
        target_mask_id=target,
        target_in_bounds=visible.copy(),
        visible_ok=visible.copy(),
        boundary_distance=np.ones((t, n), dtype=np.float32) * 5.0,
        source_boundary_distance=np.ones((n,), dtype=np.float32) * 5.0,
        mask_frame_available=np.ones((t,), dtype=bool),
        positive_observation=target > 0,
        negative_observation=negative,
        source_positive_propagated=visible & (src_mask_id[None, :] > 0),
        meta={"toy": True},
    )


class ObjectExplanationPureTests(unittest.TestCase):
    def test_measurement_bank_roundtrip(self) -> None:
        bank = toy_bank()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurement_bank.npz"
            bank.save(path)
            restored = MeasurementBank.load(path)

        self.assertEqual(restored.scene, "toy")
        np.testing.assert_array_equal(restored.carrier_id, bank.carrier_id)
        np.testing.assert_array_equal(restored.target_mask_id, bank.target_mask_id)
        self.assertEqual(restored.meta["toy"], True)

    def test_birth_groups_split_by_frame_and_mask(self) -> None:
        params = ExplanationParams(
            birth_min_surfels=3,
            birth_min_boundary_safe_ratio=0.5,
            birth_max_ambiguous_ratio=1.0,
        )

        groups = birth_groups(toy_bank(), params)

        keys = {(item["birth_frame"], item["birth_mask_id"]) for item in groups}
        self.assertEqual(keys, {(0, 7), (1, 9)})
        self.assertTrue(all(item["passes_birth_gate"] for item in groups))

    def test_posterior_negative_evidence_rejects_contested_surfels(self) -> None:
        bank = toy_bank(negative_second_object=True)
        params = ExplanationParams(
            core_posterior_threshold=0.4,
            fringe_posterior_threshold=0.2,
            reject_negative_threshold=0.5,
        )

        posterior = posterior_for_group(
            bank,
            np.arange(5, 10, dtype=np.int64),
            params,
            use_negative=True,
            use_temporal=True,
        )

        self.assertEqual(np.asarray(posterior["reject"]).shape[0], 5)
        self.assertEqual(np.asarray(posterior["core"]).shape[0], 0)

    def test_measurement_votes_are_deterministic_and_deduplicated(self) -> None:
        params = ExplanationParams(measurement_min_surfels=2, max_slots_per_frame_mask=3)
        bank = toy_bank()

        first = measurement_votes(bank, np.arange(0, 5, dtype=np.int64), params, include_temporal_targets=True)
        second = measurement_votes(bank, np.arange(0, 5, dtype=np.int64), params, include_temporal_targets=True)

        self.assertEqual(first, second)
        self.assertEqual(first, [(0, 7, 5.0), (1, 7, 5.0), (2, 7, 5.0)])


if __name__ == "__main__":
    unittest.main()
