from __future__ import annotations

import unittest

from stream4d_native.v53_mask_component_support import build_mask_component_support


class V53MaskComponentSupportTest(unittest.TestCase):
    def test_support_table_emits_all_components_not_only_dominant(self) -> None:
        mask_rows = [
            {
                "mask_observation_id": "s:0:1",
                "scene": "s",
                "frame_id": 0,
                "mask_id": 1,
                "mask_area": 100,
                "diagnostic_gt_instance": 1,
            }
        ]
        carrier_rows = [
            {
                "scene": "s",
                "frame_id": 0,
                "carrier_id": 1,
                "carrier_global_id": "s:1",
                "visible": True,
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 1,
            },
            {
                "scene": "s",
                "frame_id": 0,
                "carrier_id": 2,
                "carrier_global_id": "s:2",
                "visible": True,
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 1,
            },
        ]
        payload = build_mask_component_support_from_rows_for_test(
            carrier_rows=carrier_rows,
            mask_rows=mask_rows,
            max_union_unique_carriers=1,
        )
        full_rows = [row for row in payload["support_rows"] if row["variant"] == "I0_visible_tau0.10"]
        dominant_rows = [row for row in payload["support_rows"] if row["variant"] == "I4_dominant_component_only"]
        self.assertEqual(len(full_rows), 2)
        self.assertEqual(len(dominant_rows), 1)
        self.assertFalse(payload["summary"]["main_summary"]["dominant_component_collapse_detected"])

    def test_visible_denominator_is_current_frame_component_count(self) -> None:
        mask_rows = [
            {"mask_observation_id": "s:-1:9", "scene": "s", "frame_id": -1, "mask_id": 9, "mask_area": 100},
            {"mask_observation_id": "s:0:1", "scene": "s", "frame_id": 0, "mask_id": 1, "mask_area": 100},
            {"mask_observation_id": "s:0:2", "scene": "s", "frame_id": 0, "mask_id": 2, "mask_area": 100},
        ]
        carrier_rows = [
            {
                "scene": "s",
                "frame_id": -1,
                "carrier_id": 1,
                "carrier_global_id": "s:1",
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 9,
            },
            {
                "scene": "s",
                "frame_id": -1,
                "carrier_id": 2,
                "carrier_global_id": "s:2",
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 9,
            },
            {
                "scene": "s",
                "frame_id": 0,
                "carrier_id": 1,
                "carrier_global_id": "s:1",
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 1,
            },
            {
                "scene": "s",
                "frame_id": 0,
                "carrier_id": 2,
                "carrier_global_id": "s:2",
                "valid": True,
                "valid_uv": True,
                "confidence": 1.0,
                "visibility_prob": 1.0,
                "observed_mask_id": 2,
            },
        ]
        payload = build_mask_component_support_from_rows_for_test(
            carrier_rows=carrier_rows,
            mask_rows=mask_rows,
            max_union_unique_carriers=2,
        )
        rows = [
            row
            for row in payload["support_rows"]
            if row["variant"] == "I0_visible_tau0.10" and row["frame_id"] == 0
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["component_visible_count_in_frame"], 2)
            self.assertAlmostEqual(row["W_visible"], 0.5)
            self.assertAlmostEqual(row["R_mask"], 1.0)


def build_mask_component_support_from_rows_for_test(
    carrier_rows: list[dict],
    mask_rows: list[dict],
    max_union_unique_carriers: int,
) -> dict:
    from stream4d_native import v53_mask_component_support as module

    component_payload = module._build_components(
        carrier_rows=carrier_rows,
        mask_rows=mask_rows,
        max_union_unique_carriers=max_union_unique_carriers,
        min_visibility_prob=0.5,
        min_confidence=0.5,
    )
    support_payload = module._collect_support(
        visible_rows=component_payload["visible_rows"],
        mask_rows=mask_rows,
        component_by_carrier=component_payload["component_by_carrier"],
    )
    all_support_rows = []
    variant_summaries = []
    for variant, denominator, tau in [
        ("I0_visible_tau0.10", "visible", 0.10),
        ("I4_dominant_component_only", "dominant_only", 0.0),
    ]:
        rows, counts, entropy = module._support_rows_for_variant(
            variant=variant,
            denominator_mode=denominator,
            tau=tau,
            mask_rows=mask_rows,
            support_by_mask=support_payload["support_by_mask"],
            visible_denominator=support_payload["visible_denominator"],
            component_total_visible=support_payload["component_total_visible"],
            mask_carrier_total=support_payload["mask_carrier_total"],
        )
        all_support_rows.extend(rows)
        variant_summaries.append(
            module._variant_summary(
                variant=variant,
                rows=rows,
                components_per_mask=counts,
                entropy_by_mask=entropy,
                mask_count=len(mask_rows),
                component_count=len(component_payload["component_ids"]),
                support_visible_row_count=support_payload["support_visible_row_count"],
                visible_row_count=support_payload["visible_row_count"],
            )
        )
    return {"support_rows": all_support_rows, "summary": {"main_summary": variant_summaries[0]}}


if __name__ == "__main__":
    unittest.main()
