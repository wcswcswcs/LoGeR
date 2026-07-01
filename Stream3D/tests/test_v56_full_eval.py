from __future__ import annotations

import unittest

from stream4d_native.v56_full_eval import build_v56_full_eval


class V56FullEvalTest(unittest.TestCase):
    def test_state_separation_keeps_core_metrics(self) -> None:
        payload = build_v56_full_eval()
        phase5 = payload["phase5_summary"]
        self.assertTrue(phase5["gate"]["core_purity_drop_le_0.002"])
        self.assertEqual(phase5["core_control_margin_change"], 0.0)
        self.assertGreater(phase5["tentative_added_component_count"], 0)

    def test_no_gt_prediction_flags(self) -> None:
        payload = build_v56_full_eval()
        for key in ["phase0_summary", "phase2_summary", "phase4_summary", "phase6_summary", "native_field_summary", "final_summary"]:
            self.assertFalse(payload[key]["uses_gt_for_prediction"])

    def test_material_atom_gate_uses_actual_overlap_not_proxy(self) -> None:
        payload = build_v56_full_eval()
        phase2 = payload["phase2_summary"]
        self.assertEqual(phase2["anchor_update_atom_overlap_nonzero_count"], 0)
        self.assertEqual(phase2["anchor_update_atom_overlap_nonzero_ratio"], 0.0)
        self.assertGreater(phase2["boundary_uv_proxy_overlap_ratio"], 0.0)
        self.assertFalse(phase2["gate"]["anchor_update_atom_overlap_nonzero_ratio_ge_0.10"])
        self.assertFalse(phase2["gate"]["pass"])

    def test_native_field_exports_full_state_labels(self) -> None:
        payload = build_v56_full_eval()
        summary = payload["native_field_summary"]
        rows = payload["native_carrier_state_rows"]
        self.assertEqual(len(rows), summary["native_carrier_state_row_count"])
        self.assertEqual(
            summary["native_carrier_state_row_count"],
            summary["confirmed_core_component_count"]
            + summary["tentative_component_count"]
            + summary["quarantine_history_component_row_count"],
        )
        states = {row["state"] for row in rows}
        self.assertIn("confirmed", states)
        self.assertIn("tentative", states)
        self.assertIn("quarantine", states)
        self.assertEqual(summary["native_carrier_state_counts"]["quarantine"], summary["quarantine_history_component_row_count"])
        self.assertTrue(summary["gate"]["all_state_rows_exported"])
        self.assertTrue(summary["gate"]["quarantine_state_rows_present_if_quarantine_components"])


if __name__ == "__main__":
    unittest.main()
