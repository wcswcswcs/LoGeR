from __future__ import annotations

import unittest

from stream4d_native.v63_fact_lock import _comparison_rows, _observed_facts


class V63FactLockTest(unittest.TestCase):
    def test_observed_facts_are_extracted_from_payload(self) -> None:
        payload = {
            "primary": {
                "query_count": 80,
                "valid_material_evidence_rate": 0.5125,
                "query_to_confirm_rate": 0.2,
                "query_to_quarantine_rate": 0.3,
                "query_to_confirm_or_quarantine_rate": 0.5,
                "material_outcome_counts": {"confirm": 16, "quarantine": 24, "unresolved": 40},
            },
            "with_negative": {
                "query_count": 164,
                "valid_material_evidence_rate": 0.5914634146341463,
                "query_to_confirm_or_quarantine_rate": 0.5365853658536586,
                "material_outcome_counts": {"confirm": 37, "quarantine": 51, "unresolved": 76},
            },
            "with_negative_auc": {
                "real_query_AUC": 0.9102870813397129,
                "real_minus_shuffled_query_AUC": 0.26315789473684215,
                "real_minus_no_temporal_query_AUC": 0.22562799043062198,
                "diagnostic_real_query_AUC": 0.45226860254083484,
                "diagnostic_real_minus_shuffled_query_AUC": 0.007622504537205088,
                "diagnostic_real_minus_no_temporal_query_AUC": -0.03484573502722321,
            },
        }

        observed = _observed_facts(payload)

        self.assertEqual(observed["primary_confirm_count"], 16)
        self.assertEqual(observed["with_negative_unresolved_count"], 76)
        self.assertEqual(observed["with_negative_real_query_AUC"], 0.9102870813397129)

    def test_comparison_rows_detect_expected_value_mismatch(self) -> None:
        observed = {
            "primary_query_count": 80,
            "primary_valid_material_evidence_rate": 0.5125,
            "primary_query_to_confirm_rate": 0.2,
            "primary_query_to_quarantine_rate": 0.3,
            "primary_query_to_confirm_or_quarantine_rate": 0.5,
            "primary_confirm_count": 16,
            "primary_quarantine_count": 24,
            "primary_unresolved_count": 40,
            "with_negative_query_count": 163,
            "with_negative_valid_material_evidence_rate": 0.5914634146341463,
            "with_negative_query_to_confirm_or_quarantine_rate": 0.5365853658536586,
            "with_negative_confirm_count": 37,
            "with_negative_quarantine_count": 51,
            "with_negative_unresolved_count": 76,
            "with_negative_real_query_AUC": 0.9102870813397129,
            "with_negative_real_minus_shuffled_query_AUC": 0.26315789473684215,
            "with_negative_real_minus_no_temporal_query_AUC": 0.22562799043062198,
            "with_negative_diagnostic_real_query_AUC": 0.45226860254083484,
            "with_negative_diagnostic_real_minus_shuffled_query_AUC": 0.007622504537205088,
            "with_negative_diagnostic_real_minus_no_temporal_query_AUC": -0.03484573502722321,
        }

        rows = _comparison_rows(observed, tolerance=1.0e-12)
        mismatch = [row for row in rows if row["metric_name"] == "with_negative_query_count"][0]

        self.assertFalse(mismatch["matches_plan_expected"])
        self.assertEqual(mismatch["observed_value"], 163)
        self.assertEqual(mismatch["plan_expected_value"], 164)


if __name__ == "__main__":
    unittest.main()
