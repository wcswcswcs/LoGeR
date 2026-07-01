from __future__ import annotations

import unittest

from stream4d_native.v63_action_outcome import _action_flags, _action_outcome, _utility_score


class V63ActionOutcomeTest(unittest.TestCase):
    def test_confirm_requires_valid_kmat_heldout_evidence(self) -> None:
        outcome = _action_outcome(
            planned_action="confirm",
            candidate_type="heldout_recovery",
            valid_material_evidence=True,
            outside_residual_rate=0.0,
            has_k_mat=True,
        )
        flags = _action_flags("confirm", "heldout_recovery", outcome, True)

        self.assertEqual(outcome, "confirm")
        self.assertTrue(flags["action_success"])
        self.assertFalse(flags["false_confirm"])

    def test_decoy_rejection_treats_non_confirm_as_success(self) -> None:
        outcome = _action_outcome(
            planned_action="reject_decoy",
            candidate_type="decoy_rejection",
            valid_material_evidence=True,
            outside_residual_rate=0.0,
            has_k_mat=True,
        )
        flags = _action_flags("reject_decoy", "decoy_rejection", outcome, True)

        self.assertEqual(outcome, "quarantine")
        self.assertTrue(flags["action_success"])
        self.assertFalse(flags["false_confirm"])
        self.assertGreater(_utility_score("reject_decoy", outcome, flags, True), 0.0)

    def test_noisy_confirm_is_penalized(self) -> None:
        outcome = _action_outcome(
            planned_action="confirm",
            candidate_type="heldout_recovery",
            valid_material_evidence=False,
            outside_residual_rate=1.0,
            has_k_mat=True,
        )
        flags = _action_flags("confirm", "heldout_recovery", outcome, False)

        self.assertEqual(outcome, "defer")
        self.assertFalse(flags["action_success"])
        self.assertTrue(flags["noise_failure"])
        self.assertLess(_utility_score("confirm", outcome, flags, False), 0.0)

    def test_noop_and_semantic_controls_do_not_get_safe_defer_credit(self) -> None:
        noop_flags = _action_flags("control_noop", "control", "defer", False)
        semantic_flags = _action_flags("control_semantic_only", "control", "defer", False)

        self.assertFalse(noop_flags["action_success"])
        self.assertFalse(noop_flags["safe_defer"])
        self.assertFalse(semantic_flags["action_success"])
        self.assertFalse(semantic_flags["safe_defer"])


if __name__ == "__main__":
    unittest.main()
