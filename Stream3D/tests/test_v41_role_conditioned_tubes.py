from __future__ import annotations

import unittest

from stream4d_native.material_tube_roles import MaterialTubeEvidence, infer_tube_roles, summarize_tube_roles
from stream4d_native.object_aware_self_stitch import evaluate_role_aware_stitch_variants


class V41RoleConditionedTubeTests(unittest.TestCase):
    def _evidence(self) -> list[MaterialTubeEvidence]:
        return [
            MaterialTubeEvidence(0, 0.95, 0.95, 0.01, 0.95, 0.20, 0.01, 1.00),
            MaterialTubeEvidence(1, 0.90, 0.90, 0.02, 0.90, 0.25, 0.02, 1.01),
            MaterialTubeEvidence(2, 0.92, 0.90, 0.18, 0.40, 0.92, 0.25, 1.18),
            MaterialTubeEvidence(3, 0.85, 0.80, 0.16, 0.35, 0.88, 0.22, 1.15),
            MaterialTubeEvidence(5, 0.95, 0.95, 0.04, 0.90, 0.90, 0.04, 1.02),
            MaterialTubeEvidence(4, 0.20, 0.25, 0.20, 0.20, 0.20, 0.10, 1.05),
        ]

    def test_roles_keep_static_scene_and_object_support_separate_with_unknowns(self) -> None:
        evidences = self._evidence()
        roles = infer_tube_roles(evidences)
        role_by_id = {role.tube_id: role.role for role in roles}
        self.assertEqual(role_by_id[0], "scene")
        self.assertEqual(role_by_id[2], "object")
        self.assertEqual(role_by_id[5], "part")
        self.assertEqual(role_by_id[4], "unknown")
        summary = summarize_tube_roles(evidences, roles)
        self.assertGreater(summary["unknown_role_ratio"], 0.0)
        self.assertGreater(summary["part_role_count"], 0)
        self.assertGreater(summary["role_entropy"], 0.0)
        self.assertLess(summary["self_stitch_residual_scene_mean"], summary["self_stitch_residual_all_mean"])
        self.assertGreater(
            summary["object_support_consistency_object_mean"],
            summary["object_support_consistency_unknown_mean"],
        )

    def test_role_aware_self_stitch_reduces_dynamic_leakage_and_residual(self) -> None:
        evidences = self._evidence()
        roles = infer_tube_roles(evidences)
        rows = evaluate_role_aware_stitch_variants(evidences, roles)
        by_variant = {row["variant"]: row for row in rows}
        self.assertLess(
            by_variant["D3_role_posterior_robust_residual"]["dynamic_leakage_ratio"],
            by_variant["D0_all_tubes"]["dynamic_leakage_ratio"],
        )
        self.assertLess(
            by_variant["D3_role_posterior_robust_residual"]["self_sim3_residual_p90"],
            by_variant["D0_all_tubes"]["self_sim3_residual_p90"],
        )
        self.assertGreater(
            by_variant["D4_dynamic_tubes_negative_control"]["dynamic_leakage_ratio"],
            by_variant["D3_role_posterior_robust_residual"]["dynamic_leakage_ratio"],
        )


if __name__ == "__main__":
    unittest.main()
