from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stream4d_native.v53_semantic_guard import build_semantic_guard


class V53SemanticGuardTest(unittest.TestCase):
    def test_colorhist_fallback_guard_stays_diagnostic(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            support = root / "support.csv"
            masks = root / "masks.csv"
            objectlets = root / "objectlets.csv"
            summary = root / "summary.json"
            support.write_text(
                "\n".join(
                    [
                        "variant,mask_observation_id,scene,frame_id,mask_id,component_id,support_count,diagnostic_gt_instance,uses_gt_for_prediction,uses_gt_for_diagnostic_labels",
                        "R0_visible_tau0.05,s:0:1,s,0,1,c1,5,a,False,True",
                        "R0_visible_tau0.05,s:1:2,s,1,2,c2,5,b,False,True",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            masks.write_text(
                "\n".join(
                    [
                        "mask_observation_id,core_feature,feature_backend,core_feature_valid,uses_gt_for_prediction,uses_gt_for_diagnostic_labels",
                        's:0:1,"[1.0, 0.0]",colorhist_fallback,True,False,True',
                        's:1:2,"[0.0, 1.0]",colorhist_fallback,True,False,True',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            objectlets.write_text(
                "\n".join(
                    [
                        "variant,objectlet_id,scene,chunk_id,candidate_id,source_mask_observation_id,component_ids,underseg_proxy,uses_gt_for_prediction,uses_gt_for_diagnostic_labels",
                        'L6_test,s|L6_test|obj00000,s,s:chunk000,cand0,s:0:1,"[""c1"", ""c2""]",True,False,True',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary.write_text(
                '{"best_real_variant": "L6_test", "best_real_row": {"variant": "L6_test", "4D_ARI": 0.5, "4D_purity": 0.8, "4D_completeness": 0.7}}\n',
                encoding="utf-8",
            )
            payload = build_semantic_guard(
                support_rows_path=support,
                mask_table_path=masks,
                objectlet_summary_path=summary,
                objectlet_rows_path=objectlets,
                objectlet_variant="L6_test",
            )

        guard = payload["summary"]
        self.assertFalse(guard["dense_semantic_available"])
        self.assertEqual(guard["feature_backend"], "colorhist_fallback")
        self.assertFalse(guard["semantic_claim_allowed"])
        self.assertFalse(guard["semantic_guard_method_enabled"])
        self.assertEqual(len(payload["objectlet_semantic_rows"]), 1)
        self.assertGreater(payload["objectlet_semantic_rows"][0]["semantic_contradiction_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
