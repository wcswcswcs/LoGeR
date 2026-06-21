from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stream4d_native.v53_native_carrier_materialization import build_native_carrier_materialization


class V53NativeCarrierMaterializationTest(unittest.TestCase):
    def test_objectlet_components_materialize_to_d4rt_carrier_support_not_ap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            carrier = root / "carrier.csv"
            mask = root / "mask.csv"
            objectlets = root / "objectlets.csv"
            carrier.write_text(
                "\n".join(
                    [
                        "scene,carrier_id,carrier_global_id,frame_id,chunk_id,submap_id,window_index,carrier_index,uv_x,uv_y,visible,confidence,visibility_prob,valid,valid_uv,observed_mask_id,uses_gt_for_prediction",
                        "s,1,s:1,0,0,0,0,0,0.1,0.2,True,0.9,0.9,True,True,1,False",
                        "s,2,s:2,0,0,0,0,1,0.2,0.2,True,0.9,0.9,True,True,1,False",
                        "s,2,s:2,1,0,0,0,2,0.2,0.3,True,0.9,0.9,True,True,2,False",
                        "s,3,s:3,1,0,0,0,3,0.3,0.3,True,0.9,0.9,True,True,2,False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            mask.write_text(
                "\n".join(
                    [
                        "mask_observation_id,scene,frame_id,mask_id,uses_gt_for_prediction,uses_gt_for_diagnostic_labels",
                        "s:0:1,s,0,1,False,False",
                        "s:1:2,s,1,2,False,False",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            objectlets.write_text(
                "\n".join(
                    [
                        "variant,objectlet_id,scene,chunk_id,candidate_id,source_mask_observation_id,component_ids,uses_gt_for_prediction",
                        'L6_test,s|L6_test|obj00000,s,s:chunk000,cand0,s:0:1,"[""c00000""]",False',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = build_native_carrier_materialization(
                carrier_table_path=carrier,
                mask_table_path=mask,
                objectlet_rows_path=objectlets,
                objectlet_variant="L6_test",
            )

        summary = payload["summary"]
        self.assertTrue(summary["native_carrier_materialization_pass"])
        self.assertTrue(summary["method_safe_native_support_available"])
        self.assertFalse(summary["method_safe_ap_available"])
        self.assertFalse(summary["is_scannet_ap_export"])
        self.assertEqual(summary["native_unique_carrier_count"], 3)
        self.assertEqual(summary["native_observation_row_count"], 4)
        self.assertEqual(payload["carrier_rows"][0]["native_support_kind"], "d4rt_carrier_global_id")


if __name__ == "__main__":
    unittest.main()
