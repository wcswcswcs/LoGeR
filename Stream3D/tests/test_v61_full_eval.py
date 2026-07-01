from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from stream4d_native.v47_common import write_json
from stream4d_native.v61_full_eval import V61FullEvalConfig, build_v61_final_decision


class V61FullEvalTest(unittest.TestCase):
    def test_core_go_can_block_query_claim(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="v61_full_eval_test_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        phase0 = root / "phase0.json"
        graph = root / "graph.json"
        embedding = root / "embedding.json"
        refinement = root / "refine.json"
        query = root / "query.json"
        stress = root / "stress.json"
        native = root / "native.json"
        write_json(phase0, {"gate": {"pass": True}, "material_state_coverage_rate": 0.0})
        write_json(graph, {"gate": {"pass": True}, "material_nodes_with_candidate_rate": 1.0})
        write_json(
            embedding,
            {
                "gate": {"pass": True},
                "core_purity": 0.99,
                "core_completeness": 0.99,
                "expanded_completeness": 0.99,
                "same_category_merge_rate": 0.0,
                "real_minus_shuffled_ARI": 0.9,
                "real_minus_no_temporal_ARI": 0.9,
            },
        )
        write_json(refinement, {"gate": {"pass": False}, "quarantine_precision_diagnostic": 1.0, "quarantined_node_count": 3})
        write_json(query, {"gate": {"pass": False}, "query_to_confirm_or_quarantine_rate": 0.1})
        write_json(stress, {"gate": {"pass": True}, "stress_real_minus_mask_only_ARI_pass_count": 3})
        write_json(native, {"gate": {"pass": True}, "method_safe_native_support_available": True, "ap_diagnostic_status": "not_run"})
        result = build_v61_final_decision(
            V61FullEvalConfig(
                phase0_summary_path=phase0,
                graph_summary_path=graph,
                embedding_summary_path=embedding,
                refinement_summary_path=refinement,
                query_summary_path=query,
                stress_summary_path=stress,
                native_summary_path=native,
            )
        )
        summary = result["summary"]
        self.assertEqual(summary["decision_label"], "GO_SOMA_MANIFOLD_GLOBAL_EMBEDDING")
        self.assertIn("active_material_query_claim", summary["blocked_claims"])
        self.assertFalse(summary["query_gate_pass"])


if __name__ == "__main__":
    unittest.main()
