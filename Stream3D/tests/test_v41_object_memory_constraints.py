from __future__ import annotations

import unittest

from stream4d_native.semantic_material_memory import MemoryObservation, SemanticMaterialMemory


class V41ObjectMemoryConstraintTests(unittest.TestCase):
    def test_memory_cannot_create_object_without_semantic_evidence(self) -> None:
        memory = SemanticMaterialMemory(min_material_consistency=0.5)
        result = memory.update([MemoryObservation(3, False, 0.95, 10)])
        self.assertEqual(result.diagnostics["object_count"], 0)
        self.assertEqual(result.diagnostics["memory_birth_without_semantic_support_count"], 1)

    def test_memory_reactivates_with_semantic_and_material_support(self) -> None:
        memory = SemanticMaterialMemory(min_material_consistency=0.5)
        first = memory.update([MemoryObservation(3, True, 0.95, 0)])
        self.assertEqual(first.diagnostics["object_count"], 1)
        memory.objects[3].active = False
        second = memory.update([MemoryObservation(3, True, 0.90, 4)])
        self.assertEqual(second.diagnostics["reactivation_success"], 1)
        self.assertEqual(memory.objects[3].reactivation_count, 1)


if __name__ == "__main__":
    unittest.main()

