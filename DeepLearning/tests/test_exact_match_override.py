from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.inference.exact_match_override import (
    apply_exact_hash_overrides,
    build_consistent_hash_label_map,
)


class ExactMatchOverrideTests(unittest.TestCase):
    def test_build_consistent_hash_label_map_keeps_only_consistent_labels(self):
        pairs = [
            ("hash_a", 2),
            ("hash_b", 1),
            ("hash_a", 2),
            ("hash_c", 0),
            ("hash_c", 1),
        ]

        result = build_consistent_hash_label_map(pairs)

        self.assertEqual(result, {"hash_a": 2, "hash_b": 1})

    def test_apply_exact_hash_overrides_only_updates_matching_ids(self):
        predictions = {10: 0, 11: 2, 12: 1}
        test_hashes = {10: "hash_a", 11: "hash_x", 12: "hash_b"}
        label_map = {"hash_a": 2, "hash_b": 1}

        updated, overridden = apply_exact_hash_overrides(predictions, test_hashes, label_map)

        self.assertEqual(updated, {10: 2, 11: 2, 12: 1})
        self.assertEqual(overridden, {10: {"from": 0, "to": 2, "hash": "hash_a"}})


if __name__ == "__main__":
    unittest.main()
