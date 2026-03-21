import unittest


from dl_pipeline.common.config import deep_merge_dicts


class ConfigMergeTests(unittest.TestCase):
    def test_deep_merge_preserves_base_and_overrides_nested_values(self):
        base = {
            "model": {"name": "resnet18", "pretrained": True, "dropout": 0.2},
            "train": {"batch_size": 32, "epochs": 10},
        }
        override = {
            "model": {"dropout": 0.5},
            "train": {"epochs": 20},
            "experiment_name": "exp_001",
        }

        merged = deep_merge_dicts(base, override)

        self.assertEqual(merged["model"]["name"], "resnet18")
        self.assertTrue(merged["model"]["pretrained"])
        self.assertEqual(merged["model"]["dropout"], 0.5)
        self.assertEqual(merged["train"]["batch_size"], 32)
        self.assertEqual(merged["train"]["epochs"], 20)
        self.assertEqual(merged["experiment_name"], "exp_001")


if __name__ == "__main__":
    unittest.main()
