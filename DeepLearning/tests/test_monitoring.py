import unittest

from dl_pipeline.training.monitoring import resolve_monitor_config


class MonitoringConfigTests(unittest.TestCase):
    def test_resolve_monitor_config_defaults_to_val_acc(self):
        monitor = resolve_monitor_config({})

        self.assertEqual(monitor.metric, "val_acc")
        self.assertEqual(monitor.mode, "max")

    def test_resolve_monitor_config_accepts_val_loss_override(self):
        monitor = resolve_monitor_config({"monitor_metric": "val_loss"})

        self.assertEqual(monitor.metric, "val_loss")
        self.assertEqual(monitor.mode, "min")

    def test_resolve_monitor_config_allows_explicit_mode(self):
        monitor = resolve_monitor_config(
            {
                "monitor_metric": "val_f1",
                "monitor_mode": "max",
            }
        )

        self.assertEqual(monitor.metric, "val_f1")
        self.assertEqual(monitor.mode, "max")


if __name__ == "__main__":
    unittest.main()
