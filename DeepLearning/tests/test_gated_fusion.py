import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.inference.gated_fusion import (
    apply_agreement_aware_rescue_only_fusion,
    apply_gated_anchor_fusion,
    apply_rescue_only_fusion,
    select_best_agreement_aware_rescue_only_params,
    select_best_rescue_only_params,
    select_best_gated_fusion_params,
    summarize_dual_verifier_scores,
)


class GatedFusionTests(unittest.TestCase):
    def test_summarize_dual_verifier_scores_tracks_selected_and_best_excess(self):
        summary = summarize_dual_verifier_scores(
            score_jesse=np.array([0.70, 0.20, 0.65], dtype=np.float32),
            score_mila=np.array([0.60, 0.55, 0.75], dtype=np.float32),
            theta_jesse=0.50,
            theta_mila=0.50,
        )

        self.assertEqual(summary["predictions"].tolist(), [1, 2, 2])
        self.assertTrue(np.allclose(summary["selected_excess"], [0.20, 0.05, 0.25], atol=1e-6))
        self.assertTrue(np.allclose(summary["best_excess"], [0.20, 0.05, 0.25], atol=1e-6))

    def test_apply_gated_anchor_fusion_can_rescue_primary_reject(self):
        fused = apply_gated_anchor_fusion(
            anchor_predictions=np.array([0, 0, 0], dtype=np.int64),
            anchor_scores=np.array([0.48, 0.40, 0.51], dtype=np.float32),
            secondary_predictions=np.array([1, 2, 1], dtype=np.int64),
            secondary_selected_excess=np.array([0.10, 0.02, 0.08], dtype=np.float32),
            secondary_best_excess=np.array([0.10, 0.02, 0.08], dtype=np.float32),
            other_label=0,
            low_conf_threshold=0.55,
            rescue_margin=0.05,
            swap_margin=0.08,
            veto_excess_threshold=-0.02,
        )

        self.assertEqual(fused.tolist(), [1, 0, 1])

    def test_apply_gated_anchor_fusion_only_swaps_targets_when_primary_is_weak(self):
        fused = apply_gated_anchor_fusion(
            anchor_predictions=np.array([1, 1, 2], dtype=np.int64),
            anchor_scores=np.array([0.52, 0.61, 0.50], dtype=np.float32),
            secondary_predictions=np.array([2, 2, 1], dtype=np.int64),
            secondary_selected_excess=np.array([0.12, 0.12, 0.04], dtype=np.float32),
            secondary_best_excess=np.array([0.12, 0.12, 0.04], dtype=np.float32),
            other_label=0,
            low_conf_threshold=0.55,
            rescue_margin=0.05,
            swap_margin=0.08,
            veto_excess_threshold=-0.02,
        )

        self.assertEqual(fused.tolist(), [2, 1, 2])

    def test_apply_gated_anchor_fusion_can_veto_to_other_when_secondary_is_confidently_negative(self):
        fused = apply_gated_anchor_fusion(
            anchor_predictions=np.array([1, 2, 1], dtype=np.int64),
            anchor_scores=np.array([0.50, 0.58, 0.53], dtype=np.float32),
            secondary_predictions=np.array([0, 0, 0], dtype=np.int64),
            secondary_selected_excess=np.array([-0.03, 0.01, -0.06], dtype=np.float32),
            secondary_best_excess=np.array([-0.03, 0.01, -0.06], dtype=np.float32),
            other_label=0,
            low_conf_threshold=0.55,
            rescue_margin=0.05,
            swap_margin=0.08,
            veto_excess_threshold=-0.02,
        )

        self.assertEqual(fused.tolist(), [0, 2, 0])

    def test_select_best_gated_fusion_params_prefers_higher_accuracy_then_fewer_changes(self):
        best, records = select_best_gated_fusion_params(
            labels=np.array([2, 1, 1, 0], dtype=np.int64),
            anchor_predictions=np.array([1, 0, 1, 1], dtype=np.int64),
            anchor_scores=np.array([0.52, 0.48, 0.62, 0.51], dtype=np.float32),
            secondary_predictions=np.array([2, 1, 1, 0], dtype=np.int64),
            secondary_selected_excess=np.array([0.11, 0.08, 0.03, -0.04], dtype=np.float32),
            secondary_best_excess=np.array([0.11, 0.08, 0.03, -0.04], dtype=np.float32),
            other_label=0,
            low_conf_threshold_values=[0.50, 0.55],
            rescue_margin_values=[0.05],
            swap_margin_values=[0.08, 0.12],
            veto_excess_threshold_values=[-0.02],
        )

        self.assertEqual(best["low_conf_threshold"], 0.55)
        self.assertEqual(best["swap_margin"], 0.08)
        self.assertAlmostEqual(best["accuracy"], 1.0, places=6)
        self.assertEqual(len(records), 4)

    def test_apply_rescue_only_fusion_can_require_anchor_argmax_match(self):
        fused = apply_rescue_only_fusion(
            anchor_predictions=np.array([0, 0, 1], dtype=np.int64),
            anchor_argmax=np.array([1, 2, 1], dtype=np.int64),
            anchor_scores=np.array([0.20, 0.22, 0.70], dtype=np.float32),
            secondary_predictions=np.array([1, 1, 2], dtype=np.int64),
            secondary_excess_jesse=np.array([0.18, 0.18, -0.05], dtype=np.float32),
            secondary_excess_mila=np.array([-0.20, -0.10, 0.12], dtype=np.float32),
            other_label=0,
            low_conf_threshold=0.55,
            rescue_margin_jesse=0.10,
            rescue_margin_mila=0.10,
            require_anchor_argmax_match=True,
        )

        self.assertEqual(fused.tolist(), [1, 0, 1])

    def test_select_best_rescue_only_params_prefers_accuracy_then_fewer_changes(self):
        best, records = select_best_rescue_only_params(
            labels=np.array([1, 0, 2, 0], dtype=np.int64),
            anchor_predictions=np.array([0, 0, 0, 0], dtype=np.int64),
            anchor_argmax=np.array([1, 1, 2, 2], dtype=np.int64),
            anchor_scores=np.array([0.20, 0.30, 0.25, 0.45], dtype=np.float32),
            secondary_predictions=np.array([1, 1, 2, 2], dtype=np.int64),
            secondary_excess_jesse=np.array([0.18, 0.09, -0.20, -0.30], dtype=np.float32),
            secondary_excess_mila=np.array([-0.10, -0.20, 0.16, 0.04], dtype=np.float32),
            other_label=0,
            low_conf_threshold_values=[0.40, 0.55],
            rescue_margin_jesse_values=[0.08, 0.12],
            rescue_margin_mila_values=[0.10, 0.18],
            require_anchor_argmax_match_options=[False, True],
        )

        self.assertEqual(best["low_conf_threshold"], 0.4)
        self.assertEqual(best["rescue_margin_jesse"], 0.12)
        self.assertEqual(best["rescue_margin_mila"], 0.1)
        self.assertFalse(best["require_anchor_argmax_match"])
        self.assertAlmostEqual(best["accuracy"], 1.0, places=6)
        self.assertEqual(len(records), 16)

    def test_apply_agreement_aware_rescue_only_fusion_uses_different_thresholds_by_agreement(self):
        fused = apply_agreement_aware_rescue_only_fusion(
            anchor_predictions=np.array([0, 0, 0, 0, 1], dtype=np.int64),
            anchor_argmax=np.array([1, 2, 2, 1, 1], dtype=np.int64),
            anchor_scores=np.array([0.20, 0.20, 0.20, 0.20, 0.70], dtype=np.float32),
            secondary_predictions=np.array([1, 1, 2, 2, 2], dtype=np.int64),
            secondary_excess_jesse=np.array([0.08, 0.08, -0.20, -0.20, -0.10], dtype=np.float32),
            secondary_excess_mila=np.array([-0.20, -0.20, 0.08, 0.08, 0.12], dtype=np.float32),
            other_label=0,
            low_conf_threshold=0.55,
            rescue_margin_jesse_agree=0.06,
            rescue_margin_jesse_disagree=0.10,
            rescue_margin_mila_agree=0.06,
            rescue_margin_mila_disagree=0.10,
        )

        self.assertEqual(fused.tolist(), [1, 0, 2, 0, 1])

    def test_select_best_agreement_aware_rescue_only_params_prefers_accuracy_then_fewer_changes(self):
        best, records = select_best_agreement_aware_rescue_only_params(
            labels=np.array([1, 0, 2, 0], dtype=np.int64),
            anchor_predictions=np.array([0, 0, 0, 0], dtype=np.int64),
            anchor_argmax=np.array([1, 2, 2, 1], dtype=np.int64),
            anchor_scores=np.array([0.20, 0.20, 0.25, 0.25], dtype=np.float32),
            secondary_predictions=np.array([1, 1, 2, 2], dtype=np.int64),
            secondary_excess_jesse=np.array([0.08, 0.08, -0.20, -0.20], dtype=np.float32),
            secondary_excess_mila=np.array([-0.20, -0.20, 0.08, 0.08], dtype=np.float32),
            other_label=0,
            low_conf_threshold_values=[0.40, 0.55],
            rescue_margin_jesse_agree_values=[0.06, 0.10],
            rescue_margin_jesse_disagree_values=[0.10, 0.14],
            rescue_margin_mila_agree_values=[0.06, 0.10],
            rescue_margin_mila_disagree_values=[0.10, 0.14],
        )

        self.assertEqual(best["low_conf_threshold"], 0.4)
        self.assertEqual(best["rescue_margin_jesse_agree"], 0.06)
        self.assertEqual(best["rescue_margin_jesse_disagree"], 0.1)
        self.assertEqual(best["rescue_margin_mila_agree"], 0.06)
        self.assertEqual(best["rescue_margin_mila_disagree"], 0.1)
        self.assertAlmostEqual(best["accuracy"], 1.0, places=6)
        self.assertEqual(len(records), 32)


if __name__ == "__main__":
    unittest.main()
