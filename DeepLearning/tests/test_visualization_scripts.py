from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    for path in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualizationScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bad_probe = _load_module("export_bad_image_probe", "scripts/export_bad_image_probe.py")
        cls.champion_probe = _load_module(
            "export_test_champion_risk_visualization",
            "scripts/export_test_champion_risk_visualization.py",
        )

    def test_bad_image_probe_load_raw_rgb_converts_bgr_npy_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "sample.npy"
            bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
            np.save(npy_path, bgr)

            rgb = self.bad_probe._load_raw_rgb(npy_path)

            expected = np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8)
            np.testing.assert_array_equal(rgb, expected)

    def test_bad_image_probe_candidate_lines_split_prediction_block(self):
        row = pd.Series(
            {
                "selection_source": "crop_fallback",
                "num_faces_detected": 0,
                "selected_face_index": -1,
                "pred_e088": 0,
                "selected_score_jesse": np.nan,
                "selected_score_mila": np.nan,
                "selected_excess_jesse": np.nan,
                "selected_excess_mila": np.nan,
                "probe_rank_score": 1.0,
                "pred_e086": 0,
                "pred_e087": 0,
            }
        )

        lines = self.bad_probe._build_candidate_lines(row)

        self.assertGreaterEqual(len(lines), 5)
        self.assertIn("source=crop_fallback  faces=0  selected_face=-1", lines)
        self.assertIn("probe_score=1.000  pred_e088=0", lines)
        self.assertIn("pred_e086=0  pred_e087=0", lines)

    def test_champion_probe_load_raw_bgr_preserves_bgr_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "sample.npy"
            bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
            np.save(npy_path, bgr)

            raw_bgr = self.champion_probe._load_raw_bgr(npy_path)

            np.testing.assert_array_equal(raw_bgr, bgr)

    def test_champion_probe_load_raw_rgb_converts_bgr_npy_to_rgb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = Path(tmpdir) / "sample.npy"
            bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
            np.save(npy_path, bgr)

            raw_rgb = self.champion_probe._load_raw_rgb(npy_path)

            expected = np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8)
            np.testing.assert_array_equal(raw_rgb, expected)

    def test_bad_image_probe_builds_test_only_fallback_candidates(self):
        audit_df = pd.DataFrame(
            [
                {"split": "train", "id": 9, "selection_source": "crop_fallback", "num_faces_detected": 0, "selected_face_index": -1},
                {"split": "test", "id": 92, "selection_source": "crop_fallback", "num_faces_detected": 0, "selected_face_index": -1},
                {"split": "test", "id": 93, "selection_source": "detect_align_selected", "num_faces_detected": 1, "selected_face_index": 0},
            ]
        )
        split_df = pd.DataFrame(
            [
                {"id": 92, "image_path": "data/processed/test/92.png", "source_path": "data/raw/test_92.npy"},
                {"id": 93, "image_path": "data/processed/test/93.png", "source_path": "data/raw/test_93.npy"},
            ]
        )
        pred_map = {92: 0, 93: 1}

        candidates = self.bad_probe._build_test_fallback_candidates(audit_df, split_df, pred_map)

        self.assertEqual(candidates["id"].tolist(), [92])
        self.assertEqual(candidates.iloc[0]["pred_e088"], 0)
        self.assertEqual(candidates.iloc[0]["probe_rank_score"], 1.0)

    def test_bad_image_probe_can_filter_rescued_pred0_candidates(self):
        audit_df = pd.DataFrame(
            [
                {"split": "test", "id": 138, "selection_source": "detect_align_rescue_pad18", "num_faces_detected": 1, "selected_face_index": 0},
                {"split": "test", "id": 879, "selection_source": "detect_align_rescue_pad18", "num_faces_detected": 1, "selected_face_index": 0},
                {"split": "test", "id": 92, "selection_source": "crop_fallback", "num_faces_detected": 0, "selected_face_index": -1},
            ]
        )
        split_df = pd.DataFrame(
            [
                {"id": 138, "image_path": "data/processed/test/138.png", "source_path": "data/raw/test_138.npy"},
                {"id": 879, "image_path": "data/processed/test/879.png", "source_path": "data/raw/test_879.npy"},
                {"id": 92, "image_path": "data/processed/test/92.png", "source_path": "data/raw/test_92.npy"},
            ]
        )
        pred_map = {138: 0, 879: 1, 92: 0}

        candidates = self.bad_probe._build_test_candidates(
            audit_df,
            split_df,
            pred_map,
            selection_sources=["detect_align_rescue_pad18"],
            pred_values=[0],
        )

        self.assertEqual(candidates["id"].tolist(), [138])
        self.assertEqual(candidates.iloc[0]["selection_source"], "detect_align_rescue_pad18")
        self.assertEqual(candidates.iloc[0]["pred_e088"], 0)


if __name__ == "__main__":
    unittest.main()
