from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path


PANEL_W = 1320
PANEL_H = 860
HEADER_H = 84
GAP = 16
RAW_BOX_W = 820
TEXT = (25, 25, 25)
BG = (248, 248, 248)


def _load_raw_rgb(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _read_bgr(image_path: Path) -> np.ndarray | None:
    if not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def _load_png_rgb(path: Path) -> np.ndarray | None:
    bgr = _read_bgr(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit_rgb(image_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _load_submission_map(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return {int(row["id"]): int(row["class"]) for _, row in df.iterrows()}


def _make_missing_panel(message: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 230, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (160, 160, 160), 2)
    y = height // 2 - 12
    for line in message.split("\n"):
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        x = max(8, (width - size[0]) // 2)
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1, cv2.LINE_AA)
        y += 28
    return panel


def _draw_header(canvas: np.ndarray, text: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HEADER_H), (235, 235, 235), -1)
    cv2.putText(canvas, text, (GAP, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, TEXT, 2, cv2.LINE_AA)


def _format_pred(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return str(int(value))


def _format_score(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _build_test_fallback_candidates(
    audit_df: pd.DataFrame,
    split_df: pd.DataFrame,
    pred_map: dict[int, int],
    pred_e086_map: dict[int, int] | None = None,
    pred_e087_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    return _build_test_candidates(
        audit_df,
        split_df,
        pred_map,
        pred_e086_map=pred_e086_map,
        pred_e087_map=pred_e087_map,
        selection_sources=["crop_fallback"],
    )


def _build_test_candidates(
    audit_df: pd.DataFrame,
    split_df: pd.DataFrame,
    pred_map: dict[int, int],
    pred_e086_map: dict[int, int] | None = None,
    pred_e087_map: dict[int, int] | None = None,
    selection_sources: list[str] | None = None,
    pred_values: list[int] | None = None,
) -> pd.DataFrame:
    pred_e086_map = pred_e086_map or {}
    pred_e087_map = pred_e087_map or {}
    test_audit = audit_df[audit_df["split"] == "test"].copy()
    if selection_sources:
        allowed_sources = {str(v) for v in selection_sources}
        test_audit = test_audit[test_audit["selection_source"].astype(str).isin(allowed_sources)].copy()
    merged = test_audit.merge(split_df[["id", "image_path", "source_path"]], on="id", how="left")
    merged["pred_e088"] = merged["id"].astype(int).map(pred_map)
    merged["pred_e086"] = merged["id"].astype(int).map(pred_e086_map)
    merged["pred_e087"] = merged["id"].astype(int).map(pred_e087_map)
    if pred_values is not None:
        allowed_preds = {int(v) for v in pred_values}
        merged = merged[merged["pred_e088"].isin(allowed_preds)].copy()
    merged["probe_rank_score"] = 1.0
    merged = merged.sort_values(["probe_rank_score", "id"], ascending=[False, True]).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    return merged


def _build_candidate_lines(row: pd.Series) -> list[str]:
    return [
        f"source={row['selection_source']}  faces={int(row['num_faces_detected'])}  selected_face={int(row['selected_face_index'])}",
        f"probe_score={float(row['probe_rank_score']):.3f}  pred_e088={_format_pred(row.get('pred_e088', 'NA'))}",
        f"score_j={_format_score(row.get('selected_score_jesse', np.nan))}  score_m={_format_score(row.get('selected_score_mila', np.nan))}",
        f"excess_j={_format_score(row.get('selected_excess_jesse', np.nan))}  excess_m={_format_score(row.get('selected_excess_mila', np.nan))}",
        f"pred_e086={_format_pred(row.get('pred_e086', 'NA'))}  pred_e087={_format_pred(row.get('pred_e087', 'NA'))}",
    ]


def _render_candidate_canvas(raw_rgb: np.ndarray | None, selected_rgb: np.ndarray | None, row: pd.Series) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    header = (
        f"rank={int(row['rank'])} | split=test | id={int(row['id'])} | source={row['selection_source']} "
        f"| faces={int(row['num_faces_detected'])} | pred={_format_pred(row.get('pred_e088', 'NA'))}"
    )
    _draw_header(canvas, header)

    raw_w = RAW_BOX_W
    raw_h = PANEL_H - HEADER_H - GAP * 2
    raw_panel = _make_missing_panel("raw\nmissing", raw_w, raw_h) if raw_rgb is None else _fit_rgb(raw_rgb, raw_w, raw_h)
    canvas[HEADER_H + GAP:HEADER_H + GAP + raw_h, GAP:GAP + raw_w] = raw_panel

    right_x = GAP * 2 + raw_w
    right_w = PANEL_W - right_x - GAP
    y = HEADER_H + GAP
    cv2.putText(canvas, "SEL", (right_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
    selected_panel = _make_missing_panel("SEL\nmissing", right_w, 180) if selected_rgb is None else _fit_rgb(selected_rgb, right_w, 180)
    canvas[y + 28:y + 28 + 180, right_x:right_x + right_w] = selected_panel

    text_y = y + 250
    for line in _build_candidate_lines(row):
        cv2.putText(canvas, line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT, 1, cv2.LINE_AA)
        text_y += 30
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test-only exp_088 crop-fallback candidates.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta.yaml",
    )
    parser.add_argument(
        "--submission",
        default="data/submissions/20260403_222929_exp_088_vit_adaface_multiface_selected_20ep_last2block_ce_neighborhoodaware_fixed055_hfliptta_submission.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/visualizations/bad_image_probe/exp_088_test_crop_logic_v2",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--selection-sources", nargs="*", default=["crop_fallback"])
    parser.add_argument("--pred-values", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    processed_dir = project_path(config["data"]["processed_dir"])
    splits_dir = project_path(config["data"]["splits_dir"])

    audit_df = pd.read_csv(processed_dir / "selection_audit.csv")
    test_split_df = pd.read_csv(splits_dir / "test.csv")

    pred_e088 = _load_submission_map(project_path(args.submission))
    pred_e086 = _load_submission_map(project_path("data/submissions/20260403_195050_exp_086_buffalo_l_detect_align_multiface_dual_verifier_lookalike_submission.csv"))
    pred_e087 = _load_submission_map(project_path("data/submissions/20260403_203126_exp_087_multiface_exact_raw_hash_override_submission.csv"))

    export_df = _build_test_candidates(
        audit_df,
        test_split_df,
        pred_e088,
        pred_e086_map=pred_e086,
        pred_e087_map=pred_e087,
        selection_sources=list(args.selection_sources),
        pred_values=args.pred_values,
    )
    export_df = export_df.head(int(args.top_n)).copy()

    out_root = ensure_dir(project_path(args.out_dir))
    images_root = ensure_dir(out_root / "images")

    for old_png in images_root.glob("*.png"):
        old_png.unlink()

    for _, row in export_df.iterrows():
        raw_path = project_path(str(row["source_path"]))
        sel_path = project_path(str(row["image_path"]))
        raw_rgb = _load_raw_rgb(raw_path) if raw_path.is_file() else None
        selected_rgb = _load_png_rgb(sel_path)
        canvas = _render_candidate_canvas(raw_rgb, selected_rgb, row)
        stem = (
            f"rank_{int(row['rank']):03d}_test_id{int(row['id']):04d}_"
            f"{str(row['selection_source'])}_pred{_format_pred(row.get('pred_e088', 'NA'))}"
        )
        cv2.imwrite(
            str(images_root / f"{stem}.png"),
            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        )

    export_df.to_csv(out_root / "priority_candidates.csv", index=False)
    audit_df[audit_df["split"] == "test"].to_csv(out_root / "all_test_audit.csv", index=False)
    (out_root / "README.md").write_text(
        "# bad image probe\n\n"
        "- source: exp_088 crop pipeline selection_audit.csv\n"
        "- scope: test only\n"
        f"- candidates: selection_source in {list(args.selection_sources)}\n"
        f"- prediction filter: {args.pred_values if args.pred_values is not None else 'none'}\n"
        "- no hash / dhash anchor heuristics\n"
        "- `priority_candidates.csv`: fallback candidate list\n"
        "- `all_test_audit.csv`: full test audit rows from exp_088 preprocessing\n"
        "- `images/`: rendered raw + selected crop panels\n",
        encoding="utf-8",
    )
    print(f"done: exported {len(export_df)} test crop-fallback candidates -> {out_root}")


if __name__ == "__main__":
    main()
