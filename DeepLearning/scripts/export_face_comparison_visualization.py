"""
导出 raw / HAAR / MTCNN 三联图，便于人工比较同一样本的输入质量。

在 DeepLearning 根目录执行：
  python scripts/export_face_comparison_visualization.py
  python scripts/export_face_comparison_visualization.py --split train
  python scripts/export_face_comparison_visualization.py --ids 14 40 65
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.paths import ensure_dir, project_path


SPLIT_NAMES = frozenset({"train", "val", "test"})
PANEL_SIZE = 224
LABEL_BAR_H = 28
HEADER_H = 48
GAP = 8
BG = (245, 245, 245)
TEXT = (20, 20, 20)
PANEL_TITLES = ("raw", "HAAR", "MTCNN")


def _parse_splits(arg: str) -> list[str]:
    raw = arg.strip().lower()
    if raw == "all":
        return ["train", "val"]
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    for p in parts:
        if p not in SPLIT_NAMES:
            raise SystemExit(f"未知 split: {p!r}，允许: train, val, test, all")
    return parts


def _load_raw_rgb(npy_path: Path) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"原始图像格式异常: {npy_path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _load_png_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit_panel(image_rgb: np.ndarray, panel_size: int = PANEL_SIZE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("图像尺寸非法")
    scale = min(panel_size / w, panel_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    canvas = np.full((panel_size, panel_size, 3), 255, dtype=np.uint8)
    y0 = (panel_size - new_h) // 2
    x0 = (panel_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _make_missing_panel(message: str, panel_size: int = PANEL_SIZE) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), 230, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_size - 1, panel_size - 1), (160, 160, 160), 2)
    y = panel_size // 2 - 8
    for line in message.split("\n"):
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        x = max(8, (panel_size - size[0]) // 2)
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        y += 24
    return panel


def _draw_label_bar(canvas: np.ndarray, x0: int, y0: int, width: int, text: str) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + LABEL_BAR_H), (225, 225, 225), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + LABEL_BAR_H), (200, 200, 200), 1)
    cv2.putText(canvas, text, (x0 + 8, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)


def _compose_triptych(*, split_name: str, sid: int, cls: int, raw_rgb: np.ndarray | None, haar_rgb: np.ndarray | None, mtcnn_rgb: np.ndarray | None) -> np.ndarray:
    panel_w = PANEL_SIZE
    total_w = GAP * 4 + panel_w * 3
    total_h = HEADER_H + LABEL_BAR_H + PANEL_SIZE + GAP * 2
    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    header = f"{split_name} | id={sid} | class={cls}"
    cv2.putText(canvas, header, (GAP, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2, cv2.LINE_AA)

    panels = []
    for image_rgb, missing_text in (
        (raw_rgb, "raw\nmissing"),
        (haar_rgb, "HAAR\nmissing"),
        (mtcnn_rgb, "MTCNN\nmissing"),
    ):
        if image_rgb is None:
            panels.append(_make_missing_panel(missing_text))
        else:
            panels.append(_fit_panel(image_rgb))

    for idx, (title, panel) in enumerate(zip(PANEL_TITLES, panels)):
        x0 = GAP + idx * (panel_w + GAP)
        y_label = HEADER_H
        y_img = y_label + LABEL_BAR_H
        _draw_label_bar(canvas, x0, y_label, panel_w, title)
        canvas[y_img:y_img + PANEL_SIZE, x0:x0 + panel_w] = panel

    return canvas


def _try_load_image(loader, path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return loader(path)
    except Exception:
        return None


def _resolve_processed_split_dir(base_dir: Path, split_name: str) -> Path:
    if split_name == "val":
        return base_dir / "train"
    return base_dir / split_name


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 raw / HAAR / MTCNN 三联图")
    parser.add_argument("--splits-dir", default="data/splits/exp_001")
    parser.add_argument("--split", default="all", help="train,val,test,all")
    parser.add_argument("--haar-dir", default="data/processed/exp_001_faces_224")
    parser.add_argument("--mtcnn-dir", default="data/processed/exp_008_mtcnn_faces_112")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--ids", nargs="*", type=int, default=[])
    args = parser.parse_args()

    split_names = _parse_splits(args.split)
    splits_dir = project_path(args.splits_dir)
    haar_dir = project_path(args.haar_dir)
    mtcnn_dir = project_path(args.mtcnn_dir)
    if args.out_dir:
        out_root = project_path(args.out_dir)
    else:
        out_root = project_path("data", "visualizations", "face_compare", f"{'_'.join(split_names)}_{splits_dir.name}")
    ensure_dir(out_root)

    exported = 0
    selected_ids = set(args.ids)
    sources_md: list[str] = []

    for split_name in split_names:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"找不到 CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if selected_ids:
            df = df[df["id"].astype(int).isin(selected_ids)].copy()

        count = 0
        for idx, row in df.reset_index(drop=True).iterrows():
            sid = int(row["id"])
            cls = int(row["class"]) if "class" in row else -1
            raw_path = project_path(str(row["source_path"]))
            haar_path = project_path(str(row["image_path"]))
            mtcnn_path = _resolve_processed_split_dir(mtcnn_dir, split_name) / f"{sid}.png"

            raw_rgb = _try_load_image(_load_raw_rgb, raw_path)
            haar_rgb = _try_load_image(_load_png_rgb, haar_path)
            mtcnn_rgb = _try_load_image(_load_png_rgb, mtcnn_path)

            triptych = _compose_triptych(
                split_name=split_name,
                sid=sid,
                cls=cls,
                raw_rgb=raw_rgb,
                haar_rgb=haar_rgb,
                mtcnn_rgb=mtcnn_rgb,
            )
            stem = f"{split_name}_{idx:03d}_id{sid:03d}_class{cls}"
            out_path = out_root / f"{stem}.png"
            Image.fromarray(triptych).save(out_path, format="PNG", compress_level=3)
            count += 1
            exported += 1
        sources_md.append(f"- `{csv_path.relative_to(project_path())}` → 导出 {count} 张")

    readme = out_root / "README.md"
    readme.write_text(
        "# Face Comparison\n\n"
        + "\n".join(sources_md)
        + "\n\n"
        + "- 每张图包含 `raw / HAAR / MTCNN` 三列\n"
        + "- `raw` 为原始 `.npy`，已按比赛数据实际存储格式从 BGR 转成 RGB\n"
        + "- `HAAR` 使用当前主线 `exp_001_faces_224`\n"
        + "- `MTCNN` 使用现有 `exp_008_mtcnn_faces_112`，可直接肉眼比较裁剪稳定性\n",
        encoding="utf-8",
    )

    print(f"完成: 导出 {exported} 张对比图 -> {out_root}")


if __name__ == "__main__":
    main()
