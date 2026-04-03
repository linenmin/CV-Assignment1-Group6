"""
将 split CSV 中的图片导出为独立 PNG，便于逐张人工检查。

在仓库 DeepLearning 根目录执行：
  python scripts/export_train_split_visualization.py
      # 默认导出 train + val（共 80 张，exp_001）
  python scripts/export_train_split_visualization.py --split train
  python scripts/export_train_split_visualization.py --split val
  python scripts/export_train_split_visualization.py --split all --with-test
  python scripts/export_train_split_visualization.py --no-annotate --upscale 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.paths import ensure_dir, project_path

SPLIT_NAMES = frozenset({"train", "val", "test"})


def _parse_splits(arg: str, with_test: bool) -> list[str]:
    raw = arg.strip().lower()
    if raw == "all":
        out = ["train", "val"]
        if with_test:
            out.append("test")
        return out
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    for p in parts:
        if p not in SPLIT_NAMES:
            raise SystemExit(f"未知 split: {p!r}，允许: train, val, test, all")
    return parts


def _export_one_split(
    *,
    split_name: str,
    csv_path: Path,
    out_root: Path,
    annotate: bool,
    upscale: int,
) -> tuple[int, int, int]:
    """返回 (csv 行数, 成功, 失败)。"""
    df = pd.read_csv(csv_path)
    need = {"id", "image_path", "class"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{csv_path.name} 需含列 {need}，当前: {df.columns.tolist()}")

    ok, fail = 0, 0
    for idx, row in df.reset_index(drop=True).iterrows():
        sid = int(row["id"])
        cls = int(row["class"])
        rel = Path(str(row["image_path"]))
        src = project_path(rel) if not rel.is_absolute() else rel
        stem = f"{split_name}_{idx:03d}_id{sid:03d}_class{cls}"
        dst = out_root / f"{stem}.png"

        if not src.is_file():
            print(f"[skip] [{split_name}] 缺失文件 id={sid}: {src}")
            fail += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] [{split_name}] 无法解码 id={sid}: {src}")
            fail += 1
            continue

        h0, w0 = img.shape[:2]
        if upscale > 1:
            img = cv2.resize(
                img,
                (w0 * upscale, h0 * upscale),
                interpolation=cv2.INTER_CUBIC,
            )

        if annotate:
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.35, min(w, h) / 512.0)
            thick = max(1, int(round(scale * 2)))
            lines = [f"{split_name}", f"id={sid}", f"class={cls}", f"src {w0}x{h0}"]
            y0 = int(22 * scale) + thick
            line_h = int(22 * scale) + thick * 2
            pad = int(6 * scale)
            box_h = line_h * len(lines) + pad * 2
            box_w = min(w, int(240 * scale) + pad * 2)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            for i, line in enumerate(lines):
                y = y0 + i * line_h
                cv2.putText(
                    img,
                    line,
                    (pad, y),
                    font,
                    scale,
                    (255, 255, 255),
                    thick,
                    cv2.LINE_AA,
                )

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(dst, format="PNG", compress_level=3)
        except OSError as e:
            print(f"[skip] [{split_name}] 写入失败 id={sid}: {dst} ({e})")
            fail += 1
            continue
        ok += 1

    return len(df), ok, fail


def main() -> None:
    p = argparse.ArgumentParser(description="导出 split 中每张图为 PNG 供人工检查")
    p.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits/exp_001",
        help="含 train.csv / val.csv 的目录（相对 DeepLearning 根）",
    )
    p.add_argument(
        "--split",
        type=str,
        default="all",
        help="逗号分隔，如 train,val；all 表示 train+val；加 --with-test 时 all 含 test",
    )
    p.add_argument(
        "--with-test",
        action="store_true",
        help="与 --split all 联用：额外导出 test.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录（默认按 split 组合自动生成）",
    )
    p.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在图上左上角标注 split / id / class（默认开启）",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=1,
        metavar="N",
        help="保存前将图像放大 N 倍（整数>=1）；默认 1",
    )
    args = p.parse_args()
    if args.upscale < 1:
        raise SystemExit("--upscale 须为 >=1 的整数")

    split_list = _parse_splits(args.split, args.with_test)
    splits_dir = project_path(args.splits_dir)

    if args.out_dir:
        out_root = project_path(args.out_dir)
    else:
        sub = "_".join(split_list) + f"_{splits_dir.name}"
        out_root = project_path("data", "visualizations", "split_png", sub)
    ensure_dir(out_root)

    total_rows = 0
    total_ok = total_fail = 0
    sources_md: list[str] = []

    for split_name in split_list:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"找不到 CSV: {csv_path}")
        nrows, ok, fail = _export_one_split(
            split_name=split_name,
            csv_path=csv_path,
            out_root=out_root,
            annotate=args.annotate,
            upscale=args.upscale,
        )
        total_rows += nrows
        total_ok += ok
        total_fail += fail
        rel = csv_path.relative_to(project_path())
        sources_md.append(f"- `{rel}` → 成功 {ok} / {nrows} 行")

    print(f"完成: 共 {len(split_list)} 个 split，累计 {total_rows} 行 CSV，成功 {total_ok} 张，失败 {total_fail} 张 -> {out_root}")

    index_md = out_root / "README.md"
    index_md.write_text(
        "# 导出 PNG\n\n"
        + "\n".join(sources_md)
        + "\n\n"
        + f"- 文件名: `{{split}}_{{行序}}_idXXX_classC.png`（行序为该 split 的 CSV 顺序）\n"
        + f"- PNG 经 Pillow RGB 写出\n"
        + f"- 相册裂图时请确认网盘已同步本地，或 `--upscale 2` 重导\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
