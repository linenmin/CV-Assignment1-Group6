from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict import _collect_embeddings, _load_model_from_checkpoint
from dl_pipeline.common.config import load_experiment_config
from dl_pipeline.common.paths import ensure_dir, project_path
from dl_pipeline.data.datamodule import FaceDataModule
from dl_pipeline.evaluation.case_diagnostics import prepare_test_audit_index, rank_class_neighbors
from dl_pipeline.inference.prototype import (
    combine_embedding_sets,
    compute_class_prototypes,
    compute_similarity_matrix,
    neighborhood_aware_predictions,
)


PANEL_W = 1500
PANEL_H = 980
HEADER_H = 84
GAP = 16
RAW_W = 760
TEXT = (25, 25, 25)
BG = (248, 248, 248)


def _load_raw_rgb(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), allow_pickle=False)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _load_png_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
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


def _render_case_canvas(
    row: pd.Series,
    raw_rgb: np.ndarray | None,
    selected_rgb: np.ndarray | None,
    class1_neighbors: list[dict[str, object]],
    class2_neighbors: list[dict[str, object]],
    gallery_lookup: pd.DataFrame,
) -> np.ndarray:
    canvas = np.full((PANEL_H, PANEL_W, 3), BG, dtype=np.uint8)
    header = (
        f"id={int(row['id'])} | pred={int(row['pred'])} | source_093={row['selection_source_093']} "
        f"| source_088={row['selection_source_088']}"
    )
    _draw_header(canvas, header)

    raw_h = PANEL_H - HEADER_H - GAP * 2
    raw_panel = _make_missing_panel("raw\nmissing", RAW_W, raw_h) if raw_rgb is None else _fit_rgb(raw_rgb, RAW_W, raw_h)
    canvas[HEADER_H + GAP:HEADER_H + GAP + raw_h, GAP:GAP + RAW_W] = raw_panel

    right_x = GAP * 2 + RAW_W
    right_w = PANEL_W - right_x - GAP
    selected_panel = _make_missing_panel("SEL\nmissing", right_w, 210) if selected_rgb is None else _fit_rgb(selected_rgb, right_w, 210)
    cv2.putText(canvas, "SEL", (right_x, HEADER_H + GAP + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
    canvas[HEADER_H + GAP + 28:HEADER_H + GAP + 28 + 210, right_x:right_x + right_w] = selected_panel

    text_y = HEADER_H + GAP + 270
    lines = [
        f"final={float(row['final_score']):.4f}  base={float(row['base_score']):.4f}  neighbor={float(row['neighbor_score']):.4f}",
        f"cls1={float(row['score_class1']):.4f}  cls2={float(row['score_class2']):.4f}  margin={float(row['class_margin']):.4f}",
        f"verifier_j={float(row['selected_score_jesse_093']):.4f}  verifier_m={float(row['selected_score_mila_093']):.4f}",
        f"excess_j={float(row['selected_excess_jesse_093']):.4f}  excess_m={float(row['selected_excess_mila_093']):.4f}",
    ]
    for line in lines:
        cv2.putText(canvas, line, (right_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT, 1, cv2.LINE_AA)
        text_y += 28

    def draw_neighbor_strip(title: str, neighbors: list[dict[str, object]], y0: int) -> None:
        cv2.putText(canvas, title, (right_x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT, 2, cv2.LINE_AA)
        thumb_w = 180
        thumb_h = 170
        x = right_x
        for neighbor in neighbors:
            gid = int(neighbor["id"])
            grow = gallery_lookup.loc[gid]
            image_rgb = _load_png_rgb(project_path(str(grow["image_path"])))
            panel = _make_missing_panel(str(gid), thumb_w, thumb_h) if image_rgb is None else _fit_rgb(image_rgb, thumb_w, thumb_h)
            canvas[y0 + 12:y0 + 12 + thumb_h, x:x + thumb_w] = panel
            label = f"id={gid} sim={float(neighbor['similarity']):.3f}"
            cv2.putText(canvas, label, (x, y0 + 12 + thumb_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT, 1, cv2.LINE_AA)
            x += thumb_w + 12

    draw_neighbor_strip("Nearest class1", class1_neighbors, text_y + 10)
    draw_neighbor_strip("Nearest class2", class2_neighbors, text_y + 250)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export embedding-neighbor diagnostics for selected test cases.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_093_vit_adaface_multiface_selected_pad18_rescue_neighborhoodaware_fixed055_hfliptta.yaml",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        default=[48, 289, 361, 543, 612, 699, 1333, 1525],
    )
    parser.add_argument(
        "--out-dir",
        default="data/visualizations/case_embedding_diagnostics/exp_093_focus_cases",
    )
    parser.add_argument("--neighbors-per-class", type=int, default=3)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    splits_dir = project_path(config["data"]["splits_dir"])
    processed_dir = project_path(config["data"]["processed_dir"])
    output_root = ensure_dir(project_path(args.out_dir))
    images_root = ensure_dir(output_root / "images")
    for old_png in images_root.glob("*.png"):
        old_png.unlink()

    metrics_path = project_path("outputs", config["inference"]["checkpoint_source_experiment"], "metrics.json")
    checkpoint_path = json.loads(metrics_path.read_text(encoding="utf-8"))["best_model_path"]

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    gallery_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True).set_index("id")
    test_df_indexed = test_df.set_index("id")

    datamodule = FaceDataModule(
        train_csv=splits_dir / "train.csv",
        val_csv=splits_dir / "val.csv",
        test_csv=splits_dir / "test.csv",
        image_size=config["data"]["face_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        normalization=config["data"]["normalization"],
    )
    datamodule.setup()

    model = _load_model_from_checkpoint(config, checkpoint_path)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 and config["train"]["accelerator"] != "cpu" else "cpu"
    model = model.to(device)

    use_horizontal_flip_tta = config.get("inference", {}).get("tta_horizontal_flip", False)
    train_embeddings, train_labels = _collect_embeddings(model, datamodule.train_dataloader(), device, use_horizontal_flip_tta)
    val_embeddings, val_labels = _collect_embeddings(model, datamodule.val_dataloader(), device, use_horizontal_flip_tta)
    test_embeddings, test_ids = _collect_embeddings(model, datamodule.predict_dataloader(), device, use_horizontal_flip_tta)

    gallery_embeddings, gallery_labels = combine_embedding_sets([(train_embeddings, train_labels), (val_embeddings, val_labels)])
    gallery_ids = np.concatenate([train_df["id"].to_numpy(dtype=np.int64), val_df["id"].to_numpy(dtype=np.int64)])

    prototype_labels = config["inference"].get("prototype_labels", [1, 2])
    other_label = int(config["inference"].get("other_label", 0))
    threshold = float(config["inference"].get("threshold", 0.55))
    top_k = int(config["inference"]["neighborhood_aware"].get("top_k", 15))
    base_weight = float(config["inference"]["neighborhood_aware"].get("base_weight", 0.5))

    prototypes = compute_class_prototypes(gallery_embeddings, gallery_labels, prototype_labels=prototype_labels)
    test_predictions, test_base_scores, test_neighbor_scores, test_final_scores = neighborhood_aware_predictions(
        query_embeddings=test_embeddings,
        prototypes=prototypes,
        other_label=other_label,
        threshold=threshold,
        top_k=top_k,
        base_weight=base_weight,
    )
    sim_matrix, sim_labels = compute_similarity_matrix(test_embeddings, prototypes, prototype_labels=list(prototype_labels))
    score_class1 = sim_matrix[:, sim_labels.index(1)] if 1 in sim_labels else np.zeros(len(test_ids), dtype=np.float32)
    score_class2 = sim_matrix[:, sim_labels.index(2)] if 2 in sim_labels else np.zeros(len(test_ids), dtype=np.float32)
    top2_scores, _ = sim_matrix.topk(k=min(2, sim_matrix.shape[1]), dim=1)
    class_margin = top2_scores[:, 0] - top2_scores[:, 1] if top2_scores.shape[1] > 1 else np.ones(len(test_ids), dtype=np.float32)

    pred_df = pd.DataFrame(
        {
            "id": test_ids.detach().cpu().numpy().astype(int),
            "pred": test_predictions.detach().cpu().numpy().astype(int),
            "base_score": test_base_scores.detach().cpu().numpy(),
            "neighbor_score": test_neighbor_scores.detach().cpu().numpy(),
            "final_score": test_final_scores.detach().cpu().numpy(),
            "score_class1": score_class1.detach().cpu().numpy() if hasattr(score_class1, "detach") else score_class1,
            "score_class2": score_class2.detach().cpu().numpy() if hasattr(score_class2, "detach") else score_class2,
            "class_margin": class_margin.detach().cpu().numpy() if hasattr(class_margin, "detach") else class_margin,
        }
    ).set_index("id")

    audit_093 = prepare_test_audit_index(pd.read_csv(processed_dir / "selection_audit.csv"))
    audit_088 = prepare_test_audit_index(
        pd.read_csv(project_path("data", "processed", "exp_088_multiface_selected_faces_112", "selection_audit.csv"))
    )

    summary_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []

    gallery_emb_np = gallery_embeddings.detach().cpu().numpy()
    gallery_labels_np = gallery_labels.detach().cpu().numpy()
    test_emb_np = test_embeddings.detach().cpu().numpy()
    test_id_list = test_ids.detach().cpu().numpy().astype(int).tolist()
    test_index_map = {sid: idx for idx, sid in enumerate(test_id_list)}

    for sid in [int(v) for v in args.ids]:
        if sid not in test_index_map:
            continue
        idx = test_index_map[sid]
        row = {
            "id": sid,
            **pred_df.loc[sid].to_dict(),
            "selection_source_093": audit_093.loc[sid, "selection_source"],
            "selected_score_jesse_093": audit_093.loc[sid, "selected_score_jesse"],
            "selected_score_mila_093": audit_093.loc[sid, "selected_score_mila"],
            "selected_excess_jesse_093": audit_093.loc[sid, "selected_excess_jesse"],
            "selected_excess_mila_093": audit_093.loc[sid, "selected_excess_mila"],
            "selection_source_088": audit_088.loc[sid, "selection_source"],
        }
        summary_rows.append(row)

        query_embedding = test_emb_np[idx]
        class1_neighbors = rank_class_neighbors(query_embedding, gallery_emb_np, gallery_labels_np, gallery_ids, target_label=1, top_k=args.neighbors_per_class)
        class2_neighbors = rank_class_neighbors(query_embedding, gallery_emb_np, gallery_labels_np, gallery_ids, target_label=2, top_k=args.neighbors_per_class)
        for target_name, neighbors in (("class1", class1_neighbors), ("class2", class2_neighbors)):
            for neighbor in neighbors:
                neighbor_rows.append(
                    {
                        "case_id": sid,
                        "target_class": target_name,
                        "rank": int(neighbor["rank"]),
                        "neighbor_id": int(neighbor["id"]),
                        "similarity": float(neighbor["similarity"]),
                        "neighbor_image_path": str(gallery_df.loc[int(neighbor["id"]), "image_path"]),
                    }
                )

        raw_rgb = _load_raw_rgb(project_path(str(test_df_indexed.loc[sid, "source_path"])))
        selected_rgb = _load_png_rgb(project_path(str(test_df_indexed.loc[sid, "image_path"])))
        canvas = _render_case_canvas(
            pd.Series(row),
            raw_rgb,
            selected_rgb,
            class1_neighbors,
            class2_neighbors,
            gallery_df,
        )
        out_path = images_root / f"id{sid:04d}_pred{int(row['pred'])}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    summary_df = pd.DataFrame(summary_rows).sort_values("id").reset_index(drop=True)
    neighbors_df = pd.DataFrame(neighbor_rows).sort_values(["case_id", "target_class", "rank"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    neighbors_df.to_csv(output_root / "neighbors.csv", index=False)
    (output_root / "README.md").write_text(
        "# case embedding diagnostics\n\n"
        "- model: exp_088 checkpoint reused on exp_093 selected inputs\n"
        "- cases: user-selected suspicious test ids\n"
        "- outputs: summary.csv, neighbors.csv, images/\n",
        encoding="utf-8",
    )
    print(f"done: exported diagnostics for {len(summary_df)} cases -> {output_root}")


if __name__ == "__main__":
    main()
