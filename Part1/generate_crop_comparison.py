"""Generate clean HAAR vs Multi-face crop comparison figures for Section 4.7."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DL_ROOT = Path(r"d:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group\DeepLearning")
OUT_DIR = Path(r"d:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group\Part1\figures")
OUT_DIR.mkdir(exist_ok=True)

SAMPLES = [
    {
        "id": 18,
        "split": "train",
        "label": "Jesse (class 1)",
        "caption": "Image contains both Jesse and Michael.\n"
                   "HAAR selects Michael; multi-face pipeline\n"
                   "scores all faces and selects Jesse (J = 0.843).",
    },
    {
        "id": 34,
        "split": "train",
        "label": "Jesse (class 1)",
        "caption": "Image contains 3 detected faces.\n"
                   "HAAR selects a wrong face; multi-face pipeline\n"
                   "selects the correct Jesse face (J = 0.721).",
    },
]


def load_raw(sample_id: int, split: str) -> np.ndarray:
    npy = DL_ROOT / "data" / "raw" / "kul-computer-vision-ga-1-2026" / split / f"{split}_{sample_id}.npy"
    arr = np.load(str(npy), allow_pickle=False)
    # Raw npy files are stored in BGR order
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_crop(sample_id: int, split: str, experiment: str, is_rgb_stored: bool = False) -> np.ndarray:
    """Load a crop image. is_rgb_stored=True when the file was saved in RGB order."""
    path = DL_ROOT / "data" / "processed" / experiment / split / f"{sample_id}.png"
    img = cv2.imread(str(path))
    if is_rgb_stored:
        # File was saved as RGB, imread reads it as-is -> already in RGB for matplotlib
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_comparison(samples: list[dict], output_path: Path) -> None:
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4.5 * n),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        sid = sample["id"]
        split = sample["split"]

        raw = load_raw(sid, split)
        haar_crop = load_crop(sid, split, "exp_001_faces_224")
        mf_crop = load_crop(sid, split, "exp_088_multiface_selected_faces_112", is_rgb_stored=True)
        # Upscale multi-face crop to match HAAR crop display size
        mf_crop = cv2.resize(mf_crop, (haar_crop.shape[1], haar_crop.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)

        # Original image
        axes[row, 0].imshow(raw)
        axes[row, 0].set_title(f"Original (id={sid}, {sample['label']})",
                               fontsize=11, fontweight="bold", pad=8)
        axes[row, 0].axis("off")

        # HAAR crop
        axes[row, 1].imshow(haar_crop)
        axes[row, 1].set_title("HAAR Crop (baseline)",
                               fontsize=11, fontweight="bold", color="#c0392b", pad=8)
        # Red border
        for spine in axes[row, 1].spines.values():
            spine.set_edgecolor("#c0392b")
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[row, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Multi-face selected crop
        axes[row, 2].imshow(mf_crop)
        axes[row, 2].set_title("Multi-face Selected",
                               fontsize=11, fontweight="bold", color="#27ae60", pad=8)
        # Green border
        for spine in axes[row, 2].spines.values():
            spine.set_edgecolor("#27ae60")
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[row, 2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Caption below the row
        axes[row, 1].set_xlabel(sample["caption"], fontsize=9, style="italic",
                                color="#444444", labelpad=10)

    plt.tight_layout(h_pad=4.0)
    plt.savefig(str(output_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate_comparison(SAMPLES, OUT_DIR / "crop_comparison_haar_vs_multiface.png")
