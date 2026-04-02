"""深度诊断：look-alike margin 分布 + IR101 vs ViT 分歧分析。"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

# 加载 ViT embeddings
d = torch.load(
    "outputs/exp_034_ir101_vit_score_mean_fixed055/fusion_cache/vit_frozen_embeddings.pt",
    map_location="cpu", weights_only=True,
)
gallery_emb = F.normalize(d["train_embeddings"], p=2, dim=1)
gallery_lab = d["train_labels"]
test_emb = F.normalize(d["test_embeddings"], p=2, dim=1)

# IR101 embeddings
d2 = torch.load(
    "outputs/exp_034_ir101_vit_score_mean_fixed055/fusion_cache/ir101_embeddings.pt",
    map_location="cpu", weights_only=True,
)
ir101_gallery_emb = F.normalize(d2["train_embeddings"], p=2, dim=1)
ir101_gallery_lab = d2["train_labels"]
ir101_test_emb = F.normalize(d2["test_embeddings"], p=2, dim=1)

# ViT prototypes
proto = {}
for label in [1, 2]:
    mask = gallery_lab == label
    proto[label] = F.normalize(gallery_emb[mask].mean(dim=0, keepdim=True), p=2, dim=1)

# other sub-prototypes (michael / sarah)
other_mask = gallery_lab == 0
other_emb = gallery_emb[other_mask]
sims_to_jesse = (other_emb @ proto[1].T).squeeze()
michael_mask = sims_to_jesse > sims_to_jesse.median()
sarah_mask = ~michael_mask
michael_proto = F.normalize(other_emb[michael_mask].mean(dim=0, keepdim=True), p=2, dim=1)
sarah_proto = F.normalize(other_emb[sarah_mask].mean(dim=0, keepdim=True), p=2, dim=1)

# 4 个 similarity
sim_jesse = (test_emb @ proto[1].T).squeeze().numpy()
sim_mila = (test_emb @ proto[2].T).squeeze().numpy()
sim_michael = (test_emb @ michael_proto.T).squeeze().numpy()
sim_sarah = (test_emb @ sarah_proto.T).squeeze().numpy()

best_target = np.maximum(sim_jesse, sim_mila)
best_target_label = np.where(sim_jesse > sim_mila, 1, 2)

accepted_mask = best_target >= 0.55
print(f"Accepted as target: {accepted_mask.sum()}")
print(f"Rejected as other: {(~accepted_mask).sum()}")

# Jesse-accepted: look-alike margin
jesse_accepted = accepted_mask & (best_target_label == 1)
mila_accepted = accepted_mask & (best_target_label == 2)
jesse_margins = sim_jesse[jesse_accepted] - sim_michael[jesse_accepted]
mila_margins = sim_mila[mila_accepted] - sim_sarah[mila_accepted]

print(f"\n=== Jesse-accepted ({jesse_accepted.sum()}) ===")
print(f"  sim_jesse - sim_michael: min={jesse_margins.min():.4f}, p10={np.percentile(jesse_margins, 10):.4f}, p25={np.percentile(jesse_margins, 25):.4f}, median={np.median(jesse_margins):.4f}")
for m in [0.10, 0.15, 0.20, 0.25, 0.30]:
    print(f"  margin < {m:.2f}: {(jesse_margins < m).sum()}")

print(f"\n=== Mila-accepted ({mila_accepted.sum()}) ===")
print(f"  sim_mila - sim_sarah: min={mila_margins.min():.4f}, p10={np.percentile(mila_margins, 10):.4f}, p25={np.percentile(mila_margins, 25):.4f}, median={np.median(mila_margins):.4f}")
for m in [0.10, 0.15, 0.20, 0.25, 0.30]:
    print(f"  margin < {m:.2f}: {(mila_margins < m).sum()}")

# Rejected: possible false rejects
rejected_mask = ~accepted_mask
rej_best = best_target[rejected_mask]
print(f"\n=== Rejected ({rejected_mask.sum()}) ===")
for t in [0.50, 0.45, 0.40, 0.35, 0.30]:
    print(f"  sim_to_best_target > {t:.2f}: {(rej_best > t).sum()}")

# IR101 vs ViT disagreement
ir101_proto = {}
for label in [1, 2]:
    mask = ir101_gallery_lab == label
    ir101_proto[label] = F.normalize(ir101_gallery_emb[mask].mean(dim=0, keepdim=True), p=2, dim=1)

ir101_sim_j = (ir101_test_emb @ ir101_proto[1].T).squeeze().numpy()
ir101_sim_m = (ir101_test_emb @ ir101_proto[2].T).squeeze().numpy()
ir101_best = np.maximum(ir101_sim_j, ir101_sim_m)
ir101_pred = np.where(ir101_best >= 0.55, np.where(ir101_sim_j > ir101_sim_m, 1, 2), 0)
vit_pred = np.where(best_target >= 0.55, best_target_label, 0)

disagree = ir101_pred != vit_pred
print(f"\n=== IR101 vs ViT Disagreement ===")
print(f"  Total: {disagree.sum()} / {len(disagree)}")
types = Counter()
for i in range(len(disagree)):
    if disagree[i]:
        types[f"IR101={ir101_pred[i]} ViT={vit_pred[i]}"] += 1
for k, v in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# sklearn LabelSpreading 快速试验
print("\n=== LabelSpreading Quick Test ===")
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics.pairwise import cosine_similarity

all_emb_np = torch.cat([gallery_emb, test_emb], dim=0).numpy()
affinity = cosine_similarity(all_emb_np)
# 把负值截断
affinity = np.clip(affinity, 0, None)

labels_init = np.full(len(all_emb_np), -1)
labels_init[:len(gallery_lab)] = gallery_lab.numpy()

for alpha in [0.1, 0.2, 0.3, 0.5, 0.8]:
    # sklearn LabelSpreading 不支持 precomputed，用 callable 传入 affinity
    model = LabelSpreading(kernel=lambda X, Y=None: affinity, alpha=alpha, max_iter=50)
    model.fit(all_emb_np, labels_init)
    test_preds = model.transduction_[len(gallery_lab):]
    changes = (test_preds != vit_pred).sum()
    change_types = Counter()
    for i in range(len(test_preds)):
        if test_preds[i] != vit_pred[i]:
            change_types[f"{vit_pred[i]}->{test_preds[i]}"] += 1
    print(f"  alpha={alpha}: {changes} changes from exp_032 | {dict(sorted(change_types.items(), key=lambda x:-x[1]))}")

# 也试 LabelPropagation
print("\n=== LabelPropagation Quick Test ===")
from sklearn.semi_supervised import LabelPropagation
for k in [7, 15, 30]:
    model = LabelPropagation(kernel="knn", n_neighbors=k, max_iter=100)
    model.fit(all_emb_np, labels_init)
    test_preds = model.transduction_[len(gallery_lab):]
    changes = (test_preds != vit_pred).sum()
    change_types = Counter()
    for i in range(len(test_preds)):
        if test_preds[i] != vit_pred[i]:
            change_types[f"{vit_pred[i]}->{test_preds[i]}"] += 1
    print(f"  k={k}: {changes} changes | {dict(sorted(change_types.items(), key=lambda x:-x[1]))}")
