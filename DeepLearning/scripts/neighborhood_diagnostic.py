"""诊断：rejected 样本的邻域结构 — 能否用邻域信息区分 false reject 和 true other？"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
import numpy as np

# 加载 ViT embeddings
d = torch.load(
    "outputs/exp_034_ir101_vit_score_mean_fixed055/fusion_cache/vit_frozen_embeddings.pt",
    map_location="cpu", weights_only=True,
)
gallery_emb = F.normalize(d["train_embeddings"], p=2, dim=1)
gallery_lab = d["train_labels"]
test_emb = F.normalize(d["test_embeddings"], p=2, dim=1)
N = test_emb.shape[0]

# Prototypes
proto = {}
for label in [1, 2]:
    mask = gallery_lab == label
    proto[label] = F.normalize(gallery_emb[mask].mean(dim=0, keepdim=True), p=2, dim=1)

# 每个 test 的 base score 和 prediction
sim_1 = (test_emb @ proto[1].T).squeeze()
sim_2 = (test_emb @ proto[2].T).squeeze()
base_score = torch.maximum(sim_1, sim_2)
vit_pred = torch.where(base_score >= 0.55,
    torch.where(sim_1 > sim_2, torch.tensor(1), torch.tensor(2)),
    torch.tensor(0))

# 计算 test-to-test 相似度矩阵
print("Computing test-test similarity matrix...")
test_sim = (test_emb @ test_emb.T)
test_sim.fill_diagonal_(0)  # 排除自己

# 对每个 test，计算 top-15 邻居的平均 proto score
K = 15
topk_vals, topk_idxs = test_sim.topk(K, dim=1)
neighbor_base_scores = base_score[topk_idxs]  # [N, K]
neighbor_mean_score = neighbor_base_scores.mean(dim=1)  # [N]

# 关键诊断：看 neighbor_mean_score 如何区分 accepted vs rejected
accepted = vit_pred != 0
rejected = vit_pred == 0

print(f"\n=== Accepted ({accepted.sum()}) ===")
acc_ns = neighbor_mean_score[accepted].numpy()
print(f"  neighbor_mean_score: mean={acc_ns.mean():.4f}, std={acc_ns.std():.4f}")

print(f"\n=== Rejected ({rejected.sum()}) ===")
rej_ns = neighbor_mean_score[rejected].numpy()
rej_bs = base_score[rejected].numpy()
print(f"  neighbor_mean_score: mean={rej_ns.mean():.4f}, std={rej_ns.std():.4f}")

# 在 rejected 中，按 neighbor_mean_score 分层
print(f"\n=== Rejected: neighbor_mean_score distribution ===")
for t in [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]:
    count = (rej_ns > t).sum()
    print(f"  neighbor_mean_score > {t:.2f}: {count} samples")

# 关键：那些 base_score < 0.55 但 neighbor_mean_score > 0.55 的 — 潜在 false reject
potential_rescue = rejected.numpy() & (neighbor_mean_score.numpy() > 0.55)
print(f"\n=== Potential false rejects (rejected & neighbor_score > 0.55) ===")
print(f"  Count: {potential_rescue.sum()}")
if potential_rescue.sum() > 0:
    rescue_bs = base_score.numpy()[potential_rescue]
    rescue_ns = neighbor_mean_score.numpy()[potential_rescue]
    print(f"  base_score: min={rescue_bs.min():.4f}, max={rescue_bs.max():.4f}, mean={rescue_bs.mean():.4f}")
    print(f"  neighbor_score: min={rescue_ns.min():.4f}, max={rescue_ns.max():.4f}, mean={rescue_ns.mean():.4f}")

# Neighborhood-aware scoring: final = w * base + (1-w) * neighbor
print(f"\n=== Neighborhood-aware scoring sweep ===")
for w in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    final_score = w * base_score + (1 - w) * neighbor_mean_score
    new_pred = torch.where(final_score >= 0.55,
        torch.where(sim_1 > sim_2, torch.tensor(1), torch.tensor(2)),
        torch.tensor(0))
    changes = (new_pred != vit_pred).sum().item()
    # 分析变化方向
    from collections import Counter
    types = Counter()
    for i in range(N):
        if new_pred[i] != vit_pred[i]:
            types[f"{vit_pred[i].item()}->{new_pred[i].item()}"] += 1
    print(f"  w={w:.1f}: {changes} changes | {dict(sorted(types.items(), key=lambda x:-x[1]))}")

# 也试不同 threshold
print(f"\n=== Neighborhood-aware w=0.5 + threshold sweep ===")
final_05 = 0.5 * base_score + 0.5 * neighbor_mean_score
for th in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]:
    new_pred = torch.where(final_05 >= th,
        torch.where(sim_1 > sim_2, torch.tensor(1), torch.tensor(2)),
        torch.tensor(0))
    changes = (new_pred != vit_pred).sum().item()
    types = Counter()
    for i in range(N):
        if new_pred[i] != vit_pred[i]:
            types[f"{vit_pred[i].item()}->{new_pred[i].item()}"] += 1
    print(f"  th={th:.2f}: {changes} changes | {dict(sorted(types.items(), key=lambda x:-x[1]))}")
