"""诊断 exp_042 graph refine 没有触发的原因：看 score 分布和邻域结构。"""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from pathlib import Path

# 加载 exp_032 的 embedding cache (如果存在)
# 或者直接从 fusion cache 加载
vit_cache = Path("outputs/exp_034_ir101_vit_score_mean_fixed055/fusion_cache/vit_frozen_embeddings.pt")
ir101_cache = Path("outputs/exp_034_ir101_vit_score_mean_fixed055/fusion_cache/ir101_embeddings.pt")

if vit_cache.exists():
    vit_data = torch.load(vit_cache, map_location="cpu", weights_only=True)
    print("=== ViT Embedding Cache ===")
    for k, v in vit_data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    gallery_emb = vit_data.get("gallery_embeddings") or vit_data.get("train_embeddings")
    gallery_lab = vit_data.get("gallery_labels") or vit_data.get("train_labels")
    test_emb = vit_data.get("test_embeddings")

    if gallery_emb is not None and test_emb is not None:
        import torch.nn.functional as F
        gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
        test_emb = F.normalize(test_emb, p=2, dim=1)

        # 计算 prototype
        prototypes = {}
        for label in [1, 2]:
            mask = gallery_lab == label
            prototypes[label] = F.normalize(gallery_emb[mask].mean(dim=0, keepdim=True), p=2, dim=1)

        # 计算每个 test sample 到最近 prototype 的 similarity
        sims = []
        for label in [1, 2]:
            sim = (test_emb @ prototypes[label].T).squeeze(1)
            sims.append(sim)
        sim_matrix = torch.stack(sims, dim=1)  # [N, 2]
        best_sims, best_labels = sim_matrix.max(dim=1)
        best_labels = best_labels + 1  # 1 or 2

        scores = best_sims.numpy()
        print(f"\n=== Test Set Score Distribution (ViT, N={len(scores)}) ===")
        print(f"  min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")

        # 分布直方图
        bins = [0.0, 0.30, 0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80, 1.0]
        counts, _ = np.histogram(scores, bins=bins)
        print(f"\n  Score histogram:")
        for i in range(len(counts)):
            bar = "#" * min(counts[i], 80)
            print(f"  [{bins[i]:.3f}, {bins[i+1]:.3f}): {counts[i]:5d}  {bar}")

        # 在 margin band 内的样本数
        threshold = 0.55
        for delta in [0.05, 0.10, 0.15, 0.20]:
            in_band = ((scores >= threshold - delta) & (scores < threshold + delta)).sum()
            print(f"\n  |sim - 0.55| < {delta}: {in_band} samples ({in_band/len(scores)*100:.1f}%)")

        # 邻域分析
        print(f"\n=== Neighborhood Analysis ===")
        print(f"  Gallery size: {gallery_emb.shape[0]}")
        print(f"  Test size: {test_emb.shape[0]}")

        # 对 margin band 内的样本，看 top-7 邻居中有多少是 gallery
        all_emb = torch.cat([gallery_emb, test_emb], dim=0)
        all_labels = torch.cat([gallery_lab, torch.full((test_emb.shape[0],), -1)])

        margin_mask = (scores >= threshold - 0.05) & (scores < threshold + 0.05)
        margin_indices = np.where(margin_mask)[0]
        print(f"  Samples in margin band (delta=0.05): {len(margin_indices)}")

        if len(margin_indices) > 0:
            # 对前 20 个 margin 样本做邻域分析
            gallery_votes_counts = []
            for idx in margin_indices[:50]:
                test_idx = idx
                query = test_emb[test_idx:test_idx+1]
                sims_to_all = (query @ all_emb.T).squeeze(0)
                # 排除自己 (test_idx + gallery_size)
                self_idx = gallery_emb.shape[0] + test_idx
                sims_to_all[self_idx] = -1.0
                topk_vals, topk_idxs = sims_to_all.topk(7)
                gallery_count = (topk_idxs < gallery_emb.shape[0]).sum().item()
                gallery_votes_counts.append(gallery_count)

            gallery_votes_counts = np.array(gallery_votes_counts)
            print(f"  Gallery neighbors in top-7 for margin samples (first {len(gallery_votes_counts)}):")
            for n in range(8):
                c = (gallery_votes_counts == n).sum()
                if c > 0:
                    print(f"    {n} gallery neighbors: {c} samples")
else:
    print("No embedding cache found. Run exp_034 first to generate cached embeddings.")
