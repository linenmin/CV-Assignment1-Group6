"""统计 ViT 各 block 参数量。"""
import sys
sys.path.insert(0, "src")
from dl_pipeline.models.cvlface import load_cvlface_backbone

m = load_cvlface_backbone("minchul/cvlface_adaface_vit_base_webface4m")
net = m.model.net

for i, block in enumerate(net.blocks):
    n = sum(p.numel() for p in block.parameters())
    print(f"block[{i:2d}]: {n:>10,} params")

print(f"norm:      {sum(p.numel() for p in net.norm.parameters()):>10,} params")
print(f"feature:   {sum(p.numel() for p in net.feature.parameters()):>10,} params")
print(f"patch_emb: {sum(p.numel() for p in net.patch_embed.parameters()):>10,} params")

total = sum(p.numel() for p in m.parameters())
for n_blocks in [1, 2, 3, 4]:
    trainable = sum(sum(p.numel() for p in net.blocks[24 - n_blocks + j].parameters()) for j in range(n_blocks))
    trainable += sum(p.numel() for p in net.norm.parameters())
    trainable += sum(p.numel() for p in net.feature.parameters())
    print(f"last-{n_blocks} + norm + feat: {trainable:>10,} ({trainable / total * 100:.1f}%)")

print(f"total: {total:,}")
