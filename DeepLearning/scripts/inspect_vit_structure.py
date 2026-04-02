"""检查 ViT AdaFace backbone 的内部结构，用于确认 stage-wise unfreeze 的可行性。"""
import sys
sys.path.insert(0, "src")

from dl_pipeline.models.cvlface import load_cvlface_backbone

model = load_cvlface_backbone("minchul/cvlface_adaface_vit_base_webface4m")

print("=== top-level ===")
print(type(model))
print(dir(model))

print("\n=== model.model ===")
net = model.model
print(type(net))

print("\n=== net.net (if exists) ===")
try:
    inner = net.net
    print(type(inner))
    print("named_children:", [name for name, _ in inner.named_children()])
except AttributeError:
    print("no net.net")

print("\n=== all named_modules depth 1-2 ===")
for name, mod in model.named_modules():
    depth = name.count(".")
    if depth <= 2 and name:
        print(f"  {name}: {type(mod).__name__}")

print("\n=== parameter groups ===")
total = 0
for name, p in model.named_parameters():
    total += p.numel()
print(f"Total parameters: {total:,}")

# 尝试找 transformer blocks
print("\n=== looking for transformer blocks ===")
for name, mod in model.named_modules():
    mod_type = type(mod).__name__
    if "block" in mod_type.lower() or "layer" in mod_type.lower() or "encoder" in mod_type.lower():
        depth = name.count(".")
        if depth <= 3:
            print(f"  {name}: {mod_type}")
