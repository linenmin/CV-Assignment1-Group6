"""LVFace 预训练 backbone 加载（源码位于 third_party/LVFace，由上游 MIT 许可）。

权重文件 (*.pt) 需自行从 HuggingFace 下载，见 configs 中 ``pretrained_checkpoint_path`` 注释。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from dl_pipeline.common.paths import project_path


def lvface_vendor_root() -> Path:
    """本工程内 LVFace 上游仓库根目录（git clone 目标路径）。"""
    return project_path("third_party", "LVFace")


def resolve_pretrained_checkpoint(path: str | Path) -> Path:
    """将配置中的路径解析为绝对路径；相对路径相对于工程根目录。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_path(str(p))


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        out[new_key] = value
    return out


def load_lvface_backbone(network_name: str, checkpoint_path: Path) -> nn.Module:
    """构建 LVFace ``get_model(network_name)`` 并加载 ``.pt`` 权重。

    Parameters
    ----------
    network_name:
        与 LVFace 仓库 ``backbones/__init__.py`` 中 ``get_model`` 名称一致，
        例如 LVFace-B_Glint360K 对应 ``vit_b_dp005_mask_005``（见上游 README eval 命令）。
    checkpoint_path:
        本地 ``.pt`` 文件路径（通常为纯 ``state_dict``）。
    """
    root = lvface_vendor_root()
    if not (root / "backbones" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"未找到 LVFace 源码: {root}。请在工程根目录执行：\n"
            "  git clone https://github.com/bytedance/LVFace.git third_party/LVFace"
        )

    root_str = str(root.resolve())
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        from backbones import get_model  # type: ignore[import-untyped]

        net = get_model(network_name, dropout=0, fp16=False)
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass

    resolved = Path(checkpoint_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"LVFace 权重文件不存在: {resolved}")

    raw = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
    if not isinstance(state_dict, dict):
        raise ValueError(f"无法从检查点解析 state_dict: {resolved}")

    state_dict = _strip_module_prefix(state_dict)  # type: ignore[arg-type]
    net.load_state_dict(state_dict, strict=True)
    # third_party get_model 对 ViT-B 默认 using_checkpoint=True；与部分解冻/ ArcFace 联训时，
    # torch.utils.checkpoint 在输入链上无 requires_grad 时会导致骨干梯度为 None（见训练日志警告）。
    # 关闭 checkpoint 以略增显存换取可靠反传。
    if hasattr(net, "using_checkpoint"):
        net.using_checkpoint = False
    return net
