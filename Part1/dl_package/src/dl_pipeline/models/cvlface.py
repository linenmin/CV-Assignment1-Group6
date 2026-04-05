from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


def _repo_cache_dir(repo_id: str) -> Path:
    safe_repo_name = repo_id.replace("/", "__")
    return Path.home() / ".cache" / "cvlface" / safe_repo_name


def _unload_cvlface_dynamic_modules() -> None:
    """切换不同 Hugging Face cvlface 仓库前卸载 wrapper/models，避免第二次 load 命中错误缓存。"""
    to_remove = [
        key
        for key in list(sys.modules.keys())
        if key == "wrapper" or key == "models" or key.startswith("models.")
    ]
    for key in to_remove:
        sys.modules.pop(key, None)


def load_cvlface_backbone(repo_id: str):
    from huggingface_hub import snapshot_download

    _unload_cvlface_dynamic_modules()
    cache_dir = _repo_cache_dir(repo_id)
    local_path = snapshot_download(repo_id=repo_id, local_dir=str(cache_dir))

    cwd = os.getcwd()
    sys.path.insert(0, local_path)
    os.chdir(local_path)
    try:
        wrapper = import_module("wrapper")
        config = wrapper.ModelConfig()
        model = wrapper.CVLFaceRecognitionModel(config)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model
