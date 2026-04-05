from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def compute_array_sha1(arr: np.ndarray) -> str:
    data = np.asarray(arr)
    return hashlib.sha1(data.tobytes()).hexdigest()


def build_consistent_hash_label_map(hash_label_pairs: Iterable[tuple[str, int]]) -> dict[str, int]:
    grouped: dict[str, set[int]] = {}
    for image_hash, label in hash_label_pairs:
        grouped.setdefault(str(image_hash), set()).add(int(label))
    return {image_hash: next(iter(labels)) for image_hash, labels in grouped.items() if len(labels) == 1}


def apply_exact_hash_overrides(
    predictions: dict[int, int],
    test_hashes: dict[int, str],
    hash_to_label: dict[str, int],
) -> tuple[dict[int, int], dict[int, dict[str, int | str]]]:
    updated = {int(k): int(v) for k, v in predictions.items()}
    overridden: dict[int, dict[str, int | str]] = {}
    for sid, image_hash in test_hashes.items():
        sid = int(sid)
        image_hash = str(image_hash)
        if image_hash not in hash_to_label or sid not in updated:
            continue
        target = int(hash_to_label[image_hash])
        if updated[sid] == target:
            continue
        overridden[sid] = {"from": int(updated[sid]), "to": target, "hash": image_hash}
        updated[sid] = target
    return updated, overridden
