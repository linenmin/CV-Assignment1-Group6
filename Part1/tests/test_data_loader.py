from pathlib import Path

from data_loader import _resolve_data_dir


def test_resolve_data_dir_accepts_explicit_dataset_root(tmp_path):
    dataset_root = tmp_path / "kul-computer-vision-ga-1-2026"
    dataset_root.mkdir()
    (dataset_root / "train_set.csv").write_text("id,name,class\n", encoding="utf-8")

    assert _resolve_data_dir(str(dataset_root)) == dataset_root.resolve()


def test_resolve_data_dir_finds_dataset_under_given_parent(tmp_path, monkeypatch):
    part1_root = tmp_path / "Part1"
    part1_root.mkdir()
    parent = tmp_path / "DeepLearning" / "data" / "raw"
    dataset_root = parent / "kul-computer-vision-ga-1-2026"
    dataset_root.mkdir(parents=True)
    (dataset_root / "train_set.csv").write_text("id,name,class\n", encoding="utf-8")

    monkeypatch.chdir(part1_root)
    assert _resolve_data_dir() == dataset_root.resolve()
