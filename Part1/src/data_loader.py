"""
data_loader.py
--------------
Loads the KUL Computer Vision GA-1 dataset from disk.

Usage
-----
from data_loader import load_data
train, test = load_data()
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA_FOLDER = 'kul-computer-vision-ga-1-2026'


def _resolve_data_dir(data_folder: str = DATA_FOLDER) -> Path:
    """Locate the dataset directory from common local and Kaggle layouts."""
    requested = Path(data_folder).expanduser()
    if requested.is_dir():
        return requested.resolve()

    if requested.is_absolute():
        raise FileNotFoundError(f"Cannot find dataset directory: {requested}")

    env_candidates = [
        os.environ.get('KUL_CV_GA1_DATA_DIR'),
        os.environ.get('CV_GA1_DATA_DIR'),
    ]

    roots = [
        Path('.'),
        Path('..'),
        Path('../..'),
        Path('../DeepLearning/data/raw'),
        Path('../../DeepLearning/data/raw'),
        Path('/kaggle/input'),
    ]

    for env_path in env_candidates:
        if env_path:
            roots.append(Path(env_path).expanduser())

    # When running on Kaggle, the dataset may be mounted at /kaggle/input/<dataset-slug>/...
    kaggle_root = Path('/kaggle/input')
    if kaggle_root.is_dir():
        roots.extend(path for path in kaggle_root.iterdir() if path.is_dir())

    checked = []
    for root in roots:
        for candidate in (root, root / data_folder):
            checked.append(str(candidate))
            if candidate.is_dir() and (candidate / 'train_set.csv').is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find '{data_folder}'. Checked common local/Kaggle locations: {checked}"
    )


def load_data(data_folder: str = DATA_FOLDER) -> tuple:
    """Locate the dataset directory and load train/test DataFrames.

    Parameters
    ----------
    data_folder : str
        Dataset directory name or an explicit path to the dataset root.

    Returns
    -------
    train : pd.DataFrame
        Columns: name, class, img (RGB ndarray).
    test : pd.DataFrame
        Columns: img (RGB ndarray).
    """
    data_dir = _resolve_data_dir(data_folder)
    print(f"Data directory: {data_dir}")

    train = pd.read_csv(data_dir / 'train_set.csv', index_col=0)
    train.index = train.index.rename('id')

    test = pd.read_csv(data_dir / 'test_set.csv', index_col=0)
    test.index = test.index.rename('id')

    train['img'] = [
        cv2.cvtColor(
            np.load(data_dir / 'train' / f'train_{idx}.npy', allow_pickle=False),
            cv2.COLOR_BGR2RGB,
        )
        for idx in train.index
    ]

    test['img'] = [
        cv2.cvtColor(
            np.load(data_dir / 'test' / f'test_{idx}.npy', allow_pickle=False),
            cv2.COLOR_BGR2RGB,
        )
        for idx in test.index
    ]

    print(f"Training set: {len(train)} examples | Test set: {len(test)} examples")
    return train, test
