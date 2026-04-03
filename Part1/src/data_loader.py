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
import cv2
import numpy as np
import pandas as pd

DATA_FOLDER = 'kul-computer-vision-ga-1-2026'


def load_data(data_folder: str = DATA_FOLDER) -> tuple:
    """Locate the dataset directory and load train/test DataFrames.

    Searches the current directory and up to two levels up, so the function
    works regardless of whether you launch the notebook from the project root
    or a subdirectory.

    Parameters
    ----------
    data_folder : str
        Name of the top-level dataset directory.

    Returns
    -------
    train : pd.DataFrame
        Columns: name, class, img (RGB ndarray).
    test : pd.DataFrame
        Columns: img (RGB ndarray).
    """
    data_dir = None
    for candidate in ['.', '..', '../..']:
        path = os.path.join(candidate, data_folder)
        if os.path.isdir(path):
            data_dir = os.path.abspath(path)
            break
    if data_dir is None:
        raise FileNotFoundError(
            f"Cannot find '{data_folder}'. Run from the project root or a subdirectory."
        )
    print(f"Data directory: {data_dir}")

    train = pd.read_csv(os.path.join(data_dir, 'train_set.csv'), index_col=0)
    train.index = train.index.rename('id')

    test = pd.read_csv(os.path.join(data_dir, 'test_set.csv'), index_col=0)
    test.index = test.index.rename('id')

    train['img'] = [
        cv2.cvtColor(
            np.load(os.path.join(data_dir, 'train', f'train_{idx}.npy'), allow_pickle=False),
            cv2.COLOR_BGR2RGB,
        )
        for idx in train.index
    ]

    test['img'] = [
        cv2.cvtColor(
            np.load(os.path.join(data_dir, 'test', f'test_{idx}.npy'), allow_pickle=False),
            cv2.COLOR_BGR2RGB,
        )
        for idx in test.index
    ]

    print(f"Training set: {len(train)} examples | Test set: {len(test)} examples")
    return train, test
