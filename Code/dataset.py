"""BraTS-2020 loader.

Expected layout (the official MICCAI_BraTS2020_TrainingData folder):
    <root>/BraTS20_Training_001/BraTS20_Training_001_{flair,t1,t1ce,t2,seg}.nii.gz
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("flair", "t1", "t1ce", "t2")


def labels_to_regions(seg):
    """BraTS labels {1: NCR/NET, 2: ED, 4: ET} -> stacked WT / TC / ET masks."""
    wt = np.isin(seg, [1, 2, 4])
    tc = np.isin(seg, [1, 4])
    et = seg == 4
    return np.stack([wt, tc, et]).astype(np.float32)


def zscore(volume):
    brain = volume > 0
    if brain.any():
        mean, std = volume[brain].mean(), volume[brain].std() + 1e-8
        volume = np.where(brain, (volume - mean) / std, 0)
    return volume.astype(np.float32)


def center_crop(arr, size):
    """Crop the last three axes around the volume centre to `size` (D, H, W)."""
    slices = [slice(None)] * (arr.ndim - 3)
    for dim, s in zip(arr.shape[-3:], size):
        start = max((dim - s) // 2, 0)
        slices.append(slice(start, start + s))
    return arr[tuple(slices)]


class BraTSDataset(Dataset):
    def __init__(self, root, cases=None, crop=(128, 128, 128), augment=False):
        root = Path(root)
        self.cases = cases or sorted(p for p in root.iterdir() if p.is_dir())
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def _load(self, case, suffix):
        return nib.load(str(case / f"{case.name}_{suffix}.nii.gz")).get_fdata()

    def __getitem__(self, idx):
        case = Path(self.cases[idx])
        image = np.stack([zscore(self._load(case, m)) for m in MODALITIES])
        mask = labels_to_regions(self._load(case, "seg"))
        # nibabel returns (H, W, D); move depth first -> (C, D, H, W)
        image, mask = image.transpose(0, 3, 1, 2), mask.transpose(0, 3, 1, 2)
        image, mask = center_crop(image, self.crop), center_crop(mask, self.crop)
        if self.augment:
            for axis in (1, 2, 3):
                if np.random.rand() < 0.5:
                    image, mask = np.flip(image, axis), np.flip(mask, axis)
            image = image * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy())
