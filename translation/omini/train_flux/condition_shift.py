"""Small unknown translations of training condition images."""

import random
import re

import torch
from torch.utils.data import Dataset


_CONDITION_KEY = re.compile(r"condition_\d+")


def translate_with_edge_padding(image, dx, dy):
    """Translate a CHW image; positive offsets move content right and down."""
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise ValueError("Condition image must be a CHW tensor.")
    height, width = image.shape[-2:]
    if not -width < dx < width or not -height < dy < height:
        raise ValueError("Translation must be smaller than the image dimensions.")
    rows = (torch.arange(height, device=image.device) - dy).clamp(0, height - 1)
    columns = (torch.arange(width, device=image.device) - dx).clamp(0, width - 1)
    return image.index_select(-2, rows).index_select(-1, columns)


class RandomConditionShiftDataset(Dataset):
    """Shift each input condition independently while leaving the target fixed.

    Position metadata is left at zero on purpose: the model is not told the
    unknown registration error. The wrapped dataset still handles paired flips
    and condition dropout before this augmentation runs.
    """

    def __init__(self, dataset, max_shift_pixels):
        self.dataset = dataset
        self.max_shift_pixels = int(max_shift_pixels)
        if self.max_shift_pixels < 1:
            raise ValueError("max_shift_pixels must be positive.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        condition_keys = sorted(key for key in sample if _CONDITION_KEY.fullmatch(key))
        if not condition_keys:
            raise KeyError("Dataset sample has no condition image.")
        for key in condition_keys:
            image = sample[key]
            if min(image.shape[-2:]) <= self.max_shift_pixels:
                raise ValueError("max_shift_pixels must be smaller than the image.")
            dx = dy = 0
            while dx == dy == 0:
                dx = random.randint(-self.max_shift_pixels, self.max_shift_pixels)
                dy = random.randint(-self.max_shift_pixels, self.max_shift_pixels)
            sample[key] = translate_with_edge_padding(image, dx, dy)
        return sample
