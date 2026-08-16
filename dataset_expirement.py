import numpy as np
import torchvision.transforms.v2.functional as TF
from PIL import Image
import torch
from torch.utils.data import Dataset
import csv
import random
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data"
NUM_CLASSES = 2  # background, building
CLASS_NAMES = ["background", "building"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IGNORE_INDEX = 255  # sentinel for pixels excluded from the loss (partial cross-entropy)
class MassachusettsBuildingsDataset(Dataset):
    """Loads image/mask pairs for a given split from metadata.csv."""

    def __init__(self, metadata_csv, data_root, split, crop_size=512, train=False, partial_label_fraction=1.0):
        self.data_root = Path(data_root)
        self.crop_size = crop_size
        self.train = train
        self.partial_label_fraction = partial_label_fraction
        with open(metadata_csv, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No entries found for split={split!r} in {metadata_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.data_root / row["png_image_path"]).convert("RGB")
        label = Image.open(self.data_root / row["png_label_path"]).convert("L")
        image, label = self._transform(image, label)

        image = TF.to_dtype(TF.to_image(image), dtype=torch.float32, scale=True)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask = (torch.from_numpy(np.array(label, dtype=np.uint8)) > 127).long()
        if self.train and self.partial_label_fraction < 1.0:
            mask = self._sparsify(mask)
        return image, mask

    def _sparsify(self, mask):
        # re-sampled every call, so each epoch trains on a fresh random subset of pixels
        keep = torch.rand(mask.shape) < self.partial_label_fraction
        sparse_mask = torch.full_like(mask, IGNORE_INDEX)
        sparse_mask[keep] = mask[keep]
        return sparse_mask

    def _transform(self, image, label):
        size = self.crop_size
        # guard against tiles smaller than the crop size
        if min(image.size) < size:
            image = TF.resize(image, [size, size])
            label = TF.resize(label, [size, size], interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            w, h = image.size
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            image = TF.crop(image, y, x, size, size)
            label = TF.crop(label, y, x, size, size)
            if random.random() < 0.5:
                image, label = TF.hflip(image), TF.hflip(label)
            if random.random() < 0.5:
                image, label = TF.vflip(image), TF.vflip(label)
        else:
            image = TF.center_crop(image, [size, size])
            label = TF.center_crop(label, [size, size])

        return image, label


class IoUMeter:
    """Accumulates per-class intersection/union to compute IoU over a full split."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, targets):
        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = targets == c
            self.intersection[c] += (pred_c & target_c).sum()
            self.union[c] += (pred_c | target_c).sum()

    def per_class_iou(self):
        return self.intersection / self.union.clamp(min=1e-6)

    def mean_iou(self):
        return self.per_class_iou().mean().item()
