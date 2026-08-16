"""Train and evaluate a Segformer model on the Massachusetts Buildings Dataset
using the transformers Trainer API.

Dataset: https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset
Reuses the dataset defined in dataset_expirement.py (same train/val/test splits).
"""

import os

# must be set before CUDA initializes to reduce allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, Trainer, TrainingArguments

from dataset_expirement import (
    CLASS_NAMES,
    DATA_ROOT,
    IoUMeter,
    MassachusettsBuildingsDataset,
    NUM_CLASSES,
)

MODEL_CHECKPOINT = "nvidia/mit-b0"
CROP_SIZE = 384
BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 6e-5
OUTPUT_DIR = "segformer-checkpoints"


class SegformerDatasetAdapter(Dataset):
    """Wraps MassachusettsBuildingsDataset to return the dict format Trainer expects."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        return {"pixel_values": image, "labels": mask}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    # Segformer outputs logits at 1/4 resolution; upsample to label size before comparing
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    preds = logits.argmax(dim=1)

    iou_meter = IoUMeter(NUM_CLASSES)
    iou_meter.update(preds, labels)
    per_class_iou = iou_meter.per_class_iou()

    metrics = {"mean_iou": iou_meter.mean_iou()}
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        metrics[f"{name}_iou"] = iou.item()
    return metrics


metadata_csv = DATA_ROOT / "metadata.csv"

train_ds = SegformerDatasetAdapter(
    MassachusettsBuildingsDataset(metadata_csv, DATA_ROOT, "train", CROP_SIZE, train=True)
)
val_ds = SegformerDatasetAdapter(
    MassachusettsBuildingsDataset(metadata_csv, DATA_ROOT, "val", CROP_SIZE, train=False)
)
test_ds = SegformerDatasetAdapter(
    MassachusettsBuildingsDataset(metadata_csv, DATA_ROOT, "test", CROP_SIZE, train=False)
)

print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} samples")

id2label = dict(enumerate(CLASS_NAMES))
label2id = {name: idx for idx, name in id2label.items()}
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nTest metrics:")
print(trainer.evaluate(test_ds, metric_key_prefix="test"))


