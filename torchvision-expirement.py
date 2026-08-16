import os
import time
from pathlib import Path

# must be set before CUDA initializes to reduce allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from dataset_expirement import (
    DATA_ROOT,
    NUM_CLASSES,
    CLASS_NAMES,
    MassachusettsBuildingsDataset,
    IoUMeter,
    IGNORE_INDEX,
)




def compute_loss(outputs, targets, aux_weight):
    # ignore_index excludes unlabeled pixels, enabling partial cross-entropy on sparsified train masks
    loss = F.cross_entropy(outputs["out"], targets, ignore_index=IGNORE_INDEX)
    if "aux" in outputs:
        loss = loss + aux_weight * F.cross_entropy(
            outputs["aux"], targets, ignore_index=IGNORE_INDEX
        )
    return loss


def train_one_epoch(
    model, loader, optimizer, device, aux_weight, scaler, accum_steps=1
):
    model.train()
    running_loss = 0.0
    use_amp = device.type == "cuda"
    optimizer.zero_grad()
    num_batches = 0
    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = compute_loss(outputs, masks, aux_weight) / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * accum_steps * images.size(0)
        num_batches = step + 1
    if num_batches % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(models, loader, device, aux_weight):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        model.eval()

    running_loss = 0.0
    iou_meter = IoUMeter(NUM_CLASSES)
    use_amp = device.type == "cuda"

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        probs_sum = None
        with torch.amp.autocast("cuda", enabled=use_amp):
            for model in models:
                outputs = model(images)
                probs = torch.softmax(outputs["out"], dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            # Ensemble follows probability-level averaging across members.
            mean_probs = probs_sum / len(models)
            loss = F.nll_loss(
                mean_probs.clamp_min(1e-8).log(),
                masks,
                ignore_index=IGNORE_INDEX,
            )

        running_loss += loss.item() * images.size(0)
        preds = mean_probs.argmax(dim=1)
        iou_meter.update(preds.cpu(), masks.cpu())

    return running_loss / len(loader.dataset), iou_meter


def log_iou(prefix, loss, iou_meter):
    per_class = iou_meter.per_class_iou()
    class_str = " | ".join(
        f"{name}_IoU={iou:.4f}" for name, iou in zip(CLASS_NAMES, per_class)
    )
    print(f"{prefix} loss={loss:.4f} | mIoU={iou_meter.mean_iou():.4f} | {class_str}")


NUM_EPOCHS = 30
CROP_SIZE = 384
BATCH_SIZE = 2
NUM_WORKERS = 4
CHECKPOINT1 = Path("checkpoints/best_model1.pt")
CHECKPOINT2 = Path("checkpoints/best_model2.pt")
PARTIAL_LABEL_FRACTION = 0.1  # fraction of pixels supervised per training image (partial cross-entropy); 1.0 = full supervision


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_csv = DATA_ROOT / "metadata.csv"

    train_ds = MassachusettsBuildingsDataset(
        metadata_csv,
        DATA_ROOT,
        "train",
        CROP_SIZE,
        train=True,
        partial_label_fraction=PARTIAL_LABEL_FRACTION,
    )
    val_ds = MassachusettsBuildingsDataset(
        metadata_csv, DATA_ROOT, "val", CROP_SIZE, train=False
    )
    test_ds = MassachusettsBuildingsDataset(
        metadata_csv, DATA_ROOT, "test", CROP_SIZE, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )

    print(
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} samples | device={device} | "
        f"partial_label_fraction={PARTIAL_LABEL_FRACTION}"
    )

    model1 = deeplabv3_resnet101(
        weights=None,
        weights_backbone=ResNet101_Weights.IMAGENET1K_V2,
        num_classes=NUM_CLASSES,
        aux_loss=True,
    ).to(device)
    model2 = fcn_resnet101(
        weights=None,
        weights_backbone=ResNet101_Weights.IMAGENET1K_V2,
        num_classes=NUM_CLASSES,
        aux_loss=True,
    ).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=1e-4)
    
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    CHECKPOINT1.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT2.parent.mkdir(parents=True, exist_ok=True)
    best_val_iou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss1 = train_one_epoch(
            model1, train_loader, optimizer1, device, 0.4, scaler, NUM_CLASSES
        )
        train_loss2 = train_one_epoch(
            model2, train_loader, optimizer2, device, 0.4, scaler, NUM_CLASSES
        )
        val_loss, val_iou_meter1 = evaluate([model1, model2], val_loader, device, 0.4)

        elapsed = time.time() - start
        print(
            f"\nEpoch {epoch:03d}/{NUM_EPOCHS} ({elapsed:.1f}s) | train_loss1={train_loss1:.4f} | train_loss2={train_loss2:.4f}"
        )
        log_iou("  val model", val_loss, val_iou_meter1)

        val_iou = val_iou_meter1.mean_iou()
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model1.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                CHECKPOINT1,
            )
            torch.save(
                {
                    "model_state_dict": model2.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                CHECKPOINT2,
            )
            print(f"  -> saved new best checkpoint (val_mIoU={val_iou:.4f})")

    print(f"\nTraining complete. Best val mIoU={best_val_iou:.4f}")

    checkpoint1 = torch.load(CHECKPOINT1, map_location=device)
    model1.load_state_dict(checkpoint1["model_state_dict"])
    checkpoint2 = torch.load(CHECKPOINT2, map_location=device)
    model2.load_state_dict(checkpoint2["model_state_dict"])
    test_loss, test_iou_meter = evaluate([model1, model2], test_loader, device, 0.4)
    log_iou("\nTest", test_loss, test_iou_meter)


if __name__ == "__main__":
    main()


