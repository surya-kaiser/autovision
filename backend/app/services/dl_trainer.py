"""
Deep Learning Trainer — PyTorch-based training for image classification & segmentation.

Models:
  Classification: SimpleCNN, ResNet18 (transfer learning)
  Segmentation:   U-Net, DeepLabV3 (ResNet50 backbone)

All models stream per-epoch logs via asyncio queue to the frontend.
"""
import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import ModelType, TaskType
from app.utils.logger import get_logger

logger = get_logger(__name__)

IMG_SIZE = 224

# Colour palette for segmentation overlay visualisation
SEG_PALETTE = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
    (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
]


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

def _build_simple_cnn(num_classes: int):
    """3-block CNN for image classification."""
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2), nn.Dropout2d(0.25),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(256, n_cls),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SimpleCNN(num_classes)


def _build_resnet(num_classes: int):
    """ResNet18 with ImageNet pre-trained backbone — transfer learning."""
    from torchvision import models as tv_models
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    import torch.nn as nn
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_unet(num_classes: int):
    """Standard U-Net for semantic segmentation."""
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self, inc, outc):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(inc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True),
                nn.Conv2d(outc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True),
            )

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.enc1 = _Block(3, 64)
            self.enc2 = _Block(64, 128)
            self.enc3 = _Block(128, 256)
            self.enc4 = _Block(256, 512)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _Block(512, 1024)
            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = _Block(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = _Block(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = _Block(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = _Block(128, 64)
            self.final = nn.Conv2d(64, n_cls, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.final(d1)

    return UNet(num_classes)


def _build_deeplabv3(num_classes: int):
    """DeepLabV3 with ResNet-50 backbone (pretrained on ImageNet)."""
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models import ResNet50_Weights
    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def build_classification_model(model_type: ModelType, num_classes: int):
    if model_type == ModelType.RESNET:
        return _build_resnet(num_classes)
    return _build_simple_cnn(num_classes)   # CNN or fallback


def build_segmentation_model(model_type: ModelType, num_classes: int):
    if model_type == ModelType.DEEPLABV3:
        return _build_deeplabv3(num_classes)
    return _build_unet(num_classes)          # UNET or fallback


# ═══════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════

def _get_classification_transforms(train: bool):
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _make_classification_dataset(items: List[Dict], train: bool):
    """items: [{"path": str, "class_idx": int}]"""
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    class _DS(Dataset):
        def __init__(self, items, tfm):
            self.items = items
            self.tfm = tfm

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            img = Image.open(it["path"]).convert("RGB")
            return self.tfm(img), it["class_idx"]

    return _DS(items, _get_classification_transforms(train))


def _make_segmentation_dataset(items: List[Dict], train: bool):
    """items: [{"image": str, "mask": str}]"""
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    class _DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            img = Image.open(it["image"]).convert("RGB")
            mask = Image.open(it["mask"]).convert("L")

            # Keep image/mask spatially aligned for segmentation augmentation.
            if train:
                if np.random.rand() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                if np.random.rand() < 0.3:
                    angle = float(np.random.choice([90, 180, 270]))
                    img = img.rotate(angle, resample=Image.BILINEAR)
                    mask = mask.rotate(angle, resample=Image.NEAREST)

            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            mask_t = torch.tensor(np.array(mask), dtype=torch.long)
            return img, mask_t

    return _DS(items)


# ═══════════════════════════════════════════════════════════════════════════
# Segmentation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_iou(pred, target, num_classes: int) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        pc = pred == c
        tc = target == c
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 0.0


def compute_dice(pred, target, num_classes: int) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    dices = []
    for c in range(num_classes):
        pc = pred == c
        tc = target == c
        inter = (pc & tc).sum().float()
        total = pc.sum().float() + tc.sum().float()
        if total > 0:
            dices.append((2 * inter / total).item())
    return sum(dices) / len(dices) if dices else 0.0


def compute_pixel_accuracy(pred, target) -> float:
    return (pred == target).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Training — Classification
# ═══════════════════════════════════════════════════════════════════════════

def train_dl_classification(
    model_type: ModelType,
    session_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hyperparams: Dict[str, Any],
    log_queue: "asyncio.Queue[str]",
) -> Tuple[Optional[str], Dict[str, float]]:
    """Train a PyTorch image classifier. Returns (checkpoint_path, metrics)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    def log(msg: str):
        log_queue.put_nowait(msg)

    device = _detect_device()
    log(f"Device: {device}")

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest_path = settings.UPLOAD_DIR / session_id / "preprocessed" / "split_manifest.json"
    if not manifest_path.exists():
        log("ERROR: split_manifest.json not found. Preprocess dataset first.")
        return None, {}

    manifest = json.loads(manifest_path.read_text())
    classes = manifest.get("classes", [])
    num_classes = len(classes)
    if num_classes == 0:
        log("ERROR: No classes found in manifest")
        return None, {}

    class_to_idx = {c: i for i, c in enumerate(classes)}
    log(f"Classes ({num_classes}): {classes}")

    def make_items(split):
        return [
            {"path": e["path"], "class_idx": class_to_idx.get(e["class"], 0)}
            for e in manifest["splits"].get(split, [])
        ]

    train_items = make_items("train")
    val_items = make_items("val")
    if not train_items:
        log("ERROR: No training samples found")
        return None, {}

    train_ds = _make_classification_dataset(train_items, train=True)
    val_ds = _make_classification_dataset(val_items, train=False) if val_items else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0) if val_ds else None
    log(f"Train: {len(train_ds)} samples | Val: {len(val_ds) if val_ds else 0} samples")

    # ── Model / Optimiser ────────────────────────────────────────────────────
    model = build_classification_model(model_type, num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model: {model_type.value} | Parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=hyperparams.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}
    ckpt_dir = settings.MODEL_DIR / session_id / model_type.value
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        if val_loader:
            model.eval()
            vl, vc, vt = 0.0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    vl += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    vt += labels.size(0)
                    vc += predicted.eq(labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_loss = vl / vt if vt else 0.0
            val_acc = vc / vt if vt else 0.0
            from sklearn.metrics import f1_score
            val_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
            scheduler.step(val_loss)

        log(f"  [Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {"accuracy": val_acc, "f1": val_f1, "loss": val_loss}
            torch.save({
                "model_arch": model_type.value,
                "num_classes": num_classes,
                "classes": classes,
                "img_size": IMG_SIZE,
                "state_dict": model.state_dict(),
            }, best_path)
            log(f"    -> New best model saved (val_acc={val_acc:.4f})")

    log(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return str(best_path), best_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Training — Segmentation
# ═══════════════════════════════════════════════════════════════════════════

def train_dl_segmentation(
    model_type: ModelType,
    session_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hyperparams: Dict[str, Any],
    log_queue: "asyncio.Queue[str]",
) -> Tuple[Optional[str], Dict[str, float]]:
    """Train a segmentation model (U-Net / DeepLabV3). Returns (checkpoint_path, metrics)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    def log(msg: str):
        log_queue.put_nowait(msg)

    device = _detect_device()
    log(f"Device: {device}")

    # ── Load segmentation manifest ───────────────────────────────────────────
    manifest_path = settings.UPLOAD_DIR / session_id / "preprocessed" / "seg_manifest.json"
    if not manifest_path.exists():
        log("ERROR: seg_manifest.json not found. Dataset needs images/ and masks/ directories.")
        return None, {}

    manifest = json.loads(manifest_path.read_text())
    num_classes = manifest.get("num_classes", 2)
    class_names = manifest.get("class_names", [])

    train_items = manifest["splits"].get("train", [])
    val_items = manifest["splits"].get("val", [])
    if not train_items:
        log("ERROR: No training samples found in seg_manifest")
        return None, {}

    log(f"Segmentation — {num_classes} classes | Train: {len(train_items)} | Val: {len(val_items)}")

    train_ds = _make_segmentation_dataset(train_items, train=True)
    val_ds = _make_segmentation_dataset(val_items, train=False) if val_items else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0) if val_ds else None

    # ── Model / Optimiser ────────────────────────────────────────────────────
    model = build_segmentation_model(model_type, num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model: {model_type.value} | Parameters: {param_count:,}")

    is_binary = num_classes <= 2
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_iou = 0.0
    best_metrics: Dict[str, float] = {}
    ckpt_dir = settings.MODEL_DIR / session_id / model_type.value
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):        # DeepLabV3 returns OrderedDict
                outputs = outputs["out"]

            if is_binary:
                target_bin = (masks > 0).float()
                if outputs.shape[1] >= 2:
                    logits = outputs[:, 1, :, :] - outputs[:, 0, :, :]
                else:
                    logits = outputs[:, 0, :, :]
                loss = criterion_bce(logits, target_bin)
            else:
                loss = criterion_ce(outputs, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches if n_batches else 0.0

        # Validation
        val_iou, val_dice, val_pix = 0.0, 0.0, 0.0
        if val_loader:
            model.eval()
            ious, dices, pixs = [], [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs["out"]

                    if is_binary:
                        if outputs.shape[1] >= 2:
                            logits = outputs[:, 1, :, :] - outputs[:, 0, :, :]
                        else:
                            logits = outputs[:, 0, :, :]
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).long()
                        masks_eval = (masks > 0).long()
                        eval_classes = 2
                    else:
                        preds = outputs.argmax(dim=1)
                        masks_eval = masks
                        eval_classes = num_classes

                    ious.append(compute_iou(preds, masks_eval, eval_classes))
                    dices.append(compute_dice(preds, masks_eval, eval_classes))
                    pixs.append(compute_pixel_accuracy(preds, masks_eval))
            val_iou = sum(ious) / len(ious) if ious else 0.0
            val_dice = sum(dices) / len(dices) if dices else 0.0
            val_pix = sum(pixs) / len(pixs) if pixs else 0.0
            scheduler.step(1 - val_iou)

        log(f"  [Epoch {epoch}/{epochs}] loss={train_loss:.4f}"
            f" | IoU={val_iou:.4f} Dice={val_dice:.4f} PixAcc={val_pix:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_metrics = {
                "iou": val_iou, "dice": val_dice,
                "pixel_accuracy": val_pix, "loss": train_loss,
            }
            torch.save({
                "model_arch": model_type.value,
                "num_classes": num_classes,
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "state_dict": model.state_dict(),
            }, best_path)
            log(f"    -> New best model saved (IoU={val_iou:.4f})")

    log(f"Training complete. Best IoU: {best_iou:.4f}")
    return str(best_path), best_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Inference helpers (used by the /inference route)
# ═══════════════════════════════════════════════════════════════════════════

def load_dl_checkpoint(session_id: str, model_type: str) -> Optional[Dict]:
    """Load a .pt checkpoint dict, or None if not found."""
    pt = settings.MODEL_DIR / session_id / model_type / "best.pt"
    if not pt.exists():
        return None
    import torch
    return torch.load(pt, map_location="cpu", weights_only=False)


def predict_classification(ckpt: Dict, image_bytes: bytes) -> Dict[str, Any]:
    """Run classification on a single image. Returns {label, confidence, class_probabilities}."""
    import torch
    from PIL import Image
    import io

    model_arch = ckpt["model_arch"]
    num_classes = ckpt["num_classes"]
    classes = ckpt.get("classes", [str(i) for i in range(num_classes)])

    mt = ModelType(model_arch)
    model = build_classification_model(mt, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfm = _get_classification_transforms(train=False)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

    label = classes[idx.item()] if idx.item() < len(classes) else str(idx.item())
    class_probs = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
    return {"label": label, "confidence": round(float(conf), 4), "class_probabilities": class_probs}


def predict_segmentation(ckpt: Dict, image_bytes: bytes) -> Dict[str, Any]:
    """Run segmentation on a single image. Returns {segmented_image (base64), label}."""
    import torch
    import base64
    import io
    from PIL import Image

    model_arch = ckpt["model_arch"]
    num_classes = ckpt["num_classes"]

    mt = ModelType(model_arch)
    model = build_segmentation_model(mt, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    from torchvision import transforms
    img_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = img_tfm(orig).unsqueeze(0)

    with torch.no_grad():
        out = model(inp)
        if isinstance(out, dict):
            out = out["out"]
        pred = out.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    # Create colour overlay
    overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        colour = SEG_PALETTE[c % len(SEG_PALETTE)]
        overlay[pred == c] = colour

    # Blend with resized original
    orig_resized = orig.resize((IMG_SIZE, IMG_SIZE))
    orig_arr = np.array(orig_resized)
    blended = (0.5 * orig_arr + 0.5 * overlay).astype(np.uint8)

    result_img = Image.fromarray(blended)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    unique_classes = int(len(np.unique(pred)))
    return {
        "segmented_image": f"data:image/png;base64,{b64}",
        "label": f"{unique_classes} class(es) segmented",
    }
