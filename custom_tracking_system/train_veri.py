"""
Fine-tune OSNet on VeRi-776 for Vehicle Re-Identification.

Usage:
    python train_veri.py --data ../../VeRi --out weights/osnet_veri776.pth

Hardware target: GTX 1050Ti 4GB VRAM  →  batch=32, ~2–3 h for 20 epochs.

The trained weight file is used by DualReIDExtractor in modules/reid.py
to provide a vehicle-specific feature extractor alongside the existing
OSNet (Market-1501) for persons.
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# torchreid cloned as sibling directory — no pip install needed
_reid_repo = Path(__file__).parent.parent / 'deep-person-reid'
if _reid_repo.exists() and str(_reid_repo) not in sys.path:
    sys.path.insert(0, str(_reid_repo))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class VeRiDataset(Dataset):
    """
    Reads VeRi-776 from its original directory layout.

    image_train/ contains files named:
        {vehicleID}_{cameraID}_{timestamp}_{junk}.jpg

    train_label.xml (gb2312 encoding) provides the authoritative
    vehicleID → integer class mapping.
    """

    def __init__(self, image_dir: Path, label_xml: Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples   = []   # [(image_path, class_idx)]

        # ElementTree doesn't support multi-byte encodings (gb2312).
        # Read raw bytes → decode as gbk (superset of gb2312) → re-encode to utf-8.
        raw = label_xml.read_bytes()
        text = raw.decode('gbk', errors='replace')
        text = text.replace("encoding=\"gb2312\"", "encoding=\"utf-8\"", 1)
        root = ET.fromstring(text.encode('utf-8'))
        items = root.findall('Items/Item') or root.findall('Item')

        # Build continuous class index (0 … N-1) from vehicleID strings
        vehicle_ids = sorted({item.attrib['vehicleID'] for item in items})
        vid_to_idx  = {vid: idx for idx, vid in enumerate(vehicle_ids)}
        self.num_classes = len(vehicle_ids)

        for item in items:
            img_path = self.image_dir / item.attrib['imageName']
            if img_path.exists():
                self.samples.append((img_path, vid_to_idx[item.attrib['vehicleID']]))

        print(f"[Dataset] {len(self.samples)} images | {self.num_classes} vehicle IDs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------

def build_transforms(is_train: bool):
    """Standard ReID transforms — 256×128 input, same as OSNet training."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# -----------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    t0 = time.perf_counter()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)

        # torchreid models return (features, logits) in train mode
        logits = output[1] if isinstance(output, (tuple, list)) else output

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta_s   = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
            print(f"  Epoch {epoch:02d} [{batch_idx+1}/{len(loader)}]  "
                  f"loss={total_loss/(batch_idx+1):.4f}  "
                  f"acc={correct/total*100:.1f}%  "
                  f"ETA {eta_s/60:.0f}m", flush=True)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Classification accuracy on the test split (proxy metric for Re-ID)."""
    model.eval()
    correct = 0
    total   = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)
        logits = output[1] if isinstance(output, (tuple, list)) else output
        preds  = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fine-tune OSNet on VeRi-776')
    parser.add_argument('--data',    required=True,
                        help='Path to VeRi dataset root (contains image_train/, train_label.xml)')
    parser.add_argument('--out',     default='weights/osnet_veri776.pth',
                        help='Output path for trained weights')
    parser.add_argument('--epochs',  type=int, default=20)
    parser.add_argument('--batch',   type=int, default=32,
                        help='Batch size — 32 fits in 4 GB VRAM for OSNet')
    parser.add_argument('--lr',      type=float, default=0.0003)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume',  default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    data_root = Path(args.data)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Device selection — supports CUDA (NVIDIA), DirectML (AMD/Intel on Windows), CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print("[Config] AMD/Intel GPU via DirectML")
        except ImportError:
            device = torch.device('cpu')
            print("[Config] WARNING: No GPU found — training on CPU will be very slow (~40h)")

    print(f"[Config] device={device}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    # ---- Datasets ----
    # Load toàn bộ train set trước, sau đó split 90/10 để validation dùng
    # cùng class space với training (Re-ID là open-set — không dùng test set làm val).
    full_set = VeRiDataset(
        image_dir=data_root / 'image_train',
        label_xml=data_root / 'train_label.xml',
        transform=build_transforms(is_train=True),
    )
    num_classes = full_set.num_classes

    val_size   = max(1, int(len(full_set) * 0.1))
    train_size = len(full_set) - val_size
    train_set, val_set_raw = torch.utils.data.random_split(
        full_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Val set dùng transform không augment
    from torch.utils.data import Subset
    from copy import deepcopy
    val_full = deepcopy(full_set)
    val_full.transform = build_transforms(is_train=False)
    val_set = Subset(val_full, val_set_raw.indices)

    # pin_memory only works with CUDA (not DirectML or CPU)
    use_pin = isinstance(device, torch.device) and device.type == 'cuda'

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, pin_memory=use_pin,
    )

    # ---- Model ----
    try:
        import torchreid
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=num_classes,
            pretrained=True,   # start from ImageNet weights for faster convergence
        )
        print(f"[Model] OSNet x1.0  |  {num_classes} classes  |  pretrained=ImageNet")
    except ImportError:
        raise SystemExit(
            "torchreid not found.\n"
            "Install with:  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
        )

    model = model.to(device)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 1
    best_acc    = 0.0

    if args.resume:
        # Load to CPU first — DirectML device object is not a valid map_location
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('val_acc', 0.0)
        print(f"[Resume] epoch {start_epoch}  best_acc={best_acc:.3f}")

    # ---- Training loop ----
    print(f"\n[Train] Starting — {len(train_loader)} batches/epoch\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t_epoch
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
              f"val_acc={val_acc*100:.1f}%  "
              f"time={elapsed/60:.1f}m  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        # Save checkpoint every epoch
        ckpt = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_acc':   val_acc,
            'num_classes': num_classes,
        }
        ckpt_path = out_path.with_suffix(f'.ckpt_ep{epoch:02d}.pth')
        torch.save(ckpt, ckpt_path)

        # Save best as final output path
        if is_best:
            torch.save(ckpt, out_path)
            print(f"  ★ Best model saved → {out_path}  (val_acc={best_acc*100:.2f}%)")

    print(f"\n[Done] Best val_acc={best_acc*100:.2f}%  weights → {out_path}")


if __name__ == '__main__':
    main()
