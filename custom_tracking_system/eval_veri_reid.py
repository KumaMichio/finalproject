"""
Standard VeRi-776 Re-ID evaluation (mAP / CMC) for a trained OSNet checkpoint.

Protocol: market1501-style — for each query, gallery images sharing both the
same vehicle ID and the same camera ID are discarded before ranking (they are
near-duplicate frames of the same instance, not a meaningful retrieval test).
pid/camid are parsed directly from VeRi filenames: {pid}_{camid}_{ts}_{junk}.jpg

Usage:
    python eval_veri_reid.py --data data/datasets/VeRi --weights weights/osnet_veri776_v2.pth
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

_reid_repo = Path(__file__).parent.parent / 'deep-person-reid'
if _reid_repo.exists() and str(_reid_repo) not in sys.path:
    sys.path.insert(0, str(_reid_repo))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchreid.reid.metrics.rank import evaluate_rank

NAME_RE = re.compile(r'^(\d+)_c(\d+)_')


def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class VeRiEvalSet(Dataset):
    def __init__(self, image_dir: Path, transform):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = sorted(self.image_dir.glob('*.jpg'))
        self.pids, self.camids = [], []
        for p in self.paths:
            m = NAME_RE.match(p.name)
            self.pids.append(int(m.group(1)))
            self.camids.append(int(m.group(2)))
        self.pids = np.array(self.pids)
        self.camids = np.array(self.camids)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), idx


@torch.no_grad()
def extract_features(model, loader, device, num_samples):
    feats = None
    for imgs, idxs in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = torch.nn.functional.normalize(out, dim=1).cpu()
        if feats is None:
            feats = torch.zeros(num_samples, out.shape[1])
        feats[idxs] = out
    return feats.numpy()


def main():
    parser = argparse.ArgumentParser(description='Evaluate OSNet on VeRi-776 test set (mAP/CMC)')
    parser.add_argument('--data', required=True, help='Path to VeRi dataset root')
    parser.add_argument('--weights', required=True, help='Path to trained checkpoint (.pth)')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default=None, help='Path to save JSON results (optional)')
    args = parser.parse_args()

    data_root = Path(args.data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = build_transform()
    query_set = VeRiEvalSet(data_root / 'image_query', transform)
    gallery_set = VeRiEvalSet(data_root / 'image_test', transform)
    print(f"[Data] query={len(query_set)}  gallery={len(gallery_set)}")

    query_loader = DataLoader(query_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    num_classes = ckpt['num_classes']

    import torchreid
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f"[Model] loaded {args.weights}  (epoch {ckpt.get('epoch', '?')}, "
          f"train val_acc={ckpt.get('val_acc', 0)*100:.2f}%)")

    t0 = time.perf_counter()
    q_feats = extract_features(model, query_loader, device, len(query_set))
    g_feats = extract_features(model, gallery_loader, device, len(gallery_set))
    print(f"[Extract] {len(query_set) + len(gallery_set)} feats in {time.perf_counter()-t0:.1f}s")

    distmat = 1.0 - q_feats @ g_feats.T  # cosine distance, feats are L2-normalized

    cmc, mAP = evaluate_rank(
        distmat, query_set.pids, gallery_set.pids,
        query_set.camids, gallery_set.camids,
        max_rank=50, use_metric_cuhk03=False, use_cython=False,
    )

    print(f"\n[Result] mAP={mAP*100:.2f}%")
    for r in (1, 5, 10, 20):
        print(f"  CMC@{r:<3d} = {cmc[r-1]*100:.2f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'weights': str(args.weights),
            'num_query': len(query_set),
            'num_gallery': len(gallery_set),
            'mAP': float(mAP),
            'cmc_full': [float(x) for x in cmc],
            'cmc': {str(r): float(cmc[r-1]) for r in (1, 5, 10, 20, 50)},
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[Saved] {out_path}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc * 100, marker='o', markersize=3)
        ax.set_title(f'VeRi-776 test CMC curve (mAP={mAP*100:.2f}%)')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Matching rate (%)')
        ax.grid(alpha=0.3)
        ax.set_ylim(min(cmc) * 100 - 2, 100.5)
        fig.tight_layout()
        plot_path = out_path.with_name('cmc_curve.png')
        fig.savefig(plot_path, dpi=150)
        print(f"[Saved] {plot_path}")


if __name__ == '__main__':
    main()
