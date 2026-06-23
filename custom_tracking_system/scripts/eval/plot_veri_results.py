"""Parse train_veri.log and save metrics/plots for the OSNet VeRi-776 fine-tune run."""
import re
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = Path(__file__).resolve().parents[2] / 'train_veri.log'
OUT_DIR = Path(__file__).resolve().parents[3] / 'docs' / 'assets' / 'osnet_veri776_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_RE = re.compile(
    r'^Epoch (\d+)/(\d+)\s+loss=([\d.]+)\s+train_acc=([\d.]+)%\s+val_acc=([\d.]+)%\s+time=([\d.]+)m\s+lr=([\d.eE+-]+)'
)

rows = []
for line in LOG_PATH.read_text(encoding='utf-8', errors='ignore').splitlines():
    m = EPOCH_RE.match(line)
    if m:
        epoch, total, loss, train_acc, val_acc, time_m, lr = m.groups()
        rows.append({
            'epoch': int(epoch), 'total_epochs': int(total), 'loss': float(loss),
            'train_acc': float(train_acc), 'val_acc': float(val_acc),
            'time_min': float(time_m), 'lr': float(lr),
        })

if not rows:
    raise SystemExit('No epoch summary lines found in train_veri.log')

with open(OUT_DIR / 'results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

epochs = [r['epoch'] for r in rows]
loss = [r['loss'] for r in rows]
train_acc = [r['train_acc'] for r in rows]
val_acc = [r['val_acc'] for r in rows]
best = max(rows, key=lambda r: r['val_acc'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(epochs, loss, color='tab:red')
ax1.set_title('Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(alpha=0.3)

ax2.plot(epochs, train_acc, label='train_acc', color='tab:blue')
ax2.plot(epochs, val_acc, label='val_acc', color='tab:green')
ax2.axvline(best['epoch'], color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f"Accuracy (best val_acc={best['val_acc']:.2f}% @ epoch {best['epoch']})")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle('OSNet x1.0 fine-tune on VeRi-776 (osnet_veri776_v2)')
fig.tight_layout()
fig.savefig(OUT_DIR / 'results.png', dpi=150)

print(f"Saved {OUT_DIR / 'results.csv'} and {OUT_DIR / 'results.png'}")
print(f"Best val_acc={best['val_acc']:.2f}% at epoch {best['epoch']}/{rows[-1]['total_epochs']}")
