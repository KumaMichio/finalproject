#!/usr/bin/env python3
"""
ADE/FDE evaluation for the GRU trajectory predictor (modules/trajectory_gru.py)
on a held-out split of CARLA world-space trajectories.

The checkpoint at weights/trajectory_gru.pth was trained (2026-06-12, see
docs/assets/trajectory_gru/train_log.txt) on 9 episode_*.csv files under a
`data/trajectories` directory that no longer exists. data/trajectories_hanoi_v2
(120 episodes) was collected entirely on 2026-06-17, after that training run,
so it is guaranteed unseen by the model — no separate split bookkeeping is
needed, but to keep the evaluation reproducible and explicit we still only use
the last `--holdout-frac` fraction of its episodes (by episode index).

Mirrors the mode-selection used in production
(modules/trajectory_predictor.py: EnsembleTrajectoryPredictor — best_mode =
argmax(softmax(mode_logits))) rather than an oracle best-of-N-modes pick.

Usage:
    python scripts/eval/eval_trajectory_gru.py
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'scripts' / 'train'))

from modules.trajectory_gru import TrajectoryGRU, PRED_HORIZONS  # noqa: E402
from train_trajectory_predictor import TrajectoryDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=str, default='data/trajectories_hanoi_v2')
    parser.add_argument('--ckpt', type=str, default='weights/trajectory_gru.pth')
    parser.add_argument('--holdout-frac', type=float, default=0.2)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--out', type=str,
                         default='docs/assets/trajectory_gru/ade_fde_results.json')
    args = parser.parse_args()

    data_dir = project_root / args.data
    episode_files = sorted(
        p for p in data_dir.glob('episode_*.csv') if not p.stem.endswith('_goals'))
    n_holdout = max(1, round(len(episode_files) * args.holdout_frac))
    holdout_files = episode_files[-n_holdout:]
    print(f"Held-out split: last {n_holdout}/{len(episode_files)} episodes "
          f"({holdout_files[0].name} .. {holdout_files[-1].name})")

    with tempfile.TemporaryDirectory(prefix='gru_eval_holdout_') as tmp:
        tmp_dir = Path(tmp)
        for f in holdout_files:
            shutil.copy(f, tmp_dir / f.name)
        dataset = TrajectoryDataset(str(tmp_dir), stride=args.stride)

    ckpt_path = project_root / args.ckpt
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Override the split's own normalization stats with the checkpoint's —
    # the model's weights were trained against those exact stats.
    dataset.norm_mean = np.asarray(ckpt['norm_mean'], dtype=np.float32)
    dataset.norm_std = np.asarray(ckpt['norm_std'], dtype=np.float32)

    model = TrajectoryGRU(hidden_size=ckpt['hidden_size'], num_modes=ckpt['num_modes'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    pos_std = dataset.norm_std[:2]  # meters per normalized unit
    horizon_err_m = {h: [] for h in PRED_HORIZONS}

    with torch.no_grad():
        for i in range(len(dataset)):
            obs_seq, target_rel, _ = dataset[i]
            pred_deltas, mode_logits, _ = model(obs_seq.unsqueeze(0))

            best_mode = int(torch.softmax(mode_logits, dim=-1)[0].argmax().item())
            pred_m = pred_deltas[0, best_mode].numpy() * pos_std
            target_m = target_rel.numpy() * pos_std

            err = np.linalg.norm(pred_m - target_m, axis=-1)
            for h, e in zip(PRED_HORIZONS, err):
                horizon_err_m[h].append(e)

    results = {
        f'{h}s': {'ade_m': float(np.mean(horizon_err_m[h])),
                   'n': len(horizon_err_m[h])}
        for h in PRED_HORIZONS
    }
    results['fde_m'] = results[f'{PRED_HORIZONS[-1]}s']['ade_m']
    results['ade_m_avg_all_horizons'] = float(
        np.mean([results[f'{h}s']['ade_m'] for h in PRED_HORIZONS]))
    results['n_samples'] = len(dataset)
    results['holdout_episodes'] = [f.name for f in holdout_files]
    results['note'] = ('Internal evaluation on simulated (CARLA) world-space '
                        'trajectories, not real-camera pixel-space data.')

    out_path = (project_root.parent / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
