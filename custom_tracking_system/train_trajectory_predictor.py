#!/usr/bin/env python3
"""
Train the Seq2Seq GRU trajectory predictor on CARLA trajectory data.

Input: CSV files produced by collect_trajectory_data.py
       (data/trajectories/episode_*.csv, columns: t, actor_id, class, x, y, vx, vy)

Output: weights/trajectory_gru.pth — checkpoint consumed by
        modules.trajectory_predictor.LearnedTrajectoryPredictor

Usage:
    python train_trajectory_predictor.py --data data/trajectories --epochs 50 \
        --out weights/trajectory_gru.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.trajectory_gru import (
    TrajectoryGRU, trajectory_loss, build_feature_vector, compute_intent_label,
    HORIZON_STEPS, T_OBS, NUM_NEIGHBORS, HIDDEN_SIZE, NUM_MODES, PRED_HORIZONS, CLASS_LIST,
)

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_MIN_STD = 1e-3


class TrajectoryDataset(Dataset):
    """Sliding-window samples built from CARLA trajectory CSVs.

    Each sample = T_OBS past observations (with K-nearest-neighbor context)
    + future positions at PRED_HORIZONS + an intent label.
    """

    def __init__(self, data_dir: str, stride: int = 5, num_neighbors: int = NUM_NEIGHBORS):
        self.num_neighbors = num_neighbors
        self.samples = []  # raw (un-normalized) samples; features built in __getitem__

        csv_files = sorted(Path(data_dir).glob('episode_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No episode_*.csv files found under {data_dir}")

        max_h = max(HORIZON_STEPS)

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Per-timestep numpy index for fast neighbor lookup.
            timestep_data = {}
            for t, group in df.groupby('t'):
                timestep_data[t] = (
                    group['actor_id'].to_numpy(),
                    group[['x', 'y']].to_numpy(dtype=np.float32),
                    group[['vx', 'vy']].to_numpy(dtype=np.float32),
                )

            for actor_id, actor_df in df.groupby('actor_id'):
                actor_df = actor_df.sort_values('t').reset_index(drop=True)
                n = len(actor_df)

                for start in range(0, n - T_OBS - max_h, stride):
                    obs_rows = actor_df.iloc[start:start + T_OBS]
                    last_obs = obs_rows.iloc[-1]

                    future_indices = [start + T_OBS - 1 + h for h in HORIZON_STEPS]
                    if future_indices[-1] >= n:
                        continue
                    future_rows = actor_df.iloc[future_indices]

                    obs_list = []
                    for _, row in obs_rows.iterrows():
                        neighbors = self._get_neighbors(
                            timestep_data.get(row['t']), actor_id,
                            row['x'], row['y'], row['vx'], row['vy'])
                        obs_list.append({
                            'pos': (row['x'], row['y']),
                            'vel': (row['vx'], row['vy']),
                            'cls': row['class'],
                            'neighbors': neighbors,
                        })

                    self.samples.append({
                        'obs': obs_list,
                        'last_obs_pos': (last_obs['x'], last_obs['y']),
                        'last_obs_vel': (last_obs['vx'], last_obs['vy']),
                        'future_pos': [(r['x'], r['y']) for _, r in future_rows.iterrows()],
                        'future_vel_last': (future_rows.iloc[-1]['vx'], future_rows.iloc[-1]['vy']),
                    })

        if not self.samples:
            raise RuntimeError("No training samples generated — check episode length vs T_OBS+horizons")

        self.norm_mean, self.norm_std = self._compute_norm_stats()
        logger.info("TrajectoryDataset: %d samples from %d episode file(s)",
                     len(self.samples), len(csv_files))
        logger.info("Normalization mean=%s std=%s", self.norm_mean, self.norm_std)

    def _get_neighbors(self, ts_data, actor_id, x, y, vx, vy):
        if ts_data is None:
            return []
        actor_ids, xy, vxvy = ts_data
        mask = actor_ids != actor_id
        if not mask.any():
            return []

        other_xy = xy[mask]
        other_vxvy = vxvy[mask]
        dists = np.linalg.norm(other_xy - np.array([x, y], dtype=np.float32), axis=1)
        order = np.argsort(dists)[:self.num_neighbors]

        rel = []
        for idx in order:
            dx, dy = other_xy[idx, 0] - x, other_xy[idx, 1] - y
            dvx, dvy = other_vxvy[idx, 0] - vx, other_vxvy[idx, 1] - vy
            rel.append((dx, dy, dvx, dvy))
        return rel

    def _compute_norm_stats(self):
        own_states = np.array(
            [obs['pos'] + obs['vel'] for s in self.samples for obs in s['obs']],
            dtype=np.float32)
        mean = own_states.mean(axis=0)
        std = own_states.std(axis=0)
        std = np.maximum(std, _MIN_STD)
        return mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        obs_seq = np.stack([
            build_feature_vector(o['pos'], o['vel'], o['cls'], o['neighbors'],
                                  self.norm_mean, self.norm_std)
            for o in s['obs']
        ])

        last_pos = np.array(s['last_obs_pos'], dtype=np.float32)
        pos_std = self.norm_std[:2]
        target_rel = np.stack([
            (np.array(p, dtype=np.float32) - last_pos) / pos_std
            for p in s['future_pos']
        ])

        intent_label = compute_intent_label(s['last_obs_vel'], s['future_vel_last'])

        return (
            torch.from_numpy(obs_seq).float(),
            torch.from_numpy(target_rel).float(),
            torch.tensor(intent_label, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=str, default='data/trajectories')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--out', type=str, default='weights/trajectory_gru.pth')
    args = parser.parse_args()

    dataset = TrajectoryDataset(args.data, stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Training on device: %s", device)

    model = TrajectoryGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {'traj_loss': 0.0, 'mode_loss': 0.0, 'intent_loss': 0.0, 'total': 0.0}

        for obs_seq, target_rel, intent_labels in loader:
            obs_seq = obs_seq.to(device)
            target_rel = target_rel.to(device)
            intent_labels = intent_labels.to(device)

            pred_deltas, mode_logits, intent_logits = model(obs_seq)
            loss, comps = trajectory_loss(pred_deltas, mode_logits, intent_logits,
                                           target_rel, intent_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in comps.items():
                epoch_losses[k] += v

        n_batches = len(loader)
        logger.info(
            "Epoch %3d/%d - total=%.4f traj=%.4f mode=%.4f intent=%.4f",
            epoch, args.epochs,
            epoch_losses['total'] / n_batches,
            epoch_losses['traj_loss'] / n_batches,
            epoch_losses['mode_loss'] / n_batches,
            epoch_losses['intent_loss'] / n_batches,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'norm_mean': dataset.norm_mean,
        'norm_std': dataset.norm_std,
        'hidden_size': HIDDEN_SIZE,
        'num_modes': NUM_MODES,
        'pred_horizons': PRED_HORIZONS,
        't_obs': T_OBS,
        'num_neighbors': NUM_NEIGHBORS,
        'class_list': CLASS_LIST,
    }, out_path)
    logger.info("Saved checkpoint -> %s", out_path)


if __name__ == '__main__':
    main()
