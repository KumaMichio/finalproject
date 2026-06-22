#!/usr/bin/env python3
"""
eval_route_predictor_priors.py — Mine real multi-junction turn sequences from
CARLA episode goal logs to:

  1. Evaluate the route_predictor.py assumption that every hop after the
     first is 'straight' (hop_direction = direction if hop==0 else 'straight'),
     against actual recorded turn sequences.
  2. Derive empirical direction priors (marginal + 1st-order Markov, conditioned
     on the previous hop's direction) usable to replace the hardcoded
     'straight, confidence=1.0' placeholder for hop>=1.

Input:  custom_tracking_system/data/trajectories_hanoi_v2/episode_*_goals.csv
Output: custom_tracking_system/data/route_predictor_priors.json

Usage:
    python scripts/eval/eval_route_predictor_priors.py
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOALS_DIR = ROOT / 'data' / 'trajectories_hanoi_v2'
OUT_PATH = ROOT / 'data' / 'route_predictor_priors.json'

DIRECTIONS = ['straight', 'left', 'right', 'u_turn']


def angle_to_label(angle_deg: float) -> str:
    """Same convention as scripts/train/train_goal_classifier.py:angle_to_label."""
    a = float(angle_deg)
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    if abs(a) < 30:
        return 'straight'
    if abs(a) > 150:
        return 'u_turn'
    return 'right' if a > 0 else 'left'


def load_all_goals() -> pd.DataFrame:
    files = sorted(glob.glob(str(GOALS_DIR / 'episode_*_goals.csv')))
    if not files:
        raise FileNotFoundError(f"No episode_*_goals.csv found in {GOALS_DIR}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df['label'] = df['turn_angle_deg'].apply(angle_to_label)
    return df


def build_sequences(df: pd.DataFrame) -> list[list[str]]:
    """Group by (episode_id, actor_id), sort by t_enter -> list of direction labels."""
    sequences = []
    for (_, _), g in df.groupby(['episode_id', 'actor_id']):
        g = g.sort_values('t_enter')
        sequences.append(list(g['label']))
    return sequences


def main():
    df = load_all_goals()
    print(f"Total goal events: {len(df)}  across {df['episode_id'].nunique()} episodes")

    sequences = build_sequences(df)
    multi_hop = [s for s in sequences if len(s) >= 2]
    print(f"Total actor sequences: {len(sequences)}  "
          f"({len(multi_hop)} with >=2 junctions, usable for hop1+ eval)")

    # ------------------------------------------------------------------
    # 1. "Always straight" baseline accuracy for hop>=1 (current behavior)
    # ------------------------------------------------------------------
    hop_n_labels = defaultdict(list)  # hop_index (0-based) -> [label, ...]
    transitions = defaultdict(lambda: defaultdict(int))  # prev_label -> {next_label: count}

    for seq in sequences:
        for i, lab in enumerate(seq):
            hop_n_labels[i].append(lab)
        for prev, nxt in zip(seq[:-1], seq[1:]):
            transitions[prev][nxt] += 1

    straight_baseline_hits = 0
    straight_baseline_total = 0
    for seq in multi_hop:
        for lab in seq[1:]:  # hop 1+
            straight_baseline_total += 1
            if lab == 'straight':
                straight_baseline_hits += 1
    baseline_acc = straight_baseline_hits / max(straight_baseline_total, 1)

    print(f"\n[Route Predictor hop>=1 'always straight' baseline]")
    print(f"  n={straight_baseline_total}  accuracy={baseline_acc:.4f}")

    # ------------------------------------------------------------------
    # 2. Marginal prior per hop index (hop 0, 1, 2, 3...)
    # ------------------------------------------------------------------
    marginal_by_hop = {}
    print(f"\n[Marginal direction distribution by hop index]")
    for hop_idx in sorted(hop_n_labels):
        labels = hop_n_labels[hop_idx]
        if len(labels) < 20:
            continue
        dist = {d: labels.count(d) / len(labels) for d in DIRECTIONS}
        marginal_by_hop[hop_idx] = {'n': len(labels), 'dist': dist}
        print(f"  hop{hop_idx}: n={len(labels):5d}  " +
              "  ".join(f"{d}={dist[d]:.3f}" for d in DIRECTIONS))

    # Overall hop>=1 marginal (pooled across hop indices) -- the single
    # number most directly usable as a drop-in replacement for the
    # hardcoded {'straight':1.0,...} in route_predictor.py / demo scripts.
    pooled_hop1plus = []
    for hop_idx, labels in hop_n_labels.items():
        if hop_idx >= 1:
            pooled_hop1plus.extend(labels)
    overall_hop1plus_dist = {d: pooled_hop1plus.count(d) / max(len(pooled_hop1plus), 1)
                              for d in DIRECTIONS}
    print(f"\n[Pooled hop>=1 marginal prior] n={len(pooled_hop1plus)}")
    for d in DIRECTIONS:
        print(f"  {d}: {overall_hop1plus_dist[d]:.4f}")

    # ------------------------------------------------------------------
    # 3. 1st-order Markov transition matrix: P(next | prev)
    # ------------------------------------------------------------------
    markov = {}
    print(f"\n[Transition matrix P(next | prev)]")
    for prev in DIRECTIONS:
        counts = transitions.get(prev, {})
        total = sum(counts.values())
        if total == 0:
            continue
        dist = {d: counts.get(d, 0) / total for d in DIRECTIONS}
        markov[prev] = {'n': total, 'dist': dist}
        print(f"  prev={prev:9s} n={total:5d}  " +
              "  ".join(f"{d}={dist[d]:.3f}" for d in DIRECTIONS))

    # Accuracy if we picked argmax(P(next|prev)) instead of hardcoded 'straight'
    markov_hits = 0
    markov_total = 0
    for seq in multi_hop:
        for prev, nxt in zip(seq[:-1], seq[1:]):
            pred = max(markov[prev]['dist'], key=markov[prev]['dist'].get) if prev in markov else 'straight'
            markov_total += 1
            if pred == nxt:
                markov_hits += 1
    markov_acc = markov_hits / max(markov_total, 1)
    print(f"\n[Markov argmax accuracy for hop>=1] n={markov_total}  accuracy={markov_acc:.4f}"
          f"  (vs always-straight baseline {baseline_acc:.4f})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        'n_episodes': int(df['episode_id'].nunique()),
        'n_goal_events': int(len(df)),
        'n_sequences': len(sequences),
        'n_multi_hop_sequences': len(multi_hop),
        'always_straight_baseline_accuracy': round(baseline_acc, 4),
        'always_straight_baseline_n': straight_baseline_total,
        'markov_argmax_accuracy': round(markov_acc, 4),
        'markov_argmax_n': markov_total,
        'marginal_by_hop_index': {str(k): v for k, v in marginal_by_hop.items()},
        'pooled_hop1plus_prior': overall_hop1plus_dist,
        'markov_transition_matrix': markov,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == '__main__':
    main()
