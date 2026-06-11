"""
Learned Trajectory Predictor — Seq2Seq GRU, multi-modal + intent classification.

Trained offline (see train_trajectory_predictor.py) on world-coordinate
(meters) trajectories collected from CARLA (see collect_trajectory_data.py),
then used at inference time by modules.trajectory_predictor.LearnedTrajectoryPredictor
as one branch of an ensemble alongside the existing Kalman filter.

Model overview
--------------
Encoder: GRU over T_OBS past observations, each observation =
    [x, y, vx, vy] (normalized) + one-hot class + K nearest-neighbor
    relative [dx, dy, dvx, dvy] (normalized), K = NUM_NEIGHBORS.

Decoder: GRUCell unrolled over len(PRED_HORIZONS) steps, run once per mode
(NUM_MODES hypotheses). Each step predicts a normalized displacement; the
cumulative sum gives the predicted (normalized) displacement from the last
observed position at each horizon.

Auxiliary heads (from the encoder's final hidden state):
  - mode probabilities (softmax over NUM_MODES)
  - intent classification (straight / turn_left / turn_right / stop)

All shared constants (horizons, normalization helpers, feature assembly) live
here so train_trajectory_predictor.py and LearnedTrajectoryPredictor build
identical input tensors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Shared configuration
# ----------------------------------------------------------------------

DT = 0.1                                   # seconds/step (CARLA fixed_delta_seconds)
PRED_HORIZONS = (0.5, 1.0, 1.5, 2.0, 3.0)  # seconds — must match trajectory_predictor.PRED_HORIZONS
HORIZON_STEPS = tuple(round(h / DT) for h in PRED_HORIZONS)  # (5, 10, 15, 20, 30)

T_OBS = 8                                  # observation window (0.8s @ 10Hz)
NUM_MODES = 3
NUM_NEIGHBORS = 3
CLASS_LIST = ('car', 'motorcycle', 'person')
INTENT_LABELS = ('straight', 'turn_left', 'turn_right', 'stop')
NUM_INTENT_CLASSES = len(INTENT_LABELS)

INPUT_DIM = 4 + len(CLASS_LIST) + NUM_NEIGHBORS * 4  # = 19
HIDDEN_SIZE = 96
MODE_EMB_DIM = 8


# ----------------------------------------------------------------------
# Feature assembly (shared by training dataset and LearnedTrajectoryPredictor)
# ----------------------------------------------------------------------

def class_to_onehot(obj_class: str) -> np.ndarray:
    onehot = np.zeros(len(CLASS_LIST), dtype=np.float32)
    if obj_class in CLASS_LIST:
        onehot[CLASS_LIST.index(obj_class)] = 1.0
    else:
        onehot[0] = 1.0  # default to 'car' for unknown classes
    return onehot


def compute_intent_label(vel_obs, vel_future, speed_eps: float = 0.5,
                          turn_threshold_deg: float = 20.0) -> int:
    """Derive an intent label from observed vs. future-horizon velocity.

    Args:
        vel_obs:    (vx, vy) at the end of the observation window.
        vel_future: (vx, vy) at the final prediction horizon (t+3s).

    Returns:
        Index into INTENT_LABELS.
    """
    speed_obs = float(np.linalg.norm(vel_obs))
    speed_future = float(np.linalg.norm(vel_future))

    if speed_obs < speed_eps:
        return INTENT_LABELS.index('straight')

    if speed_future < 0.3 * speed_obs:
        return INTENT_LABELS.index('stop')

    heading_obs = np.arctan2(vel_obs[1], vel_obs[0])
    heading_future = np.arctan2(vel_future[1], vel_future[0])
    delta_deg = np.degrees(heading_future - heading_obs)
    delta_deg = (delta_deg + 180.0) % 360.0 - 180.0

    if delta_deg > turn_threshold_deg:
        return INTENT_LABELS.index('turn_left')
    if delta_deg < -turn_threshold_deg:
        return INTENT_LABELS.index('turn_right')
    return INTENT_LABELS.index('straight')


def build_feature_vector(pos, vel, obj_class: str, neighbor_rel: list,
                          norm_mean: np.ndarray, norm_std: np.ndarray) -> np.ndarray:
    """Assemble the INPUT_DIM feature vector for a single timestep.

    Args:
        pos: (x, y) world meters.
        vel: (vx, vy) world m/s.
        obj_class: one of CLASS_LIST (or unknown -> treated as 'car').
        neighbor_rel: list of up to NUM_NEIGHBORS (dx, dy, dvx, dvy) tuples,
                       relative to `pos`/`vel`, sorted nearest-first.
        norm_mean, norm_std: shape (4,) — mean/std of [x, y, vx, vy] from
                       the training set (used to normalize own state; only
                       std is used for relative neighbor features).

    Returns:
        np.ndarray of shape (INPUT_DIM,), dtype float32.
    """
    own = (np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32) - norm_mean) / norm_std

    neighbor_feats = np.zeros(NUM_NEIGHBORS * 4, dtype=np.float32)
    for i, rel in enumerate(neighbor_rel[:NUM_NEIGHBORS]):
        neighbor_feats[i * 4:(i + 1) * 4] = np.asarray(rel, dtype=np.float32) / norm_std

    return np.concatenate([own, class_to_onehot(obj_class), neighbor_feats]).astype(np.float32)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class TrajectoryGRU(nn.Module):
    """Seq2Seq GRU trajectory predictor with multi-modal + intent outputs."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_size: int = HIDDEN_SIZE,
                 num_modes: int = NUM_MODES, num_horizons: int = len(PRED_HORIZONS),
                 num_intent_classes: int = NUM_INTENT_CLASSES,
                 mode_emb_dim: int = MODE_EMB_DIM):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modes = num_modes
        self.num_horizons = num_horizons

        self.encoder = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)

        self.mode_embeddings = nn.Embedding(num_modes, mode_emb_dim)
        self.decoder_cell = nn.GRUCell(2 + mode_emb_dim, hidden_size)
        self.displacement_head = nn.Linear(hidden_size, 2)

        self.mode_prob_head = nn.Linear(hidden_size, num_modes)
        self.intent_head = nn.Linear(hidden_size, num_intent_classes)

    def forward(self, obs_seq: torch.Tensor):
        """
        Args:
            obs_seq: (batch, T_OBS, input_dim) — past observations, normalized.

        Returns:
            pred_deltas:  (batch, num_modes, num_horizons, 2) — cumulative
                          normalized displacement from the last observed
                          position, per mode and horizon.
            mode_logits:  (batch, num_modes)
            intent_logits: (batch, num_intent_classes)
        """
        batch_size = obs_seq.size(0)
        _, h_n = self.encoder(obs_seq)
        encoder_hidden = h_n.squeeze(0)  # (batch, hidden_size)

        mode_logits = self.mode_prob_head(encoder_hidden)
        intent_logits = self.intent_head(encoder_hidden)

        all_modes = []
        for m in range(self.num_modes):
            mode_emb = self.mode_embeddings(
                torch.full((batch_size,), m, dtype=torch.long, device=obs_seq.device))

            h = encoder_hidden
            delta_prev = torch.zeros(batch_size, 2, device=obs_seq.device)
            cumulative = torch.zeros(batch_size, 2, device=obs_seq.device)

            steps = []
            for _ in range(self.num_horizons):
                cell_input = torch.cat([delta_prev, mode_emb], dim=-1)
                h = self.decoder_cell(cell_input, h)
                delta = self.displacement_head(h)
                cumulative = cumulative + delta
                steps.append(cumulative)
                delta_prev = delta

            all_modes.append(torch.stack(steps, dim=1))  # (batch, num_horizons, 2)

        pred_deltas = torch.stack(all_modes, dim=1)  # (batch, num_modes, num_horizons, 2)
        return pred_deltas, mode_logits, intent_logits


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------

def trajectory_loss(pred_deltas, mode_logits, intent_logits,
                     target_rel, intent_labels,
                     lambda_mode: float = 0.1, lambda_intent: float = 0.2):
    """
    Args:
        pred_deltas:   (batch, num_modes, num_horizons, 2) — model output.
        mode_logits:   (batch, num_modes)
        intent_logits: (batch, num_intent_classes)
        target_rel:    (batch, num_horizons, 2) — ground-truth normalized
                       displacement from the last observed position.
        intent_labels: (batch,) long tensor, values in [0, num_intent_classes).

    Returns:
        total_loss, dict of component losses (for logging).
    """
    # Per-mode displacement error, averaged over horizons and xy.
    diff = pred_deltas - target_rel.unsqueeze(1)              # (batch, M, H, 2)
    per_mode_err = diff.pow(2).sum(dim=-1).mean(dim=-1)        # (batch, M)

    best_mode = per_mode_err.argmin(dim=1)                     # (batch,)
    traj_loss = per_mode_err.gather(1, best_mode.unsqueeze(1)).mean()

    mode_loss = F.cross_entropy(mode_logits, best_mode)
    intent_loss = F.cross_entropy(intent_logits, intent_labels)

    total = traj_loss + lambda_mode * mode_loss + lambda_intent * intent_loss
    return total, {
        'traj_loss': traj_loss.item(),
        'mode_loss': mode_loss.item(),
        'intent_loss': intent_loss.item(),
        'total': total.item(),
    }
