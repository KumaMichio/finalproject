"""
Visualize trajectory data collected from CARLA.
Usage:
    python visualize_trajectories.py --dir data/trajectories_hanoi
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False


CLASS_COLORS = {
    "car":        "#4fc3f7",
    "motorcycle": "#ff8a65",
    "bus":        "#a5d6a7",
    "truck":      "#ce93d8",
    "person":     "#fff176",
}


def load_all_episodes(traj_dir: Path):
    frames = []
    for f in sorted(traj_dir.glob("episode_*.csv")):
        if "_goals" in f.name:
            continue
        if not HAS_PD:
            import csv
            with open(f, newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    frames.append(row)
        else:
            df = pd.read_csv(f, dtype={"lat": str, "lon": str})
            frames.append(df)

    if not frames:
        return None
    if HAS_PD:
        return pd.concat(frames, ignore_index=True)
    return frames


def plot_trajectory_map(df, out_path: Path):
    """Vẽ trajectory map, clip về vùng có mật độ cao (loại bỏ edge outliers)."""
    # Lọc xe tốc độ cao bất thường (xe lao ra ngoài OpenDRIVE boundary)
    if HAS_PD:
        df_clean = df[df["speed"].astype(float) < 40.0]   # < 144 km/h
    else:
        df_clean = [r for r in df if float(r.get("speed", 0)) < 40.0]

    # Tính axis limits theo percentile để loại bỏ outlier path
    if HAS_PD:
        xs_all = df_clean["x"].astype(float)
        ys_all = df_clean["y"].astype(float)
    else:
        xs_all = np.array([float(r["x"]) for r in df_clean])
        ys_all = np.array([float(r["y"]) for r in df_clean])

    if len(xs_all) == 0:
        print("No clean data for trajectory map.")
        return

    x_lo, x_hi = np.percentile(xs_all, 1), np.percentile(xs_all, 99)
    y_lo, y_hi = np.percentile(ys_all, 1), np.percentile(ys_all, 99)
    pad_x = (x_hi - x_lo) * 0.05 + 20
    pad_y = (y_hi - y_lo) * 0.05 + 20

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    classes = df_clean["class"].unique() if HAS_PD else set(r["class"] for r in df_clean)

    for cls in classes:
        color = CLASS_COLORS.get(cls, "#ffffff")
        if HAS_PD:
            sub = df_clean[df_clean["class"] == cls]
            for actor_id, grp in sub.groupby("actor_id"):
                grp = grp.sort_values("t")
                ax.plot(grp["x"].values, grp["y"].values,
                        color=color, alpha=0.25, linewidth=0.7)
        else:
            xs = [float(r["x"]) for r in df_clean if r["class"] == cls]
            ys = [float(r["y"]) for r in df_clean if r["class"] == cls]
            ax.scatter(xs, ys, c=color, s=0.5, alpha=0.2)

    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

    patches = [mpatches.Patch(color=CLASS_COLORS.get(c, "#fff"), label=c) for c in classes]
    ax.legend(handles=patches, loc="upper right", framealpha=0.3,
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)

    n_filtered = (len(df) - len(df_clean)) if HAS_PD else (len(df) - len(df_clean))
    ax.set_title(f"Trajectory Map — Hanoi District 3\n(speed < 144 km/h, {n_filtered:,} outlier rows removed)",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("CARLA X (m)", color="#aaa")
    ax.set_ylabel("CARLA Y (m)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_class_distribution(df, out_path: Path):
    if HAS_PD:
        counts = df["class"].value_counts()
    else:
        from collections import Counter
        counts_raw = Counter(r["class"] for r in df)
        counts = counts_raw

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Bar chart
    labels = list(counts.keys()) if not HAS_PD else counts.index.tolist()
    values = list(counts.values()) if not HAS_PD else counts.values.tolist()
    colors = [CLASS_COLORS.get(l, "#fff") for l in labels]
    axes[0].bar(labels, values, color=colors, edgecolor="#333")
    axes[0].set_title("Rows per class", color="white")
    axes[0].set_ylabel("Count", color="#aaa")
    for label in axes[0].get_xticklabels():
        label.set_color("#ccc")
    for label in axes[0].get_yticklabels():
        label.set_color("#ccc")

    # Pie chart
    axes[1].pie(values, labels=labels, colors=colors,
                autopct="%1.1f%%", textprops={"color": "white"},
                wedgeprops={"edgecolor": "#333"})
    axes[1].set_title("Class distribution", color="white")

    plt.suptitle("Class Distribution in Trajectory Data", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_speed_histogram(df, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#0f0f1a")
    axes_flat = axes.flatten()

    classes_order = ["car", "motorcycle", "bus", "truck", "person"]
    for i, cls in enumerate(classes_order):
        ax = axes_flat[i]
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        if HAS_PD:
            speeds = df[df["class"] == cls]["speed"].dropna().astype(float).values
        else:
            speeds = np.array([float(r["speed"]) for r in df
                               if r["class"] == cls and r["speed"]])

        if len(speeds) == 0:
            ax.set_title(cls, color="#666")
            continue

        color = CLASS_COLORS.get(cls, "#fff")
        ax.hist(speeds, bins=40, color=color, alpha=0.8, edgecolor="#333")
        ax.axvline(np.mean(speeds), color="white", linestyle="--", linewidth=1,
                   label=f"mean {np.mean(speeds):.1f} m/s")
        ax.set_title(f"{cls}  (n={len(speeds):,})", color="white")
        ax.set_xlabel("speed (m/s)", color="#aaa")
        ax.set_ylabel("count", color="#aaa")
        ax.legend(fontsize=7, labelcolor="white", framealpha=0.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#ccc")

    # Hide last subplot
    axes_flat[-1].set_visible(False)

    plt.suptitle("Speed Distribution by Class", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_phase_breakdown(df, out_path: Path):
    if HAS_PD:
        phase_counts = df["phase"].value_counts()
        has_post = "post_accident" in phase_counts.index and phase_counts.get("post_accident", 0) > 0
    else:
        from collections import Counter
        phase_counts = Counter(r.get("phase", "normal") for r in df)
        has_post = phase_counts.get("post_accident", 0) > 0

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    labels = list(phase_counts.keys()) if not HAS_PD else phase_counts.index.tolist()
    values = list(phase_counts.values()) if not HAS_PD else phase_counts.values.tolist()
    phase_colors = {"normal": "#4fc3f7", "post_accident": "#ef5350"}
    colors = [phase_colors.get(l, "#aaa") for l in labels]

    bars = ax.bar(labels, values, color=colors, edgecolor="#333")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", color="white", fontsize=9)

    ax.set_title("Trajectory Rows by Phase\n(normal vs post_accident)",
                 color="white", fontsize=11)
    ax.set_ylabel("Row count", color="#aaa")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#ccc")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_goals_analysis(traj_dir: Path, out_path: Path):
    """Turn angle distribution at junctions from goals CSV."""
    all_goals = []
    for f in sorted(traj_dir.glob("episode_*_goals.csv")):
        if HAS_PD:
            g = pd.read_csv(f)
            if len(g):
                all_goals.append(g)
        else:
            import csv
            with open(f, newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                all_goals.extend(list(reader))

    if not all_goals:
        print("No goal events found, skipping goals analysis.")
        return

    if HAS_PD:
        gdf = pd.concat(all_goals, ignore_index=True)
        angles = gdf["turn_angle_deg"].astype(float).values
        classes = gdf["class"].values
    else:
        angles = np.array([float(r["turn_angle_deg"]) for r in all_goals])
        classes = [r["class"] for r in all_goals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Turn angle histogram
    axes[0].hist(angles, bins=60, color="#4fc3f7", edgecolor="#333", alpha=0.85)
    axes[0].axvline(-45, color="#ff8a65", linestyle="--", linewidth=1, label="±45° (left/right)")
    axes[0].axvline(45, color="#ff8a65", linestyle="--", linewidth=1)
    axes[0].axvline(0, color="#a5d6a7", linestyle="-", linewidth=1.5, label="straight (0°)")
    axes[0].set_title("Junction Turn Angle Distribution", color="white")
    axes[0].set_xlabel("Turn angle (°, negative=left, positive=right)", color="#aaa")
    axes[0].set_ylabel("Count", color="#aaa")
    axes[0].legend(labelcolor="white", framealpha=0.2)
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_color("#ccc")

    # Pie: left / straight / right
    left    = np.sum(angles < -30)
    straight = np.sum(np.abs(angles) <= 30)
    right   = np.sum(angles > 30)
    axes[1].pie(
        [left, straight, right],
        labels=[f"Left ({left})", f"Straight ({straight})", f"Right ({right})"],
        colors=["#ff8a65", "#4fc3f7", "#a5d6a7"],
        autopct="%1.1f%%",
        textprops={"color": "white"},
        wedgeprops={"edgecolor": "#333"},
    )
    axes[1].set_title("Turn Type at Junctions", color="white")

    plt.suptitle(f"Goal Events Analysis  (total={len(angles):,})", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_episode_stats(df, traj_dir: Path, out_path: Path):
    """Rows per episode and speed stats."""
    if not HAS_PD:
        print("pandas required for episode stats plot")
        return

    ep_stats = df.groupby("episode_id").agg(
        rows=("actor_id", "count"),
        actors=("actor_id", "nunique"),
        mean_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        post_acc=("phase", lambda x: (x == "post_accident").sum()),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0f0f1a")
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Rows per episode
    axes_flat[0].bar(ep_stats["episode_id"], ep_stats["rows"], color="#4fc3f7", edgecolor="#333")
    axes_flat[0].set_title("Traj rows per episode", color="white")
    axes_flat[0].set_xlabel("Episode", color="#aaa")
    axes_flat[0].set_ylabel("Row count", color="#aaa")

    # Unique actors per episode
    axes_flat[1].bar(ep_stats["episode_id"], ep_stats["actors"], color="#ff8a65", edgecolor="#333")
    axes_flat[1].set_title("Unique actors per episode", color="white")
    axes_flat[1].set_xlabel("Episode", color="#aaa")
    axes_flat[1].set_ylabel("Actor count", color="#aaa")

    # Mean speed per episode
    axes_flat[2].plot(ep_stats["episode_id"], ep_stats["mean_speed"],
                      color="#a5d6a7", linewidth=1.5, marker="o", markersize=3)
    axes_flat[2].set_title("Mean speed per episode (m/s)", color="white")
    axes_flat[2].set_xlabel("Episode", color="#aaa")
    axes_flat[2].set_ylabel("Speed (m/s)", color="#aaa")

    # Post-accident rows per episode
    axes_flat[3].bar(ep_stats["episode_id"], ep_stats["post_acc"],
                     color="#ef5350", edgecolor="#333")
    axes_flat[3].set_title("Post-accident rows per episode", color="white")
    axes_flat[3].set_xlabel("Episode", color="#aaa")
    axes_flat[3].set_ylabel("Row count", color="#aaa")

    for ax in axes_flat:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#ccc")

    plt.suptitle("Per-Episode Statistics", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def print_summary(df, traj_dir: Path):
    goals_dfs = []
    for f in sorted(traj_dir.glob("episode_*_goals.csv")):
        if HAS_PD:
            goals_dfs.append(pd.read_csv(f))

    n_episodes = len([f for f in traj_dir.glob("episode_[0-9]*.csv")
                      if "_goals" not in f.name])

    if HAS_PD:
        n_rows = len(df)
        classes = df["class"].value_counts().to_dict()
        n_post = (df["phase"] == "post_accident").sum()
        n_goals = sum(len(g) for g in goals_dfs) if goals_dfs else 0
        unique_actors = df["actor_id"].nunique()
        mean_speed = df["speed"].mean()
        max_speed  = df["speed"].max()
        ep_counts  = df.groupby("episode_id").size()
        rows_per_ep_mean = ep_counts.mean()
    else:
        n_rows = len(df)
        from collections import Counter
        classes = dict(Counter(r["class"] for r in df))
        n_post = sum(1 for r in df if r.get("phase") == "post_accident")
        n_goals = 0
        unique_actors = len(set(r["actor_id"] for r in df))
        speeds = [float(r["speed"]) for r in df if r.get("speed")]
        mean_speed = sum(speeds) / max(len(speeds), 1)
        max_speed = max(speeds) if speeds else 0
        rows_per_ep_mean = n_rows / max(n_episodes, 1)

    print("\n" + "="*55)
    print("  TRAJECTORY COLLECTION SUMMARY")
    print("="*55)
    print(f"  Episodes completed   : {n_episodes}")
    print(f"  Total traj rows      : {n_rows:,}")
    print(f"  Rows / episode (avg) : {rows_per_ep_mean:,.0f}")
    print(f"  Unique actors        : {unique_actors}")
    print(f"  Post-accident rows   : {n_post:,} ({n_post/max(n_rows,1)*100:.1f}%)")
    print(f"  Goal events          : {n_goals}")
    print(f"  Mean speed           : {mean_speed:.2f} m/s  ({mean_speed*3.6:.1f} km/h)")
    print(f"  Max speed            : {max_speed:.2f} m/s  ({max_speed*3.6:.1f} km/h)")
    print(f"  Class breakdown:")
    for cls, cnt in sorted(classes.items(), key=lambda x: -x[1]):
        pct = cnt / max(n_rows, 1) * 100
        speeds_c = df[df["class"] == cls]["speed"].mean() if HAS_PD else 0
        print(f"    {cls:<12} {cnt:>8,}  ({pct:5.1f}%)  "
              f"avg_speed={speeds_c:.1f} m/s")
    print("="*55 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="custom_tracking_system/data/trajectories_hanoi",
                        help="Trajectory output directory")
    args = parser.parse_args()

    traj_dir = Path(args.dir)
    if not traj_dir.exists():
        print(f"Directory not found: {traj_dir}")
        return

    csv_files = list(traj_dir.glob("episode_[0-9]*.csv"))
    non_goal = [f for f in csv_files if "_goals" not in f.name]
    if not non_goal:
        print(f"No episode CSV files found in {traj_dir}")
        return

    print(f"Loading {len(non_goal)} episode files...")
    df = load_all_episodes(traj_dir)
    if df is None:
        print("No data loaded.")
        return

    if not HAS_MPL:
        print("matplotlib not installed — skipping plots")
        print_summary(df, traj_dir)
        return

    out_dir = traj_dir / "preview"
    out_dir.mkdir(exist_ok=True)

    print_summary(df, traj_dir)

    print("Generating visualizations...")
    plot_trajectory_map(df,            out_dir / "01_trajectory_map.png")
    plot_class_distribution(df,        out_dir / "02_class_distribution.png")
    plot_speed_histogram(df,           out_dir / "03_speed_histogram.png")
    plot_phase_breakdown(df,           out_dir / "04_phase_breakdown.png")
    plot_goals_analysis(traj_dir,      out_dir / "05_goals_analysis.png")
    plot_episode_stats(df, traj_dir,   out_dir / "06_episode_stats.png")

    print(f"\nAll previews saved to: {out_dir}")


if __name__ == "__main__":
    main()
