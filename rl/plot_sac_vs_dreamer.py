"""Compare training efficiency of SAC and Dreamer.

Parses SAC text log (`train_sac.log`) and Dreamer JSONL metrics
(`metrics.jsonl`), then produces two plots:

1. Episode reward as a function of environment steps (training curve).
2. Sample-efficiency curve: minimum steps required to first reach a
   given reward threshold, for both algorithms.

Usage (from repo root):
    python rl/plot_sac_vs_dreamer.py
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
SAC_LINE_RE = re.compile(
    r"Episode:\s*(\d+)\s*\|\s*Episode Reward:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def parse_sac_log(path, steps_per_episode):
    """Return (steps, rewards) arrays parsed from a SAC text log."""
    episodes, rewards = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SAC_LINE_RE.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
    episodes = np.asarray(episodes, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    # SAC log has no per-step counter, so we approximate environment steps
    # using the (fixed) episode length from task.json.
    steps = (episodes + 1) * steps_per_episode
    return steps, rewards


def parse_dreamer_jsonl(path):
    """Return (steps, rewards) for Dreamer using `train_return` records."""
    steps, rewards = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train_return" in rec and "step" in rec:
                steps.append(int(rec["step"]))
                rewards.append(float(rec["train_return"]))
    return np.asarray(steps), np.asarray(rewards, dtype=np.float64)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def moving_average(x, window):
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    # 'same' keeps length identical so it stays aligned with the step axis.
    return np.convolve(x, kernel, mode="same")


# ---------------------------------------------------------------------------
# Sample efficiency
# ---------------------------------------------------------------------------
def steps_to_reach(steps, rewards_smooth, thresholds):
    """For each threshold, return the first step where smoothed reward >= it.

    Returns NaN where the threshold is never reached.
    """
    out = np.full_like(thresholds, np.nan, dtype=np.float64)
    for i, thr in enumerate(thresholds):
        mask = rewards_smooth >= thr
        if mask.any():
            out[i] = float(steps[np.argmax(mask)])  # argmax on bool -> first True
    return out


def plot_sample_efficiency(sac_steps, sac_rewards,
                           dr_steps, dr_rewards,
                           out_path, smooth=20, n_points=80):
    """Plot threshold vs first-reached step for SAC and Dreamer.

    Uses *smoothed* episode rewards (moving-average window = ``smooth``) so
    that a single lucky episode does not declare the threshold reached.
    The question we answer is: "how many env steps until the algorithm
    *robustly* attains this reward level".
    """
    sac_smooth = moving_average(sac_rewards, smooth)
    dr_smooth  = moving_average(dr_rewards,  smooth)

    # Use the overlap of the two smoothed reward ranges so the comparison is fair.
    lo = max(sac_smooth.min(), dr_smooth.min())
    hi = min(sac_smooth.max(), dr_smooth.max())
    if hi <= lo:
        print("[warn] reward ranges do not overlap; skip efficiency plot")
        return
    thresholds = np.linspace(lo, hi, n_points)

    sac_first = steps_to_reach(sac_steps, sac_smooth, thresholds)
    dr_first  = steps_to_reach(dr_steps,  dr_smooth,  thresholds)

    # Sample efficiency: fewer steps to reach the same reward = better.
    # Speedup = SAC_steps / Dreamer_steps:
    #   > 1  -> Dreamer needs fewer steps (Dreamer is faster)
    #   = 1  -> tie
    #   < 1  -> SAC needs fewer steps (SAC is faster)
    with np.errstate(divide="ignore", invalid="ignore"):
        speedup = sac_first / dr_first
    valid = np.isfinite(speedup) & (dr_first > 0) & (sac_first > 0)
    mean_sp   = float(np.nanmean(speedup[valid])) if valid.any() else float("nan")
    median_sp = float(np.nanmedian(speedup[valid])) if valid.any() else float("nan")
    max_sp    = float(np.nanmax(speedup[valid]))    if valid.any() else float("nan")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- Top: steps-to-threshold curves ------------------------------------
    ax1.plot(thresholds, sac_first, color="tab:blue",   lw=2.0, marker="o",
             ms=3, label="SAC")
    ax1.plot(thresholds, dr_first,  color="tab:orange", lw=2.0, marker="s",
             ms=3, label="Dreamer")
    ax1.set_ylabel("Min. env. steps to first reach threshold\n(lower is better)")
    ax1.set_title(f"Sample efficiency: steps to reach reward level "
                  f"(smoothed w={smooth})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # --- Bottom: speedup bar chart -----------------------------------------
    # Bar height = how many times fewer steps Dreamer needs vs SAC at that
    # reward level. Bars > 1 mean Dreamer is faster, bars < 1 mean SAC is
    # faster (we colour them differently for readability).
    bar_w = (thresholds[1] - thresholds[0]) * 0.9 if len(thresholds) > 1 else 1.0
    sp_plot = np.where(valid, speedup, np.nan)
    bar_colors = np.where(sp_plot >= 1.0, "tab:green", "tab:red")
    ax2.bar(thresholds, sp_plot, width=bar_w,
            color=bar_colors, alpha=0.75, edgecolor="none")
    ax2.axhline(1.0, color="gray", lw=1.0, ls="--", label="tie (x1.0)")
    ax2.axhline(mean_sp,   color="black",  lw=1.2, ls="-",
                label=f"mean   x{mean_sp:.2f}")
    ax2.axhline(median_sp, color="purple", lw=1.2, ls=":",
                label=f"median x{median_sp:.2f}")
    ax2.axhline(max_sp,    color="darkgreen", lw=1.2, ls="-.",
                label=f"max    x{max_sp:.2f}")
    ax2.set_xlabel("Reward threshold")
    ax2.set_ylabel("How many times fewer steps\nDreamer needs vs SAC")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[done] efficiency figure saved to {out_path}")
    print(f"[info] Dreamer is on average x{mean_sp:.2f} faster than SAC "
          f"(median x{median_sp:.2f}, max x{max_sp:.2f}) "
          f"across {int(valid.sum())} thresholds.")

    # Print a small numeric summary at a few representative thresholds.
    sample_thrs = np.linspace(lo, hi, 6)
    sac_pick = steps_to_reach(sac_steps, sac_smooth, sample_thrs)
    dr_pick  = steps_to_reach(dr_steps,  dr_smooth,  sample_thrs)
    print("\n[summary] steps to first reach threshold (smoothed reward):")
    print(f"  {'reward':>8s} | {'SAC':>10s} | {'Dreamer':>10s} | speedup")
    for thr, s, d in zip(sample_thrs, sac_pick, dr_pick):
        sp = (s / d) if (d and not np.isnan(d) and not np.isnan(s)) else float("nan")
        s_txt = f"{int(s)}"  if not np.isnan(s) else "  n/a"
        d_txt = f"{int(d)}"  if not np.isnan(d) else "  n/a"
        sp_txt = f"x{sp:.2f}" if not np.isnan(sp) else "  n/a"
        print(f"  {thr:8.2f} | {s_txt:>10s} | {d_txt:>10s} | {sp_txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac_log",
        default="output/swimming_forward_lbs/torus/train_sac.log",
    )
    parser.add_argument(
        "--dreamer_log",
        default="output/swimming_forward_lbs/torus/metrics.jsonl",
    )
    parser.add_argument(
        "--task_json",
        default="rl/task.json",
        help="Used to read total_time / interval for SAC steps-per-episode.",
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=None,
        help="Override SAC steps-per-episode (else read from task.json).",
    )
    parser.add_argument("--smooth", type=int, default=20,
                        help="Moving-average window. 1 disables smoothing.")
    parser.add_argument(
        "--out",
        default="output/swimming_forward_lbs/torus/sac_vs_dreamer.png",
    )
    parser.add_argument(
        "--eff_out",
        default="output/swimming_forward_lbs/torus/sac_vs_dreamer_efficiency.png",
        help="Output path for the sample-efficiency plot.",
    )
    args = parser.parse_args()

    # Determine SAC steps_per_episode from task.json if not explicitly given.
    if args.steps_per_episode is None:
        with open(args.task_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        steps_per_episode = int(round(cfg["total_time"] / cfg["interval"]))
    else:
        steps_per_episode = args.steps_per_episode
    print(f"[info] SAC steps_per_episode = {steps_per_episode}")

    sac_steps, sac_rewards = parse_sac_log(args.sac_log, steps_per_episode)
    dr_steps, dr_rewards = parse_dreamer_jsonl(args.dreamer_log)
    print(f"[info] SAC     : {len(sac_rewards)} episodes, max step = {sac_steps.max() if len(sac_steps) else 0}")
    print(f"[info] Dreamer : {len(dr_rewards)} episodes, max step = {dr_steps.max() if len(dr_steps) else 0}")

    sac_smooth = moving_average(sac_rewards, args.smooth)
    dr_smooth = moving_average(dr_rewards, args.smooth)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sac_steps, sac_rewards, color="tab:blue",  alpha=0.25, lw=0.8)
    ax.plot(sac_steps, sac_smooth,  color="tab:blue",  lw=2.0,
            label=f"SAC (smoothed w={args.smooth})")
    ax.plot(dr_steps,  dr_rewards,  color="tab:orange", alpha=0.25, lw=0.8)
    ax.plot(dr_steps,  dr_smooth,   color="tab:orange", lw=2.0,
            label=f"Dreamer (smoothed w={args.smooth})")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("SAC vs Dreamer — training efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[done] figure saved to {args.out}")

    # Second plot: sample efficiency (threshold vs steps-to-reach).
    # Use *smoothed* rewards so that a single noisy episode does not bias the
    # "first time reached" measurement -- we want to know when the algorithm
    # *robustly* attains a reward level.
    plot_sample_efficiency(sac_steps, sac_rewards,
                           dr_steps,  dr_rewards,
                           args.eff_out, smooth=args.smooth)


if __name__ == "__main__":
    main()
