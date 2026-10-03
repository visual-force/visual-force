"""
scripts/analyze_dataset.py  --  Print dataset statistics.

Usage:
    python scripts/analyze_dataset.py --data-dir data/test_data_4_21_1
"""

import argparse
import csv
import os

import numpy as np


FORCE_KEYS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyze_episode(ep_dir, trim_seconds=2.0):
    force_path    = os.path.join(ep_dir, "force_timestamps.csv")
    frame_path    = os.path.join(ep_dir, "frame_timestamps.csv")
    has_mask      = os.path.isfile(os.path.join(ep_dir, "mask.mp4"))

    force_rows = load_csv(force_path)
    frame_rows = load_csv(frame_path)
    if not force_rows or not frame_rows:
        raise ValueError("force and frame timestamp CSVs must contain data rows")

    force_t = np.array([float(r["t_rel_s"]) for r in force_rows])
    frame_t = np.array([float(r["t_rel_s"]) for r in frame_rows])

    duration = frame_t.max() if len(frame_t) > 0 else 0.0
    t_lo, t_hi = trim_seconds, duration - trim_seconds
    keep_mask_force = (force_t >= t_lo) & (force_t <= t_hi)
    keep_mask_frame = (frame_t >= t_lo) & (frame_t <= t_hi)

    n_frames_total  = len(frame_rows)
    n_frames_kept   = int(keep_mask_frame.sum())
    n_force_total   = len(force_rows)
    n_force_kept    = int(keep_mask_force.sum())

    fps = n_frames_total / duration if duration > 0 else 0.0
    force_hz = n_force_total / duration if duration > 0 else 0.0

    force_data = {}
    for k in FORCE_KEYS:
        vals = np.array([float(r[k]) for r in force_rows])
        vals_trimmed = vals[keep_mask_force]
        force_data[k] = vals_trimmed
    if not len(force_data[FORCE_KEYS[0]]):
        raise ValueError("no force samples remain after trimming")

    return {
        "duration":       duration,
        "fps":            fps,
        "force_hz":       force_hz,
        "n_frames_total": n_frames_total,
        "n_frames_kept":  n_frames_kept,
        "n_force_total":  n_force_total,
        "n_force_kept":   n_force_kept,
        "has_mask":       has_mask,
        "force":          force_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/test_data_4_21_1")
    parser.add_argument("--trim-seconds", type=float, default=2.0)
    parser.add_argument(
        "--force-key",
        choices=FORCE_KEYS,
        default="Fz",
        help="Primary force key to highlight",
    )
    args = parser.parse_args()
    if args.trim_seconds < 0:
        parser.error("--trim-seconds must be non-negative")

    ep_dirs = sorted(
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if d.startswith("EP") and os.path.isdir(os.path.join(args.data_dir, d))
    )
    if not ep_dirs:
        print(f"No EP* folders found under {args.data_dir}")
        return

    print(f"Dataset: {args.data_dir}")
    print(f"Trim: {args.trim_seconds}s each side")
    print(f"Episodes found: {len(ep_dirs)}")
    print("=" * 70)

    all_force = {k: [] for k in FORCE_KEYS}
    target_sequences = []
    total_frames = 0
    analyzed_episodes = 0

    for ep_dir in ep_dirs:
        ep_name = os.path.basename(ep_dir)
        try:
            s = analyze_episode(ep_dir, args.trim_seconds)
        except Exception as e:
            print(f"{ep_name}: ERROR - {e}")
            continue

        fz = s["force"][args.force_key]
        print(f"{ep_name}  dur={s['duration']:.1f}s  "
              f"fps={s['fps']:.1f}  force_hz={s['force_hz']:.1f}  "
              f"frames={s['n_frames_kept']}/{s['n_frames_total']}  "
              f"mask={'yes' if s['has_mask'] else 'NO'}")
        print(f"  {args.force_key}: mean={fz.mean():.3f}  std={fz.std():.3f}  "
              f"min={fz.min():.3f}  max={fz.max():.3f}  range={fz.max()-fz.min():.3f} N")

        for k in FORCE_KEYS:
            all_force[k].extend(s["force"][k].tolist())
        target_sequences.append(fz)
        total_frames += s["n_frames_kept"]
        analyzed_episodes += 1

    if analyzed_episodes == 0:
        raise RuntimeError("No episodes could be analyzed")

    # Global stats
    print("=" * 70)
    print(
        f"GLOBAL  total_frames={total_frames}  "
        f"episodes={analyzed_episodes}/{len(ep_dirs)} analyzed"
    )
    print()

    target = np.array(all_force[args.force_key])
    print(f"--- {args.force_key} (prediction target) ---")
    print(f"  mean       = {target.mean():.4f} N")
    print(f"  std        = {target.std():.4f} N")
    print(f"  min        = {target.min():.4f} N")
    print(f"  max        = {target.max():.4f} N")
    print(f"  range      = {target.max() - target.min():.4f} N")
    print(f"  median     = {np.median(target):.4f} N")
    print(f"  p10 / p90  = {np.percentile(target,10):.4f} / {np.percentile(target,90):.4f} N")

    # Naive-constant baseline: predict the mean every time
    naive_mae = np.abs(target - target.mean()).mean()
    naive_mse = ((target - target.mean()) ** 2).mean()
    print()
    print(f"--- Naive baseline (always predict mean={target.mean():.3f} N) ---")
    print(f"  MAE = {naive_mae:.4f} N   (if your val_MAE > this, model is worse than a constant)")
    print(f"  MSE = {naive_mse:.4f} N²")
    print()

    # Autocorrelation lag-1 (temporal smoothness)
    lagged = [values for values in target_sequences if len(values) > 1]
    if lagged:
        lag_x = np.concatenate([values[:-1] for values in lagged])
        lag_y = np.concatenate([values[1:] for values in lagged])
        if lag_x.std() > 0 and lag_y.std() > 0:
            ac = np.corrcoef(lag_x, lag_y)[0, 1]
        else:
            ac = float("nan")
    else:
        ac = float("nan")
    print(f"--- Temporal structure ---")
    if np.isfinite(ac):
        smoothness = "very smooth" if ac > 0.99 else "smooth" if ac > 0.95 else "moderate"
        print(f"  lag-1 autocorrelation of {args.force_key} = {ac:.4f}  ({smoothness})")
    else:
        print(f"  lag-1 autocorrelation of {args.force_key} = unavailable")
    print()

    print(f"--- All force/moment channels (global, trimmed) ---")
    header = f"{'key':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'range':>8}"
    print(f"  {header}")
    for k in FORCE_KEYS:
        v = np.array(all_force[k])
        print(f"  {k:>4}  {v.mean():>8.3f}  {v.std():>8.3f}  "
              f"{v.min():>8.3f}  {v.max():>8.3f}  {v.max()-v.min():>8.3f}")


if __name__ == "__main__":
    main()
