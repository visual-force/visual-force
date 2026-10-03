"""
Train an action-conditioned delta-force model on flip data.

This is a deterministic supervised critic:
    wrist obs history + candidate delta-action trajectory -> delta-force trajectory

It is separate from the original VisualForce image-only estimator and does not
change old checkpoints or policy-server behavior.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.flip_delta_force_dataset import (
    FlipDeltaForceDataset,
    split_train_val_by_episode,
    standardize,
    unstandardize,
)
from src.model.action_delta_force import ActionConditionedDeltaForceNet
from src.training_utils import (
    batch_to_device,
    fit_standardizers,
    load_wandb,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--pseudo-label-name", default="visualforce_pseudo_force_fz.npz")
    parser.add_argument("--force-key", default="Fz")
    parser.add_argument("--force-mode", choices=["magnitude", "signed"], default="magnitude")
    parser.add_argument(
        "--action-mode",
        choices=["obs_delta", "obs_delta_pos_gripper"],
        default="obs_delta_pos_gripper",
        help=(
            "obs_delta subtracts the full 8D agent state from action. "
            "obs_delta_pos_gripper subtracts xyz and gripper, preserving quat action."
        ),
    )
    parser.add_argument("--obs-steps", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--image-size", nargs=2, type=int, default=[240, 320])
    parser.add_argument(
        "--image-normalization",
        choices=["zero_one", "minus_one_one"],
        default="zero_one",
        help="minus_one_one matches the DP image normalizer.",
    )
    parser.add_argument(
        "--critic-image-mode",
        choices=["rgb", "edge", "none"],
        default="rgb",
        help=(
            "Observation image input for the force critic. edge uses precomputed "
            "masked Sobel frames; none trains an action-only critic."
        ),
    )
    parser.add_argument(
        "--edge-obs-name",
        default="visualforce_edge_obs.npz",
        help="Per-episode edge observation npz used when --critic-image-mode=edge.",
    )
    parser.add_argument(
        "--include-agent-pos",
        action="store_true",
        help="Condition the critic on the same low-dimensional agent_pos history as DP.",
    )
    parser.add_argument(
        "--include-current-force",
        action="store_true",
        help="Condition the critic on the current ForceLens pseudo-force scalar.",
    )
    parser.add_argument(
        "--target-mode",
        choices=[
            "delta_trajectory",
            "final_delta",
            "max_delta",
            "mean_delta",
            "future_peak",
        ],
        default="delta_trajectory",
        help=(
            "delta_trajectory predicts per-step force deltas. Chunk-level modes "
            "predict one scalar for the whole action chunk."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cum-loss-weight", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/flip_delta_force")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="force_estimation")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.obs_steps <= 0 or args.pred_horizon <= 0:
        parser.error("--obs-steps and --pred-horizon must be positive")
    if any(size <= 0 for size in args.image_size):
        parser.error("--image-size values must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0 or args.cum_loss_weight < 0:
        parser.error("--weight-decay and --cum-loss-weight must be non-negative")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")
    if args.save_every < 0:
        parser.error("--save-every must be non-negative")
    return args


def make_low_dim(
    batch: dict[str, torch.Tensor],
    agent_pos_stats=None,
    current_force_stats=None,
):
    parts = []
    if agent_pos_stats is not None:
        parts.append(standardize(batch["agent_pos"], agent_pos_stats).flatten(start_dim=1))
    if current_force_stats is not None:
        parts.append(standardize(batch["current_force"], current_force_stats))
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def evaluate(
    model,
    loader,
    action_stats,
    target_stats,
    device,
    cum_loss_weight,
    agent_pos_stats=None,
    current_force_stats=None,
    target_mode="delta_trajectory",
):
    model.eval()
    loss_sum = 0.0
    step_abs_sum = 0.0
    cum_abs_sum = 0.0
    n_steps = 0
    n_cum = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch_to_device(batch, device)
            action = standardize(batch["action_delta"], action_stats)
            low_dim = make_low_dim(
                batch,
                agent_pos_stats=agent_pos_stats,
                current_force_stats=current_force_stats,
            )
            target = batch["target_delta_force"]
            target_n = standardize(target, target_stats)
            pred_n = model(batch["image"], action, low_dim)
            pred = unstandardize(pred_n, target_stats)
            step_loss = F.mse_loss(pred_n, target_n)
            if target_mode == "delta_trajectory":
                cum_loss = F.mse_loss(
                    pred_n.cumsum(dim=1),
                    target_n.cumsum(dim=1),
                )
            else:
                cum_loss = torch.zeros((), device=device)
            loss = step_loss + cum_loss_weight * cum_loss
            loss_sum += float(loss.item()) * len(target)
            step_abs_sum += (pred - target).abs().sum().item()
            cum_abs_sum += (
                pred.cumsum(dim=1) - target.cumsum(dim=1)
            ).abs().sum().item()
            n_steps += target.numel()
            n_cum += target.numel()
    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "step_mae": step_abs_sum / max(1, n_steps),
        "cum_mae": cum_abs_sum / max(1, n_cum),
    }


def main():
    args = parse_args()
    wandb = load_wandb(args.no_wandb)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = FlipDeltaForceDataset(
        args.dataset_path,
        pseudo_label_name=args.pseudo_label_name,
        obs_steps=args.obs_steps,
        pred_horizon=args.pred_horizon,
        image_size=tuple(args.image_size),
        force_key=args.force_key,
        force_mode=args.force_mode,
        action_mode=args.action_mode,
        image_normalization=args.image_normalization,
        critic_image_mode=args.critic_image_mode,
        edge_obs_name=args.edge_obs_name,
        target_mode=args.target_mode,
    )
    if len(dataset.episodes) < 2:
        raise RuntimeError("At least two episodes are required for an episode-level split")
    train_set, val_set = split_train_val_by_episode(
        dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("Training and validation splits must both contain samples")
    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": use_cuda,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
    )
    stats_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    stat_keys = ["action_delta", "target_delta_force"]
    if args.include_agent_pos:
        stat_keys.append("agent_pos")
    if args.include_current_force:
        stat_keys.append("current_force")
    fitted_stats = fit_standardizers(stats_loader, stat_keys)
    action_stats = fitted_stats["action_delta"]
    target_stats = fitted_stats["target_delta_force"]
    agent_pos_stats = fitted_stats.get("agent_pos")
    current_force_stats = fitted_stats.get("current_force")
    low_dim_dim = 0
    if args.include_agent_pos:
        low_dim_dim += dataset.low_dim_dim
    if args.include_current_force:
        low_dim_dim += 1

    model = ActionConditionedDeltaForceNet(
        image_channels=dataset.image_channels,
        action_dim=dataset.action_dim,
        pred_horizon=args.pred_horizon,
        force_dim=dataset.force_dim,
        low_dim_dim=low_dim_dim,
        output_horizon=dataset.target_horizon,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Dataset: {len(dataset)} samples, train={len(train_set)}, val={len(val_set)}, "
        f"episodes={len(dataset.episodes)}"
    )
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}")

    use_wandb = wandb is not None
    if use_wandb:
        run_name = args.wandb_run or f"flip_delta_force_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.run.summary["n_params"] = n_params

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    periodic = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch}", leave=False, unit="batch"):
            batch = batch_to_device(batch, device)
            action = standardize(batch["action_delta"], action_stats)
            low_dim = make_low_dim(
                batch,
                agent_pos_stats=agent_pos_stats,
                current_force_stats=current_force_stats,
            )
            target = batch["target_delta_force"]
            target_n = standardize(target, target_stats)

            optimizer.zero_grad(set_to_none=True)
            pred_n = model(batch["image"], action, low_dim)
            step_loss = F.mse_loss(pred_n, target_n)
            if args.target_mode == "delta_trajectory":
                cum_loss = F.mse_loss(
                    pred_n.cumsum(dim=1),
                    target_n.cumsum(dim=1),
                )
            else:
                cum_loss = torch.zeros((), device=device)
            loss = step_loss + args.cum_loss_weight * cum_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)

        scheduler.step()
        train_loss = loss_sum / max(1, len(train_set))
        val = evaluate(
            model,
            val_loader,
            action_stats,
            target_stats,
            device,
            args.cum_loss_weight,
            agent_pos_stats=agent_pos_stats,
            current_force_stats=current_force_stats,
            target_mode=args.target_mode,
        )
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:04d}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={val['loss']:.5f} "
            f"step_MAE={val['step_mae']:.4f}N cum_MAE={val['cum_mae']:.4f}N "
            f"lr={lr_now:.2e}"
        )
        if use_wandb:
            wandb.log(
                {
                    "metrics/train_loss": train_loss,
                    "metrics/val_loss": val["loss"],
                    "metrics/val_step_mae": val["step_mae"],
                    "metrics/val_cum_mae": val["cum_mae"],
                    "lr": lr_now,
                },
                step=epoch,
            )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "action_stats": action_stats,
            "target_stats": target_stats,
            "agent_pos_stats": agent_pos_stats,
            "current_force_stats": current_force_stats,
            "image_channels": dataset.image_channels,
            "action_dim": dataset.action_dim,
            "force_dim": dataset.force_dim,
            "low_dim_dim": low_dim_dim,
            "output_horizon": dataset.target_horizon,
            "train_loss": train_loss,
            "val": val,
        }
        torch.save(payload, out_dir / "latest.pt")
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(payload, out_dir / "best.pt")
            print(f"  -> saved best checkpoint (val_loss={best_val:.5f})")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val
                wandb.run.summary["best_epoch"] = epoch

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = out_dir / f"ep{epoch:04d}_valloss{val['loss']:.5f}.pt"
            torch.save(payload, ckpt_path)
            periodic.append(ckpt_path)
            while len(periodic) > 3:
                old = periodic.pop(0)
                if old.exists():
                    old.unlink()

    if use_wandb:
        wandb.finish()
    print(f"Training complete. Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
