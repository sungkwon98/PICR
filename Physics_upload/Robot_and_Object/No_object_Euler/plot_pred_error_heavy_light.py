"""Mean N-step prediction-error curves for the No-Object-Euler world model.

Companion to ``animate_episode.py``.  Where the animation overlays, frame
by frame, the model's ``N``-step rollout against the recorded ground
truth, this script *quantifies* that gap: it rolls the model out for
``--pred_horizon`` steps from every valid start frame of every episode
in the **heavy** and **light** context datasets, measures the Euclidean
gripper-tip and cube-center position error at each rollout horizon
``h = 1..N``, and plots the error averaged over all those rollouts.

Three curves are drawn per quantity: the ``heavy`` mean, the ``light``
mean, and the pooled ``heavy+light`` mean (bold, with a +/-1 std band).

The rollout reuses ``animate_episode`` verbatim (same checkpoint loader,
same per-frame ``precompute_pred_trajectories``).  This folder's
``precompute_pred_trajectories`` also returns per-frame ``F_c`` /
``alpha`` / ``rho_consts`` -- those are ignored here and the same first
two return values (predicted gripper and cube trajectories) are
consumed exactly like in the MCGDF sibling script.

Usage::

    python plot_pred_error_heavy_light.py
    python plot_pred_error_heavy_light.py --pred_horizon 15 --max_episodes 10
    python plot_pred_error_heavy_light.py --filter_steps 3
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import FrankaForwardKinematics  # noqa: E402
from animate_episode import (  # noqa: E402
    compute_joint_positions,
    list_episode_names,
    load_checkpoint_model,
    load_episode,
    load_episode_extras,
    precompute_pred_trajectories,
)

# Heavy/light datasets live under the Physics_jj tree.  The
# No-Object-Euler model itself is loaded from this folder's local
# ``outputs_mcgdf/`` (the trainer reuses that output dir name).
_PHYSICS_JJ = "/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics_jj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./outputs_mcgdf/run_20260608_171124/best.pt",
        help="Path to the trained No-Object-Euler checkpoint (best.pt/last.pt).",
    )
    parser.add_argument(
        "--heavy_dataset", type=str,
        default=f"{_PHYSICS_JJ}/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_heavy_context_10ep.hdf5",
    )
    parser.add_argument(
        "--light_dataset", type=str,
        default=f"{_PHYSICS_JJ}/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep.hdf5",
    )
    parser.add_argument(
        "--output", type=str,
        default="./eval_outputs/no_object_euler_pred_error_heavy_light.png",
        help="Output PNG path for the averaged error figure.",
    )
    parser.add_argument(
        "--pred_horizon", type=int, default=10,
        help="Number of rollout steps N over which the error is measured.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=0,
        help="Cap episodes per dataset (0 = all).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument(
        "--torque_key", type=str, default="applied_torque",
        choices=["applied_torque", "computed_torque"],
    )
    parser.add_argument(
        "--friction_key", type=str, default="joint_dynamic_friction_coeff",
        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"],
    )
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin",
        action="store_false", default=True,
        help="Match the training default: subtract env_origin from object position.",
    )
    parser.add_argument("--deterministic_context", action="store_true", default=True)
    parser.add_argument(
        "--sample_context", dest="deterministic_context", action="store_false",
    )
    parser.add_argument(
        "--filter_steps", type=int, default=-1,
        help="Closed-loop filter window length for the online latent "
             "update before each per-frame rollout.  -1 (default) means "
             "'use the value baked into the checkpoint config'; 0 "
             "disables the filter; >0 overrides.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _episode_args(args: argparse.Namespace, dataset_file: str, ep_name: str) -> argparse.Namespace:
    """Build the lightweight Namespace expected by the animate_episode loaders."""
    return argparse.Namespace(
        dataset_file=dataset_file,
        episode_index=0,
        episode_name=ep_name,
        robot_dof=args.robot_dof,
        subtract_env_origin=args.subtract_env_origin,
        max_frames=0,
        torque_key=args.torque_key,
        friction_key=args.friction_key,
        tool_z_offset=args.tool_z_offset,
    )


def accumulate_errors_for_dataset(
    dataset_file: str,
    args: argparse.Namespace,
    model,
    history_len: int,
    eff_filter_steps: int,
    device: torch.device,
    fk: FrankaForwardKinematics,
) -> tuple[list[list[float]], list[list[float]], int]:
    """Roll the model out across every episode and bin errors by horizon.

    Returns ``(gripper_err_by_h, cube_err_by_h, n_episodes)`` where each
    list is indexed by horizon ``h = 0..N-1`` and holds every per-rollout
    error observed at that horizon (pooled over all start frames and
    episodes).
    """
    n = args.pred_horizon
    gripper_err_by_h: list[list[float]] = [[] for _ in range(n)]
    cube_err_by_h: list[list[float]] = [[] for _ in range(n)]

    if not os.path.isfile(dataset_file):
        print(f"[WARN] dataset not found, skipping: {dataset_file}")
        return gripper_err_by_h, cube_err_by_h, 0

    names = list_episode_names(dataset_file)
    if args.max_episodes > 0:
        names = names[: args.max_episodes]

    for ep_name in names:
        ep_args = _episode_args(args, dataset_file, ep_name)
        episode = load_episode(ep_args)
        extras = load_episode_extras(ep_args, episode["name"])

        # Recorded ground-truth Cartesian targets for every frame.
        gt_gripper_all = compute_joint_positions(episode["joint_pos"], fk, device)[:, -1, :]
        gt_cube_all = episode["object_pos"]
        T_gt = min(gt_gripper_all.shape[0], gt_cube_all.shape[0])

        # No-Object-Euler's precompute returns a 5-tuple:
        # (pred_gripper, pred_cube, fc_per_t, alpha_per_t, rho_consts).
        # Only the first two are needed for the error curves.
        (
            pred_gripper_per_t,
            pred_cube_per_t,
            *_rest,
        ) = precompute_pred_trajectories(
            model=model, episode=episode, extras=extras,
            history_len=history_len, pred_horizon=n, device=device,
            deterministic_context=args.deterministic_context,
            fk=fk, robot_dof=args.robot_dof,
            filter_steps=eff_filter_steps,
        )

        for t, (g, c) in enumerate(zip(pred_gripper_per_t, pred_cube_per_t)):
            if g is None or c is None:
                continue
            steps = min(g.shape[0], c.shape[0])
            for h in range(steps):
                fidx = t + 1 + h
                if fidx >= T_gt:
                    break
                eg = float(np.linalg.norm(g[h] - gt_gripper_all[fidx]))
                ec = float(np.linalg.norm(c[h] - gt_cube_all[fidx]))
                if np.isfinite(eg):
                    gripper_err_by_h[h].append(eg)
                if np.isfinite(ec):
                    cube_err_by_h[h].append(ec)

    return gripper_err_by_h, cube_err_by_h, len(names)


def _mean_std(err_by_h: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-horizon mean, std and sample count (NaN where a horizon is empty)."""
    n = len(err_by_h)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for h, vals in enumerate(err_by_h):
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            mean[h] = arr.mean()
            std[h] = arr.std()
            count[h] = arr.size
    return mean, std, count


def _pool(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [a[h] + b[h] for h in range(len(a))]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, cfg = load_checkpoint_model(ckpt_path, device)
    history_len = int(cfg.get("history_len", 5))

    # Resolve effective filter window.  -1 → use the checkpoint's value;
    # otherwise the CLI override wins.
    if args.filter_steps < 0:
        eff_filter_steps = int(getattr(model, "filter_steps", 0))
    else:
        eff_filter_steps = int(args.filter_steps)

    fk = FrankaForwardKinematics(
        robot_dof=args.robot_dof, tool_z_offset=args.tool_z_offset,
    ).to(device)
    fk.eval()

    print("===== No-Object-Euler mean prediction-error (heavy/light averaged) =====")
    print(f"checkpoint:   {ckpt_path}")
    print(
        f"pred_horizon: {args.pred_horizon}   history_len: {history_len}   "
        f"filter_steps: {eff_filter_steps}"
    )

    heavy_g, heavy_c, n_heavy = accumulate_errors_for_dataset(
        args.heavy_dataset, args, model, history_len, eff_filter_steps, device, fk,
    )
    light_g, light_c, n_light = accumulate_errors_for_dataset(
        args.light_dataset, args, model, history_len, eff_filter_steps, device, fk,
    )
    comb_g = _pool(heavy_g, light_g)
    comb_c = _pool(heavy_c, light_c)

    h_axis = np.arange(1, args.pred_horizon + 1)
    series = {
        "heavy": ("tab:red", heavy_g, heavy_c),
        "light": ("tab:blue", light_g, light_c),
        "heavy+light": ("black", comb_g, comb_c),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["Gripper-tip position error", "Cube-center position error"]
    for col, (ax, title) in enumerate(zip(axes, titles)):
        for label, (color, g_by_h, c_by_h) in series.items():
            err_by_h = g_by_h if col == 0 else c_by_h
            mean, std, _count = _mean_std(err_by_h)
            bold = label == "heavy+light"
            ax.plot(
                h_axis, mean * 1000.0, color=color,
                linewidth=2.6 if bold else 1.6,
                linestyle="-" if bold else "--",
                marker="o", markersize=4, label=label,
            )
            if bold:
                lo = np.clip(mean - std, 0.0, None) * 1000.0
                hi = (mean + std) * 1000.0
                ax.fill_between(h_axis, lo, hi, color=color, alpha=0.12)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("rollout horizon $h$ (steps ahead)")
        ax.set_ylabel("mean position error [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(
        f"No-Object-Euler open-loop {args.pred_horizon}-step prediction error  "
        f"(heavy n={n_heavy}, light n={n_light} episodes; "
        f"{'deterministic' if args.deterministic_context else 'sampled'} z; "
        f"filter_steps={eff_filter_steps})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    # Console summary at every horizon.
    cg_mean, _, cg_cnt = _mean_std(comb_g)
    cc_mean, _, _ = _mean_std(comb_c)
    print("\nhorizon |  gripper [mm] |  cube [mm] |  #samples")
    for h in range(args.pred_horizon):
        gm = cg_mean[h] * 1000.0 if np.isfinite(cg_mean[h]) else float("nan")
        cm = cc_mean[h] * 1000.0 if np.isfinite(cc_mean[h]) else float("nan")
        print(f"  {h + 1:>4}  |  {gm:>11.3f}  |  {cm:>8.3f}  |  {int(cg_cnt[h])}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
