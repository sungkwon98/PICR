"""Mean N-step prediction-error curves for the Context DeLaN world model.

Companion to ``animate_episode.py``.  Mirrors
``Physics_jj/Robot_and_Object/MCGDF/plot_pred_error_heavy_light.py`` but
adapted to the Context DeLaN model's API: the model has no per-step
``damping`` / ``friction`` / ``object_context`` inputs, the latent ``z``
is encoded once from a history window (no closed-loop filter), and the
checkpoint loader / FK chain come from this folder's ``eval_utils``.

Where the animation overlays, frame by frame, the model's ``N``-step rollout
against the recorded ground truth, this script *quantifies* that gap: it
rolls the model out for ``--pred_horizon`` steps from every valid start
frame of every episode in the **heavy** and **light** context datasets,
measures the Euclidean gripper-tip and cube-center position error at each
rollout horizon ``h = 1..N``, and plots the error averaged over all those
rollouts.

Three curves are drawn per quantity: the ``heavy`` mean, the ``light`` mean,
and the pooled ``heavy+light`` mean (bold, with a +/-1 std band).

The rollout reuses ``animate_episode`` verbatim (same checkpoint loader,
same per-frame ``precompute_pred_trajectories``) so the numbers here are
exactly the gap you see in the animation, only aggregated.

Usage::

    python plot_pred_error_heavy_light.py
    python plot_pred_error_heavy_light.py --pred_horizon 15 --max_episodes 10
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_utils import (  # noqa: E402
    episode_names,
    load_episode_arrays,
    load_episode_robot_joint_positions,
)
from animate_episode import (  # noqa: E402
    compute_joint_positions,
    precompute_pred_trajectories,
)
# load_checkpoint_model is re-exported into animate_episode via its
# ``from eval_utils import ...`` block, but importing it directly is
# cleaner because that name is what callers expect.
from eval_utils import load_checkpoint_model  # noqa: E402


_PHYSICS_JJ = "/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics_jj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./outputs_context_delan/run_20260514_092259/best.pt",
        help="Path to the trained Context DeLaN checkpoint (best.pt/last.pt).",
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
        default="./eval_outputs/context_delan_pred_error_heavy_light.png",
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
        "--torque_key", type=str, default=None,
        choices=[None, "applied_torque", "computed_torque"],
        help="Torque field used as the model's per-step input.  Default: "
             "read from the checkpoint config.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def accumulate_errors_for_dataset(
    dataset_file: str,
    args: argparse.Namespace,
    model,
    layout,
    cfg: dict,
    history_len: int,
    device: torch.device,
) -> tuple[list[list[float]], list[list[float]], int]:
    """Roll the model across every episode and bin errors by horizon.

    Returns ``(gripper_err_by_h, cube_err_by_h, n_episodes)`` where each
    list is indexed by horizon ``h = 0..N-1`` and holds every per-rollout
    error observed at that horizon (pooled over all start frames and
    episodes).

    Frame conventions: gripper positions are in the robot-base frame
    (numpy FK on absolute joint angles).  Cube positions are taken from
    the *state-vector* slice, which is the same frame the model predicts
    in (``load_episode_arrays`` does not subtract the env-origin).  Both
    ``pred`` and ``gt`` therefore live in the same frame for each
    quantity, so the error magnitudes are meaningful.
    """
    n = args.pred_horizon
    gripper_err_by_h: list[list[float]] = [[] for _ in range(n)]
    cube_err_by_h: list[list[float]] = [[] for _ in range(n)]

    if not os.path.isfile(dataset_file):
        print(f"[WARN] dataset not found, skipping: {dataset_file}")
        return gripper_err_by_h, cube_err_by_h, 0

    names = episode_names(dataset_file)
    if args.max_episodes > 0:
        names = names[: args.max_episodes]

    torque_key = args.torque_key or str(cfg["torque_key"])
    robot_state_dim = layout.robot_state_dim

    for ep_name in names:
        states, torques, _phys_ctx, _robot_dyn, _obj_dyn = load_episode_arrays(
            dataset_file, ep_name, layout, torque_key,
        )
        absolute_joint_pos = load_episode_robot_joint_positions(
            dataset_file, ep_name, layout.robot_dof,
        )
        T_min = min(states.shape[0], torques.shape[0], absolute_joint_pos.shape[0])
        states = states[:T_min]
        torques = torques[:T_min]
        absolute_joint_pos = absolute_joint_pos[:T_min]

        # Ground-truth Cartesian targets at every frame:
        # gripper -> FK on absolute joint angles (robot-base frame),
        # cube    -> state-vector slice (matches model prediction frame).
        joints_3d = compute_joint_positions(absolute_joint_pos, args.tool_z_offset)
        gt_gripper_all = joints_3d[:, -1, :]  # (T, 3)
        gt_cube_all = states[:, robot_state_dim : robot_state_dim + 3]  # (T, 3)

        (
            pred_gripper_per_t,
            pred_cube_per_t,
            *_rest,
        ) = precompute_pred_trajectories(
            model=model, layout=layout,
            states=states, torques=torques, joint_pos_abs=absolute_joint_pos,
            pred_horizon=n, history_len=history_len,
            device=device, tool_z_offset=args.tool_z_offset,
        )

        T_gt = min(gt_gripper_all.shape[0], gt_cube_all.shape[0])
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
    model, cfg, layout, ckpt_path = load_checkpoint_model(ckpt_path, device)
    history_len = int(cfg["history_len"])

    print("===== Context DeLaN mean prediction-error (heavy/light averaged) =====")
    print(f"checkpoint:    {ckpt_path}")
    print(
        f"context_enc:   {cfg.get('context_encoder', '?')}   "
        f"latent_dim: {cfg.get('latent_dim', '?')}"
    )
    print(f"pred_horizon:  {args.pred_horizon}   history_len: {history_len}")

    heavy_g, heavy_c, n_heavy = accumulate_errors_for_dataset(
        args.heavy_dataset, args, model, layout, cfg, history_len, device,
    )
    light_g, light_c, n_light = accumulate_errors_for_dataset(
        args.light_dataset, args, model, layout, cfg, history_len, device,
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
        f"Context DeLaN open-loop {args.pred_horizon}-step prediction error  "
        f"(heavy n={n_heavy}, light n={n_light} episodes; "
        f"encoder={cfg.get('context_encoder', '?')}, "
        f"latent_dim={cfg.get('latent_dim', '?')})",
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
