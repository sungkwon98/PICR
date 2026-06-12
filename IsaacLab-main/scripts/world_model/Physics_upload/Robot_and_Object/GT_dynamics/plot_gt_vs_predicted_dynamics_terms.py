from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_utils import (
    first_object_motion_timestep,
    load_checkpoint_model,
    load_episode_arrays,
    resolve_episode_name,
    rollout_stepwise,
    stack_aux,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GT vs predicted robot/object dynamics terms.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../../../../reinforcement_learning/skrl/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--start_t", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=10, help="Future rollout steps. 0 means rest of episode.")
    parser.add_argument("--torque_key", type=str, default=None, choices=[None, "applied_torque", "computed_torque"])
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--output_name", type=str, default="gt_vs_predicted_robot_object_dynamics_terms.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    return parser.parse_args()


def build_plot_terms(
    gt_future: np.ndarray,
    pred_future: np.ndarray,
    gt_torques: np.ndarray,
    gt_robot_dynamics: dict[str, np.ndarray],
    gt_object_dynamics: dict[str, np.ndarray],
    pred_aux: list[dict[str, torch.Tensor]],
    robot_dof: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    robot_state_dim = 2 * robot_dof
    gt_residual_tau = gt_robot_dynamics["inverse_dynamics_tau"] - gt_torques
    return [
        ("robot q", gt_future[:, :robot_dof], pred_future[:, :robot_dof]),
        ("robot qdot", gt_future[:, robot_dof:robot_state_dim], pred_future[:, robot_dof:robot_state_dim]),
        ("object pos", gt_future[:, robot_state_dim : robot_state_dim + 3], pred_future[:, robot_state_dim : robot_state_dim + 3]),
        (
            "object lin vel",
            gt_future[:, robot_state_dim + 7 : robot_state_dim + 10],
            pred_future[:, robot_state_dim + 7 : robot_state_dim + 10],
        ),
        ("robot qdd", gt_robot_dynamics["qdd"], stack_aux(pred_aux, "ddq")),
        ("residual tau", gt_residual_tau, stack_aux(pred_aux, "tau_residual")),
        ("object lin acc", gt_object_dynamics["root_lin_acc_w"], stack_aux(pred_aux, "object_lin_acc")),
        ("object ang acc", gt_object_dynamics["root_ang_acc_w"], stack_aux(pred_aux, "object_ang_acc")),
        ("object ext force", gt_object_dynamics["external_force_est_w"], stack_aux(pred_aux, "object_external_force")),
    ]


def plot_terms(
    terms: list[tuple[str, np.ndarray, np.ndarray]],
    time: np.ndarray,
    contact_t: int | None,
    dt: float,
    output_path: str,
    episode_name: str,
    start_t: int,
    failed_step: int | None,
) -> None:
    fig, axes = plt.subplots(nrows=len(terms), ncols=1, figsize=(10, 2.2 * len(terms)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (term_name, gt_values, pred_values) in zip(axes, terms):
        dims = min(gt_values.shape[-1], pred_values.shape[-1], 3)
        for dim in range(dims):
            ax.plot(time, gt_values[:, dim], linewidth=1.2, label=f"GT dim {dim}" if dim == 0 else None)
            ax.plot(
                time,
                pred_values[:, dim],
                linestyle="--",
                linewidth=1.2,
                label=f"Pred dim {dim}" if dim == 0 else None,
            )
        if contact_t is not None:
            contact_time = contact_t * dt
            if time[0] <= contact_time <= time[-1]:
                ax.axvline(contact_time, color="tab:green", linestyle=":", linewidth=1.4)
        ax.set_ylabel(term_name)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("episode time [s]")
    title = f"GT vs Predicted Robot/Object Dynamics Terms ({episode_name}, start_t={start_t})"
    if failed_step is not None:
        title += f" | rollout stopped at step {failed_step}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_term_metrics(terms: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    print("Mean squared error over plotted rollout:")
    for term_name, gt_values, pred_values in terms:
        err = (pred_values - gt_values) ** 2
        print(f"  {term_name}: mean={float(np.mean(err)):.8f}, max={float(np.max(err)):.8f}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, layout, checkpoint_path = load_checkpoint_model(args.checkpoint, device)
    episode_name = resolve_episode_name(args.dataset_file, args.episode_index, args.episode_name)
    torque_key = args.torque_key or str(cfg["torque_key"])
    states, torques, object_context, robot_dynamics, object_dynamics = load_episode_arrays(
        args.dataset_file, episode_name, layout, torque_key
    )

    history_len = int(cfg["history_len"])
    start_t = history_len - 1 if args.start_t is None else args.start_t
    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1}).")
    if start_t >= states.shape[0] - 1:
        raise ValueError(f"start_t must leave at least one future step. Episode length is {states.shape[0]}.")

    max_available_steps = states.shape[0] - start_t - 1
    requested_steps = max_available_steps if args.rollout_steps <= 0 else min(args.rollout_steps, max_available_steps)
    history_states = torch.from_numpy(states[start_t - history_len + 1 : start_t + 1]).unsqueeze(0).to(device)
    future_torques = torch.from_numpy(torques[start_t : start_t + requested_steps]).unsqueeze(0).to(device)
    context = torch.from_numpy(object_context[start_t]).unsqueeze(0).to(device)
    pred_future, pred_aux, failed_step, failure_reason = rollout_stepwise(model, history_states[:, -1], future_torques, context)

    actual_steps = pred_future.shape[0]
    gt_future = states[start_t + 1 : start_t + actual_steps + 1]
    gt_torques = torques[start_t : start_t + actual_steps]
    gt_robot_window = {key: value[start_t : start_t + actual_steps] for key, value in robot_dynamics.items()}
    gt_object_window = {key: value[start_t : start_t + actual_steps] for key, value in object_dynamics.items()}
    terms = build_plot_terms(
        gt_future=gt_future,
        pred_future=pred_future,
        gt_torques=gt_torques,
        gt_robot_dynamics=gt_robot_window,
        gt_object_dynamics=gt_object_window,
        pred_aux=pred_aux,
        robot_dof=layout.robot_dof,
    )

    robot_state_dim = layout.robot_state_dim
    contact_t = first_object_motion_timestep(
        object_pos=states[:, robot_state_dim : robot_state_dim + 3],
        dt=float(cfg["dt"]),
        displacement_threshold=args.object_displacement_threshold,
        velocity_threshold=args.object_velocity_threshold,
        consecutive_steps=args.contact_consecutive_steps,
        settle_steps=args.contact_settle_steps,
    )
    time = (np.arange(actual_steps, dtype=np.float32) + start_t + 1) * float(cfg["dt"])
    output_path = os.path.join(args.output_dir, args.output_name)
    plot_terms(
        terms=terms,
        time=time,
        contact_t=contact_t,
        dt=float(cfg["dt"]),
        output_path=output_path,
        episode_name=episode_name,
        start_t=start_t,
        failed_step=failed_step,
    )

    print("===== GT vs Predicted Robot/Object Dynamics Terms =====")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_file: {args.dataset_file}")
    print(f"episode: {episode_name}")
    print(f"torque_key: {torque_key}")
    print(f"start_t: {start_t}")
    print(f"requested_rollout_steps: {requested_steps}")
    print(f"evaluated_rollout_steps: {actual_steps}")
    print(f"first_contact_proxy_timestep: {contact_t}")
    if failed_step is not None:
        print(f"rollout_stopped_at_step: {failed_step}")
        print(f"rollout_stop_reason: {failure_reason}")
    print_term_metrics(terms)
    print("Note: residual tau target is robot_dynamics/inverse_dynamics_tau - recorded torque.")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
