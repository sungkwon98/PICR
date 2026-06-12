from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from eval_utils import (
    episode_names,
    first_object_motion_timestep,
    franka_joint_and_gripper_positions,
    load_checkpoint_model,
    load_episode_arrays,
    load_episode_robot_joint_positions,
    rollout_stepwise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean accumulated gripper/object position error for Context DeLaN."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default="./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5")
    parser.add_argument("--start_t", type=int, default=None, help="Default: history_len - 1.")
    parser.add_argument("--rollout_steps", type=int, default=0, help="0 means use the rest of each episode.")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 means evaluate all episodes.")
    parser.add_argument("--torque_key", type=str, default=None, choices=[None, "applied_torque", "computed_torque"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--skip_failed", action="store_true", default=False)
    parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
    parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
    parser.add_argument("--contact_consecutive_steps", type=int, default=3)
    parser.add_argument("--contact_settle_steps", type=int, default=5)
    return parser.parse_args()


def _masked_sum(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sum(values[mask])) if np.any(mask) else 0.0


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def evaluate_episode(
    model,
    cfg: dict,
    layout,
    hdf5_path: str,
    episode_name: str,
    device: torch.device,
    start_t_arg: int | None,
    rollout_steps_arg: int,
    torque_key: str,
    tool_z_offset: float,
    object_displacement_threshold: float,
    object_velocity_threshold: float,
    contact_consecutive_steps: int,
    contact_settle_steps: int,
) -> dict[str, float | int | str | bool]:
    states, torques, _, _, _ = load_episode_arrays(hdf5_path, episode_name, layout, torque_key)
    absolute_joint_pos = load_episode_robot_joint_positions(hdf5_path, episode_name, layout.robot_dof)
    t_count = min(states.shape[0], torques.shape[0], absolute_joint_pos.shape[0])
    states = states[:t_count]
    torques = torques[:t_count]
    absolute_joint_pos = absolute_joint_pos[:t_count]

    history_len = int(cfg["history_len"])
    start_t = history_len - 1 if start_t_arg is None else start_t_arg
    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1}).")
    if start_t >= states.shape[0] - 1:
        raise ValueError(f"start_t must leave at least one future step. Episode length is {states.shape[0]}.")

    requested_steps = states.shape[0] - start_t - 1 if rollout_steps_arg <= 0 else min(rollout_steps_arg, states.shape[0] - start_t - 1)
    history_states = torch.from_numpy(states[start_t - history_len + 1 : start_t + 1]).unsqueeze(0).to(device)
    history_torques = torch.from_numpy(torques[start_t - history_len + 1 : start_t + 1]).unsqueeze(0).to(device)
    future_torques = torch.from_numpy(torques[start_t : start_t + requested_steps]).unsqueeze(0).to(device)
    pred_states, _, z, failed_step, failure_reason = rollout_stepwise(model, history_states, history_torques, future_torques)

    actual_steps = pred_states.shape[0]
    gt_future = states[start_t + 1 : start_t + actual_steps + 1]
    gt_future_q = absolute_joint_pos[start_t + 1 : start_t + actual_steps + 1]
    relative_to_absolute_offset = absolute_joint_pos[start_t] - states[start_t, : layout.robot_dof]
    pred_q = pred_states[:, : layout.robot_dof] + relative_to_absolute_offset
    gt_gripper = franka_joint_and_gripper_positions(gt_future_q, tool_z_offset)["gripper"]
    pred_gripper = franka_joint_and_gripper_positions(pred_q, tool_z_offset)["gripper"]
    robot_state_dim = layout.robot_state_dim
    gt_object = gt_future[:, robot_state_dim : robot_state_dim + 3]
    pred_object = pred_states[:, robot_state_dim : robot_state_dim + 3]
    real_episode_object = states[:, robot_state_dim : robot_state_dim + 3]
    contact_t = first_object_motion_timestep(
        object_pos=real_episode_object,
        dt=float(cfg["dt"]),
        displacement_threshold=object_displacement_threshold,
        velocity_threshold=object_velocity_threshold,
        consecutive_steps=contact_consecutive_steps,
        settle_steps=contact_settle_steps,
    )

    gripper_error = np.linalg.norm(pred_gripper - gt_gripper, axis=-1)
    object_error = np.linalg.norm(pred_object - gt_object, axis=-1)
    episode_timesteps = np.arange(start_t + 1, start_t + actual_steps + 1)
    if contact_t is None:
        pre_contact_mask = np.ones(actual_steps, dtype=bool)
        post_contact_mask = np.zeros(actual_steps, dtype=bool)
        contact_t_value = -1
    else:
        pre_contact_mask = episode_timesteps < contact_t
        post_contact_mask = episode_timesteps >= contact_t
        contact_t_value = int(contact_t)

    dt = float(cfg["dt"])
    failed = failed_step is not None and actual_steps < requested_steps
    return {
        "episode": episode_name,
        "start_t": start_t,
        "contact_t": contact_t_value,
        "requested_steps": requested_steps,
        "evaluated_steps": actual_steps,
        "pre_contact_steps": int(np.sum(pre_contact_mask)),
        "post_contact_steps": int(np.sum(post_contact_mask)),
        "failed": failed,
        "failed_step": -1 if failed_step is None else failed_step,
        "failure_reason": "" if failure_reason is None else failure_reason,
        "latent_abs_mean": float(z.abs().mean()),
        "gripper_accumulated_error": float(np.sum(gripper_error)),
        "object_accumulated_error": float(np.sum(object_error)),
        "gripper_pre_contact_accumulated_error": _masked_sum(gripper_error, pre_contact_mask),
        "object_pre_contact_accumulated_error": _masked_sum(object_error, pre_contact_mask),
        "gripper_post_contact_accumulated_error": _masked_sum(gripper_error, post_contact_mask),
        "object_post_contact_accumulated_error": _masked_sum(object_error, post_contact_mask),
        "gripper_area_error": float(np.sum(gripper_error) * dt),
        "object_area_error": float(np.sum(object_error) * dt),
        "gripper_pre_contact_area_error": _masked_sum(gripper_error, pre_contact_mask) * dt,
        "object_pre_contact_area_error": _masked_sum(object_error, pre_contact_mask) * dt,
        "gripper_post_contact_area_error": _masked_sum(gripper_error, post_contact_mask) * dt,
        "object_post_contact_area_error": _masked_sum(object_error, post_contact_mask) * dt,
        "gripper_mean_error": float(np.mean(gripper_error)),
        "object_mean_error": float(np.mean(object_error)),
        "gripper_pre_contact_mean_error": _masked_mean(gripper_error, pre_contact_mask),
        "object_pre_contact_mean_error": _masked_mean(object_error, pre_contact_mask),
        "gripper_post_contact_mean_error": _masked_mean(gripper_error, post_contact_mask),
        "object_post_contact_mean_error": _masked_mean(object_error, post_contact_mask),
        "gripper_max_error": float(np.max(gripper_error)),
        "object_max_error": float(np.max(object_error)),
    }


def _mean(rows: list[dict[str, float | int | str | bool]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _std(rows: list[dict[str, float | int | str | bool]], key: str) -> float:
    return float(np.std([float(row[key]) for row in rows]))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, cfg, layout, checkpoint_path = load_checkpoint_model(args.checkpoint, device)
    torque_key = args.torque_key or str(cfg["torque_key"])
    names = episode_names(args.dataset_file)
    if args.max_episodes > 0:
        names = names[: args.max_episodes]
    if not names:
        raise RuntimeError("No episodes selected for evaluation.")

    rows: list[dict[str, float | int | str | bool]] = []
    skipped_failed = 0
    for idx, name in enumerate(names, start=1):
        try:
            row = evaluate_episode(
                model=model,
                cfg=cfg,
                layout=layout,
                hdf5_path=args.dataset_file,
                episode_name=name,
                device=device,
                start_t_arg=args.start_t,
                rollout_steps_arg=args.rollout_steps,
                torque_key=torque_key,
                tool_z_offset=args.tool_z_offset,
                object_displacement_threshold=args.object_displacement_threshold,
                object_velocity_threshold=args.object_velocity_threshold,
                contact_consecutive_steps=args.contact_consecutive_steps,
                contact_settle_steps=args.contact_settle_steps,
            )
        except Exception as exc:
            print(f"[{idx}/{len(names)}] {name}: skipped due to error: {exc}")
            continue
        if args.skip_failed and bool(row["failed"]):
            skipped_failed += 1
            continue
        rows.append(row)
        if idx % 100 == 0 or idx == len(names):
            print(f"[{idx}/{len(names)}] evaluated={len(rows)} skipped_failed={skipped_failed}")

    if not rows:
        raise RuntimeError("No valid episodes were evaluated.")

    if args.output_csv is not None:
        output_csv = os.path.abspath(args.output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-episode metrics: {output_csv}")

    pre_steps = np.asarray([int(row["pre_contact_steps"]) for row in rows], dtype=np.int64)
    post_steps = np.asarray([int(row["post_contact_steps"]) for row in rows], dtype=np.int64)
    failed_count = sum(bool(row["failed"]) for row in rows)

    print("===== Context DeLaN Mean Accumulated Position Error =====")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_file: {args.dataset_file}")
    print(f"context_encoder: {cfg['context_encoder']}")
    print(f"torque_key: {torque_key}")
    print(f"episodes_requested: {len(names)}")
    print(f"episodes_evaluated: {len(rows)}")
    print(f"episodes_failed_rollout_included: {failed_count}")
    print(f"episodes_failed_rollout_skipped: {skipped_failed}")
    print(f"mean_latent_abs: {_mean(rows, 'latent_abs_mean'):.8f}")
    print(f"episodes_with_pre_contact_steps: {int(np.sum(pre_steps > 0))}")
    print(f"episodes_with_post_contact_steps: {int(np.sum(post_steps > 0))}")
    print("--- total rollout interval ---")
    print(f"mean_gripper_accumulated_error_m_step: {_mean(rows, 'gripper_accumulated_error'):.8f}")
    print(f"std_gripper_accumulated_error_m_step: {_std(rows, 'gripper_accumulated_error'):.8f}")
    print(f"mean_object_accumulated_error_m_step: {_mean(rows, 'object_accumulated_error'):.8f}")
    print(f"std_object_accumulated_error_m_step: {_std(rows, 'object_accumulated_error'):.8f}")
    print(f"mean_gripper_area_error_m_s: {_mean(rows, 'gripper_area_error'):.8f}")
    print(f"mean_object_area_error_m_s: {_mean(rows, 'object_area_error'):.8f}")
    print("--- before first contact ---")
    print(f"mean_gripper_pre_contact_accumulated_error_m_step: {_mean(rows, 'gripper_pre_contact_accumulated_error'):.8f}")
    print(f"mean_object_pre_contact_accumulated_error_m_step: {_mean(rows, 'object_pre_contact_accumulated_error'):.8f}")
    print(f"mean_gripper_pre_contact_area_error_m_s: {_mean(rows, 'gripper_pre_contact_area_error'):.8f}")
    print(f"mean_object_pre_contact_area_error_m_s: {_mean(rows, 'object_pre_contact_area_error'):.8f}")
    print("--- after first contact ---")
    print(f"mean_gripper_post_contact_accumulated_error_m_step: {_mean(rows, 'gripper_post_contact_accumulated_error'):.8f}")
    print(f"mean_object_post_contact_accumulated_error_m_step: {_mean(rows, 'object_post_contact_accumulated_error'):.8f}")
    print(f"mean_gripper_post_contact_area_error_m_s: {_mean(rows, 'gripper_post_contact_area_error'):.8f}")
    print(f"mean_object_post_contact_area_error_m_s: {_mean(rows, 'object_post_contact_area_error'):.8f}")
    print("Note: accumulated_error is sum(error_t); area_error is accumulated_error * dt.")


if __name__ == "__main__":
    main()
