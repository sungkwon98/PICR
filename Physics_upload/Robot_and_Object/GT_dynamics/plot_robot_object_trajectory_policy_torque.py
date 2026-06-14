from __future__ import annotations

"""
Policy-driven trajectory plot for the hybrid robot-object GT dynamics model.

This differs from ``plot_robot_object_trajectory.py`` in one key way: future
robot torques are not read from the HDF5 episode. Instead, this script builds a
policy observation from the current world-model predicted state, asks the skrl
policy for an action, steps IsaacLab once with that action, reads the resulting
actuator torque, and feeds that torque into the GT dynamics world model.

Important limitation: the IsaacLab actuator torque is computed by the simulator
for the simulator's current state after stepping with the policy action. The
policy observation is based on the predicted state, but the simulator is not
reset to the predicted state at every step.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot hybrid GT world-model rollout driven by policy-generated env torques.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Physics-Constant-Franka-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default="/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/reinforcement_learning/skrl/logs/skrl/franka_lift/2026-04-13_18-16-25_ppo_torch/checkpoints/agent_36000.pt",
    help="skrl policy checkpoint used to generate actions.",
)
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--wm_checkpoint", type=str, default="./outputs_robot_object_gt_dynamics/run_20260514_004522/best.pt")
parser.add_argument("--dataset_file", type=str, default="./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5")
parser.add_argument("--episode_index", type=int, default=0)
parser.add_argument("--episode_name", type=str, default=None)
parser.add_argument("--start_t", type=int, default=5)
parser.add_argument("--rollout_steps", type=int, default=35)
parser.add_argument(
    "--target",
    type=str,
    default="gripper",
    help="Robot body to plot: 0, 1, ..., 7, or gripper. Joint 0 is the fixed robot base origin.",
)
parser.add_argument("--output_dir", type=str, default="./eval_outputs")
parser.add_argument("--output_name", type=str, default="robot_object_policy_torque_trajectory.png")
parser.add_argument("--tool_z_offset", type=float, default=0.1034)
parser.add_argument("--object_displacement_threshold", type=float, default=0.005)
parser.add_argument("--object_velocity_threshold", type=float, default=0.02)
parser.add_argument("--contact_consecutive_steps", type=int, default=3)
parser.add_argument("--contact_settle_steps", type=int, default=5)
parser.add_argument(
    "--max_abs_position",
    type=float,
    default=5.0,
    help="Stop rollout before plotting if any predicted robot/object position exceeds this absolute value in meters.",
)
parser.add_argument(
    "--max_abs_velocity",
    type=float,
    default=50.0,
    help="Stop rollout before plotting if any predicted robot/object velocity exceeds this absolute value.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random

import gymnasium as gym
import h5py
import matplotlib
import numpy as np
import skrl
import torch
from packaging import version

matplotlib.use("Agg")

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from eval_utils import (
    first_object_motion_timestep,
    franka_joint_and_gripper_positions,
    load_checkpoint_model,
    load_episode_arrays,
    load_episode_robot_joint_positions,
    parse_robot_target_name,
    resolve_episode_name,
)
from plot_robot_object_trajectory import plot_trajectory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_policy_episode_arrays(hdf5_path: str, episode_name: str) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r", locking=False) as file:
        episode = file["data"][episode_name]
        obs = episode["obs"]
        return {
            "target_object_position": np.asarray(obs["target_object_position"], dtype=np.float32),
            "prev_action": np.asarray(obs["actions"], dtype=np.float32),
            "actions": np.asarray(episode["actions"], dtype=np.float32),
        }


def _restore_command_from_episode(env, episode_data, env_id: int) -> None:
    data = episode_data.data
    target = None
    if "obs" in data:
        obs = data["obs"]
        if isinstance(obs, dict) and "target_object_position" in obs:
            target_tensor = obs["target_object_position"]
            if target_tensor is not None and target_tensor.numel() >= 7:
                target = target_tensor[0].flatten()[:7].to(env.device)
    if target is not None and hasattr(env, "command_manager") and "object_pose" in env.command_manager.active_terms:
        env.command_manager.get_term("object_pose").pose_command_b[env_id] = target


def reset_to_dataset_initial_state(env_unwrapped, dataset_file: str, episode_index: int) -> None:
    dataset = HDF5DatasetFileHandler()
    dataset.open(os.path.abspath(dataset_file))
    episode_name = list(dataset.get_episode_names())[episode_index]
    episode = dataset.load_episode(episode_name, env_unwrapped.device)
    env_unwrapped.reset()
    env_unwrapped.reset_to(episode.get_initial_state(), torch.tensor([0], device=env_unwrapped.device), is_relative=True)
    _restore_command_from_episode(env_unwrapped, episode, 0)
    dataset.close()


def build_policy_observation(
    pred_state: torch.Tensor,
    target_object_position: torch.Tensor,
    prev_action: torch.Tensor,
    ref_obs,
) -> torch.Tensor | dict:
    """Build policy observation: [joint_pos, joint_vel, object_pos, target_pose, prev_action]."""
    policy_tensor = torch.cat(
        [
            pred_state[:, :9],
            pred_state[:, 9:18],
            pred_state[:, 18:21],
            target_object_position,
            prev_action,
        ],
        dim=-1,
    )
    if isinstance(ref_obs, dict):
        obs_for_policy = {key: value.clone() if torch.is_tensor(value) else value for key, value in ref_obs.items()}
        ref_policy = obs_for_policy.get("policy", None)
        if ref_policy is None:
            raise KeyError("Wrapped env observation dictionary has no 'policy' key.")
        obs_for_policy["policy"] = match_policy_dim(policy_tensor, ref_policy)
        return obs_for_policy
    return match_policy_dim(policy_tensor, ref_obs)


def match_policy_dim(policy_tensor: torch.Tensor, ref_policy: torch.Tensor) -> torch.Tensor:
    policy_dim = ref_policy.shape[-1]
    if policy_tensor.shape[-1] == policy_dim:
        return policy_tensor
    if policy_tensor.shape[-1] < policy_dim:
        pad = torch.zeros(
            (policy_tensor.shape[0], policy_dim - policy_tensor.shape[-1]),
            device=policy_tensor.device,
            dtype=policy_tensor.dtype,
        )
        return torch.cat([policy_tensor, pad], dim=-1)
    return policy_tensor[:, :policy_dim]


def policy_action_from_predicted_state(
    runner: Runner,
    ref_obs,
    pred_state: torch.Tensor,
    target_object_position: torch.Tensor,
    prev_action: torch.Tensor,
) -> torch.Tensor:
    obs_for_policy = build_policy_observation(pred_state, target_object_position, prev_action, ref_obs)
    outputs = runner.agent.act(obs_for_policy, timestep=0, timesteps=0)
    return outputs[-1].get("mean_actions", outputs[0])


def rollout_with_policy_torques(
    model,
    env,
    env_unwrapped,
    runner: Runner,
    obs,
    initial_state: torch.Tensor,
    object_context: torch.Tensor,
    policy_targets: torch.Tensor,
    initial_prev_action: torch.Tensor,
    rollout_steps: int,
    max_abs_position: float,
    max_abs_velocity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None, str | None, object]:
    state = initial_state
    prev_action = initial_prev_action
    pred_states: list[torch.Tensor] = []
    torques: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    failed_step: int | None = None
    failure_reason: str | None = None

    with torch.inference_mode():
        for step_idx in range(rollout_steps):
            if not torch.isfinite(state).all():
                failed_step = step_idx
                failure_reason = "non-finite predicted state before policy/action step"
                break
            target_idx = min(step_idx, policy_targets.shape[0] - 1)
            target_pose = policy_targets[target_idx : target_idx + 1]
            action = policy_action_from_predicted_state(runner, obs, state, target_pose, prev_action)
            obs, _, terminated, truncated, _ = env.step(action)

            robot = env_unwrapped.scene["robot"]
            torque = robot.data.applied_torque[0:1].detach().to(device=state.device, dtype=state.dtype)
            try:
                state, _ = model.dynamics(state, torque, object_context)
            except RuntimeError as exc:
                failed_step = step_idx
                failure_reason = str(exc).splitlines()[0]
                break
            if not torch.isfinite(state).all():
                failed_step = step_idx + 1
                failure_reason = "non-finite predicted state after dynamics step"
                break
            robot_q = state[:, :9]
            robot_dq = state[:, 9:18]
            object_pos = state[:, 18:21]
            object_lin_vel = state[:, 25:28]
            object_ang_vel = state[:, 28:31]
            max_position = torch.cat([robot_q, object_pos], dim=-1).abs().max()
            max_velocity = torch.cat([robot_dq, object_lin_vel, object_ang_vel], dim=-1).abs().max()
            if float(max_position.item()) > max_abs_position:
                failed_step = step_idx + 1
                failure_reason = (
                    f"predicted position exceeded --max_abs_position={max_abs_position}: "
                    f"max_abs_position={float(max_position.item()):.6f}"
                )
                break
            if float(max_velocity.item()) > max_abs_velocity:
                failed_step = step_idx + 1
                failure_reason = (
                    f"predicted velocity exceeded --max_abs_velocity={max_abs_velocity}: "
                    f"max_abs_velocity={float(max_velocity.item()):.6f}"
                )
                break

            pred_states.append(state.detach().cpu())
            torques.append(torque.detach().cpu())
            actions.append(action[:, :8].detach().cpu())
            prev_action = action[:, :8].to(device=state.device, dtype=state.dtype)
            if terminated.any() or truncated.any():
                failed_step = step_idx + 1
                failure_reason = "IsaacLab environment terminated or truncated"
                break

    if not pred_states:
        raise RuntimeError(f"Policy-torque rollout failed before producing any prediction: {failure_reason}")
    return (
        torch.cat(pred_states, dim=0).numpy(),
        torch.cat(torques, dim=0).numpy(),
        torch.cat(actions, dim=0).numpy(),
        failed_step,
        failure_reason,
        obs,
    )


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    device = torch.device(args_cli.device)
    set_seed(int(experiment_cfg.get("seed", 42)))

    dataset_file = os.path.abspath(args_cli.dataset_file)
    episode_name = resolve_episode_name(dataset_file, args_cli.episode_index, args_cli.episode_name)
    model, cfg, layout, wm_checkpoint_path = load_checkpoint_model(args_cli.wm_checkpoint, device)
    torque_key = str(cfg["torque_key"])
    states, _, object_context, _, _ = load_episode_arrays(dataset_file, episode_name, layout, torque_key)
    absolute_joint_pos = load_episode_robot_joint_positions(dataset_file, episode_name, layout.robot_dof)
    policy_arrays = load_policy_episode_arrays(dataset_file, episode_name)

    t_count = min(
        states.shape[0],
        object_context.shape[0],
        absolute_joint_pos.shape[0],
        policy_arrays["target_object_position"].shape[0],
        policy_arrays["prev_action"].shape[0],
    )
    states = states[:t_count]
    object_context = object_context[:t_count]
    absolute_joint_pos = absolute_joint_pos[:t_count]
    policy_targets_np = policy_arrays["target_object_position"][:t_count]
    prev_actions_np = policy_arrays["prev_action"][:t_count]

    history_len = int(cfg["history_len"])
    start_t = history_len - 1 if args_cli.start_t is None else args_cli.start_t
    if start_t < history_len - 1:
        raise ValueError(f"start_t must be >= history_len - 1 ({history_len - 1}).")
    if start_t >= states.shape[0] - 1:
        raise ValueError(f"start_t must leave at least one future step. Episode length is {states.shape[0]}.")
    rollout_steps = min(args_cli.rollout_steps, states.shape[0] - start_t - 1)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    if isinstance(env_unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
        env_unwrapped = env.unwrapped
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    policy_checkpoint_path = os.path.abspath(args_cli.policy_checkpoint)
    print(f"[INFO] Loading policy checkpoint from: {policy_checkpoint_path}")
    runner.agent.load(policy_checkpoint_path)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    reset_to_dataset_initial_state(env_unwrapped, dataset_file, args_cli.episode_index)
    obs, _ = env.reset()
    reset_to_dataset_initial_state(env_unwrapped, dataset_file, args_cli.episode_index)

    # Bring the simulator actuator state close to the selected episode time.
    with torch.inference_mode():
        for t in range(start_t):
            action_t = torch.from_numpy(policy_arrays["actions"][t : t + 1]).to(device)
            obs, _, terminated, truncated, _ = env.step(action_t)
            if terminated.any() or truncated.any():
                raise RuntimeError(f"Environment terminated during dataset warmup at t={t}.")

    initial_state = torch.from_numpy(states[start_t : start_t + 1]).to(device)
    context = torch.from_numpy(object_context[start_t : start_t + 1]).to(device)
    policy_targets = torch.from_numpy(policy_targets_np[start_t : start_t + rollout_steps]).to(device)
    prev_action = torch.from_numpy(prev_actions_np[start_t : start_t + 1, :8]).to(device)
    pred_future, policy_torques, policy_actions, failed_step, failure_reason, _ = rollout_with_policy_torques(
        model=model,
        env=env,
        env_unwrapped=env_unwrapped,
        runner=runner,
        obs=obs,
        initial_state=initial_state,
        object_context=context,
        policy_targets=policy_targets,
        initial_prev_action=prev_action,
        rollout_steps=rollout_steps,
        max_abs_position=args_cli.max_abs_position,
        max_abs_velocity=args_cli.max_abs_velocity,
    )
    env.close()

    actual_steps = pred_future.shape[0]
    gt_future = states[start_t + 1 : start_t + actual_steps + 1]
    target_name = parse_robot_target_name(args_cli.target)
    gt_future_absolute_q = absolute_joint_pos[start_t + 1 : start_t + actual_steps + 1]
    relative_to_absolute_offset = absolute_joint_pos[start_t] - states[start_t, : layout.robot_dof]
    pred_absolute_q = pred_future[:, : layout.robot_dof] + relative_to_absolute_offset
    real_episode_trajectories = franka_joint_and_gripper_positions(absolute_joint_pos, args_cli.tool_z_offset)
    gt_trajectories = franka_joint_and_gripper_positions(gt_future_absolute_q, args_cli.tool_z_offset)
    pred_trajectories = franka_joint_and_gripper_positions(pred_absolute_q, args_cli.tool_z_offset)
    real_episode_target = real_episode_trajectories[target_name]
    gt_target = gt_trajectories[target_name]
    pred_target = pred_trajectories[target_name]
    robot_state_dim = layout.robot_state_dim
    real_episode_object = states[:, robot_state_dim : robot_state_dim + 3]
    gt_object = gt_future[:, robot_state_dim : robot_state_dim + 3]
    pred_object = pred_future[:, robot_state_dim : robot_state_dim + 3]
    contact_t = first_object_motion_timestep(
        object_pos=states[:, robot_state_dim : robot_state_dim + 3],
        dt=float(cfg["dt"]),
        displacement_threshold=args_cli.object_displacement_threshold,
        velocity_threshold=args_cli.object_velocity_threshold,
        consecutive_steps=args_cli.contact_consecutive_steps,
        settle_steps=args_cli.contact_settle_steps,
    )
    time = (np.arange(actual_steps, dtype=np.float32) + start_t + 1) * float(cfg["dt"])
    output_path = os.path.join(args_cli.output_dir, args_cli.output_name)
    os.makedirs(args_cli.output_dir, exist_ok=True)
    target_rmse, object_rmse = plot_trajectory(
        real_episode_target=real_episode_target,
        real_episode_object=real_episode_object,
        gt_target=gt_target,
        pred_target=pred_target,
        gt_object=gt_object,
        pred_object=pred_object,
        time=time,
        contact_t=contact_t,
        dt=float(cfg["dt"]),
        output_path=output_path,
        episode_name=episode_name,
        target_name=target_name,
        start_t=start_t,
    )

    print("===== Policy-Torque Robot/Object Trajectory Plot =====")
    print(f"world_model_checkpoint: {wm_checkpoint_path}")
    print(f"policy_checkpoint: {policy_checkpoint_path}")
    print(f"dataset_file: {dataset_file}")
    print(f"episode: {episode_name}")
    print(f"target: {target_name}")
    print(f"start_t: {start_t}")
    print(f"first_contact_proxy_timestep: {contact_t}")
    print(f"requested_rollout_steps: {rollout_steps}")
    print(f"evaluated_rollout_steps: {actual_steps}")
    if failed_step is not None:
        print(f"rollout_stopped_at_step: {failed_step}")
        print(f"rollout_stop_reason: {failure_reason}")
    print(f"{target_name.replace(' ', '_')}_position_rmse_m: {target_rmse:.8f}")
    print(f"object_position_rmse_m: {object_rmse:.8f}")
    print(f"policy_action_abs_mean: {float(np.mean(np.abs(policy_actions))):.8f}")
    print(f"policy_torque_abs_mean: {float(np.mean(np.abs(policy_torques))):.8f}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
