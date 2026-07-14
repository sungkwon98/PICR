from __future__ import annotations

"""Render an Isaac Lab GT-vs-world-model comparison video for Franka cube lift rollouts."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parents[1]
_WORLD_MODEL_DIR = _THIS_FILE.parents[2]
_REPO_ROOT = _THIS_FILE.parents[4]
_DEFAULT_CONFIG = _PACKAGE_DIR / "configs" / "isaaclab_visualization.yaml"

for _path in (
    _REPO_ROOT / "source" / "isaaclab",
    _REPO_ROOT / "source" / "isaaclab_assets",
    _REPO_ROOT / "source" / "isaaclab_tasks",
    _REPO_ROOT / "source" / "isaaclab_rl",
    _REPO_ROOT / "source" / "isaaclab_mimic",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if __package__ in (None, ""):
    sys.path.insert(0, str(_WORLD_MODEL_DIR))

from isaaclab.app import AppLauncher


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a headless Isaac Lab GT-vs-WM comparison video.")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--start_t", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--video_width", type=int, default=None)
    parser.add_argument("--video_height", type=int, default=None)
    parser.add_argument("--env_spacing", type=float, default=None)
    parser.add_argument("--warmup_frames", type=int, default=None)
    parser.add_argument("--no_apply_domain_params", action="store_true", default=False)
    parser.add_argument("--no_clamp_imagination_joints", action="store_true", default=False)
    parser.add_argument("--no_couple_imagination_gripper", action="store_true", default=False)
    parser.add_argument("--use_ground_truth_gripper", action="store_true", default=False)
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    args.enable_cameras = True
    # AppLauncher uses this flag to keep viewport rendering alive in headless mode.
    args.video = True
    return args


args_cli = parse_cli()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below requires Isaac Sim to be launched first."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from robot_object_wm.data.dataset import h5_open  # noqa: E402
from robot_object_wm.eval.episode import load_episode_data, resolve_episode_name, rollout_episode  # noqa: E402
from robot_object_wm.eval.rollout import load_checkpoint_model  # noqa: E402


@dataclass
class DomainValue:
    value: np.ndarray
    source: str


@dataclass
class VisualEpisode:
    name: str
    robot_root_pose_w: np.ndarray
    robot_root_velocity_w: np.ndarray
    joint_pos_abs: np.ndarray
    joint_vel: np.ndarray
    object_root_pose_w: np.ndarray
    object_root_velocity_w: np.ndarray
    source_origin_w: np.ndarray
    domain: dict[str, DomainValue]

    @property
    def T(self) -> int:
        return int(self.joint_pos_abs.shape[0])


@dataclass
class RenderTrack:
    robot_root_pose_w: np.ndarray
    robot_root_velocity_w: np.ndarray
    joint_pos_abs: np.ndarray
    joint_vel: np.ndarray
    object_root_pose_w: np.ndarray
    object_root_velocity_w: np.ndarray

    @property
    def T(self) -> int:
        return int(self.joint_pos_abs.shape[0])


@dataclass
class ComparisonTracks:
    gt: RenderTrack
    wm: RenderTrack
    episode_name: str
    start_t: int
    dt: float
    failed_step: int | None
    failure_reason: str | None

    @property
    def T(self) -> int:
        return min(self.gt.T, self.wm.T)


class FfmpegVideoWriter:
    def __init__(self, output_path: str, *, width: int, height: int, fps: int) -> None:
        self.output_path = os.path.abspath(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.output_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed.")
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            frame = np.asarray(Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.BILINEAR))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame ({self.height}, {self.width}, 3), got {frame.shape}.")
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}:\n{stderr[-4000:]}")


def main() -> str:
    config = load_config(args_cli)
    loaded = load_checkpoint_model(config["checkpoint"], device=args_cli.device)
    wm_cfg = loaded.config

    visual_episode = load_visual_episode(
        config["dataset_file"],
        episode_index=int(config["episode_index"]),
        episode_name=config.get("episode_name"),
        robot_dof=wm_cfg.robot_dof,
        max_frames=0,
        domain_t=int(config["start_t"]),
    )
    wm_episode = load_episode_data(
        config["dataset_file"],
        episode_index=int(config["episode_index"]),
        episode_name=config.get("episode_name"),
        robot_dof=wm_cfg.robot_dof,
        action_dim=wm_cfg.action_dim,
        torque_dim=wm_cfg.torque_dim,
        torque_key=wm_cfg.torque_key,
        subtract_env_origin=wm_cfg.subtract_env_origin,
        dt=wm_cfg.dt,
        state_prediction_mode=getattr(wm_cfg, "state_prediction_mode", "full"),
        privileged_collision_observation=wm_cfg.privileged_collision_observation,
        privileged_collision_group=wm_cfg.privileged_collision_group,
        privileged_collision_pairs=wm_cfg.privileged_collision_pairs,
    )
    tracks = build_comparison_tracks(
        model=loaded.model,
        wm_cfg=wm_cfg,
        visual_episode=visual_episode,
        wm_episode=wm_episode,
        start_t=int(config["start_t"]),
        rollout_steps=int(config["rollout_steps"]),
        max_frames=int(config["max_frames"]),
        device=next(loaded.model.parameters()).device,
    )

    env_cfg = parse_env_cfg(
        config["task"],
        device=args_cli.device,
        num_envs=2,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 2
    env_cfg.scene.env_spacing = float(config["env_spacing"])
    env_cfg.viewer.resolution = (int(config["video_width"]), int(config["video_height"]))
    if hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    env = gym.make(config["task"], cfg=env_cfg, render_mode="rgb_array")

    try:
        env.reset()
        if bool(config["apply_domain_params"]):
            apply_episode_domain_parameters(env.unwrapped, visual_episode.domain)
        configure_camera(env.unwrapped, config)
        output_path = render_comparison_video(env.unwrapped, tracks, visual_episode, config)
    finally:
        env.close()

    print(f"Saved Isaac Lab comparison video: {output_path}")
    return output_path


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in config file: {config_path}")

    config: dict[str, Any] = {
        "task": "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        "dataset_file": "../dataset/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5",
        "checkpoint": "../outputs_wm_dynamics/run_20260622_100149/best.pt",
        "output": "../eval_outputs/isaaclab_gt_vs_wm.mp4",
        "episode_index": 0,
        "episode_name": None,
        "start_t": 25,
        "rollout_steps": 200,
        "max_frames": 250,
        "fps": 25,
        "video_width": 1280,
        "video_height": 720,
        "env_spacing": 2.0,
        "warmup_frames": 5,
        "apply_domain_params": True,
        "clamp_imagination_joints": True,
        "couple_imagination_gripper": True,
        "use_ground_truth_gripper": False,
        "camera_eye": None,
        "camera_target": None,
    }
    config.update(raw)
    if "video_width" not in raw and "width" in raw:
        config["video_width"] = raw["width"]
    if "video_height" not in raw and "height" in raw:
        config["video_height"] = raw["height"]

    cli_keys = (
        "dataset_file",
        "checkpoint",
        "output",
        "task",
        "episode_index",
        "episode_name",
        "start_t",
        "rollout_steps",
        "max_frames",
        "fps",
        "video_width",
        "video_height",
        "env_spacing",
        "warmup_frames",
    )
    for key in cli_keys:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.no_apply_domain_params:
        config["apply_domain_params"] = False
    if args.no_clamp_imagination_joints:
        config["clamp_imagination_joints"] = False
    if args.no_couple_imagination_gripper:
        config["couple_imagination_gripper"] = False
    if args.use_ground_truth_gripper:
        config["use_ground_truth_gripper"] = True

    base_dir = config_path.parent
    for key in ("dataset_file", "checkpoint", "output"):
        config[key] = resolve_path(config[key], base_dir)
    config["episode_name"] = config.get("episode_name") or None
    return config


def resolve_path(value: str | os.PathLike[str], base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_visual_episode(
    dataset_file: str,
    *,
    episode_index: int,
    episode_name: str | None,
    robot_dof: int,
    max_frames: int = 0,
    domain_t: int = 0,
) -> VisualEpisode:
    resolved_name = resolve_episode_name(dataset_file, episode_index, episode_name)
    with h5_open(dataset_file) as file:
        episode = file["data"][resolved_name]
        object_pose = require_array(episode, "states/rigid_object/object/root_pose", dtype=np.float32)
        object_velocity = require_array(episode, "states/rigid_object/object/root_velocity", dtype=np.float32)
        joint_pos = optional_array(episode, "states/articulation/robot/joint_position", dtype=np.float32)
        if joint_pos is None:
            joint_pos = require_array(episode, "obs/joint_pos", dtype=np.float32)
        joint_vel = optional_array(episode, "states/articulation/robot/joint_velocity", dtype=np.float32)
        if joint_vel is None:
            joint_vel = require_array(episode, "obs/joint_vel", dtype=np.float32)

        robot_root_pose = optional_array(episode, "states/articulation/robot/root_pose", dtype=np.float32)
        robot_root_velocity = optional_array(episode, "states/articulation/robot/root_velocity", dtype=np.float32)
        initial_robot_root = optional_array(episode, "initial_state/articulation/robot/root_pose", dtype=np.float32)

        t_count = min(object_pose.shape[0], object_velocity.shape[0], joint_pos.shape[0], joint_vel.shape[0])
        if max_frames > 0:
            t_count = min(t_count, max_frames)
        if robot_root_pose is None:
            if initial_robot_root is None:
                raise KeyError(
                    f"Episode '{resolved_name}' has no states/articulation/robot/root_pose or initial robot root pose."
                )
            robot_root_pose = np.repeat(initial_robot_root.reshape(-1, initial_robot_root.shape[-1])[:1], t_count, axis=0)
        if robot_root_velocity is None:
            robot_root_velocity = np.zeros((t_count, 6), dtype=np.float32)
        t_count = min(t_count, robot_root_pose.shape[0], robot_root_velocity.shape[0])

        if initial_robot_root is not None:
            source_origin = initial_robot_root.reshape(-1, initial_robot_root.shape[-1])[0, :3].astype(np.float32)
        else:
            source_origin = robot_root_pose[0, :3].astype(np.float32)

        domain = load_domain_parameters(episode, t_index=domain_t, episode_T=t_count)

    return VisualEpisode(
        name=resolved_name,
        robot_root_pose_w=fit_last_dim(robot_root_pose[:t_count], 7).astype(np.float32),
        robot_root_velocity_w=fit_last_dim(robot_root_velocity[:t_count], 6).astype(np.float32),
        joint_pos_abs=fit_last_dim(joint_pos[:t_count], robot_dof).astype(np.float32),
        joint_vel=fit_last_dim(joint_vel[:t_count], robot_dof).astype(np.float32),
        object_root_pose_w=fit_last_dim(object_pose[:t_count], 7).astype(np.float32),
        object_root_velocity_w=fit_last_dim(object_velocity[:t_count], 6).astype(np.float32),
        source_origin_w=source_origin,
        domain=domain,
    )


def build_comparison_tracks(
    *,
    model: torch.nn.Module,
    wm_cfg,
    visual_episode: VisualEpisode,
    wm_episode,
    start_t: int,
    rollout_steps: int,
    max_frames: int,
    device: torch.device,
) -> ComparisonTracks:
    start_t = max(int(start_t), int(wm_cfg.history_len) - 1)
    start_t = min(start_t, visual_episode.T - 2, wm_episode.T - 2)
    available = min(visual_episode.T, wm_episode.T) - start_t - 1
    steps = available if rollout_steps <= 0 else min(int(rollout_steps), available)
    if steps <= 0:
        raise ValueError(f"No rollout room for episode '{visual_episode.name}': start_t={start_t}, T={visual_episode.T}.")

    rollout = rollout_episode(
        model,
        wm_episode,
        start_t=start_t,
        rollout_steps=steps,
        history_len=wm_cfg.history_len,
        device=device,
    )
    if rollout.steps <= 0:
        raise RuntimeError(f"World-model rollout failed immediately: {rollout.failure_reason}")

    frame_count = rollout.steps + 1
    if max_frames > 0:
        frame_count = min(frame_count, int(max_frames))
    gt_indices = np.arange(start_t, start_t + frame_count, dtype=np.int64)

    gt = RenderTrack(
        robot_root_pose_w=visual_episode.robot_root_pose_w[gt_indices],
        robot_root_velocity_w=visual_episode.robot_root_velocity_w[gt_indices],
        joint_pos_abs=visual_episode.joint_pos_abs[gt_indices],
        joint_vel=visual_episode.joint_vel[gt_indices],
        object_root_pose_w=visual_episode.object_root_pose_w[gt_indices],
        object_root_velocity_w=visual_episode.object_root_velocity_w[gt_indices],
    )
    wm = make_world_model_track(
        rollout.pred_states[: frame_count - 1],
        visual_episode=visual_episode,
        wm_episode=wm_episode,
        wm_cfg=wm_cfg,
        start_t=start_t,
    )
    return ComparisonTracks(
        gt=gt,
        wm=wm,
        episode_name=visual_episode.name,
        start_t=start_t,
        dt=float(wm_cfg.dt),
        failed_step=rollout.failed_step,
        failure_reason=rollout.failure_reason,
    )


def make_world_model_track(
    pred_states: np.ndarray,
    *,
    visual_episode: VisualEpisode,
    wm_episode,
    wm_cfg,
    start_t: int,
) -> RenderTrack:
    frame_count = int(pred_states.shape[0]) + 1
    src_indices = np.arange(start_t, start_t + frame_count, dtype=np.int64)

    robot_root_pose = visual_episode.robot_root_pose_w[src_indices].copy()
    robot_root_velocity = visual_episode.robot_root_velocity_w[src_indices].copy()
    joint_pos = visual_episode.joint_pos_abs[src_indices].copy()
    joint_vel = visual_episode.joint_vel[src_indices].copy()
    object_pose = visual_episode.object_root_pose_w[src_indices].copy()
    object_velocity = visual_episode.object_root_velocity_w[src_indices].copy()

    if pred_states.size == 0:
        return RenderTrack(robot_root_pose, robot_root_velocity, joint_pos, joint_vel, object_pose, object_velocity)

    layout = wm_episode.state_layout
    joint_offset = wm_episode.joint_pos_abs[start_t] - wm_episode.state[start_t, layout.robot_q_slice]
    pred_joint_pos = pred_states[:, layout.robot_q_slice] + joint_offset[None, :]
    joint_pos[1:] = fit_last_dim(pred_joint_pos, joint_pos.shape[-1])
    if layout.has_joint_vel:
        joint_vel[1:] = fit_last_dim(pred_states[:, layout.robot_dq_slice], joint_vel.shape[-1])
    else:
        joint_vel[1:] = fit_last_dim(
            finite_difference_rows(joint_pos[:frame_count], float(wm_cfg.dt))[1:],
            joint_vel.shape[-1],
        )

    pred_object_pos = pred_states[:, layout.object_pos_slice]
    if bool(wm_cfg.subtract_env_origin):
        pred_object_pos = pred_object_pos + visual_episode.source_origin_w[None, :]
    pred_object_quat = normalize_quat_array(pred_states[:, layout.object_quat_slice])

    object_pose[1:, :3] = pred_object_pos
    object_pose[1:, 3:7] = pred_object_quat
    if layout.has_object_lin_vel:
        object_velocity[1:, :3] = pred_states[:, layout.object_lin_vel_slice]
    else:
        object_velocity[1:, :3] = finite_difference_rows(object_pose[:frame_count, :3], float(wm_cfg.dt))[1:]
    if layout.has_object_ang_vel:
        object_velocity[1:, 3:6] = pred_states[:, layout.object_ang_vel_slice]
    else:
        object_velocity[1:, 3:6] = angular_velocity_from_quat_rows(object_pose[:frame_count, 3:7], float(wm_cfg.dt))[1:]
    return RenderTrack(robot_root_pose, robot_root_velocity, joint_pos, joint_vel, object_pose, object_velocity)


def render_comparison_video(env, tracks: ComparisonTracks, episode: VisualEpisode, config: dict[str, Any]) -> str:
    width = int(config["video_width"])
    height = int(config["video_height"])
    comparison_env_ids = compute_comparison_env_ids(env, config)
    writer = FfmpegVideoWriter(config["output"], width=width, height=height, fps=int(config["fps"]))
    try:
        for _ in range(max(0, int(config["warmup_frames"]))):
            write_comparison_state(env, tracks, episode, config, frame=0, env_ids=comparison_env_ids)
            env.render()
        for frame in range(tracks.T):
            write_comparison_state(env, tracks, episode, config, frame=frame, env_ids=comparison_env_ids)
            image = env.render()
            if image is None:
                raise RuntimeError("Environment render returned None; render_mode must be 'rgb_array'.")
            image = annotate_frame(np.asarray(image), tracks, episode, config, frame)
            writer.write(image)
    finally:
        writer.close()
    return os.path.abspath(config["output"])


def write_comparison_state(
    env,
    tracks: ComparisonTracks,
    episode: VisualEpisode,
    config: dict[str, Any],
    *,
    frame: int,
    env_ids: torch.Tensor,
) -> None:
    device = env.device
    env_origins = env.scene.env_origins[env_ids].detach().cpu().numpy()

    robot_root_pose = np.stack(
        [
            rebase_pose(tracks.gt.robot_root_pose_w[frame], episode.source_origin_w, env_origins[0]),
            rebase_pose(tracks.wm.robot_root_pose_w[frame], episode.source_origin_w, env_origins[1]),
        ],
        axis=0,
    )
    object_root_pose = np.stack(
        [
            rebase_pose(tracks.gt.object_root_pose_w[frame], episode.source_origin_w, env_origins[0]),
            rebase_pose(tracks.wm.object_root_pose_w[frame], episode.source_origin_w, env_origins[1]),
        ],
        axis=0,
    )
    robot_root_velocity = np.stack([tracks.gt.robot_root_velocity_w[frame], tracks.wm.robot_root_velocity_w[frame]], axis=0)
    object_root_velocity = np.stack(
        [tracks.gt.object_root_velocity_w[frame], tracks.wm.object_root_velocity_w[frame]], axis=0
    )
    joint_pos = np.stack([tracks.gt.joint_pos_abs[frame], tracks.wm.joint_pos_abs[frame]], axis=0)
    joint_vel = np.stack([tracks.gt.joint_vel[frame], tracks.wm.joint_vel[frame]], axis=0)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    joint_pos = fit_last_dim(joint_pos, robot.num_joints)
    joint_vel = fit_last_dim(joint_vel, robot.num_joints)
    sanitize_imagined_joint_state(robot, joint_pos, joint_vel, env_ids, config)
    robot.write_root_pose_to_sim(torch.as_tensor(robot_root_pose, dtype=torch.float32, device=device), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.as_tensor(robot_root_velocity, dtype=torch.float32, device=device), env_ids=env_ids)
    robot.write_joint_state_to_sim(
        torch.as_tensor(joint_pos, dtype=torch.float32, device=device),
        torch.as_tensor(joint_vel, dtype=torch.float32, device=device),
        env_ids=env_ids,
    )
    obj.write_root_pose_to_sim(torch.as_tensor(object_root_pose, dtype=torch.float32, device=device), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.as_tensor(object_root_velocity, dtype=torch.float32, device=device), env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def sanitize_imagined_joint_state(
    robot,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    env_ids: torch.Tensor,
    config: dict[str, Any],
) -> None:
    if joint_pos.shape[0] < 2:
        return

    bad_q = ~np.isfinite(joint_pos[1])
    bad_dq = ~np.isfinite(joint_vel[1])
    if np.any(bad_q):
        joint_pos[1, bad_q] = joint_pos[0, bad_q]
    if np.any(bad_dq):
        joint_vel[1, bad_dq] = joint_vel[0, bad_dq]

    limits = get_robot_joint_pos_limits(robot, env_ids, joint_pos.shape[-1])
    if bool(config.get("clamp_imagination_joints", True)) and limits is not None:
        lower, upper = limits
        valid = np.isfinite(lower) & np.isfinite(upper)
        if np.any(valid):
            joint_pos[1, valid] = np.clip(joint_pos[1, valid], lower[valid], upper[valid])

    finger_ids = find_gripper_joint_ids(robot, joint_pos.shape[-1])
    if not finger_ids:
        return
    if bool(config.get("use_ground_truth_gripper", False)):
        joint_pos[1, finger_ids] = joint_pos[0, finger_ids]
        joint_vel[1, finger_ids] = joint_vel[0, finger_ids]
        return
    if not bool(config.get("couple_imagination_gripper", True)):
        return

    finger_q = float(np.mean(joint_pos[1, finger_ids]))
    finger_dq = float(np.mean(joint_vel[1, finger_ids]))
    if limits is not None:
        lower, upper = limits
        lo = float(np.max(lower[finger_ids]))
        hi = float(np.min(upper[finger_ids]))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            finger_q = float(np.clip(finger_q, lo, hi))
    joint_pos[1, finger_ids] = finger_q
    joint_vel[1, finger_ids] = finger_dq


def get_robot_joint_pos_limits(
    robot,
    env_ids: torch.Tensor,
    num_joints: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    data = getattr(robot, "data", None)
    limits = getattr(data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = getattr(data, "joint_pos_limits", None)
    if limits is None:
        return None

    limits_np = limits.detach().cpu().numpy() if isinstance(limits, torch.Tensor) else np.asarray(limits)
    if limits_np.ndim == 3:
        wm_env_id = 0
        if env_ids.numel() >= 2:
            wm_env_id = int(env_ids.detach().cpu()[1].item())
        limits_np = limits_np[min(max(wm_env_id, 0), limits_np.shape[0] - 1)]
    if limits_np.ndim != 2 or limits_np.shape[-1] < 2:
        return None

    lower = fit_last_dim(limits_np[:, 0], num_joints, fill=-np.inf)
    upper = fit_last_dim(limits_np[:, 1], num_joints, fill=np.inf)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return lo.astype(np.float32), hi.astype(np.float32)


def find_gripper_joint_ids(robot, num_joints: int) -> list[int]:
    names = getattr(robot, "joint_names", None) or []
    finger_ids = [idx for idx, name in enumerate(names) if "finger" in str(name).lower()]
    finger_ids = [idx for idx in finger_ids if idx < num_joints]
    if len(finger_ids) >= 2:
        return finger_ids[:2]
    if num_joints >= 9:
        return [7, 8]
    return []


def configure_camera(env, config: dict[str, Any]) -> None:
    origins = env.scene.env_origins[:2].detach().cpu().numpy()
    auto_target = origins.mean(axis=0) + np.asarray([0.5, 0.0, 0.35], dtype=np.float32)
    target = np.asarray(config["camera_target"] if config.get("camera_target") is not None else auto_target, dtype=np.float32)
    auto_eye = target + np.asarray([0.0, -3.2, 1.25], dtype=np.float32)
    eye = np.asarray(config["camera_eye"] if config.get("camera_eye") is not None else auto_eye, dtype=np.float32)
    config["_camera_eye_resolved"] = eye.tolist()
    config["_camera_target_resolved"] = target.tolist()
    env.sim.set_camera_view(tuple(float(x) for x in eye), tuple(float(x) for x in target), env.cfg.viewer.cam_prim_path)


def compute_comparison_env_ids(env, config: dict[str, Any]) -> torch.Tensor:
    """Return env ids ordered as [screen-left, screen-right]."""
    origins = env.scene.env_origins[:2].detach().cpu().numpy()
    if origins.shape[0] < 2:
        raise ValueError("Isaac Lab comparison rendering requires at least two environments.")
    eye = np.asarray(config.get("_camera_eye_resolved"), dtype=np.float32)
    target = np.asarray(config.get("_camera_target_resolved"), dtype=np.float32)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1.0e-8:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        forward = forward / forward_norm
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        right_norm = np.linalg.norm(right)
        right = right / right_norm if right_norm > 1.0e-8 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    screen_x = (origins - target[None, :]) @ right
    order = np.argsort(screen_x, kind="stable")
    left_env = int(order[0])
    right_env = int(order[-1])
    print(
        "[INFO] Comparison layout: "
        f"env {left_env} (screen-left) = ground truth, env {right_env} (screen-right) = world model."
    )
    return torch.tensor([left_env, right_env], dtype=torch.int64, device=env.device)


def annotate_frame(
    frame: np.ndarray,
    tracks: ComparisonTracks,
    episode: VisualEpisode,
    config: dict[str, Any],
    frame_idx: int,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    width, _height = image.size
    object_error = float(
        np.linalg.norm(tracks.gt.object_root_pose_w[frame_idx, :3] - tracks.wm.object_root_pose_w[frame_idx, :3])
    )
    time_s = (tracks.start_t + frame_idx) * tracks.dt
    mass_text = domain_scalar_text(episode.domain.get("object_mass"), "m")
    mat_text = domain_vector_text(episode.domain.get("object_material"), "mu", n=2)
    header = f"Episode {tracks.episode_name} | step {tracks.start_t + frame_idx} | t={time_s:.2f}s | object err={object_error:.4f}m"
    if mass_text or mat_text:
        header += " | " + " ".join(part for part in (mass_text, mat_text) if part)
    draw_label(draw, (16, 14), header, font, fill=(255, 255, 255, 255), bg=(0, 0, 0, 150))
    draw_label(draw, (16, 46), "Ground truth replay", font, fill=(80, 220, 255, 255), bg=(0, 0, 0, 130))
    draw_label(
        draw,
        (width - 230, 46),
        "World model imagination",
        font,
        fill=(255, 190, 80, 255),
        bg=(0, 0, 0, 130),
    )
    if tracks.failed_step is not None and tracks.failure_reason:
        draw_label(draw, (16, 78), f"WM rollout truncated: {tracks.failure_reason}", font, bg=(120, 0, 0, 150))
    return np.asarray(image)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, *, fill, bg) -> None:
    bbox = draw.textbbox(xy, text, font=font)
    pad = 5
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rounded_rectangle(rect, radius=4, fill=bg)
    draw.text(xy, text, font=font, fill=fill)


def apply_episode_domain_parameters(env, domain: dict[str, DomainValue]) -> None:
    env_ids = torch.tensor([0, 1], dtype=torch.int64, device=env.device)
    env_ids_cpu = env_ids.cpu()
    obj = env.scene["object"]
    robot = env.scene["robot"]

    applied: list[str] = []
    if "object_mass" in domain:
        if set_rigid_masses(obj, domain["object_mass"].value, env_ids_cpu):
            applied.append(f"object_mass <- {domain['object_mass'].source}")
    if "object_inertia" in domain:
        if set_rigid_inertias(obj, domain["object_inertia"].value, env_ids_cpu):
            applied.append(f"object_inertia <- {domain['object_inertia'].source}")
    if "object_material" in domain:
        if set_rigid_material(obj, domain["object_material"].value, env_ids_cpu):
            applied.append(f"object_material <- {domain['object_material'].source}")
    if "robot_link_masses" in domain:
        if set_articulation_masses(robot, domain["robot_link_masses"].value, env_ids_cpu):
            applied.append(f"robot_link_masses <- {domain['robot_link_masses'].source}")
    if "robot_joint_friction" in domain:
        values = repeat_joint_values(domain["robot_joint_friction"].value, len(env_ids), robot.num_joints, env.device)
        dynamic = None
        if "robot_joint_dynamic_friction" in domain:
            dynamic = repeat_joint_values(domain["robot_joint_dynamic_friction"].value, len(env_ids), robot.num_joints, env.device)
        robot.write_joint_friction_coefficient_to_sim(values, joint_dynamic_friction_coeff=dynamic, env_ids=env_ids)
        applied.append(f"robot_joint_friction <- {domain['robot_joint_friction'].source}")
    if "robot_joint_damping" in domain:
        robot.write_joint_damping_to_sim(
            repeat_joint_values(domain["robot_joint_damping"].value, len(env_ids), robot.num_joints, env.device),
            env_ids=env_ids,
        )
        applied.append(f"robot_joint_damping <- {domain['robot_joint_damping'].source}")
    if "robot_joint_armature" in domain:
        robot.write_joint_armature_to_sim(
            repeat_joint_values(domain["robot_joint_armature"].value, len(env_ids), robot.num_joints, env.device),
            env_ids=env_ids,
        )
        applied.append(f"robot_joint_armature <- {domain['robot_joint_armature'].source}")
    if "robot_joint_stiffness" in domain:
        robot.write_joint_stiffness_to_sim(
            repeat_joint_values(domain["robot_joint_stiffness"].value, len(env_ids), robot.num_joints, env.device),
            env_ids=env_ids,
        )
        applied.append(f"robot_joint_stiffness <- {domain['robot_joint_stiffness'].source}")

    if applied:
        print("[INFO] Applied episode domain parameters:")
        for item in applied:
            print(f"  - {item}")
    else:
        print("[WARN] No episode domain parameters were found to apply.")


def set_rigid_masses(asset, value: np.ndarray, env_ids_cpu: torch.Tensor) -> bool:
    try:
        masses = asset.root_physx_view.get_masses()
        values = torch.as_tensor(np.asarray(value, dtype=np.float32).reshape(-1), dtype=masses.dtype)
        if values.numel() == 1:
            masses[env_ids_cpu, :] = values.item()
        else:
            n = min(masses.shape[-1], values.numel())
            masses[env_ids_cpu, :n] = values[:n]
        asset.root_physx_view.set_masses(masses, env_ids_cpu)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set rigid mass: {exc}")
        return False


def set_rigid_inertias(asset, value: np.ndarray, env_ids_cpu: torch.Tensor) -> bool:
    try:
        inertias = asset.root_physx_view.get_inertias()
        values = torch.as_tensor(np.asarray(value, dtype=np.float32).reshape(-1), dtype=inertias.dtype)
        if values.numel() < 9:
            return False
        if inertias.ndim == 3:
            inertias[env_ids_cpu, :, :] = values[:9].reshape(1, 1, 9)
        else:
            inertias[env_ids_cpu, :] = values[:9]
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set rigid inertia: {exc}")
        return False


def set_rigid_material(asset, value: np.ndarray, env_ids_cpu: torch.Tensor) -> bool:
    try:
        materials = asset.root_physx_view.get_material_properties()
        values = torch.as_tensor(np.asarray(value, dtype=np.float32).reshape(-1), dtype=materials.dtype)
        if values.numel() < 3:
            return False
        if materials.ndim == 3:
            materials[env_ids_cpu, :, :3] = values[:3].reshape(1, 1, 3)
        else:
            materials[env_ids_cpu, :3] = values[:3]
        asset.root_physx_view.set_material_properties(materials, env_ids_cpu)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set rigid material: {exc}")
        return False


def set_articulation_masses(asset, value: np.ndarray, env_ids_cpu: torch.Tensor) -> bool:
    try:
        masses = asset.root_physx_view.get_masses()
        values = torch.as_tensor(np.asarray(value, dtype=np.float32).reshape(-1), dtype=masses.dtype)
        if values.numel() == 0:
            return False
        n = min(masses.shape[-1], values.numel())
        masses[env_ids_cpu, :n] = values[:n]
        asset.root_physx_view.set_masses(masses, env_ids_cpu)
        return True
    except Exception as exc:
        print(f"[WARN] Could not set robot link masses: {exc}")
        return False


def repeat_joint_values(value: np.ndarray, num_envs: int, num_joints: int, device: torch.device) -> torch.Tensor:
    values = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros((num_envs, num_joints), dtype=np.float32)
    if values.size == 1:
        out[:, :] = float(values[0])
    else:
        n = min(num_joints, values.size)
        out[:, :n] = values[:n]
    return torch.as_tensor(out, dtype=torch.float32, device=device)


def load_domain_parameters(episode, *, t_index: int, episode_T: int) -> dict[str, DomainValue]:
    specs = {
        "object_mass": (
            "object_dynamics/mass",
            "object_dynamics/object_mass",
            "obs/object_mass",
            "object_mass",
        ),
        "object_inertia": ("object_dynamics/inertia", "object_dynamics/inertial"),
        "object_material": ("object_dynamics/material_properties", "object_dynamics/material"),
        "robot_link_masses": (
            "robot_dynamics/robot_link_masses",
            "robot_dynamics/raw_masses",
            "robot_dynamics/mass",
            "obs/robot_link_masses",
            "robot_link_masses",
        ),
        "robot_joint_friction": (
            "robot_dynamics/robot_joint_params/joint_friction_coeff",
            "robot_dynamics/joint_friction_coeff",
            "robot_joint_params/joint_friction_coeff",
        ),
        "robot_joint_dynamic_friction": (
            "robot_dynamics/robot_joint_params/joint_dynamic_friction_coeff",
            "robot_dynamics/joint_dynamic_friction_coeff",
            "robot_joint_params/joint_dynamic_friction_coeff",
        ),
        "robot_joint_damping": (
            "robot_dynamics/robot_joint_params/joint_damping",
            "robot_dynamics/joint_damping",
            "robot_joint_params/joint_damping",
        ),
        "robot_joint_armature": (
            "robot_dynamics/robot_joint_params/joint_armature",
            "robot_dynamics/joint_armature",
            "robot_joint_params/joint_armature",
        ),
        "robot_joint_stiffness": (
            "robot_dynamics/robot_joint_params/joint_stiffness",
            "robot_dynamics/joint_stiffness",
            "robot_joint_params/joint_stiffness",
        ),
    }
    domain: dict[str, DomainValue] = {}
    for name, paths in specs.items():
        for path in paths:
            arr = optional_array(episode, path, dtype=np.float32)
            if arr is None:
                continue
            domain[name] = DomainValue(select_domain_row(arr, t_index=t_index, episode_T=episode_T), path)
            break

    if "robot_joint_friction" not in domain:
        arm = first_available_domain(
            episode,
            ("robot_dynamics/robot_arm_joint_friction", "obs/robot_arm_joint_friction"),
            t_index,
            episode_T,
        )
        gripper = first_available_domain(
            episode,
            ("robot_dynamics/robot_gripper_joint_friction", "obs/robot_gripper_joint_friction"),
            t_index,
            episode_T,
        )
        if arm is not None and gripper is not None:
            domain["robot_joint_friction"] = DomainValue(
                np.concatenate([arm.value.reshape(-1), gripper.value.reshape(-1)]),
                f"{arm.source}+{gripper.source}",
            )
        elif arm is not None:
            domain["robot_joint_friction"] = arm
    return domain


def first_available_domain(episode, paths: tuple[str, ...], t_index: int, episode_T: int) -> DomainValue | None:
    for path in paths:
        arr = optional_array(episode, path, dtype=np.float32)
        if arr is not None:
            return DomainValue(select_domain_row(arr, t_index=t_index, episode_T=episode_T), path)
    return None


def select_domain_row(arr: np.ndarray, *, t_index: int, episode_T: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim >= 2 and values.shape[0] == episode_T:
        values = values[min(max(0, t_index), values.shape[0] - 1)]
    elif values.ndim >= 2 and values.shape[0] == 1:
        values = values[0]
    return np.asarray(values, dtype=np.float32).reshape(-1)


def require_array(group, path: str, *, dtype) -> np.ndarray:
    arr = optional_array(group, path, dtype=dtype)
    if arr is None:
        raise KeyError(f"Missing HDF5 dataset: {group.name}/{path}")
    return arr


def optional_array(group, path: str, *, dtype) -> np.ndarray | None:
    node = group
    for part in path.split("/"):
        if part not in node:
            return None
        node = node[part]
    try:
        return np.asarray(node, dtype=dtype)
    except TypeError:
        return None


def fit_last_dim(values: np.ndarray, dim: int, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    out = np.full(arr.shape[:-1] + (dim,), fill, dtype=np.float32)
    n = min(dim, arr.shape[-1])
    out[..., :n] = arr[..., :n]
    return out


def normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    quat = fit_last_dim(quat, 4)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    safe = np.where(norm > 1.0e-8, norm, 1.0)
    out = quat / safe
    bad = norm[..., 0] <= 1.0e-8
    if np.any(bad):
        out[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def finite_difference_rows(values: np.ndarray, dt: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] <= 1:
        return out
    safe_dt = max(float(dt), 1.0e-8)
    out[1:] = (arr[1:] - arr[:-1]) / safe_dt
    out[0] = out[1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def angular_velocity_from_quat_rows(quat: np.ndarray, dt: float) -> np.ndarray:
    q = normalize_quat_array(quat)
    out = np.zeros((q.shape[0], 3), dtype=np.float32)
    if q.shape[0] <= 1:
        return out
    safe_dt = max(float(dt), 1.0e-8)
    q_prev_inv = q[:-1].copy()
    q_prev_inv[:, 1:] *= -1.0
    dq = quat_multiply_rows(q[1:], q_prev_inv)
    dq = np.where(dq[:, :1] < 0.0, -dq, dq)
    xyz = dq[:, 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    w = np.clip(dq[:, :1], -1.0, 1.0)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    axis = np.divide(xyz, np.maximum(xyz_norm, 1.0e-8), out=np.zeros_like(xyz), where=xyz_norm > 1.0e-8)
    out[1:] = axis * angle / safe_dt
    out[0] = out[1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def quat_multiply_rows(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = normalize_quat_array(lhs)
    b = normalize_quat_array(rhs)
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def rebase_pose(pose_w: np.ndarray, source_origin_w: np.ndarray, target_origin_w: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w, dtype=np.float32).copy()
    pose[:3] = pose[:3] - source_origin_w + target_origin_w
    pose[3:7] = normalize_quat_array(pose[3:7][None])[0]
    return pose


def domain_scalar_text(value: DomainValue | None, label: str) -> str:
    if value is None or value.value.size == 0:
        return ""
    return f"{label}={float(value.value.reshape(-1)[0]):.3g}"


def domain_vector_text(value: DomainValue | None, label: str, *, n: int) -> str:
    if value is None or value.value.size == 0:
        return ""
    vals = value.value.reshape(-1)[:n]
    joined = ",".join(f"{float(v):.2g}" for v in vals)
    return f"{label}=({joined})"


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
