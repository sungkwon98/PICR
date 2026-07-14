#!/usr/bin/env python3
"""Collect policy-free Franka cube-drop rollouts with three RGB/RGB-D camera views.

The collector reuses grasped robot/object states from a Franka Lift HDF5 file.
Each new episode is initialized at one of those reachable held-cube states, the
arm is held fixed, and the gripper is commanded open for the entire rollout.
No policy or checkpoint is loaded.

The legacy robot/object groups are retained for compatibility with the world
model code in this directory.  RGB frames live in ``images/{front,left,right}``;
optional depth maps live in ``depth/{front,left,right}``; optional non-colorized
instance masks live in ``segmentation/instance/{front,left,right}``.  Camera
frames are captured post-step, at exactly the same time as ``states``.  The
held, pre-release frame is stored once in ``initial_state``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parents[1]
_WORLD_MODEL_DIR = _THIS_FILE.parents[2]
_REPO_ROOT = _THIS_FILE.parents[4]
_DEFAULT_SOURCE = _PACKAGE_DIR / "dataset" / "Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep.hdf5"
_DEFAULT_OUTPUT = _PACKAGE_DIR / "dataset" / "Franka_Lift_policy_free_cube_drop_multiview.hdf5"

for _path in (
    _REPO_ROOT / "source" / "isaaclab",
    _REPO_ROOT / "source" / "isaaclab_assets",
    _REPO_ROOT / "source" / "isaaclab_tasks",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from isaaclab.app import AppLauncher


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect policy-free Franka cube drops with front/left/right RGB images."
    )
    parser.add_argument("--pose-source-hdf5", type=str, default=str(_DEFAULT_SOURCE))
    parser.add_argument("--output-file", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=50, help="Recorded 50 Hz control steps per episode.")
    parser.add_argument("--decimation", type=int, default=2, help="Physics steps per recorded control step.")
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=25,
        help="Unrecorded arm-settling steps while the cube is anchored at its demonstrated grasp pose.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-held-height", type=float, default=0.18)
    parser.add_argument("--max-held-height", type=float, default=0.52)
    parser.add_argument("--max-held-speed", type=float, default=0.75)
    parser.add_argument("--max-held-angular-speed", type=float, default=5.0)
    parser.add_argument("--max-closed-finger-position", type=float, default=0.032)
    parser.add_argument("--max-grasp-distance", type=float, default=0.08)
    parser.add_argument("--arm-stiffness", type=float, default=400.0)
    parser.add_argument("--arm-damping", type=float, default=80.0)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-focal-length", type=float, default=24.0)
    parser.add_argument(
        "--no-rgb",
        action="store_true",
        help="Do not store RGB images. Useful when collecting depth-only data for pointcloud extraction.",
    )
    parser.add_argument(
        "--states-only",
        action="store_true",
        help=(
            "Do not create cameras or store image/depth/segmentation data. "
            "Use this for mesh/FK pointcloud generation from recorded robot/object states."
        ),
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Store float32 distance_to_image_plane depth maps for each camera view.",
    )
    parser.add_argument(
        "--include-instance-segmentation",
        action="store_true",
        help="Store non-colorized instance_segmentation_fast masks for pointcloud labeling.",
    )
    parser.add_argument(
        "--include-instance-id-segmentation",
        action="store_true",
        help="Store non-colorized instance_id_segmentation_fast masks for finer per-prim pointcloud labeling.",
    )
    parser.add_argument(
        "--depth-clipping-behavior",
        choices=("max", "zero", "none"),
        default="max",
        help="Depth values beyond the camera clipping range are clipped to max, set to zero, or left as inf.",
    )
    parser.add_argument(
        "--depth-storage",
        choices=("float32", "uint16_mm"),
        default="float32",
        help="Store depth as float32 meters or quantized uint16 millimeters.",
    )
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument(
        "--replay-physics-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay object mass/material and robot link-mass context from each source episode.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue a partially collected compatible output file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Write incomplete drops instead of stopping before the failed batch is committed.",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--disable-fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = not bool(args.states_only)
    return args


args_cli = parse_cli()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim imports must happen after AppLauncher creates the application."""

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (  # noqa: E402
    FrankaCubeLiftEnvCfg,
)


CAMERA_VIEWS = {
    "front": ((1.35, 0.0, 0.75), (0.50, 0.0, 0.28)),
    "left": ((0.65, 0.95, 0.75), (0.50, 0.0, 0.28)),
    "right": ((0.65, -0.95, 0.75), (0.50, 0.0, 0.28)),
}
# The Seattle Lab table support height inferred from the supplied Lift dataset.
# This is relative to each environment origin, not a world-space Z for every clone.
LIFT_TABLE_SUPPORT_Z = 0.028057001531124115


def camera_data_types(args: argparse.Namespace) -> list[str]:
    if bool(getattr(args, "states_only", False)):
        return []
    data_types = []
    if not bool(args.no_rgb):
        data_types.append("rgb")
    if bool(args.include_depth):
        data_types.append("distance_to_image_plane")
    if bool(args.include_instance_segmentation):
        data_types.append("instance_segmentation_fast")
    if bool(args.include_instance_id_segmentation):
        data_types.append("instance_id_segmentation_fast")
    return data_types


@dataclass(frozen=True)
class HeldPoseSample:
    source_demo: str
    source_frame: int
    joint_pos: np.ndarray
    object_pose: np.ndarray
    object_mass: float
    object_inertia: np.ndarray
    object_contact: np.ndarray
    robot_link_masses: np.ndarray
    robot_joint_friction: np.ndarray

    @property
    def height(self) -> float:
        return float(self.object_pose[2])


def _numeric_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return sys.maxsize, name


def _read_array(group: h5py.Group, path: str, default: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(group[path], dtype=np.float32)
    except KeyError:
        return np.asarray(default, dtype=np.float32)


def _extract_held_pose(episode: h5py.Group, name: str, args: argparse.Namespace) -> HeldPoseSample | None:
    try:
        joint_pos = np.asarray(episode["states/articulation/robot/joint_position"], dtype=np.float32)
        object_pose = np.asarray(episode["states/rigid_object/object/root_pose"], dtype=np.float32)
        object_vel = np.asarray(episode["states/rigid_object/object/root_velocity"], dtype=np.float32)
    except KeyError:
        return None

    count = min(joint_pos.shape[0], object_pose.shape[0], object_vel.shape[0])
    if count == 0 or joint_pos.shape[1] < 9 or object_pose.shape[1] < 7:
        return None
    joint_pos = joint_pos[:count]
    object_pose = object_pose[:count]
    object_vel = object_vel[:count]
    linear_speed = np.linalg.norm(object_vel[:, :3], axis=1)
    angular_speed = np.linalg.norm(object_vel[:, 3:6], axis=1)
    finite = np.isfinite(joint_pos).all(axis=1) & np.isfinite(object_pose).all(axis=1)
    valid = (
        finite
        & (joint_pos[:, 7:9] <= float(args.max_closed_finger_position)).all(axis=1)
        & (object_pose[:, 2] >= float(args.min_held_height))
        & (object_pose[:, 2] <= float(args.max_held_height))
        & (linear_speed <= float(args.max_held_speed))
        & (angular_speed <= float(args.max_held_angular_speed))
    )
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return None
    frame = int(valid_ids[np.argmax(object_pose[valid_ids, 2])])

    mass = _read_array(episode, "episode_physics_randomization/object_mass", np.array([[0.25]], np.float32))
    contact = _read_array(
        episode,
        "episode_physics_randomization/object_contact",
        np.array([[1.0, 1.0, 0.0]], np.float32),
    )
    link_masses = _read_array(
        episode,
        "episode_physics_randomization/robot_link_masses",
        np.empty((0, 11), np.float32),
    )
    arm_friction = _read_array(
        episode,
        "episode_physics_randomization/robot_arm_joint_friction",
        np.zeros((1, 7), np.float32),
    )
    gripper_friction = _read_array(
        episode,
        "episode_physics_randomization/robot_gripper_joint_friction",
        np.zeros((1, 2), np.float32),
    )
    inertia_series = _read_array(episode, "object_dynamics/inertia", np.empty((0, 9), np.float32))
    inertia = inertia_series[min(frame, inertia_series.shape[0] - 1)] if inertia_series.shape[0] else np.zeros(9)
    friction = np.concatenate((arm_friction.reshape(-1)[:7], gripper_friction.reshape(-1)[:2]))
    return HeldPoseSample(
        source_demo=name,
        source_frame=frame,
        joint_pos=joint_pos[frame, :9].copy(),
        object_pose=object_pose[frame, :7].copy(),
        object_mass=float(mass.reshape(-1)[0]),
        object_inertia=np.asarray(inertia, dtype=np.float32).reshape(9).copy(),
        object_contact=np.asarray(contact, dtype=np.float32).reshape(-1)[:3].copy(),
        robot_link_masses=np.asarray(link_masses, dtype=np.float32).reshape(-1)[:11].copy(),
        robot_joint_friction=np.asarray(friction, dtype=np.float32).reshape(9).copy(),
    )


def load_pose_sequence(path: str, args: argparse.Namespace) -> list[HeldPoseSample]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pose-source HDF5 does not exist: {path}")
    rng = np.random.default_rng(int(args.seed))
    samples: list[HeldPoseSample] = []
    with h5py.File(path, "r", locking=False) as file:
        if "data" not in file:
            raise KeyError(f"Missing /data in pose-source HDF5: {path}")
        names = sorted(file["data"].keys(), key=_numeric_demo_key)
        scan_order = rng.permutation(len(names))
        for scan_i, name_i in enumerate(scan_order, start=1):
            name = names[int(name_i)]
            sample = _extract_held_pose(file["data"][name], name, args)
            if sample is not None:
                samples.append(sample)
            if len(samples) >= int(args.num_episodes):
                break
            if scan_i % 1000 == 0:
                print(f"[INFO] Pose scan: {scan_i}/{len(names)} source episodes, {len(samples)} usable grasps")

    if not samples:
        raise RuntimeError(
            "No held poses passed the requested filters. Lower --min-held-height, increase --max-held-speed, "
            "or increase --max-closed-finger-position."
        )
    if len(samples) < int(args.num_episodes):
        base = samples.copy()
        while len(samples) < int(args.num_episodes):
            order = rng.permutation(len(base))
            samples.extend(base[int(i)] for i in order)
        samples = samples[: int(args.num_episodes)]
        print(f"[WARN] Only {len(base)} unique usable poses found; cycling them to make {len(samples)} episodes.")
    print(
        f"[INFO] Loaded {len(samples)} held-pose samples; height range "
        f"[{min(s.height for s in samples):.3f}, {max(s.height for s in samples):.3f}] m."
    )
    return samples


def make_camera_cfg(name: str, args: argparse.Namespace, control_dt: float) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Camera_{name}",
        update_period=control_dt,
        update_latest_camera_pose=True,
        data_types=camera_data_types(args),
        depth_clipping_behavior=str(args.depth_clipping_behavior),
        colorize_instance_segmentation=False,
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.camera_focal_length),
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        width=int(args.camera_width),
        height=int(args.camera_height),
    )


def build_simulation(args: argparse.Namespace):
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.sim.device = args.device
    env_cfg.sim.dt = float(args.physics_dt)
    env_cfg.sim.render_interval = int(args.decimation)
    env_cfg.sim.use_fabric = not bool(args.disable_fabric)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.ee_frame.debug_vis = False
    control_dt = float(args.physics_dt) * int(args.decimation)
    if not bool(args.states_only):
        for name in CAMERA_VIEWS:
            setattr(env_cfg.scene, f"{name}_camera", make_camera_cfg(name, args, control_dt))

    sim = sim_utils.SimulationContext(env_cfg.sim)
    scene = InteractiveScene(env_cfg.scene)
    sim.reset()

    if bool(args.states_only):
        print(f"[INFO] Simulation is ready in states-only mode ({scene.num_envs} envs).")
    else:
        origins = scene.env_origins
        for name, (eye_local, target_local) in CAMERA_VIEWS.items():
            camera = scene[f"{name}_camera"]
            eye = origins + torch.tensor(eye_local, dtype=torch.float32, device=scene.device)
            target = origins + torch.tensor(target_local, dtype=torch.float32, device=scene.device)
            camera.set_world_poses_from_view(eye, target)
        for _ in range(3):
            sim.render()
            for name in CAMERA_VIEWS:
                scene[f"{name}_camera"].update(0.0, force_recompute=True)
        print(f"[INFO] Simulation and {len(CAMERA_VIEWS)} tiled camera views are ready ({scene.num_envs} envs).")
    return sim, scene


def as_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _assign_last_dim(target: torch.Tensor, env_id: int, value: np.ndarray) -> bool:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return False
    row = target[env_id]
    if row.ndim == 1:
        n = min(row.shape[0], flat.size)
        row[:n] = torch.as_tensor(flat[:n], dtype=row.dtype)
    else:
        n = min(row.shape[-1], flat.size)
        row[..., :n] = torch.as_tensor(flat[:n], dtype=row.dtype)
    return True


def apply_physics_context(scene: InteractiveScene, samples: list[HeldPoseSample], replay: bool) -> None:
    if not replay:
        return
    robot = scene["robot"]
    obj = scene["object"]
    count = scene.num_envs
    env_ids_cpu = torch.arange(count, dtype=torch.int64, device="cpu")

    object_masses = obj.data.default_mass.detach().cpu().clone()
    object_inertias = obj.data.default_inertia.detach().cpu().clone()
    object_materials = obj.root_physx_view.get_material_properties().clone()
    robot_masses = robot.data.default_mass.detach().cpu().clone()
    robot_inertias = robot.data.default_inertia.detach().cpu().clone()
    joint_friction = robot.data.default_joint_friction_coeff.detach().clone()

    for env_id, sample in enumerate(samples):
        default_object_mass = float(object_masses[env_id].reshape(-1)[0])
        object_masses[env_id].fill_(float(sample.object_mass))
        if sample.object_inertia.size == 9 and np.any(sample.object_inertia > 0.0):
            _assign_last_dim(object_inertias, env_id, sample.object_inertia)
        elif default_object_mass > 0.0:
            object_inertias[env_id] *= float(sample.object_mass) / default_object_mass
        if sample.object_contact.size >= 3:
            object_materials[env_id, ..., :3] = torch.as_tensor(
                sample.object_contact[:3], dtype=object_materials.dtype
            )

        if sample.robot_link_masses.size:
            n = min(robot_masses.shape[1], sample.robot_link_masses.size)
            desired = torch.as_tensor(sample.robot_link_masses[:n], dtype=robot_masses.dtype)
            default = robot_masses[env_id, :n].clone()
            robot_masses[env_id, :n] = desired
            ratios = desired / torch.clamp(default, min=1.0e-8)
            robot_inertias[env_id, :n] *= ratios[:, None]
        if sample.robot_joint_friction.size:
            n = min(joint_friction.shape[1], sample.robot_joint_friction.size)
            joint_friction[env_id, :n] = torch.as_tensor(
                sample.robot_joint_friction[:n], dtype=joint_friction.dtype, device=joint_friction.device
            )

    obj.root_physx_view.set_masses(object_masses, env_ids_cpu)
    obj.root_physx_view.set_inertias(object_inertias, env_ids_cpu)
    obj.root_physx_view.set_material_properties(object_materials, env_ids_cpu)
    robot.root_physx_view.set_masses(robot_masses, env_ids_cpu)
    robot.root_physx_view.set_inertias(robot_inertias, env_ids_cpu)
    robot.write_joint_friction_coefficient_to_sim(joint_friction, env_ids=robot._ALL_INDICES)


def prepare_batch(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    active_samples: list[HeldPoseSample],
    args: argparse.Namespace,
) -> tuple[list[HeldPoseSample], torch.Tensor, torch.Tensor]:
    robot = scene["robot"]
    obj = scene["object"]
    device = scene.device
    # Fill unused slots in the final batch with a valid duplicate; they are simulated but never written.
    samples = active_samples + [active_samples[-1]] * (scene.num_envs - len(active_samples))
    apply_physics_context(scene, samples, bool(args.replay_physics_context))

    # Match Isaac Lab's high-PD Franka preset so the arm stays at the sampled
    # pose without a policy continuously compensating gravity.
    stiffness = robot.data.joint_stiffness.clone()
    damping = robot.data.joint_damping.clone()
    stiffness[:, :7] = float(args.arm_stiffness)
    damping[:, :7] = float(args.arm_damping)
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)

    robot_root = robot.data.default_root_state.clone()
    robot_root[:, :3] += scene.env_origins
    joint_pos = robot.data.default_joint_pos.clone()
    object_pose = obj.data.default_root_state[:, :7].clone()
    object_velocity = torch.zeros((scene.num_envs, 6), dtype=torch.float32, device=device)
    for env_id, sample in enumerate(samples):
        joint_pos[env_id, :9] = torch.as_tensor(sample.joint_pos, dtype=torch.float32, device=device)
        object_pose[env_id, :3] = (
            torch.as_tensor(sample.object_pose[:3], dtype=torch.float32, device=device) + scene.env_origins[env_id]
        )
        object_pose[env_id, 3:7] = torch.as_tensor(sample.object_pose[3:7], dtype=torch.float32, device=device)

    robot.write_root_pose_to_sim(robot_root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(robot_root[:, 7:]))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    obj.write_root_pose_to_sim(object_pose)
    obj.write_root_velocity_to_sim(object_velocity)
    scene.reset()

    # The source finger positions already correspond to a physical grasp.  Keep
    # that opening instead of squeezing toward zero and injecting cube motion.
    held_target = joint_pos.clone()
    robot.set_joint_position_target(held_target)
    robot.set_joint_velocity_target(torch.zeros_like(held_target))
    for _ in range(max(0, int(args.settle_steps))):
        # Keep the cube at its demonstrated grasp pose while the arm settles
        # under normal gravity.  The cube is released only after this warm-up.
        obj.write_root_pose_to_sim(object_pose)
        obj.write_root_velocity_to_sim(object_velocity)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(float(args.physics_dt))

    # Start the recorded drop from the exact demonstrated pose and from rest.
    obj.write_root_pose_to_sim(object_pose)
    obj.write_root_velocity_to_sim(torch.zeros_like(object_velocity))
    robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
    sim.forward()
    scene.update(0.0)

    grasp_center = scene["ee_frame"].data.target_pos_w[:, 0]
    grasp_distance = torch.linalg.norm(obj.data.root_pos_w - grasp_center, dim=-1)
    bad = torch.nonzero(
        grasp_distance[: len(active_samples)] > float(args.max_grasp_distance), as_tuple=False
    ).squeeze(-1)
    if bad.numel() > 0:
        details = ", ".join(
            f"{samples[int(slot)].source_demo}: {float(grasp_distance[int(slot)]):.4f} m"
            for slot in bad.detach().cpu().tolist()
        )
        raise RuntimeError(
            f"Held-pose validation exceeded --max-grasp-distance={args.max_grasp_distance:.4f} m: {details}"
        )
    print(
        f"[INFO] Prepared {len(active_samples)} held cubes; max grasp-center distance="
        f"{float(grasp_distance[: len(active_samples)].max()):.4f} m."
    )

    open_target = joint_pos.clone()
    open_target[:, 7:9] = 0.04
    default_q = robot.data.default_joint_pos
    raw_action = torch.ones((scene.num_envs, 8), dtype=torch.float32, device=device)
    raw_action[:, :7] = (open_target[:, :7] - default_q[:, :7]) / 0.5
    return samples, open_target, raw_action


def capture_camera_observations(scene: InteractiveScene, args: argparse.Namespace) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    if bool(args.states_only):
        return output
    for name in CAMERA_VIEWS:
        camera_output = scene[f"{name}_camera"].data.output
        if "rgb" in camera_output:
            image = as_numpy(camera_output["rgb"], dtype=np.uint8)
            if image.ndim != 4 or image.shape[-1] < 3:
                raise RuntimeError(f"Unexpected {name} RGB shape: {image.shape}")
            output[f"images/{name}"] = np.ascontiguousarray(image[..., :3])

        if "distance_to_image_plane" in camera_output:
            depth = as_numpy(camera_output["distance_to_image_plane"], dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., None]
            if depth.ndim != 4 or depth.shape[-1] != 1:
                raise RuntimeError(f"Unexpected {name} depth shape: {depth.shape}")
            if str(args.depth_storage) == "uint16_mm":
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth = np.clip(np.rint(depth * 1000.0), 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
            output[f"depth/{name}"] = np.ascontiguousarray(depth)

        if "instance_segmentation_fast" in camera_output:
            instance = as_numpy(camera_output["instance_segmentation_fast"], dtype=np.uint32)
            if instance.ndim == 3:
                instance = instance[..., None]
            if instance.ndim != 4 or instance.shape[-1] != 1:
                raise RuntimeError(f"Unexpected {name} instance segmentation shape: {instance.shape}")
            output[f"segmentation/instance/{name}"] = np.ascontiguousarray(instance)

        if "instance_id_segmentation_fast" in camera_output:
            instance_id = as_numpy(camera_output["instance_id_segmentation_fast"], dtype=np.uint32)
            if instance_id.ndim == 3:
                instance_id = instance_id[..., None]
            if instance_id.ndim != 4 or instance_id.shape[-1] != 1:
                raise RuntimeError(f"Unexpected {name} instance id segmentation shape: {instance_id.shape}")
            output[f"segmentation/instance_id/{name}"] = np.ascontiguousarray(instance_id)
    return output


def capture_initial_state(scene: InteractiveScene, args: argparse.Namespace) -> dict[str, np.ndarray]:
    robot = scene["robot"]
    obj = scene["object"]
    origins = scene.env_origins
    robot_state = robot.data.root_state_w.clone()
    object_state = obj.data.root_state_w.clone()
    robot_state[:, :3] -= origins
    object_state[:, :3] -= origins
    initial = {
        "articulation/robot/joint_position": as_numpy(robot.data.joint_pos),
        "articulation/robot/joint_velocity": as_numpy(robot.data.joint_vel),
        "articulation/robot/root_pose": as_numpy(robot_state[:, :7]),
        "articulation/robot/root_velocity": as_numpy(robot_state[:, 7:]),
        "rigid_object/object/root_pose": as_numpy(object_state[:, :7]),
        "rigid_object/object/root_velocity": as_numpy(object_state[:, 7:]),
    }
    initial.update(capture_camera_observations(scene, args))
    return initial


def capture_constant_properties(scene: InteractiveScene) -> dict[str, np.ndarray]:
    robot = scene["robot"]
    obj = scene["object"]
    object_mass = as_numpy(obj.root_physx_view.get_masses()).reshape(scene.num_envs, -1)[:, :1]
    return {
        "object_dynamics/mass": object_mass,
        "object_dynamics/raw_masses": object_mass,
        "object_dynamics/inertia": as_numpy(obj.root_physx_view.get_inertias()).reshape(scene.num_envs, -1)[:, :9],
        "object_dynamics/com": as_numpy(obj.root_physx_view.get_coms()).reshape(scene.num_envs, -1, 7)[:, 0],
        "object_dynamics/material_properties": as_numpy(obj.root_physx_view.get_material_properties()),
        "robot_joint_params/default_joint_armature": as_numpy(robot.data.default_joint_armature),
        "robot_joint_params/default_joint_damping": as_numpy(robot.data.default_joint_damping),
        "robot_joint_params/default_joint_friction_coeff": as_numpy(robot.data.default_joint_friction_coeff),
        "robot_joint_params/joint_armature": as_numpy(robot.data.joint_armature),
        "robot_joint_params/joint_damping": as_numpy(robot.data.joint_damping),
        "robot_joint_params/joint_dynamic_friction_coeff": as_numpy(robot.data.joint_dynamic_friction_coeff),
        "robot_joint_params/joint_friction_coeff": as_numpy(robot.data.joint_friction_coeff),
        "robot_joint_params/joint_stiffness": as_numpy(robot.data.joint_stiffness),
        "episode_physics_randomization/object_mass": as_numpy(obj.root_physx_view.get_masses()).reshape(
            scene.num_envs, -1
        )[:, :1],
        "episode_physics_randomization/object_contact": as_numpy(
            obj.root_physx_view.get_material_properties()
        ).reshape(scene.num_envs, -1, 3)[:, 0],
        "episode_physics_randomization/robot_link_masses": as_numpy(robot.root_physx_view.get_masses()),
        "episode_physics_randomization/robot_arm_joint_friction": as_numpy(robot.data.joint_friction_coeff)[:, :7],
        "episode_physics_randomization/robot_gripper_joint_friction": as_numpy(
            robot.data.joint_friction_coeff
        )[:, 7:9],
    }


def capture_camera_info(scene: InteractiveScene) -> dict[str, np.ndarray]:
    info: dict[str, np.ndarray] = {}
    origins = as_numpy(scene.env_origins)
    for name in CAMERA_VIEWS:
        camera = scene[f"{name}_camera"]
        info[f"camera_info/{name}/intrinsic_matrix"] = as_numpy(camera.data.intrinsic_matrices)
        info[f"camera_info/{name}/position"] = as_numpy(camera.data.pos_w) - origins
        info[f"camera_info/{name}/quaternion_world_convention"] = as_numpy(camera.data.quat_w_world)
        info[f"camera_info/{name}/quaternion_ros_convention"] = as_numpy(camera.data.quat_w_ros)
    return info


def capture_camera_info_for_args(scene: InteractiveScene, args: argparse.Namespace) -> dict[str, np.ndarray]:
    if bool(args.states_only):
        return {}
    return capture_camera_info(scene)


def capture_step(
    scene: InteractiveScene,
    raw_action: torch.Tensor,
    properties: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    robot = scene["robot"]
    obj = scene["object"]
    origins = scene.env_origins
    robot_state = robot.data.root_state_w.clone()
    object_state = obj.data.root_state_w.clone()
    robot_state[:, :3] -= origins
    object_state_relative = object_state.clone()
    object_state_relative[:, :3] -= origins
    object_pos_robot, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )

    mass_matrix = as_numpy(robot.root_physx_view.get_generalized_mass_matrices())
    qdd = as_numpy(robot.data.joint_acc)
    coriolis = as_numpy(robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces())
    gravity = as_numpy(robot.root_physx_view.get_gravity_compensation_forces())
    inertial = np.einsum("nij,nj->ni", mass_matrix, qdd, optimize=True).astype(np.float32)

    object_acc = as_numpy(obj.data.body_com_acc_w)[:, 0]
    object_mass = properties["object_dynamics/mass"]
    gravity_acc = np.repeat(
        np.asarray(scene.sim.cfg.gravity, dtype=np.float32).reshape(1, 3), scene.num_envs, axis=0
    )
    gravity_force = object_mass * gravity_acc
    inertial_force = object_mass * object_acc[:, :3]

    step = {
        "actions": as_numpy(raw_action),
        "processed_actions": as_numpy(robot.data.joint_pos_target),
        "obs/actions": as_numpy(raw_action),
        "obs/joint_pos": as_numpy(robot.data.joint_pos - robot.data.default_joint_pos),
        "obs/joint_vel": as_numpy(robot.data.joint_vel),
        "obs/object_position": as_numpy(object_pos_robot),
        "obs/target_object_position": np.zeros((scene.num_envs, 7), dtype=np.float32),
        "states/articulation/robot/joint_position": as_numpy(robot.data.joint_pos),
        "states/articulation/robot/joint_velocity": as_numpy(robot.data.joint_vel),
        "states/articulation/robot/root_pose": as_numpy(robot_state[:, :7]),
        "states/articulation/robot/root_velocity": as_numpy(robot_state[:, 7:]),
        "states/rigid_object/object/root_pose": as_numpy(object_state_relative[:, :7]),
        "states/rigid_object/object/root_velocity": as_numpy(object_state_relative[:, 7:]),
        "robot_torques/applied_torque": as_numpy(robot.data.applied_torque),
        "robot_torques/computed_torque": as_numpy(robot.data.computed_torque),
        "robot_torques/joint_effort_target": as_numpy(robot.data.joint_effort_target),
        "robot_torques/joint_pos_target": as_numpy(robot.data.joint_pos_target),
        "robot_torques/joint_vel_target": as_numpy(robot.data.joint_vel_target),
        "robot_dynamics/mass_matrix": mass_matrix,
        "robot_dynamics/qdd": qdd,
        "robot_dynamics/inertial": inertial,
        "robot_dynamics/coriolis": coriolis,
        "robot_dynamics/gravity": gravity,
        "robot_dynamics/inverse_dynamics_tau": inertial + coriolis + gravity,
        "object_dynamics/root_pos_w": as_numpy(obj.data.root_pos_w),
        "object_dynamics/root_quat_w": as_numpy(obj.data.root_quat_w),
        "object_dynamics/root_lin_vel_w": as_numpy(obj.data.root_lin_vel_w),
        "object_dynamics/root_ang_vel_w": as_numpy(obj.data.root_ang_vel_w),
        "object_dynamics/root_lin_acc_w": object_acc[:, :3],
        "object_dynamics/root_ang_acc_w": object_acc[:, 3:],
        "object_dynamics/gravity_acc_w": gravity_acc,
        "object_dynamics/gravity_force_w": gravity_force,
        "object_dynamics/inertial_force_w": inertial_force,
        "object_dynamics/external_force_est_w": inertial_force - gravity_force,
    }
    step.update(capture_camera_observations(scene, args))
    return step


def collect_batch(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    active_samples: list[HeldPoseSample],
    args: argparse.Namespace,
) -> tuple[
    list[HeldPoseSample],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    samples, open_target, raw_action = prepare_batch(sim, scene, active_samples, args)
    if not bool(args.states_only):
        sim.render()
        for name in CAMERA_VIEWS:
            scene[f"{name}_camera"].update(0.0, force_recompute=True)
    initial = capture_initial_state(scene, args)
    properties = capture_constant_properties(scene)
    camera_info = capture_camera_info_for_args(scene, args)

    buffers: dict[str, list[np.ndarray]] = defaultdict(list)
    robot = scene["robot"]
    for step_index in range(int(args.episode_steps)):
        robot.set_joint_position_target(open_target)
        robot.set_joint_velocity_target(torch.zeros_like(open_target))
        for substep in range(int(args.decimation)):
            scene.write_data_to_sim()
            # Keep rendering separate from stepping.  With rendering_dt > physics_dt,
            # sim.step(render=True) may advance more than one physics tick.
            sim.step(render=False)
            if not bool(args.states_only) and substep == int(args.decimation) - 1:
                sim.render()
            scene.update(float(args.physics_dt))
        for path, value in capture_step(scene, raw_action, properties, args).items():
            buffers[path].append(value)
        if not simulation_app.is_running():
            raise RuntimeError(f"Simulation stopped during recorded step {step_index}.")
    frames = {path: np.stack(values, axis=0) for path, values in buffers.items()}
    print(
        f"[INFO] Recorded {args.episode_steps} synchronized state/"
        f"{'states-only' if args.states_only else ('RGB-D' if args.include_depth else 'RGB')} "
        "steps for the current batch."
    )
    return samples, initial, frames, properties, camera_info


def dataset_kwargs(array: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none" or array.size == 0:
        return {}
    kwargs: dict[str, Any] = {"compression": args.compression, "shuffle": array.dtype.kind not in "OUS"}
    if args.compression == "gzip":
        kwargs["compression_opts"] = int(args.gzip_level)
    if array.ndim == 4 and array.shape[-1] in (1, 3):
        kwargs["chunks"] = (1, array.shape[1], array.shape[2], array.shape[3])
    return kwargs


def write_dataset(group: h5py.Group, path: str, value: np.ndarray, args: argparse.Namespace) -> None:
    parent_path, _, name = path.rpartition("/")
    parent = group.require_group(parent_path) if parent_path else group
    array = np.asarray(value)
    parent.create_dataset(name, data=array, **dataset_kwargs(array, args))


def drop_metrics(
    slot: int,
    initial: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
) -> tuple[float, bool, bool]:
    initial_z = float(initial["rigid_object/object/root_pose"][slot, 2])
    z = frames["states/rigid_object/object/root_pose"][:, slot, 2]
    gripper = frames["states/articulation/robot/joint_position"][:, slot, 7:9]
    drop_distance = initial_z - float(np.min(z))
    table_contact_observed = bool(float(np.min(z)) <= LIFT_TABLE_SUPPORT_Z + 0.04)
    gripper_open = bool((gripper[-1] >= 0.035).all())
    return drop_distance, table_contact_observed, bool(table_contact_observed and gripper_open)


def write_episode(
    data_group: h5py.Group,
    episode_index: int,
    slot: int,
    sample: HeldPoseSample,
    initial: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
    properties: dict[str, np.ndarray],
    camera_info: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> int:
    final_name = f"demo_{episode_index}"
    temp_name = f"_writing_{final_name}"
    if final_name in data_group:
        raise KeyError(f"Output already contains {final_name}")
    if temp_name in data_group:
        del data_group[temp_name]
    episode = data_group.create_group(temp_name)
    count = int(args.episode_steps)

    for path, value in initial.items():
        write_dataset(episode.require_group("initial_state"), path, value[slot : slot + 1], args)
    for path, value in frames.items():
        write_dataset(episode, path, value[:, slot], args)
    for path, value in properties.items():
        if path.startswith("episode_physics_randomization/"):
            write_dataset(episode, path, value[slot : slot + 1], args)
        else:
            repeated = np.repeat(value[slot : slot + 1], count, axis=0)
            write_dataset(episode, path, repeated, args)
    for path, value in camera_info.items():
        write_dataset(episode, path, value[slot], args)

    drop_distance, table_contact_observed, drop_complete = drop_metrics(slot, initial, frames)
    episode.attrs["num_samples"] = count
    episode.attrs["success"] = drop_complete
    episode.attrs["drop_complete"] = drop_complete
    episode.attrs["drop_distance"] = np.float32(drop_distance)
    episode.attrs["table_contact_observed"] = table_contact_observed
    episode.attrs["source_demo"] = sample.source_demo
    episode.attrs["source_frame"] = int(sample.source_frame)
    episode.attrs["source_held_height"] = np.float32(sample.height)
    episode.attrs["release_action"] = "hold arm; command both panda fingers to 0.04 m"
    if "images" in episode:
        episode["images"].attrs["alignment"] = "post_step; image[t] is synchronized with states/*[t]"
        episode["initial_state/images"].attrs["alignment"] = "held pre-release state"
    if "depth" in episode:
        episode["depth"].attrs["alignment"] = "post_step; depth[t] is synchronized with states/*[t]"
        episode["depth"].attrs["type"] = "distance_to_image_plane"
        episode["depth"].attrs["units"] = "meters"
        episode["depth"].attrs["storage"] = str(args.depth_storage)
        episode["depth"].attrs["scale_to_meters"] = np.float32(0.001 if str(args.depth_storage) == "uint16_mm" else 1.0)
        episode["initial_state/depth"].attrs["alignment"] = "held pre-release state"
    if "segmentation" in episode:
        episode["segmentation"].attrs["alignment"] = "post_step; masks[t] are synchronized with states/*[t]"
        if "instance" in episode["segmentation"]:
            episode["segmentation/instance"].attrs["type"] = "instance_segmentation_fast; non-colorized uint32 ids"
            episode["initial_state/segmentation/instance"].attrs["alignment"] = "held pre-release state"
        if "instance_id" in episode["segmentation"]:
            episode["segmentation/instance_id"].attrs["type"] = "instance_id_segmentation_fast; non-colorized uint32 ids"
            episode["initial_state/segmentation/instance_id"].attrs["alignment"] = "held pre-release state"
    if "camera_info" in episode:
        episode["camera_info"].attrs["position_frame"] = "environment origin"
        episode["camera_info"].attrs["quaternion_format"] = "wxyz; Isaac Lab world camera convention (+X forward, +Z up)"
    data_group.move(temp_name, final_name)
    return count


def existing_episode_count(data_group: h5py.Group) -> int:
    indices = sorted(_numeric_demo_key(name)[0] for name in data_group if name.startswith("demo_"))
    if indices and indices != list(range(len(indices))):
        raise RuntimeError("Resume requires contiguous demo_0 ... demo_N episode names.")
    return len(indices)


def collection_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return simulation/data choices that must stay fixed across resume runs."""
    return {
        "pose_source_hdf5": os.path.abspath(args.pose_source_hdf5),
        "seed": int(args.seed),
        "episode_steps": int(args.episode_steps),
        "physics_dt": float(args.physics_dt),
        "decimation": int(args.decimation),
        "settle_steps": int(args.settle_steps),
        "min_held_height": float(args.min_held_height),
        "max_held_height": float(args.max_held_height),
        "max_held_speed": float(args.max_held_speed),
        "max_held_angular_speed": float(args.max_held_angular_speed),
        "max_closed_finger_position": float(args.max_closed_finger_position),
        "max_grasp_distance": float(args.max_grasp_distance),
        "arm_stiffness": float(args.arm_stiffness),
        "arm_damping": float(args.arm_damping),
        "camera_width": int(args.camera_width),
        "camera_height": int(args.camera_height),
        "camera_focal_length": float(args.camera_focal_length),
        "states_only": bool(args.states_only),
        "include_rgb": (not bool(args.states_only)) and (not bool(args.no_rgb)),
        "include_depth": (not bool(args.states_only)) and bool(args.include_depth),
        "include_instance_segmentation": (not bool(args.states_only)) and bool(args.include_instance_segmentation),
        "include_instance_id_segmentation": (not bool(args.states_only)) and bool(args.include_instance_id_segmentation),
        "depth_clipping_behavior": str(args.depth_clipping_behavior),
        "depth_storage": str(args.depth_storage),
        "replay_physics_context": bool(args.replay_physics_context),
        "disable_fabric": bool(args.disable_fabric),
        "num_envs": int(args.num_envs),
        "keep_failed": bool(args.keep_failed),
    }


def normalize_collection_config(config: dict[str, Any]) -> dict[str, Any]:
    """Backfill keys added after the original RGB-only drop dataset."""
    normalized = dict(config)
    normalized.setdefault("states_only", False)
    normalized.setdefault("include_rgb", True)
    normalized.setdefault("include_depth", False)
    normalized.setdefault("include_instance_segmentation", False)
    normalized.setdefault("include_instance_id_segmentation", False)
    normalized.setdefault("depth_clipping_behavior", "max")
    normalized.setdefault("depth_storage", "float32")
    return normalized


def open_output(args: argparse.Namespace, control_dt: float) -> tuple[h5py.File, h5py.Group, int]:
    path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if os.path.exists(path) and args.overwrite:
        os.remove(path)
    if os.path.exists(path) and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume or --overwrite: {path}")
    mode = "r+" if os.path.exists(path) else "w"
    file = h5py.File(path, mode)
    data = file.require_group("data")
    start = existing_episode_count(data)
    for name in list(data.keys()):
        if name.startswith("_writing_demo_"):
            del data[name]

    if mode == "w":
        file.attrs["schema_version"] = "franka_cube_drop_multiview_v1"
        file.attrs["task"] = "Franka Lift inverse scenario: held cube released by opening gripper"
        file.attrs["policy_required"] = False
        file.attrs["temporal_alignment"] = "actions, obs, states, dynamics, torques, and images are all post-step"
        file.attrs["pose_source_hdf5"] = os.path.abspath(args.pose_source_hdf5)
        file.attrs["seed"] = int(args.seed)
        file.attrs["control_dt"] = np.float32(control_dt)
        file.attrs["camera_views"] = json.dumps({} if bool(args.states_only) else CAMERA_VIEWS)
        file.attrs["camera_data_types"] = json.dumps(camera_data_types(args))
        if bool(args.states_only):
            file.attrs["camera_resolution"] = np.asarray([0, 0], dtype=np.int32)
        else:
            file.attrs["camera_resolution"] = np.asarray([args.camera_height, args.camera_width], dtype=np.int32)
        file.attrs["support_plane_z"] = np.float32(LIFT_TABLE_SUPPORT_Z)
        file.attrs["support_plane_frame"] = "environment origin"
        file.attrs["collection_config"] = json.dumps(collection_config(args), sort_keys=True)
        data.attrs["total"] = np.int64(0)
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "Isaac-Lift-Cube-Franka-v0",
                "type": 2,
                "sim_args": {
                    "dt": float(args.physics_dt),
                    "decimation": int(args.decimation),
                    "render_interval": int(args.decimation),
                    "num_envs": int(args.num_envs),
                },
            }
        )
    else:
        schema = str(file.attrs.get("schema_version", ""))
        if schema != "franka_cube_drop_multiview_v1":
            file.close()
            raise RuntimeError(f"Cannot resume incompatible schema: {schema!r}")
        if abs(float(file.attrs.get("control_dt", -1.0)) - control_dt) > 1.0e-8:
            file.close()
            raise RuntimeError("Cannot resume with a different control timestep.")
        stored_config = normalize_collection_config(json.loads(str(file.attrs.get("collection_config", "{}"))))
        requested_config = normalize_collection_config(collection_config(args))
        if stored_config != requested_config:
            changed_keys = {
                key
                for key in set(stored_config) | set(requested_config)
                if stored_config.get(key) != requested_config.get(key)
            }
            can_widen_grasp_check = (
                changed_keys == {"max_grasp_distance"}
                and float(stored_config.get("max_grasp_distance", 0.0))
                <= float(requested_config.get("max_grasp_distance", 0.0))
            )
            if can_widen_grasp_check:
                print(
                    "[WARN] Widening resume max_grasp_distance "
                    f"{stored_config.get('max_grasp_distance')} -> {requested_config.get('max_grasp_distance')}"
                )
                file.attrs.modify("collection_config", json.dumps(requested_config, sort_keys=True))
                file.flush()
            else:
                file.close()
                raise RuntimeError(
                    "Cannot resume with a different collection configuration. "
                    f"Stored={stored_config}; requested={requested_config}"
                )
        recomputed_total = sum(
            int(data[name].attrs["num_samples"]) for name in data if name.startswith("demo_")
        )
        if int(data.attrs.get("total", -1)) != recomputed_total:
            print(
                f"[WARN] Repairing stale /data total during resume: "
                f"{int(data.attrs.get('total', -1))} -> {recomputed_total}"
            )
            data.attrs.modify("total", np.int64(recomputed_total))
            file.flush()
    return file, data, start


def estimate_storage(args: argparse.Namespace, remaining_episodes: int) -> None:
    if bool(args.states_only):
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.output_file)) or ".").free
        print(
            "[INFO] Uncompressed recorded camera payload: 0.00 GiB (states-only); "
            f"filesystem free space: {free / 2**30:.2f} GiB. HDF5 compression is {args.compression}."
        )
        return

    raw_images = 0
    if not bool(args.no_rgb):
        raw_images = (
            remaining_episodes
            * int(args.episode_steps)
            * len(CAMERA_VIEWS)
            * int(args.camera_height)
            * int(args.camera_width)
            * 3
        )
    raw_depth = 0
    if bool(args.include_depth):
        depth_bytes = 2 if str(args.depth_storage) == "uint16_mm" else 4
        raw_depth = (
            remaining_episodes
            * int(args.episode_steps)
            * len(CAMERA_VIEWS)
            * int(args.camera_height)
            * int(args.camera_width)
            * depth_bytes
        )
    raw_instance = 0
    if bool(args.include_instance_segmentation):
        raw_instance = (
            remaining_episodes
            * int(args.episode_steps)
            * len(CAMERA_VIEWS)
            * int(args.camera_height)
            * int(args.camera_width)
            * 4
        )
    raw_instance_id = 0
    if bool(args.include_instance_id_segmentation):
        raw_instance_id = (
            remaining_episodes
            * int(args.episode_steps)
            * len(CAMERA_VIEWS)
            * int(args.camera_height)
            * int(args.camera_width)
            * 4
        )
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.output_file)) or ".").free
    payload_parts = []
    if raw_images:
        payload_parts.append(f"RGB {raw_images / 2**30:.2f} GiB")
    if raw_depth:
        payload_parts.append(f"depth {raw_depth / 2**30:.2f} GiB")
    if raw_instance:
        payload_parts.append(f"instance masks {raw_instance / 2**30:.2f} GiB")
    if raw_instance_id:
        payload_parts.append(f"instance-id masks {raw_instance_id / 2**30:.2f} GiB")
    if not payload_parts:
        payload_parts.append("camera payload 0.00 GiB")
    print(
        f"[INFO] Uncompressed recorded payload: {', '.join(payload_parts)}; "
        f"filesystem free space: {free / 2**30:.2f} GiB. HDF5 compression is {args.compression}."
    )


def validate_output(path: str, expected_episodes: int) -> dict[str, float]:
    timed_shapes = {
        "actions": (8,),
        "processed_actions": (9,),
        "obs/joint_pos": (9,),
        "obs/joint_vel": (9,),
        "states/articulation/robot/joint_position": (9,),
        "states/articulation/robot/joint_velocity": (9,),
        "states/rigid_object/object/root_pose": (7,),
        "states/rigid_object/object/root_velocity": (6,),
        "robot_torques/applied_torque": (9,),
        "robot_torques/computed_torque": (9,),
        "object_dynamics/mass": (1,),
        "object_dynamics/inertia": (9,),
        "object_dynamics/material_properties": (1, 3),
    }
    initial_paths = (
        "initial_state/articulation/robot/root_pose",
        "initial_state/rigid_object/object/root_pose",
    )
    completed = 0
    min_drop = np.inf
    with h5py.File(path, "r", locking=False) as file:
        config = normalize_collection_config(json.loads(str(file.attrs.get("collection_config", "{}"))))
        states_only = bool(config.get("states_only", False))
        has_rgb = bool(config.get("include_rgb", True))
        has_depth = bool(config.get("include_depth", False))
        has_instance = bool(config.get("include_instance_segmentation", False))
        has_instance_id = bool(config.get("include_instance_id_segmentation", False))
        has_cameras = bool(has_rgb or has_depth or has_instance or has_instance_id) and not states_only
        depth_storage = str(config.get("depth_storage", "float32"))
        data = file["data"]
        names = sorted((name for name in data if name.startswith("demo_")), key=_numeric_demo_key)
        if len(names) != expected_episodes:
            raise AssertionError(f"Expected {expected_episodes} episodes, found {len(names)}")
        if names != [f"demo_{index}" for index in range(expected_episodes)]:
            raise AssertionError("Episode names are not contiguous demo_0 ... demo_N")
        if any(name.startswith("_writing_demo_") for name in data):
            raise AssertionError("Output contains an incomplete temporary episode")
        resolution = tuple(int(value) for value in np.asarray(file.attrs.get("camera_resolution", [0, 0])).reshape(-1))
        if len(resolution) != 2:
            raise AssertionError("camera_resolution must contain [height, width]")
        total = 0
        for episode_i, name in enumerate(names):
            episode = data[name]
            count = int(episode.attrs["num_samples"])
            total += count
            for path_name, trailing_shape in timed_shapes.items():
                if path_name not in episode:
                    raise AssertionError(f"{name} is missing {path_name}")
                dataset = episode[path_name]
                if dataset.shape != (count, *trailing_shape):
                    raise AssertionError(
                        f"{name}/{path_name} has shape {dataset.shape}, expected {(count, *trailing_shape)}"
                    )
                if not np.isfinite(np.asarray(dataset)).all():
                    raise AssertionError(f"{name}/{path_name} contains non-finite values")
            for path_name in initial_paths:
                if path_name not in episode or episode[path_name].shape[0] != 1:
                    raise AssertionError(f"{name} has invalid or missing {path_name}")

            state = np.asarray(episode["states/rigid_object/object/root_pose"])
            initial_z = float(episode["initial_state/rigid_object/object/root_pose"][0, 2])
            drop = initial_z - float(np.min(state[:, 2]))
            min_drop = min(min_drop, drop)
            completed += int(bool(episode.attrs.get("drop_complete", False)))
            first_views: list[np.ndarray] = []
            if not has_cameras:
                continue
            for camera in CAMERA_VIEWS:
                image = None
                if has_rgb:
                    image = episode[f"images/{camera}"]
                    if image.dtype != np.uint8 or image.shape != (count, resolution[0], resolution[1], 3):
                        raise AssertionError(f"{name}/images/{camera} is not uint8 RGB")
                    initial_image = episode[f"initial_state/images/{camera}"]
                    if initial_image.dtype != np.uint8 or initial_image.shape != (1, resolution[0], resolution[1], 3):
                        raise AssertionError(f"{name}/initial_state/images/{camera} is not uint8 RGB")
                if has_depth:
                    depth = episode[f"depth/{camera}"]
                    expected_dtype = np.uint16 if depth_storage == "uint16_mm" else np.float32
                    if depth.dtype != expected_dtype or depth.shape != (count, resolution[0], resolution[1], 1):
                        raise AssertionError(f"{name}/depth/{camera} is not {expected_dtype} HxWx1 depth")
                    depth_values = np.asarray(depth)
                    if depth_storage == "float32" and not np.isfinite(depth_values).all():
                        raise AssertionError(f"{name}/depth/{camera} contains non-finite values")
                    if float(np.nanmax(depth_values)) <= 0.0:
                        raise AssertionError(f"{name}/depth/{camera} contains no positive depth")
                    initial_depth = episode[f"initial_state/depth/{camera}"]
                    if initial_depth.dtype != expected_dtype or initial_depth.shape != (1, resolution[0], resolution[1], 1):
                        raise AssertionError(
                            f"{name}/initial_state/depth/{camera} is not {expected_dtype} HxWx1 depth"
                        )
                if has_instance:
                    instance = episode[f"segmentation/instance/{camera}"]
                    if instance.dtype != np.uint32 or instance.shape != (count, resolution[0], resolution[1], 1):
                        raise AssertionError(f"{name}/segmentation/instance/{camera} is not uint32 HxWx1")
                    initial_instance = episode[f"initial_state/segmentation/instance/{camera}"]
                    if initial_instance.dtype != np.uint32 or initial_instance.shape != (
                        1,
                        resolution[0],
                        resolution[1],
                        1,
                    ):
                        raise AssertionError(f"{name}/initial_state/segmentation/instance/{camera} is not uint32 HxWx1")
                if has_instance_id:
                    instance_id = episode[f"segmentation/instance_id/{camera}"]
                    if instance_id.dtype != np.uint32 or instance_id.shape != (count, resolution[0], resolution[1], 1):
                        raise AssertionError(f"{name}/segmentation/instance_id/{camera} is not uint32 HxWx1")
                    initial_instance_id = episode[f"initial_state/segmentation/instance_id/{camera}"]
                    if initial_instance_id.dtype != np.uint32 or initial_instance_id.shape != (
                        1,
                        resolution[0],
                        resolution[1],
                        1,
                    ):
                        raise AssertionError(
                            f"{name}/initial_state/segmentation/instance_id/{camera} is not uint32 HxWx1"
                        )
                calibration = episode[f"camera_info/{camera}"]
                calibration_shapes = {
                    "intrinsic_matrix": (3, 3),
                    "position": (3,),
                    "quaternion_world_convention": (4,),
                }
                for calibration_name, calibration_shape in calibration_shapes.items():
                    values = np.asarray(calibration[calibration_name])
                    if values.shape != calibration_shape:
                        raise AssertionError(
                            f"{name}/camera_info/{camera}/{calibration_name} has shape {values.shape}, "
                            f"expected {calibration_shape}"
                        )
                    if not np.isfinite(values).all():
                        raise AssertionError(f"{name}/camera_info/{camera}/{calibration_name} is non-finite")
                if has_rgb and image is not None and episode_i < min(16, expected_episodes):
                    first = np.asarray(image[0])
                    last = np.asarray(image[-1])
                    if float(first.std()) < 1.0:
                        raise AssertionError(f"{name}/images/{camera} appears blank")
                    if float(np.abs(last.astype(np.int16) - first.astype(np.int16)).mean()) < 0.05:
                        raise AssertionError(f"{name}/images/{camera} does not change during the drop")
                    first_views.append(first)
            if first_views:
                view_differences = [
                    float(np.abs(first_views[0].astype(np.int16) - other.astype(np.int16)).mean())
                    for other in first_views[1:]
                ]
                if min(view_differences) < 0.5:
                    raise AssertionError(f"{name} contains duplicate or nearly identical camera views")
        if int(data.attrs.get("total", -1)) != total:
            raise AssertionError("/data total attribute does not equal summed episode lengths")
        if not bool(config.get("keep_failed", False)) and completed != expected_episodes:
            raise AssertionError(f"Only {completed}/{expected_episodes} episodes completed their drops")
    return {
        "episodes": float(expected_episodes),
        "drop_complete_fraction": float(completed / max(expected_episodes, 1)),
        "minimum_drop_distance": float(min_drop),
    }


def main() -> str:
    if int(args_cli.num_episodes) <= 0 or int(args_cli.num_envs) <= 0 or int(args_cli.episode_steps) <= 1:
        raise ValueError("num-episodes and num-envs must be positive; episode-steps must exceed one.")
    if int(args_cli.decimation) <= 0 or float(args_cli.physics_dt) <= 0.0:
        raise ValueError("decimation and physics-dt must be positive.")
    if bool(args_cli.no_rgb) and not bool(args_cli.states_only) and not (
        bool(args_cli.include_depth)
        or bool(args_cli.include_instance_segmentation)
        or bool(args_cli.include_instance_id_segmentation)
    ):
        raise ValueError("--no-rgb requires at least one non-RGB camera output, e.g. --include-depth.")
    args_cli.output_file = os.path.abspath(args_cli.output_file)
    args_cli.pose_source_hdf5 = os.path.abspath(args_cli.pose_source_hdf5)
    if os.path.realpath(args_cli.output_file) == os.path.realpath(args_cli.pose_source_hdf5):
        raise ValueError("--output-file must not be the same file as --pose-source-hdf5.")
    control_dt = float(args_cli.physics_dt) * int(args_cli.decimation)

    pose_sequence = load_pose_sequence(args_cli.pose_source_hdf5, args_cli)
    output, data_group, start_episode = open_output(args_cli, control_dt)
    if start_episode > int(args_cli.num_episodes):
        output.close()
        raise RuntimeError(
            f"Output already has {start_episode} episodes, more than requested {args_cli.num_episodes}."
        )
    if start_episode == int(args_cli.num_episodes):
        output.close()
        if not args_cli.skip_validation:
            summary = validate_output(args_cli.output_file, int(args_cli.num_episodes))
            print(
                "[INFO] Validation passed: "
                f"drop_complete={summary['drop_complete_fraction']:.1%}, "
                f"minimum_drop={summary['minimum_drop_distance']:.3f} m"
            )
        print(f"[INFO] Dataset is already complete: {args_cli.output_file}")
        return args_cli.output_file
    estimate_storage(args_cli, int(args_cli.num_episodes) - start_episode)

    try:
        sim, scene = build_simulation(args_cli)
        total_samples = int(data_group.attrs.get("total", 0))
        for batch_start in range(start_episode, int(args_cli.num_episodes), int(args_cli.num_envs)):
            batch_end = min(batch_start + int(args_cli.num_envs), int(args_cli.num_episodes))
            active = pose_sequence[batch_start:batch_end]
            samples, initial, frames, properties, camera_info = collect_batch(sim, scene, active, args_cli)
            failed_slots = [slot for slot in range(len(active)) if not drop_metrics(slot, initial, frames)[2]]
            if failed_slots:
                details = ", ".join(samples[slot].source_demo for slot in failed_slots)
                message = f"Batch contains incomplete drops in slots {failed_slots} (sources: {details})."
                if not args_cli.keep_failed:
                    raise RuntimeError(
                        message + " No episode from this batch was committed; use --keep-failed to retain them."
                    )
                print(f"[WARN] {message}")
            for slot, episode_index in enumerate(range(batch_start, batch_end)):
                total_samples += write_episode(
                    data_group,
                    episode_index,
                    slot,
                    samples[slot],
                    initial,
                    frames,
                    properties,
                    camera_info,
                    args_cli,
                )
            data_group.attrs.modify("total", np.int64(total_samples))
            output.flush()
            print(
                f"[INFO] Wrote episodes {batch_start}..{batch_end - 1} "
                f"({batch_end}/{args_cli.num_episodes}); samples={total_samples}"
            )
    finally:
        output.close()

    if not args_cli.skip_validation:
        summary = validate_output(args_cli.output_file, int(args_cli.num_episodes))
        print(
            "[INFO] Validation passed: "
            f"episodes={int(summary['episodes'])}, "
            f"drop_complete={summary['drop_complete_fraction']:.1%}, "
            f"minimum_drop={summary['minimum_drop_distance']:.3f} m"
        )
    return args_cli.output_file


if __name__ == "__main__":
    try:
        result = main()
        print(f"[INFO] Dataset ready: {result}")
    except BaseException:
        # Immediate Kit shutdown can otherwise hide the Python exception.
        traceback.print_exc()
        raise
    finally:
        # Camera annotators do not write through Replicator, so there is no
        # asynchronous Replicator output to drain at shutdown.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
