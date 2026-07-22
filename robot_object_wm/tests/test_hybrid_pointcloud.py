from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from robot_object_wm.data.rollout_dataset import RobotObjectWMRolloutDataset
from robot_object_wm.data.hdf5_schema import make_robot_object_state_layout
from robot_object_wm.models.hybrid import estimate_weighted_row_rigid_transform, pose_from_cube_points, quat_angle_error


def _write_state_hdf5(path, *, steps: int = 6) -> None:
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("joint_pos", data=np.zeros((steps, 9), dtype=np.float32))
        obs.create_dataset("joint_vel", data=np.zeros((steps, 9), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((steps, 8), dtype=np.float32))
        torques = demo.create_group("robot_torques")
        torques.create_dataset("applied_torque", data=np.zeros((steps, 9), dtype=np.float32))

        states = demo.create_group("states")
        obj = states.create_group("rigid_object/object")
        pose = np.zeros((steps, 7), dtype=np.float32)
        pose[:, 0] = np.linspace(0.0, 0.05, steps)
        pose[:, 3] = 1.0
        obj.create_dataset("root_pose", data=pose)
        obj.create_dataset("root_velocity", data=np.zeros((steps, 6), dtype=np.float32))

        init = demo.create_group("initial_state/articulation/robot")
        root_pose = np.zeros((1, 7), dtype=np.float32)
        root_pose[:, 3] = 1.0
        init.create_dataset("root_pose", data=root_pose)

        dyn = demo.create_group("object_dynamics")
        dyn.create_dataset("mass", data=np.ones((steps, 1), dtype=np.float32))
        dyn.create_dataset("inertia", data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (steps, 1)))
        dyn.create_dataset("material_properties", data=np.zeros((steps, 3), dtype=np.float32))


def _write_pointcloud_hdf5(path, *, steps: int = 6, points: int = 8) -> None:
    base = np.stack(
        [
            np.linspace(-0.02, 0.02, points, dtype=np.float32),
            np.zeros(points, dtype=np.float32),
            np.zeros(points, dtype=np.float32),
        ],
        axis=-1,
    )
    object_points = np.zeros((1, steps, 2, points, 3), dtype=np.float32)
    for t in range(steps):
        object_points[0, t, 0] = base + np.asarray([0.01 * t, 0.0, 0.0], dtype=np.float32)
        object_points[0, t, 1] = base + np.asarray([0.0, 0.01 * t, 0.1], dtype=np.float32)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.create_dataset("object_points", data=object_points)
        data.create_dataset("object_point_counts", data=np.full((1, steps, 2), points, dtype=np.int32))
        data.create_dataset("episode_names", data=np.asarray(["demo_0"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        data.create_dataset("object_names", data=np.asarray(["cube", "gripper"], dtype=object), dtype=h5py.string_dtype("utf-8"))
        data.create_dataset("vertex_properties", data=np.zeros((1, 2, 3), dtype=np.float32))
        data.create_dataset("loss_object_mask", data=np.asarray([True, False], dtype=bool))
        file.attrs["rigidformer_pointcloud_config"] = json.dumps({"control_dt": 0.02})


def test_rollout_dataset_pairs_pointcloud_frames(tmp_path):
    state_path = tmp_path / "state.hdf5"
    pc_path = tmp_path / "pc.hdf5"
    _write_state_hdf5(state_path)
    _write_pointcloud_hdf5(pc_path)
    layout = make_robot_object_state_layout(
        robot_dof=9,
        action_dim=8,
        torque_dim=9,
        state_prediction_mode="full",
    )

    dataset = RobotObjectWMRolloutDataset(
        [(str(state_path), "demo_0")],
        history_len=2,
        rollout_horizon=2,
        dt=0.02,
        layout=layout,
        pointcloud_file=str(pc_path),
        pointcloud_max_points=4,
    )

    item = dataset[0]
    assert item["history_states"].shape == (2, layout.state_dim)
    assert item["future_states"].shape == (2, layout.state_dim)
    assert item["pc_object_pos_prev"].shape == (2, 4, 3)
    assert item["pc_object_pos"].shape == (2, 4, 3)
    assert item["pc_object_pos_next"].shape == (2, 4, 3)
    assert item["pc_object_pos_rollout"].shape == (4, 2, 4, 3)
    assert item["pc_object_lens"].item() == 2
    assert item["pc_loss_object_mask"].tolist() == [True, False]
    assert item["pc_gripper_part_ids"].shape == (4,)
    assert item["pc_first_robot_q_abs"].shape == (9,)
    assert item["pc_first_robot_q_obs"].shape == (9,)
    assert torch.allclose(item["pc_first_robot_q_abs"], item["pc_first_robot_q_obs"])


def test_rollout_dataset_can_supervise_cube_and_gripper_pointclouds(tmp_path):
    state_path = tmp_path / "state.hdf5"
    pc_path = tmp_path / "pc.hdf5"
    _write_state_hdf5(state_path)
    _write_pointcloud_hdf5(pc_path)
    layout = make_robot_object_state_layout(
        robot_dof=9,
        action_dim=8,
        torque_dim=9,
        state_prediction_mode="full",
    )

    dataset = RobotObjectWMRolloutDataset(
        [(str(state_path), "demo_0")],
        history_len=2,
        rollout_horizon=2,
        dt=0.02,
        layout=layout,
        pointcloud_file=str(pc_path),
        pointcloud_max_points=4,
        pointcloud_loss_object_mode="cube_gripper",
    )

    item = dataset[0]
    assert item["pc_loss_object_mask"].tolist() == [True, True]


def test_pose_from_cube_points_recovers_translation_and_rotation():
    local = torch.tensor(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    angle = torch.tensor(np.pi / 2, dtype=torch.float32)
    zero = torch.zeros((), dtype=torch.float32)
    one = torch.ones((), dtype=torch.float32)
    rot = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle), zero]),
            torch.stack([torch.sin(angle), torch.cos(angle), zero]),
            torch.stack([zero, zero, one]),
        ]
    ).unsqueeze(0)
    target_pos = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    target = torch.einsum("bnd,bdc->bnc", local, rot.transpose(-1, -2)) + target_pos[:, None]
    pos, quat = pose_from_cube_points(
        first_frame_points=local,
        target_points=target,
        first_object_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        first_object_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        env_origin=torch.zeros((1, 3), dtype=torch.float32),
        subtract_env_origin=True,
        point_lens=torch.tensor([4]),
    )
    target_quat = torch.tensor([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], dtype=torch.float32)
    assert torch.allclose(pos, target_pos, atol=1.0e-5)
    assert torch.all(quat_angle_error(quat, target_quat) < 1.0e-4)


def test_weighted_rigid_transform_masks_nonfinite_points():
    reference = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([[0.2, -0.1, 0.3], [0.0, 0.0, 0.0]], dtype=torch.float32)
    target = reference + translation[:, None]
    target[0, 0] = float("nan")
    target[1, :2] = float("nan")
    weights = torch.ones((2, 4), dtype=torch.float32)

    rotation, recovered_translation, valid = estimate_weighted_row_rigid_transform(
        reference,
        target,
        weights=weights,
        return_valid_mask=True,
    )

    assert valid.tolist() == [True, False]
    assert torch.isfinite(rotation).all()
    assert torch.isfinite(recovered_translation).all()
    assert torch.allclose(recovered_translation[0], translation[0], atol=1.0e-5)
