from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from rigidformer.franka_pointcloud import _split_indices
from rigidformer.pose_metrics import pose_errors_from_point_trajectories, summarize_pose_errors
from rigidformer.rigidformer import Rigidformer


def franka_pointcloud_episode_indices(
    hdf5_path: str | Path,
    *,
    split: str = "val",
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    with h5py.File(hdf5_path, "r", locking=False) as file:
        count = int(file["data/object_points"].shape[0])

    return _split_indices(
        count,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )


def _decode_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _point_indices(
    stored_points: int,
    max_points: int,
    *,
    random_points: bool,
    seed: int,
    episode_index: int,
) -> np.ndarray:
    if max_points <= 0:
        max_points = stored_points
    if max_points > stored_points:
        raise ValueError(f"max_points={max_points} exceeds stored points={stored_points}")
    if max_points == stored_points:
        return np.arange(stored_points, dtype=np.int64)
    if not random_points:
        return np.arange(max_points, dtype=np.int64)

    rng = np.random.default_rng(seed + int(episode_index) * 1009)
    return np.sort(rng.choice(stored_points, size=max_points, replace=False)).astype(np.int64)


def _dataset_2d_or_3d(data: h5py.Dataset, episode_index: int) -> np.ndarray:
    if data.ndim == 2:
        return np.asarray(data, dtype=np.float32)
    return np.asarray(data[episode_index], dtype=np.float32)


def _load_loss_object_mask(data: h5py.Group, episode_index: int, object_lens: int) -> np.ndarray:
    if "loss_object_mask" not in data:
        mask = np.ones((object_lens,), dtype=bool)
        if object_lens > 1:
            mask[1:] = False
        return mask

    dataset = data["loss_object_mask"]
    mask = np.asarray(dataset if dataset.ndim == 1 else dataset[episode_index], dtype=bool)
    return mask[:object_lens]


def load_franka_pointcloud_trajectory(
    hdf5_path: str | Path,
    *,
    episode_index: int,
    frame_indices: np.ndarray,
    max_points: int,
    random_points: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r", locking=False) as file:
        data = file["data"]
        points = data["object_points"]
        if points.ndim != 5 or points.shape[-1] != 3:
            raise ValueError("/data/object_points must have shape (episodes, steps, objects, points, 3)")

        num_episodes, num_steps, object_lens, stored_points, _ = points.shape
        if episode_index < 0 or episode_index >= num_episodes:
            raise IndexError(f"episode_index={episode_index} out of range for {num_episodes} episodes")
        if frame_indices[0] < 0 or frame_indices[-1] >= num_steps:
            raise IndexError(
                f"Frame range [{int(frame_indices[0])}, {int(frame_indices[-1])}] exceeds dataset steps={num_steps}"
            )

        selected_points = _point_indices(
            stored_points,
            max_points,
            random_points=random_points,
            seed=seed,
            episode_index=episode_index,
        )
        positions = np.asarray(points[episode_index, frame_indices], dtype=np.float32)
        positions = positions[:, :, selected_points, :]
        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)

        if "vertex_properties" in data:
            vertex_properties = _dataset_2d_or_3d(data["vertex_properties"], episode_index)
        else:
            vertex_properties = np.zeros((object_lens, 3), dtype=np.float32)
            if object_lens > 0:
                vertex_properties[0] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
            if object_lens > 1:
                vertex_properties[1] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        if "object_names" in data:
            object_names = [_decode_string(name) for name in np.asarray(data["object_names"])]
        else:
            object_names = [f"object_{index}" for index in range(object_lens)]

        config = json.loads(str(file.attrs.get("rigidformer_pointcloud_config", "{}")))
        control_dt = float(config.get("control_dt", 0.02))
        loss_object_mask = _load_loss_object_mask(data, episode_index, object_lens)

    return {
        "positions": torch.from_numpy(positions),
        "first_frame_pos": torch.from_numpy(positions[0]),
        "vertex_properties": torch.from_numpy(vertex_properties[:object_lens]),
        "object_lens": int(object_lens),
        "object_point_lens": torch.full((object_lens,), int(max_points), dtype=torch.long),
        "loss_object_mask": torch.from_numpy(loss_object_mask),
        "object_names": object_names[:object_lens],
        "frame_indices": frame_indices.astype(np.int64),
        "episode_index": int(episode_index),
        "control_dt": control_dt,
    }


def _valid_points(points: np.ndarray, point_lens: np.ndarray, object_indices: np.ndarray) -> np.ndarray:
    chunks = [points[object_index, : point_lens[object_index]] for object_index in object_indices]
    return np.concatenate(chunks, axis=0)


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(-1, 3)
    mask = np.isfinite(points).all(axis=-1)
    return points[mask]


def _set_equal_3d_limits(ax, points: np.ndarray, margin: float = 0.1) -> None:
    points = _finite_points(points)
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=np.float32)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * (0.5 + margin), 1e-3)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_point_cloud(
    ax,
    points: np.ndarray,
    point_lens: np.ndarray,
    object_indices: np.ndarray,
    object_names: list[str],
    colors: list[Any],
    title: str,
    axis_points: np.ndarray,
) -> None:
    for color_index, object_index in enumerate(object_indices):
        object_points = points[object_index, : point_lens[object_index]]
        label = object_names[object_index] if object_index < len(object_names) else f"object_{object_index}"
        ax.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            s=7,
            alpha=0.85,
            color=colors[color_index % len(colors)],
            edgecolors="none",
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d_limits(ax, axis_points)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper right", fontsize=7)


def _render_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    point_lens: np.ndarray,
    object_indices: np.ndarray,
    object_names: list[str],
    colors: list[Any],
    axis_points: np.ndarray,
    frame_index: int,
    rmse: float,
) -> np.ndarray:
    fig = plt.figure(figsize=(12, 5.142857), dpi=140)
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_point_cloud(
        ax_gt,
        gt,
        point_lens,
        object_indices,
        object_names,
        colors,
        f"Ground Truth | frame {frame_index}",
        axis_points,
    )
    _draw_point_cloud(
        ax_pred,
        pred,
        point_lens,
        object_indices,
        object_names,
        colors,
        f"RigidFormer Rollout | RMSE {rmse:.4f}",
        axis_points,
    )

    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    height, width = image.shape[:2]
    if height % 2:
        image = image[:-1]
    if width % 2:
        image = image[:, :-1]

    return image


def _save_final_overlay(
    output_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    point_lens: np.ndarray,
    object_indices: np.ndarray,
    object_names: list[str],
    axis_points: np.ndarray,
    frame_index: int,
) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for object_index in object_indices:
        name = object_names[object_index] if object_index < len(object_names) else f"object_{object_index}"
        gt_points = gt[object_index, : point_lens[object_index]]
        pred_points = pred[object_index, : point_lens[object_index]]
        ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            gt_points[:, 2],
            s=8,
            alpha=0.55,
            color="tab:green",
            edgecolors="none",
            label=f"{name} GT",
        )
        ax.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            pred_points[:, 2],
            s=8,
            alpha=0.55,
            color="tab:red",
            edgecolors="none",
            label=f"{name} pred",
        )

    ax.set_title(f"Final Frame Overlay | frame {frame_index}\nGT green, prediction red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d_limits(ax, axis_points)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_rmse_plot(output_path: Path, frame_indices: np.ndarray, rmse: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(frame_indices, rmse, marker="o", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Point RMSE")
    ax.set_title("Supervised Object Rollout Error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_pose_error_plot(
    output_path: Path,
    frame_indices: np.ndarray,
    position_error: np.ndarray,
    orientation_error_deg: np.ndarray,
    object_names: list[str],
    object_indices: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=160, sharex=True)

    for local_index, object_index in enumerate(object_indices):
        name = object_names[object_index] if object_index < len(object_names) else f"object_{object_index}"
        axes[0].plot(frame_indices, position_error[:, local_index], linewidth=1.2, label=name)
        axes[1].plot(frame_indices, orientation_error_deg[:, local_index], linewidth=1.2, label=name)

    axes[0].set_ylabel("Position error")
    axes[0].set_title("Supervised Object Pose Error")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


@torch.no_grad()
def rollout_franka_pointcloud_model(
    model: Rigidformer,
    positions: torch.Tensor,
    first_frame_pos: torch.Tensor,
    vertex_properties: torch.Tensor,
    object_lens: int,
    object_point_lens: torch.Tensor,
    loss_object_mask: torch.Tensor,
    delta_time: float,
    device: torch.device,
    use_amp: bool,
    *,
    teacher_force_context: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    object_lens_tensor = torch.tensor([object_lens], dtype=torch.long, device=device)
    point_lens = object_point_lens.unsqueeze(0).to(device)
    vertex_properties = vertex_properties.unsqueeze(0).to(device)
    first_frame_pos = first_frame_pos.unsqueeze(0).to(device)
    delta_times = torch.tensor([delta_time], dtype=torch.float32, device=device)
    context_mask = ~loss_object_mask[:object_lens].bool()
    context_mask_positions = context_mask.to(positions.device)
    context_mask_device = context_mask.to(device)
    has_context = bool(context_mask.any().item())

    prev = positions[0].unsqueeze(0).to(device)
    current = positions[1].unsqueeze(0).to(device)
    predicted = [prev.cpu(), current.cpu()]
    anchor_indices = None

    iterator = tqdm(
        range(positions.shape[0] - 2),
        desc="rollout",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for offset in iterator:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred, intermediates = model(
                delta_times=delta_times,
                vertex_properties=vertex_properties,
                object_pos=current,
                object_pos_prev=prev,
                object_pos_next=None,
                object_first_frame_pos=first_frame_pos,
                anchor_indices=anchor_indices,
                object_lens=object_lens_tensor,
                object_point_lens=point_lens,
                return_intermediates=True,
            )

        if anchor_indices is None:
            anchor_indices = intermediates.anchor_indices

        next_pos = pred.object_pos_next.detach()
        if teacher_force_context and has_context:
            next_pos = next_pos.clone()
            context_pos = positions[offset + 2, context_mask_positions].unsqueeze(0).to(device)
            next_pos[:, context_mask_device] = context_pos

        predicted.append(next_pos.cpu())
        prev, current = current, next_pos

    return torch.cat(predicted, dim=0)


@torch.no_grad()
def render_franka_pointcloud_rollout(
    model: Rigidformer,
    hdf5_path: str | Path,
    output_dir: str | Path,
    *,
    episode_index: int,
    start_frame: int = 0,
    steps: int = 48,
    stride: int = 1,
    max_points: int = 512,
    random_points: bool = False,
    seed: int = 0,
    video_fps: int = 12,
    teacher_force_context: bool = True,
    device: torch.device,
    use_amp: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = np.arange(start_frame, start_frame + (steps + 2) * stride, stride)
    trajectory = load_franka_pointcloud_trajectory(
        hdf5_path,
        episode_index=episode_index,
        frame_indices=frame_indices,
        max_points=max_points,
        random_points=random_points,
        seed=seed,
    )

    pred_positions = rollout_franka_pointcloud_model(
        model,
        trajectory["positions"],
        trajectory["first_frame_pos"],
        trajectory["vertex_properties"],
        trajectory["object_lens"],
        trajectory["object_point_lens"],
        trajectory["loss_object_mask"],
        stride * trajectory["control_dt"],
        device,
        use_amp,
        teacher_force_context=teacher_force_context,
        show_progress=show_progress,
    ).numpy()

    gt_positions = trajectory["positions"].numpy()
    point_lens = trajectory["object_point_lens"].numpy()
    object_lens = int(trajectory["object_lens"])
    loss_object_mask = trajectory["loss_object_mask"].numpy().astype(bool)[:object_lens]
    supervised_indices = np.flatnonzero(loss_object_mask)
    if supervised_indices.size == 0:
        supervised_indices = np.arange(object_lens, dtype=np.int64)
    all_indices = np.arange(object_lens, dtype=np.int64)
    object_names = trajectory["object_names"]

    supervised_rmse = []
    all_rmse = []
    for frame_offset in range(gt_positions.shape[0]):
        gt_supervised = _valid_points(gt_positions[frame_offset], point_lens, supervised_indices)
        pred_supervised = _valid_points(pred_positions[frame_offset], point_lens, supervised_indices)
        supervised_rmse.append(float(np.sqrt(np.mean((gt_supervised - pred_supervised) ** 2))))

        gt_all = _valid_points(gt_positions[frame_offset], point_lens, all_indices)
        pred_all = _valid_points(pred_positions[frame_offset], point_lens, all_indices)
        all_rmse.append(float(np.sqrt(np.mean((gt_all - pred_all) ** 2))))

    supervised_rmse = np.asarray(supervised_rmse, dtype=np.float32)
    all_rmse = np.asarray(all_rmse, dtype=np.float32)

    pose_errors = pose_errors_from_point_trajectories(
        trajectory["first_frame_pos"].numpy()[supervised_indices],
        gt_positions[:, supervised_indices],
        pred_positions[:, supervised_indices],
        point_lens[supervised_indices],
        int(supervised_indices.size),
    )
    pose_summary = summarize_pose_errors(
        pose_errors["position_error"],
        pose_errors["orientation_error_rad"],
        rollout_start_index=2,
    )

    valid_counts = int(point_lens[supervised_indices].sum())
    metrics = {
        "episode_index": int(episode_index),
        "frame_indices": trajectory["frame_indices"].tolist(),
        "steps": int(steps),
        "stride": int(stride),
        "max_points": int(max_points),
        "object_lens": object_lens,
        "object_names": object_names,
        "supervised_object_indices": supervised_indices.astype(int).tolist(),
        "supervised_object_names": [
            object_names[index] if index < len(object_names) else f"object_{index}" for index in supervised_indices
        ],
        "supervised_points": valid_counts,
        "teacher_force_context": bool(teacher_force_context),
        "rmse_mean_all_frames": float(supervised_rmse.mean()),
        "rmse_mean_rollout_only": float(supervised_rmse[2:].mean()) if len(supervised_rmse) > 2 else float("nan"),
        "rmse_final": float(supervised_rmse[-1]),
        "rmse_all_objects_mean": float(all_rmse.mean()),
        "rmse_all_objects_final": float(all_rmse[-1]),
        "pose_position_rmse_all": pose_summary["position_rmse_all"],
        "pose_orientation_rmse_rad_all": pose_summary["orientation_rmse_rad_all"],
        "pose_orientation_rmse_deg_all": pose_summary["orientation_rmse_deg_all"],
        "pose_position_rmse_rollout": pose_summary["position_rmse_rollout"],
        "pose_orientation_rmse_rad_rollout": pose_summary["orientation_rmse_rad_rollout"],
        "pose_orientation_rmse_deg_rollout": pose_summary["orientation_rmse_deg_rollout"],
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    np.savez_compressed(
        output_dir / "trajectory.npz",
        gt_positions=gt_positions,
        pred_positions=pred_positions,
        frame_indices=trajectory["frame_indices"],
        point_lens=point_lens,
        loss_object_mask=loss_object_mask,
        supervised_rmse=supervised_rmse,
        all_rmse=all_rmse,
        pose_position_error=pose_errors["position_error"],
        pose_orientation_error_rad=pose_errors["orientation_error_rad"],
        pose_orientation_error_deg=pose_errors["orientation_error_deg"],
    )

    axis_points = np.concatenate(
        [
            _valid_points(gt_positions[offset], point_lens, all_indices)
            for offset in range(gt_positions.shape[0])
        ]
        + [
            _valid_points(pred_positions[offset], point_lens, all_indices)
            for offset in range(pred_positions.shape[0])
        ],
        axis=0,
    )
    colors = list(plt.get_cmap("tab10").colors)

    video_path = output_dir / "rollout.mp4"
    with imageio.get_writer(video_path, fps=video_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        iterator = tqdm(
            range(gt_positions.shape[0]),
            desc="render",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for offset in iterator:
            writer.append_data(
                _render_frame(
                    gt_positions[offset],
                    pred_positions[offset],
                    point_lens,
                    all_indices,
                    object_names,
                    colors,
                    axis_points,
                    int(trajectory["frame_indices"][offset]),
                    float(supervised_rmse[offset]),
                )
            )

    final_overlay_path = output_dir / "final_overlay.png"
    rmse_plot_path = output_dir / "rmse.png"
    pose_error_plot_path = output_dir / "pose_errors.png"
    _save_final_overlay(
        final_overlay_path,
        gt_positions[-1],
        pred_positions[-1],
        point_lens,
        all_indices,
        object_names,
        axis_points,
        int(trajectory["frame_indices"][-1]),
    )
    _save_rmse_plot(rmse_plot_path, trajectory["frame_indices"], supervised_rmse)
    _save_pose_error_plot(
        pose_error_plot_path,
        trajectory["frame_indices"],
        pose_errors["position_error"],
        pose_errors["orientation_error_deg"],
        object_names,
        supervised_indices,
    )

    metrics.update(
        {
            "output_dir": str(output_dir),
            "metrics_path": str(metrics_path),
            "video": str(video_path),
            "final_overlay": str(final_overlay_path),
            "rmse_plot": str(rmse_plot_path),
            "pose_error_plot": str(pose_error_plot_path),
        }
    )

    return metrics
