from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from rigidformer import Rigidformer
from rigidformer.movi import load_movi_scene_trajectory, movi_sample_paths
from rigidformer.pose_metrics import pose_errors_from_point_trajectories, summarize_pose_errors


def _default_hidden_layers(object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if cross_depth == 1:
        return (object_depth,)

    return tuple(round(i * object_depth / (cross_depth - 1)) for i in range(cross_depth))


def _parse_hidden_layers(value: str | None, object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if value is None or value == "":
        return _default_hidden_layers(object_depth, cross_depth)

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _checkpoint_arg(checkpoint_args: dict[str, Any], key: str, default: Any) -> Any:
    return checkpoint_args[key] if key in checkpoint_args and checkpoint_args[key] is not None else default


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value

    return config


def _build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> Rigidformer:
    checkpoint_args = checkpoint.get("args", {})

    object_self_attn_depth = int(_checkpoint_arg(checkpoint_args, "object_self_attn_depth", 4))
    anchor_cross_attn_depth = int(_checkpoint_arg(checkpoint_args, "anchor_cross_attn_depth", 4))

    model = Rigidformer(
        dim = int(_checkpoint_arg(checkpoint_args, "dim", 256)),
        dim_head = int(_checkpoint_arg(checkpoint_args, "dim_head", 64)),
        heads = int(_checkpoint_arg(checkpoint_args, "heads", 4)),
        object_self_attn_depth = object_self_attn_depth,
        anchor_cross_attn_depth = anchor_cross_attn_depth,
        object_hidden_layers = _parse_hidden_layers(
            _checkpoint_arg(checkpoint_args, "object_hidden_layers", None),
            object_self_attn_depth,
            anchor_cross_attn_depth,
        ),
        anchor_self_attn = bool(_checkpoint_arg(checkpoint_args, "anchor_self_attn", False)),
        num_anchors = int(_checkpoint_arg(checkpoint_args, "num_anchors", 4)),
        vertex_properties_dim = 3,
        use_platonic_transformer = bool(_checkpoint_arg(checkpoint_args, "use_platonic_transformer", False)),
        paper_architecture = bool(_checkpoint_arg(checkpoint_args, "paper_architecture", False)),
        vertex_feature_dim = int(_checkpoint_arg(checkpoint_args, "vertex_feature_dim", 1024)),
        avp_dim = int(_checkpoint_arg(checkpoint_args, "avp_dim", 256)),
        paper_pointnet_level_dim = int(_checkpoint_arg(checkpoint_args, "paper_pointnet_level_dim", 1024)),
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _valid_points(points: np.ndarray, point_lens: np.ndarray, object_lens: int) -> np.ndarray:
    chunks = []
    for object_index in range(object_lens):
        chunks.append(points[object_index, :point_lens[object_index]])

    return np.concatenate(chunks, axis = 0)


def _set_equal_3d_limits(ax, points: np.ndarray, margin: float = 0.1) -> None:
    mins = points.min(axis = 0)
    maxs = points.max(axis = 0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * (0.5 + margin), 1e-3)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_point_cloud(
    ax,
    points: np.ndarray,
    point_lens: np.ndarray,
    object_lens: int,
    colors: list[Any],
    title: str,
    axis_points: np.ndarray,
) -> None:
    for object_index in range(object_lens):
        object_points = points[object_index, :point_lens[object_index]]
        ax.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            s = 7,
            alpha = 0.85,
            color = colors[object_index % len(colors)],
            edgecolors = "none",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d_limits(ax, axis_points)
    ax.view_init(elev = 22, azim = -55)


def _render_frame(
    gt: np.ndarray,
    pred: np.ndarray,
    point_lens: np.ndarray,
    object_lens: int,
    colors: list[Any],
    axis_points: np.ndarray,
    frame_index: int,
    rmse: float,
) -> np.ndarray:
    fig = plt.figure(figsize = (12, 5.142857), dpi = 140)

    ax_gt = fig.add_subplot(1, 2, 1, projection = "3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection = "3d")

    _draw_point_cloud(
        ax_gt,
        gt,
        point_lens,
        object_lens,
        colors,
        f"Ground Truth | frame {frame_index}",
        axis_points,
    )
    _draw_point_cloud(
        ax_pred,
        pred,
        point_lens,
        object_lens,
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
    object_lens: int,
    axis_points: np.ndarray,
    frame_index: int,
) -> None:
    fig = plt.figure(figsize = (7, 6), dpi = 160)
    ax = fig.add_subplot(1, 1, 1, projection = "3d")

    for object_index in range(object_lens):
        gt_points = gt[object_index, :point_lens[object_index]]
        pred_points = pred[object_index, :point_lens[object_index]]
        ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s = 8, alpha = 0.55, color = "tab:green", edgecolors = "none")
        ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], s = 8, alpha = 0.55, color = "tab:red", edgecolors = "none")

    ax.set_title(f"Final Frame Overlay | frame {frame_index}\nGT green, prediction red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d_limits(ax, axis_points)
    ax.view_init(elev = 22, azim = -55)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_rmse_plot(output_path: Path, frame_indices: np.ndarray, rmse: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize = (7, 4), dpi = 160)
    ax.plot(frame_indices, rmse, marker = "o", linewidth = 1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Point RMSE")
    ax.set_title("Rollout Error")
    ax.grid(True, alpha = 0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_pose_error_plot(
    output_path: Path,
    frame_indices: np.ndarray,
    position_error: np.ndarray,
    orientation_error_deg: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize = (8, 7), dpi = 160, sharex = True)

    for object_index in range(position_error.shape[1]):
        axes[0].plot(frame_indices, position_error[:, object_index], linewidth = 1.2, label = f"obj {object_index}")
        axes[1].plot(frame_indices, orientation_error_deg[:, object_index], linewidth = 1.2, label = f"obj {object_index}")

    axes[0].set_ylabel("Position error")
    axes[0].set_title("Per-Object Pose Error")
    axes[0].grid(True, alpha = 0.25)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].grid(True, alpha = 0.25)
    axes[0].legend(ncol = 4, fontsize = 8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


@torch.no_grad()
def _rollout(
    model: Rigidformer,
    positions: torch.Tensor,
    first_frame_pos: torch.Tensor,
    vertex_properties: torch.Tensor,
    object_lens: int,
    object_point_lens: torch.Tensor,
    delta_time: float,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    object_lens_tensor = torch.tensor([object_lens], dtype = torch.long, device = device)
    point_lens = object_point_lens.unsqueeze(0).to(device)
    vertex_properties = vertex_properties.unsqueeze(0).to(device)
    first_frame_pos = first_frame_pos.unsqueeze(0).to(device)
    delta_times = torch.tensor([delta_time], dtype = torch.float32, device = device)

    prev = positions[0].unsqueeze(0).to(device)
    current = positions[1].unsqueeze(0).to(device)
    predicted = [prev.cpu(), current.cpu()]
    anchor_indices = None

    for _ in tqdm(range(positions.shape[0] - 2), desc = "rollout", dynamic_ncols = True):
        with torch.amp.autocast(device_type = device.type, enabled = use_amp):
            pred, intermediates = model(
                delta_times = delta_times,
                vertex_properties = vertex_properties,
                object_pos = current,
                object_pos_prev = prev,
                object_pos_next = None,
                object_first_frame_pos = first_frame_pos,
                anchor_indices = anchor_indices,
                object_lens = object_lens_tensor,
                object_point_lens = point_lens,
                return_intermediates = True,
            )

        if anchor_indices is None:
            anchor_indices = intermediates.anchor_indices

        next_pos = pred.object_pos_next.detach()
        predicted.append(next_pos.cpu())
        prev, current = current, next_pos

    return torch.cat(predicted, dim = 0)


def main() -> None:
    parser = argparse.ArgumentParser(description = "Evaluate a RigidFormer checkpoint and render MOVi rollout visualizations.")
    parser.add_argument("--checkpoint", type = Path, required = True)
    parser.add_argument("--dataset-dir", type = Path, default = Path("data/movi/MOVi-spheres"))
    parser.add_argument("--objects-dir", type = Path, default = Path("data/movi_objects/objects"))
    parser.add_argument("--output-dir", type = Path, default = Path("eval/rigidformer-movi"))
    parser.add_argument("--split", type = str, default = "val", choices = ("train", "val", "test", "all"))
    parser.add_argument("--sample-index", type = int, default = 0)
    parser.add_argument("--sample-id", type = str, default = None)
    parser.add_argument("--start-frame", type = int, default = 0)
    parser.add_argument("--steps", type = int, default = 60)
    parser.add_argument("--stride", type = int, default = 4)
    parser.add_argument("--max-points", type = int, default = 256)
    parser.add_argument("--random-points", action = "store_true")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--video-fps", type = int, default = 12)
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action = "store_true")
    parser.add_argument("--wandb", action = "store_true")
    parser.add_argument("--wandb-project", type = str, default = "rigidformer")
    parser.add_argument("--wandb-entity", type = str, default = "tjrcjf410-seoul-national-university")
    parser.add_argument("--wandb-run-name", type = str, default = None)
    parser.add_argument("--wandb-mode", type = str, default = "online", choices = ("online", "offline", "disabled"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents = True, exist_ok = True)

    if args.sample_id is not None:
        sample_path = args.dataset_dir / args.sample_id
    else:
        paths = movi_sample_paths(args.dataset_dir, split = args.split, seed = args.seed)
        if args.sample_index >= len(paths):
            raise IndexError(f"sample-index {args.sample_index} out of range for split {args.split} with {len(paths)} samples")
        sample_path = paths[args.sample_index]

    frame_indices = np.arange(args.start_frame, args.start_frame + (args.steps + 2) * args.stride, args.stride)
    trajectory = load_movi_scene_trajectory(
        sample_path,
        args.objects_dir,
        frame_indices = frame_indices,
        max_points = args.max_points,
        random_points = args.random_points,
        seed = args.seed,
    )

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location = "cpu", weights_only = False)
    model = _build_model_from_checkpoint(checkpoint, device)

    use_amp = args.amp and device.type == "cuda"
    pred_positions = _rollout(
        model,
        trajectory["positions"],
        trajectory["first_frame_pos"],
        trajectory["vertex_properties"],
        trajectory["object_lens"],
        trajectory["object_point_lens"],
        args.stride / trajectory["frame_rate"],
        device,
        use_amp,
    ).numpy()

    gt_positions = trajectory["positions"].numpy()
    point_lens = trajectory["object_point_lens"].numpy()
    object_lens = int(trajectory["object_lens"])
    valid_counts = int(point_lens[:object_lens].sum())

    rmse = []
    for frame_offset in range(gt_positions.shape[0]):
        gt_valid = _valid_points(gt_positions[frame_offset], point_lens, object_lens)
        pred_valid = _valid_points(pred_positions[frame_offset], point_lens, object_lens)
        rmse.append(float(np.sqrt(np.mean((gt_valid - pred_valid) ** 2))))

    rmse = np.asarray(rmse, dtype = np.float32)

    pose_errors = pose_errors_from_point_trajectories(
        trajectory["first_frame_pos"].numpy(),
        gt_positions,
        pred_positions,
        point_lens,
        object_lens,
    )
    pose_summary = summarize_pose_errors(
        pose_errors["position_error"],
        pose_errors["orientation_error_rad"],
        rollout_start_index = 2,
    )

    object_metadata = trajectory["metadata"]["instances"]
    per_object_pose_metrics = []
    for object_index in range(object_lens):
        per_object_pose_metrics.append({
            "object_index": object_index,
            "shape": object_metadata[object_index].get("shape", object_metadata[object_index].get("asset_id", "")),
            "position_rmse": float(pose_summary["position_rmse_per_object"][object_index]),
            "orientation_rmse_rad": float(pose_summary["orientation_rmse_rad_per_object"][object_index]),
            "orientation_rmse_deg": float(pose_summary["orientation_rmse_deg_per_object"][object_index]),
        })

    metrics = {
        "checkpoint": str(args.checkpoint),
        "sample_id": trajectory["sample_id"],
        "split": args.split,
        "frame_indices": trajectory["frame_indices"].tolist(),
        "steps": args.steps,
        "stride": args.stride,
        "max_points": args.max_points,
        "object_lens": object_lens,
        "valid_points": valid_counts,
        "rmse_mean_all_frames": float(rmse.mean()),
        "rmse_mean_rollout_only": float(rmse[2:].mean()) if len(rmse) > 2 else float("nan"),
        "rmse_final": float(rmse[-1]),
        "pose_position_rmse_all": pose_summary["position_rmse_all"],
        "pose_orientation_rmse_rad_all": pose_summary["orientation_rmse_rad_all"],
        "pose_orientation_rmse_deg_all": pose_summary["orientation_rmse_deg_all"],
        "pose_position_rmse_rollout": pose_summary["position_rmse_rollout"],
        "pose_orientation_rmse_rad_rollout": pose_summary["orientation_rmse_rad_rollout"],
        "pose_orientation_rmse_deg_rollout": pose_summary["orientation_rmse_deg_rollout"],
        "per_object_pose_metrics": per_object_pose_metrics,
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent = 2)

    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        gt_positions = gt_positions,
        pred_positions = pred_positions,
        frame_indices = trajectory["frame_indices"],
        point_lens = point_lens,
        rmse = rmse,
        pose_position_error = pose_errors["position_error"],
        pose_orientation_error_rad = pose_errors["orientation_error_rad"],
        pose_orientation_error_deg = pose_errors["orientation_error_deg"],
    )

    pose_errors_csv_path = args.output_dir / "pose_errors.csv"
    with pose_errors_csv_path.open("w") as f:
        f.write("frame_index,object_index,shape,position_error,orientation_error_rad,orientation_error_deg\n")
        for frame_offset, frame_index in enumerate(trajectory["frame_indices"]):
            for object_index in range(object_lens):
                shape = per_object_pose_metrics[object_index]["shape"]
                f.write(
                    f"{int(frame_index)},{object_index},{shape},"
                    f"{float(pose_errors['position_error'][frame_offset, object_index])},"
                    f"{float(pose_errors['orientation_error_rad'][frame_offset, object_index])},"
                    f"{float(pose_errors['orientation_error_deg'][frame_offset, object_index])}\n"
                )

    axis_points = np.concatenate([
        _valid_points(gt_positions[i], point_lens, object_lens)
        for i in range(gt_positions.shape[0])
    ], axis = 0)
    colors = list(plt.get_cmap("tab10").colors)

    video_path = args.output_dir / "rollout.mp4"
    with imageio.get_writer(video_path, fps = args.video_fps, codec = "libx264", quality = 8, macro_block_size = 1) as writer:
        for offset in tqdm(range(gt_positions.shape[0]), desc = "render", dynamic_ncols = True):
            writer.append_data(_render_frame(
                gt_positions[offset],
                pred_positions[offset],
                point_lens,
                object_lens,
                colors,
                axis_points,
                int(trajectory["frame_indices"][offset]),
                float(rmse[offset]),
            ))

    final_overlay_path = args.output_dir / "final_overlay.png"
    rmse_plot_path = args.output_dir / "rmse.png"
    pose_error_plot_path = args.output_dir / "pose_errors.png"
    _save_final_overlay(
        final_overlay_path,
        gt_positions[-1],
        pred_positions[-1],
        point_lens,
        object_lens,
        axis_points,
        int(trajectory["frame_indices"][-1]),
    )
    _save_rmse_plot(rmse_plot_path, trajectory["frame_indices"], rmse)
    _save_pose_error_plot(
        pose_error_plot_path,
        trajectory["frame_indices"],
        pose_errors["position_error"],
        pose_errors["orientation_error_deg"],
    )

    print(json.dumps({
        "event": "evaluation_complete",
        "output_dir": str(args.output_dir),
        "video": str(video_path),
        "final_overlay": str(final_overlay_path),
        "rmse_plot": str(rmse_plot_path),
        "pose_error_plot": str(pose_error_plot_path),
        "pose_errors_csv": str(pose_errors_csv_path),
        **metrics,
    }, indent = 2), flush = True)

    if args.wandb:
        import wandb

        run = wandb.init(
            project = args.wandb_project,
            entity = args.wandb_entity,
            name = args.wandb_run_name,
            mode = args.wandb_mode,
            config = _jsonable_args(args) | metrics,
            dir = str(args.output_dir),
            job_type = "eval",
        )
        run.log({
            "eval/rmse_mean_all_frames": metrics["rmse_mean_all_frames"],
            "eval/rmse_mean_rollout_only": metrics["rmse_mean_rollout_only"],
            "eval/rmse_final": metrics["rmse_final"],
            "eval/pose_position_rmse_rollout": metrics["pose_position_rmse_rollout"],
            "eval/pose_orientation_rmse_deg_rollout": metrics["pose_orientation_rmse_deg_rollout"],
            "eval/final_overlay": wandb.Image(str(final_overlay_path)),
            "eval/rmse_plot": wandb.Image(str(rmse_plot_path)),
            "eval/pose_error_plot": wandb.Image(str(pose_error_plot_path)),
            "eval/rollout": wandb.Video(str(video_path), fps = args.video_fps, format = "mp4"),
        })
        run.finish()


if __name__ == "__main__":
    main()
