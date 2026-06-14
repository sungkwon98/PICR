from __future__ import annotations

import csv
import json
import os
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_metrics_json(payload: Mapping[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_ready(payload), file, indent=2, sort_keys=True)
    return output_path


def save_per_horizon_csv(per_horizon: Mapping[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in per_horizon.items()}
    if "horizon" not in arrays:
        first = next(iter(arrays.values()))
        arrays["horizon"] = np.arange(1, first.shape[0] + 1)
    keys = ["horizon"] + [key for key in arrays if key != "horizon"]
    rows = int(arrays["horizon"].shape[0])
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for idx in range(rows):
            writer.writerow({key: _scalar(arrays[key][idx]) for key in keys})
    return output_path


def save_error_curve(
    horizon: np.ndarray,
    mean_error: np.ndarray,
    output_path: str,
    ylabel: str = "mean position error [m]",
    title: str = "Open-loop prediction error",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizon, mean_error, marker="o")
    ax.set_xlabel("rollout horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_evaluation_curves(summary, output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}
    for key, values in summary.per_horizon.items():
        horizon = np.arange(1, len(values) + 1)
        output_path = os.path.join(output_dir, f"{key}.png")
        ylabel = key.replace("_", " ")
        written[key] = save_error_curve(
            horizon=horizon,
            mean_error=np.asarray(values),
            output_path=output_path,
            ylabel=ylabel,
            title=f"{getattr(summary, 'architecture', 'WMDynamics')} {key}",
        )
    return written


def save_prediction_error_curves(
    per_horizon: Mapping[str, Any],
    output_path: str,
    *,
    title: str = "Open-loop prediction error by horizon",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    horizon = np.asarray(per_horizon["horizon"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    series = [
        (
            axes[0],
            "target_position_error_mean_m",
            "target_position_error_std_m",
            "Target position error",
            "tab:red",
        ),
        (
            axes[1],
            "object_position_error_mean_m",
            "object_position_error_std_m",
            "Object position error",
            "tab:blue",
        ),
    ]
    for ax, mean_key, std_key, label, color in series:
        mean = np.asarray(per_horizon[mean_key], dtype=np.float64)
        std = np.asarray(per_horizon.get(std_key, np.zeros_like(mean)), dtype=np.float64)
        ax.plot(horizon, mean * 1000.0, color=color, marker="o", linewidth=2.2, label="mean")
        lo = np.clip(mean - std, 0.0, None) * 1000.0
        hi = (mean + std) * 1000.0
        ax.fill_between(horizon, lo, hi, color=color, alpha=0.15, label="+/- 1 std")
        ax.set_title(label)
        ax.set_xlabel("rollout horizon [steps]")
        ax.set_ylabel("position error [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_trajectory_plot(
    *,
    real_episode_target: np.ndarray,
    real_episode_object: np.ndarray,
    gt_target: np.ndarray,
    pred_target: np.ndarray,
    gt_object: np.ndarray,
    pred_object: np.ndarray,
    time: np.ndarray,
    output_path: str,
    episode_name: str,
    target_name: str,
    start_t: int,
) -> tuple[str, dict[str, float]]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    target_error = np.linalg.norm(pred_target - gt_target, axis=-1)
    object_error = np.linalg.norm(pred_object - gt_object, axis=-1)
    all_points = np.concatenate(
        [real_episode_target, pred_target, real_episode_object, pred_object],
        axis=0,
    )

    fig = plt.figure(figsize=(21, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.1])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(
        real_episode_target[:, 0],
        real_episode_target[:, 1],
        real_episode_target[:, 2],
        color="black",
        linewidth=1.8,
        label=f"Real full episode {target_name}",
    )
    ax3d.plot(
        pred_target[:, 0],
        pred_target[:, 1],
        pred_target[:, 2],
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label=f"Pred {target_name}",
    )
    ax3d.plot(
        real_episode_object[:, 0],
        real_episode_object[:, 1],
        real_episode_object[:, 2],
        color="tab:blue",
        linewidth=1.8,
        label="Real full episode object",
    )
    ax3d.plot(
        pred_object[:, 0],
        pred_object[:, 1],
        pred_object[:, 2],
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Pred object",
    )
    ax3d.scatter(*gt_target[0], color="tab:purple", s=35, marker="o", label=f"Real at prediction start {target_name}")
    ax3d.scatter(*gt_object[0], color="tab:cyan", s=35, marker="o", label="Real at prediction start object")
    ax3d.scatter(*pred_target[-1], color="tab:red", s=40, marker="x", label=f"Pred {target_name} end")
    ax3d.scatter(*pred_object[-1], color="tab:orange", s=40, marker="x", label="Pred object end")
    ax3d.set_title(f"{target_name.title()}/Object Trajectory ({episode_name}, start_t={start_t})")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    _set_axes_equal(ax3d, all_points)

    handles, labels = ax3d.get_legend_handles_labels()
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        fontsize=11,
        markerscale=1.7,
        handlelength=2.6,
        borderpad=1.0,
        labelspacing=0.75,
    )

    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.plot(time, target_error, color="tab:red", label=f"{target_name} error")
    ax_err.plot(time, object_error, color="tab:blue", label="object error")
    ax_err.set_title("Euclidean Position Error")
    ax_err.set_xlabel("episode time [s]")
    ax_err.set_ylabel("Euclidean error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    metrics = {
        f"{target_name}_position_rmse_m": float(np.sqrt(np.mean(target_error**2))),
        "object_position_rmse_m": float(np.sqrt(np.mean(object_error**2))),
        f"{target_name}_position_error_mean_m": float(np.mean(target_error)),
        "object_position_error_mean_m": float(np.mean(object_error)),
    }
    return output_path, metrics


def save_evaluation_artifacts(summary, output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    written = {
        "summary_json": save_metrics_json(summary.to_dict(), os.path.join(output_dir, "summary.json")),
        "per_horizon_csv": save_per_horizon_csv(summary.per_horizon, os.path.join(output_dir, "per_horizon_metrics.csv")),
    }
    curve_paths = save_evaluation_curves(summary, os.path.join(output_dir, "curves"))
    written.update({f"curve_{key}": value for key, value in curve_paths.items()})
    return written


def _set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float((maxs - mins).max()) * 0.5, 1.0e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _scalar(value: Any) -> int | float | str:
    if isinstance(value, np.generic):
        return value.item()
    if np.isscalar(value):
        return value
    return str(value)
