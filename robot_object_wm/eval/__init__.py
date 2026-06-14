"""Evaluation helpers for robot-object world models."""

from .animation import render_checkpoint_animation, render_dataset_animation, save_point_animation
from .episode import (
    EpisodeData,
    EpisodeRollout,
    compute_joint_positions,
    load_episode_data,
    per_horizon_episode_errors,
    precompute_prediction_trajectories,
    rollout_episode,
    rollout_metrics,
)
from .plots import (
    save_evaluation_artifacts,
    save_evaluation_curves,
    save_metrics_json,
    save_per_horizon_csv,
    save_prediction_error_curves,
    save_trajectory_plot,
)
from .rollout import (
    EvaluationSummary,
    LoadedWorldModel,
    RolloutResult,
    evaluate_checkpoint,
    evaluate_loader,
    euclidean_position_error,
    load_checkpoint_model,
    make_eval_loader,
    predict_batch,
    rollout_mse,
)

__all__ = [
    "EpisodeData",
    "EpisodeRollout",
    "EvaluationSummary",
    "LoadedWorldModel",
    "RolloutResult",
    "compute_joint_positions",
    "evaluate_checkpoint",
    "evaluate_loader",
    "euclidean_position_error",
    "load_checkpoint_model",
    "load_episode_data",
    "make_eval_loader",
    "per_horizon_episode_errors",
    "predict_batch",
    "precompute_prediction_trajectories",
    "render_checkpoint_animation",
    "render_dataset_animation",
    "rollout_episode",
    "rollout_metrics",
    "rollout_mse",
    "save_evaluation_artifacts",
    "save_evaluation_curves",
    "save_metrics_json",
    "save_per_horizon_csv",
    "save_point_animation",
    "save_prediction_error_curves",
    "save_trajectory_plot",
]
