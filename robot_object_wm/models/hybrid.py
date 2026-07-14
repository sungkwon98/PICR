from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

try:
    from rigidformer import Rigidformer
except ImportError as exc:  # pragma: no cover - environment dependent.
    Rigidformer = None  # type: ignore[assignment]
    _RIGIDFORMER_IMPORT_ERROR = exc
else:
    _RIGIDFORMER_IMPORT_ERROR = None

from robot_object_wm.data.hdf5_schema import RobotObjectStateLayout
from robot_object_wm.models.delan import DeLaNRobotDynamics
from robot_object_wm.models.rwm import RWMEnsemble
from robot_object_wm.models.utils import (
    FrankaGripperPointCloudFK,
    normalize_quat,
    quat_conjugate,
    quat_mul,
    quat_to_rotation_matrix,
)


HYBRID_ROBOT_MODEL_TYPES = ("rwm", "delan")
HYBRID_ROLLOUT_FEEDBACK_MODES = ("robot_native", "rigidformer_pose")
HYBRID_GRIPPER_POINTCLOUD_MODES = ("gt", "predicted_fk")


@dataclass(frozen=True)
class RigidFormerObjectConfig:
    max_points: int = 1024
    dim: int = 256
    dim_head: int = 64
    heads: int = 4
    num_anchors: int = 4
    object_self_attn_depth: int = 2
    anchor_cross_attn_depth: int = 2
    object_hidden_layers: tuple[int, ...] | None = None
    anchor_self_attn: bool = False
    use_platonic_transformer: bool = True
    paper_architecture: bool = True
    vertex_feature_dim: int = 256
    avp_dim: int = 128
    paper_pointnet_level_dim: int = 256
    pos_loss_weight: float = 10.0
    acc_loss_weight: float = 1.0


def build_rigidformer_object_model(cfg: RigidFormerObjectConfig) -> nn.Module:
    if Rigidformer is None:
        raise ImportError("RigidFormer is required for model_type='hybrid'.") from _RIGIDFORMER_IMPORT_ERROR
    object_hidden_layers = cfg.object_hidden_layers
    if object_hidden_layers is None:
        object_hidden_layers = _default_object_hidden_layers(
            cfg.object_self_attn_depth,
            cfg.anchor_cross_attn_depth,
        )
    return Rigidformer(
        dim=cfg.dim,
        dim_head=cfg.dim_head,
        heads=cfg.heads,
        object_self_attn_depth=cfg.object_self_attn_depth,
        anchor_cross_attn_depth=cfg.anchor_cross_attn_depth,
        object_hidden_layers=tuple(object_hidden_layers),
        anchor_self_attn=cfg.anchor_self_attn,
        num_anchors=cfg.num_anchors,
        vertex_properties_dim=3,
        use_platonic_transformer=cfg.use_platonic_transformer,
        paper_architecture=cfg.paper_architecture,
        vertex_feature_dim=cfg.vertex_feature_dim,
        avp_dim=cfg.avp_dim,
        paper_pointnet_level_dim=cfg.paper_pointnet_level_dim,
        pos_loss_weight=cfg.pos_loss_weight,
        acc_loss_weight=cfg.acc_loss_weight,
    )


def _default_object_hidden_layers(object_depth: int, cross_depth: int) -> tuple[int, ...]:
    if cross_depth == 1:
        return (object_depth,)
    return tuple(int(i * object_depth / (cross_depth - 1)) for i in range(cross_depth))


def normalize_hybrid_robot_model_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in HYBRID_ROBOT_MODEL_TYPES:
        raise ValueError(f"hybrid_robot_model_type must be one of {HYBRID_ROBOT_MODEL_TYPES}; got {value!r}.")
    return normalized


def normalize_hybrid_feedback_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in HYBRID_ROLLOUT_FEEDBACK_MODES:
        raise ValueError(
            f"hybrid_rollout_feedback_mode must be one of {HYBRID_ROLLOUT_FEEDBACK_MODES}; got {value!r}."
        )
    return normalized


def normalize_hybrid_gripper_pointcloud_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in HYBRID_GRIPPER_POINTCLOUD_MODES:
        raise ValueError(
            f"hybrid_gripper_pointcloud_mode must be one of {HYBRID_GRIPPER_POINTCLOUD_MODES}; got {value!r}."
        )
    return normalized


class HybridRigidFormerWMDynamics(nn.Module):
    """Robot state-space dynamics plus RigidFormer cube point-cloud dynamics."""

    def __init__(
        self,
        *,
        robot: nn.Module,
        rigidformer: nn.Module,
        layout: RobotObjectStateLayout,
        robot_backend: str,
        dt: float,
        action_type: str = "torque",
        feedback_mode: str = "robot_native",
        gripper_pointcloud_mode: str = "gt",
        subtract_env_origin: bool = True,
        pointcloud_file: str | None = None,
        rigidformer_max_points: int = 1024,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.rigidformer = rigidformer
        self.layout = layout
        self.robot_backend = normalize_hybrid_robot_model_type(robot_backend)
        self.feedback_mode = normalize_hybrid_feedback_mode(feedback_mode)
        self.gripper_pointcloud_mode = normalize_hybrid_gripper_pointcloud_mode(gripper_pointcloud_mode)
        self.dt = float(dt)
        self.action_type = str(action_type).strip().lower()
        self.subtract_env_origin = bool(subtract_env_origin)
        self.pointcloud_file = pointcloud_file
        self.rigidformer_max_points = int(rigidformer_max_points)
        self.model_type = "hybrid"
        self.robot_dof = int(layout.robot_dof)
        self.state_dim = int(layout.state_dim)
        self.torque_dim = int(layout.torque_dim)
        self.history_len = 1
        self.gripper_fk = FrankaGripperPointCloudFK(robot_dof=layout.robot_dof)
        if self.robot_backend == "rwm":
            if not isinstance(robot, RWMEnsemble):
                raise TypeError("robot_backend='rwm' requires an RWMEnsemble robot module.")
            self.history_len = int(robot.history_horizon)
            self.robot.model_type = "rwm"
            self.robot.action_type = self.action_type
        elif not isinstance(robot, DeLaNRobotDynamics):
            raise TypeError("robot_backend='delan' requires a DeLaNRobotDynamics robot module.")

    def forward(
        self,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        object_context: torch.Tensor | None = None,
        history_torques: torch.Tensor | None = None,
        deterministic_context: bool | None = None,
        *,
        return_aux: bool = False,
        return_context: bool = False,
        pointcloud_batch: dict[str, Any] | None = None,
        history_actions: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        feedback_mode: str | None = None,
    ):
        del object_context, deterministic_context
        if return_context:
            raise ValueError("HybridRigidFormerWMDynamics does not use a latent context encoder.")
        if future_torques.ndim != 3:
            raise ValueError("future_torques must have shape (batch, horizon, torque_dim).")
        if history_states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {history_states.shape[-1]}.")
        mode = normalize_hybrid_feedback_mode(feedback_mode or self.feedback_mode)
        gripper_mode = normalize_hybrid_gripper_pointcloud_mode(self.gripper_pointcloud_mode)
        horizon = int(future_torques.shape[1])
        rf_state = self._prepare_rigidformer_rollout_state(
            pointcloud_batch,
            horizon,
            history_states=history_states,
            gripper_mode=gripper_mode,
            device=history_states.device,
        )

        if self.robot_backend == "rwm":
            pred, aux = self._forward_rwm(
                history_states=history_states,
                history_torques=history_torques,
                future_torques=future_torques,
                history_actions=history_actions,
                future_actions=future_actions,
                rf_state=rf_state,
                feedback_mode=mode,
                gripper_mode=gripper_mode,
            )
        else:
            pred, aux = self._forward_delan(
                history_states=history_states,
                future_torques=future_torques,
                rf_state=rf_state,
                feedback_mode=mode,
                gripper_mode=gripper_mode,
            )
        if return_aux:
            return pred, aux
        return pred

    def rigidformer_pose_rollout(
        self,
        pointcloud_batch: dict[str, Any] | None,
        horizon: int,
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        if pointcloud_batch is None or "pc_object_pos_rollout" not in pointcloud_batch:
            return None
        pc = pointcloud_tensors(pointcloud_batch, device=device)
        sequence = pc["object_pos_rollout"]
        if sequence.shape[1] < horizon + 2:
            raise ValueError(
                f"pc_object_pos_rollout length {sequence.shape[1]} is too short for horizon {horizon}."
            )
        prev = sequence[:, 0]
        current = sequence[:, 1]
        first = pc["object_first_frame_pos"]
        anchor_indices = None
        pred_points: list[torch.Tensor] = []
        pred_pos: list[torch.Tensor] = []
        pred_quat: list[torch.Tensor] = []
        loss_mask = pc["loss_object_mask"].bool()
        context_mask = ~loss_mask[:, : current.shape[1]]
        has_context = bool(context_mask.any().detach().cpu().item())

        for step in range(horizon):
            pred, intermediates = self.rigidformer(
                delta_times=pc["delta_times"],
                vertex_properties=pc["vertex_properties"],
                object_pos=current,
                object_pos_prev=prev,
                object_first_frame_pos=first,
                anchor_indices=anchor_indices,
                object_lens=pc["object_lens"],
                object_point_lens=pc["object_point_lens"],
                return_intermediates=True,
            )
            if anchor_indices is None:
                anchor_indices = intermediates.anchor_indices
            next_points = pred.object_pos_next
            if has_context:
                gt_next = sequence[:, step + 2]
                next_points = next_points.clone()
                next_points[context_mask] = gt_next[context_mask]

            cube_points = next_points[:, 0]
            pos, quat = pose_from_cube_points(
                first_frame_points=first[:, 0],
                target_points=cube_points,
                first_object_pos_w=pc["first_object_pos_w"],
                first_object_quat=pc["first_object_quat"],
                env_origin=pc["env_origin"],
                subtract_env_origin=self.subtract_env_origin,
                point_lens=pc["object_point_lens"][:, 0],
            )
            pred_points.append(next_points)
            pred_pos.append(pos)
            pred_quat.append(quat)
            prev, current = current, next_points

        return {
            "points": torch.stack(pred_points, dim=1),
            "pos": torch.stack(pred_pos, dim=1),
            "quat": torch.stack(pred_quat, dim=1),
        }

    def _prepare_rigidformer_rollout_state(
        self,
        pointcloud_batch: dict[str, Any] | None,
        horizon: int,
        *,
        history_states: torch.Tensor | None,
        gripper_mode: str,
        device: torch.device,
    ) -> dict[str, Any] | None:
        if pointcloud_batch is None or "pc_object_pos_rollout" not in pointcloud_batch:
            return None
        pc = pointcloud_tensors(pointcloud_batch, device=device)
        sequence = pc["object_pos_rollout"]
        if sequence.shape[1] < horizon + 2:
            raise ValueError(
                f"pc_object_pos_rollout length {sequence.shape[1]} is too short for horizon {horizon}."
            )
        prev = sequence[:, 0].clone()
        current = sequence[:, 1].clone()
        state: dict[str, Any] = {
            "pc": pc,
            "sequence": sequence,
            "prev": prev,
            "current": current,
            "anchor_indices": None,
            "canonical_gripper": None,
        }
        if gripper_mode == "predicted_fk":
            if history_states is None or history_states.shape[1] < 2:
                raise ValueError("predicted_fk gripper pointcloud mode requires at least two history states.")
            canonical = self._canonical_gripper_points(pc)
            state["canonical_gripper"] = canonical
            prev[:, 1] = self._gripper_points_from_state(history_states[:, -2], pc, canonical)
            current[:, 1] = self._gripper_points_from_state(history_states[:, -1], pc, canonical)
        return state

    def _rigidformer_rollout_step(
        self,
        state: dict[str, Any],
        step: int,
        *,
        next_robot_state: torch.Tensor | None,
        gripper_mode: str,
    ) -> dict[str, torch.Tensor]:
        pc = state["pc"]
        pred, intermediates = self.rigidformer(
            delta_times=pc["delta_times"],
            vertex_properties=pc["vertex_properties"],
            object_pos=state["current"],
            object_pos_prev=state["prev"],
            object_first_frame_pos=pc["object_first_frame_pos"],
            anchor_indices=state["anchor_indices"],
            object_lens=pc["object_lens"],
            object_point_lens=pc["object_point_lens"],
            return_intermediates=True,
        )
        if state["anchor_indices"] is None:
            state["anchor_indices"] = intermediates.anchor_indices
        next_points = pred.object_pos_next
        loss_mask = pc["loss_object_mask"].bool()
        context_mask = ~loss_mask[:, : state["current"].shape[1]]
        has_context = bool(context_mask.any().detach().cpu().item())
        if has_context:
            gt_next = state["sequence"][:, step + 2]
            next_points = next_points.clone()
            next_points[context_mask] = gt_next[context_mask]
        if gripper_mode == "predicted_fk":
            if next_robot_state is None:
                raise ValueError("predicted_fk gripper pointcloud mode requires next_robot_state.")
            if state["current"].shape[1] < 2:
                raise ValueError("predicted_fk gripper pointcloud mode requires a gripper object at index 1.")
            next_points = next_points.clone()
            next_points[:, 1] = self._gripper_points_from_state(next_robot_state, pc, state["canonical_gripper"])

        cube_points = next_points[:, 0]
        pos, quat = pose_from_cube_points(
            first_frame_points=pc["object_first_frame_pos"][:, 0],
            target_points=cube_points,
            first_object_pos_w=pc["first_object_pos_w"],
            first_object_quat=pc["first_object_quat"],
            env_origin=pc["env_origin"],
            subtract_env_origin=self.subtract_env_origin,
            point_lens=pc["object_point_lens"][:, 0],
        )
        state["prev"], state["current"] = state["current"], next_points
        return {"points": next_points, "pos": pos, "quat": quat}

    def _canonical_gripper_points(self, pc: dict[str, torch.Tensor]) -> torch.Tensor:
        first = pc["object_first_frame_pos"]
        if first.shape[1] < 2:
            raise ValueError("predicted_fk gripper pointcloud mode requires a gripper object at index 1.")
        required = ("gripper_part_ids", "first_robot_q_abs", "first_robot_q_obs")
        missing = [key for key in required if key not in pc]
        if missing:
            raise ValueError(
                "predicted_fk gripper pointcloud mode requires pointcloud metadata keys: "
                + ", ".join(missing)
            )
        return self.gripper_fk.inverse_points(
            first[:, 1],
            pc["first_robot_q_abs"],
            pc["gripper_part_ids"],
            origin=pc["env_origin"] if self.subtract_env_origin else None,
        )

    def _state_to_absolute_q(self, state: torch.Tensor, pc: dict[str, torch.Tensor]) -> torch.Tensor:
        q_obs = state[:, self.layout.robot_q_slice]
        q_abs = q_obs.clone()
        dim = min(q_obs.shape[-1], pc["first_robot_q_abs"].shape[-1], pc["first_robot_q_obs"].shape[-1])
        q_abs[:, :dim] = q_obs[:, :dim] + pc["first_robot_q_abs"][:, :dim] - pc["first_robot_q_obs"][:, :dim]
        return q_abs

    def _gripper_points_from_state(
        self,
        state: torch.Tensor,
        pc: dict[str, torch.Tensor],
        canonical_gripper: torch.Tensor,
    ) -> torch.Tensor:
        q_abs = self._state_to_absolute_q(state, pc)
        return self.gripper_fk.forward_points(
            canonical_gripper,
            q_abs,
            pc["gripper_part_ids"],
            origin=pc["env_origin"] if self.subtract_env_origin else None,
        )

    def _forward_rwm(
        self,
        *,
        history_states: torch.Tensor,
        history_torques: torch.Tensor | None,
        future_torques: torch.Tensor,
        history_actions: torch.Tensor | None,
        future_actions: torch.Tensor | None,
        rf_state: dict[str, Any] | None,
        feedback_mode: str,
        gripper_mode: str,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        pred_steps: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []
        if self.action_type == "policy":
            if history_actions is None or future_actions is None:
                raise ValueError("Hybrid RWM action_type='policy' requires history_actions and future_actions.")
            first_actions = torch.cat([history_actions, future_actions[:, :1]], dim=1)
            action_values = future_actions
        elif self.action_type == "torque":
            if history_torques is None:
                history_torques = history_states.new_zeros(
                    history_states.shape[0],
                    history_states.shape[1],
                    self.torque_dim,
                )
            first_actions = history_torques
            action_values = future_torques
        else:
            raise ValueError(f"Hybrid RWM action_type must be 'policy' or 'torque', got {self.action_type!r}.")

        try:
            self.robot.reset()
            x_state = history_states
            prev_output_state = history_states[:, -1]
            for step in range(action_values.shape[1]):
                x_action = first_actions if step == 0 else action_values[:, step : step + 1]
                robot_pred, aleatoric, epistemic = self.robot(x_state, x_action)
                rf_step = (
                    self._rigidformer_rollout_step(
                        rf_state,
                        step,
                        next_robot_state=robot_pred,
                        gripper_mode=gripper_mode,
                    )
                    if rf_state is not None
                    else None
                )
                output_state = self._replace_with_rf_pose_step(robot_pred, prev_output_state, rf_step)
                pred_steps.append(output_state)
                feedback_state = output_state if feedback_mode == "rigidformer_pose" else robot_pred
                x_state = feedback_state.unsqueeze(1)
                prev_output_state = output_state.detach()
                aux_steps.append({"rwm_aleatoric": aleatoric, "rwm_epistemic": epistemic})
        finally:
            self.robot.reset()
        return torch.stack(pred_steps, dim=1), aux_steps

    def _forward_delan(
        self,
        *,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        rf_state: dict[str, Any] | None,
        feedback_mode: str,
        gripper_mode: str,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if not self.layout.has_joint_vel:
            raise ValueError("Hybrid DeLaN robot backend requires state_prediction_mode='full'.")
        history_window = history_states
        state = history_states[:, -1]
        prev_output_state = state
        pred_steps: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []

        for step in range(future_torques.shape[1]):
            torque = future_torques[:, step]
            q = state[:, self.layout.robot_q_slice]
            dq = state[:, self.layout.robot_dq_slice]
            x = torch.cat([history_window[:, :-1].flatten(start_dim=1), q], dim=-1)
            robot_out = self.robot(x=x, dq=dq, torque=torque)
            base_state = state.clone()
            base_state[:, self.layout.robot_q_slice] = robot_out.next_q
            base_state[:, self.layout.robot_dq_slice] = robot_out.next_dq
            rf_step = (
                self._rigidformer_rollout_step(
                    rf_state,
                    step,
                    next_robot_state=base_state,
                    gripper_mode=gripper_mode,
                )
                if rf_state is not None
                else None
            )
            output_state = self._replace_with_rf_pose_step(base_state, prev_output_state, rf_step)
            pred_steps.append(output_state)
            feedback_state = output_state if feedback_mode == "rigidformer_pose" else base_state
            history_window = torch.cat([history_window[:, 1:], feedback_state[:, None]], dim=1)
            state = feedback_state
            prev_output_state = output_state.detach()
            aux_steps.append(robot_out.aux)
        return torch.stack(pred_steps, dim=1), aux_steps

    def _replace_with_rf_pose(
        self,
        base_state: torch.Tensor,
        previous_state: torch.Tensor,
        rf_pose: dict[str, torch.Tensor] | None,
        step: int,
    ) -> torch.Tensor:
        if rf_pose is None:
            return base_state
        return self._replace_with_rf_pose_step(
            base_state,
            previous_state,
            {"pos": rf_pose["pos"][:, step], "quat": rf_pose["quat"][:, step]},
        )

    def _replace_with_rf_pose_step(
        self,
        base_state: torch.Tensor,
        previous_state: torch.Tensor,
        rf_pose: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if rf_pose is None:
            return base_state
        out = base_state.clone()
        pos = rf_pose["pos"].detach()
        quat = normalize_quat(rf_pose["quat"].detach())
        out[:, self.layout.object_pos_slice] = pos
        out[:, self.layout.object_quat_slice] = quat
        if self.layout.has_object_lin_vel:
            prev_pos = previous_state[:, self.layout.object_pos_slice]
            out[:, self.layout.object_lin_vel_slice] = (pos - prev_pos) / self.dt
        if self.layout.has_object_ang_vel:
            prev_quat = normalize_quat(previous_state[:, self.layout.object_quat_slice])
            out[:, self.layout.object_ang_vel_slice] = angular_velocity_from_quat_delta(prev_quat, quat, self.dt)
        return out


def pointcloud_tensors(batch: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(name: str) -> torch.Tensor:
        value = batch[name]
        return value.to(device, non_blocking=True) if torch.is_tensor(value) else torch.as_tensor(value, device=device)

    object_pos_rollout = tensor("pc_object_pos_rollout")
    object_lens = tensor("pc_object_lens").long()
    if object_lens.ndim == 0:
        object_lens = object_lens.expand(object_pos_rollout.shape[0])
    out = {
        "delta_times": tensor("pc_delta_times").float(),
        "vertex_properties": tensor("pc_vertex_properties").float(),
        "object_pos_prev": tensor("pc_object_pos_prev").float(),
        "object_pos": tensor("pc_object_pos").float(),
        "object_pos_next": tensor("pc_object_pos_next").float(),
        "object_pos_rollout": object_pos_rollout.float(),
        "object_first_frame_pos": tensor("pc_object_first_frame_pos").float(),
        "object_point_lens": tensor("pc_object_point_lens").long(),
        "object_lens": object_lens,
        "loss_object_mask": tensor("pc_loss_object_mask").bool(),
        "first_object_pos_w": tensor("pc_first_object_pos_w").float(),
        "first_object_quat": tensor("pc_first_object_quat").float(),
        "env_origin": tensor("pc_env_origin").float(),
    }
    if "pc_gripper_part_ids" in batch:
        out["gripper_part_ids"] = tensor("pc_gripper_part_ids").long()
    if "pc_first_robot_q_abs" in batch:
        out["first_robot_q_abs"] = tensor("pc_first_robot_q_abs").float()
    if "pc_first_robot_q_obs" in batch:
        out["first_robot_q_obs"] = tensor("pc_first_robot_q_obs").float()
    return out


def rigidformer_loss(model: HybridRigidFormerWMDynamics, batch: dict[str, Any], device: torch.device):
    pc = pointcloud_tensors(batch, device=device)
    return model.rigidformer(
        delta_times=pc["delta_times"],
        vertex_properties=pc["vertex_properties"],
        object_pos=pc["object_pos"],
        object_pos_prev=pc["object_pos_prev"],
        object_pos_next=pc["object_pos_next"],
        object_first_frame_pos=pc["object_first_frame_pos"],
        object_lens=pc["object_lens"],
        object_point_lens=pc["object_point_lens"],
        loss_object_mask=pc["loss_object_mask"],
    )


@torch.no_grad()
def rigidformer_eval_metrics(
    model: HybridRigidFormerWMDynamics,
    batch: dict[str, Any],
    future_states: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pc = pointcloud_tensors(batch, device=device)
    pred = model.rigidformer_pose_rollout(batch, 1, device=device)
    if pred is None:
        return {}
    pred_points = pred["points"][:, 0]
    target_points = pc["object_pos_next"]
    point_rmse = masked_point_rmse(
        pred_points,
        target_points,
        pc["object_point_lens"],
        pc["object_lens"],
        pc["loss_object_mask"],
    )
    layout = model.layout
    gt_pos = future_states[:, 0, layout.object_pos_slice]
    gt_quat = normalize_quat(future_states[:, 0, layout.object_quat_slice])
    pred_pos = pred["pos"][:, 0]
    pred_quat = normalize_quat(pred["quat"][:, 0])
    pos_rmse = torch.linalg.norm(pred_pos - gt_pos, dim=-1).square().mean().sqrt()
    orient_rmse = quat_angle_error(pred_quat, gt_quat).square().mean().sqrt()
    return {
        "rigidformer_point_rmse": point_rmse,
        "rigidformer_pose_position_rmse": pos_rmse,
        "rigidformer_pose_orientation_rmse_rad": orient_rmse,
        "rigidformer_pose_orientation_rmse_deg": orient_rmse * (180.0 / torch.pi),
    }


def pose_from_cube_points(
    *,
    first_frame_points: torch.Tensor,
    target_points: torch.Tensor,
    first_object_pos_w: torch.Tensor,
    first_object_quat: torch.Tensor,
    env_origin: torch.Tensor,
    subtract_env_origin: bool,
    point_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_rot = quat_to_rotation_matrix(first_object_quat)
    local_points = torch.einsum("bnd,bdc->bnc", first_frame_points - first_object_pos_w[:, None], first_rot)
    rotation, translation = estimate_row_rigid_transform(local_points, target_points, point_lens=point_lens)
    quat = matrix_to_quat(rotation)
    pos = translation - env_origin if subtract_env_origin else translation
    return pos, quat


def estimate_row_rigid_transform(
    reference_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    point_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate standard rotation R and translation t for target ~= reference @ R.T + t."""
    if reference_points.shape != target_points.shape:
        raise ValueError("reference_points and target_points must have matching shapes.")
    batch, points, _ = reference_points.shape
    if point_lens is None:
        weights = reference_points.new_ones((batch, points, 1))
    else:
        arange = torch.arange(points, device=reference_points.device)
        weights = (arange[None] < point_lens[:, None].clamp_min(0)).to(reference_points.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    ref_center = (reference_points * weights).sum(dim=1, keepdim=True) / denom
    tgt_center = (target_points * weights).sum(dim=1, keepdim=True) / denom
    ref_centered = (reference_points - ref_center) * weights
    tgt_centered = (target_points - tgt_center) * weights
    covariance = torch.einsum("bni,bnj->bij", ref_centered, tgt_centered)
    u, _s, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    det = torch.linalg.det(rotation)
    if torch.any(det < 0.0):
        vh_fixed = vh.clone()
        vh_fixed[det < 0.0, -1] *= -1.0
        rotation = vh_fixed.transpose(-1, -2) @ u.transpose(-1, -2)
    translation = tgt_center.squeeze(1) - torch.einsum(
        "bi,bji->bj",
        ref_center.squeeze(1),
        rotation,
    )
    return rotation, translation


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                    1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    qw = q_abs[..., 0]
    qx = torch.copysign(q_abs[..., 1], m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(q_abs[..., 2], m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(q_abs[..., 3], m[..., 1, 0] - m[..., 0, 1])
    return normalize_quat(torch.stack([qw, qx, qy, qz], dim=-1))


def angular_velocity_from_quat_delta(prev_quat: torch.Tensor, next_quat: torch.Tensor, dt: float) -> torch.Tensor:
    delta = normalize_quat(quat_mul(quat_conjugate(prev_quat), next_quat))
    delta = torch.where(delta[..., :1] < 0.0, -delta, delta)
    xyz = delta[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, delta[..., :1].clamp_min(1.0e-8))
    axis = xyz / sin_half.clamp_min(1.0e-8)
    return axis * angle / float(dt)


def quat_angle_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = normalize_quat(pred)
    target = normalize_quat(target)
    dot = torch.sum(pred * target, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def masked_point_rmse(
    pred_next: torch.Tensor,
    gt_next: torch.Tensor,
    point_lens: torch.Tensor,
    object_lens: torch.Tensor,
    loss_object_mask: torch.Tensor,
) -> torch.Tensor:
    batch, max_objects = pred_next.shape[:2]
    sq_error = pred_next.new_zeros(())
    count = pred_next.new_zeros(())
    for batch_index in range(batch):
        num_objects = int(object_lens[batch_index].detach().cpu().item())
        for object_index in range(min(num_objects, max_objects)):
            if not bool(loss_object_mask[batch_index, object_index].detach().cpu().item()):
                continue
            num_points = int(point_lens[batch_index, object_index].detach().cpu().item())
            if num_points <= 0:
                continue
            diff = pred_next[batch_index, object_index, :num_points] - gt_next[batch_index, object_index, :num_points]
            sq_error = sq_error + diff.square().sum()
            count = count + diff.numel()
    return torch.sqrt(sq_error / count.clamp_min(1.0))
