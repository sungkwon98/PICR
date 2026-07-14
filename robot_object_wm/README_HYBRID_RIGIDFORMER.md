# Hybrid Robot WM + RigidFormer Object Dynamics

This document covers the hybrid integration added to `robot_object_wm`.
It combines a robot state-space predictor with RigidFormer point-cloud dynamics:

- Franka robot dynamics: `RWMEnsemble` or DeLaN.
- Cube/object dynamics: RigidFormer trained on mesh/FK point-cloud trajectories.
- Object loss: teacher-forced one-step RigidFormer loss from GT gripper/cube point clouds.
- Rollout output: normal WM state vectors, so existing rollout metrics, animation, and IsaacLab rendering still work.

## Data

The hybrid model needs two aligned HDF5 files:

- State dataset:
  `rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5`
- Mesh/FK point-cloud dataset:
  `rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5`

Pairing is by `/data/episode_names` in the point-cloud HDF5 and `data/demo_*` names in the state HDF5. Point-cloud samples use `(t-1, t, t+1)` with history-2 / one-step RigidFormer prediction.

## Config Knobs

Hybrid fields are in `configs/train_config.yaml`:

```yaml
model_type: hybrid
pointcloud_file: ../../rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5
rigidformer_config: hybrid_rigidformer_config.yaml
hybrid_robot_model_type: rwm        # rwm or delan
hybrid_rollout_feedback_mode: robot_native  # robot_native or rigidformer_pose
hybrid_gripper_pointcloud_mode: gt  # gt or predicted_fk
robot_loss_weight: 1.0
rigidformer_loss_weight: 1.0
render_every: 0
eval_isaaclab_video: false
eval_isaaclab_config: isaaclab_visualization.yaml
```

RigidFormer architecture defaults are in `configs/hybrid_rigidformer_config.yaml`.

Rollout feedback modes:

- `robot_native`: robot autoregression uses the robot predictor's native predicted state; final output still uses RigidFormer cube pose where available.
- `rigidformer_pose`: RigidFormer-derived object pose/quat is inserted back into the next robot-state history.

Gripper point-cloud modes:

- `gt`: RigidFormer context gripper points are teacher-forced from the mesh/FK HDF5 trajectory.
- `predicted_fk`: the frame-0 gripper point cloud is converted back to canonical hand/finger points, then forward kinematics regenerates rollout gripper points from predicted Franka `q`. This uses the state-dataset q offset because `obs/joint_pos` is reset-relative while the mesh/FK file was generated from absolute `states/articulation/robot/joint_position`.

## Training

Run from the repository root:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.training.train \
  --config robot_object_wm/configs/train_config.yaml \
  --model_type hybrid \
  --hybrid_robot_model_type rwm \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5
```

Small smoke run:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.training.train \
  --config robot_object_wm/configs/train_config.yaml \
  --model_type hybrid \
  --hybrid_robot_model_type rwm \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --epochs 1 \
  --train_subsample_fraction 0.001 \
  --val_subsample_fraction 0.001 \
  --train_batch_fraction 0.01 \
  --wandb_mode disabled \
  --eval_after_train false
```

For DeLaN robot dynamics, use full state:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.training.train \
  --config robot_object_wm/configs/train_config.yaml \
  --model_type hybrid \
  --hybrid_robot_model_type delan \
  --state_prediction_mode full \
  --privileged_collision_observation 0 \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5
```

Render and upload evaluation media during training every N epochs:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.training.train \
  --config robot_object_wm/configs/train_config.yaml \
  --render_every 10 \
  --eval_video true \
  --eval_isaaclab_video true \
  --eval_isaaclab_config robot_object_wm/configs/isaaclab_visualization.yaml
```

Periodic renders use `last.pt` for the current epoch and write under `eval/epoch_XXXX/`.
Final post-training evaluation still uses `best.pt`.

## Evaluation

Aggregate rollout metrics:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.eval.rollout \
  --checkpoint /path/to/best.pt \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --hybrid_rollout_feedback_mode robot_native \
  --output_dir robot_object_wm/eval_outputs/hybrid_eval \
  --episode_plot \
  --prediction_metrics
```

Evaluate the alternate feedback mode:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.eval.rollout \
  --checkpoint /path/to/best.pt \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --hybrid_rollout_feedback_mode rigidformer_pose \
  --output_dir robot_object_wm/eval_outputs/hybrid_eval_rf_feedback \
  --episode_plot \
  --prediction_metrics
```

Evaluate with FK-generated gripper context:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.eval.rollout \
  --checkpoint /path/to/best.pt \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --hybrid_rollout_feedback_mode rigidformer_pose \
  --hybrid_gripper_pointcloud_mode predicted_fk \
  --output_dir robot_object_wm/eval_outputs/hybrid_eval_predicted_gripper \
  --episode_plot \
  --prediction_metrics
```

Hybrid eval adds RigidFormer metrics such as point RMSE, pose position RMSE, and pose orientation RMSE when point-cloud windows are available.

## Matplotlib Animation

Render the existing 3D matplotlib animation:

```bash
/home/sukchul/miniconda3/envs/rigidformer/bin/python -m robot_object_wm.eval.animation \
  --checkpoint /path/to/best.pt \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --hybrid_rollout_feedback_mode robot_native \
  --hybrid_gripper_pointcloud_mode gt \
  --episode_index 26 \
  --pred_horizon 10 \
  --output robot_object_wm/eval_outputs/hybrid_episode.mp4
```

## IsaacLab Visualization

Edit `configs/isaaclab_visualization.yaml` or pass overrides from the CLI.
The hybrid-specific fields are:

```yaml
pointcloud_file: ../../rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5
hybrid_rollout_feedback_mode: robot_native
hybrid_gripper_pointcloud_mode: gt
```

Run IsaacLab rendering:

```bash
./isaaclab.sh -p robot_object_wm/eval/isaaclab_visualization.py \
  --config robot_object_wm/configs/isaaclab_visualization.yaml \
  --checkpoint /path/to/best.pt \
  --dataset_file rigidformer/data/franka/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep_no_slip_trimmed_collision_augmented.hdf5 \
  --pointcloud_file rigidformer/data/franka/Franka_Lift_meshfk_posecube_1024p_11003ep.hdf5 \
  --hybrid_rollout_feedback_mode robot_native \
  --hybrid_gripper_pointcloud_mode gt \
  --output robot_object_wm/eval_outputs/isaaclab_hybrid_gt_vs_wm.mp4
```

Use `--hybrid_rollout_feedback_mode rigidformer_pose` to render the feedback-B rollout variant.
Use `--hybrid_gripper_pointcloud_mode predicted_fk` to render without teacher-forced rollout gripper point clouds.
