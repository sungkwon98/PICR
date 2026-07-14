# Robot-Object World Model

This package contains robot-object world-model training and evaluation code.
The current default training config uses an RWM position-level state:

```text
model_type: rwm
history_len: 8
rollout_horizon: 5
action_type: torque
state_prediction_mode: position
```

Other wired model types are `split`, `whole`, and `MLP`, but the privileged
collision observation described below is currently supported for `MLP` and
`rwm` only. The velocity-free `state_prediction_mode: position` layout is
currently supported for `rwm` only; the DeLaN-based dynamics paths require
velocity components in the state.

## Layout

```text
robot_object_wm/
  data/          HDF5 schema helpers and the WMDynamics rollout dataset
  eval/          checkpoint loading and open-loop rollout evaluation
  models/        DeLaN robot dynamics, object MLP dynamics, WMDynamics
  training/      single training CLI, losses, W&B/checkpoints
```

## Policy-free Franka cube-drop dataset

`data/collect_franka_cube_drop.py` builds the inverse of the Lift task without
loading or training a policy. It takes reachable held-cube poses from an
existing Lift HDF5 file, initializes the Franka and cube at those poses, holds
the arm fixed, and opens the gripper. The source dataset's randomized object
mass/material and robot link masses are replayed by default. Because there is
no policy to compensate gravity, the arm uses Isaac Lab's high-PD Franka gains
during the release; they can be changed with `--arm-stiffness` and
`--arm-damping`.

The supplied `11003ep` dataset randomizes the cube reset pose, not the Franka
reset joints. The collector therefore selects the highest valid closed-gripper
state from each source rollout. This gives varied, physically reachable initial
gripper poses that actually held the cube in the source simulation.

Smoke test from the repository root (in the Isaac Lab Python environment):

```bash
python scripts/world_model/robot_object_wm/data/collect_franka_cube_drop.py \
  --num-episodes 2 \
  --num-envs 2 \
  --episode-steps 50 \
  --camera-width 64 \
  --camera-height 64 \
  --output-file /tmp/franka_drop_smoke.hdf5 \
  --overwrite \
  --headless
```

Full 11,003-episode collection with 128x128 front/left/right RGB:

```bash
DATASET_DIR=scripts/world_model/robot_object_wm/dataset
SOURCE="$DATASET_DIR/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep.hdf5"
OUTPUT="$DATASET_DIR/Franka_Lift_policy_free_cube_drop_multiview_11003ep.hdf5"
python scripts/world_model/robot_object_wm/data/collect_franka_cube_drop.py \
  --pose-source-hdf5 "$SOURCE" \
  --output-file "$OUTPUT" \
  --num-episodes 11003 \
  --num-envs 8 \
  --episode-steps 50 \
  --camera-width 128 \
  --camera-height 128 \
  --headless
```

This full configuration has 75.6 GiB of raw RGB payload before gzip. The
collector prints a storage estimate before starting, writes complete episodes
incrementally, and supports restart with the same arguments plus `--resume`.

Each `data/demo_N` contains the legacy `actions`, `obs`, `states`,
`robot_torques`, `robot_dynamics`, `robot_joint_params`, `object_dynamics`, and
`episode_physics_randomization` groups. New visual data is stored as:

```text
images/front       (T, H, W, 3) uint8
images/left        (T, H, W, 3) uint8
images/right       (T, H, W, 3) uint8
initial_state/images/{front,left,right}  (1, H, W, 3) uint8
camera_info/{front,left,right}/...
```

All time-series fields and RGB images are post-step and synchronized. The
`initial_state` group is the held pre-release state. The existing collision
augmentation is optional; if it is used, pass the real Lift support plane
instead of `auto` because frame zero is in the air:

```bash
python scripts/world_model/robot_object_wm/data/augment_collision_info.py \
  --dataset_file /path/to/Franka_Lift_policy_free_cube_drop_multiview_11003ep.hdf5 \
  --dataset_dir '' \
  --ground_z 0.0280570015 \
  --output_suffix _collision_augmented \
  --pairs object_ground object_left_finger object_right_finger object_gripper \
  --nearest_points
```

Use `privileged_collision_observation: 0` when training directly from an
unaugmented drop dataset.

## Train

Run from the repository root with `scripts/world_model` on `PYTHONPATH`:

```bash
PYTHONPATH=scripts/world_model python -m robot_object_wm.training.train \
  --dataset_file /path/to/dataset.hdf5
```

Or run from this package directory with the TrainConfig YAML:

```bash
python training/train.py --config configs/train_config.yaml
```

The default training file is `configs/train_config.yaml`. Current defaults:

```text
dataset_file: ../dataset/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_10002ep_collision_augmented.hdf5
model_type: rwm
history_len: 8
rollout_horizon: 5
state_prediction_mode: position
privileged_collision_observation: 0
batch_size: 256
epochs: 200
train_batch_fraction: 0.1
train_subsample_fraction: 0.2
eval_after_train: true
eval_video: true
```

Command-line flags override YAML values, so quick experiments can stay small:

```bash
python training/train.py --config configs/train_config.yaml \
  --epochs=50 \
  --wandb_mode=disabled
```

Every key in `configs/train_config.yaml` is editable from the command line.
Both underscore and hyphen spellings are accepted, and booleans accept explicit
values:

```bash
python training/train.py \
  --use_context_encoder=False \
  --delan_use_film=False \
  --model_type=MLP \
  --rollout_horizon=3 \
  --eval_video=False
```

Each run writes:

```text
<output_dir>/<run_name>/best.pt
<output_dir>/<run_name>/last.pt
<output_dir>/<run_name>/eval/summary.json
<output_dir>/<run_name>/eval/per_horizon_metrics.csv
<output_dir>/<run_name>/eval/curves/*.png
<output_dir>/<run_name>/eval/episode_trajectory.png
<output_dir>/<run_name>/eval/prediction_error_curves.png
<output_dir>/<run_name>/eval/episode_animation.mp4
```

The post-training evaluation is controlled by the `eval_*` keys in
`configs/train_config.yaml`. Disable expensive parts with:

```bash
python training/train.py --config configs/train_config.yaml \
  --eval_video=False \
  --eval_prediction_metrics=False
```

## Active Model

The default config uses the generic `RWMEnsemble` path in `models/rwm.py`.
The `MLP` path uses `WholeMLPWMDynamics` in `models/whole_dynamics.py`.
It predicts the next full state directly from:

```text
history_states + current torque -> next state
```

The split DeLaN/object-MLP path in `models/world_model.py` exposes:

- `RobotDynamics`: alias for `DeLaNRobotDynamics`
- `ObjDynamics`: alias for `ObjectStateMLPDynamics`
- `WMDynamics`: rollout module that steps a history-conditioned object MLP and the DeLaN robot dynamics
- `WMDynamicsConfig`
- `build_wm_dynamics`

Optional latent-context path:

```text
history_states + history_torques -> ContextEncoder -> z
z -> object history MLP
z -> DeLaN H/g/L heads when delan_use_film=true
```

This is controlled by `use_context_encoder`, `latent_dim`, and
`delan_use_film` in `configs/train_config.yaml`.

Robot equation:

```text
H(q) ddq = tau_cmd - c(q,dq) - g(q)
```

Object side:

- predicts the next 13D object state directly
- uses rolling `history_states`, `history_torques`, current torque, and optional latent `z`

## Data

`data/rollout_dataset.py` builds rollout windows with:

- `history_states`
- `history_torques`
- `future_torques`
- `future_states`
- `object_context` = mass + inertia + material

The full base state layout is 31D:

```text
joint_pos(9) + joint_vel(9) + object_pos(3) + object_quat(4)
+ object_lin_vel(3) + object_ang_vel(3)
```

For RWM training, `state_prediction_mode: position` uses a 16D state:

```text
joint_pos(9) + object_pos(3) + object_quat(4)
```

`object_context` is 13D:

```text
mass(1) + inertia(9) + material_properties(3)
```

The dataset keeps `object_context` for logging/metadata compatibility. The
default RWM model uses `history_states` and current torque directly.

### Privileged Collision Observation

Collision-augmented HDF5 files contain:

```text
data/<episode>/privileged_collision/
  pair_names
  collision
  distance
  signed_distance
  nearest_points        # optional, present when generated with --nearest_points
```

The default collision pairs are:

```text
object_ground
object_left_finger
object_right_finger
object_gripper
```

Training can append privileged collision information to each state with:

```yaml
state_prediction_mode: full
privileged_collision_observation: 2
privileged_collision_group: privileged_collision
privileged_collision_pairs: object_ground,object_left_finger,object_right_finger,object_gripper
privileged_collision_loss_weight: 1.0
```

Modes:

```text
0: no privileged collision info
1: collision flag
2: signed distance
3: signed distance + nearest_points
```

With the default 4 pairs and the full 31D base state, state size becomes:

```text
mode 0: 31
mode 1: 35
mode 2: 35
mode 3: 59
```

With `state_prediction_mode: position`, use `privileged_collision_observation: 0`
for the intended 16D position-level RWM state.

For RWM, `rwm_config.yaml` also has:

```yaml
include_privileged_collision_in_loss: false
```

When false, privileged collision dims remain appended to the RWM state/input,
but the direct RWM state regression loss masks those dims out. The model still
predicts those dims because the recurrent rollout state shape includes them.

Mode 3 is `signed_distance(4) + nearest_points(4 * 2 * 3 = 24)`, so it adds
28 dims. `nearest_points` NaNs are converted to zeros before training. When
`subtract_env_origin` is true, nearest points are shifted into the same relative
coordinate frame as object position.

Collision observation dims are also included in the rollout loss as
`privileged_collision_mse`, weighted by `privileged_collision_loss_weight`.
This matters because rollout feeds predicted states back into the next step.

Example using signed distance:

```bash
python training/train.py --config configs/train_config.yaml \
  --dataset_file ../dataset/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep_collision_augmented.hdf5 \
  --model_type=MLP \
  --privileged_collision_observation=2 \
  --wandb_mode=disabled
```

Example using signed distance plus nearest points:

```bash
python training/train.py --config configs/train_config.yaml \
  --dataset_file ../dataset/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep_collision_augmented.hdf5 \
  --model_type=MLP \
  --privileged_collision_observation=3 \
  --wandb_mode=disabled
```

Use `data/augment_collision_info.py` to create suffixed augmented files. From
this package directory:

```bash
python data/augment_collision_info.py \
  --dataset_file dataset/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_10002ep.hdf5 \
  --output_suffix _collision_augmented \
  --pairs object_ground object_left_finger object_right_finger object_gripper \
  --nearest_points \
  --overwrite
```

More details are in `data/collision_info.md`.

## Evaluate

Aggregate rollout metrics, per-horizon CSV, and curves:

```bash
PYTHONPATH=scripts/world_model python -m robot_object_wm.eval.rollout \
  --checkpoint /path/to/best.pt \
  --dataset_file /path/to/dataset.hdf5 \
  --output_dir ./eval_outputs/wm_dynamics \
  --episode_plot \
  --prediction_metrics
```

Render an MP4/GIF with optional predicted trajectory overlay:

```bash
PYTHONPATH=scripts/world_model python -m robot_object_wm.eval.animation \
  --checkpoint /path/to/best.pt \
  --dataset_file /path/to/dataset.hdf5 \
  --output ./eval_outputs/wm_dynamics_episode.mp4 \
  --pred_horizon 10
```

Collision overlay is enabled by default for animation/evaluation video. If the
provided dataset is not augmented, the renderer also checks the sibling
`*_collision_augmented.hdf5` file. Disable the overlay with:

```bash
PYTHONPATH=scripts/world_model python -m robot_object_wm.eval.animation \
  --dataset_file /path/to/dataset.hdf5 \
  --no_collision_info
```

## Isaac Lab Visualization

Render a headless Isaac Lab video with two Franka cube-lift scenes in one frame:
ground-truth dataset replay and open-loop world-model imagination from the same
rollout history/control sequence.

```bash
./isaaclab.sh -p scripts/world_model/robot_object_wm/eval/isaaclab_visualization.py \
  --config scripts/world_model/robot_object_wm/configs/isaaclab_visualization.yaml
```

The YAML selects `dataset_file`, `checkpoint`, `episode_index` or
`episode_name`, `start_t`, `output`, and `video_width`/`video_height`.
Episode object and robot domain parameters are applied when present, including
object mass/inertia/material, robot link masses, and joint
friction/damping/armature/stiffness.
