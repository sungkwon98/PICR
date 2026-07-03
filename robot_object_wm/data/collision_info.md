# Privileged Collision Info Augmentation

This document describes the HDF5 augmentation produced by
`robot_object_wm.data.augment_collision_info`.

The augmentation is intentionally limited to the geometry available for the
current robot-object dataset:

- ground/support plane
- object cube box
- Franka Panda left/right gripper finger boxes

It does not use full arm-link collision geometry.

## Run

Use the `torch` conda environment:

```bash
/home/sukchul/miniconda3/envs/torch/bin/python -m robot_object_wm.data.augment_collision_info \
  --dataset_dir robot_object_wm/dataset \
  --output_suffix _collision_augmented \
  --overwrite
```

This writes same-directory files such as:

```text
Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_10002ep_collision_augmented.hdf5
```

If `--output_suffix` is omitted, the script augments the input files in place.

## Geometry Configuration

Object box:

- Source pose: `data/<episode>/states/rigid_object/object/root_pose`
- Quaternion order: `[w, x, y, z]`
- Default size: `--object_size auto`
- Auto size is inferred from object mass/inertia as a solid cube.
- For the current datasets this gives side length about `0.048 m`.

Gripper finger boxes:

- Source joint state: `data/<episode>/states/articulation/robot/joint_position`
- Uses absolute joint positions, not `obs/joint_pos`.
- Hand frame is computed with `FrankaForwardKinematics`.
- Finger joint origin: local hand `z = 0.0584 m`
- Left finger motion: `+y` by joint `q[7]`
- Right finger motion: `-y` by joint `q[8]`
- Right finger collision frame includes `Rz(pi)`.

Default finger box comes from the Franka/Panda collision mesh:

```text
finger_box_size_xyz   ~= [0.020974, 0.026536, 0.053717] m
finger_box_center_xyz ~= [0.000008, 0.013135, 0.026990] m
```

Ground/support plane:

- Default: `--ground_z auto`
- Auto infers the plane from frame-0 object bottom.
- For the current datasets this is about `z = 0.028057 m`.
- Use `--ground_z 0.0` if you want the world `z=0` plane instead.

## Collision Pairs

Default pair order:

```text
0 object_ground
1 object_left_finger
2 object_right_finger
3 object_gripper
```

Available pair names:

```text
object_ground
object_left_finger
object_right_finger
object_gripper
left_finger_ground
right_finger_ground
gripper_ground
left_right_finger
```

Special pair selections:

- `--pairs default`: the 4 default pairs above
- `--pairs all`: all available pairs, including `left_right_finger`
- `--pairs object_ground object_gripper`: write only selected pairs

## HDF5 Output

For each episode, the script writes:

```text
data/<episode>/privileged_collision/
  pair_names
  collision
  distance
  signed_distance
```

Default shapes for the current dataset:

```text
pair_names       (4,)
collision        (250, 4) bool
distance         (250, 4) float32
signed_distance  (250, 4) float32
```

Dataset meanings:

- `pair_names`: string names matching the second axis of all pair tensors.
- `collision`: `True` when the pair is colliding or within `--collision_margin`.
- `distance`: nonnegative separation distance in meters; `0` when colliding.
- `signed_distance`: signed clearance in meters.

For ground pairs, `signed_distance` is exact analytic box-bottom clearance:

```text
min_box_corner_z - ground_z
```

For box-box pairs, `signed_distance` is the raw `python-fcl` distance:

- positive when separated
- `0` at contact
- negative when overlapping

## Optional Outputs

Nearest points:

```bash
--nearest_points
```

Adds:

```text
data/<episode>/privileged_collision/nearest_points
```

Shape:

```text
(T, P, 2, 3)
```

Meaning:

- `nearest_points[t, p, 0]`: closest point on the first geometry in pair `p`
- `nearest_points[t, p, 1]`: closest point on the second geometry in pair `p`

Caveats:

- Stored only for separated FCL box-box pairs.
- Ground pairs are analytic and store `NaN`.
- Overlapping box-box pairs store `NaN` because FCL nearest points are not
  reliable for penetration in this usage.
- The evaluation animation renderer marks finite `nearest_points` entries for
  active collision pairs. If this dataset is absent, or all entries are `NaN`,
  only collision flags/distances are visualized.

OBB transforms:

```bash
--store_obbs
```

Adds:

```text
data/<episode>/privileged_collision/obbs/
  names
  center
  rotation
  size
```

Shapes:

```text
names     (3,)
center    (T, 3, 3)
rotation  (T, 3, 3, 3)
size      (T, 3, 3)
```

OBB name order:

```text
0 object_box
1 left_finger
2 right_finger
```

These tensors are not stored in the currently augmented datasets because they
increase file size. The boxes can be reconstructed from source state and the
script configuration when needed.

## CLI Options

```text
--dataset_file PATH     HDF5 file to augment. Can be passed multiple times.
--dataset_dir DIR       Directory of .hdf5 files. Default: robot_object_wm/dataset.
                        Use --dataset_dir '' to disable directory discovery.
--output_suffix SUFFIX  Copy each input to <stem><suffix>.hdf5 and augment the copy.
                        Example: --output_suffix _collision_augmented.
--output_dir DIR        Directory for suffixed output files. Default: same directory as source.
--group_name NAME       Output group name. Default: privileged_collision.
--pairs ...             Pair selection. Default: default.
--robot_dof N           Robot DOF used by FK. Default: 9.
--object_size VALUE     auto or cube side length in meters. Default: auto.
--ground_z VALUE        auto or fixed support-plane z value. Default: auto.
--finger_mesh PATH      Finger collision STL. Default: auto.
--device DEVICE         Torch device for FK. Default: cpu.
--collision_margin M    Treat distances <= M as collision. Default: 0.0.
--nearest_points        Store nearest_points for separated FCL box-box pairs.
--store_obbs            Store per-frame object/finger OBB transforms.
--overwrite             Replace an existing output group.
--compression TYPE      lzf, gzip, or none. Default: lzf.
--episode_name NAME     Augment only one episode.
--max_episodes N        Debug limit per file. 0 means all episodes.
--dry_run               Print planned writes without modifying files.
```

## Example Reads

```python
import h5py

path = "robot_object_wm/dataset/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep.hdf5"

with h5py.File(path, "r") as f:
    g = f["data/demo_0/privileged_collision"]
    pair_names = [x.decode() if isinstance(x, bytes) else str(x) for x in g["pair_names"][()]]
    collision = g["collision"][:]          # (T, P)
    distance = g["distance"][:]            # (T, P)
    signed_distance = g["signed_distance"][:]  # (T, P)

object_gripper_idx = pair_names.index("object_gripper")
object_gripper_collision = collision[:, object_gripper_idx]
object_gripper_distance = distance[:, object_gripper_idx]
```

## Training Window Alignment

If a rollout sample uses current index `t` and horizon `h`, align future
collision labels with future states:

```python
future_collision = collision[t + 1 : t + h + 1]
future_distance = distance[t + 1 : t + h + 1]
future_signed_distance = signed_distance[t + 1 : t + h + 1]
```

This matches the existing `future_states` convention in `rollout_dataset.py`.

## Training Observation Modes

Training can append privileged collision features to every state vector with:

```yaml
privileged_collision_observation: 0
privileged_collision_group: privileged_collision
privileged_collision_pairs: object_ground,object_left_finger,object_right_finger,object_gripper
privileged_collision_loss_weight: 1.0
```

Mode meanings:

```text
0 no privileged collision observation
1 collision flag
2 signed distance
3 signed distance + nearest points
```

With the default 4 pairs, the per-timestep state dimension changes as:

```text
mode 0: 31 + 0  = 31
mode 1: 31 + 4  = 35
mode 2: 31 + 4  = 35
mode 3: 31 + 28 = 59
```

Mode 3 uses:

```text
signed_distance: 4 dims
nearest_points: 4 pairs * 2 points * xyz(3) = 24 dims
```

`nearest_points` NaNs are converted to zeros before training. If
`subtract_env_origin` is true, finite nearest points are shifted into the same
relative coordinate frame as object position.

Example training smoke test:

```bash
/home/sukchul/miniconda3/envs/torch/bin/python -m robot_object_wm.training.train \
  --dataset_file robot_object_wm/dataset/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep_collision_augmented.hdf5 \
  --model_type MLP \
  --privileged_collision_observation 2 \
  --epochs 1 \
  --train_subsample_fraction 0.1 \
  --val_subsample_fraction 0.1 \
  --train_batch_fraction 0.05 \
  --eval_after_train false \
  --wandb_mode disabled
```
