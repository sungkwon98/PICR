# Robot-Object World Model

This package now has one active world-model architecture:

```text
RobotDynamics + ObjDynamics = WMDynamics
DeLaN robot dynamics + object-state MLP = multi-step rollout model
```

The active package exposes a single model path and one trainer/evaluator for that path.

## Layout

```text
robot_object_wm/
  data/          HDF5 schema helpers and the WMDynamics rollout dataset
  eval/          checkpoint loading and open-loop rollout evaluation
  models/        DeLaN robot dynamics, object MLP dynamics, WMDynamics
  training/      single training CLI, losses, W&B/checkpoints
```

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
  --model_type=split \
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

`models/world_model.py` exposes:

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

`data/object_mlp_dataset.py` builds rollout windows with:

- `history_states`
- `history_torques`
- `future_torques`
- `future_states`
- `object_context` = mass + inertia + material

The dataset keeps `object_context` for logging/metadata compatibility, but the current pure object MLP uses history, current torque, and latent `z`.

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
