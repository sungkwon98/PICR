# Context DeLaN Robot-Object World Model

This folder implements a context-aware DeLaN-style world model for Franka/object interaction. It follows the idea from Context DeLaN: infer a latent context `z` from recent history, then condition the structured dynamics model on that context.

Compared with `../GT_dynamics`, this model does not feed the object physical context directly into the dynamics heads. Instead:

```text
history_states, history_torques -> context_encoder -> z
state_t, torque_t, z -> Context DeLaN dynamics -> state_{t+1}
```

## Files

- `dataset.py`: HDF5 rollout dataset with history states, history torques, future torques, future states, robot dynamics labels, object dynamics labels, and optional physical context labels.
- `models.py`: MLP/LSTM context encoders, context-conditioned DeLaN robot terms, residual torque head, object acceleration head, and training losses.
- `wm_train.py`: Training script with checkpointing and TensorBoard logging.
- `eval_utils.py`: Shared HDF5 loading, checkpoint loading, FK, rollout, and contact proxy utilities.
- `plot_robot_object_trajectory.py`: Single-episode trajectory plotter matching the robot-object GT dynamics plot format.
- `eval_mean_accumulated_error.py`: Dataset-wide accumulated gripper/object position error evaluator with total, pre-contact, and post-contact intervals.

## HDF5 Schema

The code expects the same robot-object dynamics HDF5 structure used by `../GT_dynamics`:

- `data/<episode>/obs/joint_pos`
- `data/<episode>/obs/joint_vel`
- `data/<episode>/robot_torques/applied_torque` or `computed_torque`
- `data/<episode>/states/rigid_object/object/root_pose`
- `data/<episode>/states/rigid_object/object/root_velocity`
- `data/<episode>/states/articulation/robot/joint_position`
- `data/<episode>/robot_dynamics/qdd`
- `data/<episode>/robot_dynamics/mass_matrix`
- `data/<episode>/robot_dynamics/inertial`
- `data/<episode>/robot_dynamics/coriolis`
- `data/<episode>/robot_dynamics/gravity`
- `data/<episode>/robot_dynamics/inverse_dynamics_tau`
- `data/<episode>/object_dynamics/root_lin_acc_w`
- `data/<episode>/object_dynamics/root_ang_acc_w`
- `data/<episode>/object_dynamics/external_force_est_w`
- `data/<episode>/object_dynamics/inertial_force_w`
- `data/<episode>/object_dynamics/gravity_force_w`
- `data/<episode>/object_dynamics/mass`
- `data/<episode>/object_dynamics/inertia`
- `data/<episode>/object_dynamics/material_properties` when available

The physical context labels are loaded for inspection and future supervision, but the dynamics model uses the inferred latent `z`.

## Training

The trainer defaults to local `./datasets`. You can also pass a file explicitly:

```bash
python wm_train.py \
  --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --context_encoder lstm \
  --latent_dim 16 \
  --history_len 5 \
  --rollout_horizon 3
```

Use the MLP context encoder for a flattened-history ablation:

```bash
python wm_train.py \
  --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --context_encoder mlp \
  --latent_dim 16
```

Important arguments:

- `--context_encoder {lstm,mlp}`: choose the context encoder.
- `--latent_dim`: size of latent context `z`.
- `--lstm_layers`: number of recurrent layers for the LSTM encoder.
- `--history_len`: number of history states and torques used to infer `z`.
- `--rollout_horizon`: multi-step training horizon.
- `--torque_key {applied_torque,computed_torque}`: torque source.
- `--lambda_context_l2`: L2 regularization for latent context.

Checkpoints are saved to `./outputs_context_delan/<run_name>/best.pt` and `last.pt`.

## Evaluation

Single-episode trajectory plot:

```bash
python plot_robot_object_trajectory.py \
  --checkpoint ./outputs_context_delan/run_YYYYMMDD_HHMMSS/best.pt \
  --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --episode_index 0 \
  --target gripper
```

Dataset-wide accumulated error:

```bash
python eval_mean_accumulated_error.py \
  --checkpoint ./outputs_context_delan/run_YYYYMMDD_HHMMSS/best.pt \
  --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --output_csv ./eval_outputs/context_delan_accumulated_error.csv
```

The accumulated error report includes:

- total rollout interval
- before first contact proxy
- after first contact proxy

`accumulated_error` is `sum(error_t)` over evaluated timesteps. `area_error` is `accumulated_error * dt`.

## Relation to the Paper

The paper conditions a structured Lagrangian model on a latent context inferred from temporal interaction history. This implementation keeps that core structure:

- `z = context_encoder(history_states, history_torques)`
- `H(q, z) = L(q, z)L(q, z)^T + eps I`
- `g(q, z)` and `c(q, qdot, z)` are computed through a DeLaN derivative path
- `tau_eff = tau + tau_residual(state, tau, z)`

The object part extends the paper-style robot dynamics to the current IsaacLab task by learning object linear/angular acceleration from `(state, torque, z)`.

## Notes

- The LSTM encoder is the default because context is inferred from temporal history in the paper.
- The MLP encoder is included for comparison with the existing `Ver_CaDM` style.
- The plotter uses absolute simulator joint positions from `states/articulation/robot/joint_position` for real FK, and converts predicted relative joint states back to absolute joint angles before plotting.
