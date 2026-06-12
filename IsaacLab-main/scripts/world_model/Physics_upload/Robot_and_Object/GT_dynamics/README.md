# Robot and Object GT Dynamics

This folder trains a hybrid torque-driven world model for Franka lift episodes with
both robot and object dynamics labels.

The robot part keeps the DeLaN-style structured dynamics used by the robot-only
GT model:

```text
H(q), c(q, qdot), g(q), qdd
```

The object-aware part adds:

```text
tau_residual = f(robot_state, object_state, object_context, recorded_torque)
object_acc = f(robot_state, object_state, object_context, recorded_torque)
```

`tau_residual` is intended to capture contact/carrying effects not represented by
free-space robot dynamics labels. The object head predicts linear and angular
acceleration and rolls out object position, quaternion, linear velocity, and
angular velocity.

Expected HDF5 fields per episode:

```text
obs/joint_pos
obs/joint_vel
actions
robot_torques/applied_torque
robot_torques/computed_torque
robot_dynamics/qdd
robot_dynamics/mass_matrix
robot_dynamics/inertial
robot_dynamics/coriolis
robot_dynamics/gravity
robot_dynamics/inverse_dynamics_tau
states/rigid_object/object/root_pose
states/rigid_object/object/root_velocity
object_dynamics/root_lin_acc_w
object_dynamics/root_ang_acc_w
object_dynamics/mass
object_dynamics/inertia
object_dynamics/material_properties
object_dynamics/inertial_force_w
object_dynamics/gravity_force_w
object_dynamics/external_force_est_w
```

State layout:

```text
state = [
  robot_q(9),
  robot_qdot(9),
  object_position(3),
  object_quat_wxyz(4),
  object_linear_velocity(3),
  object_angular_velocity(3),
]

input = recorded robot torque(9)
context = [object_mass(1), object_inertia(9), material_properties(3)]
```

Example training command:

```bash
python wm_train.py \
  --dataset_file ../../../../reinforcement_learning/skrl/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --rollout_horizon 3 \
  --train_batch_fraction 0.1
```

Trajectory evaluation:

```bash
python plot_robot_object_trajectory.py \
  --checkpoint ./outputs_robot_object_gt_dynamics/<run>/best.pt \
  --dataset_file ../../../../reinforcement_learning/skrl/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --episode_index 0 \
  --rollout_steps 100
```

Dynamics-term evaluation:

```bash
python plot_gt_vs_predicted_dynamics_terms.py \
  --checkpoint ./outputs_robot_object_gt_dynamics/<run>/best.pt \
  --dataset_file ../../../../reinforcement_learning/skrl/datasets/Lift_RL_opt_robot_object_dynamics_10000ep.hdf5 \
  --episode_index 0 \
  --start_t 20 \
  --rollout_steps 50
```

Notes:

- The robot `M`, `c`, and `g` labels are still free-space PhysX robot dynamics.
- The residual target is `robot_dynamics/inverse_dynamics_tau - recorded_torque`.
- Object external force is supervised from `object_dynamics/external_force_est_w`.
- The default trainer includes contact windows because they are the point of this model.
