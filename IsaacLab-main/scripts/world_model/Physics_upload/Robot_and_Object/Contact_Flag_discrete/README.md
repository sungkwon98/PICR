# Robot + Object MCGDF World Model

Structured robot + object world model that augments the robot-only MCGDF
(`../../Robot_only/MCGDF`) with the fixes recommended for the **robot+object**
mismatches identified in Section 4.3 of
`scripts/world_model/Physics/delan_vs_physx_verification.tex`.

The forward dynamics solved per step is

```text
H(q) qdd = tau + r_theta(q, qdot, tau)
         - c(q, qdot) - g(q)
         - d * qdot - f * tanh(qdot / eps)          (damping omitted by default)
         - J_c(q)^T F_c(s, tau, xi)                 (NEW: contact wrench coupling)

m_o v_o_dot = m_o g + F_c_lin                       (NEW: Newton-Euler linear)
I_w(r_o) omega_o_dot + omega_o x (I_w(r_o) omega_o) = F_c_ang
                                                     (NEW: Newton-Euler angular)
I_w(r_o) = R(r_o) I_b R(r_o)^T                       (NEW: world-frame inertia)
r_o_{t+1} = r_o_t * exp((1/2) omega_o dt)            (NEW: quaternion exp-map)
```

Compared to `../GT_dynamics`, the changes are:

| # | Fix | Where |
|---|-----|-------|
| #1, #5, #6 | Single contact wrench `F_c ∈ R^6`. Robot reaction torque is `tau_contact = -J_c(q)^T F_c` via the analytic Franka Jacobian; object dynamics follow Newton-Euler with the world-frame inertia `I_w = R I_b R^T`. | `FrankaForwardKinematics`, `ContactWrenchHead`, `RobotObjectMCGDFStep.object_dynamics` in `models.py` |
| #4 | Quaternion integration uses the exponential map `q_{t+1} = q_t * exp(0.5 * omega * dt)`. | `quat_exp_integrate` in `models.py` |
| #7 | The redundant `F_ext` supervision is dropped; only `v_dot^o` and `omega_dot^o` are supervised. | `supervised_object_dynamics_loss` |
| #9 | Object world-frame position is shifted by `env_origin = initial_state/articulation/robot/root_pose[0]` so the model sees env-local coordinates. | `dataset.py::_env_origin` |
| #2 (Approach A) | Per-episode pre-contact baseline of the implicit-residual target is computed and subtracted from the per-step target, leaving the implicit-PD + numerical residual as the supervision for `r_theta` and isolating the contact contribution for `F_c`. | `dataset.py::_compute_residual_baseline`, `residual_supervised_loss` |

Everything else (DeLaN structured `H, c, g`, `r_theta`, damping/friction
handling with `omit_damping`, supervised dynamics losses) is the same as the
robot-only MCGDF.

## Expected HDF5 fields per episode

```text
obs/joint_pos
obs/joint_vel
obs/object_position
actions
robot_torques/applied_torque
robot_dynamics/qdd
robot_dynamics/mass_matrix
robot_dynamics/inertial
robot_dynamics/coriolis
robot_dynamics/gravity
robot_dynamics/inverse_dynamics_tau
robot_joint_params/joint_damping
robot_joint_params/joint_friction_coeff
robot_joint_params/joint_dynamic_friction_coeff   (preferred)
states/articulation/robot/root_pose
states/rigid_object/object/root_pose
states/rigid_object/object/root_velocity
object_dynamics/root_lin_acc_w
object_dynamics/root_ang_acc_w
object_dynamics/mass
object_dynamics/inertia
object_dynamics/material_properties
initial_state/articulation/robot/root_pose          (used for env_origin)
```

The dataset folder is `./datasets` by default; supply
`--dataset_file <path>` to override.

## Training

Default training (damping omitted to match the implicit-actuator convention,
direct residual supervision enabled):

```bash
python wm_train.py \
  --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep.hdf5 \
  --rollout_horizon 3
```

Useful flags:

- `--filter_pre_contact` — drop rollout windows whose horizon crosses contact
  onset; useful for a robot-only-style ablation.
- `--lambda_contact <w>` — L2 regularization on the contact wrench magnitude;
  default 1e-5.
- `--lambda_residual_supervised <w>` — direct supervision weight for the
  residual head (target = implicit-PD residual minus per-episode
  pre-contact baseline). Default 1e-2.
- `--no_subtract_env_origin` — keep object positions in the raw simulator
  world frame.

## Logged metrics

In addition to the GT-dynamics metrics, the trainer logs:

```text
robot_dyn_loss, object_dyn_loss
dfr_sup_loss, res_sup_loss
residual_reg, contact_reg
residual_abs_mean, contact_wrench_abs_mean
object_pos_mse, object_quat_mse, object_lin_vel_mse, object_ang_vel_mse
object_lin_acc_mse, object_ang_acc_mse
residual_sup_mse
```

## Open items

- The contact-point assumption inside `FrankaForwardKinematics` treats the
  gripper midpoint as the contact location.  For finger-by-finger contact
  modelling, the Jacobian should be split into two per-finger Jacobians and
  the contact wrench summed.
- The contact wrench head ignores any explicit gripping-plane prior; a future
  iteration could constrain the linear part of `F_c` to lie in the gripper
  closing direction or enforce a friction-cone constraint.
- Approach B of mismatch #2 (use the trained robot-only MCGDF as a frozen
  residual predictor) is not implemented yet.
