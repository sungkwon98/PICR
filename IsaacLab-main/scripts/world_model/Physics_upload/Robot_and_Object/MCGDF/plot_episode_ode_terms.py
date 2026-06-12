"""
For one episode of the MCGDF dataset, plot the seven terms of the assumed
forward-dynamics ODE per joint, along the time axis.

The ODE is (all quantities at time t):

    tau_applied = M @ qddot + c + g + d * qdot + f * sign(qdot) + delta

where ``delta`` is the residual torque (the part of the recorded applied
torque that the free-space DeLaN terms + damping + friction do not
explain).  Equivalently the residual is computed as:

    delta = tau_applied - ( M @ qddot + c + g + d * qdot + f * sign(qdot) )

Seven curves per joint, all in Nm:

    1. tau_applied       (robot_torques/applied_torque)
    2. M_qddot           (robot_dynamics/inertial   = mass_matrix @ qdd)
    3. c                 (robot_dynamics/coriolis)
    4. g                 (robot_dynamics/gravity)
    5. d * qdot          (joint_damping[0] * joint_vel)
    6. f * sign(qdot)    (joint_dynamic_friction_coeff[0] * sign(joint_vel))
    7. delta             (residual; positive => recorded tau exceeds the sum)

For the current Franka Lift dataset, terms (5) is dominated by the
implicit-actuator K_d that is also inside tau_applied (see the caveats
section of robot_only_mcgdf_description.tex), and term (6) is identically
zero because the recorded friction coefficient is zero.

Usage:

    python plot_episode_ode_terms.py \\
        --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep.hdf5 \\
        --episode_index 0
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _estimate_object_velocity(object_pos: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference (with one-sided ends) speed estimate for object trajectory."""
    vel = np.zeros_like(object_pos, dtype=np.float64)
    if object_pos.shape[0] <= 1:
        return vel
    vel[0] = (object_pos[1] - object_pos[0]) / dt
    vel[-1] = (object_pos[-1] - object_pos[-2]) / dt
    if object_pos.shape[0] > 2:
        vel[1:-1] = (object_pos[2:] - object_pos[:-2]) / (2.0 * dt)
    return vel


def first_object_motion_timestep(
    object_pos: np.ndarray,
    dt: float,
    displacement_threshold: float,
    velocity_threshold: float,
    consecutive_steps: int,
    settle_steps: int = 5,
) -> int | None:
    """Inlined copy of ``dataset.first_object_motion_timestep`` so this plot script
    has no dependency on torch (which the project's ``dataset.py`` imports)."""
    if object_pos.shape[0] == 0:
        return None
    baseline_idx = min(max(0, settle_steps), object_pos.shape[0] - 1)
    disp = np.linalg.norm(object_pos - object_pos[baseline_idx : baseline_idx + 1], axis=-1)
    speed = np.linalg.norm(_estimate_object_velocity(object_pos, dt), axis=-1)
    moving = (disp > displacement_threshold) | (speed > velocity_threshold)
    moving[:baseline_idx] = False
    if consecutive_steps <= 1:
        idx = np.flatnonzero(moving)
        return int(idx[0]) if idx.size > 0 else None
    run = 0
    for i, flag in enumerate(moving):
        run = run + 1 if bool(flag) else 0
        if run >= consecutive_steps:
            return i - consecutive_steps + 1
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/Lift_RL_opt_robot_object_dynamics_joint_params_rand_context_11003ep.hdf5",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Specific episode name (e.g. 'demo_0'); overrides --episode_index.",
    )
    parser.add_argument(
        "--torque_key",
        type=str,
        default="applied_torque",
        choices=["applied_torque", "computed_torque"],
    )
    parser.add_argument(
        "--friction_key",
        type=str,
        default="joint_dynamic_friction_coeff",
        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"],
    )
    parser.add_argument(
        "--use_smooth_sign",
        action="store_true",
        help="Use tanh(qdot/friction_eps) instead of hard sign(qdot) for friction.",
    )
    parser.add_argument(
        "--friction_eps",
        type=float,
        default=1.0e-3,
        help="Velocity scale used only when --use_smooth_sign is set.",
    )
    parser.add_argument(
        "--omit_damping",
        action="store_true",
        help=(
            "Drop the explicit d*qdot term from the ODE because the recorded "
            "applied_torque already contains the implicit-actuator -K_d*qdot "
            "contribution (see the Caveats section of "
            "robot_only_mcgdf_description.tex). When set, the equation reduces "
            "to: -tau + M*qdd + c + g + f*sign(qdot) + delta = 0."
        ),
    )
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--object_displacement_threshold",
        type=float,
        default=0.005,
        help="Object displacement (m) above which contact is presumed (proxy detector).",
    )
    parser.add_argument(
        "--object_velocity_threshold",
        type=float,
        default=0.02,
        help="Object speed (m/s) above which contact is presumed (proxy detector).",
    )
    parser.add_argument(
        "--contact_consecutive_steps",
        type=int,
        default=3,
        help="Number of consecutive moving steps required before declaring contact.",
    )
    parser.add_argument(
        "--contact_settle_steps",
        type=int,
        default=5,
        help="Number of initial steps treated as settling (ignored by the detector).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <dataset_name>_ep<idx>_ode_terms.png in cwd.",
    )
    return parser.parse_args()


def select_episode_name(file: h5py.File, args: argparse.Namespace) -> str:
    ep_names = list(file["data"].keys())

    def _key(name: str) -> tuple[int, str]:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (int(parts[1]), name)
        return (10**9, name)

    ep_names.sort(key=_key)
    if args.episode_name is not None:
        if args.episode_name not in file["data"]:
            raise KeyError(f"Episode '{args.episode_name}' not found in dataset.")
        return args.episode_name
    if args.episode_index < 0 or args.episode_index >= len(ep_names):
        raise IndexError(f"--episode_index {args.episode_index} out of range [0, {len(ep_names)})")
    return ep_names[args.episode_index]


def load_episode(args: argparse.Namespace) -> dict:
    n = args.robot_dof
    if not os.path.isfile(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

    with h5py.File(args.dataset_file, "r") as file:
        ep_name = select_episode_name(file, args)
        ep = file["data"][ep_name]

        # Mandatory fields.
        qdot = np.asarray(ep["obs"]["joint_vel"], dtype=np.float64)[:, :n]
        qdd = np.asarray(ep["robot_dynamics"]["qdd"], dtype=np.float64)[:, :n]
        tau = np.asarray(ep["robot_torques"][args.torque_key], dtype=np.float64)[:, :n]
        # Use the recorder's already-computed M @ qdd ("inertial") to match exactly.
        inertial = np.asarray(ep["robot_dynamics"]["inertial"], dtype=np.float64)[:, :n]
        c = np.asarray(ep["robot_dynamics"]["coriolis"], dtype=np.float64)[:, :n]
        g = np.asarray(ep["robot_dynamics"]["gravity"], dtype=np.float64)[:, :n]
        # Object position used for the contact-onset proxy.
        object_pos = np.asarray(ep["obs"]["object_position"], dtype=np.float64)

        # Joint params: constant within episode; use row 0.
        jp = ep["robot_joint_params"]
        damping = np.asarray(jp["joint_damping"], dtype=np.float64)[0, :n]
        friction_field = args.friction_key
        if friction_field not in jp:
            friction_field = "joint_friction_coeff"
        friction = np.asarray(jp[friction_field], dtype=np.float64)[0, :n]

    # Truncate to common length.
    T = min(
        qdot.shape[0], qdd.shape[0], tau.shape[0],
        inertial.shape[0], c.shape[0], g.shape[0],
        object_pos.shape[0],
    )

    return {
        "ep_name": ep_name,
        "T": T,
        "qdot": qdot[:T],
        "qdd": qdd[:T],
        "tau": tau[:T],
        "M_qdd": inertial[:T],
        "c": c[:T],
        "g": g[:T],
        "object_pos": object_pos[:T],
        "damping": damping,
        "friction": friction,
        "friction_field_used": friction_field,
    }


def compute_terms(
    data: dict,
    use_smooth_sign: bool,
    friction_eps: float,
    omit_damping: bool,
) -> dict[str, np.ndarray]:
    if omit_damping:
        # The implicit-actuator -K_d*qdot contribution is already inside
        # data["tau"] (applied_torque), so we drop the explicit damping term.
        d_term = np.zeros_like(data["qdot"])
    else:
        d_term = data["damping"][None, :] * data["qdot"]
    if use_smooth_sign:
        f_term = data["friction"][None, :] * np.tanh(data["qdot"] / friction_eps)
    else:
        f_term = data["friction"][None, :] * np.sign(data["qdot"])
    delta = data["tau"] - (data["M_qdd"] + data["c"] + data["g"] + d_term + f_term)
    # Plot -tau on the same side as the other RHS terms so the seven curves
    # visually sum to zero at every timestep.
    return {
        "neg_tau": -data["tau"],
        "M_qdd": data["M_qdd"],
        "c": data["c"],
        "g": data["g"],
        "d_qdot": d_term,
        "f_signqdot": f_term,
        "delta": delta,
    }


def print_summary(data: dict, terms: dict[str, np.ndarray], omit_damping: bool) -> None:
    print(f"Episode: {data['ep_name']}   T = {data['T']} steps")
    print(f"  friction field used: {data['friction_field_used']}")
    print(f"  damping  d = {np.array2string(data['damping'], precision=3)}"
          + ("   [OMITTED from ODE — already inside applied_torque]" if omit_damping else ""))
    print(f"  friction f = {np.array2string(data['friction'], precision=3)}")
    print("  per-joint |term| RMS (Nm):")
    header = " " * 4 + "joint  " + "   ".join(f"{n:>10s}" for n in terms.keys())
    print(header)
    for j in range(data["qdot"].shape[1]):
        row = f"    j{j + 1:>2d}    " + "   ".join(
            f"{np.sqrt((v[:, j] ** 2).mean()):>10.3f}" for v in terms.values()
        )
        print(row)


def plot(data: dict, terms: dict[str, np.ndarray], contact_t: int | None, args: argparse.Namespace) -> None:
    n = args.robot_dof
    T = data["T"]
    t_axis = np.arange(T)  # x-axis is the timestep index, not seconds

    # Term display config: (key, label, color, linewidth, linestyle).
    # Plotted with -tau on the RHS so all seven curves sum to zero at every t.
    term_style = [
        ("neg_tau",     r"$-\tau_{\rm applied}$",               "tab:red",    1.6, "-"),
        ("M_qdd",       r"$M\,\ddot q$ (inertial)",            "tab:blue",   1.0, "-"),
        ("c",           r"$c$ (Coriolis)",                      "tab:green",  1.0, "-"),
        ("g",           r"$g$ (gravity)",                       "tab:purple", 1.0, "-"),
        ("d_qdot",      r"$d\cdot\dot q$",                      "tab:orange", 1.0, "-"),
        ("f_signqdot",  r"$f\cdot{\rm sign}(\dot q)$",          "tab:brown",  1.0, "-"),
        ("delta",       r"$\delta = \tau - (M\ddot q + c + g + d\dot q + f\,{\rm sign}\dot q)$",
                                                                "black",      1.4, "--"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    contact_label = (
        f"first object motion @ t={contact_t}"
        if contact_t is not None
        else "no object motion detected"
    )

    for j in range(n):
        ax = axes[j]
        for key, label, color, lw, ls in term_style:
            ax.plot(t_axis, terms[key][:, j], color=color, lw=lw, ls=ls, label=label)
        ax.set_title(f"joint {j + 1}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="grey", lw=0.5)
        if contact_t is not None:
            ax.axvline(contact_t, color="magenta", lw=1.2, ls=":", label=contact_label)
        if j >= 6:
            ax.set_xlabel("timestep")
        if j % 3 == 0:
            ax.set_ylabel("torque [Nm]")
        ax.set_xlim(t_axis[0], t_axis[-1])

    # Single legend at top.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.99))

    sign_repr = f"tanh(qdot/{args.friction_eps:.0e})" if args.use_smooth_sign else "sign(qdot)"
    contact_str = f"contact onset @ t={contact_t}" if contact_t is not None else "no contact onset detected"
    if args.omit_damping:
        zero_sum_form = (
            r"Zero-sum form: $-\tau_{\rm applied} + M\ddot q + c + g + f\,{\rm sign}(\dot q) + \delta = 0$"
            r"   (damping omitted: already inside $\tau_{\rm applied}$ via the implicit actuator)"
        )
    else:
        zero_sum_form = (
            r"Zero-sum form: $-\tau_{\rm applied} + M\ddot q + c + g + d\dot q + f\,{\rm sign}(\dot q) + \delta = 0$"
        )
    fig.suptitle(
        f"ODE terms per joint — episode '{data['ep_name']}' "
        f"from {os.path.basename(args.dataset_file)}\n"
        f"(T={T} steps, dt={args.dt}s, torque_key='{args.torque_key}', "
        f"friction = {data['friction_field_used']} * {sign_repr}, {contact_str})\n"
        + zero_sum_form,
        fontsize=11,
        y=0.965,
    )
    fig.subplots_adjust(top=0.88, hspace=0.30, wspace=0.20)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.dataset_file))[0]
        output = f"{base}_ep{args.episode_index}_ode_terms.png"
    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    data = load_episode(args)
    terms = compute_terms(data, args.use_smooth_sign, args.friction_eps, args.omit_damping)
    contact_t = first_object_motion_timestep(
        object_pos=data["object_pos"],
        dt=args.dt,
        displacement_threshold=args.object_displacement_threshold,
        velocity_threshold=args.object_velocity_threshold,
        consecutive_steps=args.contact_consecutive_steps,
        settle_steps=args.contact_settle_steps,
    )
    print_summary(data, terms, args.omit_damping)
    if contact_t is not None:
        print(f"\nDetected first object motion (contact onset proxy) at timestep t = {contact_t}.")
    else:
        print("\nNo object motion detected with the current thresholds; vertical contact line will be omitted.")
    plot(data, terms, contact_t, args)


if __name__ == "__main__":
    main()
