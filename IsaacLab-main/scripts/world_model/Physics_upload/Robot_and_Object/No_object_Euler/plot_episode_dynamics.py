"""
Plot per-joint dynamics quantities for a selected episode of the MCGDF dataset.

For each of the 9 robot joints, plots the following quantities as functions
of time within the episode:

    q          joint position           obs/joint_pos
    qdot       joint velocity           obs/joint_vel
    qddot      joint acceleration       robot_dynamics/qdd          (PhysX FD)
    tau        recorded applied torque  robot_torques/applied_torque
    delta      residual torque:
                 delta_t = tau_t
                         - ( M_t @ qddot_t + c_t + g_t
                             + d * qdot_t
                             + f * sign(qdot_t) )
    M_diag     diagonal of mass matrix  robot_dynamics/mass_matrix[..., i, i]
    c          Coriolis term            robot_dynamics/coriolis
    g          gravity term             robot_dynamics/gravity
    d          joint damping coeff      robot_joint_params/joint_damping
    f          joint friction coeff     robot_joint_params/joint_dynamic_friction_coeff
                                        (or joint_friction_coeff as fallback)

The convention used for ``delta`` matches the user's formulation:

    M * qddot = tau - delta - c - g - d * qdot - f * sign(qdot)

so a positive ``delta`` means the recorded torque is *larger* than what the
ODE without delta would predict for the recorded ``qddot``.

Usage:
    python plot_episode_dynamics.py \
        --dataset_file ./datasets/Lift_RL_opt_robot_object_dynamics_joint_params_10000ep.hdf5 \
        --episode_index 0
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/Lift_RL_opt_robot_object_dynamics_joint_params_10000ep.hdf5",
    )
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Specific episode name (e.g. 'demo_0'); overrides --episode_index when given.",
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
        "--friction_eps",
        type=float,
        default=1.0e-3,
        help="Velocity scale (rad/s) for the tanh smoothing of the friction term.",
    )
    parser.add_argument(
        "--use_sign",
        action="store_true",
        help="Use hard sign(qdot) instead of tanh(qdot / friction_eps).",
    )
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <dataset_name>_ep<idx>_dynamics.png in the cwd.",
    )
    return parser.parse_args()


def select_episode_name(file: h5py.File, args: argparse.Namespace) -> str:
    ep_names = list(file["data"].keys())
    # Order by trailing integer when possible so --episode_index 0 picks demo_0.
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
        raise IndexError(
            f"--episode_index {args.episode_index} out of range [0, {len(ep_names)})"
        )
    return ep_names[args.episode_index]


def load_episode(args: argparse.Namespace) -> dict[str, np.ndarray]:
    n = args.robot_dof
    if not os.path.isfile(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

    with h5py.File(args.dataset_file, "r") as file:
        ep_name = select_episode_name(file, args)
        ep = file["data"][ep_name]

        # Mandatory fields.
        q = np.asarray(ep["obs"]["joint_pos"], dtype=np.float64)[:, :n]
        qdot = np.asarray(ep["obs"]["joint_vel"], dtype=np.float64)[:, :n]
        qdd = np.asarray(ep["robot_dynamics"]["qdd"], dtype=np.float64)[:, :n]
        tau = np.asarray(ep["robot_torques"][args.torque_key], dtype=np.float64)[:, :n]
        mass = np.asarray(ep["robot_dynamics"]["mass_matrix"], dtype=np.float64)[:, :n, :n]
        coriolis = np.asarray(ep["robot_dynamics"]["coriolis"], dtype=np.float64)[:, :n]
        gravity = np.asarray(ep["robot_dynamics"]["gravity"], dtype=np.float64)[:, :n]

        # Joint params: per-step in HDF5 but constant within an episode.
        jp = ep["robot_joint_params"]
        damping = np.asarray(jp["joint_damping"], dtype=np.float64)[0, :n]
        friction_key = args.friction_key
        if friction_key not in jp:
            friction_key = "joint_friction_coeff"
        if friction_key not in jp:
            raise KeyError(
                "Neither 'joint_dynamic_friction_coeff' nor 'joint_friction_coeff' "
                "found in robot_joint_params group."
            )
        friction = np.asarray(jp[friction_key], dtype=np.float64)[0, :n]

    # Truncate everything to a common length.
    T = min(q.shape[0], qdot.shape[0], qdd.shape[0], tau.shape[0],
            mass.shape[0], coriolis.shape[0], gravity.shape[0])

    return {
        "ep_name": ep_name,
        "T": T,
        "q": q[:T],
        "qdot": qdot[:T],
        "qdd": qdd[:T],
        "tau": tau[:T],
        "mass": mass[:T],
        "coriolis": coriolis[:T],
        "gravity": gravity[:T],
        "damping": damping,
        "friction": friction,
        "friction_key_used": friction_key,
    }


def compute_delta(
    data: dict[str, np.ndarray],
    friction_eps: float,
    use_sign: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns delta, M @ qdd, d*qdot, f*sign(qdot) (or tanh-smoothed)."""
    M_qdd = np.einsum("tij,tj->ti", data["mass"], data["qdd"])
    d_term = data["damping"][None, :] * data["qdot"]
    if use_sign:
        f_term = data["friction"][None, :] * np.sign(data["qdot"])
    else:
        f_term = data["friction"][None, :] * np.tanh(data["qdot"] / friction_eps)
    delta = data["tau"] - (M_qdd + data["coriolis"] + data["gravity"] + d_term + f_term)
    return delta, M_qdd, d_term, f_term


def print_summary(data: dict, delta: np.ndarray) -> None:
    print(f"Episode: {data['ep_name']}")
    print(f"  T = {data['T']} steps")
    print(f"  friction key used: {data['friction_key_used']}")
    print("  per-joint damping coefficients d:")
    print(f"    {np.array2string(data['damping'], precision=4)}")
    print("  per-joint friction coefficients f:")
    print(f"    {np.array2string(data['friction'], precision=4)}")
    print("  per-joint |delta| stats (Nm):")
    rms = np.sqrt((delta**2).mean(axis=0))
    mx = np.abs(delta).max(axis=0)
    print(f"    rms : {np.array2string(rms, precision=3)}")
    print(f"    max : {np.array2string(mx, precision=3)}")


def plot(data: dict, delta: np.ndarray, args: argparse.Namespace) -> None:
    n = args.robot_dof
    T = data["T"]
    t_axis = np.arange(T) * args.dt
    M_diag = np.diagonal(data["mass"], axis1=1, axis2=2)  # (T, n)

    cmap = plt.get_cmap("tab10")
    joint_colors = [cmap(i % 10) for i in range(n)]
    joint_labels = [f"j{i + 1}" for i in range(n)]

    panels: list[tuple[str, np.ndarray | None, str]] = [
        ("q  [rad]",                 data["q"],          "time series"),
        ("qdot  [rad/s]",            data["qdot"],       "time series"),
        ("qddot  [rad/s$^2$]",       data["qdd"],        "time series"),
        ("tau_applied  [Nm]",        data["tau"],        "time series"),
        ("delta  [Nm]",              delta,              "time series"),
        ("M_ii  [kg m$^2$]",         M_diag,             "time series"),
        ("c (Coriolis)  [Nm]",       data["coriolis"],   "time series"),
        ("g (Gravity)  [Nm]",        data["gravity"],    "time series"),
        ("d  [Nm s/rad]",            None,               "constant"),
        ("f  [Nm]",                  None,               "constant"),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for ax, (label, series, kind) in zip(axes, panels):
        if kind == "time series" and series is not None:
            for i in range(n):
                ax.plot(t_axis, series[:, i], color=joint_colors[i], label=joint_labels[i], lw=1.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_axis[0], t_axis[-1])

    # Constants: draw horizontal lines.
    for ax, vec in [(axes[8], data["damping"]), (axes[9], data["friction"])]:
        for i in range(n):
            ax.axhline(vec[i], color=joint_colors[i], lw=1.5, label=joint_labels[i])
        ax.margins(y=0.4)  # so the lines don't sit on the panel edge

    axes[-2].set_xlabel("time [s]")
    axes[-1].set_xlabel("time [s]")

    # One legend on the side.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="joint", fontsize=9, frameon=True)

    title = (
        f"MCGDF dynamics — episode '{data['ep_name']}' "
        f"from {os.path.basename(args.dataset_file)} "
        f"(T={T} steps, dt={args.dt}s, friction = "
        f"{'sign' if args.use_sign else f'tanh / eps={args.friction_eps:.0e}'})"
    )
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(right=0.92, hspace=0.25, top=0.95)

    output = args.output
    if output is None:
        base = os.path.splitext(os.path.basename(args.dataset_file))[0]
        output = f"{base}_ep{args.episode_index}_dynamics.png"
    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    data = load_episode(args)
    delta, _M_qdd, _d_term, _f_term = compute_delta(data, args.friction_eps, args.use_sign)
    print_summary(data, delta)
    plot(data, delta, args)


if __name__ == "__main__":
    main()
