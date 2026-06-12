"""Cross-model prediction-error comparison for the three robot+object
world models in this folder.

For each of the three sibling models -- ``MCGDF``, ``No_object_Euler``,
and ``Context_DeLaN`` -- this script:

1. Loads the trained checkpoint via the model's own ``animate_episode``
   helpers.
2. Rolls the model out for ``--pred_horizon`` steps from every valid
   start frame of every episode in the *heavy* and *light* context
   datasets.
3. Bins gripper-tip and cube-center position errors by rollout horizon.

The output is a single 2x2 figure: rows = (heavy, light) datasets,
columns = (gripper, cube) errors.  Each panel overlays the three models
with a +/-1 std band so they can be compared directly.

Because the three sibling folders each define their own
``animate_episode.py`` / ``models.py`` modules with overlapping names
but different signatures, this script swaps ``sys.path`` and the
``sys.modules`` cache between model loads.  Each model is fully torn
down before the next is imported.

Usage::

    python compare_models_pred_error_heavy_light.py
    python compare_models_pred_error_heavy_light.py --pred_horizon 15 --max_episodes 5
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


ROBOT_AND_OBJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PHYSICS_JJ = "/home/sungkwon/IsaacLab-RE0409/IsaacLab-main/scripts/world_model/Physics_jj"

# Module names that may exist with the same name in each model folder.
_PER_FOLDER_MODULES = (
    "animate_episode",
    "models",
    "dataset",
    "eval_utils",
    "plot_pred_error_heavy_light",
)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Per-model checkpoints (relative paths are resolved against this
    # script's folder; absolute paths are used as-is).
    parser.add_argument(
        "--mcgdf_checkpoint", type=str,
        default="MCGDF/outputs_mcgdf/run_20260608_112215/best.pt",
        help="Path to the MCGDF checkpoint.",
    )
    parser.add_argument(
        "--no_object_euler_checkpoint", type=str,
        default="No_object_Euler/outputs_mcgdf/run_20260608_171124/best.pt",
        help="Path to the No-Object-Euler checkpoint.",
    )
    parser.add_argument(
        "--context_delan_checkpoint", type=str,
        default="Context_DeLaN/outputs_context_delan/run_20260514_092259/best.pt",
        help="Path to the Context DeLaN checkpoint.",
    )

    # Datasets.
    parser.add_argument(
        "--heavy_dataset", type=str,
        default=f"{_PHYSICS_JJ}/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_heavy_context_10ep.hdf5",
    )
    parser.add_argument(
        "--light_dataset", type=str,
        default=f"{_PHYSICS_JJ}/datasets/Lift_RL_opt_robot_object_dynamics_joint_params_light_context_10ep.hdf5",
    )

    # Common rollout knobs.
    parser.add_argument("--pred_horizon", type=int, default=10)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Cap episodes per dataset (0 = all).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--robot_dof", type=int, default=9)
    parser.add_argument("--tool_z_offset", type=float, default=0.1034)
    parser.add_argument(
        "--torque_key", type=str, default="applied_torque",
        choices=["applied_torque", "computed_torque"],
    )
    parser.add_argument(
        "--friction_key", type=str, default="joint_dynamic_friction_coeff",
        choices=["joint_dynamic_friction_coeff", "joint_friction_coeff"],
    )
    parser.add_argument(
        "--no_subtract_env_origin", dest="subtract_env_origin",
        action="store_false", default=True,
    )
    parser.add_argument("--deterministic_context", action="store_true", default=True)
    parser.add_argument("--sample_context", dest="deterministic_context",
                        action="store_false")

    # No_object_Euler specific.
    parser.add_argument(
        "--filter_steps", type=int, default=-1,
        help="No-Object-Euler closed-loop filter window.  -1 = use the "
             "value baked into the checkpoint; 0 disables; >0 overrides.",
    )

    # Output.
    parser.add_argument(
        "--output", type=str,
        default="./eval_outputs/cross_model_pred_error_heavy_light.png",
        help="Output PNG path for the comparison figure.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Module-loading shim
# ----------------------------------------------------------------------------

def _activate_folder(folder_name: str) -> str:
    """Make ``Robot_and_Object/<folder_name>`` the active import root.

    Returns the absolute folder path.  Caller MUST call ``_deactivate_folder``
    when done so the next model load starts from a clean cache.
    """
    folder = os.path.join(ROBOT_AND_OBJECT_DIR, folder_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Model folder not found: {folder}")
    # Drop stale imports from the previous folder.
    for name in _PER_FOLDER_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, folder)
    return folder


def _deactivate_folder(folder: str) -> None:
    try:
        sys.path.remove(folder)
    except ValueError:
        pass
    for name in _PER_FOLDER_MODULES:
        sys.modules.pop(name, None)


def _resolve_checkpoint(path_arg: str) -> str:
    """Resolve a checkpoint path that may be relative to this script."""
    if os.path.isabs(path_arg):
        return path_arg
    return os.path.abspath(os.path.join(ROBOT_AND_OBJECT_DIR, path_arg))


# ----------------------------------------------------------------------------
# Shared error-binning helper
# ----------------------------------------------------------------------------

def _bin_errors(
    pred_g: list, pred_c: list, gt_g: np.ndarray, gt_c: np.ndarray,
    T_gt: int, g_by_h: list[list[float]], c_by_h: list[list[float]],
) -> None:
    """Pool per-horizon errors from one episode's per-frame rollouts."""
    n = len(g_by_h)
    for t, (g, c) in enumerate(zip(pred_g, pred_c)):
        if g is None or c is None:
            continue
        steps = min(g.shape[0], c.shape[0], n)
        for h in range(steps):
            fidx = t + 1 + h
            if fidx >= T_gt:
                break
            eg = float(np.linalg.norm(g[h] - gt_g[fidx]))
            ec = float(np.linalg.norm(c[h] - gt_c[fidx]))
            if np.isfinite(eg):
                g_by_h[h].append(eg)
            if np.isfinite(ec):
                c_by_h[h].append(ec)


def _empty_bins(n: int) -> tuple[list[list[float]], list[list[float]]]:
    return [[] for _ in range(n)], [[] for _ in range(n)]


def _ep_args_mcgdf_like(args: argparse.Namespace, dataset_file: str, ep_name: str) -> argparse.Namespace:
    """Namespace expected by the MCGDF / No_object_Euler ``load_episode`` loaders."""
    return argparse.Namespace(
        dataset_file=dataset_file,
        episode_index=0,
        episode_name=ep_name,
        robot_dof=args.robot_dof,
        subtract_env_origin=args.subtract_env_origin,
        max_frames=0,
        torque_key=args.torque_key,
        friction_key=args.friction_key,
        tool_z_offset=args.tool_z_offset,
    )


# ----------------------------------------------------------------------------
# Per-model accumulators
# ----------------------------------------------------------------------------

def accumulate_mcgdf(args: argparse.Namespace, device: torch.device):
    """Run the MCGDF rollout on both datasets.  Returns
    ``(heavy_g, heavy_c, n_h, light_g, light_c, n_l, ckpt_path)``."""
    folder = _activate_folder("MCGDF")
    try:
        anim = importlib.import_module("animate_episode")
        models_mod = importlib.import_module("models")
        FrankaForwardKinematics = models_mod.FrankaForwardKinematics

        ckpt_path = _resolve_checkpoint(args.mcgdf_checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"MCGDF checkpoint not found: {ckpt_path}")
        model, cfg = anim.load_checkpoint_model(ckpt_path, device)
        history_len = int(cfg.get("history_len", 5))

        fk = FrankaForwardKinematics(
            robot_dof=args.robot_dof, tool_z_offset=args.tool_z_offset,
        ).to(device)
        fk.eval()

        def _accumulate(dataset_file: str):
            n = args.pred_horizon
            g_by_h, c_by_h = _empty_bins(n)
            if not os.path.isfile(dataset_file):
                print(f"[WARN] dataset not found, skipping for MCGDF: {dataset_file}")
                return g_by_h, c_by_h, 0
            names = anim.list_episode_names(dataset_file)
            if args.max_episodes > 0:
                names = names[: args.max_episodes]
            for ep_name in names:
                ep_args = _ep_args_mcgdf_like(args, dataset_file, ep_name)
                episode = anim.load_episode(ep_args)
                extras = anim.load_episode_extras(ep_args, episode["name"])
                gt_g = anim.compute_joint_positions(episode["joint_pos"], fk, device)[:, -1, :]
                gt_c = episode["object_pos"]
                T_gt = min(gt_g.shape[0], gt_c.shape[0])
                pred_g, pred_c = anim.precompute_pred_trajectories(
                    model=model, episode=episode, extras=extras,
                    history_len=history_len, pred_horizon=n, device=device,
                    deterministic_context=args.deterministic_context,
                    fk=fk, robot_dof=args.robot_dof,
                )
                _bin_errors(pred_g, pred_c, gt_g, gt_c, T_gt, g_by_h, c_by_h)
            return g_by_h, c_by_h, len(names)

        heavy_g, heavy_c, n_h = _accumulate(args.heavy_dataset)
        light_g, light_c, n_l = _accumulate(args.light_dataset)
        return heavy_g, heavy_c, n_h, light_g, light_c, n_l, ckpt_path
    finally:
        _deactivate_folder(folder)


def accumulate_no_object_euler(args: argparse.Namespace, device: torch.device):
    """Run the No-Object-Euler rollout.  Unpacks the 5-tuple return and
    threads ``--filter_steps`` through."""
    folder = _activate_folder("No_object_Euler")
    try:
        anim = importlib.import_module("animate_episode")
        models_mod = importlib.import_module("models")
        FrankaForwardKinematics = models_mod.FrankaForwardKinematics

        ckpt_path = _resolve_checkpoint(args.no_object_euler_checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No-Object-Euler checkpoint not found: {ckpt_path}")
        model, cfg = anim.load_checkpoint_model(ckpt_path, device)
        history_len = int(cfg.get("history_len", 5))

        if args.filter_steps < 0:
            eff_filter = int(getattr(model, "filter_steps", 0))
        else:
            eff_filter = int(args.filter_steps)

        fk = FrankaForwardKinematics(
            robot_dof=args.robot_dof, tool_z_offset=args.tool_z_offset,
        ).to(device)
        fk.eval()

        def _accumulate(dataset_file: str):
            n = args.pred_horizon
            g_by_h, c_by_h = _empty_bins(n)
            if not os.path.isfile(dataset_file):
                print(f"[WARN] dataset not found, skipping for No-Object-Euler: {dataset_file}")
                return g_by_h, c_by_h, 0
            names = anim.list_episode_names(dataset_file)
            if args.max_episodes > 0:
                names = names[: args.max_episodes]
            for ep_name in names:
                ep_args = _ep_args_mcgdf_like(args, dataset_file, ep_name)
                episode = anim.load_episode(ep_args)
                extras = anim.load_episode_extras(ep_args, episode["name"])
                gt_g = anim.compute_joint_positions(episode["joint_pos"], fk, device)[:, -1, :]
                gt_c = episode["object_pos"]
                T_gt = min(gt_g.shape[0], gt_c.shape[0])
                pred_g, pred_c, *_rest = anim.precompute_pred_trajectories(
                    model=model, episode=episode, extras=extras,
                    history_len=history_len, pred_horizon=n, device=device,
                    deterministic_context=args.deterministic_context,
                    fk=fk, robot_dof=args.robot_dof,
                    filter_steps=eff_filter,
                )
                _bin_errors(pred_g, pred_c, gt_g, gt_c, T_gt, g_by_h, c_by_h)
            return g_by_h, c_by_h, len(names)

        heavy_g, heavy_c, n_h = _accumulate(args.heavy_dataset)
        light_g, light_c, n_l = _accumulate(args.light_dataset)
        return (heavy_g, heavy_c, n_h, light_g, light_c, n_l, ckpt_path, eff_filter)
    finally:
        _deactivate_folder(folder)


def accumulate_context_delan(args: argparse.Namespace, device: torch.device):
    """Run the Context DeLaN rollout.  Different API: uses ``layout`` and
    a numpy FK chain rather than the PyTorch ``FrankaForwardKinematics``."""
    folder = _activate_folder("Context_DeLaN")
    try:
        anim = importlib.import_module("animate_episode")
        eval_utils = importlib.import_module("eval_utils")

        ckpt_path = _resolve_checkpoint(args.context_delan_checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Context DeLaN checkpoint not found: {ckpt_path}")
        model, cfg, layout, ckpt_path = eval_utils.load_checkpoint_model(ckpt_path, device)
        history_len = int(cfg["history_len"])
        torque_key = args.torque_key or str(cfg["torque_key"])
        robot_state_dim = layout.robot_state_dim

        def _accumulate(dataset_file: str):
            n = args.pred_horizon
            g_by_h, c_by_h = _empty_bins(n)
            if not os.path.isfile(dataset_file):
                print(f"[WARN] dataset not found, skipping for Context DeLaN: {dataset_file}")
                return g_by_h, c_by_h, 0
            names = eval_utils.episode_names(dataset_file)
            if args.max_episodes > 0:
                names = names[: args.max_episodes]
            for ep_name in names:
                states, torques, _phys, _rdyn, _odyn = eval_utils.load_episode_arrays(
                    dataset_file, ep_name, layout, torque_key,
                )
                absolute_q = eval_utils.load_episode_robot_joint_positions(
                    dataset_file, ep_name, layout.robot_dof,
                )
                T_min = min(states.shape[0], torques.shape[0], absolute_q.shape[0])
                states = states[:T_min]
                torques = torques[:T_min]
                absolute_q = absolute_q[:T_min]
                joints_3d = anim.compute_joint_positions(absolute_q, args.tool_z_offset)
                gt_g = joints_3d[:, -1, :]
                gt_c = states[:, robot_state_dim : robot_state_dim + 3]
                T_gt = min(gt_g.shape[0], gt_c.shape[0])
                pred_g, pred_c, *_rest = anim.precompute_pred_trajectories(
                    model=model, layout=layout,
                    states=states, torques=torques, joint_pos_abs=absolute_q,
                    pred_horizon=n, history_len=history_len,
                    device=device, tool_z_offset=args.tool_z_offset,
                )
                _bin_errors(pred_g, pred_c, gt_g, gt_c, T_gt, g_by_h, c_by_h)
            return g_by_h, c_by_h, len(names)

        heavy_g, heavy_c, n_h = _accumulate(args.heavy_dataset)
        light_g, light_c, n_l = _accumulate(args.light_dataset)
        return (
            heavy_g, heavy_c, n_h, light_g, light_c, n_l, ckpt_path,
            str(cfg.get("context_encoder", "?")),
            int(cfg.get("latent_dim", -1)),
        )
    finally:
        _deactivate_folder(folder)


# ----------------------------------------------------------------------------
# Statistics + plotting
# ----------------------------------------------------------------------------

def _mean_std(err_by_h: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(err_by_h)
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for h, vals in enumerate(err_by_h):
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            mean[h] = arr.mean()
            std[h] = arr.std()
            count[h] = arr.size
    return mean, std, count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("===== Cross-model prediction-error (heavy/light) =====")
    print(f"pred_horizon: {args.pred_horizon}   max_episodes: {args.max_episodes}")
    print(f"heavy_dataset: {args.heavy_dataset}")
    print(f"light_dataset: {args.light_dataset}\n")

    # Run each model in isolation.  Each call swaps sys.path / sys.modules
    # so the three folders' overlapping module names do not interfere.
    print("-- MCGDF --")
    mcgdf_data = accumulate_mcgdf(args, device)

    print("\n-- No-Object-Euler --")
    noe_data = accumulate_no_object_euler(args, device)

    print("\n-- Context DeLaN --")
    cd_data = accumulate_context_delan(args, device)

    # Bundle into a uniform dict keyed by model name.
    results = {
        "MCGDF": {
            "heavy_g": mcgdf_data[0], "heavy_c": mcgdf_data[1], "n_h": mcgdf_data[2],
            "light_g": mcgdf_data[3], "light_c": mcgdf_data[4], "n_l": mcgdf_data[5],
            "ckpt": mcgdf_data[6],
            "subtitle": "(MCGDF)",
        },
        "No-Object-Euler": {
            "heavy_g": noe_data[0], "heavy_c": noe_data[1], "n_h": noe_data[2],
            "light_g": noe_data[3], "light_c": noe_data[4], "n_l": noe_data[5],
            "ckpt": noe_data[6],
            "subtitle": f"(NOE, filter_steps={noe_data[7]})",
        },
        "Context DeLaN": {
            "heavy_g": cd_data[0], "heavy_c": cd_data[1], "n_h": cd_data[2],
            "light_g": cd_data[3], "light_c": cd_data[4], "n_l": cd_data[5],
            "ckpt": cd_data[6],
            "subtitle": f"(Context DeLaN, enc={cd_data[7]}, ld={cd_data[8]})",
        },
    }
    model_colors = {
        "MCGDF": "tab:blue",
        "No-Object-Euler": "tab:green",
        "Context DeLaN": "tab:red",
    }

    # ---- 2x2 figure: rows = (heavy, light), cols = (gripper, cube) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    h_axis = np.arange(1, args.pred_horizon + 1)
    dataset_labels = [("heavy", "heavy"), ("light", "light")]
    qty_labels = [("gripper", "Gripper-tip"), ("cube", "Cube-center")]

    for row_i, (ds_key, ds_pretty) in enumerate(dataset_labels):
        for col_i, (qt_key, qt_pretty) in enumerate(qty_labels):
            ax = axes[row_i, col_i]
            for model_name, color in model_colors.items():
                bucket = results[model_name]
                if ds_key == "heavy":
                    err_by_h = bucket["heavy_g"] if qt_key == "gripper" else bucket["heavy_c"]
                    n_ep = bucket["n_h"]
                else:
                    err_by_h = bucket["light_g"] if qt_key == "gripper" else bucket["light_c"]
                    n_ep = bucket["n_l"]
                mean, std, _count = _mean_std(err_by_h)
                ax.plot(
                    h_axis, mean * 1000.0, color=color, linewidth=2.0,
                    marker="o", markersize=4,
                    label=f"{model_name} (n={n_ep})",
                )
                lo = np.clip(mean - std, 0.0, None) * 1000.0
                hi = (mean + std) * 1000.0
                ax.fill_between(h_axis, lo, hi, color=color, alpha=0.12)
            ax.set_title(f"{ds_pretty} dataset — {qt_pretty} position error", fontsize=11)
            ax.grid(True, alpha=0.3)
            if row_i == 1:
                ax.set_xlabel("rollout horizon $h$ (steps ahead)")
            ax.set_ylabel("mean error [mm]")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Cross-model open-loop {args.pred_horizon}-step prediction error",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    # ---- Console summary -----------------------------------------------
    print("\n========== Pooled heavy+light, mean error [mm] ==========")
    print(
        f"{'horizon':>8}  |"
        f"  {'MCGDF g':>10}  {'MCGDF c':>10}"
        f"  |  {'NOE g':>10}  {'NOE c':>10}"
        f"  |  {'CtxD g':>10}  {'CtxD c':>10}"
    )
    pooled = {}
    for model_name, bucket in results.items():
        pooled_g = [a + b for a, b in zip(bucket["heavy_g"], bucket["light_g"])]
        pooled_c = [a + b for a, b in zip(bucket["heavy_c"], bucket["light_c"])]
        mg, _, _ = _mean_std(pooled_g)
        mc, _, _ = _mean_std(pooled_c)
        pooled[model_name] = (mg, mc)
    for h in range(args.pred_horizon):
        cells = []
        for model_name in ("MCGDF", "No-Object-Euler", "Context DeLaN"):
            mg, mc = pooled[model_name]
            gv = mg[h] * 1000.0 if np.isfinite(mg[h]) else float("nan")
            cv = mc[h] * 1000.0 if np.isfinite(mc[h]) else float("nan")
            cells.append(f"  {gv:>10.3f}  {cv:>10.3f}")
        print(f"{h + 1:>8}  |" + "  |".join(cells))

    print("\n---- Checkpoints used ----")
    for model_name, bucket in results.items():
        print(f"  {model_name}: {bucket['ckpt']}   {bucket['subtitle']}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
