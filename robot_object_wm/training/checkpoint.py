from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from robot_object_wm.config import default_run_name, safe_wandb_artifact_name, to_plain_config


@dataclass
class TrainingRun:
    model_name: str
    run_name: str
    output_dir: str
    best_path: str
    last_path: str
    wandb_run: Any | None
    wandb_artifacts: bool

    @property
    def wandb_url(self) -> str | None:
        if self.wandb_run is None:
            return None
        return getattr(self.wandb_run, "url", None)

    def update_config(self, values: Mapping[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.config.update(to_plain_config(values), allow_val_change=True)
        except Exception as exc:  # pragma: no cover - W&B service dependent.
            print(f"[WARN] W&B config update failed: {exc}")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float] | None,
        best_val_loss: float,
        is_best: bool,
    ) -> None:
        if self.wandb_run is None:
            return
        payload: dict[str, float | int | bool] = {
            "epoch": epoch,
            "best_val_loss": float(best_val_loss),
            "is_best": bool(is_best),
        }
        payload.update({f"train/{key}": float(value) for key, value in train_metrics.items()})
        if val_metrics:
            payload.update({f"val/{key}": float(value) for key, value in val_metrics.items()})
        try:
            self.wandb_run.log(payload, step=epoch)
        except Exception as exc:  # pragma: no cover - W&B service dependent.
            print(f"[WARN] W&B metric logging failed at epoch {epoch}: {exc}")

    def log_best_artifact(self, epoch: int, val_loss: float) -> None:
        if self.wandb_run is None or not self.wandb_artifacts:
            return
        try:
            import wandb

            artifact = wandb.Artifact(
                name=safe_wandb_artifact_name(f"{self.model_name}-{self.run_name}-best"),
                type="model",
                metadata={
                    "architecture": self.model_name,
                    "run_name": self.run_name,
                    "epoch": int(epoch),
                    "val_loss": float(val_loss),
                },
            )
            artifact.add_file(self.best_path, name="best.pt")
            self.wandb_run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
        except Exception as exc:  # pragma: no cover - W&B service dependent.
            print(f"[WARN] W&B artifact logging failed at epoch {epoch}: {exc}")

    def log_evaluation(
        self,
        output_dir: str,
        *,
        metrics: Mapping[str, float] | None = None,
        media: Mapping[str, str] | None = None,
        step: int | None = None,
        fps: int = 25,
    ) -> None:
        if self.wandb_run is None:
            return
        try:
            import wandb

            payload: dict[str, Any] = {}
            if metrics:
                payload.update({f"eval/{key}": float(value) for key, value in metrics.items()})
            for key, path in (media or {}).items():
                if not os.path.isfile(path):
                    continue
                ext = Path(path).suffix.lower()
                if ext in (".png", ".jpg", ".jpeg"):
                    payload[f"eval/media/{key}"] = wandb.Image(path)
                elif ext in (".mp4", ".mov", ".m4v", ".gif"):
                    payload[f"eval/media/{key}"] = wandb.Video(path, fps=fps, format=ext.lstrip("."))
            if payload:
                self.wandb_run.log(payload, step=step)

            if self.wandb_artifacts and os.path.isdir(output_dir):
                artifact = wandb.Artifact(
                    name=safe_wandb_artifact_name(f"{self.model_name}-{self.run_name}-eval"),
                    type="evaluation",
                    metadata={
                        "architecture": self.model_name,
                        "run_name": self.run_name,
                        "output_dir": os.path.abspath(output_dir),
                    },
                )
                artifact.add_dir(output_dir)
                self.wandb_run.log_artifact(artifact, aliases=["latest"])
        except Exception as exc:  # pragma: no cover - W&B service dependent.
            print(f"[WARN] W&B eval logging failed: {exc}")

    def finish(self) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:  # pragma: no cover - W&B service dependent.
            print(f"[WARN] W&B finish failed: {exc}")


def start_training_run(
    cfg: Any,
    model_name: str,
    config: Mapping[str, Any],
) -> TrainingRun:
    run_name = cfg.run_name or default_run_name()
    run_output_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=bool(getattr(cfg, "resume_from", None)))
    best_path = os.path.join(run_output_dir, "best.pt")
    last_path = os.path.join(run_output_dir, "last.pt")

    wandb_run = None
    if cfg.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is required when --wandb-mode is not 'disabled'. "
                "Install wandb or pass --wandb-mode disabled."
            ) from exc

        wandb_run = wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity or None,
            name=cfg.wandb_name or f"{model_name}-{run_name}",
            mode=cfg.wandb_mode,
            dir=run_output_dir,
            config=to_plain_config(config),
            tags=["world_model", model_name],
            group=model_name,
            save_code=True,
        )

    return TrainingRun(
        model_name=model_name,
        run_name=run_name,
        output_dir=run_output_dir,
        best_path=best_path,
        last_path=last_path,
        wandb_run=wandb_run,
        wandb_artifacts=bool(cfg.wandb_artifacts),
    )


def checkpoint_and_log_epoch(
    run: TrainingRun,
    epoch: int,
    checkpoint: Mapping[str, Any],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float] | None,
    best_val_loss: float,
) -> float:
    has_validation = val_metrics is not None and "loss" in val_metrics
    val_loss = float(val_metrics["loss"]) if has_validation else None
    is_best = bool(has_validation and val_loss is not None and val_loss < best_val_loss)
    if is_best and val_loss is not None:
        best_val_loss = val_loss

    checkpoint_to_save = dict(checkpoint)
    checkpoint_to_save["best_val_loss"] = float(best_val_loss)
    torch.save(checkpoint_to_save, run.last_path)

    if is_best and val_loss is not None:
        torch.save(checkpoint_to_save, run.best_path)
        run.log_best_artifact(epoch=epoch, val_loss=val_loss)
    run.log_epoch(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_val_loss=best_val_loss,
        is_best=is_best,
    )
    return best_val_loss
