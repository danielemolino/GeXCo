"""Command-line entry point for GeXCo world-model pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import copy
import random
import numpy as np
from gexco.utils.checkpoint import atomic_save, split_training_records
from typing import Any

import torch
from gexco.config import default_config, load_config
from torch.utils.data import DataLoader

from gexco.pretraining.data import DrrWorldModelDataset
from gexco.pretraining.losses import predictive_loss
from gexco.pretraining.models import WorldModel, build_world_model
from gexco.pretraining.schedulers import CosineWDScheduler, WarmupCosineLRScheduler
from gexco.pretraining.sigreg import SIGReg
from gexco.utils.tensors import seed_everything


def train(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Train the JEPA world model from a parsed configuration.

    Args:
        config: Configuration matching ``configs/pretraining/default.yaml``.
        output_dir: Directory receiving ``last.pt``.

    Returns:
        Final checkpoint path.
    """
    training = config["training"]
    data_config = config["data"]
    if not data_config.get("manifest"):
        raise ValueError("Set data.manifest to a JSON/JSONL multi-angle DRR manifest.")
    seed_everything(int(training.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrrWorldModelDataset(
        data_config["manifest"],
        num_targets=int(training["num_targets"]),
        image_size=int(data_config["image_size"]),
        seed=int(training.get("seed", 42)),
        stats_path=data_config.get("stats_path"),
        normalization=str(data_config.get("normalization", "dataset_percentile_fixed")),
    )
    training_records, validation_records = split_training_records(
        dataset.records, fraction=float(data_config.get("validation_fraction", 0.1)), seed=int(training.get("seed", 42))
    )
    validation_dataset = copy.copy(dataset)
    dataset.records = training_records
    validation_dataset.records = validation_records
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(training["batch_size"]), shuffle=False, num_workers=0
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "split.json").write_text(
        json.dumps(
            {
                "train": [r["case_id"] for r in training_records],
                "validation": [r["case_id"] for r in validation_records],
            },
            indent=2,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 8)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Training split is smaller than batch_size")
    encoder_config = config["encoder"]
    predictor_config = config["predictor"]
    projector_config = config["projector"]
    model: WorldModel = build_world_model(
        input_size=int(encoder_config["input_size"]),
        patch_size=int(encoder_config["patch_size"]),
        embed_dim=int(encoder_config["embed_dim"]),
        depth=int(encoder_config["depth"]),
        num_heads=int(encoder_config["num_heads"]),
        predictor_depth=int(predictor_config["depth"]),
        predictor_num_heads=int(predictor_config["num_heads"]),
        projector_hidden_dim=int(projector_config["hidden_dim"]),
        projector_dim=int(projector_config["output_dim"]),
        use_projector=bool(projector_config["enabled"]),
        pose_scale_learnable=bool(encoder_config["pose_scale_learnable"]),
    ).to(device)
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": float(training["weight_decay_initial"]),
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "WD_exclude": True,
            },
        ],
        lr=float(training["lr_init"]),
        betas=tuple(training["betas"]),
    )
    total_steps = max(1, int(training["epochs"]) * len(loader))
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        warmup_steps=int(training["lr_warmup_epochs"]) * len(loader),
        start_lr=float(training["lr_init"]),
        peak_lr=float(training["lr_peak"]),
        final_lr=float(training["lr_final"]),
        total_steps=total_steps,
    )
    weight_decay_scheduler = CosineWDScheduler(
        optimizer,
        initial_weight_decay=float(training["weight_decay_initial"]),
        final_weight_decay=float(training["weight_decay_final"]),
        total_steps=total_steps,
    )
    regularizer = SIGReg(
        knots=int(config["sigreg"]["knots"]),
        num_proj=int(config["sigreg"]["projections"]),
    ).to(device)
    serialized_config = {
        "input_size": int(encoder_config["input_size"]),
        "in_channels": int(encoder_config["in_channels"]),
        "patch_size": int(encoder_config["patch_size"]),
        "embed_dim": int(encoder_config["embed_dim"]),
        "depth": int(encoder_config["depth"]),
        "num_heads": int(encoder_config["num_heads"]),
        "mlp_ratio": float(encoder_config["mlp_ratio"]),
        "pose_scale_learnable": bool(encoder_config["pose_scale_learnable"]),
        "no_encoder_pose": False,
        "predictor_embed_dim": int(predictor_config["embed_dim"]),
        "predictor_depth": int(predictor_config["depth"]),
        "predictor_num_heads": int(predictor_config["num_heads"]),
        "use_adaln": True,
        "use_action_mlp_embed": False,
        "action_mlp_ratio": 4.0,
        "action_mode": "delta",
        "num_targets": int(training["num_targets"]),
        "use_projector": bool(projector_config["enabled"]),
        "proj_hidden_dim": int(projector_config["hidden_dim"]),
        "proj_dim": int(projector_config["output_dim"]),
        "effective_proj_dim": int(projector_config["output_dim"]),
        "epochs": int(training["epochs"]),
        "batch_size_drr": int(training["batch_size"]),
        "start_lr": float(training["lr_init"]),
        "ref_lr": float(training["lr_peak"]),
        "final_lr": float(training["lr_final"]),
        "warmup_epochs": int(training["lr_warmup_epochs"]),
        "weight_decay": float(training["weight_decay_initial"]),
        "final_weight_decay": float(training["weight_decay_final"]),
        "lambda_sigreg": float(config["sigreg"]["lambda"]),
        "lambda_mim": 1.0,
        "sigreg_knots": int(config["sigreg"]["knots"]),
        "sigreg_num_proj": int(config["sigreg"]["projections"]),
        "seed": int(training.get("seed", 42)),
    }

    def save_state(path, epoch, best_loss):
        atomic_save(
            {
                "epoch": epoch,
                "encoder": model.encoder.state_dict(),
                "view_predictor": model.predictor.state_dict(),
                "enc_projector": None if model.enc_projector is None else model.enc_projector.state_dict(),
                "pred_projector": None if model.pred_projector is None else model.pred_projector.state_dict(),
                "sigreg": regularizer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": serialized_config,
                "public_config": config,
                "best_validation_loss": best_loss,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            },
            path,
        )

    best_loss = float("inf")
    start_epoch = 0
    if training.get("resume"):
        payload = torch.load(training["resume"], map_location="cpu", weights_only=False)
        for key, module in [
            ("encoder", model.encoder),
            ("view_predictor", model.predictor),
            ("enc_projector", model.enc_projector),
            ("pred_projector", model.pred_projector),
        ]:
            if module is not None:
                module.load_state_dict(payload[key], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_loss = float(payload["best_validation_loss"])
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    global_step = start_epoch * len(loader)
    model.train()
    for _epoch in range(start_epoch, int(training["epochs"])):
        epoch_total = 0.0
        model.train()
        for batch in loader:
            anchor = batch["anchor"].to(device)
            targets = batch["targets"].to(device)
            deltas = batch["angular_deltas"].to(device)
            batch_size, num_targets = targets.shape[:2]
            anchor_angles = batch["anchor_angle"].to(device).reshape(batch_size)
            target_angles = batch["target_angles"].to(device)
            all_images = torch.cat((anchor[:, None], targets), dim=1)
            all_angles = torch.cat((anchor_angles[:, None], target_angles), dim=1)
            all_tokens = model.encode(all_images.flatten(0, 1), all_angles.flatten()).reshape(
                batch_size, num_targets + 1, -1, int(projector_config["output_dim"])
            )
            anchor_tokens = all_tokens[:, 0]
            target_tokens = all_tokens[:, 1:]
            # One primary anchor repeated K times for the original SIGReg batch weighting.
            repeated_anchor = anchor_tokens[:, None].expand(-1, num_targets, -1, -1)
            predicted = model.predictor(repeated_anchor.flatten(0, 1), deltas.flatten())
            if model.pred_projector is not None:
                predicted = model.pred_projector(predicted)
            predicted = predicted.reshape_as(target_tokens)
            loss = predictive_loss(predicted, target_tokens)
            regularizer_embeddings = torch.cat(
                (
                    repeated_anchor.mean(dim=2).reshape(batch_size * num_targets, -1),
                    target_tokens.mean(dim=2).reshape(batch_size * num_targets, target_tokens.shape[-1]),
                ),
                dim=0,
            )
            loss = loss + float(config["sigreg"]["lambda"]) * regularizer(regularizer_embeddings)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite world-model loss")
            loss.backward()
            optimizer.step()
            scheduler.step(global_step)
            weight_decay_scheduler.step(global_step)
            global_step += 1
            epoch_total += float(loss.detach())
        # Select encoder on a patient-disjoint holdout of training data only.
        rng_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        seed_everything(42)
        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.inference_mode():
            for batch in validation_loader:
                anchor = batch["anchor"].to(device)
                targets = batch["targets"].to(device)
                batch_size, num_targets = targets.shape[:2]
                predicted = model(anchor, batch["angular_deltas"].to(device), batch["anchor_angle"].to(device))
                target_tokens = model.encode(
                    targets.flatten(0, 1), batch["target_angles"].to(device).flatten()
                ).reshape_as(predicted)
                anchor_tokens = model.encode(anchor, batch["anchor_angle"].to(device))
                anchor_globals = anchor_tokens.mean(dim=1)[:, None].expand(-1, num_targets, -1).flatten(0, 1)
                target_globals = target_tokens.mean(dim=2).flatten(0, 1)
                heldout_loss = predictive_loss(predicted, target_tokens) + float(
                    config["sigreg"]["lambda"]
                ) * regularizer(torch.cat((anchor_globals, target_globals), dim=0))
                validation_loss += float(heldout_loss) * batch_size
                validation_count += batch_size
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state[3])
        validation_loss /= validation_count
        improved = validation_loss < best_loss
        best_loss = min(best_loss, validation_loss)
        save_state(destination / "last.pt", _epoch + 1, best_loss)
        if improved:
            save_state(destination / "best.pt", _epoch + 1, best_loss)
        row = {"epoch": _epoch + 1, "train_loss": epoch_total / len(loader), "validation_loss": validation_loss}
        with (destination / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return destination / "last.pt"


def main() -> None:
    """Parse CLI arguments and launch world-model pretraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config("pretraining"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pretraining"))
    arguments = parser.parse_args()
    configuration = load_config(arguments.config)
    checkpoint = train(configuration, arguments.output_dir)
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
