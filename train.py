"""
Training script for Audio-Visual Speech Enhancement
"""
import os
import sys
import glob
import torch
import argparse
import yaml
import warnings

import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
from look2hear.system import make_optimizer
from look2hear.utils import print_only

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from swanlab.integration.pytorch_lightning import SwanLabLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/av_convtasnet.yml",
    help="Path to configuration file",
)
parser.add_argument(
    "--warm_start",
    default=None,
    help=(
        "Path to a checkpoint to warm-start from: loads ONLY the model weights "
        "(no optimizer/scheduler/epoch state), so training restarts fresh with "
        "the config's lr. Takes precedence over auto-resume from last.ckpt."
    ),
)


def find_latest_checkpoint(exp_dir):
    """Return the checkpoint to resume from, or None to train from scratch.

    Prefers ``last.ckpt`` (written by ModelCheckpoint(save_last=True)); falls
    back to the most recently modified ``*.ckpt`` in ``exp_dir``.
    """
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.isfile(last):
        print_only(f"Found last.ckpt, resuming from: {last}")
        return last

    candidates = glob.glob(os.path.join(exp_dir, "*.ckpt"))
    if candidates:
        latest = max(candidates, key=os.path.getmtime)
        print_only(f"Found latest checkpoint, resuming from: {latest}")
        return latest

    print_only(f"No checkpoint in {exp_dir}, training from scratch")
    return None


def load_model_weights(model, ckpt_path):
    """Load only model weights from a checkpoint (no optimizer/scheduler state).

    Accepts a Lightning checkpoint (``state_dict`` with ``audio_model.``-prefixed
    keys), a bare module ``state_dict``, or a ``serialize()`` payload (``state_dict``
    holding the model's own keys).
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        full_sd = state["state_dict"]
        # Lightning system wraps the net as ``self.audio_model`` -> keys are
        # ``audio_model.*``. A serialize() payload's state_dict is already the
        # model's own keys, so fall back to it when no prefix matches.
        model_sd = {
            k[len("audio_model."):]: v
            for k, v in full_sd.items()
            if k.startswith("audio_model.")
        }
        if not model_sd:
            model_sd = full_sd
    else:
        model_sd = state

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print_only(f"Warm-start: loaded model weights from {ckpt_path}")
    if missing:
        print_only(f"  missing keys ({len(missing)}): {missing}")
    if unexpected:
        print_only(f"  unexpected keys ({len(unexpected)}): {unexpected}")


def main(config, warm_start=None):
    """Main training function"""

    # Print configuration
    print_only("=" * 50)
    print_only("Training Configuration:")
    print_only("=" * 50)
    print_only(f"Experiment: {config['exp']['exp_name']}")
    print_only(f"Audio Model: {config['audionet']['audionet_name']}")
    print_only(f"Video Model: {config.get('videonet', {}).get('videonet_name', 'bundled')}")
    print_only("=" * 50)

    # Instantiate datamodule
    print_only(f"Instantiating datamodule <{config['datamodule']['data_name']}>")
    datamodule = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    train_loader, val_loader, _ = datamodule.make_loader

    # Define audio-visual model. The video encoder is now bundled inside the
    # model, so its config is passed through here (prefixed video_*) instead of
    # building a separate video net.
    print_only(f"Instantiating AudioNet <{config['audionet']['audionet_name']}>")
    video_cfg = config.get("videonet", {}).get("videonet_config", {})
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        video_relu_type=video_cfg.get("relu_type", "prelu"),
        video_pretrain=video_cfg.get("pretrain", None),
        **config["audionet"]["audionet_config"],
    )

    # Warm-start: load trained weights only (no optimizer/scheduler/epoch state),
    # so the optimizer below is rebuilt from the config lr. Disables auto-resume.
    if warm_start is not None:
        load_model_weights(model, warm_start)

    # Define optimizer. Only optimize trainable params -- the bundled video
    # encoder is frozen, so it is excluded here.
    print_only(f"Instantiating Optimizer <{config['optimizer']['optim_name']}>")
    optimizer = make_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        **config["optimizer"],
    )

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(f"Instantiating Scheduler <{config['scheduler']['sche_name']}>")
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer, **config["scheduler"]["sche_config"]
        )

    # Setup experiment directory
    exp_dir = os.path.join(os.getcwd(), "Experiments", config["exp"]["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    # Save configuration
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define loss function
    print_only(
        f"Instantiating Loss, Train <{config['loss']['train']['sdr_type']}>, "
        f"Val <{config['loss']['val']['sdr_type']}>"
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    # Define system
    print_only(f"Instantiating System <{config['training']['system']}>")
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint = ModelCheckpoint(
        exp_dir,
        filename="{epoch}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))

    # Setup GPUs
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # Setup logger. Only instantiate SwanLab on global rank 0: SwanLabLogger's
    # `experiment` calls swanlab.init without a rank guard, so under DDP every
    # spawned process would otherwise start its own run. Lightning broadcasts the
    # rank-0 logger to the other ranks internally.
    logger = False
    is_rank_zero = (
        int(os.environ.get("LOCAL_RANK", 0)) == 0
        and int(os.environ.get("NODE_RANK", 0)) == 0
        and int(os.environ.get("GLOBAL_RANK", 0)) == 0
    )
    if is_rank_zero:
        log_cfg = config.get("logger", {}) or {}
        log_dir = os.path.join(exp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        print_only("Instantiating SwanLabLogger")
        logger = SwanLabLogger(
            project=log_cfg.get("project", "Real-Wold-AVSE"),
            experiment_name=log_cfg.get("experiment_name", config["exp"]["exp_name"]),
            save_dir=log_dir,
        )

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True) if gpus and len(gpus) > 1 else "auto",
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        logger=logger,
        sync_batchnorm=True if gpus and len(gpus) > 1 else False,
    )

    # Resume: auto-pick last.ckpt for a full resume (optimizer/scheduler/epoch
    # restored). Skipped when warm-starting, so the optimizer starts fresh.
    resume_ckpt = None
    if warm_start is not None:
        print_only("warm_start set; skipping auto-resume so the optimizer starts fresh.")
    else:
        resume_ckpt = find_latest_checkpoint(exp_dir)

    # Train
    print_only("=" * 50)
    print_only("Starting Training...")
    print_only("=" * 50)
    trainer.fit(system, ckpt_path=resume_ckpt)

    print_only("Finished Training")

    # Save best model. Under DDP every rank re-executes this code, so guard the
    # write to rank 0 only -- otherwise all ranks torch.save() to the same path
    # concurrently and can corrupt best_model.pth. Write to a temp file and
    # atomically rename so a reader never sees a partial checkpoint.
    if trainer.is_global_zero:
        try:
            state_dict = torch.load(checkpoint.best_model_path)
            system.load_state_dict(state_dict=state_dict["state_dict"])
            system.cpu()

            to_save = system.audio_model.serialize()
            best_path = os.path.join(exp_dir, "best_model.pth")
            tmp_path = best_path + ".tmp"
            torch.save(to_save, tmp_path)
            os.replace(tmp_path, best_path)
            print_only(f"Best model saved to {best_path}")
        except Exception as e:
            print_only(f"Warning: Could not save best model: {e}")


if __name__ == "__main__":
    args = parser.parse_args()

    # Load configuration
    with open(args.conf_dir) as f:
        config = yaml.safe_load(f)

    # Start training
    main(config, warm_start=args.warm_start)
