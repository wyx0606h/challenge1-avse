"""
PyTorch Lightning module for Audio-Visual Speech Enhancement
"""
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AudioVisualLightningModule(pl.LightningModule):
    """
    Lightning Module for Audio-Visual Speech Enhancement Training

    Args:
        audio_model: Audio-visual enhancement model. The lip-reading video
            encoder is bundled inside this model, so raw lip frames are passed
            straight through to it.
        optimizer: Optimizer
        loss_func: Dictionary of loss functions for train/val
        train_loader: Training data loader
        val_loader: Validation data loader
        scheduler: Learning rate scheduler
        config: Training configuration
    """

    def __init__(
        self,
        audio_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config

        self.default_monitor = "val_loss"
        self.save_hyperparameters(self.config, ignore=['audio_model'])

        self.validation_step_outputs = []

    def forward(self, wav, mouth=None):
        """
        Forward pass

        Args:
            wav: [B, T] audio waveform
            mouth: [B, Tv, H, W] or [B, 1, Tv, H, W] lip frames

        Returns:
            Enhanced speech
        """
        # The video encoder lives inside audio_model (and is frozen there).
        return self.audio_model(wav, mouth)

    def training_step(self, batch, batch_idx):
        """Training step"""
        mixtures, targets, mouth, _ = batch

        # Forward pass
        est_sources = self(mixtures, mouth)

        # Ensure correct shape
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        # Calculate loss
        loss = self.loss_func["train"](est_sources, targets)

        # Log. on_step=True so the progress bar shows the live per-step loss
        # (without it PL still creates a ``train_loss_step`` entry but never
        # updates it, so the bar reads 0.000 even while training is fine).
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        mixtures, targets, mouth, _ = batch
        est_sources = self(mixtures, mouth)

        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        loss = self.loss_func["val"](est_sources, targets)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        self.validation_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """Validation epoch end"""
        if len(self.validation_step_outputs) > 0:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            val_loss = torch.mean(self.all_gather(avg_loss))

            self.log(
                "lr",
                self.optimizer.param_groups[0]["lr"],
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            # Logged via the standard Lightning channel so any logger
            # (SwanLab here) picks it up; no logger-specific experiment API.
            self.log("val_pit_sisnr", -val_loss, on_epoch=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched.get("interval") == "batch":
                    sched["interval"] = "step"
                epoch_schedulers.append(sched)

        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Save additional info to checkpoint"""
        checkpoint["training_config"] = self.config
        return checkpoint
