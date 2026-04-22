import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MeanMetric, MaxMetric
from torchvision.transforms.functional import pil_to_tensor
import logging

import config
from train.dataset import get_dataloaders
from train.model import UNet, count_parameters
from losses import SecurityLoss
from utils import evaluate_batch, visualize_results

class JalDrishtiLightning(pl.LightningModule):
    """
    The Industrial PyTorch Lightning Architecture for JalDrishti.
    Replaces massive boilerplate loops with a robust, automated state machine.
    """
    def __init__(self, net: torch.nn.Module, criterion: nn.Module):
        super().__init__()
        self.net = net
        self.criterion = criterion
        
        # TorchMetrics for flawless, native hardware synchronization
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_psnr = MeanMetric()
        self.val_ssim = MeanMetric()
        self.val_psnr_best = MaxMetric()

        # Save hyperparameters ignoring the massive Model objects
        self.save_hyperparameters(ignore=['net', 'criterion'])

    def forward(self, x):
        """Lightning wrapper for inference passes."""
        return self.net(x)

    def configure_optimizers(self):
        """Bind the AdamW engine and learning rate parameters."""
        optimizer = optim.AdamW(self.net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        raw_imgs, target_imgs = batch
        
        # Automatic Forward Pass
        preds = self.forward(raw_imgs)
        
        # PINN Constraints (Total Variation and DCP Physics calculate inside SecurityLoss)
        loss, _ = self.criterion(preds, target_imgs)
        
        # Update and log metrics natively
        self.train_loss(loss)
        self.log("Train/Loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        raw_imgs, target_imgs = batch
        
        preds = self.forward(raw_imgs)
        loss, _ = self.criterion(preds, target_imgs)
        
        # Use existing utility logic to evaluate the batch
        batch_psnr, batch_ssim = evaluate_batch(preds, target_imgs)
        
        # Log to torchmetrics
        self.val_loss(loss)
        self.val_psnr(batch_psnr)
        self.val_ssim(batch_ssim)
        
        self.log("Val/Loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val/PSNR", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val/SSIM", self.val_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Automate single-batch Tensorboard visual logging
        if batch_idx == 0:
            example = (raw_imgs[0].cpu(), preds[0].cpu(), target_imgs[0].cpu())
            viz_path = os.path.join(config.RESULTS_DIR, f"epoch_{self.current_epoch:03d}_val.png")
            
            # Use our existing visualization utility (saves to disk)
            visualize_results(example[0], example[1], example[2], save_path=viz_path)
            
            """
            Because we're in PyTorch Lightning, we can also inject this cleanly into Tensorboard!
            We read the saved PIL image back in and push it directly into the dashboard.
            """
            from PIL import Image
            if os.path.exists(viz_path) and self.logger is not None:
                pil_img = Image.open(viz_path)
                self.logger.experiment.add_image("Val/Visualization", pil_to_tensor(pil_img), self.current_epoch)

    def on_validation_epoch_end(self):
        """Capture the best absolute PSNR."""
        psnr = self.val_psnr.compute()
        self.val_psnr_best(psnr)
        if psnr == self.val_psnr_best.compute():
            self.log("Val/Best_PSNR", self.val_psnr_best.compute(), prog_bar=True)


def train_model():
    """Execute the PyTorch Lightning Training Architecture."""
    print("=" * 60)
    print("MARITIME SECURITY ENHANCEMENT - LIGHTNING ENGINE")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 60)

    # 1. Data Subsystem
    train_loader, val_loader = get_dataloaders(
        raw_dir=config.RAW_DIR,
        enhanced_dir=config.ENHANCED_DIR,
        batch_size=config.BATCH_SIZE,
        train_split=config.TRAIN_SPLIT,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # 2. Model Initialization
    net = UNet(in_channels=config.IMAGE_CHANNELS, out_channels=config.IMAGE_CHANNELS, base=config.UNET_INIT_FEATURES)
    num_params = count_parameters(net)
    print(f"Model parameters: {num_params:,}")

    # 3. Master Loss Initialization (Total Variation, Perceptual, DCP Physics)
    criterion = SecurityLoss()

    # 4. Lightning Module Wrapper
    lightning_model = JalDrishtiLightning(net=net, criterion=criterion)

    # 5. Lightning Checkpoint Callback (Saves the literal best_model.pth automatically)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='Val/Loss',
        mode='min'
    )

    # 6. Tensorboard Initialization
    logger = TensorBoardLogger(save_dir=config.RESULTS_DIR, name="lightning_logs")

    # 7. Lightning Trainer Build
    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    # 8. Fire
    print("\n[+] Engaging Physics-Informed Lightning Trainer...")
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    train_model()