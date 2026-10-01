"""
==========================================================
train.py

AI Speech Enhancer V3 Training

Residual Attention U-Net
Direct Clean Spectrogram Regression

==========================================================
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    PATIENCE,
    CHECKPOINT_DIR
)

from dataset_loader import SpeechEnhancementDataset

from models.unet import ResidualAttentionUNet


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================

VALIDATION_RATIO = 0.20


# ==========================================================
# LOSS FUNCTION
# ==========================================================

class SpectrogramLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.mse = nn.MSELoss()

    def forward(
        self,
        prediction,
        target
    ):

        l1_loss = self.l1(
            prediction,
            target
        )

        mse_loss = self.mse(
            prediction,
            target
        )

        # L1 is the main loss.
        # MSE helps keep larger errors under control.

        total_loss = (
            0.8 * l1_loss +
            0.2 * mse_loss
        )

        return total_loss


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    batches = 0

    for batch in loader:

        noisy = batch["noisy"].to(
            DEVICE,
            non_blocking=True
        )

        clean = batch["clean"].to(
            DEVICE,
            non_blocking=True
        )

        # --------------------------------------------------
        # Check input
        # --------------------------------------------------

        if not torch.isfinite(
            noisy
        ).all():

            raise RuntimeError(
                "NaN/Inf detected in noisy input."
            )

        if not torch.isfinite(
            clean
        ).all():

            raise RuntimeError(
                "NaN/Inf detected in clean target."
            )

        # --------------------------------------------------
        # Clear gradients
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        prediction = model(
            noisy
        )

        # --------------------------------------------------
        # Check prediction
        # --------------------------------------------------

        if not torch.isfinite(
            prediction
        ).all():

            raise RuntimeError(
                "Model produced NaN/Inf values."
            )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss = criterion(
            prediction,
            clean
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Loss became NaN/Inf."
            )

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

        batches += 1

    if batches == 0:

        return float("inf")

    return total_loss / batches


# ==========================================================
# VALIDATION
# ==========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    batches = 0

    for batch in loader:

        noisy = batch["noisy"].to(
            DEVICE,
            non_blocking=True
        )

        clean = batch["clean"].to(
            DEVICE,
            non_blocking=True
        )

        prediction = model(
            noisy
        )

        if not torch.isfinite(
            prediction
        ).all():

            raise RuntimeError(
                "NaN/Inf detected during validation."
            )

        loss = criterion(
            prediction,
            clean
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Validation loss became NaN/Inf."
            )

        total_loss += loss.item()

        batches += 1

    if batches == 0:

        return float("inf")

    return total_loss / batches


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    loss
):

    checkpoint = {

        "epoch": epoch,

        "model": model.state_dict(),

        "optimizer": optimizer.state_dict(),

        "scheduler": scheduler.state_dict(),

        "loss": loss,

        "learning_rate":
            optimizer.param_groups[0]["lr"]
    }

    torch.save(
        checkpoint,
        path
    )


# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():

    print()
    print(
        "=================================================="
    )

    print(
        "          AI SPEECH ENHANCER V3"
    )

    print(
        "          TRAINING"
    )

    print(
        "=================================================="
    )

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print()

    # ======================================================
    # DATASET
    # ======================================================

    dataset = SpeechEnhancementDataset()

    total_samples = len(
        dataset
    )

    if total_samples < 10:

        raise RuntimeError(
            "\n"
            "Dataset is too small.\n\n"
            f"Found only {total_samples} paired audio files.\n"
            "Please restore the full dataset before training.\n"
            "Your earlier dataset contained about 1217 pairs."
        )

    validation_size = max(
        1,
        int(
            total_samples *
            VALIDATION_RATIO
        )
    )

    training_size = (
        total_samples -
        validation_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=generator
    )

    print(
        "Total samples:",
        total_samples
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    print()

    # ======================================================
    # DATALOADERS
    # ======================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = ResidualAttentionUNet()

    model = model.to(
        DEVICE
    )

    # ======================================================
    # LOSS
    # ======================================================

    criterion = SpectrogramLoss()

    # ======================================================
    # OPTIMIZER
    # ======================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ======================================================
    # SCHEDULER
    # ======================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # ======================================================
    # CHECKPOINT PATHS
    # ======================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        CHECKPOINT_DIR /
        "best.pt"
    )

    latest_checkpoint = (
        CHECKPOINT_DIR /
        "latest.pt"
    )

    # ======================================================
    # REMOVE OLD V3 CHECKPOINTS
    # ======================================================
    #
    # IMPORTANT:
    # This training starts from scratch.
    #
    # ======================================================

    if best_checkpoint.exists():

        best_checkpoint.unlink()

        print(
            "Removed old best.pt"
        )

    if latest_checkpoint.exists():

        latest_checkpoint.unlink()

        print(
            "Removed old latest.pt"
        )

    print()

    # ======================================================
    # TRAINING VARIABLES
    # ======================================================

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "=================================================="
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            "=================================================="
        )

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        validation_loss = validate(
            model,
            validation_loader,
            criterion
        )

        # --------------------------------------------------
        # Learning rate
        # --------------------------------------------------

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Valid Loss : {validation_loss:.6f}"
        )

        print(
            f"Learning Rate : {current_lr:.8f}"
        )

        # --------------------------------------------------
        # Save latest
        # --------------------------------------------------

        save_checkpoint(
            latest_checkpoint,
            epoch,
            model,
            optimizer,
            scheduler,
            validation_loss
        )

        print(
            "Latest checkpoint saved."
        )

        # --------------------------------------------------
        # Best model
        # --------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                validation_loss
            )

            print(
                "New BEST model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement."
            )

            print(
                "Patience:",
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        print()

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                "Early stopping triggered."
            )

            break

    # ======================================================
    # FINISHED
    # ======================================================

    print()
    print(
        "=================================================="
    )

    print(
        "             TRAINING COMPLETED"
    )

    print(
        "=================================================="
    )

    print()

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        best_checkpoint
    )

    print()

    print(
        "Latest checkpoint:"
    )

    print(
        latest_checkpoint
    )

    print()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()