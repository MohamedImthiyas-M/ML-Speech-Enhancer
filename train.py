"""
==========================================================
train.py

Professional Trainer
Residual Attention U-Net Speech Enhancer

Features
--------
✓ GPU / CPU Support
✓ Mixed Precision (AMP)
✓ Validation
✓ Checkpoints
✓ Resume Training
✓ Progress Bar
✓ Best Model Saving

==========================================================
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    EPOCHS,
    CHECKPOINT_DIR
)

from dataset_loader import SpeechEnhancementDataset

from models.unet import ResidualAttentionUNet

from models.loss import HybridSpeechLoss

# ==========================================================
# DATASET
# ==========================================================

dataset = SpeechEnhancementDataset()

train_size = int(
    0.9 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================================
# MODEL
# ==========================================================

model = ResidualAttentionUNet().to(
    DEVICE
)
torch.cuda.empty_cache()
criterion = HybridSpeechLoss()

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scaler = GradScaler("cuda")

# ==========================================================
# SAVE MODEL
# ==========================================================

def save_checkpoint(

        epoch,

        model,

        optimizer,

        loss,

        filename

):

    torch.save(

        {

            "epoch": epoch,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "loss": loss

        },

        filename

    )

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_checkpoint(path):

    if not os.path.exists(path):

        return 0

    checkpoint = torch.load(

        path,

        map_location=DEVICE

    )

    model.load_state_dict(

        checkpoint["model"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer"]

    )

    print(

        "Resumed from Epoch",

        checkpoint["epoch"]

    )

    return checkpoint["epoch"] + 1


   # ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Train Epoch {epoch+1}/{EPOCHS}"
    )

    for batch in progress:


        # ----------------------------------
        # Load batch
        # ----------------------------------

        noisy = batch["noisy"].to(DEVICE)

        clean = batch["clean"].to(DEVICE)

        optimizer.zero_grad()

        # ----------------------------------
        # Forward
        # ----------------------------------

        with autocast(device_type="cuda"):

            prediction = model(noisy)

            loss = criterion(
                prediction,
                clean
            )

        # ----------------------------------
        # Backward
        # ----------------------------------

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    return running_loss / len(train_loader)

   # ==========================================================
# VALIDATION
# ==========================================================

def validate():

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            noisy = batch["noisy"].to(DEVICE)

            clean = batch["clean"].to(DEVICE)

            prediction = model(noisy)

            loss = criterion(

                prediction,

                clean

            )

            running_loss += loss.item()

    return running_loss / len(val_loader)

   # ==========================================================
# TRAINING
# ==========================================================

best_loss = float("inf")

start_epoch = load_checkpoint(

    CHECKPOINT_DIR / "latest.pt"

)

for epoch in range(

    start_epoch,

    EPOCHS

):

    train_loss = train_one_epoch(

        epoch

    )

    val_loss = validate()

    print()

    print(

        f"Epoch {epoch+1}"

    )

    print(

        f"Train Loss : {train_loss:.5f}"

    )

    print(

        f"Valid Loss : {val_loss:.5f}"

    )

    # --------------------------------------
    # Save latest checkpoint
    # --------------------------------------

    save_checkpoint(

        epoch,

        model,

        optimizer,

        val_loss,

        CHECKPOINT_DIR / "latest.pt"

    )

    # --------------------------------------
    # Save best checkpoint
    # --------------------------------------

    if val_loss < best_loss:

        best_loss = val_loss

        save_checkpoint(

            epoch,

            model,

            optimizer,

            val_loss,

            CHECKPOINT_DIR / "best.pt"

        )

        print("Best model updated.")

      
