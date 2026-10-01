from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    CLEAN_DIR,
    NOISY_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)

from preprocessing import preprocess_training_pair


# ==========================================================
# DATASET
# ==========================================================

class SpeechEnhancementDataset(Dataset):

    def __init__(
        self,
        noisy_dir=NOISY_DIR,
        clean_dir=CLEAN_DIR
    ):

        self.noisy_dir = Path(
            noisy_dir
        )

        self.clean_dir = Path(
            clean_dir
        )

        self.files = []

        # --------------------------------------------------
        # Find matching noisy/clean pairs
        # --------------------------------------------------

        for file in self.noisy_dir.iterdir():

            if file.suffix.lower() not in [
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",
                ".m4a"
            ]:
                continue

            clean_file = (
                self.clean_dir /
                file.name
            )

            if clean_file.exists():

                self.files.append(
                    file.name
                )

        self.files.sort()

        print(
            "Total audio pairs:",
            len(self.files)
        )

    def __len__(self):

        return len(
            self.files
        )

    def __getitem__(
        self,
        index
    ):

        filename = self.files[index]

        noisy_path = (
            self.noisy_dir /
            filename
        )

        clean_path = (
            self.clean_dir /
            filename
        )

        (
            noisy_spec,
            clean_spec,
            noisy_phase,
            noisy_mean,
            noisy_std
        ) = preprocess_training_pair(
            noisy_path,
            clean_path
        )

        # --------------------------------------------------
        # Add channel dimension
        # --------------------------------------------------

        noisy_spec = noisy_spec.unsqueeze(0)

        clean_spec = clean_spec.unsqueeze(0)

        return {
            "noisy": noisy_spec.float(),

            "clean": clean_spec.float(),

            "phase": noisy_phase.float(),

            "mean": noisy_mean.float(),

            "std": noisy_std.float(),

            "filename": filename
        }


# ==========================================================
# DATALOADER
# ==========================================================

def create_dataloader(
    shuffle=True
):

    dataset = SpeechEnhancementDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        ),

        prefetch_factor=(
            2
            if NUM_WORKERS > 0
            else None
        )
    )

    return loader