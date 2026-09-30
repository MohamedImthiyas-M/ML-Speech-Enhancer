"""
==========================================================
dataset_loader.py

PyTorch Dataset Loader for
Residual Attention U-Net Speech Enhancement


Input:
    noisy speech


Target:
    clean speech


Output:

    noisy spectrogram
    clean spectrogram

==========================================================
"""


import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


from config import (
    CLEAN_DIR,
    NOISY_DIR,
    BATCH_SIZE
)


from preprocessing import (
    preprocess_audio
)

from config import NUM_WORKERS
from config import DEVICE




# ==========================================================
# SPEECH DATASET
# ==========================================================


class SpeechEnhancementDataset(Dataset):

    """
    Dataset for paired noisy-clean speech
    """


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


        # Find matching files

        for file in self.noisy_dir.iterdir():


            if file.suffix.lower() in [
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",
                ".m4a"
            ]:


                clean_file = (
                    self.clean_dir /
                    file.name
                )


                if clean_file.exists():

                    self.files.append(
                        file.name
                    )


        print(
            "Total audio pairs:",
            len(self.files)
        )



    # ------------------------------------------------------
    # Dataset length
    # ------------------------------------------------------

    def __len__(self):

        return len(
            self.files
        )



    # ------------------------------------------------------
    # Get item
    # ------------------------------------------------------

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



        # -------------------------------
        # Preprocess noisy audio
        # -------------------------------

        noisy_spec, noisy_phase = preprocess_audio(
            noisy_path
        )



        # -------------------------------
        # Preprocess clean audio
        # -------------------------------

        clean_spec, _ = preprocess_audio(
            clean_path
        )



        # Add channel dimension

        noisy_spec = noisy_spec.unsqueeze(
            0
        )


        clean_spec = clean_spec.unsqueeze(
            0
        )



        return {

           "noisy": noisy_spec.float(),

           "clean": clean_spec.float(),

           "phase": noisy_phase.float(),

           "filename": filename

        }







# ==========================================================
# DATA LOADER CREATION
# ==========================================================


def create_dataloader(shuffle=True):

    dataset = SpeechEnhancementDataset()

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        num_workers=NUM_WORKERS,

        pin_memory=(DEVICE.type == "cuda"),

        persistent_workers=(NUM_WORKERS > 0),

        prefetch_factor=2 if NUM_WORKERS > 0 else None

    )


    return loader
