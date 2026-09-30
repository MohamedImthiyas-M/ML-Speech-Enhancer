"""
==========================================================
loss.py

Hybrid Loss for Speech Enhancement

Components

1. L1 Loss

2. Spectral Convergence Loss

3. Multi-Resolution STFT Loss

==========================================================
"""

import torch
import torch.nn as nn


# ==========================================================
# SPECTRAL CONVERGENCE
# ==========================================================

class SpectralConvergenceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, prediction, target):

        numerator = torch.norm(
            target - prediction,
            p="fro"
        )

        denominator = torch.norm(
            target,
            p="fro"
        )

        return numerator / (
            denominator + 1e-8
        )


# ==========================================================
# MULTI-RESOLUTION STFT LOSS
# ==========================================================

class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction,
        target
    ):

        total_loss = 0.0

        fft_sizes = [
            256,
            512,
            1024
        ]

        for fft in fft_sizes:

            pred = torch.nn.functional.interpolate(

                prediction,

                scale_factor=1.0,

                mode="bilinear",

                align_corners=False

            )

            gt = torch.nn.functional.interpolate(

                target,

                scale_factor=1.0,

                mode="bilinear",

                align_corners=False

            )

            total_loss += self.l1(
                pred,
                gt
            )

        return total_loss / len(
            fft_sizes
        )


# ==========================================================
# HYBRID LOSS
# ==========================================================

class HybridSpeechLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.sc = SpectralConvergenceLoss()

        self.mrstft = MultiResolutionSTFTLoss()

    def forward(
        self,
        prediction,
        target
    ):

        l1 = self.l1(
            prediction,
            target
        )

        sc = self.sc(
            prediction,
            target
        )

        mr = self.mrstft(
            prediction,
            target
        )

        total = (

            0.4 * l1

            +

            0.2 * sc

            +

            0.4 * mr

        )

        return total
