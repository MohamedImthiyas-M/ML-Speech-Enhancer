"""
==========================================================
attention.py

Attention Gate for Residual Attention U-Net

Based on:
Attention U-Net (Oktay et al.)

==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate

    g = decoder feature (gating signal)

    x = encoder feature (skip connection)

    Output:
        Attention filtered skip feature
    """

    def __init__(
        self,
        gate_channels,
        skip_channels,
        inter_channels
    ):

        super().__init__()

        # Decoder feature projection
        self.W_g = nn.Sequential(

            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(inter_channels)

        )

        # Encoder feature projection
        self.W_x = nn.Sequential(

            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(inter_channels)

        )

        # Attention coefficient
        self.psi = nn.Sequential(

            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                bias=True
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid()

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        # ------------------------------------
        # Match spatial dimensions
        # ------------------------------------

        if g.shape[2:] != x.shape[2:]:

            g = F.interpolate(
                g,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        # ------------------------------------
        # Linear projections
        # ------------------------------------

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        # ------------------------------------
        # Attention map
        # ------------------------------------

        psi = self.relu(
            g1 + x1
        )

        psi = self.psi(psi)

        # ------------------------------------
        # Apply attention
        # ------------------------------------

        return x * psi
