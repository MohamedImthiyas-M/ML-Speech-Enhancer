"""
==========================================================
blocks.py

Building blocks for Residual Attention U-Net

Contains

• ConvBlock
• ResidualConv
• DownSample
• UpSample
• DecoderBlock

==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import AttentionGate
# ==========================================================
# BASIC DOUBLE CONVOLUTION
# ==========================================================

class ConvBlock(nn.Module):

    """
    Conv
    BatchNorm
    ReLU

    Conv
    BatchNorm
    ReLU
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# RESIDUAL CONVOLUTION BLOCK
# ==========================================================

class ResidualConv(nn.Module):

    """
    Residual Convolution

          Input
            │
        ConvBlock
            │
        + Shortcut
            │
          ReLU

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels
        )

        # Match channels if needed

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv(x)

        out = out + identity

        out = self.relu(out)

        return out


# ==========================================================
# ENCODER BLOCK
# ==========================================================

class DownSample(nn.Module):

    """
    Encoder

    ResidualConv

        ↓

    MaxPool

    Returns

    features
    pooled
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ResidualConv(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled


# ==========================================================
# UPSAMPLING
# ==========================================================

class UpSample(nn.Module):

    """
    Bilinear Upsampling

    followed by

    3x3 Conv

    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.up(x)


# ==========================================================
# DECODER BLOCK
# ==========================================================

# ==========================================================
# DECODER BLOCK WITH ATTENTION
# ==========================================================

class DecoderBlock(nn.Module):

    """
    Decoder Block

    Input
      │
      ▼
    Upsample
      │
      ▼
    Attention Gate
      │
      ▼
    Concatenate
      │
      ▼
    ResidualConv
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        # Upsample decoder feature

        self.up = UpSample(
            in_channels,
            out_channels
        )

        # Filter encoder feature

        self.attention = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 1)
        )

        # Fuse features

        self.conv = ResidualConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        decoder_feature,
        encoder_feature
    ):

        decoder_feature = self.up(
            decoder_feature
        )

        # Apply attention to encoder feature

        encoder_feature = self.attention(
            decoder_feature,
            encoder_feature
        )

        # Handle odd image sizes

        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:

            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        x = torch.cat(
            [
                encoder_feature,
                decoder_feature
            ],
            dim=1
        )

        x = self.conv(x)

        return x
