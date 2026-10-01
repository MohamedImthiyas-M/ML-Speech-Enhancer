"""
==========================================================
unet.py

Residual Attention U-Net V3
Direct Clean Spectrogram Regression

Input:
    Noisy normalized log spectrogram

Output:
    Predicted clean normalized log spectrogram

==========================================================
"""

import torch
import torch.nn as nn

from config import (
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    BASE_CHANNELS,
    BOTTLENECK_BLOCKS
)

from models.blocks import (
    ConvBlock,
    ResidualConv,
    DownSample,
    DecoderBlock
)


class ResidualAttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        c = BASE_CHANNELS

        # ======================================================
        # Initial Convolution
        # ======================================================

        self.stem = ConvBlock(
            INPUT_CHANNELS,
            c
        )

        # ======================================================
        # Encoder
        # ======================================================

        self.enc1 = DownSample(
            c,
            c * 2
        )

        self.enc2 = DownSample(
            c * 2,
            c * 4
        )

        self.enc3 = DownSample(
            c * 4,
            c * 8
        )

        self.enc4 = DownSample(
            c * 8,
            c * 16
        )

        # ======================================================
        # Bottleneck
        # ======================================================

        bottleneck = []

        for _ in range(BOTTLENECK_BLOCKS):

            bottleneck.append(
                ResidualConv(
                    c * 16,
                    c * 16
                )
            )

        self.bottleneck = nn.Sequential(
            *bottleneck
        )

        # ======================================================
        # Decoder
        # ======================================================

        self.dec4 = DecoderBlock(
            in_channels=c * 16,
            skip_channels=c * 16,
            out_channels=c * 8
        )

        self.dec3 = DecoderBlock(
            in_channels=c * 8,
            skip_channels=c * 8,
            out_channels=c * 4
        )

        self.dec2 = DecoderBlock(
            in_channels=c * 4,
            skip_channels=c * 4,
            out_channels=c * 2
        )

        self.dec1 = DecoderBlock(
            in_channels=c * 2,
            skip_channels=c * 2,
            out_channels=c
        )

        self.dec0 = DecoderBlock(
            in_channels=c,
            skip_channels=c,
            out_channels=c
        )

        # ======================================================
        # Final Output Layer
        # ======================================================
        #
        # IMPORTANT:
        # No Sigmoid here.
        #
        # The network must be able to output both positive
        # and negative normalized spectrogram values.
        #
        # ======================================================

        self.output = nn.Conv2d(
            c,
            OUTPUT_CHANNELS,
            kernel_size=1
        )

    def forward(self, x):

        # ======================================================
        # Encoder
        # ======================================================

        x0 = self.stem(x)

        s1, p1 = self.enc1(x0)

        s2, p2 = self.enc2(p1)

        s3, p3 = self.enc3(p2)

        s4, p4 = self.enc4(p3)

        # ======================================================
        # Bottleneck
        # ======================================================

        b = self.bottleneck(p4)

        # ======================================================
        # Decoder
        # ======================================================

        d4 = self.dec4(
            b,
            s4
        )

        d3 = self.dec3(
            d4,
            s3
        )

        d2 = self.dec2(
            d3,
            s2
        )

        d1 = self.dec1(
            d2,
            s1
        )

        d0 = self.dec0(
            d1,
            x0
        )

        # ======================================================
        # Direct Clean Spectrogram Prediction
        # ======================================================

        output = self.output(d0)

        return output