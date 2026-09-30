import torch

from models.blocks import DecoderBlock


decoder = torch.randn(
    2,
    512,
    16,
    16
)

encoder = torch.randn(
    2,
    256,
    32,
    32
)

block = DecoderBlock(
    in_channels=512,
    skip_channels=256,
    out_channels=256
)

output = block(
    decoder,
    encoder
)

print(output.shape)
