import torch

from models.attention import AttentionGate


gate = AttentionGate(
    gate_channels=512,
    skip_channels=256,
    inter_channels=128
)

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

output = gate(
    decoder,
    encoder
)

print("Decoder :", decoder.shape)
print("Encoder :", encoder.shape)
print("Output  :", output.shape)
