import torch

from models.loss import HybridSpeechLoss

loss_fn = HybridSpeechLoss()

pred = torch.randn(
    2,
    1,
    257,
    1001
)

target = torch.randn(
    2,
    1,
    257,
    1001
)

loss = loss_fn(
    pred,
    target
)

print("Loss:", loss.item())
