import torch

from models.unet import ResidualAttentionUNet

model = ResidualAttentionUNet()

print(model)

x = torch.randn(
    2,
    1,
    256,
    256
)

with torch.no_grad():

    y = model(x)

print("Input :", x.shape)

print("Output:", y.shape)
