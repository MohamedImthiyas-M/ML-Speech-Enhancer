from dataset_loader import create_dataloader

loader = create_dataloader()

batch = next(iter(loader))

print(batch["noisy"].shape)

print(batch["clean"].shape)

print(batch["phase"].shape)

print(batch["filename"][0])
