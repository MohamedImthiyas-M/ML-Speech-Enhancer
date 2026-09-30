from dataset_loader import SpeechEnhancementDataset

dataset = SpeechEnhancementDataset()

sample = dataset[0]

print(type(sample))

print(sample)

if isinstance(sample, dict):
    print("Keys:", sample.keys())

    print("Noisy :", sample["noisy"].shape)
    print("Clean :", sample["clean"].shape)
    print("Phase :", sample["phase"].shape)
