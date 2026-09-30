from pathlib import Path
import torch


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent



# ==========================================================
# DATASET PATHS
# ==========================================================

DATASET_DIR = PROJECT_ROOT / "dataset"


CLEAN_DIR = DATASET_DIR / "clean_speech"

NOISY_DIR = DATASET_DIR / "noisy_speech"



# ==========================================================
# PROJECT FOLDERS
# ==========================================================

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEMP_DIR = PROJECT_ROOT / "temp"

MODELS_DIR = PROJECT_ROOT / "models"



for directory in [
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    MODELS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# AUDIO CONFIGURATION
# ==========================================================

SAMPLE_RATE = 16000


AUDIO_DURATION = 4


MAX_AUDIO_LENGTH = (
    SAMPLE_RATE *
    AUDIO_DURATION
)



# ==========================================================
# STFT CONFIGURATION
# ==========================================================

N_FFT = 512

HOP_LENGTH = 128

WIN_LENGTH = 512



# ==========================================================
# SPECTROGRAM CONFIGURATION
# ==========================================================

EPSILON = 1e-7



# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

BATCH_SIZE = 1

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0      # Start with 0 on Windows

GRAD_CLIP = 1.0

PATIENCE = 10

CHECKPOINT_INTERVAL = 5



# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

INPUT_CHANNELS = 1

OUTPUT_CHANNELS = 1

BASE_CHANNELS = 16

# Number of residual blocks in bottleneck
BOTTLENECK_BLOCKS = 2


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("==============================")
print("AI Speech Enhancer Config")
print("==============================")
print("Dataset:", DATASET_DIR)
print("Clean:", CLEAN_DIR)
print("Noisy:", NOISY_DIR)
print("Device:", DEVICE)
print("==============================")
