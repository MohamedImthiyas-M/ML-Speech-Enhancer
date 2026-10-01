from pathlib import Path

import torch
import torchaudio

from config import (
    DEVICE,
    OUTPUT_DIR,
    CHECKPOINT_DIR
)

from models.unet import ResidualAttentionUNet

from preprocessing import (
    preprocess_audio,
    spectrogram_to_waveform,
    normalize_output_waveform
)


# ==========================================================
# CHECKPOINT NAMES
# ==========================================================

CHECKPOINT_NAMES = [

    "best.pt",

    "latest.pt",

    "best_model.pth",

    "last_model.pth",

    "best_model.pt",

    "last_model.pt"
]


# ==========================================================
# FIND CHECKPOINT
# ==========================================================

def find_checkpoint():

    for name in CHECKPOINT_NAMES:

        path = (
            CHECKPOINT_DIR /
            name
        )

        if path.exists():

            return path

    return None


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    checkpoint_path = find_checkpoint()

    if checkpoint_path is None:

        raise FileNotFoundError(
            "No model checkpoint found.\n"
            f"Expected checkpoint inside:\n"
            f"{CHECKPOINT_DIR}"
        )

    print()
    print(
        "Loading checkpoint:"
    )

    print(
        checkpoint_path
    )

    # ------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # ------------------------------------------------------
    # Extract model weights
    # ------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model" in checkpoint:

            state_dict = checkpoint[
                "model"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # ------------------------------------------------------
    # Check weights
    # ------------------------------------------------------

    for name, tensor in state_dict.items():

        if torch.is_tensor(tensor):

            if not torch.isfinite(
                tensor
            ).all():

                raise RuntimeError(
                    f"Invalid values found in "
                    f"model parameter: {name}"
                )

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------

    model = ResidualAttentionUNet()

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    # ------------------------------------------------------
    # Checkpoint information
    # ------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "loss" in checkpoint:

            print(
                "Checkpoint loss:",
                checkpoint["loss"]
            )

        if "learning_rate" in checkpoint:

            print(
                "Checkpoint learning rate:",
                checkpoint["learning_rate"]
            )

    return model


# ==========================================================
# ENHANCE AUDIO
# ==========================================================

@torch.no_grad()
def enhance_audio(
    model,
    input_path
):

    input_path = Path(
        input_path
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{input_path}"
        )

    print()
    print(
        "Input:",
        input_path
    )

    print(
        "Running AI enhancement..."
    )

    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------

    (
        noisy_spectrogram,
        noisy_phase,
        noisy_mean,
        noisy_std
    ) = preprocess_audio(
        input_path
    )

    # ------------------------------------------------------
    # Model input
    # ------------------------------------------------------

    model_input = (
        noisy_spectrogram
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = model(
        model_input
    )

    # ------------------------------------------------------
    # Remove batch/channel dimensions
    # ------------------------------------------------------

    prediction = (
        prediction
        .squeeze(0)
        .squeeze(0)
    )

    # ------------------------------------------------------
    # Check prediction
    # ------------------------------------------------------

    if not torch.isfinite(
        prediction
    ).all():

        raise RuntimeError(
            "Model produced NaN or Inf values."
        )

    # ------------------------------------------------------
    # Move supporting tensors
    # ------------------------------------------------------

    prediction = prediction.to(
        DEVICE
    )

    noisy_phase = noisy_phase.to(
        DEVICE
    )

    noisy_mean = noisy_mean.to(
        DEVICE
    )

    noisy_std = noisy_std.to(
        DEVICE
    )

    # ------------------------------------------------------
    # Convert prediction back to waveform
    #
    # THIS IS THE IMPORTANT FIX:
    #
    # normalized prediction
    #       ↓
    # denormalize
    #       ↓
    # exp
    #       ↓
    # ISTFT
    # ------------------------------------------------------

    waveform = spectrogram_to_waveform(
        prediction,
        noisy_mean,
        noisy_std,
        noisy_phase
    )

    # ------------------------------------------------------
    # Normalize final audio
    # ------------------------------------------------------

    waveform = normalize_output_waveform(
        waveform
    )

    waveform = waveform.unsqueeze(
        0
    )

    # ------------------------------------------------------
    # Output path
    # ------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        f"enhanced_{input_path.stem}.wav"
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    torchaudio.save(
        str(output_path),
        waveform.cpu(),
        16000
    )

    print()
    print(
        "Enhancement completed."
    )

    print(
        "Output:",
        output_path
    )

    return output_path


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "======================================"
    )

    print(
        "       AI SPEECH ENHANCER V3"
    )

    print(
        "======================================"
    )

    model = load_model()

    print()
    print(
        "Enter the path of the noisy audio file."
    )

    audio_path = input(
        "Audio path: "
    ).strip().strip('"')

    enhance_audio(
        model,
        audio_path
    )


if __name__ == "__main__":

    main()