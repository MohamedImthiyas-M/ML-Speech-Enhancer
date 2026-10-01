import torch
import torchaudio
import torch.nn.functional as F

from config import (
    SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    EPSILON
)


WINDOW = torch.hann_window(WIN_LENGTH)


# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(path):

    waveform, sample_rate = torchaudio.load(str(path))

    # Convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(
            waveform,
            dim=0,
            keepdim=True
        )

    # Resample if required
    if sample_rate != SAMPLE_RATE:

        resampler = torchaudio.transforms.Resample(
            sample_rate,
            SAMPLE_RATE
        )

        waveform = resampler(waveform)

    return waveform


# ==========================================================
# FIX AUDIO LENGTH
# ==========================================================

def fix_length(waveform):

    current_length = waveform.shape[-1]

    if current_length < MAX_AUDIO_LENGTH:

        padding = MAX_AUDIO_LENGTH - current_length

        waveform = F.pad(
            waveform,
            (0, padding)
        )

    elif current_length > MAX_AUDIO_LENGTH:

        waveform = waveform[
            :,
            :MAX_AUDIO_LENGTH
        ]

    return waveform


# ==========================================================
# NORMALIZE AUDIO
# ==========================================================

def normalize_audio(waveform):

    peak = torch.max(
        torch.abs(waveform)
    )

    if peak > EPSILON:

        waveform = waveform / peak

    return waveform


# ==========================================================
# WAVEFORM -> LOG SPECTROGRAM
# ==========================================================

def waveform_to_spectrogram(waveform):

    window = WINDOW.to(
        waveform.device
    )

    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(stft)

    phase = torch.angle(stft)

    log_spectrogram = torch.log(
        magnitude + EPSILON
    )

    return log_spectrogram, phase


# ==========================================================
# NORMALIZE SPECTROGRAM
# ==========================================================

def normalize_spectrogram(
    spectrogram,
    mean=None,
    std=None
):

    if mean is None:
        mean = spectrogram.mean()

    if std is None:
        std = spectrogram.std()

    normalized = (
        spectrogram - mean
    ) / (
        std + EPSILON
    )

    return normalized, mean, std


# ==========================================================
# PREPROCESS AUDIO PAIR
# ==========================================================

def preprocess_audio(path):

    waveform = load_audio(path)

    waveform = fix_length(
        waveform
    )

    waveform = normalize_audio(
        waveform
    )

    spectrogram, phase = waveform_to_spectrogram(
        waveform
    )

    spectrogram, mean, std = normalize_spectrogram(
        spectrogram
    )

    return (
        spectrogram,
        phase,
        mean,
        std
    )


# ==========================================================
# PREPROCESS TRAINING PAIR
# ==========================================================

def preprocess_training_pair(
    noisy_path,
    clean_path
):

    # ------------------------------------------------------
    # NOISY
    # ------------------------------------------------------

    noisy_waveform = load_audio(
        noisy_path
    )

    noisy_waveform = fix_length(
        noisy_waveform
    )

    noisy_waveform = normalize_audio(
        noisy_waveform
    )

    noisy_log, noisy_phase = waveform_to_spectrogram(
        noisy_waveform
    )

    # ------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------

    clean_waveform = load_audio(
        clean_path
    )

    clean_waveform = fix_length(
        clean_waveform
    )

    clean_waveform = normalize_audio(
        clean_waveform
    )

    clean_log, _ = waveform_to_spectrogram(
        clean_waveform
    )

    # ------------------------------------------------------
    # IMPORTANT
    #
    # Use the NOISY statistics for BOTH.
    # ------------------------------------------------------

    noisy_mean = noisy_log.mean()

    noisy_std = noisy_log.std()

    noisy_normalized = (
        noisy_log - noisy_mean
    ) / (
        noisy_std + EPSILON
    )

    clean_normalized = (
        clean_log - noisy_mean
    ) / (
        noisy_std + EPSILON
    )

    return (
        noisy_normalized,
        clean_normalized,
        noisy_phase,
        noisy_mean,
        noisy_std
    )


# ==========================================================
# SPECTROGRAM -> WAVEFORM
# ==========================================================

def spectrogram_to_waveform(
    normalized_spectrogram,
    mean,
    std,
    phase
):

    # ------------------------------------------------------
    # Denormalize
    # ------------------------------------------------------

    log_magnitude = (
        normalized_spectrogram * std
    ) + mean

    # ------------------------------------------------------
    # Convert log magnitude -> magnitude
    # ------------------------------------------------------

    magnitude = torch.exp(
        torch.clamp(
            log_magnitude,
            min=-20.0,
            max=20.0
        )
    )

    # ------------------------------------------------------
    # Complex spectrogram
    # ------------------------------------------------------

    complex_spec = (
        magnitude *
        torch.exp(
            1j * phase
        )
    )

    # ------------------------------------------------------
    # ISTFT
    # ------------------------------------------------------

    window = WINDOW.to(
        magnitude.device
    )

    waveform = torch.istft(
        complex_spec,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=MAX_AUDIO_LENGTH
    )

    return waveform


# ==========================================================
# SAVE-SAFE AUDIO NORMALIZATION
# ==========================================================

def normalize_output_waveform(waveform):

    peak = torch.max(
        torch.abs(waveform)
    )

    if peak > EPSILON:

        waveform = waveform / peak

    return waveform