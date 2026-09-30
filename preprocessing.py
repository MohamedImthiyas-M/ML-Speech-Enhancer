"""
==========================================================
preprocessing.py

Audio preprocessing pipeline for
Residual Attention U-Net Speech Enhancement

Pipeline:

Audio File
    |
    ↓
Load Audio
    |
    ↓
Convert Mono
    |
    ↓
Resample
    |
    ↓
Fix Length
    |
    ↓
Normalize Waveform
    |
    ↓
STFT
    |
    ↓
Log Magnitude Spectrogram
    |
    ↓
Normalize Spectrogram


Functions:

load_audio()
fix_length()
normalize_audio()
waveform_to_spectrogram()
normalize_spectrogram()
get_phase()
spectrogram_to_waveform()

==========================================================
"""


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



# ==========================================================
# HANN WINDOW
# ==========================================================

WINDOW = torch.hann_window(
    WIN_LENGTH
)



# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(path):
    """
    Load audio file

    Output:
        waveform:
        Shape -> [1, samples]

    """

    waveform, sample_rate = torchaudio.load(
        str(path)
    )


    # -------------------------------
    # Convert stereo to mono
    # -------------------------------

    if waveform.shape[0] > 1:

        waveform = torch.mean(
            waveform,
            dim=0,
            keepdim=True
        )


    # -------------------------------
    # Resample
    # -------------------------------

    if sample_rate != SAMPLE_RATE:


        resampler = torchaudio.transforms.Resample(
            sample_rate,
            SAMPLE_RATE
        )


        waveform = resampler(
            waveform
        )


    return waveform





# ==========================================================
# FIX AUDIO LENGTH
# ==========================================================

def fix_length(waveform):
    """
    Make every audio same length

    Short audio:
        zero padding

    Long audio:
        trimming

    """

    current_length = waveform.shape[-1]


    if current_length < MAX_AUDIO_LENGTH:


        padding = (
            MAX_AUDIO_LENGTH -
            current_length
        )


        waveform = F.pad(
            waveform,
            (
                0,
                padding
            )
        )


    elif current_length > MAX_AUDIO_LENGTH:


        waveform = waveform[
            :,
            :MAX_AUDIO_LENGTH
        ]


    return waveform





# ==========================================================
# NORMALIZE WAVEFORM
# ==========================================================

def normalize_audio(waveform):
    """
    Normalize waveform amplitude
    """

    peak = torch.max(
        torch.abs(waveform)
    )


    if peak > 0:

        waveform = (
            waveform /
            peak
        )


    return waveform





# ==========================================================
# WAVEFORM -> SPECTROGRAM
# ==========================================================

def waveform_to_spectrogram(
        waveform
):
    """
    Convert waveform into
    log magnitude spectrogram

    Output:

    [frequency, time]

    """


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


    magnitude = torch.abs(
        stft
    )

    # Save phase for reconstruction
    phase = torch.angle(
        stft
    )  

    # Log compression
    log_spectrogram = torch.log(
        magnitude + EPSILON
    )

    return log_spectrogram, phase





# ==========================================================
# NORMALIZE SPECTROGRAM
# ==========================================================

def normalize_spectrogram(
        spectrogram
):
    """
    Standard score normalization

    Helps neural network training
    """

    mean = spectrogram.mean()

    std = spectrogram.std()


    spectrogram = (
        spectrogram - mean
    ) / (
        std + EPSILON
    )


    return spectrogram


# ==========================================================
# SPECTROGRAM -> WAVEFORM
# ==========================================================

def spectrogram_to_waveform(
        spectrogram,
        phase
):
    """
    Convert predicted spectrogram
    back into waveform

    """


    window = WINDOW.to(
        spectrogram.device
    )


    # Reverse normalization
    # handled outside model


    magnitude = torch.exp(
        spectrogram
    )


    complex_spec = (
        magnitude *
        torch.exp(
            1j * phase
        )
    )


    waveform = torch.istft(

        complex_spec,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        win_length=WIN_LENGTH,

        window=window

    )


    return waveform





# ==========================================================
# COMPLETE PREPROCESS FUNCTION
# ==========================================================

def preprocess_audio(path):
    """
    Complete pipeline

    Input:
        audio path


    Output:

        normalized spectrogram
        phase

    """


    waveform = load_audio(
        path
    )


    waveform = fix_length(
        waveform
    )


    waveform = normalize_audio(
        waveform
    )


    spectrogram, phase = waveform_to_spectrogram(
        waveform
    )

    spectrogram = normalize_spectrogram(
        spectrogram
    )

    return (
        spectrogram,
        phase
    )
