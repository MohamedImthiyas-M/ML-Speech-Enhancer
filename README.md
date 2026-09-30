# ML-Speech-Enhancer
AI-powered speech enhancement system using a Residual Attention U-Net to reduce noise and improve speech clarity.
# AI Speech Enhancer

AI-powered speech enhancement system designed to reduce background noise and improve speech clarity.

The project uses a Residual Attention U-Net architecture with time-frequency audio processing.

---

##  What does this project do?

The system takes a noisy speech recording:

    Noisy Audio
         ↓
        STFT
         ↓
    Spectrogram
         ↓
    Residual Attention U-Net
         ↓
    Enhanced Spectrogram
         ↓
       ISTFT
         ↓
    Enhanced Audio

Example:

    noisy_voice.wav
          ↓
    AI Speech Enhancer
          ↓
    enhanced_voice.wav

---

#  Quick Start

You can run this project on Windows, Linux, or macOS.

For training, an NVIDIA GPU is strongly recommended.

---

# 1. Requirements

You need:

- Python 3.10 or newer
- Git
- PyTorch
- torchaudio
- NumPy
- tqdm
- soundfile
- librosa

### GPU

A CUDA-compatible NVIDIA GPU is recommended for training.

CPU can be used for testing and inference, but training will be much slower.

---

# 2. Download the Project

Open a terminal or Command Prompt.

Run:

```bash
git clone https://github.com/MohamedImthiyas-M/ML-Speech-Enhancer.git
