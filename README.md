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
```

Then enter the project folder:
```bash
cd ML-Speech-Enhancer
```

# 3. Create a Virtual Environment

This keeps the project's Python packages separate from your other projects.

Run:
```bash
python -m venv .venv
```
**Windows**

Activate the environment:
```bash
.venv\Scripts\activate
```
After activation, you should see something similar to:

(.venv) C:\...\ML-Speech-Enhancer>

**Linux / macOS**

Use:
```bash
source .venv/bin/activate
```

# 4. Install Dependencies

Upgrade pip:
```bash
python -m pip install --upgrade pip
```

Then install the project dependencies:
```bash
pip install -r requirements.txt
```

# 5. Check PyTorch and GPU

Run:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If you have a correctly configured NVIDIA GPU, you should see:

**CUDA available: True**

If you see:

**CUDA available: False**

the project can still run on the CPU, but training will be much slower.

# 6. Dataset Setup

The model is trained using paired noisy and clean speech.

Create the following folder structure:
```bash
dataset/
│
├── clean_speech/
│   ├── 1.wav
│   ├── 2.wav
│   ├── 3.wav
│   └── ...
│
└── noisy_speech/
    ├── 1.wav
    ├── 2.wav
    ├── 3.wav
    └── ...
```

The noisy and clean files must have matching filenames.

For example:
```bash
clean_speech/1.wav
noisy_speech/1.wav
```
are one training pair.

Another pair:
```bash
clean_speech/2.wav
noisy_speech/2.wav
```

## Important

The dataset is not included in this GitHub repository.

Download or prepare a speech dataset separately and place the files into:

**dataset/clean_speech/**

and:

**dataset/noisy_speech/**

# 7. Test the Dataset

Before starting training, check whether the dataset is detected correctly.

Run:
```bash
python test_dataset.py
```
You should see something similar to:

Total audio pairs: 1217

The exact number depends on the dataset you use.

If the audio pairs are detected successfully, continue.

# 8. Test the Model

Run:
```bash
python test_unet.py
```
This checks whether the U-Net model can process a spectrogram correctly.

If the test completes without an error, the model architecture is working.

# 9. Test the Dataset Shape

Run:
```bash
python test_shape.py
```
You should see something similar to:

Noisy : torch.Size([1, 257, ...])
Clean : torch.Size([1, 257, ...])
Phase : torch.Size([257, ...])

The exact dimensions may vary depending on the audio configuration.

# 10. Train the AI Model

Once the dataset and model tests work, start training:
```bash
python train.py
```
The training process will:

Load noisy speech
Load clean speech
Convert audio into spectrograms
Train the Residual Attention U-Net
Calculate the training loss
Use GPU acceleration when available
Save model checkpoints

You should see something similar to:
```bash
Total audio pairs: 1217

Train Epoch 1/100:
```
Training time depends on:

GPU
CPU
Dataset size
Number of epochs
Audio duration
Model configuration

# 11. Model Checkpoints

During training, model checkpoints are saved in:

checkpoints/

The checkpoint directory is created automatically when needed.

# 12. Enhancing an Audio File

After training, use:
```bash
python inference.py
```
The inference system processes the audio using the trained model.

The pipeline is:
```bash
Input Audio
     ↓
Audio Loading
     ↓
STFT
     ↓
Spectrogram
     ↓
Trained U-Net
     ↓
Enhanced Spectrogram
     ↓
Inverse STFT
     ↓
Enhanced Audio
```
The supported audio formats depend on the implementation of the audio conversion module and installed audio libraries.

# 13. Output Files

Enhanced audio files are saved in:

outputs/

For example:

outputs/
└── enhanced_audio.wav

The outputs directory is created automatically when needed.

# 14. Graphical User Interface

After the GUI is completed, start the application using:
```bash
python app.py
```
The application will provide a user-friendly interface for selecting an audio file and processing it with the trained model.

# 🧠 Model Architecture

This project uses a Residual Attention U-Net architecture.

The complete processing pipeline is:
```bash
Noisy Audio
     ↓
STFT
     ↓
Log Magnitude Spectrogram
     ↓
Encoder
     ↓
Residual Blocks
     ↓
Attention Gates
     ↓
Bottleneck
     ↓
Attention Gates
     ↓
Decoder
     ↓
Skip Connections
     ↓
Predicted Clean Spectrogram
     ↓
Inverse STFT
     ↓
Enhanced Audio
```

# 📁 Project Structure

```bash
ML-Speech-Enhancer/
│
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── attention.py
│   ├── blocks.py
│   └── loss.py
│
├── dataset/
│   ├── clean_speech/
│   └── noisy_speech/
│
├── checkpoints/
│
├── outputs/
│
├── temp/
│
├── config.py
├── preprocessing.py
├── dataset_loader.py
├── train.py
├── inference.py
├── audio_converter.py
├── app.py
│
├── test_attention.py
├── test_blocks.py
├── test_dataset.py
├── test_loss.py
├── test_shape.py
├── test_unet.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

# ⚙️ Configuration

Most project settings are controlled from:
```bash
config.py
```
Important settings include:
```bash
SAMPLE_RATE = 16000

AUDIO_DURATION = 2

N_FFT = 512

HOP_LENGTH = 128

WIN_LENGTH = 512

BATCH_SIZE = 1

EPOCHS = 100

LEARNING_RATE = 1e-4

BASE_CHANNELS = 16
```
These settings can be changed depending on your hardware and experiment.

# 💻 Recommended Hardware
## Training

Recommended:

NVIDIA GPU
4 GB or more VRAM
8 GB or more system RAM
SSD storage

A GPU with more VRAM can allow larger batch sizes and longer audio segments.

## Inference

Inference can run on:

**NVIDIA GPU**
**CPU**

CPU inference will generally be slower.

# 🛠️ Troubleshooting
CUDA Out Of Memory

If you get:

torch.OutOfMemoryError: CUDA out of memory

try reducing:
```bash
BATCH_SIZE = 1
```
and:
```bash
AUDIO_DURATION = 2
```

If necessary, reduce:
```bash
BASE_CHANNELS = 8
```
This can help when using a GPU with limited VRAM.

**CUDA is False**

If:

CUDA available: False

appears even though you have an NVIDIA GPU:

Check your NVIDIA driver.
Check your Python version.
Check your PyTorch installation.
Install the appropriate PyTorch build for your system.

The project can still run on CPU.

**Dataset Shows 0 Pairs**

Check that your files are arranged correctly:
```bash
dataset/
│
├── clean_speech/
│   ├── 1.wav
│   └── 2.wav
│
└── noisy_speech/
    ├── 1.wav
    └── 2.wav
```
The filenames must match.

**For example:**

clean_speech/1.wav
noisy_speech/1.wav

is a valid pair.

But:

clean_speech/1.wav
noisy_speech/noisy_1.wav

will not be recognized as a matching pair by the current dataset loader.

# Training Pipeline

The complete training process is:
```bash
Audio Dataset
     ↓
Audio Loading
     ↓
Mono Conversion
     ↓
Resampling
     ↓
Normalization
     ↓
STFT
     ↓
Log Spectrogram
     ↓
Residual Attention U-Net
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Optimizer
     ↓
Checkpoint
```

# 📌 Important Notes

This project is intended for:

Research
Learning
Experimentation
AI/ML development
Speech enhancement development

The quality of enhancement depends heavily on:

Training dataset
Noise types
Dataset size
Audio quality
Training duration
Model configuration
Hardware

A trained model should be evaluated using appropriate speech-quality metrics and listening tests before being considered production-ready.
