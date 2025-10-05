# Installation Guide

This guide provides step-by-step instructions for setting up the OAI inpainting project on different platforms.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

```bash
# Update package manager
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install CUDA (if using GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install CUDA (if using GPU with compatible hardware)
brew install cuda
```

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/)
2. Install CUDA from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
3. Install Git from [git-scm.com](https://git-scm.com/download/win)

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd OAI-inpainting
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements-dev.txt
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

### 5. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test project imports
python -c "from utils.paths import get_project_root; print(f'Project root: {get_project_root()}')"
```

## Data Setup

### 1. Download OAI Dataset

Place your OAI X-ray images in the `data/oai/gt_img/` directory.

### 2. Generate Dataset Splits

```bash
cd data/oai
python split.py
```

### 3. Download Pretrained Models

Place pretrained model weights in the appropriate directories:

- AOT-GAN: `data/pretrained/aot-gan/`
- ICT: `data/pretrained/ict/`
- RePaint: `data/pretrained/repaint/`

## Configuration

### 1. Update Configuration Files

Edit the configuration files in `configs/` to match your system:

```bash
# Edit AOT-GAN config
nano configs/aot-gan/oai_config.yml

# Edit ICT config
nano configs/ict/oai_config.yml

# Edit RePaint config
nano configs/repaint/oai_config.yml
```

### 2. Set Environment Variables

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export CUDA_VISIBLE_DEVICES=0  # Set GPU device
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Add project to Python path
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration files
   - Use gradient checkpointing
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Import Errors**
   - Ensure virtual environment is activated
   - Check PYTHONPATH is set correctly
   - Reinstall dependencies: `pip install -r requirements-dev.txt`

3. **Permission Errors**
   - Check file permissions
   - Use `sudo` for system-wide installations (not recommended)

4. **Path Issues**
   - Use absolute paths in configuration files
   - Check path separators (use `/` on all platforms)

### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Review error logs in `results/logs/`
- Create an issue on the project repository

## Next Steps

After successful installation:

1. Read the [usage guide](usage.md)
2. Start with [experimental procedures](experiments.md)

## Platform Compatibility

This project has been tested on:

- Ubuntu 20.04/22.04 LTS
- macOS 12/13 (Intel and Apple Silicon)
- Windows 10/11
- Python 3.8, 3.9, 3.10, 3.11
- PyTorch 1.12, 1.13, 2.0, 2.1
- CUDA 11.8, 12.0, 12.1
