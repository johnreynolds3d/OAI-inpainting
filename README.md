# OAI X-ray Inpainting Project

This repository contains implementations and experiments for X-ray image inpainting using multiple state-of-the-art deep learning approaches on the Osteoarthritis Initiative (OAI) dataset.

## 🎯 Project Overview

The project implements and compares three different inpainting methods for X-ray image analysis:

- **AOT-GAN**: Attention-based Outpainting Transformer for Generative Adversarial Networks
- **ICT**: Image Completion Transformer
- **RePaint**: Repaint-based diffusion model for inpainting

## 🏗️ Project Structure

```
OAI-inpainting/
├── configs/                     # Configuration files
│   ├── aot-gan/
│   ├── ict/
│   └── repaint/
├── scripts/                     # Unified training/testing scripts
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── setup_data.py           # Data setup script
├── src/                         # Core source code
│   ├── paths.py
│   ├── config.py
│   ├── data.py
│   ├── experiment_tracking.py
│   └── data_versioning.py
├── data/                        # Data management
│   ├── oai/                     # OAI dataset
│   └── pretrained/              # Pretrained models
├── results/                     # Results and metrics
│   ├── metrics/
│   ├── plots/
│   └── logs/
├── docs/                        # Documentation
├── models/                      # Model implementations
│   ├── aot-gan/                # AOT-GAN implementation
│   ├── ict/                    # ICT implementation
│   ├── repaint/                # RePaint implementation
│   └── classifier/             # Classification utilities
└── results/                     # Generated results
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for GPU access)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnreynolds3d/OAI-inpainting/blob/master/notebooks/OAI_Inpainting_Colab.ipynb)

**Direct Colab Link**: https://colab.research.google.com/github/johnreynolds3d/OAI-inpainting/blob/master/notebooks/OAI_Inpainting_Colab.ipynb

#### Google Colab Setup Steps:

1. **Upload Data to Google Drive**:
   - Upload the entire `OAI_untracked/` directory to:
     ```
     /content/drive/MyDrive/Colab Notebooks/OAI_untracked/
     ```
   - This includes all 539 OAI images and pretrained models (~16GB total)

2. **Run the Colab Notebook**:
   - Click the "Open in Colab" button above
   - The notebook will automatically:
     - Mount Google Drive
     - Detect your data structure
     - Copy data to local storage for performance
     - Generate train/valid/test splits
     - Create masks and edge maps

3. **Verify Setup**:
   ```bash
   # Check image count (should show 540: 539 files + 1 for total line)
   ls -la "/content/drive/MyDrive/Colab Notebooks/OAI_untracked/data/oai/img/" | wc -l
   ```

4. **Start Training**:
   - Run the pipeline cells in order
   - All models will be trained and evaluated automatically

#### Colab Data Structure:
```
/content/drive/MyDrive/Colab Notebooks/OAI_untracked/
├── data/
│   ├── oai/
│   │   └── img/                    # 539 OAI X-ray images
│   │       ├── 6.C.1_*.png        # Osteoporotic cases
│   │       └── 6.E.1_*.png        # Normal cases
│   └── pretrained/                 # Pretrained models
│       ├── aot-gan/               # AOT-GAN models
│       ├── ict/                   # ICT models
│       └── repaint/               # RePaint models
└── README.md                      # Data documentation
```

#### Important Colab Notes:
- **Upload time**: May take 30-60 minutes depending on connection
- **Storage**: Ensure sufficient Google Drive space (16GB+)
- **Permissions**: Make sure Colab can access the files
- **Large files**: Some model files are >1GB

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/johnreynolds3d/OAI-inpainting.git
cd OAI-inpainting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev,ml]"

# Install pre-commit hooks
pre-commit install
```

### Data Setup

The project requires additional untracked data files (images and pretrained models). Use the automated setup script:

```bash
# Auto-detect and set up data (recommended)
python scripts/setup_data.py

# Preview what would be copied (dry run)
python scripts/setup_data.py --dry-run

# Specify custom source directory
python scripts/setup_data.py --source-dir /path/to/OAI_untracked

# Force overwrite existing files
python scripts/setup_data.py --force
```

**Required Untracked Data Structure:**
```
OAI_untracked/                    # Can be named differently
├── data/
│   ├── oai/
│   │   └── img/                  # OAI X-ray images (required)
│   │       ├── 6.C.1_*.png      # Osteoporotic cases
│   │       └── 6.E.1_*.png      # Normal cases
│   └── pretrained/               # Pretrained models (optional)
│       ├── aot-gan/             # AOT-GAN pretrained models
│       │   ├── celebahq/        # CelebA-HQ models
│       │   └── places2/         # Places2 models
│       ├── ict/                 # ICT pretrained models
│       │   ├── Transformer/     # Transformer models
│       │   └── Upsample/        # Upsampler models
│       └── repaint/             # RePaint pretrained models
│           ├── 256x256_classifier.pt
│           ├── 256x256_diffusion.pt
│           ├── celeba256_250000.pt
│           └── places256_300000.pt
└── README.md                    # Data documentation
```

**Data Requirements:**
- **Total size**: ~16GB
- **Total files**: 559
- **OAI Images**: 539 PNG files (~11.6 MB) - Required
- **AOT-GAN Models**: 6 files (~2GB) - Optional
- **ICT Models**: 9 files (~8.3GB) - Optional
- **RePaint Models**: 4 files (~6.4GB) - Optional

**Setup Script Features:**
- Auto-detects untracked data directories in common locations
- Validates data structure and estimates sizes
- Supports dry-run mode for previewing operations
- Handles missing optional components gracefully
- Provides detailed progress tracking and error messages

### Training

```bash
# Train AOT-GAN
python scripts/train.py --model aot-gan --config configs/aot-gan/oai_config.yml

# Train ICT
python scripts/train.py --model ict --config configs/ict/oai_config.yml

# Note: RePaint is inference-only
```

### Testing

```bash
# Test all models
python scripts/test.py --model aot-gan --config configs/aot-gan/oai_config.yml
python scripts/test.py --model ict --config configs/ict/oai_config.yml
python scripts/test.py --model repaint --config configs/repaint/oai_config.yml
```

### Evaluation

```bash
# Evaluate all models
python scripts/evaluate.py --models aot-gan ict repaint --subset subset_4
```

## 📊 Dataset

The project uses the OAI (Osteoarthritis Initiative) X-ray dataset:

- **Ground truth images**: `data/oai/img/` (populated by setup script)
- **Balanced splits**: Train/validation/test with class balance
- **Multiple mask types**: Random squares, inverted masks, edge maps
- **Evaluation subset**: `subset_4` with 2 osteoporotic + 2 normal images

### Dataset Generation

After running the setup script, generate the dataset splits:

```bash
cd data/oai
python split.py  # Generates balanced splits and masks
```

### Troubleshooting Data Setup

#### Local Setup Issues

**"Could not find untracked data directory"**
- Use `--source-dir` to specify the exact path: `python scripts/setup_data.py --source-dir /path/to/OAI_untracked`

**"Some required components are missing"**
- Ensure your untracked data has the correct structure (see Data Setup section above)

**"Insufficient disk space"**
- Check available space: `df -h`
- Clean up unnecessary files or use `--force` to proceed anyway

**"Permission denied"**
- Fix permissions: `chmod -R 755 .`

#### Google Colab Issues

**"OAI data not found in Google Drive"**
- Verify data is uploaded to: `/content/drive/MyDrive/Colab Notebooks/OAI_untracked/`
- Check file permissions in Google Drive
- Ensure directory structure matches exactly

**"Upload failed or incomplete"**
- Large files (>1GB) may timeout - try uploading in smaller batches
- Check Google Drive storage space (need 16GB+)
- Verify internet connection stability

**"Colab can't access files"**
- Check Google Drive sharing permissions
- Ensure files are not in "Shared with me" folder
- Try moving files to "My Drive" root directory

**"subset_4 data not found"**
- Run the data splitting cell in the Colab notebook
- This generates the test subset automatically
- Check that `data/oai/split.py` runs successfully

#### General Data Issues

**"Image count mismatch"**
- Expected: 539 PNG files in `data/oai/img/`
- Check: `ls -la data/oai/img/ | wc -l` (should show 540: 539 files + 1 for total line)
- Verify file naming: `6.C.1_*.png` (osteoporotic), `6.E.1_*.png` (normal)

**"Model files missing"**
- AOT-GAN: 6 files (~2GB) in `pretrained/aot-gan/`
- ICT: 9 files (~8.3GB) in `pretrained/ict/`
- RePaint: 4 files (~6.4GB) in `pretrained/repaint/`
- Missing models will use random initialization (reduced performance)

## 🔧 Configuration

All models use platform-agnostic YAML configuration files:

- **AOT-GAN**: `configs/aot-gan/oai_config.yml`
- **ICT**: `configs/ict/oai_config.yml`
- **RePaint**: `configs/repaint/oai_config.yml`

### Key Configuration Options

```yaml
# Example: AOT-GAN configuration
data:
  train_images: "./data/oai/train/img"
  train_masks: "./data/oai/train/mask"

model:
  name: "aotgan"
  block_num: 8
  gan_type: "smgan"

training:
  batch_size: 8
  image_size: 512
  lr_g: 1e-4
  lr_d: 1e-4
```

## 📈 Results

Generated results are organized by model and dataset:

```
results/
├── AOT-GAN/OAI/
├── ICT/OAI/
└── RePaint/OAI/
```

### Evaluation Metrics

- **Inpainting Quality**: PSNR, SSIM
- **Classification Performance**: Accuracy, Precision, Recall, F1-score
- **Statistical Analysis**: Comprehensive comparison reports

## 🧪 Experiments

### Reproducibility

- **Platform-agnostic**: Works on Linux, macOS, Windows
- **Version control**: All configurations tracked in git
- **Pre-commit hooks**: Automatic code formatting and linting with Ruff
- **Documentation**: Comprehensive guides and tutorials

### Comparison Studies

1. **Pretrained vs OAI-trained**: Compare pretrained models with OAI-specific training
2. **Cross-dataset evaluation**: Test models on different X-ray datasets
3. **Ablation studies**: Analyze individual components
4. **Statistical significance**: Rigorous statistical testing

## 📚 Documentation

- **[Installation Guide](docs/installation.md)**: Setup instructions
- **[Usage Guide](docs/usage.md)**: Detailed usage instructions
- **[API Documentation](docs/api.md)**: Code API reference
- **[Experiments](docs/experiments.md)**: Experimental procedures
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## 🔬 Research Applications

This project enables research in:

- **Medical Image Analysis**: X-ray inpainting for diagnostic purposes
- **Osteoporosis Detection**: Classification of bone density from X-ray images
- **Inpainting Quality Assessment**: Comparison of different inpainting approaches
- **Transfer Learning**: Adapting pretrained models to medical imaging

## 🛠️ Development

### Code Quality

This project uses modern Python development tools:

- **Ruff**: Fast Python linter and formatter (replaces Black + flake8)
- **MyPy**: Static type checking
- **Pre-commit**: Automated code quality checks
- **Pytest**: Testing framework with coverage reporting

### Development Commands

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues automatically
ruff check . --fix

# Run type checking
mypy src/ scripts/

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Pre-commit Hooks

The project includes pre-commit hooks that automatically:
- Format code with Ruff
- Check for linting issues
- Validate YAML/JSON files
- Check for large files
- Ensure proper line endings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This project includes third-party models and code with their own licenses:
- **AOT-GAN**: Apache 2.0 License
- **RePaint**: MIT License + CC BY-NC-SA 4.0 (Huawei)
- **ICT**: Research use only

Please review the individual license files in the `models/` directory for specific terms.

## 🔬 Research Use

This project is intended for academic research and educational purposes. The OAI dataset is used under appropriate research agreements. Commercial use may require additional permissions from third-party licensors.

## 🙏 Acknowledgments

- Original AOT-GAN implementation
- ICT model authors
- RePaint implementation
- OAI dataset providers
- Open source community

## 📞 Support

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the [docs/](docs/) directory
- **Tutorials**: See [docs/](docs/)

## 🔄 Updates

- **v1.0.0**: Initial release with three inpainting models
- **v1.1.0**: Added unified scripts and improved documentation
- **v1.2.0**: Platform-agnostic configuration system
- **v1.3.0**: Comprehensive evaluation framework

---

**Note**: This project is designed for research purposes. Ensure compliance with medical imaging regulations and ethical guidelines when using X-ray data.
