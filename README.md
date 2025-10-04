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
│   └── evaluate.py
├── utils/                       # Shared utilities
│   ├── paths.py
│   ├── config.py
│   └── data.py
├── data/                        # Data management
│   ├── oai/                     # OAI dataset
│   └── pretrained/              # Pretrained models
├── results/                     # Results and metrics
│   ├── metrics/
│   ├── plots/
│   └── logs/
├── notebooks/                   # Analysis notebooks
│   ├── data_analysis/
│   ├── model_comparison/
│   └── visualization/
├── docs/                        # Documentation
├── AOT-GAN-for-Inpainting/      # AOT-GAN implementation
├── ICT/                         # ICT implementation
├── RePaint/                     # RePaint implementation
├── classifier/                  # Classification utilities
└── output/                      # Generated results
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd OAI-inpainting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

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

- **Ground truth images**: `data/oai/gt_img/`
- **Balanced splits**: Train/validation/test with class balance
- **Multiple mask types**: Random squares, inverted masks, edge maps
- **Evaluation subset**: `subset_4` with 2 osteoporotic + 2 normal images

### Dataset Generation

```bash
cd data/oai
python split.py  # Generates balanced splits and masks
```

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
output/
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
- **Pre-commit hooks**: Automatic code formatting with Black
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [CONTRIBUTING.md](docs/contributing.md) for detailed guidelines.

## 📄 License

[Add your license information here]

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
- **Tutorials**: See [notebooks/tutorials/](notebooks/tutorials/)

## 🔄 Updates

- **v1.0.0**: Initial release with three inpainting models
- **v1.1.0**: Added unified scripts and improved documentation
- **v1.2.0**: Platform-agnostic configuration system
- **v1.3.0**: Comprehensive evaluation framework

---

**Note**: This project is designed for research purposes. Ensure compliance with medical imaging regulations and ethical guidelines when using X-ray data.
