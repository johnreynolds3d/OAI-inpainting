# OAI X-ray Inpainting Project

This repository contains implementations and experiments for X-ray image inpainting using multiple state-of-the-art deep learning approaches.

## Overview

The project implements and compares three different inpainting methods:
- **AOT-GAN**: Attention-based Outpainting Transformer for Generative Adversarial Networks
- **ICT**: Image Completion Transformer
- **RePaint**: Repaint-based diffusion model for inpainting

## Project Structure

```
OAI-inpainting/
├── AOT-GAN-for-Inpainting/    # AOT-GAN implementation
├── ICT/                       # Image Completion Transformer
├── RePaint/                   # RePaint diffusion model
├── classifier/                # Classification utilities
├── OAI_data/                  # X-ray dataset
└── output/                    # Generated results
```

## Dataset

The project uses the OAI (Osteoarthritis Initiative) X-ray dataset, which contains:
- Ground truth images in `OAI_data/gt_img/`
- Training, validation, and test splits
- Various mask types for inpainting evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OAI-inpainting
```

2. Install dependencies (create virtual environment recommended):
```bash
pip install torch torchvision
pip install -r requirements.txt  # if available
```

## Usage

### AOT-GAN
```bash
cd AOT-GAN-for-Inpainting
python src/train.py --config your_config.yml
```

### ICT
```bash
cd ICT
python run.py --config your_config.yml
```

### RePaint
```bash
cd RePaint
python test.py --config your_config.yml
```

## Results

Generated results are stored in the `output/` directory, organized by method and dataset.

## Contributing

Please ensure that:
- Large model checkpoints are not committed to git
- Experiment results are properly documented
- Code follows the existing style guidelines

## License

[Add your license information here]

## Acknowledgments

- Original AOT-GAN implementation
- ICT model authors
- RePaint diffusion model
- OAI dataset providers
