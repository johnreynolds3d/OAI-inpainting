# Usage Guide

This guide explains how to use the OAI inpainting project for training, testing, and evaluation.

## Quick Start

### 1. Training a Model

```bash
# Train AOT-GAN
python scripts/train.py --model aot-gan --config configs/aot-gan/oai_config.yml

# Train ICT
python scripts/train.py --model ict --config configs/ict/oai_config.yml

# Note: RePaint is inference-only
python scripts/train.py --model repaint --config configs/repaint/oai_config.yml
```

### 2. Testing a Model

```bash
# Test AOT-GAN
python scripts/test.py --model aot-gan --config configs/aot-gan/oai_config.yml

# Test ICT
python scripts/test.py --model ict --config configs/ict/oai_config.yml

# Test RePaint
python scripts/test.py --model repaint --config configs/repaint/oai_config.yml
```

### 3. Evaluating Results

```bash
# Evaluate all models
python scripts/evaluate.py --models aot-gan ict repaint --subset subset_4

# Evaluate specific models
python scripts/evaluate.py --models aot-gan ict --subset test
```

## Detailed Usage

### Training

#### AOT-GAN Training

```bash
python scripts/train.py \
    --model aot-gan \
    --config configs/aot-gan/oai_config.yml \
    --resume  # Optional: resume from checkpoint
```

**Configuration Options:**
- `batch_size`: Batch size for training (default: 8)
- `image_size`: Input image size (default: 512)
- `lr_g`: Generator learning rate (default: 1e-4)
- `lr_d`: Discriminator learning rate (default: 1e-4)
- `num_workers`: Number of data loading workers (default: 4)

#### ICT Training

```bash
python scripts/train.py \
    --model ict \
    --config configs/ict/oai_config.yml
```

**Configuration Options:**
- `BATCH_SIZE`: Batch size for training (default: 32)
- `INPUT_SIZE`: Input image size (default: 256)
- `LR`: Learning rate (default: 0.0001)
- `MAX_ITERS`: Maximum training iterations (default: 5e6)

### Testing

#### AOT-GAN Testing

```bash
python scripts/test.py \
    --model aot-gan \
    --config configs/aot-gan/oai_config.yml \
    --subset subset_4  # or 'test' for full test set
```

#### ICT Testing

```bash
python scripts/test.py \
    --model ict \
    --config configs/ict/oai_config.yml \
    --subset subset_4
```

#### RePaint Testing

```bash
python scripts/test.py \
    --model repaint \
    --config configs/repaint/oai_config.yml \
    --subset subset_4
```

### Evaluation

#### Comprehensive Evaluation

```bash
python scripts/evaluate.py \
    --models aot-gan ict repaint \
    --subset subset_4 \
    --output results/evaluation/comprehensive_20240101
```

#### Specific Model Evaluation

```bash
python scripts/evaluate.py \
    --models aot-gan \
    --subset test \
    --output results/evaluation/aot_gan_only
```

## Configuration

### Configuration Files

All configuration files are located in the `configs/` directory:

```
configs/
├── aot-gan/
│   └── oai_config.yml
├── ict/
│   └── oai_config.yml
└── repaint/
    └── oai_config.yml
```

### Customizing Configurations

#### AOT-GAN Configuration

```yaml
# configs/aot-gan/oai_config.yml
data:
  train_images: "./data/oai/train/img"
  train_masks: "./data/oai/train/mask"
  # ... other data paths

model:
  name: "aotgan"
  block_num: 8
  rates: "1+2+4+8"
  gan_type: "smgan"

training:
  batch_size: 8
  image_size: 512
  lr_g: 1e-4
  lr_d: 1e-4
  # ... other training parameters
```

#### ICT Configuration

```yaml
# configs/ict/oai_config.yml
MODE: 2  # 1: train, 2: test, 3: eval
MODEL: 2
MASK: 3
EDGE: 1

# Dataset paths
TRAIN_FLIST: "./data/oai/train/img"
VAL_FLIST: "./data/oai/valid/img"
TEST_FLIST: "./data/oai/test/img"

# Training parameters
BATCH_SIZE: 32
INPUT_SIZE: 256
LR: 0.0001
MAX_ITERS: 5e6
```

#### RePaint Configuration

```yaml
# configs/repaint/oai_config.yml
model:
  name: "repaint"
  model_path: "./data/pretrained/256x256_diffusion.pt"
  classifier_path: "./data/pretrained/256x256_classifier.pt"

data:
  gt_path: "./data/oai/test/img"
  mask_path: "./data/oai/test/mask"
  output_path: "./output/RePaint/OAI"

inference:
  sample_num: 1
  jump_length: 10
  jump_n_sample: 10
  n_steps: 1000
```

## Data Management

### Dataset Structure

The project expects the following data structure:

```
data/
├── oai/
│   ├── gt_img/           # Original OAI images
│   ├── train/
│   │   ├── img/          # Training images
│   │   ├── mask/         # Training masks
│   │   ├── mask/inv/     # Inverted masks
│   │   └── edge/         # Edge maps
│   ├── valid/
│   │   ├── img/          # Validation images
│   │   ├── mask/         # Validation masks
│   │   ├── mask/inv/     # Inverted masks
│   │   └── edge/         # Edge maps
│   └── test/
│       ├── img/          # Test images
│       ├── mask/         # Test masks
│       ├── mask/inv/     # Inverted masks
│       ├── edge/         # Edge maps
│       └── subset_4/     # Subset for evaluation
└── pretrained/
    ├── aot-gan/          # AOT-GAN pretrained models
    ├── ict/              # ICT pretrained models
    └── repaint/          # RePaint pretrained models
```

### Generating Dataset Splits

```bash
cd data/oai
python split.py
```

This will:
1. Create balanced train/validation/test splits
2. Generate random square masks
3. Create inverted masks for RePaint
4. Generate Canny edge maps
5. Create subset_4 for evaluation

## Results and Outputs

### Output Structure

```
output/
├── AOT-GAN/
│   └── OAI/
│       ├── test/         # Full test set results
│       └── subset_4/     # Subset results
├── ICT/
│   └── OAI/
│       ├── test/         # Full test set results
│       └── subset_4/     # Subset results
└── RePaint/
    └── OAI/
        ├── test/         # Full test set results
        └── subset_4/     # Subset results
```

### Results Analysis

```bash
# Generate comprehensive comparison report
python scripts/evaluate.py \
    --models aot-gan ict repaint \
    --subset subset_4 \
    --output results/evaluation/comprehensive

# View results
ls results/evaluation/comprehensive/
# comparison_results.csv
# comparison_report.png
```

## Advanced Usage

### Custom Training

```python
from utils.data import create_data_loaders
from utils.config import load_config

# Load configuration
config = load_config("configs/aot-gan/oai_config.yml")

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    data_dir="data/oai",
    batch_size=config.training.batch_size,
    image_size=config.training.image_size
)

# Custom training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Your training code here
        pass
```

### Custom Evaluation

```python
from scripts.evaluate import calculate_inpainting_metrics

# Calculate custom metrics
metrics = calculate_inpainting_metrics(
    gt_images=gt_images,
    inpainted_images=inpainted_images,
    masks=masks
)

print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.4f}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size in configuration
   - Use gradient checkpointing
   - Clear GPU cache

2. **Path Errors**
   - Check configuration file paths
   - Ensure data directory structure is correct
   - Use absolute paths if needed

3. **Import Errors**
   - Check PYTHONPATH is set correctly
   - Ensure virtual environment is activated
   - Reinstall dependencies

### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Review error logs in `results/logs/`
- Create an issue on the project repository

## Best Practices

1. **Use Configuration Files**: Always use configuration files instead of hardcoding parameters
2. **Monitor Training**: Use TensorBoard to monitor training progress
3. **Save Checkpoints**: Regularly save model checkpoints during training
4. **Validate Results**: Always validate results on held-out test sets
5. **Document Experiments**: Keep detailed logs of all experiments
6. **Version Control**: Use git to track changes and experiments
