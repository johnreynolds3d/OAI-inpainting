# Results Directory

This directory contains all experimental results, metrics, and logs from the OAI inpainting project.

## Structure

```
results/
├── metrics/          # Quantitative evaluation metrics
├── plots/           # Visualization plots and figures
├── logs/            # Training and inference logs
│   ├── aot-gan/     # AOT-GAN training logs
│   ├── ict/         # ICT training logs
│   └── repaint/     # RePaint inference logs
└── evaluation/      # Model comparison results
```

## Usage

### Metrics
- Store quantitative evaluation metrics (PSNR, SSIM, classification accuracy, etc.)
- CSV files with numerical results
- JSON files with detailed metrics

### Plots
- Visualization plots for model comparison
- Training curves and loss plots
- Classification performance plots
- Inpainting quality visualizations

### Logs
- Training logs from each model
- TensorBoard logs
- Inference logs
- Error logs and debugging information

### Evaluation
- Comprehensive model comparison results
- Statistical analysis
- Performance benchmarks
- Reproducibility reports

## File Naming Convention

- Use descriptive names with timestamps
- Include model name and dataset information
- Use consistent file extensions (.csv, .json, .png, .pdf)
- Organize by experiment date and model

## Examples

```
metrics/
├── aot_gan_oai_20240101_metrics.csv
├── ict_oai_20240101_metrics.csv
└── repaint_oai_20240101_metrics.csv

plots/
├── model_comparison_accuracy.png
├── training_curves.png
└── inpainting_quality_metrics.png

logs/
├── aot-gan/
│   ├── train_20240101.log
│   └── tensorboard/
└── ict/
    ├── train_20240101.log
    └── tensorboard/

evaluation/
├── comprehensive_comparison_20240101.json
├── statistical_analysis.csv
└── reproducibility_report.md
```
