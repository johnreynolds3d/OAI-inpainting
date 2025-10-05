# OAI Inpainting Project Setup Guide

This guide explains how to set up the OAI Inpainting project with the required untracked data files.

## Overview

The OAI Inpainting project requires additional data files that are not tracked in git due to their size:
- **OAI X-ray images** (~539 PNG files)
- **Pretrained models** for AOT-GAN, ICT, and RePaint

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/johnreynolds3d/OAI-inpainting.git
cd OAI-inpainting
```

### 2. Set Up Untracked Data

The project includes a setup script that automatically populates the required data files:

```bash
# Auto-detect and set up data (recommended)
python setup_data.py

# Preview what would be copied (dry run)
python setup_data.py --dry-run

# Specify custom source directory
python setup_data.py --source-dir /path/to/OAI_untracked

# Force overwrite existing files
python setup_data.py --force
```

### 3. Generate Dataset Splits

Once the data is set up, generate the train/validation/test splits:

```bash
cd data/oai
python split.py
```

## Untracked Data Structure

Your untracked data should be organized as follows:

```
OAI_untracked/                    # Can be named differently
├── data/
│   ├── oai/
│   │   └── img/                  # OAI X-ray images (required)
│   │       ├── 6.C.1_*.png
│   │       └── 6.E.1_*.png
│   └── pretrained/               # Pretrained models (optional)
│       ├── aot-gan/
│       │   ├── celebahq/
│       │   └── places2/
│       ├── ict/
│       │   ├── Transformer/
│       │   └── Upsample/
│       └── repaint/
│           ├── 256x256_classifier.pt
│           ├── 256x256_diffusion.pt
│           └── *.pt
```

## Setup Script Features

The `setup_data.py` script provides:

- **Auto-detection**: Automatically finds untracked data directories
- **Structure verification**: Validates required components are present
- **Dry run mode**: Preview operations without copying files
- **Size estimation**: Shows data sizes and file counts
- **Progress tracking**: Detailed output of copy operations
- **Error handling**: Graceful handling of missing components

## Common Locations

The script searches for untracked data in these locations:

1. `../OAI_untracked/` (parent directory)
2. `./OAI_untracked/` (project directory)
3. `~/OAI_untracked/` (home directory)
4. Current working directory

## Troubleshooting

### "Could not find untracked data directory"

**Solution**: Use the `--source-dir` parameter to specify the exact path:

```bash
python setup_data.py --source-dir /absolute/path/to/OAI_untracked
```

### "Some required components are missing"

**Solution**: Ensure your untracked data has the correct structure:

```bash
# Check structure
ls -la /path/to/OAI_untracked/data/oai/img/ | head -5
ls -la /path/to/OAI_untracked/data/pretrained/
```

### "Insufficient disk space"

**Solution**:
- Check available disk space: `df -h`
- Clean up unnecessary files
- Use `--force` to proceed anyway (not recommended)

### "Permission denied"

**Solution**: Ensure you have write permissions to the project directory:

```bash
# Fix permissions
chmod -R 755 .
```

## Manual Setup (Alternative)

If the setup script doesn't work, you can manually copy the data:

```bash
# Copy OAI images (required)
cp -r /path/to/OAI_untracked/data/oai/img data/oai/

# Copy pretrained models (optional)
cp -r /path/to/OAI_untracked/data/pretrained data/
```

## Verification

After setup, verify the data is in place:

```bash
# Check OAI images
ls data/oai/img/ | wc -l  # Should show ~539 files

# Check pretrained models
ls data/pretrained/*/

# Test the split script
cd data/oai && python split.py
```

## Project Structure After Setup

```
OAI-inpainting/
├── setup_data.py              # Setup script
├── SETUP.md                   # This guide
├── data/
│   ├── oai/
│   │   ├── img/               # OAI images (from untracked data)
│   │   ├── data.csv           # Metadata
│   │   ├── split.py           # Split generation script
│   │   ├── train/             # Generated splits
│   │   ├── valid/
│   │   └── test/
│   └── pretrained/            # Pretrained models (from untracked data)
│       ├── aot-gan/
│       ├── ict/
│       └── repaint/
├── configs/                   # Configuration files
├── scripts/                   # Training/testing scripts
└── ...
```

## Next Steps

After successful setup:

1. **Generate splits**: `cd data/oai && python split.py`
2. **Train models**: Use the scripts in the `scripts/` directory
3. **Run experiments**: Follow the usage guide in `docs/usage.md`

## Support

If you encounter issues:

1. Check this guide for troubleshooting steps
2. Verify your untracked data structure matches the requirements
3. Use `--dry-run` to preview operations
4. Check file permissions and disk space
5. Create an issue on GitHub with detailed error information

## Data Sources

- **OAI Images**: Osteoarthritis Initiative dataset
- **Pretrained Models**:
  - AOT-GAN: CelebA-HQ and Places2 pretrained models
  - ICT: FFHQ, ImageNet, and Places2_Nature models
  - RePaint: Diffusion and classifier models
