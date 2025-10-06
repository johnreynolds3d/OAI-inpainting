# OAI Untracked Data

This directory contains the large data files for the OAI X-ray Inpainting project that are not tracked in Git.

## 📊 Data Overview

- **Total size**: ~16GB
- **Total files**: 559
- **OAI X-ray images**: 539 PNG files
- **Pretrained models**: 19 model files (.pt/.pth)

## 📁 Directory Structure

```
OAI_untracked/
├── data/
│   ├── oai/
│   │   └── img/                    # OAI X-ray images (539 files, ~11.6MB)
│   │       ├── 6.C.1_*.png        # Osteoporotic cases
│   │       └── 6.E.1_*.png        # Normal cases
│   └── pretrained/                 # Pretrained models (19 files, ~15GB)
│       ├── aot-gan/               # AOT-GAN pretrained models
│       │   ├── celebahq/          # CelebA-HQ pretrained models
│       │   └── places2/           # Places2 pretrained models
│       ├── ict/                   # ICT pretrained models
│       │   ├── Transformer/       # Transformer models
│       │   └── Upsample/          # Upsampler models
│       └── repaint/               # RePaint pretrained models
│           ├── 256x256_classifier.pt
│           ├── 256x256_diffusion.pt
│           ├── celeba256_250000.pt
│           └── places256_300000.pt
└── README.md                      # This file
```

## 🎯 Usage in Google Colab

1. **Upload this entire directory** to Google Drive at:
   ```
   /content/drive/MyDrive/Colab Notebooks/OAI_untracked/
   ```

2. **The Colab notebook will automatically**:
   - Mount Google Drive
   - Detect this data structure
   - Copy data to local storage for performance
   - Generate train/valid/test splits
   - Create masks and edge maps

## 📋 Data Details

### OAI X-ray Images
- **Source**: Osteoarthritis Initiative (OAI) dataset
- **Format**: PNG files
- **Naming**: `6.C.1_*` (osteoporotic), `6.E.1_*` (normal)
- **Size**: ~11.6MB total
- **Usage**: Ground truth images for inpainting

### Pretrained Models
- **AOT-GAN**: 6 files (~2GB)
  - CelebA-HQ: D0000000.pt, G0000000.pt, O0000000.pt
  - Places2: D0000000.pt, G0000000.pt, O0000000.pt
- **ICT**: 9 files (~8.3GB)
  - Transformer: FFHQ.pth, ImageNet.pth, Places2_Nature.pth
  - Upsample: 6 model files for different datasets
- **RePaint**: 4 files (~6.4GB)
  - Diffusion and classifier models

## 🔧 Setup Instructions

1. **Upload to Google Drive**:
   ```bash
   # Copy entire directory to Google Drive
   cp -r /path/to/OAI_untracked /content/drive/MyDrive/Colab\ Notebooks/
   ```

2. **Verify structure**:
   ```bash
   ls -la "/content/drive/MyDrive/Colab Notebooks/OAI_untracked/data/oai/img/" | wc -l
   # Should show 540 (539 files + 1 for total line)
   ```

3. **Run Colab notebook**:
   - The notebook will automatically detect and use this data
   - No manual configuration needed

## ⚠️ Important Notes

- **Large files**: Some model files are >1GB
- **Upload time**: May take 30-60 minutes depending on connection
- **Storage**: Ensure sufficient Google Drive space (16GB+)
- **Permissions**: Make sure Colab can access the files

## 🔄 Updates

- **Data changes**: Re-upload the entire directory
- **Code changes**: Use `git pull` in Colab (handled automatically)
- **New models**: Add to appropriate `pretrained/` subdirectory

## 📞 Support

If you encounter issues:
1. Check file permissions in Google Drive
2. Verify directory structure matches exactly
3. Ensure sufficient storage space
4. Check Colab notebook logs for specific errors
