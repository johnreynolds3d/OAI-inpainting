# 🚀 Complete Colab Walkthrough: Testing All Model Variants

## Overview

This guide walks you through testing all 9 inpainting model variants on the OAI dataset using Google Colab.

**Total time:** ~30-60 minutes  
**Models tested:** 9 variants (AOT-GAN, ICT, RePaint)  
**Test images:** 4 from subset_4  
**Results:** Saved to Google Drive automatically

---

## 📋 Complete Cell Execution Order

### Step 1: Check GPU (Cell 2)

**Cell:** "Check GPU availability"
```python
# Check GPU availability
import torch
```

**What it does:** Verifies CUDA/GPU is enabled

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True
GPU: Tesla T4 (or similar)
```

**If GPU not available:** Runtime → Change runtime type → GPU

---

### Step 2: Mount Google Drive (Cell 6)

**Cell:** "Mount Google Drive and setup data"
```python
from google.colab import drive
drive.mount("/content/drive")
```

**What it does:** Mounts your Google Drive

**Action needed:** Click "Connect to Google Drive" and grant permissions

**Expected output:**
```
Mounted at /content/drive
✅ Google Drive mounted successfully
✅ OAI data found in Google Drive: /content/drive/MyDrive/Colab Notebooks/OAI_untracked
```

---

### Step 3: Clone Repository (Cell 3)

**Cell:** "Alternative setup method (if Git fails)"
```python
setup_repository_alternative()
```

**What it does:**
- Navigates to `/content/drive/MyDrive/Colab Notebooks/`
- Downloads latest code from GitHub
- Creates relative symlinks:
  - `data -> ../OAI_untracked/data/`
  - `results -> ../OAI_untracked/results/`
- Installs dependencies

**Expected output:**
```
✅ Working directory: /content/drive/MyDrive/Colab Notebooks
📥 Downloading from: https://github.com/...
📦 Extracting repository...
✅ Repository downloaded and extracted successfully
📂 Current directory: .../Colab Notebooks/OAI-inpainting
🗑️  Removed broken data file
🗑️  Removed broken results file
✅ Created data symlink -> ../OAI_untracked/data/
✅ Created results symlink -> ../OAI_untracked/results/
✅ Installed torch
✅ Installed torchvision
... (all dependencies)
✅ Alternative setup complete!
📁 Data location: .../OAI_untracked/data
📁 Results location: .../OAI_untracked/results
```

**Time:** ~3-5 minutes

---

### Step 4: Verify Installation (Cell 5)

**Cell:** "Verify installation and setup"
```python
from src.paths import get_project_root
```

**What it does:** Checks imports work and lists models

**Expected output:**
```
✅ Core modules imported successfully
📁 Project root: ...

📋 Available models:
  - aot-gan
  - classifier
  - ict
  - repaint
```

---

### Step 5: Generate Dataset Splits (Cell - after "Update Repository")

**Cell:** "GENERATE DATASET SPLITS"
```python
# 🔄 GENERATE DATASET SPLITS (Required before testing!)
```

**What it does:**
- Runs `split.py` to create train/valid/test splits
- Generates random masks
- Creates Canny edge maps
- Creates inverted masks
- **Creates subset_4 (4 test images)**

**Expected output:**
```
============================================================
🔄 GENERATING DATASET SPLITS
============================================================
✅ Splits already exist!
✅ Found 4 files in subset_4
```

OR (first time):
```
⚠️  Splits not found - generating now...
This will create:
  • Train/valid/test splits
  • Random masks
  • Edge maps
  • Inverted masks
  • subset_4 evaluation set (4 images)

✅ Dataset splits generated successfully!
✅ subset_4 created with 4 images

📋 Verification:
  ✅ img/subset_4/: 4 files
  ✅ mask/subset_4/: 4 files
  ✅ edge/subset_4/: 4 files
  ✅ mask_inv/subset_4/: 4 files

✅ Ready for comprehensive testing!
```

**Time:** ~5-10 minutes (first time only)

---

### Step 6: Import Test Scripts (Cell - "PIPELINE RUNNER SETUP")

**Cell:** "PIPELINE RUNNER SETUP"
```python
from colab_comprehensive_test import ModelTester
from colab_pipeline import run_full_pipeline, run_phase
```

**What it does:** Imports testing functions from Git-tracked scripts

**Expected output:**
```
✅ Pipeline runner imported successfully!

🎯 Available functions:

📊 COMPREHENSIVE TESTING (Recommended):
  - ModelTester().run_comprehensive_test() - Test ALL 9 model variants
     • AOT-GAN: CelebA-HQ, Places2, OAI
     • ICT: FFHQ, ImageNet, Places2_Nature, OAI
     • RePaint: CelebA-HQ, ImageNet, Places2

🔄 PHASED PIPELINE (For training):
  - run_full_pipeline() - Run all 5 phases
  - run_phase(1-5) - Individual phases

💡 Ready to use! Go to the next cell to run commands.
```

---

### Step 7: Run Comprehensive Test (MAIN ACTION!)

**Cell:** "QUICK START - Test ALL Model Variants"
```python
tester = ModelTester(timeout_per_model=600, verbose=True)
results = tester.run_comprehensive_test(models=["all"])
```

**What it does:** Tests all 9 model variants on 4 subset_4 images

**Expected output:**
```
[08:32:19] 🚀 COMPREHENSIVE MODEL TESTING ON OAI SUBSET_4
[08:32:19] i Verifying subset_4 data...
[08:32:19] ✅ Found 4 files in subset_4/
[08:32:20] 🚀 TESTING AOT-GAN VARIANTS
[08:32:20] 🔄 Testing AOT-GAN CelebA-HQ...
[08:34:45] ✅ Completed in 145.3s
[08:34:45] 🔄 Testing AOT-GAN Places2...
[08:37:10] ✅ Completed in 145.1s
[08:37:10] 🔄 Testing AOT-GAN OAI...
[08:37:10] ⚠️  Model file not found: ...
[08:37:10] 🚀 TESTING ICT VARIANTS
[08:37:11] 🔄 Testing ICT FFHQ...
[08:40:15] ✅ Completed in 184.2s
... (continues for all models)

📊 DETAILED RESULTS:
AOT-GAN CelebA-HQ        : ✅ PASSED (145.3s)
AOT-GAN Places2          : ✅ PASSED (145.1s)
AOT-GAN OAI              : ⏭️ SKIPPED (Model not available)
ICT FFHQ                 : ✅ PASSED (184.2s)
ICT ImageNet             : ✅ PASSED (189.5s)
ICT Places2_Nature       : ✅ PASSED (191.3s)
ICT OAI                  : ⏭️ SKIPPED (Model not available)
RePaint CelebA-HQ        : ✅ PASSED (421.7s)
RePaint ImageNet         : ✅ PASSED (435.2s)
RePaint Places2          : ✅ PASSED (428.9s)

📁 OUTPUT DIRECTORIES:
  AOT-GAN CelebA-HQ        : results/AOT-GAN/CelebA-HQ/subset_4
  AOT-GAN Places2          : results/AOT-GAN/Places2/subset_4
  ...

✅ Results saved to: results/comprehensive_test_results.json

============================================================
🎉 COMPREHENSIVE TEST COMPLETE!
============================================================
✅ Successful: 6
❌ Failed: 0
⏭️ Skipped: 3
```

**Time:** ~30-60 minutes (varies by GPU)

---

### Step 8: Generate Comparison Strips

**Add a new cell after the comprehensive test:**

```python
# 🎨 GENERATE COMPARISON STRIPS
# Creates horizontal strips: GT, GT+Mask, AOT-GAN, ICT, RePaint outputs

print("🎨 Generating comparison strips...")

from scripts.generate_comparison_strips import main as generate_strips

strip_paths = generate_strips()

print(f"\n✅ Generated {len(strip_paths)} comparison strips")
print("📁 Location: results/comparison_strips/")
print("\n📸 Individual strips:")
for path in strip_paths:
    print(f"  - {path.name}")
```

**What it creates:**
- Individual comparison strips for each image
- Summary figure with all images stacked vertically
- Each strip shows: GT → GT+Mask → All model outputs

**Expected output:**
```
🎨 Generating comparison strips...
============================================================
🎨 COMPARISON STRIP GENERATOR
============================================================
📁 GT images: .../data/oai/test/img/subset_4
📁 Masks: .../data/oai/test/mask/subset_4
📁 Results: .../results
📁 Output: .../results/comparison_strips

✅ Found 4 test images

🖼️  Creating strip for 6.C.1_9068305_20081124_001.png
  ✅ Found 8 images for strip
  ✅ Saved: comparison_6.C.1_9068305_20081124_001.png
... (repeats for all 4 images)

📊 Creating summary figure...
  ✅ Summary saved: all_comparisons_summary.png

============================================================
📊 SUMMARY
============================================================
✅ Created 4 comparison strips
✅ Summary figure: all_comparisons_summary.png
📁 Output: results/comparison_strips
```

**Files created:**
```
results/comparison_strips/
├── comparison_image1.png  # Horizontal strip
├── comparison_image2.png
├── comparison_image3.png
├── comparison_image4.png
└── all_comparisons_summary.png  # All strips stacked
```

---

### Step 9: Visualize Results (Existing Cells)

**Cell:** "Visualize Results"
- Shows JSON summary with timing

**Cell:** "Display Sample Results"
- Shows sample images in grid

**Cell:** "Download Results"
- Packages everything as ZIP

---

## 🎯 Complete Workflow Summary

```
1. Check GPU                      → 10 seconds
2. Mount Google Drive             → 30 seconds (+ auth)
3. Clone repo & setup             → 3-5 minutes
4. Verify installation            → 10 seconds
5. Generate dataset splits        → 5-10 minutes (first time)
6. Import test scripts            → 5 seconds
7. Run comprehensive test         → 30-60 minutes ⭐
8. Generate comparison strips     → 2-3 minutes
9. View/download results          → 1 minute

Total: ~40-80 minutes (most is automated testing)
```

---

## 📊 What You Get

### 1. Comprehensive Test Results

**JSON file:** `results/comprehensive_test_results.json`
```json
{
  "timestamp": "2025-10-07T08:32:19",
  "duration": "0:45:32",
  "results": [
    {
      "model": "AOT-GAN CelebA-HQ",
      "success": true,
      "elapsed": 145.3,
      "output_dir": "results/AOT-GAN/CelebA-HQ/subset_4"
    },
    ...
  ],
  "summary": {
    "total": 9,
    "successful": 6,
    "failed": 0,
    "skipped": 3
  }
}
```

### 2. Model Outputs

**Directory structure:**
```
results/
├── AOT-GAN/
│   ├── CelebA-HQ/subset_4/  # 4 inpainted images
│   ├── Places2/subset_4/
│   └── OAI/subset_4/ (if trained)
├── ICT/
│   ├── FFHQ/subset_4/
│   ├── ImageNet/subset_4/
│   ├── Places2_Nature/subset_4/
│   └── OAI/subset_4/ (if trained)
└── RePaint/
    ├── CelebA-HQ/subset_4/
    ├── ImageNet/subset_4/
    └── Places2/subset_4/
```

### 3. Comparison Strips

**Visual comparisons:**
```
results/comparison_strips/
├── comparison_image1.png  # GT | GT+Mask | AOT-GAN variants | ICT variants | RePaint variants
├── comparison_image2.png
├── comparison_image3.png
├── comparison_image4.png
└── all_comparisons_summary.png  # All 4 strips stacked vertically
```

**Each strip shows (left to right):**
1. Ground Truth (GT)
2. GT + Red Mask Overlay
3. AOT-GAN CelebA-HQ
4. AOT-GAN Places2
5. AOT-GAN OAI (if available)
6. ICT FFHQ
7. ICT ImageNet
8. ICT Places2_Nature
9. ICT OAI (if available)
10. RePaint CelebA-HQ
11. RePaint ImageNet
12. RePaint Places2

---

## 🔧 Troubleshooting

### "subset_4 not found"
**Fix:** Run "GENERATE DATASET SPLITS" cell (Step 5)

### "results is not a directory"
**Fix:** Run this in a new cell:
```python
from pathlib import Path
Path("results").unlink() if Path("results").is_file() else None
Path("../OAI_untracked/results").mkdir(parents=True, exist_ok=True)
Path("results").symlink_to("../OAI_untracked/results/")
```

### "Model not found"
**Expected:** Some models may be skipped if:
- OAI-trained models don't exist yet (normal)
- Pretrained models not uploaded to Google Drive

### "Timeout"
**Fix:** Increase timeout:
```python
tester = ModelTester(timeout_per_model=1200, verbose=True)  # 20 minutes per model
```

---

## 💡 Tips

1. **First run takes longer** (~10 min for splits) - subsequent runs skip this
2. **Results persist** in Google Drive - won't be lost on disconnect
3. **Can run multiple times** - won't duplicate work
4. **Comparison strips** are great for visual assessment and papers
5. **Download ZIP** if you want everything offline

---

## 🎓 Next Steps: Training (Optional)

If you want to train models on OAI data (6-8 hours):

```python
from scripts.colab_pipeline import run_full_pipeline

# Full pipeline
run_full_pipeline(timeout_hours=8)

# Or individual phases
run_phase(2)  # AOT-GAN training only
run_phase(3)  # ICT training only
```

This will create OAI-trained variants that appear in future comprehensive tests!

---

Made with ❤️ for medical imaging research

