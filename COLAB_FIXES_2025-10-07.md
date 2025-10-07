# 🔧 Colab Notebook Fixes - October 7, 2025

## 🎯 Summary
Fixed all issues preventing comprehensive model testing in Google Colab. Achieved **100% success rate** (8/8 pretrained models) in ~9 minutes.

---

## 🐛 Issues Fixed

### 1. **Cell 7: Massive Output from Data Copy Error**
**Problem:**
- Cell 3 created symlink: `data/oai → Google Drive`
- Cell 7 tried to copy from Google Drive to `data/`
- Circular reference caused `SameFileError` for all 539 images
- Error message printed character-by-character → 401,213 token output file

**Solution:**
- Detect existing symlinks before attempting copy
- Use `os.walk()` instead of `shutil.copytree()` for better control
- Truncate error messages to 200 characters max
- Add progress updates during file copying

**Files Modified:** `notebooks/OAI_Inpainting_Colab.ipynb` (Cell 7)

---

### 2. **Results Directory: NotADirectoryError**
**Problem:**
- Cell 4 created symlink: `results → ../OAI_untracked/results/`
- Google Drive restrictions prevented creating nested directories through symlinks
- Tests failed when creating `results/AOT-GAN/CelebA-HQ/subset_4`

**Solution:**
- Replace results symlink with real directory
- Keeps data as symlink (read-only, persistent in Drive)
- Results in local directory (fast, no restrictions)
- Added Cell 7: "FIX: Results Directory Issue" for troubleshooting

**Files Modified:** `notebooks/OAI_Inpainting_Colab.ipynb` (Cells 4, 6, 7)

---

### 3. **ICT: Wrong Script and Arguments**
**Problem:**
- Initially tried `run.py --config` (single-image inference script)
- Then tried `main.py --config` (mode not set, args unavailable)
- Config file approach was too complex

**Solution:**
- Use `test.py` which calls `main(mode=2)` with test arguments enabled
- Use command-line args: `--path, --model, --input, --mask, --prior, --output`
- Point `--path` to model checkpoint directory (contains config.yml)
- Set `--condition_num 1` (we have 1 edge map, not 8 transformer priors)

**Files Modified:** `scripts/colab_comprehensive_test.py` (test_ict_variant)

---

### 4. **ICT: Edge Files in condition_1/ Subdirectory**
**Problem:**
- ICT expects edge files in `condition_N/` subdirectories for testing
- Our edges were in `subset_4/` directly
- FileNotFoundError: `.../edge/subset_4/condition_1/image.png`

**Solution:**
- Create `condition_1/` subdirectory in edge directory
- Symlink each edge file into `condition_1/`
- Set `--condition_num 1` to match our single-edge-per-image setup

**Files Modified:** `scripts/colab_comprehensive_test.py` (test_ict_variant)

---

### 5. **ICT ImageNet: Multi-GPU Configuration**
**Problem:**
- ImageNet config has `GPU: [0,1]` (trained on 2 GPUs)
- Colab only has 1 GPU
- AssertionError: Invalid device id

**Solution:**
- Temporarily modify `config.yml` before testing
- Replace `GPU: [0,1]` with `GPU: [0]`
- Restore original config after test completes
- Only modifies configs that need it (checks for [0,1] first)

**Files Modified:** `scripts/colab_comprehensive_test.py` (test_ict_variant)

---

### 6. **ICT ImageNet: DataParallel Checkpoint Loading**
**Problem:**
- ImageNet trained with DataParallel (multi-GPU)
- PyTorch adds `module.` prefix to all parameter names
- Single-GPU loading fails with key mismatch errors

**Solution:**
- Added `_strip_dataparallel_prefix()` method to BaseModel
- Detects `module.` prefixes in checkpoint
- Strips prefixes automatically: `module.encoder.0` → `encoder.0`
- Applied to both generator and discriminator loading
- Backward compatible with single-GPU checkpoints

**Files Modified:** `models/ict/Guided_Upsample/src/models.py`

---

### 7. **RePaint ImageNet: Class-Conditional Checkpoint**
**Problem:**
- ImageNet checkpoint trained with class labels (1000 ImageNet classes)
- Contains `label_emb.weight` and `label_emb.bias` parameters
- RePaint uses unconditional generation for inpainting
- RuntimeError: Unexpected key(s) in state_dict: "label_emb.weight"

**Solution:**
- Detect `label_emb` keys in checkpoint
- Strip class-conditional keys before loading
- Load cleaned state dict into unconditional UNetModel
- Aligns with RePaint's design: unconditional inference, condition on known regions

**Files Modified:** `models/repaint/test.py`

---

## 📊 Final Test Results

### ✅ **All 8 Pretrained Models Working**
| Model | Time | Architecture |
|-------|------|--------------|
| AOT-GAN CelebA-HQ | 11.0s | GAN |
| AOT-GAN Places2 | 5.3s | GAN |
| ICT FFHQ | 11.1s | Transformer |
| ICT ImageNet | 11.0s | Transformer |
| ICT Places2_Nature | 10.0s | Transformer |
| RePaint CelebA-HQ | 149.9s | Diffusion |
| RePaint ImageNet | 157.6s | Diffusion |
| RePaint Places2 | 159.1s | Diffusion |

**Success Rate: 100% (8/8 available models)**
**Total Test Time: 8 minutes 35 seconds**

### ⏭️ **2 Models Skipped (Expected)**
- AOT-GAN OAI (requires training)
- ICT OAI (requires training)

---

## 🔑 Key Technical Insights

1. **Google Drive Restrictions**: Using symlinks to Google Drive for results caused directory creation issues. Local directories work better for write-heavy operations.

2. **DataParallel Compatibility**: Multi-GPU trained models need `module.` prefix stripping when loading on single GPU - common PyTorch issue.

3. **Class-Conditional vs Unconditional**: ImageNet models trained with class labels can be used unconditionally by stripping label embedding layers.

4. **ICT's Unique Requirements**: Two-stage architecture (Transformer + Guided Upsample) expects specific directory structures and multiple "conditions" per image.

5. **Portable Design**: Maintained relative symlinks throughout to ensure notebook works both locally and in Colab.

---

## 📝 Commits Made

1. `26c3c62` - Fix Cell 7 data copy issue
2. `dc784ff` - Fix results directory handling in Cell 4
3. `dd23d31` - Add dedicated fix cell for NotADirectoryError
4. `4b65784` - Revert to portable relative paths
5. `459d7d3` - Fix ICT to use Guided_Upsample/main.py
6. `35b81dd` - Fix ICT to use test.py instead of main.py
7. `68ae1fb` - Fix ICT to use proper checkpoint path
8. `7a5579c` - Fix ICT condition_1 subdirectory and GPU issues
9. `36af72a` - Fix ICT ImageNet GPU with config modification
10. `715de28` - Fix ICT DataParallel checkpoint loading
11. `10aafc0` - Fix RePaint ImageNet class-conditional checkpoint

**Total: 11 commits, all pushed to GitHub** ✅

---

## 🚀 Usage

The notebook is now production-ready. Users can:

1. Run `Runtime → Restart session and run all`
2. All 8 pretrained models will test successfully
3. Results automatically saved to `results/` directory
4. JSON summary at `results/comprehensive_test_results.json`

---

## 🎓 Educational Value

This debugging session demonstrates:
- Platform-agnostic design principles
- Handling third-party model integration challenges
- PyTorch checkpoint compatibility issues
- Google Colab environment constraints
- Systematic debugging methodology

---

*Generated: October 7, 2025*
*Project: OAI X-ray Inpainting - Comprehensive Model Testing*
