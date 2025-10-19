# 🔍 Google Colab Diagnostics Commands

Quick commands to verify directory structure and symlinks in Google Colab.

## 🚀 Quick One-Liner (Copy-Paste into Colab Cell)

```python
# QUICK DIAGNOSTIC - Copy this entire block into a Colab cell
import os
from pathlib import Path

print("=" * 70)
print("🔍 QUICK DIRECTORY DIAGNOSTIC")
print("=" * 70)

cwd = Path.cwd()
print(f"\n📁 Current directory: {cwd}")

# Check data symlink
data = Path("data")
if data.exists():
    is_link = data.is_symlink()
    target = os.readlink(data) if is_link else "N/A"
    resolved = data.resolve()
    print(f"\n{'✅' if is_link else '⚠️ '} data/: {'SYMLINK' if is_link else 'REAL DIR'}")
    if is_link:
        print(f"   Target: {target}")
    print(f"   Resolved: {resolved}")
    print(f"   {'✅ CORRECT' if str(resolved).endswith('OAI_untracked/data') else '❌ WRONG'}")
else:
    print(f"\n❌ data/: NOT FOUND")

# Check results symlink
results = Path("results")
if results.exists():
    is_link = results.is_symlink()
    target = os.readlink(results) if is_link else "N/A"
    resolved = results.resolve()
    print(f"\n{'✅' if is_link else '⚠️ '} results/: {'SYMLINK' if is_link else 'REAL DIR'}")
    if is_link:
        print(f"   Target: {target}")
    print(f"   Resolved: {resolved}")
    print(f"   {'✅ CORRECT' if str(resolved).endswith('OAI_untracked/results') else '❌ WRONG'}")
else:
    print(f"\n❌ results/: NOT FOUND")

# Check data availability
oai_img = Path("data/oai/img")
if oai_img.exists():
    count = len(list(oai_img.glob("*.png")))
    print(f"\n✅ Data access: {count} images in data/oai/img/")
else:
    print(f"\n❌ Cannot access data/oai/img/")

# Test write
try:
    test = Path("results/.test")
    test.write_text("test")
    test.unlink()
    print(f"✅ Write access: Can write to results/")
except Exception as e:
    print(f"❌ Write failed: {e}")

print("\n" + "=" * 70)
data_ok = data.exists() and data.is_symlink() and str(data.resolve()).endswith("OAI_untracked/data")
results_ok = results.exists() and results.is_symlink() and str(results.resolve()).endswith("OAI_untracked/results")

if data_ok and results_ok:
    print("🎉 PERFECT! Setup matches local environment")
elif data_ok:
    print("⚠️  data/ OK, but results/ needs fixing")
    print("   Re-run Cell 1 (Complete Setup)")
else:
    print("❌ Setup needed - run Cell 1 (Complete Setup)")
print("=" * 70)
```

## 📋 Step-by-Step Diagnostic Commands

Run these in **separate cells** for detailed investigation:

### Command 1: Check Current Location
```python
from pathlib import Path
import os

print("📍 Current Location:")
print(f"  Working dir: {Path.cwd()}")
print(f"  Absolute: {Path.cwd().resolve()}")
```

### Command 2: Check Symlinks
```python
import os
from pathlib import Path

for name in ["data", "results"]:
    p = Path(name)
    if p.exists():
        if p.is_symlink():
            target = os.readlink(p)
            resolved = p.resolve()
            print(f"✅ {name}/ is SYMLINK")
            print(f"   Target: {target}")
            print(f"   Resolved: {resolved}")
        else:
            print(f"⚠️  {name}/ is REAL DIRECTORY (not symlink!)")
            print(f"   Path: {p.resolve()}")
    else:
        print(f"❌ {name}/ NOT FOUND")
```

### Command 3: Verify Data Access
```python
from pathlib import Path

oai_img = Path("data/oai/img")
print(f"data/oai/img/ exists: {oai_img.exists()}")

if oai_img.exists():
    images = list(oai_img.glob("*.png"))
    print(f"✅ Found {len(images)} PNG files")
    if images:
        print(f"   Sample: {images[0].name}")
else:
    print("❌ Cannot access OAI images")
```

### Command 4: Check Split Data
```python
from pathlib import Path

for split in ["train", "valid", "test"]:
    split_path = Path(f"data/oai/{split}/img")
    if split_path.exists():
        count = len(list(split_path.glob("*.png")))
        print(f"✅ {split:5} split: {count:3} images")
    else:
        print(f"❌ {split:5} split: NOT FOUND")

# Check subset_4
subset_4 = Path("data/oai/test/img/subset_4")
if subset_4.exists():
    count = len(list(subset_4.glob("*.png")))
    print(f"✅ subset_4: {count} images")
else:
    print(f"❌ subset_4: NOT FOUND (run Cell 2 to generate)")
```

### Command 5: Test Write Access
```python
from pathlib import Path

test_file = Path("results/.write_test.txt")
try:
    test_file.write_text("Testing write access")
    content = test_file.read_text()
    test_file.unlink()
    print(f"✅ Write test PASSED")
    print(f"   Location: {test_file.resolve()}")
    print(f"   {'💾 Will persist in Google Drive!' if test_file.resolve().parts[1] == 'content' and 'drive' in test_file.resolve().parts else '⚠️  May not persist'}")
except Exception as e:
    print(f"❌ Write test FAILED: {e}")
```

### Command 6: Compare Paths
```python
from pathlib import Path

data_resolved = Path("data").resolve() if Path("data").exists() else None
results_resolved = Path("results").resolve() if Path("results").exists() else None

print("🔍 Path Analysis:")
print(f"\ndata/ resolves to:")
print(f"  {data_resolved}")
print(f"  Ends with 'OAI_untracked/data': {str(data_resolved).endswith('OAI_untracked/data') if data_resolved else False}")

print(f"\nresults/ resolves to:")
print(f"  {results_resolved}")
print(f"  Ends with 'OAI_untracked/results': {str(results_resolved).endswith('OAI_untracked/results') if results_resolved else False}")

print(f"\n{'✅ MATCH' if (data_resolved and results_resolved and str(data_resolved).endswith('OAI_untracked/data') and str(results_resolved).endswith('OAI_untracked/results')) else '❌ MISMATCH'}: Local vs Colab structure")
```

### Command 7: Full Structure Tree
```python
from pathlib import Path
import os

print("🌳 Directory Tree:")
print()

def show_tree(path, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return

    try:
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if item.is_symlink():
                target = os.readlink(item)
                print(f"{prefix}{current_prefix}{item.name} -> {target}")
            elif item.is_dir():
                print(f"{prefix}{current_prefix}{item.name}/")
                show_tree(item, next_prefix, max_depth, current_depth + 1)
            else:
                # Only show key files
                if item.suffix in [".py", ".yml", ".md", ".csv"]:
                    print(f"{prefix}{current_prefix}{item.name}")
    except PermissionError:
        pass

show_tree(Path.cwd(), max_depth=2)
```

## 🎯 Comprehensive Diagnostic Script

For a complete diagnostic report, run the standalone script:

```bash
!python colab_diagnostics.py
```

This will generate a comprehensive report covering:
- ✅ Current working directory
- ✅ Project root detection
- ✅ Complete directory structure
- ✅ Data directory deep inspection
- ✅ Results directory deep inspection
- ✅ Parent directory sibling check
- ✅ Symlink verification (target vs expected)
- ✅ Read/write access tests
- ✅ Path comparison (local vs Colab)
- ✅ Final verdict with recommendations

## 🔧 Troubleshooting

### If data/ is NOT a symlink:
```python
# Fix data symlink
from pathlib import Path
import shutil

if Path("data").exists() and not Path("data").is_symlink():
    shutil.rmtree("data")  # Remove real directory
    Path("data").symlink_to("../OAI_untracked/data")
    print("✅ Fixed: data/ is now a symlink")
```

### If results/ is NOT a symlink:
```python
# Fix results symlink
from pathlib import Path
import shutil

if Path("results").exists() and not Path("results").is_symlink():
    # Backup existing results if any
    if any(Path("results").iterdir()):
        print("⚠️  results/ has content - creating backup...")
        shutil.move("results", "results_backup")
    else:
        shutil.rmtree("results")

    # Create symlink
    parent_results = Path("../OAI_untracked/results")
    parent_results.mkdir(parents=True, exist_ok=True)
    Path("results").symlink_to(parent_results)
    print("✅ Fixed: results/ is now a symlink")
    print("💾 Results will now persist in Google Drive!")
```

### Check OAI_untracked Location
```python
from pathlib import Path

parent = Path.cwd().parent
oai_untracked = parent / "OAI_untracked"

print(f"Looking for OAI_untracked at:")
print(f"  {oai_untracked}")
print(f"  Exists: {oai_untracked.exists()}")

if oai_untracked.exists():
    print(f"\n✅ Found OAI_untracked!")
    print(f"   data/ exists: {(oai_untracked / 'data').exists()}")
    print(f"   results/ exists: {(oai_untracked / 'results').exists()}")
else:
    print(f"\n❌ OAI_untracked not found as sibling!")
    print(f"\n💡 Make sure your Google Drive has this structure:")
    print(f"   /content/drive/MyDrive/Colab Notebooks/")
    print(f"   ├── OAI-inpainting/")
    print(f"   └── OAI_untracked/")
```

## 📊 Expected Output (Correct Setup)

When everything is correct, you should see:

```
✅ data/: SYMLINK
   Target: ../OAI_untracked/data
   Resolved: /content/drive/MyDrive/Colab Notebooks/OAI_untracked/data
   ✅ CORRECT

✅ results/: SYMLINK
   Target: ../OAI_untracked/results
   Resolved: /content/drive/MyDrive/Colab Notebooks/OAI_untracked/results
   ✅ CORRECT

✅ READ TEST: Found 539 images in data/oai/img/
✅ WRITE TEST: Can write to results/ directory

🎉 PERFECT! Both data/ and results/ point to OAI_untracked/
   This matches local environment structure
   Results will persist in Google Drive!
```

## ⚠️ What to Look For

### ✅ GOOD (Symlinks):
```
data/ → ../OAI_untracked/data/      (symlink)
results/ → ../OAI_untracked/results/ (symlink)
```

### ❌ BAD (Real Directories):
```
data/ (real directory)              ← Wrong!
results/ (real directory)           ← Won't persist!
```

If you see real directories instead of symlinks, re-run Cell 1 (Complete Setup) in the notebook.
