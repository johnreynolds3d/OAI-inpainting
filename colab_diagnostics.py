#!/usr/bin/env python3
"""
Colab Directory Structure Diagnostics
======================================
Run this in Google Colab to verify directory structure, symlinks,
and read/write operations match local environment.

Usage in Colab:
!python colab_diagnostics.py
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("🔍 COLAB DIRECTORY STRUCTURE DIAGNOSTICS")
print("=" * 80)

# 1. Current Working Directory
print("\n[1] 📁 Current Working Directory")
print("─" * 80)
cwd = Path.cwd()
print(f"Current directory: {cwd}")
print(f"Absolute path: {cwd.resolve()}")

# 2. Project Root Detection
print("\n[2] 🏠 Project Root Detection")
print("─" * 80)
if (cwd / "models").exists() and (cwd / "scripts").exists():
    print("✅ Currently in project root")
    project_root = cwd
elif (cwd / "OAI-inpainting" / "models").exists():
    print("✅ Project found in subdirectory")
    project_root = cwd / "OAI-inpainting"
else:
    print("⚠️  Project root not found from current location")
    project_root = cwd

print(f"Project root: {project_root}")

# 3. Directory Structure
print("\n[3] 📂 Directory Structure")
print("─" * 80)
os.chdir(project_root)
print(f"Changed to: {Path.cwd()}")
print()

# Check key directories
key_dirs = ["data", "results", "models", "scripts", "notebooks", "configs"]
for dirname in key_dirs:
    path = Path(dirname)
    if path.exists():
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve()
            print(f"✅ {dirname:15} → {target:30} (symlink)")
            print(f"   {'':15}   Resolves to: {resolved}")
        elif path.is_dir():
            print(f"✅ {dirname:15} (real directory)")
        else:
            print(f"⚠️  {dirname:15} (exists but not a directory)")
    else:
        print(f"❌ {dirname:15} (NOT FOUND)")
print()

# 4. Data Directory Deep Inspection
print("\n[4] 🔬 Data Directory Inspection")
print("─" * 80)
data_path = Path("data")
if data_path.exists():
    print(f"data/ exists: {(data_path.is_symlink() and 'SYMLINK') or 'REAL DIR'}")
    print(f"Resolved path: {data_path.resolve()}")

    # Check OAI subdirectory
    oai_path = data_path / "oai"
    if oai_path.exists():
        print("\n📊 data/oai/ structure:")

        # Check key subdirectories
        subdirs = [
            "img",
            "train",
            "valid",
            "test",
            "train/img",
            "valid/img",
            "test/img",
            "test/img/subset_4",
        ]
        for subdir in subdirs:
            p = oai_path / subdir
            if p.exists():
                count = len(list(p.glob("*.png"))) if p.is_dir() else 0
                print(f"  ✅ {subdir:25} ({count} PNG files)")
            else:
                print(f"  ❌ {subdir:25} (NOT FOUND)")

        # Check for split info CSVs
        print("\n📄 Split info files:")
        for csv_file in [
            "train_split_info.csv",
            "valid_split_info.csv",
            "test_split_info.csv",
        ]:
            csv_path = oai_path / csv_file
            if csv_path.exists():
                print(f"  ✅ {csv_file}")
            else:
                print(f"  ❌ {csv_file}")
    else:
        print("❌ data/oai/ not found")

    # Check pretrained directory
    pretrained_path = data_path / "pretrained"
    if pretrained_path.exists():
        model_files = list(pretrained_path.rglob("*.pth")) + list(
            pretrained_path.rglob("*.pt")
        )
        print(f"\n🤖 data/pretrained/: {len(model_files)} model files")

        for model_type in ["aot-gan", "ict", "repaint"]:
            model_dir = pretrained_path / model_type
            if model_dir.exists():
                count = len(
                    list(model_dir.rglob("*.pth")) + list(model_dir.rglob("*.pt"))
                )
                print(f"  ✅ {model_type:15} ({count} files)")
            else:
                print(f"  ❌ {model_type:15} (NOT FOUND)")
    else:
        print("❌ data/pretrained/ not found")
else:
    print("❌ data/ directory not found")

# 5. Results Directory Deep Inspection
print("\n[5] 💾 Results Directory Inspection")
print("─" * 80)
results_path = Path("results")
if results_path.exists():
    print(f"results/ exists: {(results_path.is_symlink() and 'SYMLINK') or 'REAL DIR'}")
    print(f"Resolved path: {results_path.resolve()}")

    # Check if results directory is writable
    test_file = results_path / ".test_write"
    try:
        test_file.touch()
        test_file.unlink()
        print("✅ Results directory is WRITABLE")
    except Exception as e:
        print(f"❌ Results directory is NOT WRITABLE: {e}")

    # Check for existing results
    if results_path.is_dir():
        subdirs = list(results_path.iterdir())
        if subdirs:
            print("\n📊 Existing results found:")
            for subdir in subdirs:
                if subdir.is_dir():
                    file_count = len(list(subdir.rglob("*")))
                    print(f"  • {subdir.name:25} ({file_count} files)")
        else:
            print("\n📊 Results directory is empty (ready for new results)")
else:
    print("❌ results/ directory not found")

# 6. Parent Directory Structure
print("\n[6] 🌳 Parent Directory Structure (Sibling Check)")
print("─" * 80)
parent = project_root.parent
print(f"Parent directory: {parent}")
print("\nSiblings:")
for item in sorted(parent.iterdir()):
    if item.is_dir():
        marker = "📁"
        if item.name == "OAI-inpainting":
            marker = "🎯"
        elif item.name == "OAI_untracked":
            marker = "💾"
        print(f"  {marker} {item.name}")

# Check expected sibling structure
oai_untracked = parent / "OAI_untracked"
if oai_untracked.exists():
    print("\n✅ OAI_untracked found as sibling!")
    print(f"   Path: {oai_untracked.resolve()}")

    # Check its contents
    if (oai_untracked / "data").exists():
        print("   ✅ data/ subdirectory exists")
    else:
        print("   ❌ data/ subdirectory missing")

    if (oai_untracked / "results").exists():
        print("   ✅ results/ subdirectory exists")
    else:
        print("   ⚠️  results/ subdirectory missing (will be created)")
else:
    print("❌ OAI_untracked NOT found as sibling!")
    print(f"   Expected at: {oai_untracked}")

# 7. Symlink Verification
print("\n[7] 🔗 Symlink Verification")
print("─" * 80)


def check_symlink(link_name, expected_target):
    """Check if symlink exists and points to expected target."""
    link_path = project_root / link_name

    if not link_path.exists():
        print(f"❌ {link_name}: Does not exist")
        return False

    if not link_path.is_symlink():
        print(f"⚠️  {link_name}: Exists but is NOT a symlink (real directory)")
        return False

    actual_target = os.readlink(link_path)
    resolved = link_path.resolve()

    print(f"✅ {link_name}:")
    print(f"   Link target: {actual_target}")
    print(f"   Resolves to: {resolved}")
    print(f"   Expected: {expected_target}")

    if actual_target == expected_target:
        print("   ✅ CORRECT relative path!")
        return True
    else:
        print("   ⚠️  Different target than expected")
        return False


check_symlink("data", "../OAI_untracked/data")
check_symlink("results", "../OAI_untracked/results")

# 8. Read/Write Test
print("\n[8] 🧪 Read/Write Test")
print("─" * 80)

# Test reading from data
test_read = Path("data/oai/img")
if test_read.exists():
    images = list(test_read.glob("*.png"))
    print(f"✅ READ TEST: Found {len(images)} images in data/oai/img/")
    if images:
        print(f"   Sample: {images[0].name}")
else:
    print("❌ READ TEST FAILED: Cannot access data/oai/img/")

# Test writing to results
test_write = Path("results/.diagnostic_test")
try:
    test_write.write_text("test")
    content = test_write.read_text()
    test_write.unlink()
    print("✅ WRITE TEST: Can write to results/ directory")
    print(f"   Test file: {test_write.resolve()}")
except Exception as e:
    print("❌ WRITE TEST FAILED: Cannot write to results/")
    print(f"   Error: {e}")

# 9. Path Comparison: Local vs Colab
print("\n[9] 📊 Path Comparison Summary")
print("─" * 80)

data_resolved = Path("data").resolve() if Path("data").exists() else None
results_resolved = Path("results").resolve() if Path("results").exists() else None

print("Expected structure (works everywhere):")
print("  Parent/")
print("  ├── OAI-inpainting/")
print("  │   ├── data -> ../OAI_untracked/data/")
print("  │   └── results -> ../OAI_untracked/results/")
print("  └── OAI_untracked/")
print("      ├── data/")
print("      └── results/")
print()

print("Current structure:")
print(f"  data/    → {data_resolved}")
print(f"  results/ → {results_resolved}")
print()

# Check if paths match expected pattern
data_correct = (
    str(data_resolved).endswith("OAI_untracked/data") if data_resolved else False
)
results_correct = (
    str(results_resolved).endswith("OAI_untracked/results")
    if results_resolved
    else False
)

if data_correct and results_correct:
    print("✅ PERFECT! Both data/ and results/ point to OAI_untracked/")
    print("   This matches local environment structure")
    print("   Results will persist in Google Drive!")
elif data_correct and not results_correct:
    print("⚠️  ISSUE: data/ is correct, but results/ doesn't match!")
    print("   data/ → OAI_untracked/data/ ✅")
    print(f"   results/ → {results_resolved} ❌")
    print("   ")
    print("   This means:")
    print("   • Reads will work (data is correct)")
    print("   • Writes will go to DIFFERENT location than local")
    print("   • Results may not persist in Google Drive")
    print("   ")
    print("   To fix: Delete results/ and re-run setup cell")
else:
    print("❌ MISMATCH: Directory structure doesn't match expected pattern")
    print(
        "   Expected both to end with 'OAI_untracked/data' and 'OAI_untracked/results'"
    )

# 10. Environment Summary
print("\n[10] 📋 Environment Summary")
print("─" * 80)

# Detect environment
try:
    import google.colab  # noqa: F401

    env = "Google Colab"
except ImportError:
    env = "Local Machine"

print(f"Environment: {env}")
print(f"Python: {sys.version.split()[0]}")
print(f"Working directory: {Path.cwd()}")
print(f"Project root: {project_root}")

# Check if torch is available
try:
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: Not installed")

print("\n" + "=" * 80)
print("🎉 DIAGNOSTICS COMPLETE!")
print("=" * 80)

# Final verdict
if data_correct and results_correct:
    print("\n✅ VERDICT: Perfect setup!")
    print("   • Directory structure matches local environment")
    print("   • Both data/ and results/ use relative symlinks")
    print("   • Results will persist in Google Drive")
    print("   • Ready for testing/training!")
elif data_correct:
    print("\n⚠️  VERDICT: Partial setup")
    print("   • data/ is correct (reads will work)")
    print("   • results/ needs fixing (writes won't match local)")
    print("   • Re-run Cell 1 (setup) to fix")
else:
    print("\n❌ VERDICT: Setup needed")
    print("   • Directory structure doesn't match expected pattern")
    print("   • Run Cell 1 (Complete Setup) in the notebook")
