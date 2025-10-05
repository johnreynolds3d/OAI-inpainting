#!/usr/bin/env python3
"""
OAI Inpainting Project Data Setup Script

This script populates a newly-cloned instance of the OAI-inpainting project
with untracked data files (images and pretrained models) from a separate directory.

Usage:
    python setup_data.py [--source-dir PATH] [--dry-run] [--force]

Requirements:
    - The untracked data should be organized in the following structure:
      OAI_untracked/
      ├── data/
      │   ├── oai/
      │   │   └── img/          # OAI X-ray images
      │   └── pretrained/       # Pretrained models
      │       ├── aot-gan/
      │       ├── ict/
      │       └── repaint/

    - Run this script from the project root directory
    - Ensure you have sufficient disk space for the data
"""

import argparse
import contextlib
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple


def get_project_root() -> Path:
    """Get the project root directory (where this script is located)."""
    return Path(__file__).parent.absolute()


def find_untracked_dir(start_path=None) -> Path:
    """
    Find the untracked data directory.

    Args:
        start_path: Starting path for search. If None, searches from project root.

    Returns:
        Path to the untracked directory.

    Raises:
        FileNotFoundError: If untracked directory is not found.
    """
    if start_path is None:
        start_path = get_project_root()

    # Common names for untracked directories
    possible_names = [
        "OAI_untracked",
        "OAI_untracked_data",
        "untracked_data",
        "data_untracked",
    ]

    # Search in common locations
    search_paths = [
        start_path.parent,  # Parent of project directory
        start_path,  # Project directory itself
        Path.home(),  # User home directory
        Path.cwd(),  # Current working directory
    ]

    for search_path in search_paths:
        for name in possible_names:
            candidate = search_path / name
            if candidate.exists() and candidate.is_dir():
                # Verify it has the expected structure
                if (candidate / "data" / "oai" / "img").exists():
                    return candidate

    raise FileNotFoundError(
        f"Could not find untracked data directory. "
        f"Searched for: {possible_names} in {[str(p) for p in search_paths]}"
    )


def verify_untracked_structure(untracked_dir: Path) -> Dict[str, bool]:
    """
    Verify the structure of the untracked directory.

    Args:
        untracked_dir: Path to the untracked directory.

    Returns:
        Dictionary with verification results.
    """
    required_paths = {
        "oai_images": untracked_dir / "data" / "oai" / "img",
        "aot_gan_models": untracked_dir / "data" / "pretrained" / "aot-gan",
        "ict_models": untracked_dir / "data" / "pretrained" / "ict",
        "repaint_models": untracked_dir / "data" / "pretrained" / "repaint",
    }

    results = {}
    for name, path in required_paths.items():
        results[name] = path.exists() and path.is_dir()

    return results


def get_file_count(path: Path) -> int:
    """Get the number of files in a directory (recursive)."""
    if not path.exists():
        return 0
    return len(list(path.rglob("*"))) if path.is_dir() else 1


def estimate_size(path: Path) -> str:
    """Estimate the size of a directory in human-readable format."""
    if not path.exists():
        return "0 B"

    total_size = 0
    if path.is_file():
        total_size = path.stat().st_size
    else:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

    # Convert to human-readable format
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size = int(total_size / 1024.0)

    return f"{total_size:.1f} PB"


def copy_directory(src: Path, dst: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Copy a directory from source to destination.

    Args:
        src: Source directory path.
        dst: Destination directory path.
        dry_run: If True, only simulate the copy operation.

    Returns:
        Tuple of (files_copied, files_skipped).
    """
    files_copied = 0
    files_skipped = 0

    if not src.exists():
        print(f"⚠️  Warning: Source directory does not exist: {src}")
        return files_copied, files_skipped

    # Create destination directory
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Copy files
    for src_file in src.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src)
            dst_file = dst / rel_path

            if dst_file.exists():
                files_skipped += 1
                if not dry_run:
                    print(f"⏭️  Skipped (exists): {rel_path}")
            else:
                files_copied += 1
                if not dry_run:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    print(f"✅ Copied: {rel_path}")
                else:
                    print(f"🔄 Would copy: {rel_path}")

    return files_copied, files_skipped


def setup_project_data(
    untracked_dir: Path, project_root: Path, dry_run: bool = False, force: bool = False
) -> bool:
    """
    Set up project data by copying from untracked directory.

    Args:
        untracked_dir: Path to untracked data directory.
        project_root: Path to project root directory.
        dry_run: If True, only simulate the operations.
        force: If True, overwrite existing files.

    Returns:
        True if setup was successful, False otherwise.
    """
    print("🚀 Setting up OAI Inpainting project data...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Untracked data: {untracked_dir}")
    print(f"🔍 Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Verify untracked structure
    verification = verify_untracked_structure(untracked_dir)
    print("📋 Verifying untracked data structure:")
    for component, exists in verification.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {component}: {'Found' if exists else 'Missing'}")
    print()

    if not all(verification.values()):
        print(
            "❌ Some required components are missing. "
            "Please check your untracked data structure."
        )
        return False

    # Define copy operations
    copy_operations = [
        {
            "name": "OAI Images",
            "src": untracked_dir / "data" / "oai" / "img",
            "dst": project_root / "data" / "oai" / "img",
            "required": True,
        },
        {
            "name": "AOT-GAN Models",
            "src": untracked_dir / "data" / "pretrained" / "aot-gan",
            "dst": project_root / "data" / "pretrained" / "aot-gan",
            "required": False,
        },
        {
            "name": "ICT Models",
            "src": untracked_dir / "data" / "pretrained" / "ict",
            "dst": project_root / "data" / "pretrained" / "ict",
            "required": False,
        },
        {
            "name": "RePaint Models",
            "src": untracked_dir / "data" / "pretrained" / "repaint",
            "dst": project_root / "data" / "pretrained" / "repaint",
            "required": False,
        },
    ]

    # Show size estimates
    print("📊 Data size estimates:")
    total_size = 0
    for op in copy_operations:
        if op["src"].exists():
            size_str = estimate_size(op["src"])
            file_count = get_file_count(op["src"])
            print(f"  {op['name']}: {size_str} ({file_count} files)")
            # Add to total (rough estimate)
            with contextlib.suppress(Exception):
                total_size += (
                    op["src"].stat().st_size
                    if op["src"].is_file()
                    else sum(
                        f.stat().st_size for f in op["src"].rglob("*") if f.is_file()
                    )
                )
    print()

    # Check available disk space
    try:
        import shutil

        free_space = shutil.disk_usage(project_root).free
        if total_size > free_space:
            size_str = estimate_size(Path("/tmp"))
            print(
                f"⚠️  Warning: Estimated data size ({size_str}) "
                f"may exceed available disk space ({size_str})"
            )
            if not force:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != "y":
                    return False
    except Exception:
        pass

    # Perform copy operations
    total_copied = 0
    total_skipped = 0

    for op in copy_operations:
        print(f"📦 Processing {op['name']}...")

        if not op["src"].exists():
            if op["required"]:
                print(f"❌ Required component missing: {op['name']}")
                return False
            else:
                print(f"⏭️  Optional component missing: {op['name']}")
                continue

        copied, skipped = copy_directory(op["src"], op["dst"], dry_run)
        total_copied += copied
        total_skipped += skipped

        print(f"  📄 Files: {copied} copied, {skipped} skipped")
        print()

    # Summary
    print("📋 Setup Summary:")
    print(f"  📄 Total files copied: {total_copied}")
    print(f"  ⏭️  Total files skipped: {total_skipped}")

    if dry_run:
        print("🔍 This was a dry run. No files were actually copied.")
        print("   Run without --dry-run to perform the actual copy.")
    else:
        print("✅ Data setup completed successfully!")
        print(
            "   You can now run the split.py script to generate "
            "train/validation/test splits."
        )

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up OAI Inpainting project with untracked data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_data.py                           # Auto-detect untracked directory
  python setup_data.py --source-dir ../OAI_untracked  # Specify source directory
  python setup_data.py --dry-run                 # Preview what would be copied
  python setup_data.py --force                   # Overwrite existing files
        """,
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        help="Path to the untracked data directory (auto-detected if not specified)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without actually copying files",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting",
    )

    args = parser.parse_args()

    # Get project root
    project_root = get_project_root()

    # Find untracked directory
    try:
        if args.source_dir:
            untracked_dir = Path(args.source_dir).resolve()
            if not untracked_dir.exists():
                print(f"❌ Error: Source directory does not exist: {untracked_dir}")
                sys.exit(1)
        else:
            untracked_dir = find_untracked_dir()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - Ensure your untracked data is in a directory named 'OAI_untracked'")
        print("   - Use --source-dir to specify the exact path")
        print("   - Check that the directory contains the expected structure")
        sys.exit(1)

    # Run setup
    success = setup_project_data(untracked_dir, project_root, args.dry_run, args.force)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
