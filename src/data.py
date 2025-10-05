"""
Data utilities for the OAI inpainting project.
Platform-agnostic data loading and preprocessing.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OAIDataset(Dataset):
    """OAI dataset for inpainting tasks."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        subset: Optional[str] = None,
    ):
        """
        Initialize OAI dataset.

        Args:
            data_dir: Path to OAI data directory
            split: Dataset split ("train", "valid", "test")
            transform: Image transforms
            mask_transform: Mask transforms
            subset: Subset name (e.g., "subset_4")
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.subset = subset
        self.transform = transform
        self.mask_transform = mask_transform

        # Set up paths
        if subset:
            self.img_dir = self.data_dir / "test" / "img" / subset
            self.mask_dir = self.data_dir / "test" / "mask" / subset
            self.mask_inv_dir = self.data_dir / "test" / "mask" / "inv" / subset
            self.edge_dir = self.data_dir / "test" / "edge" / subset
        else:
            self.img_dir = self.data_dir / split / "img"
            self.mask_dir = self.data_dir / split / "mask"
            self.mask_inv_dir = self.data_dir / split / "mask" / "inv"
            self.edge_dir = self.data_dir / split / "edge"

        # Load image paths
        self.image_paths = sorted(list(self.img_dir.glob("*.png")))

        if not self.image_paths:
            raise ValueError(f"No images found in {self.img_dir}")

        # Load labels if available
        self.labels = self._load_labels()

        print(f"📁 Loaded {len(self.image_paths)} images from {self.img_dir}")

    def _load_labels(self) -> Optional[List[int]]:
        """Load labels for the dataset."""
        # Try to load from CSV file
        if self.subset:
            labels_file = self.data_dir / "test" / f"{self.subset}_info.csv"
        else:
            labels_file = self.data_dir / f"{self.split}_info.csv"

        if labels_file.exists():
            df = pd.read_csv(labels_file)
            if "is_osteo" in df.columns:
                return df["is_osteo"].astype(int).tolist()

        return None

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # Create dummy mask if not found
            mask = Image.new("L", image.size, 255)

        # Load inverted mask
        mask_inv_path = self.mask_inv_dir / img_path.name
        if mask_inv_path.exists():
            mask_inv = Image.open(mask_inv_path).convert("L")
        else:
            mask_inv = Image.new("L", image.size, 0)

        # Load edge map
        edge_path = self.edge_dir / img_path.name
        if edge_path.exists():
            edge = Image.open(edge_path).convert("L")
        else:
            edge = Image.new("L", image.size, 0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask_inv = self.mask_transform(mask_inv)
            edge = self.mask_transform(edge)

        # Prepare output
        output = {
            "image": image,
            "mask": mask,
            "mask_inv": mask_inv,
            "edge": edge,
            "filename": img_path.name,
            "path": str(img_path),
        }

        # Add label if available
        if self.labels is not None:
            output["label"] = self.labels[idx]

        return output


def get_default_transforms(
    image_size: int = 512, is_training: bool = True
) -> transforms.Compose:
    """Get default image transforms."""
    if is_training:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transform


def get_mask_transforms(image_size: int = 512) -> transforms.Compose:
    """Get mask transforms."""
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    subset: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.

    Args:
        data_dir: Path to OAI data directory
        batch_size: Batch size for data loaders
        image_size: Image size for transforms
        num_workers: Number of workers for data loading
        subset: Subset name (e.g., "subset_4")

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_default_transforms(image_size, is_training=True)
    val_transform = get_default_transforms(image_size, is_training=False)
    mask_transform = get_mask_transforms(image_size)

    # Create datasets
    if subset:
        # Use subset for testing
        test_dataset = OAIDataset(
            data_dir,
            split="test",
            transform=val_transform,
            mask_transform=mask_transform,
            subset=subset,
        )
        return (
            None,
            None,
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )
    else:
        # Use full dataset
        train_dataset = OAIDataset(
            data_dir,
            split="train",
            transform=train_transform,
            mask_transform=mask_transform,
        )
        val_dataset = OAIDataset(
            data_dir,
            split="valid",
            transform=val_transform,
            mask_transform=mask_transform,
        )
        test_dataset = OAIDataset(
            data_dir,
            split="test",
            transform=val_transform,
            mask_transform=mask_transform,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


def load_oai_metadata(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load OAI metadata."""
    data_dir = Path(data_dir)
    metadata_file = data_dir / "data.csv"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    return pd.read_csv(metadata_file)


def get_class_distribution(
    data_dir: Union[str, Path], split: str = "train"
) -> Dict[str, int]:
    """Get class distribution for a dataset split."""
    data_dir = Path(data_dir)

    # Try to load from CSV
    labels_file = data_dir / f"{split}_info.csv"
    if labels_file.exists():
        df = pd.read_csv(labels_file)
        if "is_osteo" in df.columns:
            distribution = df["is_osteo"].value_counts().to_dict()
            return {str(k): int(v) for k, v in distribution.items()}

    # Fallback: count files in directories
    img_dir = data_dir / split / "img"
    if img_dir.exists():
        total_images = len(list(img_dir.glob("*.png")))
        return {"unknown": total_images}

    return {"unknown": 0}


def validate_dataset(data_dir: Union[str, Path]) -> bool:
    """Validate dataset structure."""
    data_dir = Path(data_dir)

    required_dirs = ["train", "valid", "test"]
    required_subdirs = ["img", "mask"]

    for split in required_dirs:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"❌ Missing split directory: {split_dir}")
            return False

        for subdir in required_subdirs:
            subdir_path = split_dir / subdir
            if not subdir_path.exists():
                print(f"❌ Missing subdirectory: {subdir_path}")
                return False

            # Check if directory has images
            images = list(subdir_path.glob("*.png"))
            if not images:
                print(f"⚠️  No images found in: {subdir_path}")

    print("✅ Dataset structure validation passed")
    return True


def create_subset_info(data_dir: Union[str, Path], subset: str = "subset_4") -> None:
    """Create subset info file."""
    data_dir = Path(data_dir)
    subset_dir = data_dir / "test" / "img" / subset

    if not subset_dir.exists():
        print(f"❌ Subset directory not found: {subset_dir}")
        return

    # Get image files
    image_files = sorted(list(subset_dir.glob("*.png")))

    # Create info DataFrame
    info_data = []
    for img_file in image_files:
        # Extract label from filename or use dummy
        # This is a placeholder - you may need to adjust based on your naming convention
        is_osteo = 0  # Default value
        info_data.append({"filename": img_file.name, "is_osteo": is_osteo})

    # Save to CSV
    info_df = pd.DataFrame(info_data)
    info_file = data_dir / "test" / f"{subset}_info.csv"
    info_df.to_csv(info_file, index=False)

    print(f"✅ Created subset info file: {info_file}")
    print(f"📊 Subset contains {len(info_data)} images")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 Set random seed to {seed}")
