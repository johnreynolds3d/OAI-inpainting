"""
Data versioning and lineage tracking utilities.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DataVersioning:
    """Data versioning and lineage tracking."""

    def __init__(self, version_dir: Path):
        """
        Initialize data versioning.

        Args:
            version_dir: Directory to store version information
        """
        self.version_dir = version_dir
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.version_dir / "data_versions.json"

        if self.version_file.exists():
            with open(self.version_file) as f:
                self.versions = json.load(f)
        else:
            self.versions = {}

    def create_version(
        self,
        data_path: Path,
        version_name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new data version.

        Args:
            data_path: Path to the data
            version_name: Name of the version
            description: Description of the version
            metadata: Additional metadata

        Returns:
            Version ID
        """
        version_id = f"{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate data hash
        data_hash = self._calculate_hash(data_path)

        # Get file information
        file_info = self._get_file_info(data_path)

        version_info = {
            "version_id": version_id,
            "version_name": version_name,
            "description": description,
            "data_path": str(data_path),
            "data_hash": data_hash,
            "file_info": file_info,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        self.versions[version_id] = version_info
        self._save_versions()

        return version_id

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version information."""
        return self.versions.get(version_id)

    def list_versions(self) -> Dict[str, Any]:
        """List all versions."""
        return self.versions

    def verify_version(self, version_id: str, data_path: Path) -> bool:
        """
        Verify data integrity for a version.

        Args:
            version_id: Version ID to verify
            data_path: Current data path

        Returns:
            True if data is intact, False otherwise
        """
        version_info = self.get_version(version_id)
        if not version_info:
            return False

        current_hash = self._calculate_hash(data_path)
        return current_hash == version_info["data_hash"]

    def _calculate_hash(self, data_path: Path) -> str:
        """Calculate hash of data directory."""
        hasher = hashlib.sha256()

        if data_path.is_file():
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        elif data_path.is_dir():
            for file_path in sorted(data_path.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)

        return hasher.hexdigest()

    def _get_file_info(self, data_path: Path) -> Dict[str, Any]:
        """Get file information for data path."""
        if data_path.is_file():
            stat = data_path.stat()
            return {
                "type": "file",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        elif data_path.is_dir():
            files = list(data_path.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            return {
                "type": "directory",
                "file_count": len(files),
                "total_size": total_size,
                "modified": datetime.fromtimestamp(
                    data_path.stat().st_mtime
                ).isoformat(),
            }

        return {"type": "unknown"}

    def _save_versions(self) -> None:
        """Save versions to file."""
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)


class DataLineage:
    """Data lineage tracking."""

    def __init__(self, lineage_dir: Path):
        """
        Initialize data lineage tracking.

        Args:
            lineage_dir: Directory to store lineage information
        """
        self.lineage_dir = lineage_dir
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_file = self.lineage_dir / "data_lineage.json"

        if self.lineage_file.exists():
            with open(self.lineage_file) as f:
                self.lineage = json.load(f)
        else:
            self.lineage = {}

    def add_transformation(
        self,
        input_paths: List[Path],
        output_path: Path,
        transformation_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a data transformation to lineage.

        Args:
            input_paths: Input data paths
            output_path: Output data path
            transformation_type: Type of transformation
            parameters: Transformation parameters
            metadata: Additional metadata

        Returns:
            Transformation ID
        """
        transformation_id = (
            f"{transformation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        transformation_info = {
            "transformation_id": transformation_id,
            "transformation_type": transformation_type,
            "input_paths": [str(p) for p in input_paths],
            "output_path": str(output_path),
            "parameters": parameters,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        self.lineage[transformation_id] = transformation_info
        self._save_lineage()

        return transformation_id

    def get_lineage(self, data_path: Path) -> List[Dict[str, Any]]:
        """
        Get lineage for a data path.

        Args:
            data_path: Data path to get lineage for

        Returns:
            List of transformations
        """
        data_path_str = str(data_path)
        lineage = []

        for transformation in self.lineage.values():
            if (
                data_path_str in transformation["input_paths"]
                or data_path_str == transformation["output_path"]
            ):
                lineage.append(transformation)

        return sorted(lineage, key=lambda x: x["timestamp"])

    def _save_lineage(self) -> None:
        """Save lineage to file."""
        with open(self.lineage_file, "w") as f:
            json.dump(self.lineage, f, indent=2)
