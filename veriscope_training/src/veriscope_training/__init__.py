"""VeriScope server-side phishing model training pipeline."""

from veriscope_training.config import AppConfig, DatasetSourceConfig
from veriscope_training.paths import ProjectPaths, discover_project_root

__all__ = [
    "AppConfig",
    "DatasetSourceConfig",
    "ProjectPaths",
    "discover_project_root",
]

__version__ = "0.1.0"
