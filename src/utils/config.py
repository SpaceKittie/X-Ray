"""Utilities for loading configuration files."""
import os
import yaml
from pathlib import Path

def load_paths():
    """Load paths from config/paths.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "paths.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
