import yaml
import os
from pathlib import Path

def read_yaml(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def create_directories(paths: list):
    for path in paths:
        os.makedirs(path, exist_ok=True)
