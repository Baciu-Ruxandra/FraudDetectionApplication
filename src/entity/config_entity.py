from dataclasses import dataclass
from pathlib import Path

# --- DATA INGESTION CONFIG ---
@dataclass
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path

# --- DATA PREPROCESSING CONFIG ---
@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    preprocessed_train_path: Path
    preprocessed_test_path: Path

# --- MODEL TRAINER CONFIG ---
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    threshold: float
    smote_ratio: float
    random_state: int
    n_estimators: int
    max_depth: int
    learning_rate: float
