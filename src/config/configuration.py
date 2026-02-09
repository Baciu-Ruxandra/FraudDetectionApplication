import os
from pathlib import Path
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelTrainerConfig
)
from src.logger import logger


class ConfigurationManager:
    def __init__(self, config_filepath: str = "config/config.yaml"):
        self.config = read_yaml(config_filepath)
        create_directories(["artifacts"])
        logger.info("ConfigurationManager initialized successfully.")

    # --- DATA INGESTION CONFIG ---
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([cfg["root_dir"]])
        return DataIngestionConfig(
            root_dir=cfg["root_dir"],
            train_data_path=cfg["train_data_path"],
            test_data_path=cfg["test_data_path"]
        )

    # --- DATA PREPROCESSING CONFIG ---
    def get_preprocessing_config(self) -> DataPreprocessingConfig:
        cfg = self.config["data_preprocessing"]
        create_directories([cfg["root_dir"]])
        return DataPreprocessingConfig(
            root_dir=cfg["root_dir"],
            preprocessed_train_path=cfg["preprocessed_train_path"],
            preprocessed_test_path=cfg["preprocessed_test_path"]
        )
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        create_directories([cfg["root_dir"]])
        return ModelTrainerConfig(
            root_dir=cfg["root_dir"],
            model_path=cfg["model_path"],
            threshold=cfg["threshold"],
            smote_ratio=cfg["smote_ratio"],
            random_state=cfg["random_state"],
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"]
        )


