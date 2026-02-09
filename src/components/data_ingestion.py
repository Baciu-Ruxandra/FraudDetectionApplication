import pandas as pd
from src.logger import logger
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process...")

        # Read train and test CSVs
        df_train = pd.read_csv(self.config.train_data_path)
        df_test = pd.read_csv(self.config.test_data_path)

        logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
        logger.info("Data ingestion completed successfully.")

        return df_train, df_test
