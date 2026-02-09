import pandas as pd
import numpy as np
from src.logger import logger
from src.entity.config_entity import DataPreprocessingConfig


class DataPreprocessor:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def preprocess_transactions(self, df: pd.DataFrame, is_train=True) -> pd.DataFrame:
        logger.info("Preprocessing dataset: creating time and age features.")
        df = df.copy()

        # Parse timestamps and date of birth
        ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        dob = pd.to_datetime(df["dob"], errors="coerce")

        # Derived temporal features
        df["hour"] = ts.dt.hour
        df["day"] = ts.dt.dayofweek
        df["month"] = ts.dt.month

        # Compute integer age
        ref = ts.fillna(pd.Timestamp.utcnow())
        df["age"] = ((ref - dob).dt.days / 365).astype(int)

        # Drop rows without target
        if "is_fraud" in df.columns:
            df = df.dropna(subset=["is_fraud"]).copy()
            df["is_fraud"] = df["is_fraud"].astype(int)

        # Drop unused columns
        df.drop(columns=[c for c in ["trans_date_trans_time", "dob"] if c in df.columns],
                inplace=True, errors="ignore")

        logger.info(f"Preprocessing completed. Final shape: {df.shape}")

        # Save to proper file
        try:
            output_path = (
                self.config.preprocessed_train_path
                if is_train
                else self.config.preprocessed_test_path
            )
            df.to_csv(output_path, index=False)
            logger.info(f"Preprocessed data saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save preprocessed data: {e}")

        return df
