import pandas as pd
from src.components.prediction_service import FraudPredictionService
from src.config.configuration import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessor
from src.logger import logger


class PredictionPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        model_cfg = self.config_manager.get_model_trainer_config()
        self.model_path = model_cfg.model_path
        self.preprocessor = DataPreprocessor(self.config_manager.get_preprocessing_config())
        self.service = FraudPredictionService(model_path=self.model_path)

    def predict(self, csv_path: str):
        logger.info(f"Running prediction pipeline for file: {csv_path}")

        # 1️⃣ Load data
        df_new = pd.read_csv(csv_path)

        # 2️⃣ Preprocess it (same feature engineering)
        df_prep = self.preprocessor.preprocess_transactions(df_new)

        # 3️⃣ Run inference
        results = self.service.predict(df_prep)

        logger.info(f"Prediction completed on {len(results)} samples.")
        return results
    
    def predict_df(self, df_input: pd.DataFrame):
        df_prep = self.preprocessor.preprocess_transactions(df_input)
        results = self.service.predict(df_prep)
        return results


