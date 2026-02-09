import pandas as pd
import joblib
from src.logger import logger


class FraudPredictionService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def predict(self, df_input: pd.DataFrame):
        logger.info(f"Running prediction on {len(df_input)} new samples.")
        y_prob = self.model.predict_proba(df_input)[:, 1]
        y_pred = (y_prob >= 0.57).astype(int)

        df_input = df_input.copy()
        df_input["fraud_probability"] = y_prob
        df_input["is_fraud_predicted"] = y_pred

        return df_input
