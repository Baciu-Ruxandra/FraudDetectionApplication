import pandas as pd
import joblib
import numpy as np
from src.config import MODEL_PATH
from src.components.data_preprocessing import preprocess_transactions

def load_model():
    return joblib.load(MODEL_PATH)

def run_inference(df_input: pd.DataFrame, model=None, threshold: float = 0.57):
    if model is None:
        model = load_model()
    df_proc = preprocess_transactions(df_input.copy())
    X = df_proc.drop(columns=["is_fraud"], errors="ignore")
    proba = model.predict_proba(X)[:, 1]
    df_output = df_input.copy()
    df_output["Fraud_Probability"] = (proba * 100).round(2)
    df_output["Final Decision"] = np.where(proba >= threshold, "High", "Low")
    return df_output
