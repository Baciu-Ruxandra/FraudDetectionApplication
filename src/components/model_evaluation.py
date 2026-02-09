import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from src.logger import logger


class ModelEvaluation:
    def __init__(self, model, df_test):
        self.model = model
        self.df_test = df_test

    def evaluate(self):
        logger.info("Starting model evaluation...")

        X_test = self.df_test.drop(columns=["is_fraud"])
        y_test = self.df_test["is_fraud"]

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc = roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)

        logger.info(f"ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred, digits=3)}")
        logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

        return {"roc_auc": roc, "pr_auc": pr}
