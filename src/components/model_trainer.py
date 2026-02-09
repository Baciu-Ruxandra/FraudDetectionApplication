import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)
from src.logger import logger
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_and_save_model(self, df: pd.DataFrame):
        logger.info("Starting model training...")

        numeric_features = [
            "lat", "long", "merch_lat", "merch_long", "city_pop",
            "age", "hour", "day", "month"
        ]
        amt_feature = ["amt"]
        categorical_features = ["category", "gender"]
        target_col = "is_fraud"

        X = df[numeric_features + amt_feature + categorical_features]
        y = df[target_col]

        amt_log_and_scale = Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
            ("scale", StandardScaler())
        ])

        preprocess = ColumnTransformer([
            ("amt_log_scaled", amt_log_and_scale, amt_feature),
            ("num_scaled", StandardScaler(), numeric_features),
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
        smote = SMOTE(sampling_strategy=self.config.smote_ratio, random_state=self.config.random_state, k_neighbors=3)

        rf_model = RandomForestClassifier(
            n_estimators=57,
            max_depth=17,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=7,
            bootstrap=False,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.config.random_state
        )

        xgb_model = XGBClassifier(
            n_estimators=1295,
            learning_rate=0.06075805702761636,
            max_depth=4,
            min_child_weight=0.6170217483403153,
            gamma=5.324289357128436,
            subsample=0.6353970008207678,
            colsample_bytree=0.6739417822102108,
            reg_alpha=0.09761125443110447,
            reg_lambda=48.696409415209004,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            n_jobs=-1,
            random_state=self.config.random_state,
            eval_metric="logloss"
        )


        rf_pipeline = ImbPipeline([
            ("preprocess", preprocess),
            ("smote", smote),
            ("rf", rf_model)
        ])

        xgb_pipeline = ImbPipeline([
            ("preprocess", preprocess),
            ("smote", smote),
            ("xgb", xgb_model)
        ])

        logger.info("Evaluating base models with cross-validation...")
        cv_rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_xgb_scores = cross_val_score(xgb_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        logger.info(f"RF mean ROC-AUC: {cv_rf_scores.mean():.4f}")
        logger.info(f"XGB mean ROC-AUC: {cv_xgb_scores.mean():.4f}")

        rf_pipeline.fit(X, y)
        xgb_pipeline.fit(X, y)

        voter = VotingClassifier(
            estimators=[("rf", rf_pipeline), ("xgb", xgb_pipeline)],
            voting="soft"
        )

        oof_proba = cross_val_predict(voter, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        pr_auc = average_precision_score(y, oof_proba)
        roc_auc = roc_auc_score(y, oof_proba)
        logger.info(f"OOF PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

        target_recall = 0.95
        prec, rec, thr = precision_recall_curve(y, oof_proba)
        mask = rec[:-1] >= target_recall
        if mask.any():
            best_t = float(thr[mask][np.argmax(prec[:-1][mask])])
        else:
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            best_t = float(thr[np.argmax(f1)])
        logger.info(f"Chosen threshold: {best_t:.4f} (target recall={target_recall})")

        voter.fit(X, y)
        y_pred = (voter.predict_proba(X)[:, 1] >= best_t).astype(int)
        logger.info(f"Classification report:\n{classification_report(y, y_pred, digits=3)}")
        logger.info(f"Confusion matrix:\n{confusion_matrix(y, y_pred)}")

        joblib.dump(voter, self.config.model_path)
        logger.info(f"Model saved successfully at: {self.config.model_path}")

        return voter
