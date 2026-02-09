from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logger


class TrainingPipeline:
    def __init__(self):
        logger.info("Initializing TrainingPipeline...")
        self.config_manager = ConfigurationManager()
        logger.info("ConfigurationManager initialized successfully.")

    def main(self):
        logger.info("🚀 Starting training pipeline")

        # 1️⃣ Data ingestion
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        df_train, df_test = data_ingestion.initiate_data_ingestion()

        # 2️⃣ Preprocessing
        preprocessing_config = self.config_manager.get_preprocessing_config()
        preprocessor = DataPreprocessor(preprocessing_config)
        df_train_prep = preprocessor.preprocess_transactions(df_train, is_train=True)
        df_test_prep = preprocessor.preprocess_transactions(df_test, is_train=False)

        # 3️⃣ Model training
        trainer_config = self.config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(trainer_config)
        model = model_trainer.train_and_save_model(df_train_prep)

        # 4️⃣ Evaluation
        evaluator = ModelEvaluation(model, df_test_prep)
        results = evaluator.evaluate()
        logger.info(f"Training pipeline completed successfully. Metrics: {results}")

if __name__ == "__main__":
    try:
        logger.info(">>> Starting Fraud Detection Training Pipeline <<<")
        pipeline = TrainingPipeline()
        pipeline.main()
        logger.info(">>> Pipeline completed successfully <<<")
    except Exception as e:
        logger.exception(e)
        raise e
