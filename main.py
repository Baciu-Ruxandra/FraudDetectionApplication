from src.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        STAGE_NAME = "Model Retraining"
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        train_pipeline = TrainingPipeline()
        train_pipeline.main()

        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logging.exception(e)
        raise e
