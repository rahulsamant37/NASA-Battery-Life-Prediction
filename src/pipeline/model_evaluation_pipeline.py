from src import logger
from pathlib import Path
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEval

STAGE_NAME="Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def inititate_model_evaluation(self):
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_eval_config()
            model_eval = ModelEval(config=model_eval_config)
            model_eval.log_into_mlflow()
        except Exception as e:
            raise e