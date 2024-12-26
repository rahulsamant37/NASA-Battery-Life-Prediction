from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src import logger
from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def inititate_data_transformation(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"),'r') as f:
                status=f.read().split(" ")[-1]
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                transformer = DataTransformation(config=data_transformation_config)
                train_df, test_df = transformer.train_test_spliting()
            else:
                raise Exception("Your data scheme is not valid")
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.inititate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e