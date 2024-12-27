from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvalConfig)
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.environ.get('MLFLOW_TRACKING_URI', 'https://dagshub.com/rahulsamantcoc2/NASA-Battery-Life-Prediction.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('MLFLOW_TRACKING_USERNAME', 'rahulsamantcoc2')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('MLFLOW_TRACKING_PASSWORD', '33607bcb15d4e7a7cca29f0f443d16762cc15549')

class ConfigurationManager:
    def __init__(self,
                 config_filepath= CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])
        data_tranformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
        return data_tranformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config=self.config.model_trainer
        params=self.params.model_trainer
        schema=self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            epochs=params.epochs,
            batch_size=params.batch_size,
            optimizer=params.optimizer,
            loss=params.loss,
            validation_split=params.validation_split,
            gru_units=params.gru_units,
            dropout_rate=params.dropout_rate,
            target_column=schema.name
        )
        return model_trainer_config
    
    def get_model_eval_config(self) -> ModelEvalConfig:
        config=self.config.model_evaluation
        params=self.params.model_trainer
        schema=self.schema.TARGET_COLUMN
        create_directories([config.root_dir])
        model_eval_config=ModelEvalConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri=os.environ['MLFLOW_TRACKING_URI']
        )
        return model_eval_config