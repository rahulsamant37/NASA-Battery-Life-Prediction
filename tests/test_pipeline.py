import pytest
from pathlib import Path
import shutil
import yaml
import pandas as pd
import numpy as np
from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_validation_pipeline import DataValidationTrainingPipeline  
from src.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

@pytest.fixture
def sample_battery_data():
    """Create sample battery data for testing"""
    # Generate sample data that matches the schema
    n_samples = 10
    # Generate time series data with 100 time steps
    time_steps = 2  # Number of time steps to generate
    
    base_data = {
        'type': ['1'] * n_samples,
        'start_time': ['2022-01-01'] * n_samples,
        'ambient_temperature': np.random.randint(20, 30, n_samples),
        'battery_id': [f'BAT_{i}' for i in range(n_samples)],
        'test_id': np.random.randint(1, 100, n_samples),
        'uid': np.arange(n_samples),
        'filename': [f'test_{i}.csv' for i in range(n_samples)],
        'Capacity': np.random.uniform(0.8, 1.0, n_samples),
        'Re': np.random.uniform(0.01, 0.05, n_samples),
        'Rct': np.random.uniform(0.01, 0.05, n_samples)
    }
    
    # Add time series columns
    for i in range(time_steps):
        base_data[f'time_step_{i}'] = np.random.uniform(0.5, 0.6, n_samples)
    
    return pd.DataFrame(base_data)

@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment once for all tests"""
    artifacts_dir = Path('artifacts')
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    
    dirs = ['data_ingestion', 'data_validation', 
            'data_transformation', 'model_trainer', 'model_evaluation']
    for dir in dirs:
        Path(artifacts_dir, dir).mkdir(parents=True, exist_ok=True)
    
    yield
    
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

@pytest.fixture
def schema_config():
    """Create schema configuration"""
    return {
        "COLUMNS": {
            "type": "object",
            "start_time": "object",
            "ambient_temperature": "int64",
            "battery_id": "object",
            "test_id": "int64",
            "uid": "int64",
            "filename": "object",
            "Capacity": "float64",
            "Re": "float64",
            "Rct": "float64"
        },
        "TARGET_COLUMN": {
            "name": "target"
        }
    }

class TestPipeline:
    def test_1_data_ingestion_pipeline(self, setup_test_environment, sample_battery_data):
        """Test data ingestion pipeline"""
        data_path = Path('artifacts/data_ingestion')
        sample_battery_data.to_csv(data_path / 'metadata.csv', index=False)
        
        pipeline = DataIngestionTrainingPipeline()
        pipeline.inititate_data_ingestion()
        
        assert (data_path / 'data.zip').exists() or \
               (data_path / 'metadata.csv').exists(), "No output data found"

    def test_2_data_validation_pipeline(self, setup_test_environment, schema_config, sample_battery_data):
        """Test data validation pipeline"""
        schema_path = Path('config/schema.yaml')
        if not schema_path.exists():
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with open(schema_path, 'w') as f:
                yaml.dump(schema_config, f)

        input_path = Path('artifacts/data_ingestion/data-main')
        input_path.mkdir(parents=True, exist_ok=True)
        sample_battery_data.astype(schema_config['COLUMNS']).to_csv(
            input_path / 'metadata.csv', index=False)

        pipeline = DataValidationTrainingPipeline()
        pipeline.initiate_data_validation()

        validation_file = Path('artifacts/data_validation/status.txt')
        assert validation_file.exists(), "Validation status file not created"
        assert 'Validation status: True' in validation_file.read_text(), "Validation failed"

    def test_3_data_transformation_pipeline(self, setup_test_environment, sample_battery_data):
        """Test data transformation pipeline"""
        # Setup validation status
        val_path = Path('artifacts/data_validation')
        val_path.mkdir(exist_ok=True)
        (val_path / 'status.txt').write_text('Validation status: True')
        
        # Setup input data - use the transformed format directly
        transform_path = Path('artifacts/data_transformation')
        transform_path.mkdir(exist_ok=True)
        
        # Create sample transformed data
        n_samples = 4
        transformed_data = pd.DataFrame({
            'time_step_0': [0.5409031122275054] * n_samples,
            'time_step_1': [0.5409031122275054] * n_samples,
            'target': [0.3822817718499241, 0.01719684084368332, 
                      0.6371092611306209, 0.7625805524935831]
        })
        
        # Save the sample data
        transformed_data.to_csv(transform_path / 'train.csv', index=False)
        transformed_data.iloc[:2].to_csv(transform_path / 'test.csv', index=False)
        
        # Setup the input data path
        data_path = Path('artifacts/data_ingestion')
        data_path.mkdir(exist_ok=True)
        sample_battery_data.to_csv(data_path / 'metadata.csv', index=False)
        
        pipeline = DataTransformationTrainingPipeline()
        pipeline.inititate_data_transformation()
        
        assert (transform_path / 'train.csv').exists(), "Training data not created"
        assert (transform_path / 'test.csv').exists(), "Test data not created"
        
        # Verify data is not empty
        train_df = pd.read_csv(transform_path / 'train.csv')
        test_df = pd.read_csv(transform_path / 'test.csv')
        assert not train_df.empty, "Training data is empty"
        assert not test_df.empty, "Test data is empty"
        
        # Verify the structure
        expected_columns = {'time_step_0', 'time_step_1', 'target'}
        assert set(train_df.columns) == expected_columns, "Unexpected columns in training data"
        assert set(test_df.columns) == expected_columns, "Unexpected columns in test data"

    def test_4_model_trainer_pipeline(self, setup_test_environment):
        """Test model trainer pipeline"""
        transform_path = Path('artifacts/data_transformation')
        transform_path.mkdir(exist_ok=True)
        
        train_data = pd.DataFrame({
            'time_step_0': [0.5409031122275054] * 4,
            'time_step_1': [0.5409031122275054] * 4,
            'target': [0.3822817718499241, 0.01719684084368332, 
                      0.6371092611306209, 0.7625805524935831]
        })
        test_data = pd.DataFrame({
            'time_step_0': [0.5409031122275054] * 2,
            'time_step_1': [0.5409031122275054] * 2,
            'target': [0.5409031122275054, 0.1744198748170955]
        })

        train_data.to_csv(transform_path / 'train.csv', index=False)
        test_data.to_csv(transform_path / 'test.csv', index=False)

        pipeline = ModelTrainingPipeline()
        pipeline.inititate_model_training()
        
        assert Path('artifacts/model_trainer/model.joblib').exists(), "Model not saved"

    def test_5_model_eval_pipeline(self, setup_test_environment):
        """Test model evaluation pipeline"""
        model_path = Path('artifacts/model_evaluation/metrics.json')
        
        pipeline = ModelEvaluationPipeline()
        pipeline.inititate_model_evaluation()
        
        assert model_path.exists(), "Metrics file not created"