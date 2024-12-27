from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE: str
    unzip_data_dir:Path
    all_schema:dict

@dataclass
class DataTransformationConfig:
    root_dir:Path
    data_path: Path
    seq_length: int = 2
    test_size: float = 0.25
    random_state: int = 42

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    validation_split: float
    gru_units: int
    dropout_rate: float
    target_column: str