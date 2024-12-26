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
