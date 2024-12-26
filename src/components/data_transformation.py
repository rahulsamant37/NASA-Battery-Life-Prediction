import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import os
import joblib
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Tuple
from src import logger
from src.utils.common import create_directories
from src.entity.config_entity import (DataTransformationConfig)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.encoders = {}
        create_directories([self.config.root_dir])
    
    def process_start_time(self, time_str):
        try:
            if isinstance(time_str, str):
                time_vals = [float(x) for x in time_str.strip('[]').split()]
            else:
                time_vals = time_str
                
            year, month, day, hour, minute, second = map(float, time_vals)
            return pd.Timestamp(int(year), int(month), int(day), 
                              int(hour), int(minute), int(second))
        except Exception as e:
            logger.warning(f"Error processing time: {e}")
            return pd.NaT

    def clean_and_encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            
            numeric_cols = ['Capacity', 'Re', 'Rct']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            label_encoder = LabelEncoder()
            df['type'] = label_encoder.fit_transform(df['type'])
            self.encoders['type_encoder'] = label_encoder
            
            df['battery_id'] = df['battery_id'].str.replace('B', '').astype(int)
            df['start_time'] = df['start_time'].apply(self.process_start_time)
            
            df['hour'] = df['start_time'].dt.hour
            df['day'] = df['start_time'].dt.day
            df['month'] = df['start_time'].dt.month
            
            df['ambient_temperature'] = pd.to_numeric(df['ambient_temperature'], errors='coerce')
            
            numeric_columns = ['ambient_temperature', 'Capacity', 'Re', 'Rct']
            for col in numeric_columns:
                if col in df.columns:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.encoders[f'{col}_scaler'] = scaler
            
            logger.info("Data cleaning and encoding completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in clean_and_encode_data: {e}")
            raise e

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.config.seq_length):
            sequence = data[i:i + self.config.seq_length + 1]
            if not np.any(np.isnan(sequence)):
                X.append(sequence[:-1])
                y.append(sequence[-1])
        return np.array(X), np.array(y)

    def save_data(self, X: np.ndarray, y: np.ndarray, filename: str) -> pd.DataFrame:
        X_2d = X.reshape(X.shape[0], -1)
        feature_cols = [f'time_step_{i}' for i in range(X.shape[1])]
        
        X_df = pd.DataFrame(X_2d, columns=feature_cols)
        y_df = pd.DataFrame(y, columns=['target'])
        
        combined_df = pd.concat([X_df, y_df], axis=1)
        save_path = os.path.join(self.config.root_dir, filename)
        combined_df.to_csv(save_path, index=False)
        
        logger.info(f"Data saved to: {save_path}")
        return combined_df

    def train_test_spliting(self):
        try:
            logger.info("Started data transformation")
            
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Read data from {self.config.data_path}, shape: {data.shape}")
            
            processed_df = self.clean_and_encode_data(data)
            capacity_data = processed_df['Capacity'].values
            
            X, y = self.create_sequences(capacity_data)
            
            if len(X) == 0:
                raise ValueError("No valid sequences could be created. Check if there are enough consecutive non-null Capacity values.")
            
            logger.info(f"Created sequences with shape X: {X.shape}, y: {y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            train_df = self.save_data(X_train, y_train, "train.csv")
            test_df = self.save_data(X_test, y_test, "test.csv")
            
            # Save encoders using the imported save_bin function
            encoder_path = Path(self.config.root_dir) / "encoders.joblib"
            joblib.dump(self.encoders, encoder_path)
            
            logger.info("Data transformation completed")
            logger.info(f"Training set shape: {train_df.shape}")
            logger.info(f"Test set shape: {test_df.shape}")
            
            return train_df, test_df

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e