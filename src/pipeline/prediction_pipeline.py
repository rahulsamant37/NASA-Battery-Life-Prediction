import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PredictionPipeline:
    def __init__(self):
        self.encoder = joblib.load(Path('artifacts/data_transformation/encoders.joblib'))
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def preprocess_data(self, input_data: pd.DataFrame) -> np.ndarray:
        try:
            df = input_data.copy()
            
            # Convert numeric columns with proper error handling
            numeric_cols = ['Re', 'Rct']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Apply the same scaling as training
                if f'{col}_scaler' in self.encoder:
                    scaler = self.encoder[f'{col}_scaler']
                    df[col] = scaler.transform(df[[col]])

            # Process type column
            if 'type' in df.columns:
                df['type'] = self.encoder['type_encoder'].transform(df['type'])
            
            # Process temperature if present
            if 'ambient_temperature' in df.columns:
                df['ambient_temperature'] = pd.to_numeric(df['ambient_temperature'], errors='coerce')
                if 'ambient_temperature_scaler' in self.encoder:
                    scaler = self.encoder['ambient_temperature_scaler']
                    df['ambient_temperature'] = scaler.transform(df[['ambient_temperature']])

            # Extract time features
            if 'start_time' in df.columns:
                df['start_time'] = df['start_time'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
                df['hour'] = df['start_time'].apply(lambda x: x[3])
                df['day'] = df['start_time'].apply(lambda x: x[2])
                df['month'] = df['start_time'].apply(lambda x: x[1])
            
            # Since training data only had 2 timesteps, let's combine our features into a single value per timestep
            # We'll use the average of all features as our timestep value
            features = ['type', 'ambient_temperature', 'Re', 'Rct', 'hour', 'day', 'month']
            feature_data = df[features].values
            
            # Convert to mean value for each timestep
            timestep_values = np.mean(feature_data, axis=1)
            
            return timestep_values
            
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def predict(self, input_data: pd.DataFrame) -> float:
        try:
            processed_data = self.preprocess_data(input_data)
            
            # Ensure we have exactly 2 timesteps
            if len(processed_data) < 2:
                # Duplicate the single timestep to create two
                processed_data = np.tile(processed_data, 2)[:2]
            elif len(processed_data) > 2:
                processed_data = processed_data[-2:]  # Take last 2 timesteps
            
            # Reshape for model: (batch_size, timesteps, features)
            # Here each timestep is a single feature
            X = processed_data.reshape(1, 2, 1)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            
            return float(prediction[0][0])
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")