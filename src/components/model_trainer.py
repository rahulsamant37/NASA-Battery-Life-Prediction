import os
import time
import pandas as pd
import numpy as np
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Prepare data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1).values
        X_test = test_data.drop([self.config.target_column], axis=1).values
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]
        
        seq_length = 2
        num_features = X_train.shape[1]
        # Ensure the total features are divisible by seq_length
        if num_features % seq_length != 0:
            raise ValueError(f"Number of features ({num_features}) must be divisible by sequence length ({seq_length}).")
        
        reshaped_features = num_features // seq_length
        X_train = X_train.reshape(X_train.shape[0], seq_length, reshaped_features)
        X_test = X_test.reshape(X_test.shape[0], seq_length, reshaped_features)
        
        model_GRU = Sequential()
        model_GRU.add(Input(shape=(seq_length, reshaped_features)))  # Use Input layer for the first layer
        model_GRU.add(GRU(units=20, return_sequences=True))
        model_GRU.add(GRU(units=50, activation='relu'))
        model_GRU.add(Dense(units=1, activation=None))


        # Compile model
        model_GRU.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=['mae']
        )
        
        history = model_GRU.fit(
            X_train, 
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        joblib.dump(model_GRU, os.path.join(self.config.root_dir, self.config.model_name))