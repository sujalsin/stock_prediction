import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from visualization import ModelVisualizer

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.models = {}
        self.feature_names = None
        self.visualizer = ModelVisualizer()
        
    def prepare_data(self, data: pd.DataFrame, target_col: str, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, data.columns.get_loc(target_col)])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model with improved architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')  # Huber loss is more robust to outliers
        return model
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """Train multiple models with improved parameters"""
        self.feature_names = feature_names
        metrics_dict = {}
        
        # Train LSTM with early stopping and reduced learning rate on plateau
        print("\nTraining LSTM model...")
        lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
        
        # Add callbacks for better training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        history = lstm_model.fit(
            X, y,
            epochs=100,  # Increased epochs
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        self.models['lstm'] = lstm_model
        
        # Plot learning curves
        self.visualizer.plot_learning_curves(history.history)
        
        # Prepare data for traditional models
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Train Random Forest with improved parameters
        print("\nTraining Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=200,  # Increased number of trees
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_reshaped, y)
        self.models['random_forest'] = rf_model
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='rmse'
        )
        
        xgb_model.fit(
            X_reshaped,
            y,
            verbose=True
        )
        self.models['xgboost'] = xgb_model
        
        # Calculate and store metrics
        for name, model in self.models.items():
            if name == 'lstm':
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X_reshaped)
            
            metrics_dict[name] = {
                'mse': np.mean((y - y_pred) ** 2),
                'mae': np.mean(np.abs(y - y_pred)),
                'rmse': np.sqrt(np.mean((y - y_pred) ** 2))
            }
            
            # Plot predictions
            self.visualizer.plot_predictions(y, y_pred, f'{name.upper()} Model Predictions')
            self.visualizer.plot_error_distribution(y, y_pred)
            
            # Analyze feature importance (for RF and XGB)
            if name in ['random_forest', 'xgboost']:
                self.visualizer.analyze_feature_importance(model, feature_names, X)
        
        # Plot metrics comparison
        self.visualizer.plot_metrics_comparison(metrics_dict)
        
        return metrics_dict
    
    def predict(self, X: np.ndarray, model_name: str = 'lstm') -> np.ndarray:
        """Make predictions using the specified model"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        if model_name == 'lstm':
            return model.predict(X)
        else:
            X_reshaped = X.reshape(X.shape[0], -1)
            return model.predict(X_reshaped)
