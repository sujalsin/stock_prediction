import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, List
from visualization import ModelVisualizer

class BaseModel:
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LSTMModel(BaseModel):
    def __init__(self, input_shape: Tuple[int, int]):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(input_shape[1])  # Output dimension matches input features
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.history = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        
        # Plot learning curves after training
        ModelVisualizer.plot_learning_curves(self.history.history)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.01,
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model: Any = None, 
                      feature_names: List[str] = None, X: np.ndarray = None) -> Dict[str, float]:
        """Calculate various performance metrics and generate visualizations."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Generate visualization plots
        ModelVisualizer.plot_predictions(y_true, y_pred)
        ModelVisualizer.plot_error_distribution(y_true, y_pred)
        
        # Analyze feature importance if model and feature names are provided
        if model is not None and feature_names is not None and X is not None:
            ModelVisualizer.analyze_feature_importance(model, feature_names, X)
        
        return metrics
    
    @staticmethod
    def calculate_improvement(baseline_metrics: Dict[str, float], 
                            model_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate percentage improvement over baseline."""
        improvements = {}
        for metric in baseline_metrics:
            if baseline_metrics[metric] != 0:
                improvement = ((baseline_metrics[metric] - model_metrics[metric]) 
                             / baseline_metrics[metric]) * 100
                improvements[metric] = improvement
        return improvements
    
    @staticmethod
    def compare_models(models_metrics: Dict[str, Dict[str, float]]) -> None:
        """Compare and visualize metrics across different models."""
        ModelVisualizer.plot_metrics_comparison(models_metrics)
