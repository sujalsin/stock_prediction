import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any

class ModelVisualizer:
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Model Predictions vs Actual Values") -> None:
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prediction_plot.png')
        plt.close()

    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot the distribution of prediction errors
        """
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('error_distribution.png')
        plt.close()

    @staticmethod
    def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> None:
        """
        Plot comparison of different metrics across models
        """
        metrics = list(next(iter(metrics_dict.values())).keys())
        models = list(metrics_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison')
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            
            values = [metrics_dict[model][metric] for model in models]
            axes[row, col].bar(models, values)
            axes[row, col].set_title(f'{metric.upper()} Comparison')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig('metrics_comparison.png')
        plt.close()

    @staticmethod
    def analyze_feature_importance(model, feature_names, X):
        """Analyze and visualize feature importance using a simplified approach"""
        try:
            # Create a simpler feature importance visualization using correlation analysis
            print("\nAnalyzing feature importance using correlation analysis...")
            
            # Reshape X if needed (assuming X is your input data)
            if len(X.shape) == 3:
                X_reshaped = X.reshape(-1, X.shape[-1])
            else:
                X_reshaped = X
                
            # Calculate correlation with target (using absolute correlation)
            correlations = np.abs(np.corrcoef(X_reshaped.T))
            
            # Average correlation for each feature
            avg_correlations = np.mean(correlations, axis=1)
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            feature_importance = pd.Series(avg_correlations, index=feature_names)
            feature_importance.sort_values(ascending=True).plot(kind='barh')
            plt.title('Feature Importance based on Average Correlation')
            plt.xlabel('Average Absolute Correlation')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            print("Feature importance analysis completed. Results saved to 'feature_importance.png'")
            
        except Exception as e:
            print(f"Warning: Could not complete feature importance analysis. Error: {str(e)}")

    @staticmethod
    def plot_learning_curves(history: Dict[str, List[float]]) -> None:
        """
        Plot training history for deep learning models
        """
        plt.figure(figsize=(12, 6))
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.close()
