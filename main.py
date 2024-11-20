import os
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from model_training import StockPredictor
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        print("Starting Stock Price Prediction Pipeline...")
        
        # Initialize preprocessor and model
        preprocessor = DataPreprocessor(window_size=60)
        model = StockPredictor()
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        data_path = "data/AAPL.csv"  # Make sure this path is correct
        df = preprocessor.load_data(data_path)
        print(f"Loaded {len(df)} data points")
        
        # Create features
        print("Creating technical indicators...")
        df = preprocessor.create_features(df)
        print(f"Final dataset shape: {df.shape}")
        
        # Prepare data for modeling
        print("\nPreparing sequences for modeling...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Train models and get metrics
        feature_names = df.columns.tolist()
        metrics = model.train_models(X_train, y_train, feature_names)
        
        # Print results
        print("\nModel Performance Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()} Model:")
            for metric_name, value in model_metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Make predictions on test set
        print("\nMaking predictions on test set...")
        for model_name in ['lstm', 'random_forest', 'xgboost']:
            y_pred = model.predict(X_test, model_name)
            print(f"\n{model_name.upper()} Test Metrics:")
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        print("\nPipeline completed successfully!")
        print("Results and plots have been saved in the 'results' directory.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
