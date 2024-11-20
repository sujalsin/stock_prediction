import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from models import LSTMModel, RandomForestModel, XGBoostModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.close()

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return rmse, r2

def main():
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(window_size=60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = preprocessor.load_data('AAPL.csv')
    df = preprocessor.create_features(df)
    
    # Prepare data for models
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Initialize models
    lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    rf_model = RandomForestModel()
    xgb_model = XGBoostModel()
    
    # Train and evaluate LSTM
    print("\nTraining LSTM model...")
    lstm_model.train(X_train, y_train)
    lstm_pred = lstm_model.predict(X_test)
    lstm_metrics = evaluate_model(y_test[:, 3], lstm_pred[:, 3], 'LSTM')  # Using close price
    plot_predictions(y_test[:, 3], lstm_pred[:, 3], 'LSTM Predictions')
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest model...")
    rf_model.train(X_train.reshape(X_train.shape[0], -1), y_train[:, 3])  # Using close price
    rf_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    rf_metrics = evaluate_model(y_test[:, 3], rf_pred, 'Random Forest')
    plot_predictions(y_test[:, 3], rf_pred, 'Random Forest Predictions')
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost model...")
    xgb_model.train(X_train.reshape(X_train.shape[0], -1), y_train[:, 3])  # Using close price
    xgb_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    xgb_metrics = evaluate_model(y_test[:, 3], xgb_pred, 'XGBoost')
    plot_predictions(y_test[:, 3], xgb_pred, 'XGBoost Predictions')
    
    # Compare models
    models = ['LSTM', 'Random Forest', 'XGBoost']
    rmse_scores = [lstm_metrics[0], rf_metrics[0], xgb_metrics[0]]
    r2_scores = [lstm_metrics[1], rf_metrics[1], xgb_metrics[1]]
    
    print("\nModel Comparison:")
    for model, rmse, r2 in zip(models, rmse_scores, r2_scores):
        print(f"{model:13} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()
