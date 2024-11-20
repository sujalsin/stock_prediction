import yfinance as yf
from data_preprocessing import DataPreprocessor
from models import LSTMModel, RandomForestModel, XGBoostModel, ModelEvaluator
import numpy as np
from datetime import datetime, timedelta

def run_complete_analysis():
    # Download recent stock data (using AAPL as an example)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    stock_data = yf.download('AAPL', start=start_date, end=end_date)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(window_size=60)
    
    # Create features
    df = preprocessor.create_features(stock_data)
    
    # Get feature names
    feature_names = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'MA50',
        'BB_middle', 'BB_upper', 'BB_lower',
        'MACD', 'Signal_Line',
        'RSI', 'Momentum', 'ROC',
        'OBV', 'Volume_MA20',
        'Volatility'
    ]
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Initialize models
    lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    rf_model = RandomForestModel()
    xgb_model = XGBoostModel()
    
    # Dictionary to store metrics
    models_metrics = {}
    
    # Train and evaluate LSTM
    print("Training LSTM model...")
    lstm_model.train(X_train, y_train)
    lstm_pred = lstm_model.predict(X_test)
    lstm_metrics = ModelEvaluator.evaluate_model(
        y_test[:, 3],  # Close price column
        lstm_pred[:, 3],  # Close price predictions
        model=lstm_model.model,
        feature_names=feature_names,
        X=X_test
    )
    models_metrics['LSTM'] = lstm_metrics
    
    # Reshape data for RF and XGBoost (they expect 2D input)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_train_close = y_train[:, 3]  # Close price
    y_test_close = y_test[:, 3]  # Close price
    
    # Train and evaluate Random Forest
    print("Training Random Forest model...")
    rf_model.train(X_train_2d, y_train_close)
    rf_pred = rf_model.predict(X_test_2d)
    rf_metrics = ModelEvaluator.evaluate_model(
        y_test_close,
        rf_pred,
        model=rf_model.model,
        feature_names=[f"{name}_{i}" for name in feature_names for i in range(60)],
        X=X_test_2d
    )
    models_metrics['RandomForest'] = rf_metrics
    
    # Train and evaluate XGBoost
    print("Training XGBoost model...")
    xgb_model.train(X_train_2d, y_train_close)
    xgb_pred = xgb_model.predict(X_test_2d)
    xgb_metrics = ModelEvaluator.evaluate_model(
        y_test_close,
        xgb_pred,
        model=xgb_model.model,
        feature_names=[f"{name}_{i}" for name in feature_names for i in range(60)],
        X=X_test_2d
    )
    models_metrics['XGBoost'] = xgb_metrics
    
    # Compare all models
    print("Generating model comparison visualization...")
    ModelEvaluator.compare_models(models_metrics)
    
    print("\nAnalysis complete! The following visualization files have been generated:")
    print("1. prediction_plot.png - Model predictions vs actual values")
    print("2. error_distribution.png - Distribution of prediction errors")
    print("3. feature_importance.png - Feature importance analysis")
    print("4. metrics_comparison.png - Comparison of model performance metrics")
    print("5. learning_curves.png - LSTM model learning curves")

if __name__ == "__main__":
    run_complete_analysis()
