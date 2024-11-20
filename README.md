# Stock Price Prediction ML Pipeline

This project implements a comprehensive machine learning pipeline for predicting stock price movements using multiple models including Random Forest, XGBoost, and LSTM neural networks.

## Project Structure

```
stock_prediction/
├── data_preprocessing.py  # Data preprocessing and feature engineering
├── models.py             # Model implementations
├── main.py              # Main script to run the pipeline
└── requirements.txt     # Project dependencies
```

## Features

- Data preprocessing and feature engineering
  - Technical indicators (Moving Averages, MACD, RSI)
  - Time series sequence preparation
  - Data normalization
- Multiple model implementations:
  - Random Forest
  - XGBoost
  - LSTM Neural Network
- Model evaluation and comparison
- Visualization of predictions

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data file (AAPL.csv) is in the correct location:
```
../data/AAPL.csv
```

3. Run the pipeline:
```bash
python main.py
```

## Model Details

### Random Forest
- Ensemble learning method using multiple decision trees
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 20

### XGBoost
- Gradient boosting implementation
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 7
  - learning_rate: 0.01

### LSTM Neural Network
- Deep learning model for sequence prediction
- Architecture:
  - 2 LSTM layers (50 units each)
  - Dropout layers (0.2)
  - Dense layers for final prediction

## Output

The pipeline will output:
- Performance metrics for each model (MSE, RMSE, MAE, R²)
- Visualization of predictions vs actual values
- Comparison of model improvements over baseline
