# Stock Price Prediction ML Pipeline

This project implements a comprehensive machine learning pipeline for predicting stock price movements using multiple models including Random Forest, XGBoost, and LSTM neural networks. The system analyzes historical stock data to forecast future price movements.

## Features

- **Data Processing & Feature Engineering**
  - Technical indicators calculation (Moving Averages, MACD, RSI)
  - Automated data cleaning and preprocessing
  - Time series sequence preparation
  - Data normalization and scaling
  
- **Multiple Model Implementations**
  - Random Forest for robust ensemble predictions
  - XGBoost for gradient boosting-based forecasting
  - LSTM Neural Network for sequence learning
  
- **Analysis & Visualization**
  - Model performance evaluation metrics
  - Interactive prediction visualizations
  - Model comparison analytics
  - Feature importance analysis

## Project Structure

```
stock_prediction/
├── data/                # Data directory for stock price datasets
├── models/             # Trained model files
├── data_preprocessing.py  # Data preprocessing and feature engineering
├── models.py             # Model implementations
├── main.py              # Main script to run the pipeline
├── utils.py             # Utility functions
└── requirements.txt     # Project dependencies
```

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

The system expects stock data in CSV format with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

Place your data file in the `data/` directory. Default expected filename is `AAPL.csv` for Apple stock data.

## Usage

1. Prepare your data:
   - Ensure your stock data CSV is in the correct format
   - Place it in the `data/` directory

2. Run the pipeline:
```bash
python main.py
```

3. Optional arguments:
```bash
python main.py --stock_symbol AAPL --start_date 2020-01-01 --end_date 2023-12-31
```

## Model Details

### Random Forest
- **Purpose**: Ensemble learning for robust predictions
- **Key Features**:
  - Multiple decision trees for reduced overfitting
  - Feature importance analysis
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 2
  - min_samples_leaf: 1

### XGBoost
- **Purpose**: Gradient boosting for high accuracy
- **Key Features**:
  - Gradient boosting implementation
  - Handles missing values
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 7
  - learning_rate: 0.01
  - subsample: 0.8

### LSTM Neural Network
- **Purpose**: Deep learning for sequence prediction
- **Architecture**:
  - Input Layer
  - 2 LSTM layers (50 units each)
  - Dropout layers (0.2)
  - Dense layers for prediction
- **Training Parameters**:
  - Batch size: 32
  - Epochs: 100
  - Optimizer: Adam
  - Loss: Mean Squared Error

## Output and Results

The pipeline generates:
1. **Performance Metrics**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score

2. **Visualizations**:
   - Actual vs Predicted price plots
   - Model comparison charts
   - Feature importance plots
   - Training/validation loss curves

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
