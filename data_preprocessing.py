import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class DataPreprocessor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess the stock data."""
        df = pd.read_csv(filepath)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df
        
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], n_sigmas: float = 3) -> pd.DataFrame:
        """Handle outliers using the z-score method"""
        df_clean = df.copy()
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            df_clean.loc[z_scores > n_sigmas, column] = mean
        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as features with improved calculations."""
        df = df.copy()
        
        # Handle outliers in price and volume data
        price_columns = ['Open', 'High', 'Low', 'Close']
        df = self.handle_outliers(df, price_columns + ['Volume'])
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Std'] = df['Close'].rolling(window=20).std() / df['Close']
        
        # Moving averages with different windows
        for window in [5, 10, 20, 50]:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA{window}_Slope'] = df[f'MA{window}'].pct_change() * 100
        
        # Exponential moving averages
        for window in [5, 10, 20, 50]:
            df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Bollinger Bands
        for window in [20, 50]:
            middle = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            df[f'BB_upper_{window}'] = middle + 2 * std
            df[f'BB_lower_{window}'] = middle - 2 * std
            df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / middle
        
        # MACD with different parameters
        for (fast, slow) in [(12, 26), (8, 21)]:
            exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = signal
            df[f'MACD_Hist_{fast}_{slow}'] = macd - signal
        
        # RSI with multiple timeframes
        for window in [14, 28]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA20_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # On-Balance Volume (OBV) and its EMA
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'Momentum_{window}'] = df['Close'].diff(window)
            df[f'ROC_{window}'] = df['Close'].pct_change(window) * 100
        
        # Advanced features
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Remove NaN values
        df = df.dropna()
        
        return df
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series prediction."""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:(i + self.window_size)])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        # Scale the features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'Range', 'Price_Std',
            'MA5', 'MA5_Slope', 'MA10', 'MA10_Slope', 'MA20', 'MA20_Slope', 'MA50', 'MA50_Slope',
            'EMA5', 'EMA10', 'EMA20', 'EMA50',
            'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_upper_50', 'BB_lower_50', 'BB_width_50',
            'MACD_12_26', 'MACD_Signal_12_26', 'MACD_Hist_12_26', 'MACD_8_21', 'MACD_Signal_8_21', 'MACD_Hist_8_21',
            'RSI_14', 'RSI_28',
            'Volume_MA20', 'Volume_MA20_Ratio',
            'OBV', 'OBV_EMA',
            'Volatility',
            'Momentum_5', 'ROC_5', 'Momentum_10', 'ROC_10', 'Momentum_20', 'ROC_20',
            'ATR', 'Stoch_K', 'Stoch_D'
        ]
        data = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = self.prepare_sequences(data)
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
