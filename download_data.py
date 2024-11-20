import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """Download stock data for the specified symbol"""
    try:
        print(f"Downloading {symbol} stock data for the past {years} years...")
        stock = yf.Ticker(symbol)
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Download data
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            raise ValueError(f"No data received for {symbol}")
            
        print(f"Downloaded {len(df)} data points")
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol} data: {str(e)}")
        raise

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # List of stock symbols to download
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    for symbol in symbols:
        try:
            # Download data
            df = download_stock_data(symbol)
            
            # Save to CSV
            output_path = f"data/{symbol}.csv"
            df.to_csv(output_path)
            print(f"Saved {symbol} data to {output_path}")
            
            # Print some statistics
            print(f"\nData Statistics for {symbol}:")
            print(f"Date Range: {df.index.min()} to {df.index.max()}")
            print(f"Number of trading days: {len(df)}")
            print(f"Average daily volume: {df['Volume'].mean():,.0f}")
            print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}\n")
            
        except Exception as e:
            print(f"Failed to process {symbol}: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()
