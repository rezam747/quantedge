"""
Data fetching and cleaning module.
Handles data retrieval from various sources and preprocessing.
"""

import pandas as pd
import yfinance as yf
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')


class DataHandler:
    """
    Handles data fetching and cleaning operations.
    """
    
    def __init__(self, symbol="BTC-USD", stop_loss_pct=5, target_pct=15, data_source="yahoo"):
        """
        Initialize DataHandler.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC-USD')
            stop_loss_pct (float): Stop loss percentage
            target_pct (float): Target profit percentage
            data_source (str): Data source ('yahoo' or 'binance')
        """
        self.symbol = symbol
        self.stop_loss_pct = stop_loss_pct / 100
        self.target_pct = target_pct / 100
        self.data_source = data_source
        self.data = None
        
        # Initialize Binance client if needed
        if data_source == "binance":
            self.binance_client = Client(None, None)  # Use without API keys for public data
            self.symbol = self.symbol.replace('-', '')  # Convert BTC-USD to BTCUSD format

    def fetch_data(self, start_date):
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            start_date (str): Start date for historical data (required)
            
        Returns:
            pd.DataFrame: Historical price data
        """
        print(f"Fetching {self.symbol} data from {start_date}...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date)
        self.data.index = pd.to_datetime(self.data.index)
        print(f"✓ Fetched {len(self.data)} data points")
        return self.data

    def clean_data(self):
        """
        Clean the dataset by handling missing values.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("Cleaning data...")
        # Forward fill missing values
        self.data = self.data.fillna(method='ffill')
        # Backward fill any remaining missing values
        self.data = self.data.fillna(method='bfill')
        print("✓ Data cleaned")
        return self.data

    def calculate_labels(self):
        """
        Calculate trade outcome labels based on stop-loss and target.
        
        Returns:
            pd.DataFrame: Data with labels
        """
        print("Calculating signal labels...")
        labels = []
        entry_prices = []
        exit_prices = []
        exit_dates = []
        trade_durations = []
        
        for i in range(len(self.data) - 1):
            entry_price = self.data['Close'].iloc[i]
            entry_date = self.data.index[i]
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            target = entry_price * (1 + self.target_pct)
            
            future_data = self.data.iloc[i+1:]
            hit_stop = future_data['Low'] <= stop_loss
            hit_target = future_data['High'] >= target
            
            if hit_stop.any() and hit_target.any():
                stop_idx = hit_stop.idxmax()
                target_idx = hit_target.idxmax()
                
                if target_idx < stop_idx:
                    labels.append(1)
                    exit_date = target_idx
                    exit_price = target
                else:
                    labels.append(0)
                    exit_date = stop_idx
                    exit_price = stop_loss
            elif hit_stop.any():
                labels.append(0)
                exit_date = hit_stop.idxmax()
                exit_price = stop_loss
            elif hit_target.any():
                labels.append(1)
                exit_date = hit_target.idxmax()
                exit_price = target
            else:
                labels.append(0)
                exit_date = self.data.index[-1]
                exit_price = self.data['Close'].iloc[-1]
            
            entry_prices.append(entry_price)
            exit_prices.append(exit_price)
            exit_dates.append(exit_date)
            trade_durations.append((pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days)
        
        # Add last row
        labels.append(0)
        entry_prices.append(self.data['Close'].iloc[-1])
        exit_prices.append(self.data['Close'].iloc[-1])
        exit_dates.append(self.data.index[-1])
        trade_durations.append(0)
        
        # Add to dataframe
        self.data['signal_labels'] = labels
        self.data['Entry_Price'] = entry_prices
        self.data['Exit_Price'] = exit_prices
        self.data['Exit_Date'] = exit_dates
        self.data['Trade_Duration'] = trade_durations
        self.data['Return'] = (self.data['Exit_Price'] - self.data['Entry_Price']) / self.data['Entry_Price'] * 100
        
        num_positive = sum(labels)
        print(f"✓ Signal labels calculated: {num_positive} positive signals ({num_positive/len(labels)*100:.1f}%)")
        
        return self.data
    
    def get_data(self):
        """
        Get the processed data.
        
        Returns:
            pd.DataFrame: Processed data
        """
        return self.data
    
    def export_data_html(self, filename="crypto_data.html"):
        """
        Export the full dataset to an HTML file.
        
        Args:
            filename (str): Output filename
        """
        if self.data is not None:
            self.data.to_html(filename)
            print(f"Full dataset exported to {filename}")
        else:
            print("No data to export.")
