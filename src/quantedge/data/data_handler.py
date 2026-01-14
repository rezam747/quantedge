"""
Data fetching and cleaning module.
Handles data retrieval from various sources and preprocessing.
"""

import pandas as pd
import yfinance as yf
from binance.client import Client
import warnings
from datetime import datetime
from pathlib import Path
import os
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
        

    def fetch_data(self, start_date, save_data = False):
        """
        Fetch historical data from Yahoo Finance and save to CSV.
        
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
        
        # Save fetched data to CSV
        if save_data:
            self._save_fetched_data()
        
        return self.data
    
    def _save_fetched_data(self):
        """
        Save fetched data to CSV file in fetched_data folder with timestamp and symbol.
        Creates the folder if it doesn't exist.
        """
        try:
            # Get the directory where this file is located
            data_dir = Path(__file__).parent
            fetched_data_dir = data_dir / "fetched_data"
            
            # Create folder if it doesn't exist
            fetched_data_dir.mkdir(exist_ok=True)
            
            # Create filename with timestamp, symbol, and source
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            symbol_clean = self.symbol.replace('-', '_').replace('/', '_')
            filename = f"{symbol_clean}_{self.data_source}_{timestamp}.csv"
            filepath = fetched_data_dir / filename
            
            # Save to CSV
            self.data.to_csv(filepath)
            print(f"✓ Fetched data saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not save fetched data to CSV: {e}")

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
        
        # Add to dataframe (only signal_labels, remove Entry_Price, Exit_Price, Exit_Date, Trade_Duration, Return)
        self.data['signal_labels'] = labels
        
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
