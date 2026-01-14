"""
Feature engineering module.
Creates technical indicators and features for model training.
"""

import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split


class FeatureEngineer:
    """
    Handles feature engineering and data splitting.
    """
    
    def __init__(self, data):
        """
        Initialize FeatureEngineer.
        
        Args:
            data (pd.DataFrame): Input data
        """
        self.data = data.copy()
    
    def create_features(self):
        """
        Create technical analysis features with expanded set.
        
        Returns:
            pd.DataFrame: Data with engineered features
        """
        print("Creating technical indicators and features...")
        # Price features
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        self.data['EMA5'] = self.data['Close'].ewm(span=5, adjust=False).mean()
        self.data['EMA20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['EMA50'] = self.data['Close'].ewm(span=50, adjust=False).mean()
        self.data['EMA200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        
        # Price changes
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Price_Change_5'] = self.data['Close'].pct_change(periods=5)
        self.data['Price_Change_20'] = self.data['Close'].pct_change(periods=20)
        
        # Rate of Change
        self.data['ROC_5'] = self.data['Close'].pct_change(periods=5) * 100
        self.data['ROC_20'] = self.data['Close'].pct_change(periods=20) * 100
        
        # Momentum
        self.data['Momentum_10'] = self.data['Close'] - self.data['Close'].shift(10)
        
        # Historical Volatility
        self.data['Hist_Vol_10'] = self.data['Close'].pct_change().rolling(window=10).std()
        self.data['Hist_Vol_30'] = self.data['Close'].pct_change().rolling(window=30).std()
        
        # Sharpe Ratio (rolling, risk-free rate assumed 0)
        self.data['Sharpe_10'] = self.data['Close'].pct_change().rolling(window=10).mean() / self.data['Close'].pct_change().rolling(window=10).std()
        self.data['Sharpe_30'] = self.data['Close'].pct_change().rolling(window=30).mean() / self.data['Close'].pct_change().rolling(window=30).std()
        
        # Day of week (Monday=1, Sunday=7)
        self.data['DayOfWeek'] = self.data.index.dayofweek + 1
        # Month
        self.data['Month'] = self.data.index.month
        
        # Volume features
        self.data['Volume_MA5'] = self.data['Volume'].rolling(window=5).mean()
        self.data['Volume_MA20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA20']
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        # On-Balance Volume (OBV)
        self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
        # Volume Spike
        self.data['Volume_Spike'] = self.data['Volume'] / self.data['Volume'].rolling(window=20).max()
        
        # RSI with different periods
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=14)
        self.data['RSI_5'] = ta.momentum.rsi(self.data['Close'], window=5)
        self.data['RSI_20'] = ta.momentum.rsi(self.data['Close'], window=20)
        # Lagged RSI
        self.data['RSI_Lag1'] = self.data['RSI'].shift(1)
        
        # Williams %R
        self.data['Williams_%R'] = ta.momentum.williams_r(self.data['High'], self.data['Low'], self.data['Close'], lbp=14)
        # CCI
        self.data['CCI'] = ta.trend.cci(self.data['High'], self.data['Low'], self.data['Close'], window=20)
        # ADX
        self.data['ADX'] = ta.trend.adx(self.data['High'], self.data['Low'], self.data['Close'], window=14)
        
        # MACD
        self.data['MACD'] = ta.trend.macd_diff(self.data['Close'])
        # Lagged MACD
        self.data['MACD_Lag1'] = self.data['MACD'].shift(1)
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(self.data['Close'])
        self.data['BB_Upper'] = bb_indicator.bollinger_hband()
        self.data['BB_Middle'] = bb_indicator.bollinger_mavg()
        self.data['BB_Lower'] = bb_indicator.bollinger_lband()
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        # Stochastic Oscillator
        self.data['Stoch_K'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Stoch_D'] = ta.momentum.stoch_signal(self.data['High'], self.data['Low'], self.data['Close'])
        # Lagged Stoch_K
        self.data['Stoch_K_Lag1'] = self.data['Stoch_K'].shift(1)
        
        # ATR
        self.data['ATR'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['ATR_Ratio'] = self.data['ATR'] / self.data['Close']
        
        # Price position relative to moving averages
        self.data['Price_MA5_Ratio'] = self.data['Close'] / self.data['MA5']
        self.data['Price_MA20_Ratio'] = self.data['Close'] / self.data['MA20']
        self.data['Price_MA50_Ratio'] = self.data['Close'] / self.data['MA50']
        self.data['Price_MA200_Ratio'] = self.data['Close'] / self.data['MA200']
        
        # Moving average crossovers
        self.data['MA5_MA20_Cross'] = (self.data['MA5'] > self.data['MA20']).astype(int)
        self.data['MA20_MA50_Cross'] = (self.data['MA20'] > self.data['MA50']).astype(int)
        self.data['MA50_MA200_Cross'] = (self.data['MA50'] > self.data['MA200']).astype(int)
        
        # Volatility
        self.data['Price_Volatility'] = self.data['Close'].rolling(window=20).std() / self.data['Close']
        # Rolling max/min and distance
        self.data['Rolling_High_20'] = self.data['High'].rolling(window=20).max()
        self.data['Rolling_Low_20'] = self.data['Low'].rolling(window=20).min()
        self.data['Dist_High_20'] = self.data['Close'] - self.data['Rolling_High_20']
        self.data['Dist_Low_20'] = self.data['Close'] - self.data['Rolling_Low_20']
        
        # Candlestick patterns
        self.data['Bull_Engulfing'] = ((self.data['Close'] > self.data['Open']) & (self.data['Open'].shift(1) > self.data['Close'].shift(1)) & (self.data['Open'] < self.data['Close'].shift(1)) & (self.data['Close'] > self.data['Open'].shift(1))).astype(int)
        self.data['Bear_Engulfing'] = ((self.data['Close'] < self.data['Open']) & (self.data['Open'].shift(1) < self.data['Close'].shift(1)) & (self.data['Open'] > self.data['Close'].shift(1)) & (self.data['Close'] < self.data['Open'].shift(1))).astype(int)
        self.data['Hammer'] = (((self.data['High'] - self.data['Low']) > 3 * (self.data['Open'] - self.data['Close'])) & ((self.data['Close'] - self.data['Low']) / (.001 + self.data['High'] - self.data['Low']) > 0.6) & ((self.data['Open'] - self.data['Low']) / (.001 + self.data['High'] - self.data['Low']) > 0.6)).astype(int)
        self.data['Doji'] = (abs(self.data['Close'] - self.data['Open']) <= (self.data['High'] - self.data['Low']) * 0.1).astype(int)
        
        # Add lagged features for key indicators
        for col in ['RSI', 'MACD', 'Stoch_K', 'ATR', 'BB_Width', 'Price_Volatility']:
            self.data[f'{col}_Lag1'] = self.data[col].shift(1)
        
        # Drop rows with NaN values from rolling calculations
        self.data = self.data.dropna()
        print(f"✓ Created {len(self.data.columns)} features")
        return self.data
    
    def split_data(self, feature_columns, train_ratio=0.7):
        """
        Split data into train and test sets.
        
        Args:
            feature_columns (list): List of feature column names
            train_ratio (float): Ratio of training data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Splitting data into train/test sets...")
        # Remove duplicate features
        features = list(dict.fromkeys(feature_columns))
        X = self.data[features]
        y = self.data['signal_labels']
        
        # Split: train vs test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)
        
        print(f"✓ Data split complete:")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        
        return (X_train, X_test, y_train, y_test)
    
    def get_data(self):
        """
        Get the data with features.
        
        Returns:
            pd.DataFrame: Data with features
        """
        return self.data
