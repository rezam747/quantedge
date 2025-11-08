import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')

class CryptoAnalysis:
    def __init__(self, symbol="BTC-USD", stop_loss_pct=5, target_pct=15, data_source="yahoo"):
        self.symbol = symbol
        self.stop_loss_pct = stop_loss_pct / 100
        self.target_pct = target_pct / 100
        self.data_source = data_source
        self.data = None
        self.model = None
        
        # Initialize Binance client if needed
        if data_source == "binance":
            self.binance_client = Client(None, None)  # Use without API keys for public data
            self.symbol = self.symbol.replace('-', '')  # Convert BTC-USD to BTCUSD format

    def fetch_data(self, start_date="2020-01-01"):
        """Fetch historical data from Yahoo Finance"""
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date)
        self.data.index = pd.to_datetime(self.data.index)
        return self.data

    def clean_data(self):
        """Clean the dataset by handling missing values"""
        # Forward fill missing values
        self.data = self.data.fillna(method='ffill')
        # Backward fill any remaining missing values
        self.data = self.data.fillna(method='bfill')
        return self.data

    def calculate_labels(self):
        """Calculate trade outcome labels based on stop-loss and target"""
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
        self.data['Label'] = labels
        self.data['Entry_Price'] = entry_prices
        self.data['Exit_Price'] = exit_prices
        self.data['Exit_Date'] = exit_dates
        self.data['Trade_Duration'] = trade_durations
        self.data['Return'] = (self.data['Exit_Price'] - self.data['Entry_Price']) / self.data['Entry_Price'] * 100
        
        return self.data

    def create_features(self):
        """Create technical analysis features with expanded set"""
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
        return self.data

    def split_data(self):
        """Split data into train, validation, and test sets"""
        features = [
            'MA5', 'MA20', 'MA50', 'MA200',
            'EMA5', 'EMA20', 'EMA50', 'EMA200',
            'Price_Change', 'Price_Change_5', 'Price_Change_20',
            'ROC_5', 'ROC_20', 'Momentum_10',
            'Hist_Vol_10', 'Hist_Vol_30', 'Sharpe_10', 'Sharpe_30',
            'DayOfWeek', 'Month',
            'Volume_Ratio', 'Volume_Change', 'OBV', 'Volume_Spike',
            'RSI', 'RSI_5', 'RSI_20', 'RSI_Lag1',
            'Williams_%R', 'CCI', 'ADX',
            'MACD', 'MACD_Lag1',
            'BB_Width', 'BB_Position',
            'Stoch_K', 'Stoch_D', 'Stoch_K_Lag1',
            'ATR', 'ATR_Ratio',
            'Price_MA5_Ratio', 'Price_MA20_Ratio', 'Price_MA50_Ratio', 'Price_MA200_Ratio',
            'MA5_MA20_Cross', 'MA20_MA50_Cross', 'MA50_MA200_Cross',
            'Price_Volatility', 'Rolling_High_20', 'Rolling_Low_20', 'Dist_High_20', 'Dist_Low_20',
            'Bull_Engulfing', 'Bear_Engulfing', 'Hammer', 'Doji',
            'ATR_Lag1', 'BB_Width_Lag1', 'Price_Volatility_Lag1'
        ]
        # Remove duplicate features
        features = list(dict.fromkeys(features))
        X = self.data[features]
        y = self.data['Label']
        # First split: 70% train, 30% remaining
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=False)
        # Second split: Split remaining 30% into 15% validation and 15% test
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        return (X_train, X_val, X_test, y_train, y_val, y_test)

    def train_model(self, X_train, y_train):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions)
        }

    def plot_results(self, X_train, X_val, X_test):
        """Plot interactive results using plotly"""
        # Create predictions for each period separately
        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)
        test_predictions = self.model.predict(X_test)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Price', 'RSI', 'Volume'),
                           row_heights=[0.5, 0.25, 0.25])

        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)

        # Add buy signals for training data (green)
        train_buy_signals = X_train[train_predictions == 1].index
        if len(train_buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=train_buy_signals,
                y=self.data.loc[train_buy_signals, 'Low'] * 0.99,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                ),
                name='Train Buy Signal'
            ), row=1, col=1)

        # Add buy signals for validation data (yellow)
        val_buy_signals = X_val[val_predictions == 1].index
        if len(val_buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=val_buy_signals,
                y=self.data.loc[val_buy_signals, 'Low'] * 0.99,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='gold',
                ),
                name='Validation Buy Signal'
            ), row=1, col=1)

        # Add buy signals for test data (blue)
        test_buy_signals = X_test[test_predictions == 1].index
        if len(test_buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=test_buy_signals,
                y=self.data.loc[test_buy_signals, 'Low'] * 0.99,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='blue',
                ),
                name='Test Buy Signal'
            ), row=1, col=1)

        # Add RSI
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)

        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Add Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for i, row in self.data.iterrows()]
        
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Price and Trading Signals',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save the interactive HTML plot
        fig.write_html('trading_signals.html')

    def plot_test_predictions(self, X_test, y_test):
        """Plot test data predictions and their outcomes"""
        # Get predictions for test data
        test_predictions = self.model.predict(X_test)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Price with Prediction Outcomes', 'RSI', 'Volume'),
                           row_heights=[0.5, 0.25, 0.25])

        # Add candlestick
        test_data = self.data.loc[X_test.index]
        fig.add_trace(go.Candlestick(
            x=test_data.index,
            open=test_data['Open'],
            high=test_data['High'],
            low=test_data['Low'],
            close=test_data['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Add Bollinger Bands for test period
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)

        # Add successful predictions (hit target)
        successful_signals = X_test[(test_predictions == 1) & (y_test == 1)].index
        if len(successful_signals) > 0:
            fig.add_trace(go.Scatter(
                x=successful_signals,
                y=test_data.loc[successful_signals, 'Low'] * 0.99,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                ),
                name='Hit Target'
            ), row=1, col=1)

            # Add target prices for successful trades
            for idx in successful_signals:
                entry_price = test_data.loc[idx, 'Entry_Price']
                target_price = entry_price * (1 + self.target_pct)
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                
                # Add target line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    y0=entry_price,
                    x1=test_data.loc[idx, 'Exit_Date'],
                    y1=target_price,
                    line=dict(color="green", width=1, dash="dot"),
                    row=1, col=1
                )
                
                # Add stop-loss line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    y0=entry_price,
                    x1=test_data.loc[idx, 'Exit_Date'],
                    y1=stop_loss,
                    line=dict(color="red", width=1, dash="dot"),
                    row=1, col=1
                )

        # Add failed predictions (hit stop-loss)
        failed_signals = X_test[(test_predictions == 1) & (y_test == 0)].index
        if len(failed_signals) > 0:
            fig.add_trace(go.Scatter(
                x=failed_signals,
                y=test_data.loc[failed_signals, 'Low'] * 0.99,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='red',
                ),
                name='Hit Stop-Loss'
            ), row=1, col=1)

            # Add target and stop-loss lines for failed trades
            for idx in failed_signals:
                entry_price = test_data.loc[idx, 'Entry_Price']
                target_price = entry_price * (1 + self.target_pct)
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                
                # Add target line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    y0=entry_price,
                    x1=test_data.loc[idx, 'Exit_Date'],
                    y1=target_price,
                    line=dict(color="green", width=1, dash="dot"),
                    row=1, col=1
                )
                
                # Add stop-loss line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    y0=entry_price,
                    x1=test_data.loc[idx, 'Exit_Date'],
                    y1=stop_loss,
                    line=dict(color="red", width=1, dash="dot"),
                    row=1, col=1
                )

        # Add RSI
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)

        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Add Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for i, row in test_data.iterrows()]
        
        fig.add_trace(go.Bar(
            x=test_data.index,
            y=test_data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=3, col=1)

        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Test Period Predictions and Outcomes',
            xaxis_rangeslider_visible=False,
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save the interactive HTML plot
        fig.write_html('test_predictions.html')

        # Print statistics about test predictions
        successful_count = len(successful_signals)
        failed_count = len(failed_signals)
        total_signals = successful_count + failed_count
        
        print("\nTest Period Prediction Statistics:")
        print(f"Total Buy Signals: {total_signals}")
        if total_signals > 0:
            print(f"Successful (Hit Target): {successful_count} ({successful_count/total_signals*100:.1f}% of signals)")
            print(f"Failed (Hit Stop-Loss): {failed_count} ({failed_count/total_signals*100:.1f}% of signals)")
            print(f"Win Rate: {successful_count/total_signals*100:.1f}%")
            
            # Calculate returns
            successful_returns = self.target_pct * successful_count
            failed_returns = -self.stop_loss_pct * failed_count
            total_return = successful_returns + failed_returns
            print(f"\nReturns Analysis:")
            print(f"Total Return: {total_return*100:.1f}%")
            print(f"Average Return per Trade: {(total_return/total_signals)*100:.1f}%")
            
            # Calculate average trade duration
            avg_duration = test_data.loc[test_predictions == 1, 'Trade_Duration'].mean()
            print(f"Average Trade Duration: {avg_duration:.1f} days")

    def export_data_html(self, filename="crypto_data.html"):
        """Export the full dataset to an HTML file."""
        if self.data is not None:
            self.data.to_html(filename)
            print(f"Full dataset exported to {filename}")
        else:
            print("No data to export.")

def main():
    import os
    from datetime import datetime
    
    # Create a timestamped subfolder in reports/
    base_reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    reports_dir = os.path.join(base_reports_dir, timestamp)
    os.makedirs(reports_dir, exist_ok=True)

    print("\n===== Training Random Forest Model =====")
    
    # Initialize and train model
    analysis = CryptoAnalysis(symbol="BTC-USD", stop_loss_pct=5, target_pct=15, data_source="yahoo")
    
    # Fetch and prepare data
    print("Fetching data...")
    analysis.fetch_data()
    print("Cleaning data...")
    analysis.clean_data()
    print("Calculating labels...")
    analysis.calculate_labels()
    print("Creating features...")
    analysis.create_features()
    
    # Export full dataset as HTML
    html_data_filename = os.path.join(reports_dir, "data_table.html")
    analysis.export_data_html(html_data_filename)
    print(f"Exported data table to {html_data_filename}")
    
    # Split data and train model
    print("Splitting data and training model...")
    X_train, X_val, X_test, y_train, y_val, y_test = analysis.split_data()
    analysis.train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    test_predictions = analysis.model.predict(X_test)
    train_predictions = analysis.model.predict(X_train)
    
    # Calculate comprehensive metrics
    metrics = analysis.evaluate_model(X_test, y_test)
    train_metrics = analysis.evaluate_model(X_train, y_train)
    
    # Get confusion matrix and classification report
    cm = confusion_matrix(y_test, test_predictions)
    class_report = classification_report(y_test, test_predictions, output_dict=True)
    
    # Count signals
    num_train_signals = sum(train_predictions)
    num_test_signals = sum(test_predictions)
    num_train_target_1 = sum(y_train)
    num_test_target_1 = sum(y_test)
    
    # Create model information HTML
    print("Creating model information page...")
    model_info_html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Random Forest Model Information</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f8f8; padding: 20px; }}
            h1, h2 {{ color: #222; }}
            .info-section {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 18px; font-weight: bold; color: #4CAF50; }}
        </style>
    </head>
    <body>
        <h1>Random Forest Model Information</h1>
        
        <div class="info-section">
            <h2>Model Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Model Type</td><td>Random Forest Classifier</td></tr>
                <tr><td>Number of Estimators</td><td>200</td></tr>
                <tr><td>Max Depth</td><td>10</td></tr>
                <tr><td>Min Samples Split</td><td>10</td></tr>
                <tr><td>Min Samples Leaf</td><td>5</td></tr>
                <tr><td>Class Weight</td><td>Balanced</td></tr>
                <tr><td>Random State</td><td>42</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Training Data Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Training Samples</td><td>{len(X_train)}</td></tr>
                <tr><td>Training Samples with Target=1</td><td>{num_train_target_1} ({num_train_target_1/len(y_train)*100:.2f}%)</td></tr>
                <tr><td>Training Accuracy</td><td class="metric">{train_metrics['accuracy']:.4f}</td></tr>
                <tr><td>Training Precision</td><td class="metric">{train_metrics['precision']:.4f}</td></tr>
                <tr><td>Training Recall</td><td class="metric">{train_metrics['recall']:.4f}</td></tr>
                <tr><td>Buy Signals Generated (Training)</td><td>{num_train_signals}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Test Data Performance</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Test Samples</td><td>{len(X_test)}</td></tr>
                <tr><td>Test Samples with Target=1</td><td>{num_test_target_1} ({num_test_target_1/len(y_test)*100:.2f}%)</td></tr>
                <tr><td>Test Accuracy</td><td class="metric">{metrics['accuracy']:.4f}</td></tr>
                <tr><td>Test Precision</td><td class="metric">{metrics['precision']:.4f}</td></tr>
                <tr><td>Test Recall</td><td class="metric">{metrics['recall']:.4f}</td></tr>
                <tr><td>Buy Signals Generated (Test)</td><td>{num_test_signals}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Confusion Matrix (Test Data)</h2>
            <table>
                <tr><th></th><th>Predicted: 0</th><th>Predicted: 1</th></tr>
                <tr><th>Actual: 0</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
                <tr><th>Actual: 1</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Classification Report (Test Data)</h2>
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                <tr><td>Class 0</td><td>{class_report['0']['precision']:.4f}</td><td>{class_report['0']['recall']:.4f}</td><td>{class_report['0']['f1-score']:.4f}</td><td>{int(class_report['0']['support'])}</td></tr>
                <tr><td>Class 1</td><td>{class_report['1']['precision']:.4f}</td><td>{class_report['1']['recall']:.4f}</td><td>{class_report['1']['f1-score']:.4f}</td><td>{int(class_report['1']['support'])}</td></tr>
                <tr><td><strong>Weighted Avg</strong></td><td>{class_report['weighted avg']['precision']:.4f}</td><td>{class_report['weighted avg']['recall']:.4f}</td><td>{class_report['weighted avg']['f1-score']:.4f}</td><td>{int(class_report['weighted avg']['support'])}</td></tr>
            </table>
        </div>
        
        <div class="info-section">
            <h2>Trading Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Symbol</td><td>BTC-USD</td></tr>
                <tr><td>Stop Loss Percentage</td><td>5%</td></tr>
                <tr><td>Target Percentage</td><td>15%</td></tr>
                <tr><td>Data Source</td><td>Yahoo Finance</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    model_info_filename = os.path.join(reports_dir, "model_info.html")
    with open(model_info_filename, "w", encoding="utf-8") as f:
        f.write(model_info_html)
    print(f"Exported model information to {model_info_filename}")
    
    # Plot interactive trading signals
    print("Plotting trading signals...")
    train_idx = X_train.index
    test_idx = X_test.index
    df = analysis.data.copy()
    
    fig = go.Figure()
    
    # Plot price line
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        mode='lines', 
        name='Close Price', 
        line=dict(color='black', width=1)
    ))
    
    # Plot training signals with target=1 (green dots)
    train_target_1_mask = (df.index.isin(train_idx)) & (df['Label'] == 1)
    fig.add_trace(go.Scatter(
        x=df.index[train_target_1_mask],
        y=df['Close'][train_target_1_mask],
        mode='markers',
        marker=dict(color='green', size=8, symbol='circle'),
        name='Training Target=1'
    ))
    
    # Plot test signals with target=1 (blue dots)
    test_target_1_mask = (df.index.isin(test_idx)) & (df['Label'] == 1)
    fig.add_trace(go.Scatter(
        x=df.index[test_target_1_mask],
        y=df['Close'][test_target_1_mask],
        mode='markers',
        marker=dict(color='blue', size=8, symbol='circle'),
        name='Test Target=1'
    ))
    
    fig.update_layout(
        title="Random Forest Trading Signals - BTC-USD",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )
    
    signals_filename = os.path.join(reports_dir, "trading_signals.html")
    fig.write_html(signals_filename)
    print(f"Exported trading signals plot to {signals_filename}")
    
    # Generate dashboard.html with 3 tabs
    print("Generating dashboard...")
    dashboard_path = os.path.join(reports_dir, "dashboard.html")
    dashboard_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Crypto Trading Dashboard - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f8f8; margin: 0; padding: 20px; }}
            h1 {{ color: #222; text-align: center; }}
            .tabs {{ display: flex; border-bottom: 2px solid #ccc; margin-bottom: 20px; justify-content: center; }}
            .tab {{ padding: 15px 40px; cursor: pointer; background: #eee; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 5px; font-size: 16px; transition: all 0.3s; }}
            .tab:hover {{ background: #ddd; }}
            .tab.active {{ background: #fff; border-bottom: 2px solid #fff; font-weight: bold; color: #4CAF50; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            .iframe-box {{ background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            iframe {{ width: 100%; height: 700px; border: none; border-radius: 6px; }}
        </style>
        <script>
            function showTab(tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].classList.remove("active");
                }}
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].classList.remove("active");
                }}
                document.getElementById(tabName).classList.add("active");
                document.getElementById(tabName + "-tab").classList.add("active");
            }}
            window.onload = function() {{ showTab('data_table'); }};
        </script>
    </head>
    <body>
        <h1>🚀 Crypto Trading Dashboard - Random Forest Model</h1>
        <p style="text-align: center; color: #666;">Generated: {timestamp}</p>
        
        <div class="tabs">
            <div class="tab active" id="data_table-tab" onclick="showTab('data_table')">📊 Data Table</div>
            <div class="tab" id="model_info-tab" onclick="showTab('model_info')">🤖 Model Information</div>
            <div class="tab" id="signals-tab" onclick="showTab('signals')">📈 Trading Signals</div>
        </div>
        
        <div class="tab-content active" id="data_table">
            <div class="iframe-box">
                <iframe src="data_table.html"></iframe>
            </div>
        </div>
        
        <div class="tab-content" id="model_info">
            <div class="iframe-box">
                <iframe src="model_info.html"></iframe>
            </div>
        </div>
        
        <div class="tab-content" id="signals">
            <div class="iframe-box">
                <iframe src="trading_signals.html"></iframe>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_content)
    print(f"\n✅ Dashboard generated successfully at {dashboard_path}")

if __name__ == "__main__":
    main()
