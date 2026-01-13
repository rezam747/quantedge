"""
Example: BTC-USD Trading Analysis with Random Forest
This example demonstrates how to use the crypto trading analysis package
to analyze BTC-USD with custom parameters.
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path so imports work from anywhere
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from quantedge.data.data_handler import DataHandler
from quantedge.features.feature_engineer import FeatureEngineer
from quantedge.models.random_forest_model import RandomForestModel
from quantedge.visualization.dashboard_generator import DashboardGenerator


# ===== USER CONFIGURATION =====
# All parameters that the user needs to define

# Trading parameters
SYMBOL = "BTC-USD"
STOP_LOSS_PCT = 5  # Percentage
TARGET_PCT = 15    # Percentage
EACH_TRADE_VALUE = 2000  # USD value per trade position
INITIAL_BALANCE = 20000  # Total capital
DATA_SOURCE = "yahoo"
START_DATE = "2020-01-01"  # Start date for fetching historical data

# Model parameters (Random Forest)
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42
}

# Data split ratios
TRAIN_RATIO = 0.8

# Feature columns for model training
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
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


def main():
    """
    Main function to run the BTC-USD trading analysis.
    """
    # Create timestamped subfolder in reports/
    reports_dir = project_root / "reports" / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f"🚀 CRYPTO TRADING ANALYSIS - {SYMBOL}")
    print("="*60)
    print(f"Symbol: {SYMBOL}")
    print(f"Start Date: {START_DATE}")
    print(f"Stop Loss: {STOP_LOSS_PCT}%")
    print(f"Target: {TARGET_PCT}%")
    print(f"Each Trade Value: ${EACH_TRADE_VALUE}")
    print(f"Initial Balance: ${INITIAL_BALANCE}")
    print("="*60)
    
    # ===== STEP 1: DATA FETCHING AND CLEANING =====
    print("\n📊 STEP 1: Data Fetching and Cleaning")
    print("-" * 60)
    
    data_handler = DataHandler(
        symbol=SYMBOL,
        stop_loss_pct=STOP_LOSS_PCT,
        target_pct=TARGET_PCT,
        data_source=DATA_SOURCE
    )
    
    data_handler.fetch_data(start_date=START_DATE)
    data_handler.clean_data()
    data_handler.calculate_labels()
    
    # ===== STEP 2: FEATURE ENGINEERING =====
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 60)
    
    feature_engineer = FeatureEngineer(data_handler.get_data())
    feature_engineer.create_features()
    
    X_train, X_test, y_train, y_test = feature_engineer.split_data(
        feature_columns=FEATURE_COLUMNS,
        train_ratio=TRAIN_RATIO
    )
    
    # ===== STEP 3: MODEL TRAINING =====
    print("\n🤖 STEP 3: Model Training")
    print("-" * 60)
    
    rf_model = RandomForestModel(**RANDOM_FOREST_PARAMS)
    rf_model.train(X_train, y_train)
    
    # ===== STEP 4: MODEL EVALUATION =====
    print("\n📈 STEP 4: Model Evaluation")
    print("-" * 60)
    
    train_metrics = rf_model.get_detailed_metrics(X_train, y_train, data_type="training")
    test_metrics = rf_model.get_detailed_metrics(X_test, y_test, data_type="test")
    
    train_predictions = rf_model.predict(X_train)
    test_predictions = rf_model.predict(X_test)
    
    # ===== STEP 5: GENERATE REPORT =====
    dashboard_gen = DashboardGenerator(
        data=feature_engineer.get_data(),
        symbol=SYMBOL
    )
    
    dashboard_gen.generate_full_report(
        reports_dir=str(reports_dir),
        timestamp=reports_dir.name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        model_params=RANDOM_FOREST_PARAMS,
        model=rf_model,
        stop_loss_pct=STOP_LOSS_PCT,
        target_pct=TARGET_PCT,
        initial_balance=INITIAL_BALANCE,
        each_trade_value=EACH_TRADE_VALUE
    )


if __name__ == "__main__":
    main()
