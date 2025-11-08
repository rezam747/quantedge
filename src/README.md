# Crypto Trading Analysis - Codebase Structure

## 📁 Project Organization

This project follows a modular architecture for cryptocurrency trading analysis using machine learning.

```
quantedge/
├── src/                         # Core package modules
│   ├── data/                    # Data fetching and cleaning
│   │   ├── __init__.py
│   │   └── data_handler.py     # DataHandler class
│   │
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineer.py # FeatureEngineer class
│   │
│   ├── models/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   └── random_forest_model.py  # RandomForestModel class
│   │
│   ├── visualization/           # Plotting and dashboards
│   │   ├── __init__.py
│   │   └── dashboard_generator.py  # DashboardGenerator class
│   │
│   ├── tests/                   # Unit tests
│   │   └── test_crypto_analysis.py
│   │
│   ├── crypto_analysis_backup.py   # Backup of original monolithic script
│   └── README.md                # This file
│
├── examples/                    # Example scripts with configurations
│   ├── btc_trading_example.py  # BTC-USD example with all parameters
│   └── README.md               # Examples documentation
│
├── reports/                     # Generated reports (timestamped folders)
└── requirements.txt             # Python dependencies
```

## 🔧 Module Descriptions

### 1. **data/data_handler.py**
Handles all data-related operations:
- `DataHandler` class:
  - `fetch_data()` - Fetches historical data from Yahoo Finance
  - `clean_data()` - Cleans and preprocesses data
  - `calculate_labels()` - Generates trading labels based on stop-loss and target
  - `export_data_html()` - Exports data to HTML format

### 2. **features/feature_engineer.py**
Creates technical indicators and features:
- `FeatureEngineer` class:
  - `create_features()` - Generates 60+ technical indicators:
    - Moving averages (MA, EMA)
    - Momentum indicators (RSI, MACD, ROC)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators (OBV, Volume Ratio)
    - Candlestick patterns
  - `split_data()` - Splits data into train/validation/test sets

### 3. **models/random_forest_model.py**
Machine learning model operations:
- `RandomForestModel` class:
  - `train()` - Trains the Random Forest classifier
  - `predict()` - Makes predictions
  - `evaluate()` - Evaluates model performance
  - `get_detailed_metrics()` - Returns comprehensive metrics

### 4. **visualization/dashboard_generator.py**
Creates visualizations and dashboards:
- `DashboardGenerator` class:
  - `create_trading_signals_plot()` - Interactive price chart with signals
  - `create_model_info_html()` - Detailed model performance page
  - `create_dashboard()` - Main dashboard with 5 tabs
  - `create_training_data_table()` - Training data with predictions
  - `create_testing_data_table()` - Testing data with predictions

## 🚀 Usage

### Run the BTC-USD example:
```bash
python examples/btc_trading_example.py
```

All configuration parameters are defined at the top of the example file:
- Trading symbol, stop-loss, target percentages
- Data source and start date
- Model hyperparameters
- Feature selection

### Output:
The script generates a timestamped folder in `reports/` containing:
- `dashboard.html` - Main dashboard with 3 tabs
- `data_table.html` - Complete dataset table
- `model_info.html` - Model performance metrics
- `trading_signals.html` - Interactive price chart with signals

### Dashboard Tabs:
1. **📊 Full Data Table** - Complete dataset with all calculated features and signal_labels
2. **🤖 Model Information** - Model configuration and performance metrics
3. **📈 Trading Signals** - Interactive chart showing:
   - Black line: Price
   - Green dots: Training data where signal_labels = 1
   - Blue dots: Testing data where signal_labels = 1
4. **🎓 Training Data Table** - Training dataset with `predicted_signal_labels` column
5. **🧪 Testing Data Table** - Testing dataset with `predicted_signal_labels` column

## 🔮 Future Enhancements

The modular structure makes it easy to:
- Add new data sources (in `src/data/`)
- Create new features (in `src/features/`)
- Add different models (in `src/models/`)
- Customize visualizations (in `src/visualization/`)
- Create custom examples with different parameters (in `examples/`)

## 📝 Example: Adding a New Model

```python
# Create: src/models/xgboost_model.py
from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, **params):
        self.model = XGBClassifier(**params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    
    # ... (other methods)
```

Then import and use in your example:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from models.xgboost_model import XGBoostModel

xgb_model = XGBoostModel(n_estimators=100, max_depth=5)
xgb_model.train(X_train, y_train)
```

## 🧪 Testing

Run tests:
```bash
pytest src/tests/
```

## 📄 License

See LICENSE file in project root.
