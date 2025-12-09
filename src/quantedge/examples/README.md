# Examples

This folder contains example scripts demonstrating how to use the crypto trading analysis package.

## 📊 BTC-USD Trading Example

**File**: `btc_trading_example.py`

A complete example showing how to analyze BTC-USD trading signals using Random Forest.

### Usage

```bash
python examples/btc_trading_example.py
```

### Configuration

All user-configurable parameters are at the top of the file:

#### Trading Parameters
- `SYMBOL`: Trading pair (default: "BTC-USD")
- `STOP_LOSS_PCT`: Stop loss percentage (default: 5)
- `TARGET_PCT`: Target profit percentage (default: 15)
- `DATA_SOURCE`: Data source (default: "yahoo")
- `START_DATE`: Start date for historical data (default: "2020-01-01")

#### Model Parameters
- `RANDOM_FOREST_PARAMS`: Dictionary of Random Forest hyperparameters
  - `n_estimators`: Number of trees (default: 200)
  - `max_depth`: Maximum tree depth (default: 10)
  - `min_samples_split`: Minimum samples for split (default: 10)
  - `min_samples_leaf`: Minimum samples per leaf (default: 5)
  - `class_weight`: Class balancing (default: 'balanced')
  - `random_state`: Random seed (default: 42)

#### Data Split
- `TRAIN_RATIO`: Training data ratio (default: 0.7)
- `VAL_TEST_SPLIT`: Validation/Test split (default: 0.5)

#### Features
- `FEATURE_COLUMNS`: List of technical indicators to use (60+ features)

### Output

The example generates a timestamped report folder with:
- `dashboard.html` - Main dashboard with 5 tabs
- `data_table.html` - Full dataset
- `model_info.html` - Model performance metrics
- `trading_signals.html` - Interactive chart
- `training_data_table.html` - Training data with predictions
- `testing_data_table.html` - Testing data with predictions

### Dashboard Tabs

1. **📊 Full Data Table** - Complete dataset with all calculated features and signal_labels
2. **🤖 Model Information** - Comprehensive model performance metrics and configuration
3. **📈 Trading Signals** - Interactive price chart showing:
   - Green dots: Training data where signal_labels = 1
   - Blue dots: Testing data where signal_labels = 1
4. **🎓 Training Data Table** - Training dataset with `predicted_signal_labels` column
5. **🧪 Testing Data Table** - Testing dataset with `predicted_signal_labels` column

## Creating Your Own Example

To create a custom analysis:

1. Copy `btc_trading_example.py`
2. Modify the configuration section at the top
3. Adjust parameters to your needs:
   - Change `SYMBOL` for different cryptocurrencies
   - Adjust `START_DATE` for different time periods
   - Tune model parameters in `RANDOM_FOREST_PARAMS`
   - Modify stop-loss and target percentages

Example for ETH-USD:

```python
SYMBOL = "ETH-USD"
START_DATE = "2021-01-01"
STOP_LOSS_PCT = 3
TARGET_PCT = 10
```

## Next Steps

After running the example:
1. Open the generated `dashboard.html` in your browser
2. Review the model performance in the Model Information tab
3. Analyze trading signals in the Trading Signals tab
4. Examine training/testing data with predictions
5. Adjust parameters and re-run for better results
