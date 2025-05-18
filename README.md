# Crypto Trading Analysis Package

A comprehensive cryptocurrency analysis and trading prediction tool that leverages machine learning and technical analysis to analyze cryptocurrency price movements and generate trading signals.

## Features

- **Data Collection**
  - Historical OHLCV data import from Yahoo Finance
  - Support for multiple cryptocurrencies
  - Real-time data fetching capabilities

- **Technical Analysis**
  - Multiple moving averages (MA, EMA)
  - RSI, MACD, Bollinger Bands
  - Volume analysis
  - Momentum indicators
  - Volatility measures
  - Candlestick pattern recognition

- **Machine Learning**
  - Multiple model support (Random Forest, XGBoost, LightGBM)
  - Feature engineering
  - Model evaluation metrics
  - Cross-validation
  - Performance visualization

- **Trading Signals**
  - Stop-loss and target price calculations
  - Trade outcome labeling
  - Risk management metrics
  - Performance analysis

## Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv wvenv
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source wvenv/bin/activate
     ```
   - On Windows:
     ```bash
     wvenv\Scripts\activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Basic Usage**
   ```bash
   python src/crypto_analysis.py
   ```

2. **Custom Parameters**
   ```python
   from src.crypto_analysis import CryptoAnalysis

   # Initialize with custom parameters
   analyzer = CryptoAnalysis(
       symbol="BTC-USD",          # Cryptocurrency symbol
       stop_loss_pct=5,           # Stop loss percentage
       target_pct=15,             # Target profit percentage
       data_source="yahoo",       # Data source
       model_type="random_forest" # ML model type
   )

   # Run analysis
   analyzer.fetch_data()
   analyzer.clean_data()
   analyzer.calculate_labels()
   analyzer.create_features()
   ```

## Project Structure

```
├── src/
│   └── crypto_analysis.py    # Main analysis script
├── reports/                  # Generated reports and visualizations
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- yfinance == 0.1.87
- scikit-learn >= 1.0.0
- plotly >= 5.18.0
- ta >= 0.10.0
- python-binance >= 1.0.0
- pandas-datareader >= 0.10.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

## Output

The analysis generates:
- Interactive HTML reports with visualizations
- Model performance metrics
- Trading signal analysis
- Technical indicator charts
- Performance statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


