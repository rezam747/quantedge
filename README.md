# QuantEdge - Crypto Trading Analysis with Machine Learning

A comprehensive cryptocurrency analysis and trading prediction tool that leverages machine learning and technical analysis to analyze cryptocurrency price movements and generate trading signals.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- Data Collection
- Technical Analysis
- Machine Learning
- Trading Signals
- Visualization & Reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Bash (macOS/Linux) or WSL (Windows)

### 1. Clone the Repository
```bash
git clone https://github.com/rezam747/quantedge.git
cd quantedge
```

### 2. Setup Environment & Install Dependencies
The easiest way is to use the provided setup script:

```bash
./doit.sh
```

This script will:
- ✅ Create a Python virtual environment (`.venv`)
- ✅ Install all required dependencies
- ✅ Install development tools (pytest, black, flake8, mypy)
- ✅ Setup VSCode debugger configuration (optional)
- ✅ Activate the virtual environment

---

## 📚 Usage

### Run the Example
Execute the BTC-USD trading analysis example:

```bash
# From project root
python src/quantedge/analytics/examples/btc_trading_example.py
```

### Example Output
The script generates a timestamped report in `src/quantedge/analytics/reports/`:
```
reports/
└── 2025-12-08_18-38-28/
    ├── dashboard.html              # Main dashboard (5 tabs)
    ├── data_table.html             # Full dataset with features
    ├── model_info.html             # Model metrics & confusion matrix
    ├── trading_signals.html        # Interactive price chart
    ├── training_data_table.html    # Training data with predictions
    └── testing_data_table.html     # Test data with predictions
```

---

## 📋 Dependencies

### Core Dependencies
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **yfinance** - Data fetching
- **scikit-learn** - Machine learning
- **ta** - Technical analysis indicators
- **plotly** - Interactive visualizations
- **xgboost, lightgbm** - Advanced ML models

### Development Tools
- **pytest** - Testing framework
- **black** - Code formatter
- **isort** - Import sorter
- **flake8** - Linter
- **mypy** - Type checker
- **doit** - Task automation

See `requirements.txt` for complete list with versions.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example code

---

## 🔗 Related Links

- [Yahoo Finance API](https://finance.yahoo.com)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Plotly Documentation](https://plotly.com)
- [Technical Analysis Indicators](https://ta.readthedocs.io)

---

**Happy Trading! 🚀📈**



