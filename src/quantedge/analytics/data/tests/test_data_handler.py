"""
Unit tests for the DataHandler class.
"""
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from quantedge.analytics.data.data_handler import DataHandler

# Sample test data
TEST_DATA = {
    'Open': [100, 105, 110, 108, 112],
    'High': [110, 115, 120, 118, 122],
    'Low': [95, 100, 105, 103, 107],
    'Close': [105, 110, 115, 112, 120],
    'Volume': [1000, 1500, 2000, 1800, 2500]
}

def create_test_dataframe():
    """Create a test DataFrame with datetime index."""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    df = pd.DataFrame(TEST_DATA, index=dates)
    return df

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    return create_test_dataframe()

@pytest.fixture
def data_handler():
    """Fixture to create a DataHandler instance for testing."""
    return DataHandler(symbol="BTC-USD", stop_loss_pct=5, target_pct=10, data_source="yahoo")

@patch('yfinance.Ticker')
def test_fetch_data(mock_ticker, data_handler, sample_data):
    """Test fetching data from Yahoo Finance."""
    # Setup mock
    mock_history = MagicMock()
    mock_history.index = sample_data.index
    mock_ticker.return_value.history.return_value = sample_data
    
    # Call the method
    result = data_handler.fetch_data("2023-01-01")
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'Open' in result.columns
    assert 'High' in result.columns
    assert 'Low' in result.columns
    assert 'Close' in result.columns
    assert 'Volume' in result.columns
    assert len(result) == 5

def test_clean_data(data_handler, sample_data):
    """Test cleaning data with missing values."""
    # Create test data with NaN values
    test_data = sample_data.copy()
    test_data.iloc[2:4, :] = np.nan
    data_handler.data = test_data
    
    # Call the method
    result = data_handler.clean_data()
    
    # Assertions
    assert not result.isnull().any().any()
    assert len(result) == 5

def test_calculate_labels(data_handler, sample_data):
    """Test calculating trade labels."""
    # Setup test data
    data_handler.data = sample_data
    
    # Call the method
    result = data_handler.calculate_labels()
    
    # Assertions
    assert 'signal_labels' in result.columns
    assert 'Entry_Price' in result.columns
    assert 'Exit_Price' in result.columns
    assert 'Exit_Date' in result.columns
    assert 'Trade_Duration' in result.columns
    assert 'Return' in result.columns
    assert len(result) == 5
    assert all(isinstance(x, (int, np.integer)) for x in result['signal_labels'])
    assert all(isinstance(x, (int, np.integer)) for x in result['Trade_Duration'])

def test_get_data(data_handler, sample_data):
    """Test getting processed data."""
    # Setup test data
    data_handler.data = sample_data
    
    # Call the method
    result = data_handler.get_data()
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) == 5

def test_export_data_html(data_handler, sample_data, tmp_path):
    """Test exporting data to HTML."""
    # Setup test data
    data_handler.data = sample_data
    test_file = tmp_path / "test_export.html"
    
    # Call the method
    data_handler.export_data_html(str(test_file))
    
    # Assertions
    assert os.path.exists(test_file)
    assert os.path.getsize(test_file) > 0

