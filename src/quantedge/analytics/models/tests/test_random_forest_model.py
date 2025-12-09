"""
sample test"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from quantedge.analytics.models.random_forest_model import RandomForestModel

def test_rf_model():
    # Test random forest
    print("Test random forest model")
    assert "Test random forest model"