"""
QuantEdge - Cryptocurrency Trading Analysis with Machine Learning
"""
import sys
import os

# Automatically add src directory to Python path when package is imported
_src_path = os.path.join(os.path.dirname(__file__), '..', '..')
_src_path = os.path.abspath(_src_path)

if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

__version__ = "0.1.0"
