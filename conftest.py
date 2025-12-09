"""
pytest configuration file
Automatically configures Python path for all tests and execution contexts
"""
import sys
import os

# Add src directory to path for all pytest runs
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
