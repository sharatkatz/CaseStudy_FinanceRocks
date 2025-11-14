#!/usr/bin/env python
import pytest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestEDA:
    """Test cases for EDA functionality."""

    def test_imports(self):
        """Test that all required modules can be imported."""
        import pandas
        import matplotlib
        import seaborn
        assert True

    def test_dataframe_creation(self):
        """Test basic DataFrame operations."""
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'category': ['A', 'B', 'A']
        })
        assert df.shape == (3, 2)
        assert list(df.columns) == ['value', 'category']

    def test_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['A', 'B', None]
        })
        missing = df.isnull().sum()
        assert missing['col1'] == 1
        assert missing['col2'] == 1
