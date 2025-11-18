import pytest
import sys
import os

# Add src to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Use absolute imports
from CaseStudy_FinanceRocks.core import ExploratoryDataAnalysis

def test_import():
    """Test that the package can be imported."""
    assert ExploratoryDataAnalysis is not None

def test_class_initialization():
    """Test basic class functionality."""
    # Test that class can be referenced
    assert hasattr(ExploratoryDataAnalysis, '__init__')
