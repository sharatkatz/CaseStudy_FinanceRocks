import sys, os
import pytest # type: ignore
import pandas # type: ignore
import numpy as np # type: ignore
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import matplotlib
import seaborn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from CaseStudy_FinanceRocks import ExploratoryDataAnalysis


class TestReportMissings:
    """Test cases for the report_missings method."""
    # -----------------------------------------------------------------------------
    # This fixture creates a mock instance of the ExploratoryDataAnalysis class
    # with a predefined customer_data DataFrame containing missing values.
    # This allows us to test the report_missings method in isolation,
    # without relying on external files or data sources.

    # -----------------------------------------------------------------------------
    # The patch function is used to temporarily replace certain parts of the code
    # with mock objects during testing, allowing isolation of the code under test.
    # The Mock class is used to create mock objects that simulate the behavior of real objects.
    # The patch function is being used here to mock out file system interactions and data loading functions
    # within the ExploratoryDataAnalysis class. Specifically, it mocks:
    """
    1. os.path.join: This is used in the class to construct file paths. 
    By mocking it, we prevent any actual file path operations.
    2. pathlib.Path: This is used to handle file paths in a more object-oriented way. 
    Mocking it prevents any real file system access.
    3. pandas.read_parquet: This function is used to read Parquet files into pandas DataFrames. 
    By mocking it, we avoid loading any real data from disk.
    """
    # This allows us to create a controlled test environment where we can define the data directly 
    # in the test, ensuring that our tests are not dependent on external files or data sources.
    # usage: 
    # with patch('os.path.join'), \
    #      patch('pathlib.Path'), \
    #      patch('pandas.read_parquet'):
    #         # test code here
    #   # -----------------------------------------------------------------------------
    #     -----------------------------------------------------------------------------
    # The Mock class is used to create mock objects that simulate the behavior of real objects.
    # In this case, we create a mock instance of the ExploratoryDataAnalysis class.

    @pytest.fixture
    def mock_eda_instance(self):
        """Create a mock EDA instance with test data."""
        with patch('os.path.join'), \
             patch('pathlib.Path'), \
             patch('pandas.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame({
                'col1': [1, 2, np.nan, 4],
                'col2': [np.nan, np.nan, 3, 4],
                'col3': [1, 2, 3, 4],
                'col4': [np.nan, np.nan, np.nan, np.nan]
            })
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
            return eda

    def test_report_missings_returns_series(self, mock_eda_instance):
        """Test that report_missings returns a pandas Series."""
        result = mock_eda_instance.report_missings()
        assert isinstance(result, pandas.Series)

    def test_report_missings_contains_only_columns_with_missing_values(self, mock_eda_instance):
        """Test that only columns with missing values are included."""
        result = mock_eda_instance.report_missings()
        assert 'col3' not in result.index
        assert 'col1' in result.index
        assert 'col2' in result.index
        assert 'col4' in result.index

    def test_report_missings_correct_counts(self, mock_eda_instance):
        """Test that missing value counts are correct."""
        result = mock_eda_instance.report_missings()
        assert result['col1'] == 1
        assert result['col2'] == 2
        assert result['col4'] == 4

    def test_report_missings_no_missing_values(self):
        """Test when there are no missing values."""
        with patch('os.path.join'), \
             patch('pathlib.Path'), \
             patch('pandas.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame({
                'col1': [1, 2, 3, 4],
                'col2': [5, 6, 7, 8]
            })
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
            
            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pandas.Series)

    def test_report_missings_empty_dataframe(self):
        """Test with an empty dataframe."""
        with patch('os.path.join'), \
             patch('pathlib.Path'), \
             patch('pandas.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame()
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
            
            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pandas.Series)


class TestVisualization:
    def test_visualize_missings(self):
        """Test missing values visualization."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                # Create mock dataframe
                mock_df = MagicMock()
                mock_df.columns = ['col1', 'col2']
                mock_df.isnull.return_value = pandas.DataFrame({
                    'col1': [False, True, False],
                    'col2': [True, False, True]
                })
                mock_read_parquet.return_value = mock_df
                
                with patch('os.path.join', return_value='/fake/path/data.parquet'):
                    with patch('os.makedirs'):
                        with patch('matplotlib.pyplot.savefig'):
                            with patch('matplotlib.pyplot.close'):
                                with patch('matplotlib.pyplot.figure'):
                                    with patch('seaborn.heatmap') as mock_heatmap:  # ← SPECIFIC function
                                        eda = ExploratoryDataAnalysis('/fake/path', 'data.parquet')
                                        eda.visualize_missings()
                                        
                                        # Verify visualization was called
                                        mock_heatmap.assert_called_once()

    def test_create_hist_plots(self):
        """Test histogram plot creation."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                mock_df = MagicMock()
                mock_df.columns = ['numeric_col', 'categorical_col']
                mock_read_parquet.return_value = mock_df
                
                # Mock column type checks
                with patch('pandas.api.types.is_numeric_dtype') as mock_is_numeric:
                    with patch('pandas.api.types.is_object_dtype') as mock_is_object:
                        mock_is_numeric.side_effect = lambda x: x == 'numeric_col'
                        mock_is_object.side_effect = lambda x: x == 'categorical_col'
                        
                        with patch('os.path.join', return_value='/fake/path/data.parquet'):
                            with patch('os.makedirs'):
                                with patch('matplotlib.pyplot.savefig'):
                                    with patch('matplotlib.pyplot.close'):
                                        with patch('seaborn.histplot') as mock_histplot:    # ← SPECIFIC function
                                            with patch('seaborn.countplot') as mock_countplot:  # ← SPECIFIC function
                                                eda = ExploratoryDataAnalysis('/fake/path', 'data.parquet')
                                                eda.create_hist_plots()
                                                
                                                # Verify plots were created
                                                mock_histplot.assert_called()
                                                mock_countplot.assert_called()

    def test_create_boxplots(self):
        """Test box plot creation."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                mock_df = MagicMock()
                mock_df.columns = ['numeric_col1', 'numeric_col2']
                mock_read_parquet.return_value = mock_df
                
                with patch('pandas.api.types.is_numeric_dtype', return_value=True):
                    with patch('os.path.join', return_value='/fake/path/data.parquet'):
                        with patch('os.makedirs'):
                            with patch('matplotlib.pyplot.savefig'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('seaborn.boxplot') as mock_boxplot:  # ← SPECIFIC function
                                        eda = ExploratoryDataAnalysis('/fake/path', 'data.parquet')
                                        eda.create_boxplots()
                                        
                                        mock_boxplot.assert_called()

    def test_correlation_matrix(self):
        """Test correlation matrix visualization."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                mock_df = MagicMock()
                mock_df.columns = ['col1', 'col2', 'col3']
                # Mock correlation matrix
                mock_df.corr.return_value = pandas.DataFrame({
                    'col1': [1.0, 0.5, 0.3],
                    'col2': [0.5, 1.0, 0.7],
                    'col3': [0.3, 0.7, 1.0]
                })
                mock_read_parquet.return_value = mock_df
                
                with patch('os.path.join', return_value='/fake/path/data.parquet'):
                    with patch('os.makedirs'):
                        with patch('matplotlib.pyplot.savefig'):
                            with patch('matplotlib.pyplot.close'):
                                with patch('matplotlib.pyplot.figure'):
                                    with patch('seaborn.heatmap') as mock_heatmap:  # ← SPECIFIC function
                                        eda = ExploratoryDataAnalysis('/fake/path', 'data.parquet')
                                        eda.visualize_correlation_matrix()
                                        
                                        mock_heatmap.assert_called_once()
