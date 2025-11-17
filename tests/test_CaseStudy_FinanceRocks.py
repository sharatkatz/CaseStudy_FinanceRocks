import pytest # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from unittest.mock import Mock, patch, MagicMock
import os

from .core import ExploratoryDataAnalysis


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
    2. Path: This is used to handle file paths in a more object-oriented way. 
    Mocking it prevents any real file system access.
    3. pd.read_parquet: This function is used to read Parquet files into pandas DataFrames. 
    By mocking it, we avoid loading any real data from disk.
    """
    # This allows us to create a controlled test environment where we can define the data directly 
    # in the test, ensuring that our tests are not dependent on external files or data sources.
    # usage: 
    # with patch('CaseStudy_FinanceRocks.os.path.join'), \
    #      patch('CaseStudy_FinanceRocks.Path'), \
    #      patch('CaseStudy_FinanceRocks.pd.read_parquet'):
    #         # test code here
    #   # -----------------------------------------------------------------------------
    #     -----------------------------------------------------------------------------
    # The Mock class is used to create mock objects that simulate the behavior of real objects.
    # In this case, we create a mock instance of the ExploratoryDataAnalysis class.

    @pytest.fixture
    def mock_eda_instance(self):
        """Create a mock EDA instance with test data."""
        with patch('CaseStudy_FinanceRocks.os.path.join'), \
             patch('CaseStudy_FinanceRocks.Path'), \
             patch('CaseStudy_FinanceRocks.pd.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pd.DataFrame({
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
        assert isinstance(result, pd.Series)

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
        with patch('CaseStudy_FinanceRocks.os.path.join'), \
             patch('CaseStudy_FinanceRocks.Path'), \
             patch('CaseStudy_FinanceRocks.pd.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pd.DataFrame({
                'col1': [1, 2, 3, 4],
                'col2': [5, 6, 7, 8]
            })
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
            
            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pd.Series)

    @patch('CaseStudy_FinanceRocks.pprint')
    def test_report_missings_prints_output(self, mock_pprint, mock_eda_instance):
        """Test that the method prints the missing values report."""
        with patch('builtins.print') as mock_print:
            result = mock_eda_instance.report_missings()
            mock_print.assert_called_once_with("Missing Values Report:")
            mock_pprint.assert_called_once()
            call_args = mock_pprint.call_args[0][0]
            assert isinstance(call_args, dict)

    def test_report_missings_empty_dataframe(self):
        """Test with an empty dataframe."""
        with patch('CaseStudy_FinanceRocks.os.path.join'), \
             patch('CaseStudy_FinanceRocks.Path'), \
             patch('CaseStudy_FinanceRocks.pd.read_parquet'):
            
            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pd.DataFrame()
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
            
            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pd.Series)
            class TestCreateBarplotsByPackage:
                """Test cases for the create_barplots_bypackage method."""

                @pytest.fixture
                def mock_eda_instance_with_packages(self):
                    """Create a mock EDA instance with test data including packages."""
                    with patch('CaseStudy_FinanceRocks.os.path.join'), \
                         patch('CaseStudy_FinanceRocks.Path'), \
                         patch('CaseStudy_FinanceRocks.pd.read_parquet'):
                        
                        eda = Mock(spec=ExploratoryDataAnalysis)
                        eda.customer_data = pd.DataFrame({
                            'package': ['basic', 'basic', 'premium', 'premium'],
                            'category_col': ['A', 'B', 'A', 'C'],
                            'numeric_col': [1, 2, 3, 4],
                            'object_col': ['x', 'y', 'z', 'x']
                        })
                        eda.sub_columns = ['category_col', 'numeric_col', 'object_col']
                        eda.package_var = 'package'
                        eda.unique_packages = ['basic', 'premium']
                        eda.plot_dir = '/mock/plot/dir'
                        eda.create_barplots_bypackage = ExploratoryDataAnalysis.create_barplots_bypackage.__get__(eda)
                        return eda

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_creates_plots_for_categorical_columns(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that bar plots are created for categorical/object columns only."""
                    mock_join.return_value = '/mock/path'
                    
                    with patch('builtins.print'):
                        result = mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    # Should create plots for 2 packages * 2 categorical columns = 4 plots
                    assert mock_plt.figure.call_count == 4
                    assert mock_sns.countplot.call_count == 4
                    assert mock_plt.savefig.call_count == 4

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_creates_correct_directory_structure(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that correct directories are created."""
                    mock_join.side_effect = lambda *args: '/'.join(args)
                    
                    with patch('builtins.print'):
                        mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    # Check that makedirs was called
                    assert mock_makedirs.called
                    mock_makedirs.assert_called_with('/mock/plot/dir/bar_plots_by_package', exist_ok=True)

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_closes_figures_properly(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that figures are properly closed after creation."""
                    mock_join.return_value = '/mock/path'
                    
                    with patch('builtins.print'):
                        mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    # Check that cleanup methods are called
                    assert mock_plt.cla.call_count == 4
                    assert mock_plt.clf.call_count == 4
                    assert mock_plt.close.call_count >= 4
                    mock_plt.close.assert_any_call('all')

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_returns_none(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that the method returns None."""
                    with patch('builtins.print'):
                        result = mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    assert result is None

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_prints_status_messages(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that status messages are printed."""
                    mock_join.return_value = '/mock/path'
                    
                    with patch('builtins.print') as mock_print:
                        mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    # Should print for each plot + final message
                    assert mock_print.call_count == 5  # 4 plots + 1 final message
                    
                    # Check final message
                    final_call = mock_print.call_args_list[-1][0][0]
                    assert "All bar plots by package have been saved" in final_call

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_no_categorical_columns(self, mock_join, mock_makedirs, mock_sns, mock_plt):
                    """Test behavior when there are no categorical columns."""
                    with patch('CaseStudy_FinanceRocks.os.path.join'), \
                         patch('CaseStudy_FinanceRocks.Path'), \
                         patch('CaseStudy_FinanceRocks.pd.read_parquet'):
                        
                        eda = Mock(spec=ExploratoryDataAnalysis)
                        eda.customer_data = pd.DataFrame({
                            'package': ['basic', 'premium'],
                            'numeric_col1': [1, 2],
                            'numeric_col2': [3, 4]
                        })
                        eda.sub_columns = ['numeric_col1', 'numeric_col2']
                        eda.package_var = 'package'
                        eda.unique_packages = ['basic', 'premium']
                        eda.plot_dir = '/mock/plot/dir'
                        eda.create_barplots_bypackage = ExploratoryDataAnalysis.create_barplots_bypackage.__get__(eda)
                        
                        with patch('builtins.print'):
                            eda.create_barplots_bypackage()
                        
                        # No plots should be created
                        assert mock_plt.figure.call_count == 0
                        assert mock_sns.countplot.call_count == 0

                @patch('CaseStudy_FinanceRocks.plt')
                @patch('CaseStudy_FinanceRocks.sns')
                @patch('CaseStudy_FinanceRocks.os.makedirs')
                @patch('CaseStudy_FinanceRocks.os.path.join')
                def test_filters_data_by_package(self, mock_join, mock_makedirs, mock_sns, mock_plt, mock_eda_instance_with_packages):
                    """Test that data is correctly filtered by package."""
                    mock_join.return_value = '/mock/path'
                    
                    with patch('builtins.print'):
                        mock_eda_instance_with_packages.create_barplots_bypackage()
                    
                    # Verify that countplot was called with filtered data
                    assert mock_sns.countplot.called
                    for call in mock_sns.countplot.call_args_list:
                        # Each call should have x parameter with filtered data
                        assert 'x' in call[1] or len(call[0]) > 0


if __name__ == "__main__":
    """Test cases for the report_missings method."""
    with patch('CaseStudy_FinanceRocks.os.path.join'), \
         patch('CaseStudy_FinanceRocks.Path'), \
         patch('CaseStudy_FinanceRocks.pd.read_parquet'):
             eda = Mock(spec=ExploratoryDataAnalysis)
             eda.customer_data = pd.DataFrame({
                'col1': [1, 2, np.nan, 4],
                'col2': [np.nan, np.nan, 3, 4],
                'col3': [1, 2, 3, 4],
                'col4': [np.nan, np.nan, np.nan, np.nan]
             })
             eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(eda)
             print(eda.report_missings())